from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from starter.agent import Agent
from starter.contextual_retrieval import (
    ContextualRetrievalPolicy,
    rank_contextual_candidates,
)
from starter.hybrid_retrieval import HybridRetrievalConfig


@dataclass(frozen=True)
class FakeResult:
    parent_asin: str
    score: float
    rank: int


class FakeRetriever:
    def __init__(self, results: list[FakeResult]) -> None:
        self.results = results
        self.calls: list[tuple[object, int]] = []

    def retrieve(self, query: object, top_n: int) -> list[FakeResult]:
        self.calls.append((query, top_n))
        return self.results[:top_n]


class ContextualRankingTest(unittest.TestCase):
    def test_negative_ids_are_removed_and_bm25_prefix_is_protected(self) -> None:
        policy = ContextualRetrievalPolicy(
            policy_id="test",
            protected_lexical_count=2,
            state_lexical_weight=1.0,
        )
        result = rank_contextual_candidates(
            [
                FakeResult("NEGATIVE", 4, 1),
                FakeResult("A", 3, 2),
                FakeResult("B", 2, 3),
                FakeResult("C", 1, 4),
            ],
            [FakeResult("C", 2, 1), FakeResult("B", 1, 2)],
            [],
            {"NEGATIVE", "A", "B", "C"},
            {"NEGATIVE"},
            policy,
            limit=3,
        )

        self.assertEqual([item.parent_asin for item in result[:2]], ["A", "B"])
        self.assertEqual(result[2].parent_asin, "C")
        self.assertNotIn("NEGATIVE", {item.parent_asin for item in result})

    def test_ties_are_deterministic(self) -> None:
        policy = ContextualRetrievalPolicy(
            policy_id="test", protected_lexical_count=0, dense_weight=1.0
        )
        dense = [FakeResult("B", 1, 1), FakeResult("A", 1, 1)]
        orders = [
            [
                item.parent_asin
                for item in rank_contextual_candidates(
                    [], [], values, {"A", "B"}, set(), policy, limit=2
                )
            ]
            for values in (dense, list(reversed(dense)))
        ]
        self.assertEqual(orders, [["A", "B"], ["A", "B"]])


class ContextualAgentStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.catalog = Path(self.temporary.name) / "catalog.jsonl"
        self.catalog.write_text(
            "".join(
                json.dumps({"parent_asin": asin, "title": asin}) + "\n"
                for asin in ("A", "B", "C", "D")
            ),
            encoding="utf-8",
        )
        self.anchor = FakeRetriever(
            [
                FakeResult("A", 4, 1),
                FakeResult("B", 3, 2),
                FakeResult("C", 2, 3),
                FakeResult("D", 1, 4),
            ]
        )
        self.agent = Agent(
            self.catalog,
            config=HybridRetrievalConfig(mode="contextual"),
            anchor_retriever=self.anchor,
        )
        self.agent.reset("s", {})

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_explicit_negative_feedback_rotates_previous_recommendations(self) -> None:
        first = self.agent.respond("s", "show me products", 1, 2)
        second = self.agent.respond("s", "Those options are not quite right yet.", 2, 2)

        self.assertEqual(
            first["recommendations"],
            [{"parent_asin": "A"}, {"parent_asin": "B"}],
        )
        self.assertEqual(
            second["recommendations"],
            [{"parent_asin": "C"}, {"parent_asin": "D"}],
        )

    def test_intent_override_clears_known_negative_products(self) -> None:
        self.agent.respond("s", "show me products", 1, 2)
        self.agent.respond("s", "Those options are not quite right yet.", 2, 2)
        response = self.agent.respond(
            "s", "Actually, ignore my earlier preference and start over.", 3, 2
        )

        self.assertEqual(
            response["recommendations"],
            [{"parent_asin": "A"}, {"parent_asin": "B"}],
        )
        self.assertEqual(
            self.agent._active_raw_intent["s"],
            "Actually, ignore my earlier preference and start over.",
        )


if __name__ == "__main__":
    unittest.main()
