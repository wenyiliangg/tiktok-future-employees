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


class QueryRetriever:
    def __init__(self, results: dict[str, list[FakeResult]]) -> None:
        self.results = results
        self.calls: list[tuple[object, int]] = []

    def retrieve(self, query: object, top_n: int) -> list[FakeResult]:
        self.calls.append((query, top_n))
        value = self.results.get(str(query), [])
        if value and value[0].parent_asin == "FAIL":
            raise RuntimeError("injected history failure")
        return value[:top_n]


class FailOnRepeatedQueryRetriever(QueryRetriever):
    def __init__(self, results: dict[str, list[FakeResult]], fail_query: str) -> None:
        super().__init__(results)
        self.fail_query = fail_query
        self.query_counts: dict[str, int] = {}

    def retrieve(self, query: object, top_n: int) -> list[FakeResult]:
        key = str(query)
        self.query_counts[key] = self.query_counts.get(key, 0) + 1
        if key == self.fail_query and self.query_counts[key] > 1:
            raise RuntimeError("injected repeated-query failure")
        return super().retrieve(query, top_n)


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
        self.assertEqual(
            [call[0] for call in self.anchor.calls],
            ["show me products", "Those options are not quite right yet."],
        )

    def test_feedback_memory_policy_reuses_last_informative_anchor_query(self) -> None:
        anchor = FakeRetriever(self.anchor.results)
        agent = Agent(
            self.catalog,
            config=HybridRetrievalConfig(mode="contextual"),
            anchor_retriever=anchor,
            contextual_policy=ContextualRetrievalPolicy(
                policy_id="test.feedback-memory",
                protected_lexical_count=2,
                negative_feedback_uses_active_intent=True,
            ),
        )
        agent.reset("memory", {})

        first = agent.respond("memory", "show me products", 1, 2)
        second = agent.respond("memory", "Those options are not quite right yet.", 2, 2)

        self.assertEqual(
            first["recommendations"],
            [{"parent_asin": "A"}, {"parent_asin": "B"}],
        )
        self.assertEqual(
            second["recommendations"],
            [{"parent_asin": "C"}, {"parent_asin": "D"}],
        )
        self.assertEqual(
            [call[0] for call in anchor.calls],
            ["show me products", "show me products"],
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

    def test_override_history_only_changes_unprotected_tail(self) -> None:
        current = [
            FakeResult(value, 101 - rank, rank)
            for rank, value in enumerate(
                ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J"),
                start=1,
            )
        ]
        current.extend(FakeResult("TARGET", 1, rank) for rank in (50,))
        history = [FakeResult("TARGET", 100, 1)]
        anchor = QueryRetriever(
            {
                "old identifying phrase": history,
                "Actually, I need the new requirement.": current,
            }
        )
        agent = Agent(
            self.catalog,
            config=HybridRetrievalConfig(mode="contextual"),
            anchor_retriever=anchor,
            contextual_policy=ContextualRetrievalPolicy(
                policy_id="contextual.override-history-tail.v1",
                protected_lexical_count=8,
                state_lexical_weight=0.5,
                negative_feedback_uses_active_intent=True,
            ),
        )
        agent._catalog_ids = frozenset(
            {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "TARGET"}
        )
        agent.reset("history", {})
        agent.respond("history", "old identifying phrase", 1, 10)

        response = agent.respond(
            "history", "Actually, I need the new requirement.", 2, 10
        )
        ranked = [item["parent_asin"] for item in response["recommendations"]]

        self.assertEqual(ranked[:8], list("ABCDEFGH"))
        self.assertIn("TARGET", ranked[8:])
        self.assertEqual(
            agent._historical_raw_evidence["history"], "old identifying phrase"
        )

    def test_history_is_reset_and_disabled_policy_does_not_archive(self) -> None:
        self.agent.respond("s", "old phrase", 1, 2)
        self.agent.respond("s", "Actually, start over", 2, 2)
        self.assertEqual(self.agent._historical_raw_evidence["s"], "")

        history_policy = ContextualRetrievalPolicy(
            policy_id="contextual.override-history-tail.v1",
            protected_lexical_count=8,
            state_lexical_weight=0.5,
        )
        agent = Agent(
            self.catalog,
            config=HybridRetrievalConfig(mode="contextual"),
            anchor_retriever=self.anchor,
            contextual_policy=history_policy,
        )
        agent.reset("reset", {})
        agent.respond("reset", "old phrase", 1, 2)
        agent.respond("reset", "Actually, start over", 2, 2)
        self.assertEqual(agent._historical_raw_evidence["reset"], "old phrase")
        agent.reset("reset", {})
        self.assertEqual(agent._historical_raw_evidence["reset"], "")

    def test_history_policy_is_identical_before_an_override(self) -> None:
        anchor = QueryRetriever({"initial request": self.anchor.results})
        agent = Agent(
            self.catalog,
            config=HybridRetrievalConfig(mode="contextual"),
            anchor_retriever=anchor,
            contextual_policy=ContextualRetrievalPolicy(
                policy_id="contextual.override-history-tail.v1",
                protected_lexical_count=8,
                state_lexical_weight=0.5,
            ),
        )
        agent.reset("pre-override", {})

        response = agent.respond("pre-override", "initial request", 1, 4)

        self.assertEqual(
            response["recommendations"],
            [{"parent_asin": value} for value in ("A", "B", "C", "D")],
        )
        self.assertEqual([call[0] for call in anchor.calls], ["initial request"])

    def test_history_retrieval_failure_preserves_current_ranking(self) -> None:
        anchor = FailOnRepeatedQueryRetriever(
            {
                "old evidence": self.anchor.results,
                "Actually, use new evidence": self.anchor.results,
            },
            fail_query="old evidence",
        )
        agent = Agent(
            self.catalog,
            config=HybridRetrievalConfig(mode="contextual"),
            anchor_retriever=anchor,
            contextual_policy=ContextualRetrievalPolicy(
                policy_id="contextual.override-history-tail.v1",
                protected_lexical_count=8,
                state_lexical_weight=0.5,
            ),
        )
        agent.reset("failure", {})
        agent.respond("failure", "old evidence", 1, 4)

        response = agent.respond("failure", "Actually, use new evidence", 2, 4)

        self.assertEqual(
            response["recommendations"],
            [{"parent_asin": value} for value in ("A", "B", "C", "D")],
        )
        self.assertEqual(agent._component_failure_counts["override_history"], 1)


if __name__ == "__main__":
    unittest.main()
