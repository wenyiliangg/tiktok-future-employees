from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from starter.agent import Agent
from starter.catalog_signature_index import (
    CatalogSignatureMatch,
    CatalogSignaturePolicy,
)
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


class FakeSignatureIndex:
    def __init__(self, parent_asin: str) -> None:
        self.parent_asin = parent_asin

    def match(self, _message: object) -> CatalogSignatureMatch:
        return CatalogSignatureMatch(self.parent_asin, "unique catalog phrase", 3)


class FailingSignatureIndex:
    def match(self, _message: object) -> CatalogSignatureMatch:
        raise RuntimeError("signature failure")


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

    def test_unique_signature_promotes_one_head_candidate_and_respects_rejection(
        self,
    ) -> None:
        anchor = FakeRetriever(self.anchor.results)
        policy = ContextualRetrievalPolicy(
            policy_id="test.signature",
            protected_lexical_count=2,
            negative_feedback_uses_active_intent=True,
        )
        signature_policy = CatalogSignaturePolicy(
            policy_id="test.signature",
            retrieval_policy_id=policy.policy_id,
            compatible_retrieval_policy_id=policy.policy_id,
            enabled=True,
        )
        agent = Agent(
            self.catalog,
            config=HybridRetrievalConfig(mode="contextual"),
            anchor_retriever=anchor,
            contextual_policy=policy,
            clarification_config=None,
            signature_policy=signature_policy,
            signature_index=FakeSignatureIndex("C"),
        )
        agent.reset("signature", {})

        first = agent.respond("signature", "unique catalog phrase", 1, 2)
        second = agent.respond(
            "signature", "Those options are not quite right yet.", 2, 2
        )

        self.assertEqual(
            first["recommendations"],
            [{"parent_asin": "C"}, {"parent_asin": "A"}],
        )
        self.assertEqual(
            second["recommendations"],
            [{"parent_asin": "B"}, {"parent_asin": "D"}],
        )

    def test_signature_lookup_failure_preserves_champion_order(self) -> None:
        policy = ContextualRetrievalPolicy(
            policy_id="test.signature.failure", protected_lexical_count=2
        )
        signature_policy = CatalogSignaturePolicy(
            policy_id="test.signature.failure",
            retrieval_policy_id=policy.policy_id,
            compatible_retrieval_policy_id=policy.policy_id,
            enabled=True,
        )
        agent = Agent(
            self.catalog,
            config=HybridRetrievalConfig(mode="contextual"),
            anchor_retriever=FakeRetriever(self.anchor.results),
            contextual_policy=policy,
            signature_policy=signature_policy,
            signature_index=FailingSignatureIndex(),
        )
        agent.reset("signature-failure", {})

        response = agent.respond("signature-failure", "unique catalog phrase", 1, 2)

        self.assertEqual(
            response["recommendations"],
            [{"parent_asin": "A"}, {"parent_asin": "B"}],
        )
        self.assertEqual(
            agent.diagnostics_snapshot()["component_failure_counts"],
            {"catalog_signature": 1},
        )


if __name__ == "__main__":
    unittest.main()
