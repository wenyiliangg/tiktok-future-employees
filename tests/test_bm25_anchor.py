from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent
from starter.bm25_anchor import BM25AnchorRetriever
from starter.hybrid_retrieval import HybridRetrievalConfig, RetrievalMode
from starter.search_models import RetrievalResult


class FakeRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[tuple[object, int]] = []

    def retrieve(self, query: object, top_n: int) -> list[RetrievalResult]:
        self.calls.append((query, top_n))
        return self.results[:top_n]


class BM25AnchorRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.catalog = Path(self.temporary.name) / "catalog.jsonl"
        rows = [
            {
                "parent_asin": "ANCHOR",
                "title": "plain garment",
                "features": ["target-specific unsupported phrase"],
            },
            {"parent_asin": "OTHER", "title": "different product"},
            {"parent_asin": "BACKFILL", "title": "red sneakers"},
        ]
        self.catalog.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_promoted_contextual_mode_is_default_and_risky_flags_are_safe(self) -> None:
        config = HybridRetrievalConfig()
        self.assertIs(config.mode, RetrievalMode.CONTEXTUAL)
        self.assertFalse(config.enable_feature_reranker)
        self.assertFalse(config.enable_boundary_fallback)
        self.assertFalse(config.route_policies["boundary"].always_attempt_fallback)

    def test_default_bm25_does_not_invoke_backfill_retrievers(self) -> None:
        anchor = FakeRetriever([RetrievalResult("ANCHOR", 10.0, 1)])
        lexical = FakeRetriever([RetrievalResult("BACKFILL", 2.0, 1)])
        dense = FakeRetriever([RetrievalResult("OTHER", 1.0, 1)])
        agent = Agent(
            self.catalog,
            config=HybridRetrievalConfig(mode="bm25"),
            anchor_retriever=anchor,
            lexical_retriever=lexical,
            dense_retriever=dense,
        )
        agent.reset("s", {})

        response = agent.respond("s", "unsupported phrase", 1, 1)

        self.assertEqual(response["recommendations"], [{"parent_asin": "ANCHOR"}])
        self.assertEqual(len(anchor.calls), 1)
        self.assertEqual(lexical.calls, [])
        self.assertEqual(dense.calls, [])

    def test_anchor_uses_raw_turn_text_lost_by_structured_extraction(self) -> None:
        agent = Agent(self.catalog)
        agent.reset("s", {})

        response = agent.respond(
            "s", "I need the target-specific unsupported phrase", 1, 1
        )

        self.assertEqual(response["recommendations"], [{"parent_asin": "ANCHOR"}])
        self.assertEqual(agent._state.query_for("s").text, "")
        self.assertEqual(
            agent._state.state_for("s").raw_current_turn_text,
            "I need the target-specific unsupported phrase",
        )

    def test_dense_and_structured_candidates_only_backfill_anchor_vacancies(
        self,
    ) -> None:
        anchor = FakeRetriever([RetrievalResult("ANCHOR", 10.0, 1)])
        lexical = FakeRetriever([RetrievalResult("BACKFILL", 2.0, 1)])
        dense = FakeRetriever([RetrievalResult("OTHER", 1.0, 1)])
        agent = Agent(
            self.catalog,
            config=HybridRetrievalConfig(mode="anchored", final_candidate_count=2),
            anchor_retriever=anchor,
            lexical_retriever=lexical,
            dense_retriever=dense,
        )
        agent.reset("s", {})

        response = agent.respond("s", "I need red sneakers", 1, 2)

        self.assertEqual(response["recommendations"][0], {"parent_asin": "ANCHOR"})
        self.assertEqual(dense.calls[0][0], "I need red sneakers")
        structured_query = lexical.calls[0][0]
        self.assertEqual(structured_query.category.strength, "soft")
        self.assertEqual(structured_query.color.strength, "soft")

    def test_anchor_retriever_is_deterministic(self) -> None:
        retriever = BM25AnchorRetriever(self.catalog)
        first = retriever.retrieve("unsupported phrase", top_n=3)
        second = retriever.retrieve("unsupported phrase", top_n=3)
        self.assertEqual(
            [(item.parent_asin, item.rank) for item in first],
            [(item.parent_asin, item.rank) for item in second],
        )


if __name__ == "__main__":
    unittest.main()
