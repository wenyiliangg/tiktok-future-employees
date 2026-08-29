from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from starter.agent import Agent
from starter.hybrid_retrieval import (
    Candidate,
    HybridRetrievalConfig,
    dense_candidate,
    lexical_candidate,
    merge_candidates,
    reciprocal_rank_fusion,
)


@dataclass(frozen=True)
class FakeResult:
    parent_asin: str
    score: float
    rank: int


class FakeRetriever:
    def __init__(
        self, results: list[FakeResult] | None = None, error: Exception | None = None
    ) -> None:
        self.results = results or []
        self.error = error
        self.calls: list[tuple[object, int]] = []

    def retrieve(self, query: object, top_n: int = 200) -> list[FakeResult]:
        self.calls.append((query, top_n))
        if self.error is not None:
            raise self.error
        return self.results[:top_n]


class HybridCandidateTest(unittest.TestCase):
    def test_adapters_preserve_one_based_rank_score_and_source(self) -> None:
        lexical = lexical_candidate(FakeResult("A", 2.5, 1))
        dense = dense_candidate(FakeResult("B", 0.75, 2))
        self.assertEqual(
            (
                lexical.parent_asin,
                lexical.lexical_score,
                lexical.lexical_rank,
                lexical.sources,
            ),
            ("A", 2.5, 1, {"lexical"}),
        )
        self.assertEqual(
            (dense.parent_asin, dense.dense_score, dense.dense_rank, dense.sources),
            ("B", 0.75, 2, {"dense"}),
        )
        with self.assertRaisesRegex(ValueError, "one-based"):
            dense_candidate(FakeResult("B", 0.1, 0))

    def test_exact_merge_deduplicates_and_filters_invalid_asins(self) -> None:
        merged = merge_candidates(
            [FakeResult("LEX", 3.0, 1), FakeResult("BOTH", 2.0, 2)],
            [
                FakeResult("BOTH", 0.9, 1),
                FakeResult("DENSE", 0.8, 2),
                FakeResult("BAD", 1.0, 3),
            ],
            {"LEX", "BOTH", "DENSE"},
        )
        by_id = {candidate.parent_asin: candidate for candidate in merged}
        self.assertEqual(set(by_id), {"LEX", "BOTH", "DENSE"})
        self.assertEqual(by_id["BOTH"].sources, {"lexical", "dense"})
        self.assertEqual(by_id["BOTH"].lexical_rank, 2)
        self.assertEqual(by_id["BOTH"].dense_rank, 1)
        self.assertEqual(by_id["BOTH"].lexical_score, 2.0)
        self.assertEqual(by_id["BOTH"].dense_score, 0.9)

    def test_rrf_formula_weights_missing_sources_limit_and_empty_inputs(self) -> None:
        config = HybridRetrievalConfig(
            lexical_weight=2.0,
            dense_weight=3.0,
            rrf_k=10,
            final_candidate_count=2,
        )
        candidates = [
            Candidate("BOTH", lexical_rank=1, dense_rank=2),
            Candidate("LEX", lexical_rank=2),
            Candidate("DENSE", dense_rank=1),
        ]
        result = reciprocal_rank_fusion(candidates, config)
        self.assertEqual([item.parent_asin for item in result], ["BOTH", "DENSE"])
        self.assertAlmostEqual(result[0].fusion_score, 2 / 11 + 3 / 12)
        self.assertAlmostEqual(candidates[1].fusion_score, 2 / 12)
        self.assertEqual(reciprocal_rank_fusion([], config), [])

    def test_rrf_tie_break_is_deterministic(self) -> None:
        config = HybridRetrievalConfig(rrf_k=60, final_candidate_count=10)
        candidates = [
            Candidate("Z", lexical_rank=2),
            Candidate("B", dense_rank=2),
            Candidate("A", dense_rank=2),
            Candidate("BEST", lexical_rank=1),
        ]
        orders = [
            [
                item.parent_asin
                for item in reciprocal_rank_fusion(list(reversed(candidates)), config)
            ],
            [item.parent_asin for item in reciprocal_rank_fusion(candidates, config)],
        ]
        self.assertEqual(orders[0], orders[1])
        self.assertEqual(orders[0], ["BEST", "Z", "A", "B"])

    def test_invalid_mode_and_config_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid retrieval mode"):
            HybridRetrievalConfig(mode="automatic")
        with self.assertRaisesRegex(ValueError, "positive integer"):
            HybridRetrievalConfig(final_candidate_count=0)


class AgentIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.catalog = Path(self.temporary.name) / "catalog.jsonl"
        rows = [
            {"parent_asin": "A", "title": "white sneakers", "categories": ["shoes"]},
            {"parent_asin": "B", "title": "black sneakers", "categories": ["shoes"]},
            {"parent_asin": "C", "title": "red sneakers", "categories": ["shoes"]},
        ]
        self.catalog.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_agent(
        self,
        mode: str,
        lexical: FakeRetriever | None = None,
        dense: FakeRetriever | None = None,
        dense_factory=None,
    ) -> Agent:
        return Agent(
            self.catalog,
            config=HybridRetrievalConfig(mode=mode, final_candidate_count=10),
            lexical_retriever=lexical,
            dense_retriever=dense,
            dense_factory=dense_factory,
        )

    def test_agent_contract_signatures_are_unchanged(self) -> None:
        self.assertEqual(
            list(inspect.signature(Agent.reset).parameters),
            ["self", "session_id", "user_profile"],
        )
        self.assertEqual(
            list(inspect.signature(Agent.respond).parameters),
            ["self", "session_id", "user_message", "turn", "top_k"],
        )

    def test_modes_call_only_required_retrievers_and_preserve_schema(self) -> None:
        for mode, expected_lexical, expected_dense in (
            ("lexical", 1, 0),
            ("dense", 0, 1),
            ("hybrid", 1, 1),
        ):
            with self.subTest(mode=mode):
                lexical = FakeRetriever(
                    [FakeResult("A", 2.0, 1), FakeResult("B", 1.0, 2)]
                )
                dense = FakeRetriever(
                    [FakeResult("B", 0.9, 1), FakeResult("C", 0.8, 2)]
                )
                agent = self.make_agent(mode, lexical, dense)
                agent.reset("session", {})
                response = agent.respond("session", "white sneakers", 1, 2)
                self.assertEqual(len(lexical.calls), expected_lexical)
                self.assertEqual(len(dense.calls), expected_dense)
                self.assertEqual(
                    set(response),
                    {"message", "ask_attribute", "recommendations", "usage"},
                )
                self.assertLessEqual(len(response["recommendations"]), 2)
                ids = [item["parent_asin"] for item in response["recommendations"]]
                self.assertEqual(len(ids), len(set(ids)))
                self.assertTrue(set(ids) <= {"A", "B", "C"})

    def test_agent_reranks_a_bounded_pool_before_applying_top_k(self) -> None:
        lexical = FakeRetriever(
            [FakeResult("A", 10.0, 1), FakeResult("C", 1.0, 2)]
        )
        agent = self.make_agent("lexical", lexical)
        agent.reset("rerank", {})
        response = agent.respond("rerank", "red sneakers", 1, 1)
        self.assertEqual(response["recommendations"], [{"parent_asin": "C"}])

    def test_both_retrievers_receive_same_overridden_active_state(self) -> None:
        lexical = FakeRetriever([FakeResult("A", 1.0, 1)])
        dense = FakeRetriever([FakeResult("A", 1.0, 1)])
        agent = self.make_agent("hybrid", lexical, dense)
        agent.reset("override", {"preference_tags": ["red"]})
        agent.respond("override", "I need red sneakers", 1, 3)
        agent.respond("override", "Actually, make them black instead", 2, 3)
        structured_query = lexical.calls[-1][0]
        dense_text = dense.calls[-1][0]
        self.assertEqual(structured_query.text, dense_text)
        self.assertIn("black", str(dense_text))
        self.assertNotIn("red", str(dense_text))

    def test_reset_clears_state_without_rebuilding_retrievers(self) -> None:
        lexical = FakeRetriever([FakeResult("A", 1.0, 1)])
        agent = self.make_agent("lexical", lexical)
        agent.reset("same", {})
        agent.respond("same", "red sneakers", 1, 1)
        agent.reset("same", {})
        agent.respond("same", "black sneakers", 1, 1)
        self.assertEqual(lexical.calls[-1][0].text, "sneakers black")
        self.assertNotIn("red", lexical.calls[-1][0].text)

    def test_hybrid_initialization_failure_is_logged_once_and_not_retried(self) -> None:
        lexical = FakeRetriever([FakeResult("A", 1.0, 1)])
        attempts = 0

        def fail():
            nonlocal attempts
            attempts += 1
            raise RuntimeError("missing cache")

        with self.assertLogs("starter.agent", level="WARNING") as captured:
            agent = self.make_agent("hybrid", lexical, dense_factory=fail)
            agent.reset("failure", {})
            first = agent.respond("failure", "white sneakers", 1, 2)
            second = agent.respond("failure", "black sneakers", 2, 2)
        self.assertEqual(attempts, 1)
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(first["recommendations"], [{"parent_asin": "A"}])
        self.assertEqual(second["recommendations"], [{"parent_asin": "A"}])

    def test_hybrid_query_failure_and_empty_dense_fall_back_to_lexical(self) -> None:
        for dense in (
            FakeRetriever(error=RuntimeError("encode failed")),
            FakeRetriever([]),
        ):
            with self.subTest(error=dense.error):
                lexical = FakeRetriever(
                    [FakeResult("A", 1.0, 1), FakeResult("A", 0.5, 2)]
                )
                agent = self.make_agent("hybrid", lexical, dense)
                agent.reset("fallback", {})
                with self.assertLogs("starter.agent", level="WARNING"):
                    response = agent.respond("fallback", "white sneakers", 1, 10)
                self.assertEqual(response["recommendations"], [{"parent_asin": "A"}])

    def test_empty_results_and_zero_top_k_are_safe(self) -> None:
        lexical = FakeRetriever([])
        agent = self.make_agent("lexical", lexical)
        agent.reset("empty", {})
        self.assertEqual(
            agent.respond("empty", "white sneakers", 1, 10)["recommendations"], []
        )
        self.assertEqual(
            agent.respond("empty", "white sneakers", 2, 0)["recommendations"], []
        )


if __name__ == "__main__":
    unittest.main()
