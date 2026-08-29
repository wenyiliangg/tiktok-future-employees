from __future__ import annotations

import json
import math
import sys
import tempfile
import types
import unittest
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from starter.agent import Agent
from starter.hybrid_retrieval import Candidate, HybridRetrievalConfig
from starter.semantic_reranker import (
    CrossEncoderPairScorer,
    SemanticReranker,
    SemanticRerankerConfig,
)


class FakeScorer:
    def __init__(
        self,
        scores: list[Any] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.model_name = "tests/fake-cross-encoder"
        self.model_revision: str | None = "test-revision"
        self.model_size_bytes = 1234
        self.cache_hit = False
        self.resolved_model_revision = "resolved-test-revision"
        self.scores = scores or []
        self.error = error
        self.load_calls = 0
        self.score_calls: list[tuple[list[tuple[str, str]], int]] = []

    def ensure_loaded(self) -> None:
        self.load_calls += 1
        if self.error is not None:
            raise self.error

    def score(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
    ) -> Sequence[float]:
        self.score_calls.append((list(pairs), batch_size))
        if self.error is not None:
            raise self.error
        return cast(Sequence[float], self.scores)


@dataclass(frozen=True)
class FakeResult:
    parent_asin: str
    score: float
    rank: int


class FakeRetriever:
    def __init__(self, results: list[FakeResult]) -> None:
        self.results = results
        self.calls: list[int] = []

    def retrieve(self, query: object, top_n: int = 200) -> list[FakeResult]:
        del query
        self.calls.append(top_n)
        return self.results[:top_n]


class SemanticRerankerTest(unittest.TestCase):
    def test_feature_is_disabled_by_default_and_config_is_validated(self) -> None:
        self.assertFalse(SemanticRerankerConfig().enabled)
        for kwargs, message in (
            ({"candidate_count": 0}, "candidate_count"),
            ({"batch_size": 0}, "batch_size"),
            ({"max_length": 0}, "max_length"),
            ({"model_name": ""}, "model_name"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    SemanticRerankerConfig(**kwargs)

    def test_only_bounded_prefix_is_batched_and_tail_order_is_unchanged(self) -> None:
        config = SemanticRerankerConfig(
            enabled=True,
            candidate_count=2,
            batch_size=7,
        )
        scorer = FakeScorer([0.1, 0.9])
        reranker = SemanticReranker(
            {"A": "alpha text", "B": "beta text", "C": "gamma text"},
            config=config,
            scorer=scorer,
        )
        candidates = [Candidate("A"), Candidate("B"), Candidate("C")]

        ranked = reranker.rerank("active customer intent", candidates)

        self.assertEqual([item.parent_asin for item in ranked], ["B", "A", "C"])
        self.assertEqual(
            scorer.score_calls,
            [
                (
                    [
                        ("active customer intent", "alpha text"),
                        ("active customer intent", "beta text"),
                    ],
                    7,
                )
            ],
        )
        self.assertEqual(scorer.load_calls, 1)
        self.assertEqual(candidates[0].semantic_score, 0.1)
        self.assertEqual(candidates[1].semantic_score, 0.9)
        self.assertIsNone(candidates[2].semantic_score)
        metrics = reranker.metrics_snapshot()
        self.assertEqual(metrics["query_count"], 1)
        self.assertEqual(metrics["scored_candidate_count"], 2)
        self.assertEqual(metrics["max_scored_candidates_per_query"], 2)
        self.assertEqual(metrics["failure_count"], 0)

    def test_ties_preserve_base_order_and_model_load_is_once_per_instance(self) -> None:
        config = SemanticRerankerConfig(enabled=True, candidate_count=3)
        scorer = FakeScorer([1.0, 1.0, 0.0])
        reranker = SemanticReranker(
            {"A": "a", "B": "b", "C": "c"},
            config=config,
            scorer=scorer,
        )
        candidates = [Candidate("A"), Candidate("B"), Candidate("C")]
        first = reranker.rerank("query", candidates)
        second = reranker.rerank("query", candidates)
        self.assertEqual([item.parent_asin for item in first], ["A", "B", "C"])
        self.assertEqual([item.parent_asin for item in second], ["A", "B", "C"])
        self.assertEqual(scorer.load_calls, 1)
        self.assertEqual(len(scorer.score_calls), 2)

    def test_load_inference_and_output_failures_return_exact_original_order(
        self,
    ) -> None:
        cases = (
            FakeScorer(error=RuntimeError("model unavailable")),
            FakeScorer([0.5]),
            FakeScorer([0.5, math.nan]),
            FakeScorer([[0.1, 0.2], [0.3, 0.4]]),
        )
        for scorer in cases:
            with self.subTest(scorer=scorer):
                config = SemanticRerankerConfig(enabled=True, candidate_count=2)
                reranker = SemanticReranker(
                    {"A": "a", "B": "b"}, config=config, scorer=scorer
                )
                original = [Candidate("A"), Candidate("B")]
                result = reranker.rerank("query", original)
                self.assertEqual(result, original)
                self.assertIs(result[0], original[0])
                self.assertIs(result[1], original[1])
                self.assertIsNone(original[0].semantic_score)
                self.assertIsNone(original[1].semantic_score)
                self.assertEqual(reranker.metrics_snapshot()["failure_count"], 1)

    def test_missing_product_text_falls_back_without_scoring(self) -> None:
        config = SemanticRerankerConfig(enabled=True, candidate_count=2)
        scorer = FakeScorer([1.0, 0.0])
        reranker = SemanticReranker({"A": "a"}, config=config, scorer=scorer)
        original = [Candidate("A"), Candidate("MISSING")]
        self.assertEqual(reranker.rerank("query", original), original)
        self.assertEqual(scorer.load_calls, 0)
        self.assertEqual(scorer.score_calls, [])

    def test_catalog_product_text_is_deterministic(self) -> None:
        config = SemanticRerankerConfig(enabled=True, candidate_count=1)
        first_scorer = FakeScorer([1.0])
        second_scorer = FakeScorer([1.0])
        with tempfile.TemporaryDirectory() as temporary:
            first_path = Path(temporary) / "first.jsonl"
            second_path = Path(temporary) / "second.jsonl"
            first = {
                "parent_asin": "A",
                "title": "  Blue   Trail Shoe ",
                "details": {"Style": "Sport", "Material": "Mesh"},
                "features": ["Lightweight", "Grippy sole"],
            }
            second = {
                "features": ["Lightweight", "Grippy sole"],
                "details": {"Material": "Mesh", "Style": "Sport"},
                "title": "Blue Trail Shoe",
                "parent_asin": "A",
            }
            first_path.write_text(json.dumps(first) + "\n", encoding="utf-8")
            second_path.write_text(json.dumps(second) + "\n", encoding="utf-8")
            first_reranker = SemanticReranker.from_catalog(
                first_path, config=config, scorer=first_scorer
            )
            second_reranker = SemanticReranker.from_catalog(
                second_path, config=config, scorer=second_scorer
            )
            first_reranker.rerank("query", [Candidate("A")])
            second_reranker.rerank("query", [Candidate("A")])

        first_text = first_scorer.score_calls[0][0][0][1]
        second_text = second_scorer.score_calls[0][0][0][1]
        self.assertEqual(first_text, second_text)
        self.assertEqual(
            first_text,
            "Title: Blue Trail Shoe\nFeatures: Lightweight; Grippy sole\n"
            "Attributes: Material: Mesh; Style: Sport",
        )

    def test_production_model_cache_is_shared_by_equivalent_scorers(self) -> None:
        constructed: list[tuple[str, dict[str, object]]] = []

        class FakeParameter:
            def numel(self) -> int:
                return 10

            def element_size(self) -> int:
                return 4

        class FakeTransformer:
            config = types.SimpleNamespace(_commit_hash="resolved")

            @staticmethod
            def parameters() -> list[FakeParameter]:
                return [FakeParameter()]

        class FakeCrossEncoder:
            def __init__(self, name: str, **kwargs: object) -> None:
                constructed.append((name, kwargs))
                self.model = FakeTransformer()

            def predict(
                self, pairs: Sequence[tuple[str, str]], **kwargs: object
            ) -> list[float]:
                del kwargs
                return [0.0 for _ in pairs]

        module = types.ModuleType("sentence_transformers")
        module.CrossEncoder = FakeCrossEncoder  # type: ignore[attr-defined]
        model_name = f"tests/cache-{uuid.uuid4().hex}"
        first = CrossEncoderPairScorer(
            model_name, model_revision="r1", device="cpu", max_length=128
        )
        second = CrossEncoderPairScorer(
            model_name, model_revision="r1", device="cpu", max_length=128
        )
        with patch.dict(sys.modules, {"sentence_transformers": module}):
            first.ensure_loaded()
            second.ensure_loaded()

        self.assertEqual(len(constructed), 1)
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(first.model_size_bytes, 40)
        self.assertEqual(second.model_size_bytes, 40)
        self.assertEqual(first.resolved_model_revision, "resolved")


class AgentSemanticIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.catalog = Path(self.temporary.name) / "catalog.jsonl"
        rows = [
            {"parent_asin": "A", "title": "alpha shoe"},
            {"parent_asin": "B", "title": "beta shoe"},
            {"parent_asin": "C", "title": "gamma shoe"},
        ]
        self.catalog.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_agent_fetches_rerank_window_but_returns_requested_count(self) -> None:
        semantic_config = SemanticRerankerConfig(
            enabled=True, candidate_count=3, batch_size=2
        )
        scorer = FakeScorer([0.1, 0.2, 0.9])
        semantic = SemanticReranker.from_catalog(
            self.catalog, config=semantic_config, scorer=scorer
        )
        lexical = FakeRetriever(
            [
                FakeResult("A", 3.0, 1),
                FakeResult("B", 2.0, 2),
                FakeResult("C", 1.0, 3),
            ]
        )
        agent = Agent(
            self.catalog,
            config=HybridRetrievalConfig(
                mode="lexical", lexical_candidate_count=20, final_candidate_count=10
            ),
            lexical_retriever=lexical,
            semantic_config=semantic_config,
            semantic_reranker=semantic,
        )
        agent.reset("session", {})
        response = agent.respond("session", "shoe", 1, 2)
        self.assertEqual(
            [item["parent_asin"] for item in response["recommendations"]],
            ["C", "B"],
        )
        self.assertEqual(len(scorer.score_calls[0][0]), 3)
        self.assertEqual(agent.semantic_reranker_metrics["failure_count"], 0)

    def test_agent_rejects_new_or_replaced_candidates(self) -> None:
        semantic_config = SemanticRerankerConfig(enabled=True, candidate_count=2)

        class InvalidReranker:
            @staticmethod
            def rerank(query_text: str, candidates: list[Candidate]) -> list[Candidate]:
                del query_text
                return [Candidate("NEW"), *candidates[1:]]

        lexical = FakeRetriever([FakeResult("A", 2.0, 1), FakeResult("B", 1.0, 2)])
        agent = Agent(
            self.catalog,
            config=HybridRetrievalConfig(mode="lexical", final_candidate_count=10),
            lexical_retriever=lexical,
            semantic_config=semantic_config,
            semantic_reranker=InvalidReranker(),  # type: ignore[arg-type]
        )
        agent.reset("session", {})
        response = agent.respond("session", "shoe", 1, 2)
        self.assertEqual(
            [item["parent_asin"] for item in response["recommendations"]],
            ["A", "B"],
        )
        self.assertEqual(agent.semantic_reranker_metrics["failure_count"], 1)

    def test_initialization_failure_keeps_base_ranking_operational(self) -> None:
        semantic_config = SemanticRerankerConfig(enabled=True, candidate_count=2)
        lexical = FakeRetriever([FakeResult("A", 2.0, 1), FakeResult("B", 1.0, 2)])
        with patch(
            "starter.agent.SemanticReranker.from_catalog",
            side_effect=RuntimeError("cannot build product text index"),
        ):
            agent = Agent(
                self.catalog,
                config=HybridRetrievalConfig(mode="lexical", final_candidate_count=10),
                lexical_retriever=lexical,
                semantic_config=semantic_config,
            )
        agent.reset("session", {})
        response = agent.respond("session", "shoe", 1, 2)
        self.assertEqual(
            [item["parent_asin"] for item in response["recommendations"]],
            ["A", "B"],
        )
        metrics = agent.semantic_reranker_metrics
        self.assertTrue(metrics["enabled"])
        self.assertFalse(metrics["operational"])
        self.assertEqual(metrics["initialization_failure_count"], 1)
        self.assertEqual(metrics["failure_count"], 1)

    def test_agent_restores_asin_mutation_before_fallback(self) -> None:
        semantic_config = SemanticRerankerConfig(enabled=True, candidate_count=2)

        class MutatingReranker:
            @staticmethod
            def rerank(query_text: str, candidates: list[Candidate]) -> list[Candidate]:
                del query_text
                candidates[0].parent_asin = "NEW"
                return list(reversed(candidates))

        lexical = FakeRetriever([FakeResult("A", 2.0, 1), FakeResult("B", 1.0, 2)])
        agent = Agent(
            self.catalog,
            config=HybridRetrievalConfig(mode="lexical", final_candidate_count=10),
            lexical_retriever=lexical,
            semantic_config=semantic_config,
            semantic_reranker=MutatingReranker(),  # type: ignore[arg-type]
        )
        agent.reset("session", {})
        response = agent.respond("session", "shoe", 1, 2)
        self.assertEqual(
            [item["parent_asin"] for item in response["recommendations"]],
            ["A", "B"],
        )
        self.assertEqual(agent.semantic_reranker_metrics["failure_count"], 1)


if __name__ == "__main__":
    unittest.main()
