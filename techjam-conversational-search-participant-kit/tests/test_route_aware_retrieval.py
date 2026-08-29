from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

from starter.agent import Agent
from starter.hybrid_retrieval import (
    Candidate,
    HybridRetrievalConfig,
    RouteRetrievalPolicy,
    default_route_policies,
    merge_candidates,
)
from starter.route_aware_retrieval import (
    merge_fallback_candidates,
    route_reciprocal_rank_fusion,
)


@dataclass(frozen=True)
class FakeResult:
    parent_asin: str
    score: float
    rank: int


class FakeRetriever:
    def __init__(self, results=None, error: Exception | None = None) -> None:
        self.results = list(results or [])
        self.error = error
        self.calls: list[tuple[object, int]] = []

    def retrieve(self, query: object, top_n: int = 200):
        self.calls.append((query, top_n))
        if self.error is not None:
            raise self.error
        return self.results[:top_n]


class FixedRouter:
    def __init__(self, route: str = "buying", error: Exception | None = None) -> None:
        self.route_name = route
        self.error = error
        self.calls: list[tuple[object, object]] = []

    def route(self, state: object, query: object):
        self.calls.append((state, query))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            route=self.route_name,
            confidence=0.9,
            reasons=("test",),
            policy_id=f"router.{self.route_name}",
        )


class FakeFallback:
    def __init__(self, results=None, error: Exception | None = None) -> None:
        self.results = list(results or [])
        self.error = error
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.results[: kwargs["top_n"]]


def fallback(parent_asin: str, score: float, rank: int) -> object:
    return SimpleNamespace(
        parent_asin=parent_asin,
        fallback_score=score,
        fallback_rank=rank,
        rank=rank,
    )


class RouteAwareAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.catalog = Path(self.temporary.name) / "catalog.jsonl"
        products = [
            {
                "parent_asin": "MATCH",
                "title": "White canvas sneakers",
                "categories": ["Shoes", "Sneakers"],
                "color": "white",
                "material": "canvas",
                "price": 80,
            },
            {
                "parent_asin": "WRONG_CATEGORY",
                "title": "White canvas handbag",
                "categories": ["Accessories", "Bags"],
                "color": "white",
                "material": "canvas",
                "price": 60,
            },
            {
                "parent_asin": "EXPENSIVE",
                "title": "White canvas sneakers",
                "categories": ["Shoes", "Sneakers"],
                "color": "white",
                "material": "canvas",
                "price": 140,
            },
            {
                "parent_asin": "EXCLUDED",
                "title": "White leather sneakers",
                "categories": ["Shoes", "Sneakers"],
                "color": "white",
                "material": "leather",
                "price": 70,
            },
            {
                "parent_asin": "MISSING_METADATA",
                "title": "Comfortable versatile travel item",
            },
            {
                "parent_asin": "BOUNDARY",
                "title": "Popular classic product",
                "average_rating": 5,
                "rating_number": 500,
            },
        ]
        self.catalog.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_agent(
        self,
        route: str = "buying",
        *,
        lexical=None,
        dense=None,
        fallback_generator=None,
        router=None,
        config: HybridRetrievalConfig | None = None,
    ) -> Agent:
        return Agent(
            self.catalog,
            config=config or HybridRetrievalConfig(mode="route-aware"),
            lexical_retriever=lexical or FakeRetriever(),
            dense_retriever=dense or FakeRetriever(),
            router=router or FixedRouter(route),
            fallback_generator=fallback_generator or FakeFallback(),
        )

    def test_buying_uses_configured_weights_and_candidate_pool_sizes(self) -> None:
        policies = default_route_policies()
        policies["buying"] = replace(
            policies["buying"],
            lexical_weight=4.0,
            dense_weight=0.25,
            lexical_candidate_count=7,
            dense_candidate_count=9,
        )
        lexical = FakeRetriever([FakeResult("MATCH", 1, 1)])
        dense = FakeRetriever([FakeResult("MISSING_METADATA", 1, 1)])
        agent = self.make_agent(
            lexical=lexical,
            dense=dense,
            config=HybridRetrievalConfig(mode="route-aware", route_policies=policies),
        )
        agent.reset("s", {})

        response = agent.respond("s", "I need sneakers", 1, 2)

        self.assertEqual(response["recommendations"][0]["parent_asin"], "MATCH")
        self.assertEqual(lexical.calls[0][1], 7)
        self.assertEqual(dense.calls[0][1], 9)
        candidate = agent.last_candidates("s")[0]
        self.assertGreater(
            candidate.component_scores["lexical"],
            candidate.component_scores["dense"],
        )

    def test_dense_candidate_cannot_bypass_buying_hard_constraints(self) -> None:
        lexical = FakeRetriever([FakeResult("MATCH", 1, 1)])
        dense = FakeRetriever(
            [FakeResult("EXPENSIVE", 1, 1), FakeResult("WRONG_CATEGORY", 0.9, 2)]
        )
        agent = self.make_agent(lexical=lexical, dense=dense)
        agent.reset("s", {})

        response = agent.respond("s", "I need white sneakers under $100", 1, 10)

        self.assertEqual(response["recommendations"], [{"parent_asin": "MATCH"}])
        diagnostics = agent.diagnostics_snapshot("s")["turns"][0]
        self.assertEqual(diagnostics["filter_counts"]["hard_constraint_removals"], 2)

    def test_buying_respects_category_price_and_explicit_exclusion(self) -> None:
        results = [
            FakeResult("MATCH", 4, 1),
            FakeResult("WRONG_CATEGORY", 3, 2),
            FakeResult("EXPENSIVE", 2, 3),
            FakeResult("EXCLUDED", 1, 4),
        ]
        agent = self.make_agent(lexical=FakeRetriever(results))
        agent.reset("s", {})

        response = agent.respond(
            "s", "I need white sneakers under $100, not leather", 1, 10
        )

        self.assertEqual(response["recommendations"], [{"parent_asin": "MATCH"}])

    def test_browsing_keeps_semantic_candidate_with_missing_optional_metadata(
        self,
    ) -> None:
        dense = FakeRetriever([FakeResult("MISSING_METADATA", 0.9, 1)])
        agent = self.make_agent("browsing", dense=dense)
        agent.reset("s", {})

        response = agent.respond("s", "white comfortable travel ideas", 1, 10)

        self.assertEqual(
            response["recommendations"], [{"parent_asin": "MISSING_METADATA"}]
        )
        self.assertEqual(dense.calls[0][1], 400)
        diagnostic = agent.diagnostics_snapshot("s")["turns"][0]
        self.assertEqual(diagnostic["filter_counts"]["hard_constraint_removals"], 0)

    def test_boundary_invokes_fallback_and_preserves_profile_as_soft_evidence(
        self,
    ) -> None:
        generated = FakeFallback([fallback("BOUNDARY", 0.8, 1)])
        agent = self.make_agent("boundary", fallback_generator=generated)
        agent.reset("s", {"preference_tags": ["classic"]})

        response = agent.respond("s", "show me something", 1, 10)

        self.assertEqual(response["recommendations"], [{"parent_asin": "BOUNDARY"}])
        self.assertEqual(len(generated.calls), 1)
        call = generated.calls[0]
        self.assertEqual(call["user_profile"], {"preference_tags": ["classic"]})
        diagnostic = agent.diagnostics_snapshot("s")["turns"][0]
        self.assertTrue(diagnostic["fallback_attempted"])
        self.assertTrue(diagnostic["fallback_succeeded"])
        candidate = agent.last_candidates("s")[0]
        self.assertEqual(candidate.sources, {"fallback"})
        self.assertEqual(candidate.fallback_rank, 1)

    def test_boundary_policy_does_not_invent_hard_constraints(self) -> None:
        generated = FakeFallback([fallback("MISSING_METADATA", 0.5, 1)])
        router = FixedRouter("boundary")
        agent = self.make_agent(router=router, fallback_generator=generated)
        agent.reset("s", {})

        agent.respond("s", "anything is fine", 1, 10)

        query = generated.calls[0]["query"]
        self.assertTrue(
            all(
                getattr(query, name) is None
                for name in (
                    "category",
                    "color",
                    "style",
                    "material",
                    "use_case",
                    "price",
                )
            )
        )

    def test_repeated_boundary_state_uses_reset_scoped_fallback_cache(self) -> None:
        generated = FakeFallback([fallback("BOUNDARY", 0.8, 1)])
        agent = self.make_agent("boundary", fallback_generator=generated)
        agent.reset("s", {})

        agent.respond("s", "show me something", 1, 10)
        agent.respond("s", "show me something", 2, 10)

        self.assertEqual(len(generated.calls), 1)
        turns = agent.diagnostics_snapshot("s")["turns"]
        self.assertFalse(turns[0]["fallback_cache_hit"])
        self.assertTrue(turns[1]["fallback_cache_hit"])

        agent.reset("s", {})
        agent.respond("s", "show me something", 1, 10)
        self.assertEqual(len(generated.calls), 2)

    def test_fallback_is_not_invoked_for_buying_or_browsing(self) -> None:
        for route in ("buying", "browsing"):
            with self.subTest(route=route):
                generated = FakeFallback([fallback("BOUNDARY", 1, 1)])
                agent = self.make_agent(
                    route,
                    lexical=FakeRetriever([FakeResult("MATCH", 1, 1)]),
                    fallback_generator=generated,
                )
                agent.reset(route, {})
                agent.respond(route, "I need sneakers", 1, 10)
                self.assertEqual(generated.calls, [])

    def test_uncertain_and_invalid_router_output_use_safe_fixed_hybrid(self) -> None:
        for router, routing_failed in (
            (FixedRouter("uncertain"), False),
            (FixedRouter("malformed"), True),
            (FixedRouter(error=RuntimeError("router down")), True),
        ):
            with self.subTest(router=router.route_name):
                lexical = FakeRetriever([FakeResult("MATCH", 1, 1)])
                dense = FakeRetriever([FakeResult("MISSING_METADATA", 1, 1)])
                agent = self.make_agent(router=router, lexical=lexical, dense=dense)
                agent.reset("s", {})
                with (
                    self.assertLogs("starter.agent", level="WARNING")
                    if routing_failed
                    else _null_logs()
                ):
                    agent.respond("s", "maybe sneakers", 1, 10)
                diagnostic = agent.diagnostics_snapshot("s")["turns"][0]
                self.assertEqual(diagnostic["selected_route"], "uncertain")
                self.assertEqual(
                    diagnostic["applied_policy_id"], "retrieval.safe-default.v1"
                )
                self.assertEqual(diagnostic["routing_failed"], routing_failed)
                by_id = {item.parent_asin: item for item in agent.last_candidates("s")}
                self.assertAlmostEqual(
                    by_id["MATCH"].component_scores["lexical"],
                    by_id["MISSING_METADATA"].component_scores["dense"],
                )

    def test_dense_failure_degrades_to_lexical_and_records_failure(self) -> None:
        agent = self.make_agent(
            lexical=FakeRetriever([FakeResult("MATCH", 1, 1)]),
            dense=FakeRetriever(error=RuntimeError("encoder failed")),
        )
        agent.reset("s", {})

        with self.assertLogs("starter.agent", level="WARNING"):
            response = agent.respond("s", "I need sneakers", 1, 10)

        self.assertEqual(response["recommendations"], [{"parent_asin": "MATCH"}])
        diagnostic = agent.diagnostics_snapshot("s")["turns"][0]
        self.assertIn("dense", diagnostic["component_failures"])
        self.assertIn("lexical", diagnostic["retrievers_successful"])

    def test_lexical_failure_degrades_to_dense(self) -> None:
        agent = self.make_agent(
            "browsing",
            lexical=FakeRetriever(error=RuntimeError("fts failed")),
            dense=FakeRetriever([FakeResult("MISSING_METADATA", 1, 1)]),
        )
        agent.reset("s", {})
        with self.assertLogs("starter.agent", level="WARNING"):
            response = agent.respond("s", "comfortable travel ideas", 1, 10)
        self.assertEqual(
            response["recommendations"], [{"parent_asin": "MISSING_METADATA"}]
        )

    def test_fallback_failure_does_not_crash_session(self) -> None:
        agent = self.make_agent(
            "boundary",
            lexical=FakeRetriever([FakeResult("BOUNDARY", 1, 1)]),
            fallback_generator=FakeFallback(error=RuntimeError("fallback failed")),
        )
        agent.reset("s", {})
        with self.assertLogs("starter.agent", level="WARNING"):
            response = agent.respond("s", "show me something", 1, 10)
        self.assertEqual(response["recommendations"], [{"parent_asin": "BOUNDARY"}])
        diagnostic = agent.diagnostics_snapshot("s")["turns"][0]
        self.assertIn("fallback", diagnostic["component_failures"])

    def test_provenance_ranks_scores_and_exact_deduplication_survive(self) -> None:
        lexical = FakeRetriever(
            [FakeResult("MATCH", 4.0, 2), FakeResult("MATCH", 2.0, 4)]
        )
        dense = FakeRetriever([FakeResult("MATCH", 0.9, 1)])
        generated = FakeFallback([fallback("MATCH", 0.7, 3)])
        agent = self.make_agent(
            "boundary", lexical=lexical, dense=dense, fallback_generator=generated
        )
        agent.reset("s", {})
        agent.respond("s", "show me something white", 1, 10)

        candidates = agent.last_candidates("s")
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.sources, {"lexical", "dense", "fallback"})
        self.assertEqual(candidate.lexical_rank, 2)
        self.assertEqual(candidate.lexical_score, 4.0)
        self.assertEqual(candidate.dense_rank, 1)
        self.assertEqual(candidate.dense_score, 0.9)
        self.assertEqual(candidate.fallback_rank, 3)
        self.assertEqual(candidate.fallback_score, 0.7)
        self.assertEqual(
            set(candidate.component_scores), {"lexical", "dense", "fallback"}
        )

    def test_malformed_invalid_and_duplicate_candidates_are_safe(self) -> None:
        malformed = SimpleNamespace(parent_asin="MATCH", score=1, rank=0)
        invalid = FakeResult("NOT_IN_CATALOG", 1, 1)
        agent = self.make_agent(
            "browsing",
            lexical=FakeRetriever([malformed, invalid, FakeResult("MATCH", 1, 1)]),
            dense=FakeRetriever([FakeResult("MATCH", 0.9, 1)]),
        )
        agent.reset("s", {})
        response = agent.respond("s", "travel ideas", 1, 10)
        self.assertEqual(response["recommendations"], [{"parent_asin": "MATCH"}])

    def test_reset_clears_session_state_candidates_and_diagnostics(self) -> None:
        agent = self.make_agent(lexical=FakeRetriever([FakeResult("MATCH", 1, 1)]))
        agent.reset("s", {})
        agent.respond("s", "I need white sneakers", 1, 10)
        self.assertTrue(agent.diagnostics_snapshot("s")["turns"])
        self.assertTrue(agent.last_candidates("s"))

        agent.reset("s", {})

        self.assertEqual(agent.diagnostics_snapshot("s")["turns"], [])
        self.assertEqual(agent.last_candidates("s"), [])
        query = agent._state.query_for("s")
        self.assertIsNone(query.category)

    def test_respond_and_reset_official_contracts_remain_unchanged(self) -> None:
        self.assertEqual(
            list(inspect.signature(Agent.reset).parameters),
            ["self", "session_id", "user_profile"],
        )
        self.assertEqual(
            list(inspect.signature(Agent.respond).parameters),
            ["self", "session_id", "user_message", "turn", "top_k"],
        )
        agent = self.make_agent(lexical=FakeRetriever([FakeResult("MATCH", 1, 1)]))
        agent.reset("s", {})
        response = agent.respond("s", "I need sneakers", 1, 10)
        self.assertEqual(
            set(response), {"message", "ask_attribute", "recommendations", "usage"}
        )


class RouteAwarePrimitiveTest(unittest.TestCase):
    def test_route_fusion_tie_breaking_is_deterministic(self) -> None:
        policy = default_route_policies()["uncertain"]
        candidates = [Candidate("B", lexical_rank=1), Candidate("A", dense_rank=1)]
        first = route_reciprocal_rank_fusion(candidates, policy, limit=10)
        second = route_reciprocal_rank_fusion(
            list(reversed(candidates)), policy, limit=10
        )
        self.assertEqual([item.parent_asin for item in first], ["B", "A"])
        self.assertEqual(
            [item.parent_asin for item in first],
            [item.parent_asin for item in second],
        )

    def test_exact_asin_merge_does_not_normalize_identity(self) -> None:
        merged = merge_candidates(
            [FakeResult("A", 1, 1), FakeResult(" A", 1, 2)],
            [FakeResult("a", 1, 1)],
            {"A", " A", "a"},
        )
        self.assertEqual({item.parent_asin for item in merged}, {"A", " A", "a"})

    def test_fallback_merge_rejects_invalid_and_deduplicates_exact_asins(self) -> None:
        merged = merge_fallback_candidates(
            [Candidate("A", lexical_rank=1, sources={"lexical"})],
            [fallback("A", 1, 1), fallback("", 1, 2), fallback("BAD", 1, 3)],
            {"A"},
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].sources, {"lexical", "fallback"})

    def test_route_configuration_is_complete_and_validated(self) -> None:
        config = HybridRetrievalConfig(mode="route-aware")
        self.assertEqual(
            set(config.route_policies), {"buying", "browsing", "boundary", "uncertain"}
        )
        self.assertTrue(config.route_policies["buying"].apply_hard_filters)
        self.assertFalse(config.route_policies["browsing"].apply_hard_filters)
        with self.assertRaisesRegex(ValueError, "route_policies"):
            HybridRetrievalConfig(mode="route-aware", route_policies={})
        with self.assertRaisesRegex(ValueError, "lexical_weight"):
            RouteRetrievalPolicy("bad", lexical_weight=-1, dense_weight=1)


class _null_logs:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


if __name__ == "__main__":
    unittest.main()
