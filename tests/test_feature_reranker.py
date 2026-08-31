from __future__ import annotations

import copy
import unittest

from starter.feature_reranker import (
    CatalogUnavailableError,
    FeatureReranker,
    InMemoryCatalogView,
    RerankerConfig,
)
from starter.hybrid_retrieval import Candidate
from starter.search_models import Constraint, PriceConstraint, SearchQuery


def constraint(
    value: str, strength: str = "soft", source: str = "current_turn"
) -> Constraint:
    return Constraint(value=value, strength=strength, source=source, updated_turn=1)  # type: ignore[arg-type]


def candidate(parent_asin: str, rank: int | None = None, score: float = 0.0) -> Candidate:
    return Candidate(
        parent_asin,
        lexical_score=score,
        lexical_rank=rank,
        sources={"lexical"} if rank is not None else set(),
    )


class FeatureRerankerTest(unittest.TestCase):
    def view(self, *products: dict) -> InMemoryCatalogView:
        return InMemoryCatalogView(products)

    def ids(self, values: list[Candidate]) -> list[str]:
        return [value.parent_asin for value in values]

    def reranker(self, **overrides: object) -> FeatureReranker:
        return FeatureReranker(RerankerConfig(**overrides))  # type: ignore[arg-type]

    def test_exact_category_match_ranks_above_incompatible_category(self) -> None:
        reranker = self.reranker(feature_weights={"category_match": 1.0})
        query = SearchQuery(text="shoes", category=constraint("sneakers"))
        catalog = self.view(
            {"parent_asin": "BAD", "categories": ["handbags"]},
            {"parent_asin": "GOOD", "categories": ["sneakers"]},
        )
        result = reranker.rerank(
            query, [candidate("BAD"), candidate("GOOD")], catalog, 2
        )
        self.assertEqual(self.ids(result), ["GOOD", "BAD"])

    def test_stronger_active_attribute_coverage_improves_rank(self) -> None:
        reranker = self.reranker(feature_weights={"attribute_coverage": 1.0})
        query = SearchQuery(
            text="shirt",
            color=constraint("red"),
            material=constraint("cotton"),
        )
        catalog = self.view(
            {
                "parent_asin": "ONE",
                "categories": ["shirt"],
                "details": {"Color": "red", "Material": "polyester"},
            },
            {
                "parent_asin": "TWO",
                "categories": ["shirt"],
                "details": {"Color": "red", "Material": "cotton"},
            },
        )
        result = reranker.rerank(
            query, [candidate("ONE"), candidate("TWO")], catalog, 2
        )
        self.assertEqual(self.ids(result), ["TWO", "ONE"])
        self.assertEqual(result[0].rerank_diagnostics["features"]["attribute_coverage"], 1.0)  # type: ignore[index]

    def test_price_compatible_product_ranks_above_incompatible_product(self) -> None:
        reranker = self.reranker(feature_weights={"price_compatibility": 1.0})
        query = SearchQuery(
            text="bag",
            price=PriceConstraint(minimum=20, maximum=40, strength="soft"),
        )
        catalog = self.view(
            {"parent_asin": "HIGH", "price": "$75.00"},
            {"parent_asin": "IN", "price": {"offers": ["$25", "$80"]}},
        )
        result = reranker.rerank(
            query, [candidate("HIGH"), candidate("IN")], catalog, 2
        )
        self.assertEqual(self.ids(result), ["IN", "HIGH"])

    def test_hard_constraint_violations_are_removed_by_default(self) -> None:
        reranker = FeatureReranker()
        query = SearchQuery(text="shirt", color=constraint("red", "hard"))
        catalog = self.view(
            {"parent_asin": "BLUE", "details": {"Color": "blue"}},
            {"parent_asin": "RED", "details": {"Color": "red"}},
        )
        result = reranker.rerank(
            query, [candidate("BLUE", 1, 100), candidate("RED", 2, 1)], catalog, 2
        )
        self.assertEqual(self.ids(result), ["RED"])
        removed = reranker.last_diagnostics["BLUE"]
        self.assertEqual(removed["hard_violations"], ("color",))
        self.assertEqual(removed["removal_reason"], "hard_constraint_violation")

    def test_hard_penalty_policy_keeps_but_demotes_violation(self) -> None:
        reranker = self.reranker(
            feature_weights={"lexical_rank": 1.0},
            hard_constraint_policy="penalize",
            hard_constraint_penalty=100.0,
        )
        query = SearchQuery(text="shirt", color=constraint("red", "hard"))
        catalog = self.view(
            {"parent_asin": "BLUE", "details": {"Color": "blue"}},
            {"parent_asin": "RED", "details": {"Color": "red"}},
        )
        result = reranker.rerank(
            query, [candidate("BLUE", 1), candidate("RED", 100)], catalog, 2
        )
        self.assertEqual(self.ids(result), ["RED", "BLUE"])

    def test_explicit_exclusions_are_enforced(self) -> None:
        reranker = FeatureReranker()
        query = SearchQuery(text="bag", exclusions={"material": {"leather"}})
        catalog = self.view(
            {"parent_asin": "LEATHER", "details": {"Material": "leather"}},
            {"parent_asin": "CANVAS", "details": {"Material": "canvas"}},
        )
        result = reranker.rerank(
            query, [candidate("LEATHER"), candidate("CANVAS")], catalog, 2
        )
        self.assertEqual(self.ids(result), ["CANVAS"])
        self.assertEqual(
            reranker.last_diagnostics["LEATHER"]["removal_reason"],
            "exclusion_violation",
        )

    def test_missing_catalog_metadata_is_unknown_not_a_violation(self) -> None:
        reranker = FeatureReranker()
        query = SearchQuery(
            text="red cotton shirt",
            color=constraint("red", "hard"),
            material=constraint("cotton", "hard"),
            price=PriceConstraint(maximum=40, strength="hard"),
        )
        result = reranker.rerank(
            query,
            [candidate("EMPTY", 1), candidate("PARTIAL", 2)],
            self.view(
                {"parent_asin": "EMPTY"},
                {"parent_asin": "PARTIAL", "details": {"Color": "red"}},
            ),
            2,
        )
        self.assertEqual(self.ids(result), ["PARTIAL", "EMPTY"])
        self.assertEqual(reranker.last_diagnostics["EMPTY"]["hard_violations"], ())

    def test_missing_and_malformed_retrieval_signals_are_safe(self) -> None:
        reranker = FeatureReranker()
        values = [
            Candidate("A", lexical_score=None, dense_score=None),  # type: ignore[arg-type]
            Candidate("B", lexical_score=float("nan"), lexical_rank=0),
            Candidate("C", dense_score=0.2, dense_rank=1, sources={"dense"}),
        ]
        result = reranker.rerank(
            SearchQuery(text="item"), values, self.view(*({"parent_asin": item.parent_asin} for item in values)), 3
        )
        self.assertEqual(set(self.ids(result)), {"A", "B", "C"})
        self.assertTrue(all(item.rerank_score is not None for item in result))

    def test_stable_tie_breaking_preserves_original_pool_order(self) -> None:
        reranker = self.reranker(feature_weights={})
        values = [candidate("Z"), candidate("A"), candidate("M")]
        result = reranker.rerank(
            SearchQuery(text="item"), values, self.view(*({"parent_asin": item.parent_asin} for item in values)), 3
        )
        self.assertEqual(self.ids(result), ["Z", "A", "M"])

    def test_duplicates_merge_signals_without_mutating_input(self) -> None:
        reranker = FeatureReranker()
        values = [
            Candidate("A", lexical_score=0.2, lexical_rank=3, sources={"lexical"}),
            Candidate("A", dense_score=0.9, dense_rank=1, sources={"dense"}),
            Candidate("B", lexical_score=0.1, lexical_rank=2, sources={"lexical"}),
        ]
        before = copy.deepcopy(values)
        result = reranker.rerank(
            SearchQuery(text="item"), values, self.view({"parent_asin": "A"}, {"parent_asin": "B"}), 10
        )
        self.assertEqual(len(result), 2)
        merged = next(item for item in result if item.parent_asin == "A")
        self.assertEqual((merged.lexical_rank, merged.dense_rank), (3, 1))
        self.assertEqual(merged.sources, {"lexical", "dense"})
        self.assertEqual(values, before)

    def test_top_k_zero_large_and_empty_input(self) -> None:
        reranker = FeatureReranker()
        catalog = self.view({"parent_asin": "A"}, {"parent_asin": "B"})
        query = SearchQuery(text="item")
        values = [candidate("A"), candidate("B")]
        self.assertEqual(reranker.rerank(query, values, catalog, 0), [])
        self.assertEqual(len(reranker.rerank(query, values, catalog, 20)), 2)
        self.assertEqual(reranker.rerank(query, [], catalog, 10), [])

    def test_repeated_runs_are_identical_and_output_is_from_supplied_pool(self) -> None:
        reranker = FeatureReranker()
        query = SearchQuery(text="red shirt", color=constraint("red"))
        values = [candidate("C", 3), candidate("A", 1), candidate("B", 2)]
        catalog = self.view(
            {"parent_asin": "A", "details": {"Color": "blue"}},
            {"parent_asin": "B", "details": {"Color": "red"}},
            {"parent_asin": "C"},
            {"parent_asin": "NOT_SUPPLIED", "details": {"Color": "red"}},
        )
        orders = [self.ids(reranker.rerank(query, values, catalog, 10)) for _ in range(5)]
        self.assertTrue(all(order == orders[0] for order in orders))
        self.assertLessEqual(set(orders[0]), {"A", "B", "C"})
        self.assertNotIn("NOT_SUPPLIED", orders[0])

    def test_original_retrieval_information_remains_unchanged(self) -> None:
        reranker = FeatureReranker()
        original = Candidate(
            "A",
            lexical_score=2.5,
            dense_score=0.8,
            lexical_rank=4,
            dense_rank=2,
            sources={"lexical", "dense"},
            fusion_score=0.12,
        )
        result = reranker.rerank(
            SearchQuery(text="item"), [original], self.view({"parent_asin": "A"}), 1
        )[0]
        self.assertEqual(
            (
                result.lexical_score,
                result.dense_score,
                result.lexical_rank,
                result.dense_rank,
                result.fusion_score,
            ),
            (2.5, 0.8, 4, 2, 0.12),
        )
        self.assertIsNone(original.rerank_score)
        self.assertIsNone(original.rerank_diagnostics)
        self.assertIsNotNone(result.rerank_diagnostics)

    def test_safe_fallback_preserves_unique_original_order(self) -> None:
        class UnavailableCatalog:
            def get(self, parent_asin: str):
                del parent_asin
                raise CatalogUnavailableError("offline")

        query = SearchQuery(text="item")
        values = [candidate("B"), candidate("A"), candidate("B"), candidate("C")]
        for catalog in (None, UnavailableCatalog()):
            with self.subTest(catalog=catalog):
                reranker = FeatureReranker()
                result = reranker.rerank(query, values, catalog, 2)  # type: ignore[arg-type]
                self.assertEqual(self.ids(result), ["B", "A"])
                self.assertEqual(
                    result[0].rerank_diagnostics["fallback_reason"],  # type: ignore[index]
                    "catalog_unavailable",
                )

    def test_configuration_validation_rejects_unknown_or_unsafe_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown reranking"):
            RerankerConfig(feature_weights={"magic": 1.0})
        with self.assertRaisesRegex(ValueError, "non-negative"):
            RerankerConfig(hard_constraint_penalty=-1)
        with self.assertRaisesRegex(ValueError, "hard_constraint_policy"):
            RerankerConfig(hard_constraint_policy="sometimes")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
