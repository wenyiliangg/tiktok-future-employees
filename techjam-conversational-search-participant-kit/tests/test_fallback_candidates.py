from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest

from starter.fallback_candidates import (
    FallbackCandidateGenerator,
    FallbackConfig,
    ProfileEvidenceAdapter,
    adapt_fallback_candidates,
)
from starter.search_models import Constraint, SearchQuery


def explicit_constraint(value: str) -> Constraint:
    return Constraint(
        value=value,
        strength="hard",
        source="current_turn",
        updated_turn=1,
    )


class ProfileEvidenceAdapterTest(unittest.TestCase):
    def test_profile_slots_are_soft_and_unknown_fields_are_not_invented(self) -> None:
        query = ProfileEvidenceAdapter().adapt(
            {
                "preference_tags": ("black", "leather", "comfort"),
                "summary": "Usually prefers casual products.",
                "purchase_frequency": "frequent",
                "average_prior_rating": 4.8,
            }
        )

        self.assertEqual(query.color.value, "black")
        self.assertEqual(query.material.value, "leather")
        self.assertEqual(query.style.value, "casual")
        for value in (query.color, query.material, query.style):
            self.assertEqual(value.strength, "soft")
            self.assertEqual(value.source, "profile")
        self.assertIsNone(query.category)
        self.assertIsNone(query.price)

    def test_sparse_or_malformed_profile_is_safe(self) -> None:
        adapter = ProfileEvidenceAdapter()

        for profile in (None, [], "black", {}, {"preference_tags": "black"}):
            with self.subTest(profile=profile):
                query = adapter.adapt(profile)
                self.assertEqual(query.text, "")
                self.assertIsNone(query.color)


class FallbackCandidateGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.products = [
            {
                "parent_asin": "BLACK",
                "title": "Acme Classic Black Tote",
                "categories": ["Accessories", "Bags"],
                "store": "Acme",
                "color": "Black",
                "style": "Classic",
                "material": "Leather",
                "price": 80,
                "average_rating": 4.5,
                "rating_number": 100,
            },
            {
                "parent_asin": "WHITE",
                "title": "Beacon Modern White Sneaker",
                "categories": ["Shoes", "Sneakers"],
                "store": "Beacon",
                "color": "White",
                "style": "Modern",
                "material": "Canvas",
                "price": 40,
                "average_rating": 4.5,
                "rating_number": 100,
            },
            {
                "parent_asin": "BLUE",
                "title": "Cedar Blue Winter Coat",
                "categories": ["Clothing", "Coats"],
                "store": "Cedar",
                "color": "Blue",
                "style": "Casual",
                "material": "Wool",
                "price": 140,
                "average_rating": 4.3,
                "rating_number": 50,
            },
        ]
        self.no_diversity = FallbackConfig(diversity_dimensions=())

    def test_empty_conversation_and_profile_returns_valid_diverse_candidates(self) -> None:
        generator = FallbackCandidateGenerator(self.products)

        results = generator.generate(query=SearchQuery(text=""), user_profile={}, top_n=3)

        self.assertEqual(len(results), 3)
        self.assertEqual([result.rank for result in results], [1, 2, 3])
        self.assertEqual({result.source for result in results}, {"fallback"})
        self.assertEqual(
            {result.parent_asin for result in results},
            generator.catalog_ids,
        )
        self.assertEqual(len({result.parent_asin for result in results}), 3)

    def test_empty_conversation_uses_profile_preferences_as_soft_boosts(self) -> None:
        generator = FallbackCandidateGenerator(self.products, config=self.no_diversity)

        results = generator.generate(
            query=SearchQuery(text=""),
            user_profile={"preference_tags": ["black", "leather"]},
            top_n=3,
        )

        self.assertEqual(results[0].parent_asin, "BLACK")
        self.assertGreater(results[0].fallback_score, results[1].fallback_score)
        self.assertEqual(len(results), 3, "soft profile evidence must not filter the pool")

    def test_explicit_conversation_preference_overrides_profile_history(self) -> None:
        generator = FallbackCandidateGenerator(self.products, config=self.no_diversity)
        query = SearchQuery(text="white", color=explicit_constraint("white"))

        results = generator.generate(
            query=query,
            user_profile={"preference_tags": ["black"]},
            top_n=3,
        )

        self.assertEqual(results[0].parent_asin, "WHITE")
        self.assertGreater(results[0].fallback_score, results[1].fallback_score)

    def test_explicit_exclusion_suppresses_conflicting_profile_evidence(self) -> None:
        generator = FallbackCandidateGenerator(self.products, config=self.no_diversity)
        query = SearchQuery(text="", exclusions={"color": {"black"}})

        results = generator.generate(
            query=query,
            user_profile={"preference_tags": ["black"]},
            top_n=3,
        )

        self.assertNotEqual(results[0].parent_asin, "BLACK")

    def test_explicitly_removed_profile_slot_is_not_reintroduced(self) -> None:
        products = [
            {"parent_asin": "A", "title": "Black Item", "color": "black"},
            {"parent_asin": "B", "title": "White Item", "color": "white"},
        ]
        generator = FallbackCandidateGenerator(products, config=self.no_diversity)

        results = generator.generate(
            query=SearchQuery(text=""),
            user_profile={"preference_tags": ["black"]},
            removed_constraints={"color:black"},
            top_n=2,
        )

        self.assertEqual([result.parent_asin for result in results], ["A", "B"])
        self.assertEqual(results[0].fallback_score, results[1].fallback_score)

    def test_malformed_catalog_metadata_and_optional_signals_are_safe(self) -> None:
        products = [
            {
                "parent_asin": "VALID",
                "title": None,
                "categories": {"nested": [None, 3]},
                "details": "unstructured",
                "average_rating": "not-a-number",
                "rating_number": -10,
                "price": "unavailable",
            },
            {"title": "missing id"},
            ["not", "a", "catalog row"],
        ]
        generator = FallbackCandidateGenerator(products)

        results = generator.generate(user_profile={"summary": 123}, top_n=5)

        self.assertEqual([result.parent_asin for result in results], ["VALID"])
        self.assertEqual(results[0].fallback_score, 0.0)

    def test_duplicates_are_removed_and_output_is_catalog_valid(self) -> None:
        products = [
            {"parent_asin": "B", "title": "Item"},
            {"parent_asin": "A", "title": "Item"},
            {"parent_asin": "A", "title": "Duplicate variant"},
            {"parent_asin": "", "title": "Invalid"},
            {"title": "Missing"},
        ]
        generator = FallbackCandidateGenerator(products, config=self.no_diversity)

        results = generator.generate(top_n=10)

        self.assertEqual([result.parent_asin for result in results], ["A", "B"])
        self.assertTrue(all(item.parent_asin in generator.catalog_ids for item in results))

    def test_ordering_is_deterministic_across_calls_and_catalog_order(self) -> None:
        first_generator = FallbackCandidateGenerator(self.products)
        second_generator = FallbackCandidateGenerator(reversed(self.products))

        first = first_generator.generate(user_profile={}, top_n=3)
        repeated = first_generator.generate(user_profile={}, top_n=3)
        reordered = second_generator.generate(user_profile={}, top_n=3)

        self.assertEqual(first, repeated)
        self.assertEqual(first, reordered)

    def test_diversity_caps_are_configurable(self) -> None:
        products = [
            {
                "parent_asin": "SHOE-1",
                "title": "Alpha Runner",
                "categories": ["Shoes"],
                "average_rating": 5,
                "rating_number": 100,
            },
            {
                "parent_asin": "SHOE-2",
                "title": "Beta Runner",
                "categories": ["Shoes"],
                "average_rating": 4.9,
                "rating_number": 90,
            },
            {
                "parent_asin": "HAT",
                "title": "Gamma Hat",
                "categories": ["Hats"],
                "average_rating": 4,
                "rating_number": 20,
            },
            {
                "parent_asin": "BAG",
                "title": "Delta Bag",
                "categories": ["Bags"],
                "average_rating": 4,
                "rating_number": 20,
            },
        ]
        config = FallbackConfig(
            diversity_dimensions=("category",),
            diversity_caps={"category": 1},
            diversity_penalties={"category": 0.5},
        )
        generator = FallbackCandidateGenerator(products, config=config)

        results = generator.generate(top_n=3)

        self.assertEqual(results[0].parent_asin, "SHOE-1")
        self.assertEqual({item.parent_asin for item in results[1:]}, {"HAT", "BAG"})

    def test_requested_count_is_enforced_and_larger_than_pool_is_safe(self) -> None:
        generator = FallbackCandidateGenerator(self.products)

        self.assertEqual(len(generator.generate(top_n=2)), 2)
        self.assertEqual(len(generator.generate(top_n=100)), 3)
        for invalid in (0, -1, True, 1.5, "3"):
            with self.subTest(top_n=invalid):
                self.assertEqual(generator.generate(top_n=invalid), [])  # type: ignore[arg-type]
        self.assertEqual(FallbackCandidateGenerator([]).generate(top_n=10), [])

    def test_jsonl_loader_and_shared_candidate_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text(
                json.dumps({"parent_asin": "A", "title": "Item"}) + "\n\n",
                encoding="utf-8",
            )
            generator = FallbackCandidateGenerator.from_jsonl(
                path, config=self.no_diversity
            )
            results = generator.generate(top_n=1)

        payloads = adapt_fallback_candidates(results)
        self.assertEqual(
            payloads,
            [
                {
                    "parent_asin": "A",
                    "fallback_score": 0.0,
                    "source": "fallback",
                    "rank": 1,
                }
            ],
        )

        @dataclass(frozen=True)
        class SharedCandidate:
            parent_asin: str
            fallback_score: float
            source: str
            rank: int

        adapted = adapt_fallback_candidates(results, SharedCandidate)
        self.assertEqual(adapted[0].parent_asin, "A")


if __name__ == "__main__":
    unittest.main()
