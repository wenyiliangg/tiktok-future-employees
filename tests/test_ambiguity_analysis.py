from __future__ import annotations

import unittest

from starter.ambiguity_analysis import (
    AmbiguityAnalyzer,
    AmbiguityConfig,
    ClarificationOpportunity,
)
from starter.conversation_state import Constraint, SessionState
from starter.hybrid_retrieval import Candidate


def candidates(count: int) -> list[Candidate]:
    return [Candidate(parent_asin=f"P{index}") for index in range(count)]


def slot(value: str) -> Constraint:
    return Constraint(value, "hard", "current_turn", 1)


class AmbiguityAnalyzerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = AmbiguityAnalyzer()

    def test_highly_ambiguous_pool_selects_one_deterministic_attribute(self) -> None:
        pool = candidates(8)
        catalog = {
            candidate.parent_asin: {
                "categories": ["Clothing", "Shoes"],
                "color": "red" if index < 4 else "blue",
                "material": "canvas" if index % 2 else "leather",
            }
            for index, candidate in enumerate(pool)
        }

        opportunity = self.analyzer.analyze(pool, catalog, SessionState())

        self.assertTrue(opportunity.should_ask)
        self.assertEqual(opportunity.attribute, "color")
        self.assertEqual(opportunity.options, ("blue", "red"))
        self.assertEqual(opportunity.expected_reduction, 0.5)
        self.assertIn("expected_reduction=0.500", opportunity.reason)

    def test_already_narrow_pool_does_not_ask(self) -> None:
        pool = candidates(6)
        catalog = {
            candidate.parent_asin: {
                "categories": ["Shoes"],
                "color": "black",
                "material": "canvas",
            }
            for candidate in pool
        }

        opportunity = self.analyzer.analyze(pool, catalog, SessionState())

        self.assertFalse(opportunity.should_ask)
        self.assertIsNone(opportunity.attribute)

    def test_known_attribute_is_never_selected(self) -> None:
        pool = candidates(8)
        catalog = {
            candidate.parent_asin: {
                "categories": ["Shoes"],
                "color": "red" if index < 4 else "blue",
                "material": "canvas" if index % 2 else "leather",
            }
            for index, candidate in enumerate(pool)
        }

        opportunity = self.analyzer.analyze(
            pool,
            catalog,
            SessionState(color=slot("red")),
        )

        self.assertTrue(opportunity.should_ask)
        self.assertEqual(opportunity.attribute, "material")

    def test_mostly_missing_metadata_is_not_usable(self) -> None:
        pool = candidates(6)
        catalog = {candidate.parent_asin: {} for candidate in pool}
        catalog["P0"] = {"color": "red"}
        catalog["P1"] = {"color": "blue"}

        opportunity = self.analyzer.analyze(pool, catalog, SessionState())

        self.assertFalse(opportunity.should_ask)
        color = next(
            item
            for item in self.analyzer.attribute_statistics(pool, catalog)
            if item.attribute == "color"
        )
        self.assertEqual(color.coverage, 0.333333)

    def test_one_dominant_value_barely_reduces_pool(self) -> None:
        pool = candidates(10)
        catalog = {
            candidate.parent_asin: {"color": "black" if index < 9 else "white"}
            for index, candidate in enumerate(pool)
        }

        opportunity = self.analyzer.analyze(pool, catalog, SessionState())
        color = next(
            item
            for item in self.analyzer.attribute_statistics(pool, catalog)
            if item.attribute == "color"
        )

        self.assertFalse(opportunity.should_ask)
        self.assertEqual(color.dominant_share, 0.9)
        self.assertEqual(color.expected_reduction, 0.18)

    def test_evenly_divided_values_have_high_expected_reduction(self) -> None:
        pool = candidates(10)
        catalog = {
            candidate.parent_asin: {"color": "black" if index < 5 else "white"}
            for index, candidate in enumerate(pool)
        }

        opportunity = self.analyzer.analyze(pool, catalog, SessionState())

        self.assertTrue(opportunity.should_ask)
        self.assertEqual(opportunity.attribute, "color")
        self.assertEqual(opportunity.expected_reduction, 0.5)

    def test_empty_candidate_pool_is_safe(self) -> None:
        opportunity = self.analyzer.analyze([], {}, SessionState())

        self.assertEqual(
            opportunity,
            ClarificationOpportunity(False, None, (), 0.0, "candidate_pool_is_empty"),
        )

    def test_very_small_pool_does_not_ask(self) -> None:
        pool = candidates(2)
        catalog = {"P0": {"color": "red"}, "P1": {"color": "blue"}}

        opportunity = self.analyzer.analyze(pool, catalog, SessionState())

        self.assertFalse(opportunity.should_ask)
        self.assertIn("candidate_pool_too_small", opportunity.reason)

    def test_repeated_analysis_is_deterministic(self) -> None:
        pool = candidates(6)
        catalog = {
            candidate.parent_asin: {"color": "red" if index < 3 else "blue"}
            for index, candidate in enumerate(pool)
        }

        opportunities = [self.analyzer.analyze(pool, catalog, SessionState()) for _ in range(5)]

        self.assertTrue(all(item == opportunities[0] for item in opportunities))

    def test_price_ranges_are_analyzed(self) -> None:
        pool = candidates(8)
        prices = (20, 22, 30, 35, 60, 70, 120, 150)
        catalog = {
            candidate.parent_asin: {"color": "black", "price": prices[index]}
            for index, candidate in enumerate(pool)
        }

        opportunity = self.analyzer.analyze(pool, catalog, SessionState())

        self.assertTrue(opportunity.should_ask)
        self.assertEqual(opportunity.attribute, "price")
        self.assertEqual(opportunity.options, ("0-25", "100-200", "25-50", "50-100"))
        self.assertEqual(opportunity.expected_reduction, 0.75)

    def test_relevant_binary_feature_can_be_selected(self) -> None:
        pool = candidates(6)
        catalog = {
            candidate.parent_asin: {
                "color": "black",
                "features": ["waterproof"] if index < 3 else [],
            }
            for index, candidate in enumerate(pool)
        }

        opportunity = self.analyzer.analyze(pool, catalog, SessionState())

        self.assertTrue(opportunity.should_ask)
        self.assertEqual(opportunity.attribute, "feature")
        self.assertEqual(opportunity.options, ("not:waterproof", "waterproof"))

    def test_duplicate_candidates_do_not_inflate_statistics(self) -> None:
        pool = candidates(4)
        duplicated = [*pool, pool[0], pool[1]]
        catalog = {
            candidate.parent_asin: {"color": "red" if index < 2 else "blue"}
            for index, candidate in enumerate(pool)
        }

        statistic = next(
            item
            for item in self.analyzer.attribute_statistics(duplicated, catalog)
            if item.attribute == "color"
        )

        self.assertEqual(statistic.candidate_count, 4)
        self.assertEqual(statistic.value_counts, (("blue", 2), ("red", 2)))

    def test_thresholds_are_configurable(self) -> None:
        pool = candidates(10)
        catalog = {
            candidate.parent_asin: {"color": "black" if index < 9 else "white"}
            for index, candidate in enumerate(pool)
        }
        permissive = AmbiguityAnalyzer(
            AmbiguityConfig(
                min_expected_reduction=0.1,
                max_dominant_share=0.95,
            )
        )

        opportunity = permissive.analyze(pool, catalog, SessionState())

        self.assertTrue(opportunity.should_ask)
        self.assertEqual(opportunity.attribute, "color")

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_metadata_coverage"):
            AmbiguityConfig(min_metadata_coverage=1.1)
        with self.assertRaisesRegex(ValueError, "attribute_priority"):
            AmbiguityConfig(
                attribute_priority=(
                    "category",
                    "category",
                    *AmbiguityConfig().attribute_priority[1:],
                )
            )


if __name__ == "__main__":
    unittest.main()
