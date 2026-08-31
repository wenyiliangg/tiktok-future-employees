"""Tests for campaign-level non-public shadow infrastructure."""

from __future__ import annotations

import unittest

from benchmarks.shadow_clarification_suite import (
    SCENARIOS,
    build_shadow_samples,
    select_shadow_products,
    shadow_constraints,
)


class ShadowClarificationSuiteTest(unittest.TestCase):
    def products(self) -> dict[str, dict[str, object]]:
        return {
            f"P{index:03d}": {
                "parent_asin": f"P{index:03d}",
                "title": f"canvas running shoe {index}",
                "features": [
                    "canvas upper",
                    "blue color",
                    "wide sizing",
                    "outdoor running",
                ],
                "categories": ["Clothing, Shoes & Jewelry", "Shoes", "Athletic"],
                "price": 40 + index,
            }
            for index in range(20)
        }

    def test_selection_is_nonpublic_deterministic_and_reorder_invariant(self) -> None:
        products = self.products()
        excluded = frozenset({"P003", "P011"})

        forward = select_shadow_products(products, excluded, 12)
        reverse = select_shadow_products(
            dict(reversed(list(products.items()))), excluded, 12
        )

        self.assertEqual([row[0] for row in forward], [row[0] for row in reverse])
        self.assertFalse({row[0] for row in forward} & excluded)

    def test_samples_cover_scenarios_case_and_partial_disclosure(self) -> None:
        samples = build_shadow_samples(self.products(), frozenset(), 16)

        self.assertEqual({sample.scenario_type for sample in samples}, set(SCENARIOS))
        self.assertEqual(
            {sample.case_variant for sample in samples},
            {"natural", "upper", "lower"},
        )
        self.assertTrue(any(sample.partial_disclosure for sample in samples))
        self.assertTrue(any(not sample.partial_disclosure for sample in samples))

    def test_constraint_derivation_tolerates_malformed_metadata(self) -> None:
        self.assertEqual(
            shadow_constraints({"features": None, "details": "bad", "price": "NaN"}),
            (),
        )
        self.assertEqual(
            shadow_constraints({"title": "valid item", "price": float("inf")}),
            ("valid item",),
        )


if __name__ == "__main__":
    unittest.main()
