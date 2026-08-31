from __future__ import annotations

import unittest

from benchmarks.p8_proxy_campaign import (
    SAMPLE_COUNT,
    EligibleTarget,
    _samples,
    _selection_summary,
    _stratified_select,
    _uniform_select,
)


class P8ProxyCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            EligibleTarget(f"ID{index:04d}", f"category-{index % 7}", index % 4 + 1)
            for index in range(900)
        ]

    def test_stratified_selection_is_exact_and_order_invariant(self) -> None:
        first = _stratified_select(self.rows, seed=2026083101, phase="development")
        second = _stratified_select(
            list(reversed(self.rows)), seed=2026083101, phase="development"
        )
        self.assertEqual(SAMPLE_COUNT, len(first))
        self.assertEqual(
            {row.parent_asin for row in first},
            {row.parent_asin for row in second},
        )

    def test_uniform_selection_is_exact_and_unique(self) -> None:
        selected = _uniform_select(self.rows, seed=2026083103, phase="uniform_stress")
        self.assertEqual(SAMPLE_COUNT, len(selected))
        self.assertEqual(SAMPLE_COUNT, len({row.parent_asin for row in selected}))

    def test_scenario_assignment_is_exactly_balanced(self) -> None:
        selected = _stratified_select(self.rows, seed=2026083101, phase="development")
        samples = _samples(selected, phase="development")
        counts = {
            scenario: sum(row["scenario_type"] == scenario for row in samples)
            for scenario in ("boundary", "browsing", "buying", "intent_override")
        }
        self.assertEqual({name: 100 for name in counts}, counts)

    def test_selection_summary_never_persists_target_ids(self) -> None:
        selected = _stratified_select(self.rows, seed=2026083101, phase="development")
        summary = _selection_summary(
            selected, {"PUBLIC"}, {"development": selected}, "development"
        )
        self.assertFalse(summary["target_ids_persisted"])
        self.assertNotIn("targets", summary)
        self.assertEqual(0, summary["intersections"]["public"])


if __name__ == "__main__":
    unittest.main()
