from __future__ import annotations

import unittest

from benchmarks.override_history_diagnostic import comparison_label, rank_of


class OverrideHistoryDiagnosticTest(unittest.TestCase):
    def test_rank_and_comparison_labels(self) -> None:
        self.assertEqual(rank_of("B", ["A", "B"]), 2)
        self.assertIsNone(rank_of("C", ["A", "B"]))
        self.assertEqual(comparison_label(None, 10), "gained_hit")
        self.assertEqual(comparison_label(10, None), "lost_hit")
        self.assertEqual(comparison_label(9, 8), "better_rank")
        self.assertEqual(comparison_label(8, 9), "worse_rank")
        self.assertEqual(comparison_label(8, 8), "unchanged_hit")
        self.assertEqual(comparison_label(None, None), "unchanged_miss")


if __name__ == "__main__":
    unittest.main()
