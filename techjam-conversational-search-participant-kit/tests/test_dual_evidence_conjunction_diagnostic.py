from __future__ import annotations

import unittest

from benchmarks.dual_evidence_conjunction_diagnostic import (
    promote_unique_conjunction,
    tokens,
)


class DualEvidenceConjunctionDiagnosticTest(unittest.TestCase):
    def test_shared_boilerplate_tokens_are_normalized(self) -> None:
        self.assertEqual(
            tokens("Actually, I now need blue leather"), {"blue", "leather"}
        )

    def test_unique_strong_conjunction_is_promoted(self) -> None:
        ranked, selected = promote_unique_conjunction(
            ["A", "B", "C"],
            {"B": (5.0, 4.0), "C": (3.0, 6.0)},
            minimum_side_support=3.5,
            minimum_margin=0.5,
        )

        self.assertEqual(ranked, ["B", "A", "C"])
        self.assertEqual(selected, "B")

    def test_ambiguous_conjunction_preserves_order(self) -> None:
        ranked, selected = promote_unique_conjunction(
            ["A", "B", "C"],
            {"B": (5.0, 4.0), "C": (4.2, 4.0)},
            minimum_side_support=3.5,
            minimum_margin=0.5,
        )

        self.assertEqual(ranked, ["A", "B", "C"])
        self.assertIsNone(selected)


if __name__ == "__main__":
    unittest.main()
