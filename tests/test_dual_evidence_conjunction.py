from __future__ import annotations

import unittest
from dataclasses import replace

from starter.dual_evidence_conjunction import (
    DualEvidenceConjunctionRanker,
    DualEvidencePolicy,
    dual_evidence_policy_for_retrieval,
    promote_unique_conjunction,
)


class DualEvidenceConjunctionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = dual_evidence_policy_for_retrieval(
            "contextual.override-history-conjunction.v1"
        )

    def test_selected_policy_matches_diagnostic_thresholds(self) -> None:
        self.assertTrue(self.policy.enabled)
        self.assertEqual(self.policy.max_document_frequency, 1000)
        self.assertEqual(self.policy.minimum_side_support, 2.5)
        self.assertEqual(self.policy.minimum_margin, 0.25)
        self.assertEqual(len(self.policy.fingerprint_sha256), 64)

    def test_promotes_only_unique_two_sided_support(self) -> None:
        ranked = promote_unique_conjunction(
            ["A", "B", "C"],
            ["A", "B", "C"],
            {"B": (5.0, 4.0), "C": (2.0, 8.0)},
            self.policy,
        )
        self.assertEqual(ranked, ["B", "A", "C"])

        ambiguous = promote_unique_conjunction(
            ["A", "B", "C"],
            ["A", "B", "C"],
            {"B": (5.0, 4.0), "C": (4.9, 4.0)},
            self.policy,
        )
        self.assertEqual(ambiguous, ["A", "B", "C"])

    def test_disabled_or_missing_catalog_preserves_order(self) -> None:
        disabled = replace(self.policy, enabled=False)
        ranker = DualEvidenceConjunctionRanker({}, 1, disabled)
        self.assertEqual(
            ranker.rerank(["A", "B"], ["A", "B"], "blue", "leather", {}),
            ["A", "B"],
        )

    def test_policy_validation_rejects_invalid_thresholds(self) -> None:
        with self.assertRaises(ValueError):
            DualEvidencePolicy(
                policy_id="bad",
                retrieval_policy_id="bad",
                minimum_margin=float("nan"),
            )


if __name__ == "__main__":
    unittest.main()
