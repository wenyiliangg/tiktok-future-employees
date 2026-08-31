from __future__ import annotations

import unittest

from starter.override_history_policy import (
    OverrideHistoryPolicy,
    override_history_policy_for_retrieval,
)


class OverrideHistoryPolicyTest(unittest.TestCase):
    def test_selected_policy_is_bounded_and_feedback_compatible(self) -> None:
        policy = override_history_policy_for_retrieval(
            "contextual.override-history-tail.v1"
        )

        self.assertTrue(policy.enabled)
        self.assertEqual(policy.phrase_limit, 1)
        self.assertEqual(policy.required_tail_weight, 0.5)
        self.assertTrue(
            policy.clarification_is_compatible("contextual.feedback-memory.v1")
        )
        self.assertEqual(len(policy.fingerprint_sha256), 64)

    def test_enabled_policy_requires_positive_finite_weight(self) -> None:
        with self.assertRaises(ValueError):
            OverrideHistoryPolicy(
                policy_id="bad",
                retrieval_policy_id="contextual.bad",
                compatible_retrieval_policy_id="contextual.feedback-memory.v1",
                enabled=True,
                required_tail_weight=0.0,
            )

    def test_other_retrieval_policies_disable_history(self) -> None:
        policy = override_history_policy_for_retrieval("contextual.feedback-memory.v1")

        self.assertFalse(policy.enabled)
        self.assertEqual(policy.required_tail_weight, 0.0)


if __name__ == "__main__":
    unittest.main()
