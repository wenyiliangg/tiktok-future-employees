from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluator.local_evaluator import retrieval_configuration_fingerprint
from starter.ambiguity_analysis import ClarificationOpportunity
from starter.clarification_policies import (
    clarification_policy_by_id,
    clarification_policy_candidates,
    load_clarification_policy_registry,
)
from starter.contextual_retrieval import policy_by_id
from starter.hybrid_retrieval import HybridRetrievalConfig
from starter.selective_clarification import SelectiveClarificationConfig


class ClarificationPolicyRegistryTest(unittest.TestCase):
    def test_predeclared_names_values_and_fingerprints_are_stable(self) -> None:
        policies = clarification_policy_candidates()

        self.assertEqual(
            [policy.policy_id for policy in policies],
            [
                "contextual.browsing-dense.v1",
                "clarification.issue-5c.v1",
                "clarification.browsing-only.v1",
                "clarification.feedback-memory.v1",
                "clarification.category-evidence-utility.v1",
                "clarification.category-evidence-utility-buying.v1",
            ],
        )
        self.assertEqual(
            {policy.policy_id: policy.fingerprint_sha256 for policy in policies},
            {
                "contextual.browsing-dense.v1": "04a16f9cec5162ab8a3d6ecff098c0342d205a37d24b5665a36316ba4f64f8a6",
                "clarification.issue-5c.v1": "307822c299d9c3614f06215ecb5118107a5264f1d2efb7b704cf3787018ce1ed",
                "clarification.browsing-only.v1": "405c3ff441211cc6073b3732e1bd60b7aa8e85698c8ceb7d7931fed8eeaeb6fd",
                "clarification.feedback-memory.v1": "56550e0f09f152db8be1a3988bacda499955e8b5683e95eedad44b3ce19fb7a5",
                "clarification.category-evidence-utility.v1": "ce4634f3f2e14414238812ba0eda841e2ab2e0fd71bc00f392192539571924d1",
                "clarification.category-evidence-utility-buying.v1": "6db00179643c355adf1ecfbef5fee680ce50ce316f6f5c272da9aa53ab8bf62e",
            },
        )
        self.assertEqual({policy.evaluation_seed for policy in policies}, {20260830})

        utility = clarification_policy_by_id(
            "clarification.category-evidence-utility.v1"
        )
        self.assertEqual(
            utility.clarification.question_candidates, ("other", "feature")
        )
        self.assertEqual(utility.controller.max_questions_per_session, 2)

    def test_control_and_issue_5c_exact_values_are_immutable(self) -> None:
        control = clarification_policy_by_id("contextual.browsing-dense.v1")
        issue_5c = clarification_policy_by_id("clarification.issue-5c.v1")

        self.assertFalse(control.clarification.enabled)
        self.assertTrue(issue_5c.clarification.enabled)
        self.assertEqual(issue_5c.clarification.analysis_candidate_limit, 50)
        self.assertEqual(issue_5c.clarification.browsing_min_candidates, 4)
        self.assertEqual(issue_5c.clarification.browsing_min_expected_reduction, 0.2)
        self.assertEqual(issue_5c.clarification.buying_min_candidates, 8)
        self.assertEqual(issue_5c.clarification.buying_min_expected_reduction, 0.5)
        self.assertEqual(issue_5c.clarification.eligible_routes, ("browsing", "buying"))
        self.assertEqual(issue_5c.controller.max_questions_per_session, 1)
        self.assertEqual(issue_5c.controller.max_turns, 10)

    def test_control_preserves_legacy_disabled_and_retrieval_fingerprints(self) -> None:
        control = clarification_policy_by_id("contextual.browsing-dense.v1")
        retrieval_fingerprint, _payload = retrieval_configuration_fingerprint(
            HybridRetrievalConfig(), policy_by_id(control.retrieval_policy_id)
        )

        self.assertEqual(
            control.clarification, SelectiveClarificationConfig(enabled=False)
        )
        self.assertEqual(
            retrieval_fingerprint,
            "972158c9e3905e4d0bb5390eb7224fe3fe00b9f5444337099b3422960bf0448a",
        )

    def test_browsing_only_changes_only_route_eligibility(self) -> None:
        issue_5c = clarification_policy_by_id("clarification.issue-5c.v1")
        browsing = clarification_policy_by_id("clarification.browsing-only.v1")
        issue_payload = issue_5c.fingerprint_payload()
        browsing_payload = browsing.fingerprint_payload()

        issue_payload["policy_id"] = browsing_payload["policy_id"]
        issue_payload["clarification"]["eligible_routes"] = ("browsing",)  # type: ignore[index]
        self.assertEqual(issue_payload, browsing_payload)

        opportunity = ClarificationOpportunity(True, "category", (), 0.8, "test")
        self.assertTrue(browsing.clarification.is_eligible("browsing", 50, opportunity))
        self.assertFalse(browsing.clarification.is_eligible("buying", 50, opportunity))

    def test_runtime_default_is_selected_browsing_only_policy(self) -> None:
        registry = load_clarification_policy_registry()

        self.assertEqual(
            registry.runtime_default.policy_id, "clarification.browsing-only.v1"
        )
        self.assertEqual(
            registry.selected_for_issue_6b, "clarification.browsing-only.v1"
        )

    def test_unknown_policy_and_invalid_default_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown clarification policy"):
            clarification_policy_by_id("missing")

        source = Path("config/clarification_policies.json")
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["runtime_default_policy"] = "missing"
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "policies.json"
            invalid.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "runtime_default_policy"):
                load_clarification_policy_registry(invalid)


if __name__ == "__main__":
    unittest.main()
