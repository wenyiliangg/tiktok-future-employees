from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from starter.clarification_controller import ClarificationController
from starter.clarification_policies import (
    clarification_policy_by_id,
    load_clarification_policy_by_id,
)
from starter.conversation_state import SessionState

P2_POLICY_ID = "clarification.category-evidence-utility.v1"
P4_POLICY_ID = "clarification.category-evidence-utility-buying.v1"


class P4BuyingRouteClarificationTest(unittest.TestCase):
    def test_p4_differs_from_p2_only_by_policy_id_and_buying_route(self) -> None:
        p2 = clarification_policy_by_id(P2_POLICY_ID)
        p4 = clarification_policy_by_id(P4_POLICY_ID)
        p2_payload = p2.fingerprint_payload()
        p4_payload = p4.fingerprint_payload()

        p2_payload["policy_id"] = P4_POLICY_ID
        p2_payload["clarification"]["eligible_routes"] = (  # type: ignore[index]
            "browsing",
            "boundary",
            "buying",
        )
        self.assertEqual(p2_payload, p4_payload)
        self.assertEqual(
            p4.fingerprint_sha256,
            "6db00179643c355adf1ecfbef5fee680ce50ce316f6f5c272da9aa53ab8bf62e",
        )

    def test_buying_is_the_only_new_utility_eligible_route(self) -> None:
        p2 = clarification_policy_by_id(P2_POLICY_ID).clarification
        p4 = clarification_policy_by_id(P4_POLICY_ID).clarification

        for route in ("browsing", "boundary", "uncertain"):
            for candidate_count in (3, 4, 50):
                self.assertEqual(
                    p2.utility_is_eligible(route, candidate_count),
                    p4.utility_is_eligible(route, candidate_count),
                )
        self.assertFalse(p4.utility_is_eligible("buying", 3))
        self.assertTrue(p4.utility_is_eligible("buying", 4))
        self.assertFalse(p2.utility_is_eligible("buying", 50))

    def test_controller_state_is_isolated_and_deterministic_under_p4(self) -> None:
        policy = clarification_policy_by_id(P4_POLICY_ID)

        def transcript() -> tuple[object, object, object]:
            controller = ClarificationController(policy.controller)
            controller.reset("first")
            controller.reset("second")
            first = controller.build_prompt("first", "other", SessionState(), 1)
            second_before = copy.deepcopy(controller.state_for("second"))
            controller.record_resolution("first", "other", "answered")
            follow_up = controller.build_prompt("first", "feature", SessionState(), 2)
            second_after = copy.deepcopy(controller.state_for("second"))
            self.assertEqual(second_before, second_after)
            return first, follow_up, second_after

        self.assertEqual(transcript(), transcript())

    def test_corrupt_p4_fails_without_disabling_p2_rollback(self) -> None:
        source = Path("config/clarification_policies.json")
        payload = json.loads(source.read_text(encoding="utf-8"))
        p4 = next(
            item for item in payload["policies"] if item["policy_id"] == P4_POLICY_ID
        )
        p4["fingerprint_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policies.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                load_clarification_policy_by_id(P4_POLICY_ID, path)
            rollback = load_clarification_policy_by_id(P2_POLICY_ID, path)
            self.assertEqual(rollback.policy_id, P2_POLICY_ID)


if __name__ == "__main__":
    unittest.main()
