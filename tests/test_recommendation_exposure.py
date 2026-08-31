from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from starter.agent import Agent
from starter.conversation_state import Constraint, SessionState
from starter.hybrid_retrieval import HybridRetrievalConfig
from starter.recommendation_exposure import (
    apply_recommendation_exposure,
    disabled_exposure_policy,
    exposure_policy_by_id,
    load_exposure_policy_registry,
)
from starter.selective_clarification import SelectiveClarificationConfig


class _Retriever:
    def retrieve(self, *_args: object, **kwargs: object) -> list[object]:
        values = ["A", "B", "C"]
        return [
            SimpleNamespace(parent_asin=value, score=1.0 / rank, rank=rank)
            for rank, value in enumerate(values, start=1)
        ]


class _Router:
    def route(self, *_args: object) -> object:
        return SimpleNamespace(route="browsing")


class _ExplodingController:
    def state_for(self, _session_id: str) -> object:
        raise RuntimeError("optional controller unavailable")


def _response() -> dict[str, object]:
    return {
        "message": "results",
        "ask_attribute": None,
        "recommendations": [
            {"parent_asin": "A"},
            {"parent_asin": "B"},
            {"parent_asin": "C"},
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }


class RecommendationExposurePolicyTest(unittest.TestCase):
    def test_registry_fingerprints_and_declared_variants_load(self) -> None:
        policies = load_exposure_policy_registry()
        self.assertEqual(
            [policy.policy_id for policy in policies],
            [
                "exposure.disabled.v1",
                "exposure.constraint-release-cap2.v1",
                "exposure.constraint-release-cap1.v1",
                "exposure.constraint-release-min3.v1",
            ],
        )
        self.assertEqual(len({policy.fingerprint_sha256 for policy in policies}), 4)

    def test_disabled_mode_is_exact_response_parity(self) -> None:
        response = _response()
        exposed, decision = apply_recommendation_exposure(
            response,
            policy=disabled_exposure_policy(),
            turn=1,
            route="browsing",
            active_state=SessionState(),
            clarification_state=None,
        )
        self.assertEqual(exposed, response)
        self.assertFalse(decision.gated)

    def test_primary_only_truncates_the_existing_prefix(self) -> None:
        response = _response()
        exposed, decision = apply_recommendation_exposure(
            response,
            policy=exposure_policy_by_id("exposure.constraint-release-cap2.v1"),
            turn=1,
            route="browsing",
            active_state=SessionState(),
            clarification_state=SimpleNamespace(answered_attributes=frozenset()),
        )
        self.assertTrue(decision.gated)
        self.assertEqual(exposed["recommendations"], response["recommendations"][:1])
        self.assertEqual(exposed["usage"], response["usage"])

    def test_constraint_answer_route_and_turn_cap_release(self) -> None:
        policy = exposure_policy_by_id("exposure.constraint-release-cap2.v1")
        enough = SessionState(
            category=Constraint("shoes", "hard", "current_turn", 1),
            color=Constraint("blue", "hard", "current_turn", 1),
        )
        cases = [
            (2, "browsing", enough, frozenset(), "constraint_release"),
            (
                2,
                "browsing",
                SessionState(),
                frozenset({"other"}),
                "answered_clarification_release",
            ),
            (1, "buying", SessionState(), frozenset(), "route_release"),
            (3, "browsing", SessionState(), frozenset(), "turn_cap_release"),
        ]
        for turn, route, state, answered, reason in cases:
            with self.subTest(reason=reason):
                exposed, decision = apply_recommendation_exposure(
                    _response(),
                    policy=policy,
                    turn=turn,
                    route=route,
                    active_state=state,
                    clarification_state=SimpleNamespace(answered_attributes=answered),
                )
                self.assertFalse(decision.gated)
                self.assertEqual(decision.reason, reason)
                self.assertEqual(exposed, _response())


class RecommendationExposureAgentRobustnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.catalog = Path(self.temporary.name) / "catalog.jsonl"
        self.catalog.write_text(
            "".join(json.dumps({"parent_asin": value}) + "\n" for value in "ABC"),
            encoding="utf-8",
        )

    def _agent(self, controller: object | None = None) -> Agent:
        kwargs = {}
        if controller is not None:
            kwargs["clarification_controller"] = controller
        return Agent(
            self.catalog,
            config=HybridRetrievalConfig(mode="contextual"),
            anchor_retriever=_Retriever(),
            dense_retriever=_Retriever(),
            router=_Router(),
            clarification_config=SelectiveClarificationConfig(enabled=False),
            exposure_policy=exposure_policy_by_id(
                "exposure.constraint-release-cap2.v1"
            ),
            **kwargs,
        )

    def test_empty_query_sparse_metadata_reset_and_turn_cap(self) -> None:
        agent = self._agent()
        agent.reset("session", {})
        self.assertEqual(agent.respond("session", "", 1, 3)["recommendations"], [])
        first = agent.respond("session", "browse items", 1, 3)
        self.assertEqual(len(first["recommendations"]), 1)
        released = agent.respond("session", "show me more", 3, 3)
        self.assertEqual(len(released["recommendations"]), 3)

        agent.reset("session", {})
        repeated = agent.respond("session", "look around for products", 1, 3)
        self.assertEqual(len(repeated["recommendations"]), 1)
        events = agent.exposure_diagnostics_snapshot("session")["events"]
        self.assertEqual(len(events), 1)

    def test_missing_optional_component_fails_open_without_reranking(self) -> None:
        agent = self._agent(_ExplodingController())
        agent.reset("failure", {})
        response = agent.respond("failure", "browse items", 1, 3)
        self.assertEqual(
            [item["parent_asin"] for item in response["recommendations"]],
            ["A", "B", "C"],
        )
        diagnostics = agent.exposure_diagnostics_snapshot()
        self.assertEqual(diagnostics["exposure_failure_count"], 1)


if __name__ == "__main__":
    unittest.main()
