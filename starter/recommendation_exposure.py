"""Declarative, prefix-only recommendation exposure control.

The policy consumes only already available runtime state and a validated P2
response.  It never retrieves, reranks, scans catalog data, or observes labels.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .conversation_state import SessionState, slot_dict

POLICY_PATH = Path(__file__).resolve().parents[1] / "config" / "exposure_policies.json"
DISABLED_EXPOSURE_POLICY_ID = "exposure.disabled.v1"


@dataclass(frozen=True, slots=True)
class RecommendationExposurePolicy:
    """One bounded runtime-confidence policy over an existing ranked response."""

    policy_id: str
    enabled: bool = False
    gated_width: int = 1
    max_gated_turn: int = 1
    minimum_active_normalized_constraints: int = 1
    release_on_answered_clarification: bool = True
    eligible_routes: tuple[str, ...] = ()
    fail_open: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise ValueError("exposure policy_id must be non-empty")
        for name in ("enabled", "release_on_answered_clarification", "fail_open"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        for name in (
            "gated_width",
            "max_gated_turn",
            "minimum_active_normalized_constraints",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_gated_turn > 9:
            raise ValueError("max_gated_turn must release before the final turn")
        allowed_routes = {"buying", "browsing", "boundary", "uncertain"}
        if len(set(self.eligible_routes)) != len(self.eligible_routes) or any(
            route not in allowed_routes for route in self.eligible_routes
        ):
            raise ValueError("eligible_routes contains a duplicate or invalid route")
        if self.enabled and not self.eligible_routes:
            raise ValueError("enabled exposure requires at least one eligible route")
        if not self.fail_open:
            raise ValueError("exposure policies must fail open")

    @property
    def fingerprint_sha256(self) -> str:
        rendered = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(rendered).hexdigest()


@dataclass(frozen=True, slots=True)
class ExposureDecision:
    """JSON-safe explanation of one exposure decision."""

    gated: bool
    reason: str
    turn: int
    route: str
    active_constraint_count: int
    answered_clarification: bool
    available_recommendation_count: int
    exposed_recommendation_count: int
    suppressed_recommendation_count: int

    @property
    def evidence_level(self) -> str:
        if self.answered_clarification:
            return "answered_clarification"
        if self.active_constraint_count == 0:
            return "none"
        if self.active_constraint_count == 1:
            return "sparse"
        return "multiple_constraints"


def _active_constraint_count(active_state: SessionState) -> int:
    return sum(value is not None for value in slot_dict(active_state).values())


def apply_recommendation_exposure(
    response: Mapping[str, object],
    *,
    policy: RecommendationExposurePolicy,
    turn: int,
    route: str,
    active_state: SessionState,
    clarification_state: object | None,
) -> tuple[dict[str, object], ExposureDecision]:
    """Truncate only an existing validated prefix when runtime confidence is low."""

    if not isinstance(turn, int) or isinstance(turn, bool) or turn < 1:
        raise ValueError("turn must be a positive integer")
    recommendations = response.get("recommendations")
    if not isinstance(recommendations, list):
        raise TypeError("validated recommendations must be a list")
    available = len(recommendations)
    active_count = _active_constraint_count(active_state)
    answered = bool(getattr(clarification_state, "answered_attributes", ()))

    gated = False
    reason = "disabled"
    if policy.enabled:
        if route not in policy.eligible_routes:
            reason = "route_release"
        elif turn > policy.max_gated_turn:
            reason = "turn_cap_release"
        elif policy.release_on_answered_clarification and answered:
            reason = "answered_clarification_release"
        elif active_count >= policy.minimum_active_normalized_constraints:
            reason = "constraint_release"
        elif available <= policy.gated_width:
            reason = "no_suppressible_recommendations"
        else:
            gated = True
            reason = "low_runtime_confidence"

    exposed_count = min(available, policy.gated_width) if gated else available
    exposed = copy.deepcopy(dict(response))
    if gated:
        exposed["recommendations"] = copy.deepcopy(recommendations[:exposed_count])
    decision = ExposureDecision(
        gated=gated,
        reason=reason,
        turn=turn,
        route=route,
        active_constraint_count=active_count,
        answered_clarification=answered,
        available_recommendation_count=available,
        exposed_recommendation_count=exposed_count,
        suppressed_recommendation_count=available - exposed_count,
    )
    return exposed, decision


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def load_exposure_policy_registry(
    path: str | Path = POLICY_PATH,
) -> tuple[RecommendationExposurePolicy, ...]:
    payload = _require_mapping(
        json.loads(Path(path).read_text(encoding="utf-8")), "exposure registry"
    )
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported exposure policy schema")
    rows = payload.get("policies")
    if not isinstance(rows, list):
        raise TypeError("exposure policies must be a list")
    policies: list[RecommendationExposurePolicy] = []
    seen: set[str] = set()
    for row in rows:
        values = dict(_require_mapping(row, "exposure policy"))
        expected = values.pop("fingerprint_sha256", None)
        routes = values.get("eligible_routes")
        if isinstance(routes, list):
            values["eligible_routes"] = tuple(routes)
        policy = RecommendationExposurePolicy(**values)
        if policy.policy_id in seen:
            raise ValueError("exposure policy ids must be unique")
        seen.add(policy.policy_id)
        if expected != policy.fingerprint_sha256:
            raise ValueError(f"fingerprint mismatch for {policy.policy_id}")
        policies.append(policy)
    return tuple(policies)


def exposure_policy_by_id(policy_id: str) -> RecommendationExposurePolicy:
    for policy in load_exposure_policy_registry():
        if policy.policy_id == policy_id:
            return policy
    raise ValueError(f"unknown exposure policy id: {policy_id}")


def disabled_exposure_policy() -> RecommendationExposurePolicy:
    """Return an in-memory fail-open control without filesystem dependency."""

    return RecommendationExposurePolicy(policy_id=DISABLED_EXPOSURE_POLICY_ID)
