"""Typed loader and stable fingerprints for declarative clarification policies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .clarification_controller import ClarificationControllerConfig
from .selective_clarification import SelectiveClarificationConfig

DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "clarification_policies.json"
)


@dataclass(frozen=True, slots=True)
class ClarificationPolicy:
    """One fully specified, evaluator-ready clarification configuration."""

    policy_id: str
    rationale: str
    retrieval_policy_id: str
    evaluation_seed: int
    clarification: SelectiveClarificationConfig
    controller: ClarificationControllerConfig

    def __post_init__(self) -> None:
        if not self.policy_id.strip():
            raise ValueError("policy_id must not be empty")
        if not self.rationale.strip():
            raise ValueError("rationale must not be empty")
        if not self.retrieval_policy_id.strip():
            raise ValueError("retrieval_policy_id must not be empty")
        if (
            not isinstance(self.evaluation_seed, int)
            or isinstance(self.evaluation_seed, bool)
            or self.evaluation_seed < 0
        ):
            raise ValueError("evaluation_seed must be a non-negative integer")
        if self.clarification.required_retrieval_policy_id != self.retrieval_policy_id:
            raise ValueError("clarification and retrieval policy ids must match")

    def fingerprint_payload(self) -> dict[str, object]:
        """Return all output-affecting values in canonical schema order."""

        return {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "retrieval_policy_id": self.retrieval_policy_id,
            "evaluation_seed": self.evaluation_seed,
            "clarification": asdict(self.clarification),
            "controller": asdict(self.controller),
        }

    @property
    def fingerprint_sha256(self) -> str:
        canonical = json.dumps(
            self.fingerprint_payload(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ClarificationPolicyRegistry:
    schema_version: int
    runtime_default_policy: str
    selected_for_issue_6b: str | None
    selection_status: str
    policies: tuple[ClarificationPolicy, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"unsupported clarification policy schema {self.schema_version}"
            )
        identifiers = tuple(policy.policy_id for policy in self.policies)
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError("clarification policy ids must be non-empty and unique")
        if self.runtime_default_policy not in identifiers:
            raise ValueError("runtime_default_policy is not declared")
        if (
            self.selected_for_issue_6b is not None
            and self.selected_for_issue_6b not in identifiers
        ):
            raise ValueError("selected_for_issue_6b is not declared")
        if not self.selection_status.strip():
            raise ValueError("selection_status must not be empty")

    def policy_by_id(self, policy_id: str) -> ClarificationPolicy:
        for policy in self.policies:
            if policy.policy_id == policy_id:
                return policy
        raise ValueError(f"unknown clarification policy id: {policy_id}")

    @property
    def runtime_default(self) -> ClarificationPolicy:
        return self.policy_by_id(self.runtime_default_policy)


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _clarification_config(payload: Mapping[str, Any]) -> SelectiveClarificationConfig:
    values = dict(payload)
    eligible_routes = values.get("eligible_routes")
    if isinstance(eligible_routes, list):
        values["eligible_routes"] = tuple(eligible_routes)
    return SelectiveClarificationConfig(**values)


def load_clarification_policy_registry(
    path: str | Path = DEFAULT_POLICY_PATH,
) -> ClarificationPolicyRegistry:
    """Load and validate the complete external policy registry."""

    payload = _require_mapping(
        json.loads(Path(path).read_text(encoding="utf-8")), "registry"
    )
    evaluation_seed = payload.get("evaluation_seed")
    if (
        not isinstance(evaluation_seed, int)
        or isinstance(evaluation_seed, bool)
        or evaluation_seed < 0
    ):
        raise ValueError("evaluation_seed must be a non-negative integer")
    raw_policies = payload.get("policies")
    if not isinstance(raw_policies, list):
        raise TypeError("policies must be an array")
    policies: list[ClarificationPolicy] = []
    for index, raw_policy in enumerate(raw_policies):
        policy = _require_mapping(raw_policy, f"policies[{index}]")
        clarification = _clarification_config(
            _require_mapping(policy.get("clarification"), "clarification")
        )
        controller = ClarificationControllerConfig(
            **dict(_require_mapping(policy.get("controller"), "controller"))
        )
        loaded_policy = ClarificationPolicy(
            policy_id=str(policy.get("policy_id", "")),
            rationale=str(policy.get("rationale", "")),
            retrieval_policy_id=str(policy.get("retrieval_policy_id", "")),
            evaluation_seed=evaluation_seed,
            clarification=clarification,
            controller=controller,
        )
        declared_fingerprint = policy.get("fingerprint_sha256")
        if declared_fingerprint != loaded_policy.fingerprint_sha256:
            raise ValueError(
                f"fingerprint mismatch for clarification policy {loaded_policy.policy_id}"
            )
        policies.append(loaded_policy)
    selected = payload.get("selected_for_issue_6b")
    if selected is not None and not isinstance(selected, str):
        raise TypeError("selected_for_issue_6b must be a string or null")
    return ClarificationPolicyRegistry(
        schema_version=int(payload.get("schema_version", 0)),
        runtime_default_policy=str(payload.get("runtime_default_policy", "")),
        selected_for_issue_6b=selected,
        selection_status=str(payload.get("selection_status", "")),
        policies=tuple(policies),
    )


def clarification_policy_by_id(policy_id: str) -> ClarificationPolicy:
    return load_clarification_policy_registry().policy_by_id(policy_id)


def clarification_policy_candidates() -> tuple[ClarificationPolicy, ...]:
    return load_clarification_policy_registry().policies
