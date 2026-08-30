"""Declarative policy for bounded pre-override identification evidence."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class OverrideHistoryPolicy:
    policy_id: str
    retrieval_policy_id: str
    compatible_retrieval_policy_id: str
    enabled: bool = False
    phrase_limit: int = 1
    required_tail_weight: float = 0.0

    def __post_init__(self) -> None:
        if not self.policy_id.strip() or not self.retrieval_policy_id.strip():
            raise ValueError("policy ids must not be empty")
        if not self.compatible_retrieval_policy_id.strip():
            raise ValueError("compatible_retrieval_policy_id must not be empty")
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean")
        if self.phrase_limit != 1:
            raise ValueError("the bounded policy retains exactly one phrase")
        if (
            isinstance(self.required_tail_weight, bool)
            or not math.isfinite(self.required_tail_weight)
            or self.required_tail_weight < 0
        ):
            raise ValueError("required_tail_weight must be finite and non-negative")
        if self.enabled and self.required_tail_weight <= 0:
            raise ValueError("enabled history requires a positive tail weight")

    @property
    def fingerprint_sha256(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def clarification_is_compatible(self, required_policy_id: str) -> bool:
        return required_policy_id in {
            self.retrieval_policy_id,
            self.compatible_retrieval_policy_id,
        }


def override_history_policy_for_retrieval(
    retrieval_policy_id: str,
) -> OverrideHistoryPolicy:
    if retrieval_policy_id in {
        "contextual.override-history-tail.v1",
        "contextual.override-history-conjunction.v1",
        "contextual.category-evidence.v1",
    }:
        return OverrideHistoryPolicy(
            policy_id="override-history.single-tail.v1",
            retrieval_policy_id=retrieval_policy_id,
            compatible_retrieval_policy_id="contextual.feedback-memory.v1",
            enabled=True,
            required_tail_weight=0.5,
        )
    return OverrideHistoryPolicy(
        policy_id="override-history.disabled.v1",
        retrieval_policy_id=retrieval_policy_id,
        compatible_retrieval_policy_id=retrieval_policy_id,
    )
