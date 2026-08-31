"""One-time clarification fallback policies derived before P8 proxy outcomes."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Literal

from .ambiguity_analysis import AttributeValueStatistics
from .clarification_controller import ClarificationSessionState

FallbackVariant = Literal["disabled", "open", "catalog_utility"]

ANSWERABILITY_RATES: tuple[tuple[str, float], ...] = (
    ("budget", 0.0053),
    ("color", 0.42718),
    ("feature", 0.95804),
    ("material", 0.57302),
    ("style", 0.16178),
    ("use_case", 0.01626),
)
MEAN_YIELD_WHEN_ANSWERABLE: tuple[tuple[str, float], ...] = (
    ("budget", 1.060377),
    ("color", 1.131982),
    ("feature", 1.752349),
    ("material", 1.821123),
    ("style", 1.080109),
    ("use_case", 1.066421),
)


@dataclass(frozen=True, slots=True)
class ClarificationFallbackPolicy:
    policy_id: str
    variant: FallbackVariant = "disabled"
    enabled: bool = False
    required_retrieval_policy_id: str = "contextual.category-evidence.v1"
    minimum_turn: int = 3
    maximum_turn_exclusive: int = 10
    minimum_candidate_count: int = 4
    minimum_expected_reduction: float = 0.48
    answerability_rates: tuple[tuple[str, float], ...] = ANSWERABILITY_RATES
    mean_yield_when_answerable: tuple[tuple[str, float], ...] = (
        MEAN_YIELD_WHEN_ANSWERABLE
    )

    def __post_init__(self) -> None:
        if not self.policy_id.strip() or not self.required_retrieval_policy_id.strip():
            raise ValueError("policy identifiers must not be empty")
        if self.variant not in {"disabled", "open", "catalog_utility"}:
            raise ValueError("unsupported clarification fallback variant")
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be boolean")
        if self.enabled != (self.variant != "disabled"):
            raise ValueError("enabled and variant must agree")
        if (
            not isinstance(self.minimum_turn, int)
            or isinstance(self.minimum_turn, bool)
            or not isinstance(self.maximum_turn_exclusive, int)
            or isinstance(self.maximum_turn_exclusive, bool)
            or not 1 <= self.minimum_turn < self.maximum_turn_exclusive <= 10
        ):
            raise ValueError("fallback turn bounds must be within turns one to ten")
        if (
            not isinstance(self.minimum_candidate_count, int)
            or isinstance(self.minimum_candidate_count, bool)
            or self.minimum_candidate_count < 1
        ):
            raise ValueError("minimum candidate count must be positive")
        if (
            isinstance(self.minimum_expected_reduction, bool)
            or not math.isfinite(self.minimum_expected_reduction)
            or not 0 <= self.minimum_expected_reduction <= 1
        ):
            raise ValueError("minimum expected reduction must be between zero and one")
        rate_names = tuple(name for name, _value in self.answerability_rates)
        yield_names = tuple(name for name, _value in self.mean_yield_when_answerable)
        if len(set(rate_names)) != len(rate_names) or rate_names != yield_names:
            raise ValueError("answerability and yield attributes must uniquely align")
        for _name, value in self.answerability_rates:
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("answerability rates must be finite probabilities")
        for _name, value in self.mean_yield_when_answerable:
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError("mean yields must be finite and non-negative")

    @property
    def fingerprint_sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class FallbackDecision:
    attribute: str | None
    expected_reduction: float
    utility: float
    reason: str


def disabled_fallback_policy() -> ClarificationFallbackPolicy:
    return ClarificationFallbackPolicy(policy_id="clarification-fallback.disabled.v1")


def open_fallback_policy() -> ClarificationFallbackPolicy:
    return ClarificationFallbackPolicy(
        policy_id="clarification-fallback.open-once.v1",
        variant="open",
        enabled=True,
    )


def catalog_utility_fallback_policy() -> ClarificationFallbackPolicy:
    return ClarificationFallbackPolicy(
        policy_id="clarification-fallback.catalog-utility-once.v1",
        variant="catalog_utility",
        enabled=True,
    )


def fallback_policy_by_id(policy_id: str) -> ClarificationFallbackPolicy:
    policies = {
        policy.policy_id: policy
        for policy in (
            disabled_fallback_policy(),
            open_fallback_policy(),
            catalog_utility_fallback_policy(),
        )
    }
    try:
        return policies[policy_id]
    except KeyError as error:
        raise ValueError(
            f"unknown clarification fallback policy: {policy_id}"
        ) from error


def _known_attributes(active_state: object | None) -> set[str]:
    if active_state is None:
        return set()
    known = {
        attribute
        for attribute in ("color", "material", "style", "use_case")
        if getattr(active_state, attribute, None) is not None
    }
    if getattr(active_state, "price", None) is not None:
        known.add("budget")
    return known


def choose_fallback_attribute(
    policy: ClarificationFallbackPolicy,
    statistics: Iterable[AttributeValueStatistics],
    clarification_state: ClarificationSessionState | None,
    active_state: object | None,
) -> FallbackDecision:
    """Choose one target-independent attribute or return a reasoned no-op."""

    if not policy.enabled or clarification_state is None:
        return FallbackDecision(None, 0.0, 0.0, "disabled_or_missing_state")
    excluded = set(clarification_state.asked_attributes)
    excluded.update(clarification_state.answered_attributes)
    excluded.update(clarification_state.declined_attributes)
    if clarification_state.pending_attribute is not None:
        excluded.add(clarification_state.pending_attribute)
    excluded.update(_known_attributes(active_state))
    by_attribute = {
        ("budget" if item.attribute == "price" else item.attribute): item
        for item in statistics
    }
    rates = dict(policy.answerability_rates)
    yields = dict(policy.mean_yield_when_answerable)
    viable = [
        (attribute, by_attribute[attribute])
        for attribute in rates
        if attribute not in excluded
        and attribute in by_attribute
        and by_attribute[attribute].expected_reduction
        >= policy.minimum_expected_reduction
    ]
    if not viable:
        return FallbackDecision(None, 0.0, 0.0, "no_unresolved_ambiguity")
    if policy.variant == "open":
        if "other" in excluded:
            return FallbackDecision(None, 0.0, 0.0, "open_channel_unavailable")
        maximum = max(item.expected_reduction for _name, item in viable)
        return FallbackDecision("other", maximum, maximum, "open_fallback")
    scored = [
        (
            rates[attribute] * yields[attribute] * statistic.expected_reduction,
            attribute,
            statistic.expected_reduction,
        )
        for attribute, statistic in viable
    ]
    utility, attribute, reduction = min(scored, key=lambda item: (-item[0], item[1]))
    if utility <= 0:
        return FallbackDecision(None, 0.0, 0.0, "no_positive_catalog_utility")
    return FallbackDecision(
        attribute,
        reduction,
        utility,
        "catalog_answerability_yield_reduction",
    )
