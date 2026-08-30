"""Feature-gated eligibility policy for selective clarification.

This module decides whether an already-ranked candidate pool offers enough
runtime-observable value to ask one question.  It does not retrieve, inspect
benchmark labels, mutate preferences, or compose responses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .ambiguity_analysis import ClarificationOpportunity


PRIORITY_ATTRIBUTES = frozenset(
    {
        "category",
        "material",
        "color",
        "size",
        "style",
        "brand",
        "budget",
        "feature",
        "use_case",
        "other",
    }
)


@dataclass(frozen=True, slots=True)
class SelectiveClarificationConfig:
    """Conservative, deterministic gates layered on top of Issue 5A."""

    enabled: bool = False
    required_retrieval_policy_id: str = "contextual.browsing-dense.v1"
    analysis_candidate_limit: int = 50
    browsing_min_candidates: int = 4
    browsing_min_expected_reduction: float = 0.20
    buying_min_candidates: int = 8
    buying_min_expected_reduction: float = 0.50
    eligible_routes: tuple[str, ...] = ("browsing", "buying")
    question_priority: tuple[str, ...] = ()
    priority_min_candidates: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean")
        if not self.required_retrieval_policy_id.strip():
            raise ValueError("required_retrieval_policy_id must not be empty")
        for name in (
            "analysis_candidate_limit",
            "browsing_min_candidates",
            "buying_min_candidates",
            "priority_min_candidates",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "browsing_min_expected_reduction",
            "buying_min_expected_reduction",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be between 0 and 1")
        if self.buying_min_candidates < self.browsing_min_candidates:
            raise ValueError("Buying candidate threshold must be at least Browsing")
        if self.buying_min_expected_reduction <= self.browsing_min_expected_reduction:
            raise ValueError(
                "Buying usefulness threshold must be stricter than Browsing"
            )
        if not self.eligible_routes or any(
            route not in {"browsing", "buying", "boundary", "uncertain"}
            for route in self.eligible_routes
        ):
            raise ValueError("eligible_routes contains an unsupported observable route")
        if len(set(self.question_priority)) != len(self.question_priority) or any(
            attribute not in PRIORITY_ATTRIBUTES for attribute in self.question_priority
        ):
            raise ValueError(
                "question_priority contains a duplicate or unsupported attribute"
            )

    @property
    def uses_priority_strategy(self) -> bool:
        """Return whether the opt-in ordered question strategy is active."""

        return bool(self.question_priority)

    def priority_is_eligible(self, route: str, candidate_count: int) -> bool:
        """Apply runtime-only route and pool gates for an ordered question policy."""

        return (
            self.enabled
            and self.uses_priority_strategy
            and route in self.eligible_routes
            and candidate_count >= self.priority_min_candidates
        )

    def is_eligible(
        self,
        route: str,
        candidate_count: int,
        opportunity: ClarificationOpportunity,
    ) -> bool:
        """Return whether the route-specific conservative gates all pass."""

        if not self.enabled or route not in self.eligible_routes:
            return False
        if not opportunity.should_ask or opportunity.attribute is None:
            return False
        if route == "buying":
            return (
                candidate_count >= self.buying_min_candidates
                and opportunity.expected_reduction >= self.buying_min_expected_reduction
            )
        if route == "browsing":
            return (
                candidate_count >= self.browsing_min_candidates
                and opportunity.expected_reduction
                >= self.browsing_min_expected_reduction
            )
        return False
