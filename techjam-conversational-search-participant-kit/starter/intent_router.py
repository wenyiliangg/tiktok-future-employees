"""Deterministic intent routing over active conversation state.

This module selects a retrieval-policy identifier.  It deliberately does not
retrieve products, alter scores, generate candidates, or ask questions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from starter.conversation_state import SearchQuery as StateSearchQuery
from starter.conversation_state import SessionState
from starter.search_models import SearchQuery as SharedSearchQuery


Route = Literal["buying", "browsing", "boundary", "uncertain"]
Query = StateSearchQuery | SharedSearchQuery


@dataclass(frozen=True)
class RoutingDecision:
    route: Route
    confidence: float
    reasons: tuple[str, ...]
    policy_id: str


@dataclass(frozen=True)
class RouterConfig:
    """Configurable thresholds, evidence weights, cues, and policy IDs."""

    buying_threshold: float = 2.0
    browsing_threshold: float = 2.0
    conflict_margin: float = 0.75

    category_weight: float = 2.0
    attribute_weight: float = 1.0
    price_weight: float = 1.5
    use_case_buying_weight: float = 0.75
    use_case_browsing_weight: float = 1.25
    exclusion_weight: float = 0.5
    purchase_cue_weight: float = 0.75
    broad_cue_weight: float = 0.75

    current_turn_multiplier: float = 1.25
    conversation_multiplier: float = 1.0
    hard_multiplier: float = 1.0
    soft_multiplier: float = 0.75

    purchase_cues: tuple[str, ...] = (
        "buy",
        "purchase",
        "i need",
        "i want",
        "looking for",
        "must have",
        "ready to buy",
    )
    broad_cues: tuple[str, ...] = (
        "comfortable",
        "explore",
        "exploring",
        "city",
        "trip",
        "occasion",
        "ideas",
        "inspiration",
        "browse",
        "browsing",
        "versatile",
        "mood",
        "gift",
    )
    boundary_cues: tuple[str, ...] = (
        "show me something",
        "not sure what i want",
        "don't know what i want",
        "do not know what i want",
        "no preference",
        "anything is fine",
        "use your judgment",
        "use your judgement",
    )
    uncertainty_cues: tuple[str, ...] = (
        "don't know what type",
        "do not know what type",
        "not sure what type",
        "not sure which",
        "can't decide",
        "cannot decide",
        "either",
        "maybe",
    )

    buying_policy_id: str = "retrieval.buying.v1"
    browsing_policy_id: str = "retrieval.browsing.v1"
    boundary_policy_id: str = "retrieval.boundary.v1"
    uncertain_policy_id: str = "retrieval.safe-default.v1"

    def __post_init__(self) -> None:
        numeric_fields = (
            "buying_threshold",
            "browsing_threshold",
            "conflict_margin",
            "category_weight",
            "attribute_weight",
            "price_weight",
            "use_case_buying_weight",
            "use_case_browsing_weight",
            "exclusion_weight",
            "purchase_cue_weight",
            "broad_cue_weight",
            "current_turn_multiplier",
            "conversation_multiplier",
            "hard_multiplier",
            "soft_multiplier",
        )
        for name in numeric_fields:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "buying_policy_id",
            "browsing_policy_id",
            "boundary_policy_id",
            "uncertain_policy_id",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")


def _normalise(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower().replace("’", "'")).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    pattern = rf"(?<![a-z0-9]){re.escape(_normalise(phrase))}(?![a-z0-9])"
    return bool(re.search(pattern, text))


def _matches(text: str, phrases: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(phrase for phrase in phrases if _contains_phrase(text, phrase))


def _safe_get(value: object, name: str) -> object | None:
    return getattr(value, name, None) if value is not None else None


def _slot_value(state: object, query: object, slot: str) -> object | None:
    state_value = _safe_get(state, slot)
    query_value = _safe_get(query, slot)
    if state_value is None:
        return query_value
    if query_value is None:
        return state_value

    source_precedence = {"profile": 0, "conversation": 1, "current_turn": 2}
    state_source = str(_safe_get(state_value, "source") or "conversation")
    query_source = str(_safe_get(query_value, "source") or "conversation")
    if source_precedence.get(query_source, 1) > source_precedence.get(state_source, 1):
        return query_value
    return state_value


def _constraint_multiplier(constraint: object, config: RouterConfig) -> float:
    source = _safe_get(constraint, "source")
    strength = _safe_get(constraint, "strength")
    if source == "profile":
        return 0.0
    source_multiplier = (
        config.current_turn_multiplier
        if source == "current_turn"
        else config.conversation_multiplier
    )
    strength_multiplier = (
        config.soft_multiplier if strength == "soft" else config.hard_multiplier
    )
    return source_multiplier * strength_multiplier


class IntentRouter:
    """Classify active intent and select a policy identifier deterministically."""

    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config or RouterConfig()

    def route(
        self,
        state: SessionState | object | None,
        query: Query | object | None,
    ) -> RoutingDecision:
        config = self.config
        text = _normalise(_safe_get(query, "text"))
        buying_score = 0.0
        browsing_score = 0.0
        active_slots = 0
        profile_slots = 0
        reasons: list[str] = []

        for slot in ("category", "color", "style", "material", "use_case", "price"):
            constraint = _slot_value(state, query, slot)
            if constraint is None:
                continue
            source = str(_safe_get(constraint, "source") or "conversation")
            strength = str(_safe_get(constraint, "strength") or "hard")
            multiplier = _constraint_multiplier(constraint, config)
            if source == "profile":
                profile_slots += 1
                reasons.append(f"profile_only:{slot}:ignored_for_buying")
                continue

            active_slots += 1
            reasons.append(f"active:{slot}:{source}:{strength}")
            if slot == "category":
                buying_score += config.category_weight * multiplier
            elif slot == "price":
                buying_score += config.price_weight * multiplier
            elif slot == "use_case":
                buying_score += config.use_case_buying_weight * multiplier
                browsing_score += config.use_case_browsing_weight * multiplier
            else:
                buying_score += config.attribute_weight * multiplier

        exclusions = _safe_get(query, "exclusions")
        if exclusions is None:
            exclusions = _safe_get(state, "exclusions")
        if isinstance(exclusions, dict):
            exclusion_count = sum(
                len(values) for values in exclusions.values() if isinstance(values, (set, tuple, list))
            )
            if exclusion_count:
                active_slots += exclusion_count
                buying_score += config.exclusion_weight * exclusion_count
                reasons.append(f"active:exclusions:{exclusion_count}")

        purchase_matches = _matches(text, config.purchase_cues)
        broad_matches = _matches(text, config.broad_cues)
        boundary_matches = _matches(text, config.boundary_cues)
        uncertainty_matches = _matches(text, config.uncertainty_cues)

        buying_score += config.purchase_cue_weight * len(purchase_matches)
        browsing_score += config.broad_cue_weight * len(broad_matches)
        reasons.extend(f"purchase_cue:{cue}" for cue in purchase_matches)
        reasons.extend(f"broad_cue:{cue}" for cue in broad_matches)
        reasons.extend(f"boundary_cue:{cue}" for cue in boundary_matches)
        reasons.extend(f"uncertainty_cue:{cue}" for cue in uncertainty_matches)

        if boundary_matches and active_slots == 0:
            return self._decision(
                "boundary",
                0.96,
                reasons,
                buying_score,
                browsing_score,
                "explicit_boundary_without_active_constraints",
            )

        if uncertainty_matches and active_slots > 0:
            return self._decision(
                "uncertain",
                0.62,
                reasons,
                buying_score,
                browsing_score,
                "uncertainty_conflicts_with_active_constraints",
            )

        if active_slots == 0 and browsing_score >= config.browsing_threshold:
            confidence = min(0.97, 0.60 + 0.08 * (browsing_score - config.browsing_threshold))
            return self._decision(
                "browsing",
                confidence,
                reasons,
                buying_score,
                browsing_score,
                "broad_goal_without_narrow_constraints",
            )

        scores_conflict = (
            buying_score >= config.buying_threshold
            and browsing_score >= config.browsing_threshold
            and abs(buying_score - browsing_score) < config.conflict_margin
        )
        if scores_conflict:
            return self._decision(
                "uncertain",
                0.58,
                reasons,
                buying_score,
                browsing_score,
                "buying_and_browsing_evidence_conflict",
            )

        if (
            buying_score >= config.buying_threshold
            and buying_score >= browsing_score + config.conflict_margin
        ):
            confidence = min(0.99, 0.62 + 0.06 * (buying_score - config.buying_threshold))
            return self._decision(
                "buying",
                confidence,
                reasons,
                buying_score,
                browsing_score,
                "concrete_active_constraints",
            )

        if browsing_score >= config.browsing_threshold and browsing_score > buying_score:
            confidence = min(0.97, 0.60 + 0.08 * (browsing_score - config.browsing_threshold))
            return self._decision(
                "browsing",
                confidence,
                reasons,
                buying_score,
                browsing_score,
                "broad_goal_outweighs_constraints",
            )

        if active_slots > 0 or uncertainty_matches:
            return self._decision(
                "uncertain",
                0.55,
                reasons,
                buying_score,
                browsing_score,
                "insufficient_or_mixed_active_evidence",
            )

        boundary_reason = (
            "profile_evidence_is_not_active_intent" if profile_slots else "no_usable_active_intent"
        )
        return self._decision(
            "boundary",
            0.90,
            reasons,
            buying_score,
            browsing_score,
            boundary_reason,
        )

    def _decision(
        self,
        route: Route,
        confidence: float,
        reasons: list[str],
        buying_score: float,
        browsing_score: float,
        rule: str,
    ) -> RoutingDecision:
        policy_ids = {
            "buying": self.config.buying_policy_id,
            "browsing": self.config.browsing_policy_id,
            "boundary": self.config.boundary_policy_id,
            "uncertain": self.config.uncertain_policy_id,
        }
        inspected_reasons = (
            *reasons,
            f"buying_score:{buying_score:.3f}",
            f"browsing_score:{browsing_score:.3f}",
            f"rule:{rule}",
        )
        return RoutingDecision(
            route=route,
            confidence=round(max(0.0, min(1.0, confidence)), 3),
            reasons=inspected_reasons,
            policy_id=policy_ids[route],
        )


intent_router = IntentRouter()
