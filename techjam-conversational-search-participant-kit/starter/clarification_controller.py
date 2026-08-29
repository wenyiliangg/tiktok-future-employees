"""Deterministic clarification-question state and response composition.

This module does not decide whether clarification is useful and is deliberately
not connected to :class:`starter.agent.Agent`.  Issue 5A supplies a requested
attribute; a later integration can use this controller to safely emit at most
one contract-compatible question.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, TypeAlias, cast

from .conversation_state import SessionState, slot_dict

ClarificationAttribute = Literal[
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
]
ClarificationResolution = Literal[
    "answered",
    "confirmed",
    "declined",
    "no_preference",
]
ActiveState: TypeAlias = SessionState | Mapping[str, object]


OFFICIAL_ATTRIBUTES: tuple[ClarificationAttribute, ...] = (
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
)

ATTRIBUTE_ALIASES: dict[str, ClarificationAttribute] = {
    "price": "budget",
    "price_range": "budget",
    "usecase": "use_case",
    "occasion": "use_case",
}

PROMPT_TEMPLATES: dict[ClarificationAttribute, str] = {
    "category": "What type of product would you prefer?",
    "material": "Do you have a preferred material?",
    "color": "Do you have a preferred color?",
    "size": "What size would work best for you?",
    "style": "Do you have a preferred style?",
    "brand": "Do you have a preferred brand?",
    "budget": "What price range would you prefer?",
    "feature": "Which feature matters most to you?",
    "use_case": "What will you mainly use it for?",
    "other": "What matters most to you when choosing?",
}


@dataclass(frozen=True, slots=True)
class ClarificationPrompt:
    message: str
    ask_attribute: ClarificationAttribute


@dataclass(frozen=True, slots=True)
class ClarificationControllerConfig:
    max_questions_per_session: int = 1
    max_turns: int = 10

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_questions_per_session, int)
            or isinstance(self.max_questions_per_session, bool)
            or self.max_questions_per_session < 0
        ):
            raise ValueError("max_questions_per_session must be a non-negative integer")
        if (
            not isinstance(self.max_turns, int)
            or isinstance(self.max_turns, bool)
            or not 1 <= self.max_turns <= 10
        ):
            raise ValueError("max_turns must be an integer between 1 and 10")


@dataclass(frozen=True, slots=True)
class ClarificationSessionState:
    """Read-only snapshot of one session's clarification bookkeeping."""

    asked_attributes: frozenset[ClarificationAttribute] = frozenset()
    pending_attribute: ClarificationAttribute | None = None
    answered_attributes: frozenset[ClarificationAttribute] = frozenset()
    declined_attributes: frozenset[ClarificationAttribute] = frozenset()
    clarification_count: int = 0

    @property
    def no_preference_attributes(self) -> frozenset[ClarificationAttribute]:
        return self.declined_attributes


@dataclass(slots=True)
class _MutableClarificationState:
    asked_attributes: set[ClarificationAttribute] = field(default_factory=set)
    pending_attribute: ClarificationAttribute | None = None
    answered_attributes: set[ClarificationAttribute] = field(default_factory=set)
    declined_attributes: set[ClarificationAttribute] = field(default_factory=set)
    clarification_count: int = 0
    question_turns: set[int] = field(default_factory=set)

    def snapshot(self) -> ClarificationSessionState:
        return ClarificationSessionState(
            asked_attributes=frozenset(self.asked_attributes),
            pending_attribute=self.pending_attribute,
            answered_attributes=frozenset(self.answered_attributes),
            declined_attributes=frozenset(self.declined_attributes),
            clarification_count=self.clarification_count,
        )


def normalize_attribute(value: object) -> ClarificationAttribute | None:
    """Return a contract attribute for the finite supported input vocabulary."""

    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in ATTRIBUTE_ALIASES:
        return ATTRIBUTE_ALIASES[normalized]
    if normalized in OFFICIAL_ATTRIBUTES:
        return cast(ClarificationAttribute, normalized)
    return None


def _active_value(active_state: ActiveState, attribute: str) -> object | None:
    if isinstance(active_state, SessionState):
        return slot_dict(active_state).get(attribute)
    if isinstance(active_state, Mapping):
        return active_state.get(attribute)
    return None


def _is_known(
    active_state: ActiveState | None, attribute: ClarificationAttribute
) -> bool:
    if active_state is None:
        return False
    if attribute == "budget":
        return (
            _active_value(active_state, "price") is not None
            or _active_value(active_state, "budget") is not None
        )
    return _active_value(active_state, attribute) is not None


class ClarificationController:
    """Own isolated clarification state without owning customer preferences."""

    def __init__(self, config: ClarificationControllerConfig | None = None) -> None:
        self.config = config or ClarificationControllerConfig()
        self._sessions: dict[str, _MutableClarificationState] = {}

    def reset(self, session_id: str) -> ClarificationSessionState:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must not be empty")
        state = _MutableClarificationState()
        self._sessions[session_id] = state
        return state.snapshot()

    def state_for(self, session_id: str) -> ClarificationSessionState | None:
        """Return an immutable snapshot, or ``None`` for an invalid session."""

        if not isinstance(session_id, str) or not session_id:
            return None
        state = self._sessions.get(session_id)
        return state.snapshot() if state is not None else None

    def build_prompt(
        self,
        session_id: str,
        requested_attribute: object,
        active_state: ActiveState | None,
        turn: int,
    ) -> ClarificationPrompt | None:
        """Build and record one safe prompt, or return ``None`` when ineligible."""

        if not isinstance(session_id, str) or not session_id:
            return None
        state = self._sessions.get(session_id)
        attribute = normalize_attribute(requested_attribute)
        if (
            state is None
            or attribute is None
            or not isinstance(active_state, (SessionState, Mapping))
        ):
            return None
        if not isinstance(turn, int) or isinstance(turn, bool):
            return None
        if turn < 1 or turn >= self.config.max_turns:
            return None
        if state.clarification_count >= self.config.max_questions_per_session:
            return None
        if state.pending_attribute is not None or turn in state.question_turns:
            return None
        if attribute in state.asked_attributes or _is_known(active_state, attribute):
            return None

        prompt = ClarificationPrompt(
            message=PROMPT_TEMPLATES[attribute],
            ask_attribute=attribute,
        )
        state.asked_attributes.add(attribute)
        state.pending_attribute = attribute
        state.clarification_count += 1
        state.question_turns.add(turn)
        return prompt

    def record_resolution(
        self,
        session_id: str,
        attribute: object,
        resolution: object,
    ) -> bool:
        """Record an explicit confirmed answer or explicit no-preference result."""

        if not isinstance(session_id, str) or not session_id:
            return False
        state = self._sessions.get(session_id)
        normalized_attribute = normalize_attribute(attribute)
        if (
            state is None
            or normalized_attribute is None
            or normalized_attribute not in state.asked_attributes
            or not isinstance(resolution, str)
        ):
            return False

        normalized_resolution = resolution.strip().lower().replace("-", "_")
        if normalized_resolution in {"answered", "confirmed"}:
            state.answered_attributes.add(normalized_attribute)
            state.declined_attributes.discard(normalized_attribute)
        elif normalized_resolution in {"declined", "no_preference"}:
            state.declined_attributes.add(normalized_attribute)
            state.answered_attributes.discard(normalized_attribute)
        else:
            return False

        if state.pending_attribute == normalized_attribute:
            state.pending_attribute = None
        return True


def compose_clarification_response(
    response: Mapping[str, object],
    prompt: ClarificationPrompt | None,
) -> dict[str, object]:
    """Return an independent response with an optional clarification fragment."""

    composed = copy.deepcopy(dict(response))
    if prompt is None:
        composed["ask_attribute"] = None
        return composed
    composed["message"] = prompt.message
    composed["ask_attribute"] = prompt.ask_attribute
    return composed
