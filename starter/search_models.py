"""Shared structured query and lexical retrieval result models.

This module deliberately contains no conversation parsing.  Callers are expected
to construct :class:`SearchQuery` from conversation state maintained elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ConstraintStrength = Literal["hard", "soft"]
ConstraintSource = Literal["current_turn", "conversation", "profile"]


@dataclass(frozen=True)
class Constraint:
    value: str | float
    strength: ConstraintStrength
    source: ConstraintSource
    updated_turn: int


@dataclass(frozen=True)
class PriceConstraint:
    minimum: float | None = None
    maximum: float | None = None
    strength: ConstraintStrength = "hard"
    source: ConstraintSource = "current_turn"
    updated_turn: int = 0


@dataclass(frozen=True)
class SearchQuery:
    text: str
    category: Constraint | None = None
    color: Constraint | None = None
    style: Constraint | None = None
    material: Constraint | None = None
    use_case: Constraint | None = None
    price: PriceConstraint | None = None
    exclusions: dict[str, set[str]] | None = None


@dataclass(frozen=True)
class RetrievalResult:
    parent_asin: str
    score: float
    rank: int
    matched_constraints: tuple[str, ...] = field(default_factory=tuple)
    failed_constraints: tuple[str, ...] = field(default_factory=tuple)
