"""Shared candidate adapters and fixed reciprocal-rank fusion.

Both Issue 1's lexical retriever and Issue 2A's dense retriever emit one-based
ranks. Candidates keep ranks and raw source scores inspectable; fusion never
compares the raw score scales.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class RetrievalMode(str, Enum):
    BM25 = "bm25"
    ANCHORED = "anchored"
    CONTEXTUAL = "contextual"
    LEXICAL = "lexical"
    DENSE = "dense"
    HYBRID = "hybrid"
    ROUTE_AWARE = "route-aware"

    @classmethod
    def parse(cls, value: str | RetrievalMode) -> RetrievalMode:
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as error:
            choices = ", ".join(mode.value for mode in cls)
            raise ValueError(
                f"invalid retrieval mode {value!r}; choose one of: {choices}"
            ) from error


@dataclass(frozen=True, slots=True)
class RouteRetrievalPolicy:
    """All candidate-generation behavior for one deterministic intent route."""

    policy_id: str
    lexical_weight: float
    dense_weight: float
    fallback_weight: float = 0.0
    lexical_candidate_count: int = 200
    dense_candidate_count: int = 200
    fallback_candidate_count: int = 0
    final_candidate_count: int = 10
    rrf_k: float = 60.0
    apply_hard_filters: bool = False
    apply_exclusions: bool = True
    always_attempt_fallback: bool = False
    fallback_trigger_count: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise ValueError("policy_id must not be empty")
        for name in (
            "lexical_candidate_count",
            "dense_candidate_count",
            "final_candidate_count",
            "fallback_trigger_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.fallback_candidate_count, int)
            or isinstance(self.fallback_candidate_count, bool)
            or self.fallback_candidate_count < 0
        ):
            raise ValueError("fallback_candidate_count must be a non-negative integer")
        for name in ("lexical_weight", "dense_weight", "fallback_weight", "rrf_k"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")


def default_route_policies() -> dict[str, RouteRetrievalPolicy]:
    """Return independent, validated defaults for every supported route."""

    return {
        "buying": RouteRetrievalPolicy(
            policy_id="retrieval.buying.v1",
            lexical_weight=2.0,
            dense_weight=1.0,
            lexical_candidate_count=250,
            dense_candidate_count=200,
            apply_hard_filters=True,
            apply_exclusions=True,
        ),
        "browsing": RouteRetrievalPolicy(
            policy_id="retrieval.browsing.v1",
            lexical_weight=0.75,
            dense_weight=1.5,
            lexical_candidate_count=250,
            dense_candidate_count=400,
            apply_hard_filters=False,
            apply_exclusions=True,
        ),
        "boundary": RouteRetrievalPolicy(
            policy_id="retrieval.boundary.v1",
            lexical_weight=0.5,
            dense_weight=0.5,
            fallback_weight=1.5,
            lexical_candidate_count=100,
            dense_candidate_count=200,
            fallback_candidate_count=50,
            apply_hard_filters=False,
            apply_exclusions=True,
            always_attempt_fallback=False,
        ),
        "uncertain": RouteRetrievalPolicy(
            policy_id="retrieval.safe-default.v1",
            lexical_weight=1.0,
            dense_weight=1.0,
            lexical_candidate_count=200,
            dense_candidate_count=200,
            apply_hard_filters=False,
            apply_exclusions=True,
        ),
    }


@dataclass(frozen=True, slots=True)
class HybridRetrievalConfig:
    mode: RetrievalMode | str = RetrievalMode.CONTEXTUAL
    lexical_candidate_count: int = 200
    dense_candidate_count: int = 200
    final_candidate_count: int = 10
    rerank_candidate_count: int = 100
    lexical_weight: float = 1.0
    dense_weight: float = 1.0
    rrf_k: float = 60.0
    enable_feature_reranker: bool = False
    enable_boundary_fallback: bool = False
    route_policies: Mapping[str, RouteRetrievalPolicy] = field(
        default_factory=default_route_policies
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", RetrievalMode.parse(self.mode))
        if not isinstance(self.enable_feature_reranker, bool):
            raise TypeError("enable_feature_reranker must be a boolean")
        if not isinstance(self.enable_boundary_fallback, bool):
            raise TypeError("enable_boundary_fallback must be a boolean")
        for name in (
            "lexical_candidate_count",
            "dense_candidate_count",
            "final_candidate_count",
            "rerank_candidate_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("lexical_weight", "dense_weight"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            isinstance(self.rrf_k, bool)
            or not math.isfinite(float(self.rrf_k))
            or float(self.rrf_k) < 0
        ):
            raise ValueError("rrf_k must be finite and non-negative")
        required_routes = {"buying", "browsing", "boundary", "uncertain"}
        if set(self.route_policies) != required_routes:
            raise ValueError(
                "route_policies must define exactly: buying, browsing, boundary, uncertain"
            )
        for route, policy in self.route_policies.items():
            if not isinstance(policy, RouteRetrievalPolicy):
                raise TypeError(f"route policy {route!r} must be RouteRetrievalPolicy")

    def policy_for(self, route: str) -> RouteRetrievalPolicy:
        """Resolve malformed or unsupported labels to the conservative policy."""

        return self.route_policies.get(route, self.route_policies["uncertain"])


@dataclass(slots=True)
class Candidate:
    parent_asin: str
    lexical_score: float = 0.0
    dense_score: float = 0.0
    lexical_rank: int | None = None
    dense_rank: int | None = None
    sources: set[str] = field(default_factory=set)
    fusion_score: float = 0.0
    component_scores: dict[str, float] = field(default_factory=dict)
    fallback_score: float = 0.0
    fallback_rank: int | None = None
    filter_diagnostics: dict[str, object] | None = None
    original_position: int | None = None
    rerank_score: float | None = None
    rerank_diagnostics: dict[str, object] | None = None


class RankedResult(Protocol):
    parent_asin: str
    score: float
    rank: int


def _validated_rank(rank: object, source: str) -> int:
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        raise ValueError(f"{source} rank must be a one-based positive integer")
    return rank


def lexical_candidate(result: RankedResult) -> Candidate:
    return Candidate(
        parent_asin=str(result.parent_asin),
        lexical_score=float(result.score),
        lexical_rank=_validated_rank(result.rank, "lexical"),
        sources={"lexical"},
    )


def dense_candidate(result: RankedResult) -> Candidate:
    return Candidate(
        parent_asin=str(result.parent_asin),
        dense_score=float(result.score),
        dense_rank=_validated_rank(result.rank, "dense"),
        sources={"dense"},
    )


def merge_candidates(
    lexical_results: Iterable[RankedResult],
    dense_results: Iterable[RankedResult],
    valid_catalog_ids: frozenset[str] | set[str],
) -> list[Candidate]:
    """Merge exact catalog identities, filtering non-catalog results."""

    merged: dict[str, Candidate] = {}

    def merge_source(result: RankedResult, source: str) -> None:
        raw_asin = getattr(result, "parent_asin", None)
        raw_score = getattr(result, "score", None)
        raw_rank = getattr(result, "rank", None)
        if (
            not isinstance(raw_asin, str)
            or not raw_asin
            or raw_asin not in valid_catalog_ids
        ):
            return
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float, str)):
            return
        try:
            score = float(raw_score)
            rank = _validated_rank(raw_rank, source)
        except (TypeError, ValueError):
            return
        if not math.isfinite(score):
            return

        existing = merged.get(raw_asin)
        if existing is None:
            existing = Candidate(parent_asin=raw_asin)
            merged[raw_asin] = existing
        rank_name = f"{source}_rank"
        score_name = f"{source}_score"
        current_rank = getattr(existing, rank_name)
        if current_rank is None or rank < current_rank:
            setattr(existing, rank_name, rank)
            setattr(existing, score_name, score)
        elif rank == current_rank:
            setattr(
                existing, score_name, max(float(getattr(existing, score_name)), score)
            )
        existing.sources.add(source)

    for result in lexical_results:
        merge_source(result, "lexical")
    for result in dense_results:
        merge_source(result, "dense")
    return list(merged.values())


def reciprocal_rank_fusion(
    candidates: Iterable[Candidate],
    config: HybridRetrievalConfig,
    *,
    limit: int | None = None,
) -> list[Candidate]:
    resolved_limit = config.final_candidate_count if limit is None else limit
    if (
        not isinstance(resolved_limit, int)
        or isinstance(resolved_limit, bool)
        or resolved_limit <= 0
    ):
        return []
    ranked = list(candidates)
    for candidate in ranked:
        lexical = (
            float(config.lexical_weight)
            / (float(config.rrf_k) + candidate.lexical_rank)
            if candidate.lexical_rank is not None
            else 0.0
        )
        dense = (
            float(config.dense_weight) / (float(config.rrf_k) + candidate.dense_rank)
            if candidate.dense_rank is not None
            else 0.0
        )
        candidate.component_scores = {"lexical": lexical, "dense": dense}
        candidate.fusion_score = lexical + dense

    sentinel = 2**63 - 1
    ranked.sort(
        key=lambda item: (
            -item.fusion_score,
            min(
                item.lexical_rank if item.lexical_rank is not None else sentinel,
                item.dense_rank if item.dense_rank is not None else sentinel,
            ),
            item.lexical_rank if item.lexical_rank is not None else sentinel,
            item.dense_rank if item.dense_rank is not None else sentinel,
            item.parent_asin,
        )
    )
    return ranked[:resolved_limit]


def rank_single_source(
    results: Iterable[RankedResult],
    source: RetrievalMode,
    valid_catalog_ids: frozenset[str] | set[str],
    limit: int,
) -> list[Candidate]:
    if source is RetrievalMode.LEXICAL:
        candidates = merge_candidates(results, (), valid_catalog_ids)
        candidates.sort(key=lambda item: (item.lexical_rank, item.parent_asin))
    elif source is RetrievalMode.DENSE:
        candidates = merge_candidates((), results, valid_catalog_ids)
        candidates.sort(key=lambda item: (item.dense_rank, item.parent_asin))
    else:
        raise ValueError("rank_single_source accepts only lexical or dense mode")
    return candidates[: max(0, limit)]
