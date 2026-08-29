"""Shared candidate adapters and fixed reciprocal-rank fusion.

Both Issue 1's lexical retriever and Issue 2A's dense retriever emit one-based
ranks. Candidates keep ranks and raw source scores inspectable; fusion never
compares the raw score scales.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class RetrievalMode(str, Enum):
    LEXICAL = "lexical"
    DENSE = "dense"
    HYBRID = "hybrid"

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
class HybridRetrievalConfig:
    mode: RetrievalMode | str = RetrievalMode.LEXICAL
    lexical_candidate_count: int = 200
    dense_candidate_count: int = 200
    final_candidate_count: int = 10
    rerank_candidate_count: int = 100
    lexical_weight: float = 1.0
    dense_weight: float = 1.0
    rrf_k: float = 60.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", RetrievalMode.parse(self.mode))
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


@dataclass(slots=True)
class Candidate:
    parent_asin: str
    lexical_score: float = 0.0
    dense_score: float = 0.0
    lexical_rank: int | None = None
    dense_rank: int | None = None
    sources: set[str] = field(default_factory=set)
    fusion_score: float = 0.0
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
    for result in lexical_results:
        candidate = lexical_candidate(result)
        if (
            candidate.parent_asin not in valid_catalog_ids
            or candidate.parent_asin in merged
        ):
            continue
        merged[candidate.parent_asin] = candidate
    for result in dense_results:
        incoming = dense_candidate(result)
        if incoming.parent_asin not in valid_catalog_ids:
            continue
        existing = merged.get(incoming.parent_asin)
        if existing is None:
            merged[incoming.parent_asin] = incoming
            continue
        existing.dense_score = incoming.dense_score
        existing.dense_rank = incoming.dense_rank
        existing.sources.add("dense")
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
