"""Deterministic contextual ranking with a protected BM25 prefix.

The contextual policies in this module never alter the BM25 implementation.
They decide how much of its non-negative prefix is protected and whether
state-aware lexical or selective dense evidence may fill the remaining slots.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from .hybrid_retrieval import Candidate, RankedResult


@dataclass(frozen=True, slots=True)
class ContextualRetrievalPolicy:
    policy_id: str
    protected_lexical_count: int = 10
    candidate_count: int = 100
    state_lexical_weight: float = 0.0
    dense_weight: float = 0.0
    dense_routes: tuple[str, ...] = ()
    rrf_k: float = 60.0
    negative_feedback_uses_active_intent: bool = False

    def __post_init__(self) -> None:
        if not self.policy_id.strip():
            raise ValueError("policy_id must not be empty")
        if not 0 <= self.protected_lexical_count <= 10:
            raise ValueError("protected_lexical_count must be between 0 and 10")
        if self.candidate_count < 10:
            raise ValueError("candidate_count must be at least 10")
        if not isinstance(self.negative_feedback_uses_active_intent, bool):
            raise TypeError("negative_feedback_uses_active_intent must be a boolean")
        for name in ("state_lexical_weight", "dense_weight", "rrf_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


def contextual_policy_candidates() -> tuple[ContextualRetrievalPolicy, ...]:
    """Return the frozen, ordered policies considered by offline selection."""

    return (
        ContextualRetrievalPolicy(policy_id="contextual.negative-rotation.v1"),
        ContextualRetrievalPolicy(
            policy_id="contextual.state-aware.v1",
            protected_lexical_count=8,
            state_lexical_weight=0.35,
        ),
        ContextualRetrievalPolicy(
            policy_id="contextual.browsing-dense.v1",
            protected_lexical_count=8,
            dense_weight=0.50,
            dense_routes=("browsing",),
        ),
        ContextualRetrievalPolicy(
            policy_id="contextual.feedback-memory.v1",
            protected_lexical_count=8,
            dense_weight=0.50,
            dense_routes=("browsing",),
            negative_feedback_uses_active_intent=True,
        ),
        ContextualRetrievalPolicy(
            policy_id="contextual.override-history-tail.v1",
            protected_lexical_count=8,
            state_lexical_weight=0.5,
            dense_weight=0.50,
            dense_routes=("browsing",),
            negative_feedback_uses_active_intent=True,
        ),
        ContextualRetrievalPolicy(
            policy_id="contextual.override-history-conjunction.v1",
            protected_lexical_count=8,
            state_lexical_weight=0.5,
            dense_weight=0.50,
            dense_routes=("browsing",),
            negative_feedback_uses_active_intent=True,
        ),
        ContextualRetrievalPolicy(
            policy_id="contextual.combined.v1",
            protected_lexical_count=8,
            state_lexical_weight=0.25,
            dense_weight=0.50,
            dense_routes=("browsing",),
        ),
    )


def policy_by_id(policy_id: str) -> ContextualRetrievalPolicy:
    for policy in contextual_policy_candidates():
        if policy.policy_id == policy_id:
            return policy
    choices = ", ".join(policy.policy_id for policy in contextual_policy_candidates())
    raise ValueError(
        f"unknown contextual policy {policy_id!r}; choose one of: {choices}"
    )


def _valid_results(
    results: Iterable[RankedResult],
    valid_catalog_ids: frozenset[str] | set[str],
    known_negative_ids: frozenset[str] | set[str],
) -> dict[str, RankedResult]:
    valid: dict[str, RankedResult] = {}
    for result in results:
        parent_asin = getattr(result, "parent_asin", None)
        rank = getattr(result, "rank", None)
        score = getattr(result, "score", None)
        if (
            not isinstance(parent_asin, str)
            or parent_asin not in valid_catalog_ids
            or parent_asin in known_negative_ids
            or parent_asin in valid
            or not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank < 1
            or isinstance(score, bool)
            or not isinstance(score, (int, float, str))
        ):
            continue
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric_score):
            valid[parent_asin] = result
    return valid


def rank_contextual_candidates(
    anchor_results: Iterable[RankedResult],
    state_lexical_results: Iterable[RankedResult],
    dense_results: Iterable[RankedResult],
    valid_catalog_ids: frozenset[str] | set[str],
    known_negative_ids: frozenset[str] | set[str],
    policy: ContextualRetrievalPolicy,
    *,
    limit: int,
) -> list[Candidate]:
    """Rank evidence deterministically while preserving a configured BM25 prefix."""

    if limit <= 0:
        return []
    anchor = _valid_results(anchor_results, valid_catalog_ids, known_negative_ids)
    state = _valid_results(state_lexical_results, valid_catalog_ids, known_negative_ids)
    dense = _valid_results(dense_results, valid_catalog_ids, known_negative_ids)
    anchor_order = sorted(anchor, key=lambda asin: (anchor[asin].rank, asin))
    protected_ids = anchor_order[: min(policy.protected_lexical_count, limit)]
    ranked: list[Candidate] = []
    for asin in protected_ids:
        result = anchor[asin]
        ranked.append(
            Candidate(
                parent_asin=asin,
                lexical_score=float(result.score),
                lexical_rank=result.rank,
                sources={"bm25"},
                component_scores={"protected_bm25": 1.0},
            )
        )

    sentinel = 2**63 - 1
    tail: list[tuple[Candidate, int, int, int]] = []
    for asin in set(anchor) | set(state) | set(dense):
        if asin in protected_ids:
            continue
        anchor_rank = anchor[asin].rank if asin in anchor else sentinel
        state_rank = state[asin].rank if asin in state else sentinel
        dense_rank = dense[asin].rank if asin in dense else sentinel
        bm25_score = 1.0 / (policy.rrf_k + anchor_rank) if asin in anchor else 0.0
        state_score = (
            policy.state_lexical_weight / (policy.rrf_k + state_rank)
            if asin in state
            else 0.0
        )
        dense_score = (
            policy.dense_weight / (policy.rrf_k + dense_rank) if asin in dense else 0.0
        )
        sources = set()
        if asin in anchor:
            sources.add("bm25")
        if asin in state:
            sources.add("state_lexical")
        if asin in dense:
            sources.add("dense")
        candidate = Candidate(
            parent_asin=asin,
            lexical_score=float(anchor[asin].score) if asin in anchor else 0.0,
            dense_score=float(dense[asin].score) if asin in dense else 0.0,
            lexical_rank=None if anchor_rank == sentinel else anchor_rank,
            dense_rank=None if dense_rank == sentinel else dense_rank,
            sources=sources,
            fusion_score=bm25_score + state_score + dense_score,
            component_scores={
                "bm25": bm25_score,
                "state_lexical": state_score,
                "dense": dense_score,
            },
        )
        tail.append((candidate, anchor_rank, state_rank, dense_rank))

    tail.sort(
        key=lambda item: (
            -item[0].fusion_score,
            item[1],
            item[2],
            item[3],
            item[0].parent_asin,
        )
    )
    ranked.extend(item[0] for item in tail[: max(0, limit - len(ranked))])
    return ranked
