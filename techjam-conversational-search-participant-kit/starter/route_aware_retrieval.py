"""Route-aware candidate filtering, fallback merging, and deterministic fusion.

This module deliberately stops at candidate generation.  It does not implement
feature reranking or inspect conversation history outside the active
``SearchQuery`` supplied by the state manager.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from .hybrid_retrieval import Candidate, RouteRetrievalPolicy
from .lexical_retriever import CatalogDocument, tokenize

CONSTRAINT_NAMES = ("category", "color", "style", "material", "use_case")
EXCLUSION_FIELD_ALIASES = {
    "categories": "category",
    "category": "category",
    "colour": "color",
    "color": "color",
    "feature": "features",
    "features": "features",
    "material": "material",
    "materials": "material",
    "style": "style",
    "styles": "style",
    "use case": "use_case",
    "use_case": "use_case",
    "occasion": "use_case",
    "brand": "brand",
    "store": "brand",
}


class CatalogView(Protocol):
    def get(self, parent_asin: str) -> CatalogDocument | None: ...


@dataclass(frozen=True, slots=True)
class FilterSummary:
    before_count: int
    after_count: int
    hard_constraint_removals: int
    exclusion_removals: int
    missing_document_removals: int


def _finite_number(value: object) -> float | None:
    if (
        value is None
        or isinstance(value, bool)
        or not isinstance(value, (int, float, str))
    ):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _metadata_tokens(document: CatalogDocument, name: str) -> frozenset[str]:
    return frozenset(
        token for value in document.metadata.get(name, ()) for token in tokenize(value)
    )


def _constraint_match(
    document: CatalogDocument | None, name: str, value: object
) -> bool | None:
    """Return true/false for known evidence and ``None`` for missing evidence."""

    if document is None:
        return None
    terms = frozenset(tokenize(value))
    if not terms:
        return None
    dedicated = _metadata_tokens(document, name)
    if terms <= dedicated:
        return True
    if name == "category":
        evidence = frozenset(
            token
            for field_name in ("category", "title", "features", "attributes")
            for token in tokenize(document.fields.get(field_name, ""))
        )
        if terms <= evidence:
            return True
        return False if dedicated else None
    if dedicated:
        return False
    evidence = frozenset(
        token
        for field_name in (name, "title", "features", "attributes", "description")
        for token in tokenize(document.fields.get(field_name, ""))
    )
    return True if terms <= evidence else None


def _price_match(document: CatalogDocument | None, price: object) -> bool | None:
    if document is None or price is None:
        return None
    minimum = _finite_number(getattr(price, "minimum", None))
    maximum = _finite_number(getattr(price, "maximum", None))
    if minimum is not None and minimum < 0:
        return None
    if maximum is not None and maximum < 0:
        return None
    if minimum is None and maximum is None:
        return None
    if minimum is not None and maximum is not None and minimum > maximum:
        return False
    prices = tuple(
        value
        for raw in document.available_prices
        if (value := _finite_number(raw)) is not None and value >= 0
    )
    if not prices:
        return None
    return any(
        (minimum is None or value >= minimum) and (maximum is None or value <= maximum)
        for value in prices
    )


def _exclusion_present(
    document: CatalogDocument | None, raw_field: object, value: object
) -> bool:
    if document is None:
        return False
    terms = frozenset(tokenize(value))
    if not terms:
        return False
    normalized_field = " ".join(tokenize(raw_field))
    field_name = EXCLUSION_FIELD_ALIASES.get(normalized_field, normalized_field)
    evidence_fields = {
        "category": ("category",),
        "color": ("color", "title", "features", "attributes", "description"),
        "style": ("style", "title", "features", "attributes", "description"),
        "material": ("material", "title", "features", "attributes", "description"),
        "use_case": ("use_case", "title", "features", "attributes", "description"),
        "features": ("features", "title", "attributes", "description"),
        "brand": ("brand", "title", "description"),
    }.get(field_name, (field_name, "attributes"))
    evidence = set(_metadata_tokens(document, field_name))
    for evidence_field in evidence_fields:
        evidence.update(tokenize(document.fields.get(evidence_field, "")))
    return terms <= evidence


def filter_candidates(
    query: object,
    candidates: Iterable[Candidate],
    catalog: CatalogView,
    policy: RouteRetrievalPolicy,
) -> tuple[list[Candidate], FilterSummary]:
    """Apply only the filters explicitly enabled by the selected route policy."""

    candidate_list = list(candidates)
    kept: list[Candidate] = []
    hard_removals = 0
    exclusion_removals = 0
    missing_document_removals = 0
    exclusions = getattr(query, "exclusions", None)

    for candidate in candidate_list:
        document = catalog.get(candidate.parent_asin)
        hard_violations: list[str] = []
        exclusion_violations: list[str] = []

        if policy.apply_hard_filters:
            for name in CONSTRAINT_NAMES:
                constraint = getattr(query, name, None)
                if (
                    constraint is None
                    or getattr(constraint, "strength", None) != "hard"
                ):
                    continue
                if (
                    _constraint_match(
                        document, name, getattr(constraint, "value", None)
                    )
                    is not True
                ):
                    hard_violations.append(name)
            price = getattr(query, "price", None)
            if (
                price is not None
                and getattr(price, "strength", None) == "hard"
                and _price_match(document, price) is not True
            ):
                hard_violations.append("price")

        if policy.apply_exclusions and isinstance(exclusions, Mapping):
            for raw_field, raw_values in sorted(
                exclusions.items(), key=lambda item: str(item[0])
            ):
                values: Iterable[object]
                if isinstance(raw_values, str):
                    values = (raw_values,)
                elif isinstance(raw_values, bytes):
                    values = (raw_values.decode(errors="ignore"),)
                elif isinstance(raw_values, Iterable):
                    values = sorted(raw_values, key=str)
                else:
                    continue
                for value in values:
                    if _exclusion_present(document, raw_field, value):
                        exclusion_violations.append(f"{raw_field}:{value}")

        candidate.filter_diagnostics = {
            "policy_id": policy.policy_id,
            "hard_violations": tuple(hard_violations),
            "exclusion_violations": tuple(exclusion_violations),
        }
        if hard_violations:
            hard_removals += 1
            if document is None:
                missing_document_removals += 1
            continue
        if exclusion_violations:
            exclusion_removals += 1
            continue
        kept.append(candidate)

    return kept, FilterSummary(
        before_count=len(candidate_list),
        after_count=len(kept),
        hard_constraint_removals=hard_removals,
        exclusion_removals=exclusion_removals,
        missing_document_removals=missing_document_removals,
    )


def merge_fallback_candidates(
    candidates: Iterable[Candidate],
    fallback_candidates: Iterable[object],
    valid_catalog_ids: frozenset[str] | set[str],
) -> list[Candidate]:
    """Merge fallback provenance by exact catalog identity without duplicates."""

    merged = {candidate.parent_asin: candidate for candidate in candidates}
    for fallback in fallback_candidates:
        parent_asin = getattr(fallback, "parent_asin", None)
        rank = getattr(fallback, "fallback_rank", None)
        if rank is None:
            rank = getattr(fallback, "rank", None)
        score = _finite_number(getattr(fallback, "fallback_score", None))
        if (
            not isinstance(parent_asin, str)
            or not parent_asin
            or parent_asin not in valid_catalog_ids
            or not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank < 1
            or score is None
        ):
            continue
        candidate = merged.get(parent_asin)
        if candidate is None:
            candidate = Candidate(parent_asin=parent_asin)
            merged[parent_asin] = candidate
        if candidate.fallback_rank is None or rank < candidate.fallback_rank:
            candidate.fallback_rank = rank
            candidate.fallback_score = score
        elif rank == candidate.fallback_rank:
            candidate.fallback_score = max(candidate.fallback_score, score)
        candidate.sources.add("fallback")
    return list(merged.values())


def route_reciprocal_rank_fusion(
    candidates: Iterable[Candidate],
    policy: RouteRetrievalPolicy,
    *,
    limit: int | None = None,
) -> list[Candidate]:
    """Fuse lexical, dense, and fallback ranks under one configured policy."""

    resolved_limit = policy.final_candidate_count if limit is None else limit
    if (
        not isinstance(resolved_limit, int)
        or isinstance(resolved_limit, bool)
        or resolved_limit <= 0
    ):
        return []
    ranked = list(candidates)
    for candidate in ranked:
        lexical = (
            policy.lexical_weight / (policy.rrf_k + candidate.lexical_rank)
            if candidate.lexical_rank is not None
            else 0.0
        )
        dense = (
            policy.dense_weight / (policy.rrf_k + candidate.dense_rank)
            if candidate.dense_rank is not None
            else 0.0
        )
        fallback = (
            policy.fallback_weight / (policy.rrf_k + candidate.fallback_rank)
            if candidate.fallback_rank is not None
            else 0.0
        )
        candidate.component_scores = {
            "lexical": lexical,
            "dense": dense,
            "fallback": fallback,
        }
        candidate.fusion_score = lexical + dense + fallback

    sentinel = 2**63 - 1
    ranked.sort(
        key=lambda item: (
            -item.fusion_score,
            min(
                item.lexical_rank if item.lexical_rank is not None else sentinel,
                item.dense_rank if item.dense_rank is not None else sentinel,
                item.fallback_rank if item.fallback_rank is not None else sentinel,
            ),
            item.lexical_rank if item.lexical_rank is not None else sentinel,
            item.dense_rank if item.dense_rank is not None else sentinel,
            item.fallback_rank if item.fallback_rank is not None else sentinel,
            item.parent_asin,
        )
    )
    return ranked[:resolved_limit]
