"""Deterministic feature-based reranking over an existing candidate pool."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from .hybrid_retrieval import Candidate
from .lexical_retriever import CatalogDocument, CatalogDocumentBuilder, tokenize
from .search_models import SearchQuery


FEATURE_NAMES = (
    "lexical_score",
    "lexical_rank",
    "dense_score",
    "dense_rank",
    "fusion_score",
    "category_match",
    "attribute_coverage",
    "price_compatibility",
    "color_match",
    "style_match",
    "material_match",
    "use_case_match",
    "profile_affinity",
)

DEFAULT_FEATURE_WEIGHTS = {
    "lexical_score": 0.50,
    "lexical_rank": 0.75,
    "dense_score": 0.50,
    "dense_rank": 0.75,
    "fusion_score": 1.50,
    "category_match": 3.00,
    "attribute_coverage": 1.50,
    "price_compatibility": 1.00,
    "color_match": 1.00,
    "style_match": 1.00,
    "material_match": 1.00,
    "use_case_match": 1.00,
    "profile_affinity": 0.50,
}

ConstraintPolicy = Literal["filter", "penalize"]
TIE_BREAKER = "score_original_position_asin"
ATTRIBUTE_NAMES = ("color", "style", "material", "use_case")
CONSTRAINT_NAMES = ("category", *ATTRIBUTE_NAMES)

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


class CatalogUnavailableError(RuntimeError):
    """A catalog view cannot currently serve lookups."""


class CatalogView(Protocol):
    def get(self, parent_asin: str) -> CatalogDocument | None:
        """Return normalized metadata for one canonical product identifier."""


class InMemoryCatalogView:
    """Read-only normalized catalog lookup used by the reranker."""

    def __init__(
        self,
        products: Iterable[object],
        document_builder: CatalogDocumentBuilder | None = None,
    ) -> None:
        builder = document_builder or CatalogDocumentBuilder()
        documents: dict[str, CatalogDocument] = {}
        for product in products:
            document = builder.build(product)
            if document is not None and document.parent_asin not in documents:
                documents[document.parent_asin] = document
        self._documents = documents

    @classmethod
    def from_jsonl(
        cls,
        catalog_path: str | Path,
        document_builder: CatalogDocumentBuilder | None = None,
    ) -> InMemoryCatalogView:
        path = Path(catalog_path)

        def products() -> Iterable[object]:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"invalid catalog JSON on line {line_number}"
                        ) from error

        return cls(products(), document_builder=document_builder)

    def get(self, parent_asin: str) -> CatalogDocument | None:
        return self._documents.get(parent_asin)


@dataclass(frozen=True, slots=True)
class RerankerConfig:
    """Feature weights and policies for deterministic reranking."""

    feature_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_FEATURE_WEIGHTS)
    )
    contradiction_penalty: float = 2.0
    hard_constraint_penalty: float = 1_000.0
    exclusion_penalty: float = 1_000.0
    hard_constraint_policy: ConstraintPolicy = "filter"
    exclusion_policy: ConstraintPolicy = "filter"
    missing_metadata_value: float = 0.0
    missing_retrieval_value: float = 0.0
    tie_breaker: str = TIE_BREAKER

    def __post_init__(self) -> None:
        unknown = set(self.feature_weights) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"unknown reranking feature weights: {sorted(unknown)}")
        numeric_values = {
            **dict(self.feature_weights),
            "contradiction_penalty": self.contradiction_penalty,
            "hard_constraint_penalty": self.hard_constraint_penalty,
            "exclusion_penalty": self.exclusion_penalty,
            "missing_metadata_value": self.missing_metadata_value,
            "missing_retrieval_value": self.missing_retrieval_value,
        }
        try:
            invalid = [
                name
                for name, value in numeric_values.items()
                if isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0
            ]
        except (TypeError, ValueError) as error:
            raise ValueError("reranker weights and policies must be finite numbers") from error
        if invalid:
            raise ValueError(
                f"reranker weights and policies must be finite and non-negative: {invalid}"
            )
        if self.hard_constraint_policy not in {"filter", "penalize"}:
            raise ValueError("hard_constraint_policy must be 'filter' or 'penalize'")
        if self.exclusion_policy not in {"filter", "penalize"}:
            raise ValueError("exclusion_policy must be 'filter' or 'penalize'")
        if self.tie_breaker != TIE_BREAKER:
            raise ValueError(f"tie_breaker must be {TIE_BREAKER!r}")


def _finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _valid_rank(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _canonical_id(candidate: Candidate) -> str:
    value = getattr(candidate, "parent_asin", "")
    return value.strip() if isinstance(value, str) else str(value).strip()


def _merge_duplicate(existing: Candidate, incoming: Candidate) -> None:
    """Merge the strongest source signals into a copied first occurrence."""

    for rank_name in ("lexical_rank", "dense_rank"):
        current = _valid_rank(getattr(existing, rank_name, None))
        alternative = _valid_rank(getattr(incoming, rank_name, None))
        if alternative is not None and (current is None or alternative < current):
            setattr(existing, rank_name, alternative)
    for score_name in ("lexical_score", "dense_score", "fusion_score"):
        current = _finite_float(getattr(existing, score_name, None))
        alternative = _finite_float(getattr(incoming, score_name, None))
        if alternative is not None and (current is None or alternative > current):
            setattr(existing, score_name, alternative)
    existing.sources = set(getattr(existing, "sources", set())) | set(
        getattr(incoming, "sources", set())
    )


def _deduplicate(candidates: Iterable[Candidate]) -> tuple[list[Candidate], dict[str, list[int]]]:
    unique: list[Candidate] = []
    by_id: dict[str, Candidate] = {}
    positions: dict[str, list[int]] = {}
    for position, candidate in enumerate(candidates):
        parent_asin = _canonical_id(candidate)
        if not parent_asin:
            continue
        positions.setdefault(parent_asin, []).append(position)
        if parent_asin in by_id:
            _merge_duplicate(by_id[parent_asin], candidate)
            continue
        cloned = copy.deepcopy(candidate)
        cloned.original_position = position
        cloned.rerank_score = None
        cloned.rerank_diagnostics = None
        by_id[parent_asin] = cloned
        unique.append(cloned)
    return unique, positions


def _source_present(candidate: Candidate, source: str) -> bool:
    sources = getattr(candidate, "sources", set())
    if source in sources:
        return True
    rank = _valid_rank(getattr(candidate, f"{source}_rank", None))
    score = _finite_float(getattr(candidate, f"{source}_score", None))
    return rank is not None or (score is not None and score != 0.0)


def _normalized_scores(
    candidates: list[Candidate], attribute: str, presence: list[bool], missing: float
) -> list[float]:
    raw = [_finite_float(getattr(candidate, attribute, None)) for candidate in candidates]
    observed = [value for value, present in zip(raw, presence) if present and value is not None]
    if not observed:
        return [missing] * len(candidates)
    minimum, maximum = min(observed), max(observed)
    return [
        missing
        if not present or value is None
        else 1.0
        if maximum == minimum
        else (value - minimum) / (maximum - minimum)
        for value, present in zip(raw, presence)
    ]


def _metadata_tokens(document: CatalogDocument, name: str) -> frozenset[str]:
    return frozenset(
        token
        for value in document.metadata.get(name, ())
        for token in tokenize(value)
    )


def _constraint_match(
    document: CatalogDocument | None, name: str, value: object
) -> bool | None:
    """Return True/False for known metadata and None for unknown metadata."""

    if document is None:
        return None
    terms = frozenset(tokenize(value))
    if not terms:
        return None
    dedicated = _metadata_tokens(document, name)
    if terms <= dedicated:
        return True
    if name == "category":
        # Catalog category paths can be broad (for example, "shoes") while the
        # title carries the exact active category (for example, "sneakers").
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
    evidence_fields = (
        ("category",) if name == "category" else (name, "title", "features", "attributes", "description")
    )
    evidence = frozenset(
        token
        for field_name in evidence_fields
        for token in tokenize(document.fields.get(field_name, ""))
    )
    return True if terms <= evidence else None


def _safe_bound(value: object) -> float | None:
    result = _finite_float(value)
    return result if result is not None and result >= 0 else None


def _price_match(document: CatalogDocument | None, price: object) -> bool | None:
    if document is None or price is None:
        return None
    minimum = _safe_bound(getattr(price, "minimum", None))
    maximum = _safe_bound(getattr(price, "maximum", None))
    if minimum is None and maximum is None:
        return None
    if minimum is not None and maximum is not None and minimum > maximum:
        return None
    values = tuple(
        value
        for raw in document.available_prices
        if (value := _finite_float(raw)) is not None and value >= 0
    )
    if not values:
        return None
    return any(
        (minimum is None or value >= minimum)
        and (maximum is None or value <= maximum)
        for value in values
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
    tokens = set(_metadata_tokens(document, field_name))
    for evidence_field in evidence_fields:
        tokens.update(tokenize(document.fields.get(evidence_field, "")))
    return terms <= tokens


class FeatureReranker:
    """Score and reorder only the unique products supplied by the caller."""

    def __init__(self, config: RerankerConfig | None = None) -> None:
        self.config = config or RerankerConfig()
        self.last_diagnostics: dict[str, dict[str, object]] = {}

    def rerank(
        self,
        query: SearchQuery,
        candidates: list[Candidate],
        catalog: CatalogView | None,
        top_k: int,
    ) -> list[Candidate]:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            self.last_diagnostics = {}
            return []
        unique, duplicate_positions = _deduplicate(candidates)
        if not unique:
            self.last_diagnostics = {}
            return []
        if catalog is None:
            return self._fallback(unique, duplicate_positions, top_k, "catalog_unavailable")

        try:
            documents = {
                _canonical_id(candidate): catalog.get(_canonical_id(candidate))
                for candidate in unique
            }
        except (CatalogUnavailableError, OSError):
            return self._fallback(unique, duplicate_positions, top_k, "catalog_unavailable")

        retrieval_features = self._retrieval_features(unique)
        ranked: list[Candidate] = []
        diagnostics: dict[str, dict[str, object]] = {}
        exclusions = getattr(query, "exclusions", None)

        for index, candidate in enumerate(unique):
            parent_asin = _canonical_id(candidate)
            document = documents[parent_asin]
            features = {
                name: retrieval_features[name][index]
                for name in (
                    "lexical_score",
                    "lexical_rank",
                    "dense_score",
                    "dense_rank",
                    "fusion_score",
                )
            }
            matches: dict[str, bool | None] = {}
            hard_violations: list[str] = []
            contradictions: list[str] = []

            for name in CONSTRAINT_NAMES:
                constraint = getattr(query, name, None)
                match = (
                    _constraint_match(document, name, getattr(constraint, "value", None))
                    if constraint is not None
                    else None
                )
                matches[name] = match
                features[f"{name}_match"] = (
                    1.0 if match is True else self.config.missing_metadata_value if match is None else 0.0
                )
                if match is False:
                    contradictions.append(name)
                    if getattr(constraint, "strength", None) == "hard":
                        hard_violations.append(name)

            active_attributes = [
                name for name in ATTRIBUTE_NAMES if getattr(query, name, None) is not None
            ]
            features["attribute_coverage"] = (
                sum(matches[name] is True for name in active_attributes) / len(active_attributes)
                if active_attributes
                else 0.0
            )

            price_constraint = getattr(query, "price", None)
            price_match = _price_match(document, price_constraint)
            features["price_compatibility"] = (
                1.0
                if price_match is True
                else self.config.missing_metadata_value
                if price_match is None
                else 0.0
            )
            if price_match is False:
                contradictions.append("price")
                if getattr(price_constraint, "strength", None) == "hard":
                    hard_violations.append("price")

            profile_constraints: list[str] = []
            profile_matches = 0
            for name in CONSTRAINT_NAMES:
                constraint = getattr(query, name, None)
                if constraint is not None and getattr(constraint, "source", None) == "profile":
                    profile_constraints.append(name)
                    profile_matches += matches[name] is True
            if price_constraint is not None and getattr(price_constraint, "source", None) == "profile":
                profile_constraints.append("price")
                profile_matches += price_match is True
            features["profile_affinity"] = (
                profile_matches / len(profile_constraints) if profile_constraints else 0.0
            )

            exclusion_violations: list[str] = []
            if isinstance(exclusions, Mapping):
                for raw_field, raw_values in sorted(exclusions.items(), key=lambda item: str(item[0])):
                    if isinstance(raw_values, (str, bytes)):
                        values = (raw_values,)
                    elif isinstance(raw_values, Iterable):
                        values = sorted(raw_values, key=str)
                    else:
                        continue
                    for value in values:
                        if _exclusion_present(document, raw_field, value):
                            exclusion_violations.append(f"{raw_field}:{value}")

            contributions = {
                name: features[name] * float(self.config.feature_weights.get(name, 0.0))
                for name in FEATURE_NAMES
            }
            contributions["contradictions"] = -self.config.contradiction_penalty * len(contradictions)
            contributions["hard_constraints"] = -self.config.hard_constraint_penalty * len(hard_violations)
            contributions["exclusions"] = -self.config.exclusion_penalty * len(exclusion_violations)
            score = float(sum(contributions.values()))

            removal_reasons: list[str] = []
            if hard_violations and self.config.hard_constraint_policy == "filter":
                removal_reasons.append("hard_constraint_violation")
            if exclusion_violations and self.config.exclusion_policy == "filter":
                removal_reasons.append("exclusion_violation")
            diagnostic: dict[str, object] = {
                "original_position": candidate.original_position,
                "duplicate_positions": tuple(duplicate_positions[parent_asin]),
                "features": features,
                "contributions": contributions,
                "hard_violations": tuple(hard_violations),
                "exclusion_violations": tuple(exclusion_violations),
                "contradictions": tuple(contradictions),
                "rerank_score": score,
                "removal_reason": ";".join(removal_reasons) or None,
            }
            diagnostics[parent_asin] = diagnostic
            candidate.rerank_score = score
            candidate.rerank_diagnostics = diagnostic
            if not removal_reasons:
                ranked.append(candidate)

        ranked.sort(
            key=lambda candidate: (
                -float(candidate.rerank_score or 0.0),
                candidate.original_position if candidate.original_position is not None else 2**63 - 1,
                _canonical_id(candidate),
            )
        )
        self.last_diagnostics = diagnostics
        return ranked[:top_k]

    def _retrieval_features(self, candidates: list[Candidate]) -> dict[str, list[float]]:
        lexical_present = [_source_present(candidate, "lexical") for candidate in candidates]
        dense_present = [_source_present(candidate, "dense") for candidate in candidates]
        fusion_present = [
            (_finite_float(candidate.fusion_score) or 0.0) != 0.0 for candidate in candidates
        ]
        missing = self.config.missing_retrieval_value
        return {
            "lexical_score": _normalized_scores(candidates, "lexical_score", lexical_present, missing),
            "dense_score": _normalized_scores(candidates, "dense_score", dense_present, missing),
            "fusion_score": _normalized_scores(candidates, "fusion_score", fusion_present, missing),
            "lexical_rank": [
                1.0 / rank if (rank := _valid_rank(candidate.lexical_rank)) is not None else missing
                for candidate in candidates
            ],
            "dense_rank": [
                1.0 / rank if (rank := _valid_rank(candidate.dense_rank)) is not None else missing
                for candidate in candidates
            ],
        }

    def _fallback(
        self,
        unique: list[Candidate],
        duplicate_positions: dict[str, list[int]],
        top_k: int,
        reason: str,
    ) -> list[Candidate]:
        diagnostics: dict[str, dict[str, object]] = {}
        for candidate in unique:
            parent_asin = _canonical_id(candidate)
            diagnostic: dict[str, object] = {
                "original_position": candidate.original_position,
                "duplicate_positions": tuple(duplicate_positions[parent_asin]),
                "fallback_reason": reason,
            }
            candidate.rerank_diagnostics = diagnostic
            diagnostics[parent_asin] = diagnostic
        self.last_diagnostics = diagnostics
        return unique[:top_k]
