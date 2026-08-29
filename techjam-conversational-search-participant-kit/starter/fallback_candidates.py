"""Deterministic catalog fallback candidate generation.

This module ranks valid catalog products when little or no usable intent is
available.  All conversation and profile evidence is applied as a score boost;
it is never converted into a catalog filter by this module.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Literal, TypeVar

from .conversation_state import ConversationStateManager
from .lexical_retriever import CatalogDocument, CatalogDocumentBuilder, tokenize

try:
    from .hybrid_retrieval import Candidate as _Issue2BCandidate
except ModuleNotFoundError as error:
    # Issue 3B was explicitly allowed to develop in parallel with Issue 2B.
    # Keep the generator importable on a pre-2B branch, but use the real shared
    # base automatically as soon as hybrid_retrieval is present.
    if not (error.name or "").endswith(".hybrid_retrieval"):
        raise

    @dataclass(slots=True)
    class _Issue2BCandidate:  # type: ignore[no-redef]
        parent_asin: str


FALLBACK_SOURCE: Literal["fallback"] = "fallback"
DIVERSITY_DIMENSIONS = frozenset(
    {"category", "brand", "style", "price_range", "product_family"}
)
EVIDENCE_FIELDS = ("category", "color", "style", "material", "use_case")

DEFAULT_EVIDENCE_WEIGHTS = {
    "category": 1.50,
    "color": 0.65,
    "style": 0.65,
    "material": 0.65,
    "use_case": 0.65,
    "price": 0.45,
}
DEFAULT_SOURCE_WEIGHTS = {
    "current_turn": 1.00,
    "conversation": 0.85,
    "profile": 0.50,
}
DEFAULT_DIVERSITY_CAPS = {
    "product_family": 1,
    "category": 2,
    "brand": 2,
    "style": 2,
    "price_range": 2,
}
DEFAULT_DIVERSITY_PENALTIES = {
    "product_family": 0.80,
    "category": 0.30,
    "brand": 0.20,
    "style": 0.15,
    "price_range": 0.10,
}


@dataclass(slots=True, init=False)
class FallbackCandidate(_Issue2BCandidate):
    """Issue 2B-compatible candidate carrying inspectable fallback fields.

    On current main this is a subtype of ``hybrid_retrieval.Candidate`` and its
    shared ``sources`` set contains ``"fallback"``.  The small local base above
    preserves Issue 3B's documented parallel-development behavior on older
    branches where Issue 2B is not present yet.
    """

    rank: int = 0
    source: Literal["fallback"] = FALLBACK_SOURCE

    def __init__(
        self,
        parent_asin: str,
        fallback_score: float,
        rank: int,
        source: Literal["fallback"] = FALLBACK_SOURCE,
    ) -> None:
        if source != FALLBACK_SOURCE:
            raise ValueError('fallback candidate source must be "fallback"')
        _Issue2BCandidate.__init__(self, parent_asin=parent_asin)
        shared_sources = getattr(self, "sources", None)
        if isinstance(shared_sources, set):
            shared_sources.add(FALLBACK_SOURCE)
        self.fallback_score = fallback_score
        if hasattr(self, "fallback_rank"):
            self.fallback_rank = rank
        self.rank = rank
        self.source = source

    def as_shared_payload(self) -> dict[str, str | float | int]:
        """Return only the fields promised to the shared candidate adapter."""

        return {
            "parent_asin": self.parent_asin,
            "fallback_score": self.fallback_score,
            "source": self.source,
            "rank": self.rank,
        }


SharedCandidate = TypeVar("SharedCandidate")


def adapt_fallback_candidates(
    candidates: Iterable[FallbackCandidate],
    candidate_factory: Callable[..., SharedCandidate] | None = None,
) -> list[dict[str, str | float | int] | SharedCandidate]:
    """Adapt fallback candidates to dictionaries or a shared candidate type.

    Passing no factory produces inspectable dictionaries.  Passing Issue 2B's
    shared class returns the already-compatible subclass instances.  Other
    keyword factories continue to receive the four fallback payload fields.
    """

    candidate_list = list(candidates)
    payloads = [candidate.as_shared_payload() for candidate in candidate_list]
    if candidate_factory is None:
        return payloads
    if isinstance(candidate_factory, type):
        return [
            candidate
            if isinstance(candidate, candidate_factory)
            else candidate_factory(**payload)
            for candidate, payload in zip(candidate_list, payloads)
        ]
    return [candidate_factory(**payload) for payload in payloads]


@dataclass(frozen=True)
class FallbackConfig:
    """Scoring and greedy diversity settings for fallback generation."""

    evidence_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_EVIDENCE_WEIGHTS)
    )
    source_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SOURCE_WEIGHTS)
    )
    rating_weight: float = 0.25
    popularity_weight: float = 0.15
    exclusion_penalty: float = 0.75
    diversity_dimensions: tuple[str, ...] = (
        "product_family",
        "category",
        "brand",
        "style",
        "price_range",
    )
    diversity_caps: Mapping[str, int] = field(
        default_factory=lambda: dict(DEFAULT_DIVERSITY_CAPS)
    )
    diversity_penalties: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_DIVERSITY_PENALTIES)
    )
    price_bucket_width: float = 50.0

    def __post_init__(self) -> None:
        unknown_dimensions = set(self.diversity_dimensions) - DIVERSITY_DIMENSIONS
        unknown_caps = set(self.diversity_caps) - DIVERSITY_DIMENSIONS
        unknown_penalties = set(self.diversity_penalties) - DIVERSITY_DIMENSIONS
        if unknown_dimensions or unknown_caps or unknown_penalties:
            unknown = sorted(unknown_dimensions | unknown_caps | unknown_penalties)
            raise ValueError(f"unknown diversity dimensions: {unknown}")
        if len(set(self.diversity_dimensions)) != len(self.diversity_dimensions):
            raise ValueError("diversity_dimensions must not contain duplicates")

        unknown_evidence = set(self.evidence_weights) - {*EVIDENCE_FIELDS, "price"}
        if unknown_evidence:
            raise ValueError(f"unknown evidence weights: {sorted(unknown_evidence)}")
        unknown_sources = set(self.source_weights) - {
            "current_turn",
            "conversation",
            "profile",
        }
        if unknown_sources:
            raise ValueError(f"unknown source weights: {sorted(unknown_sources)}")

        self._validate_non_negative(self.evidence_weights.values(), "evidence weights")
        self._validate_non_negative(self.source_weights.values(), "source weights")
        self._validate_non_negative(
            self.diversity_penalties.values(), "diversity penalties"
        )
        self._validate_non_negative(
            (self.rating_weight, self.popularity_weight, self.exclusion_penalty),
            "score weights",
        )
        if any(
            isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0
            for cap in self.diversity_caps.values()
        ):
            raise ValueError("diversity caps must be positive integers")
        if (
            isinstance(self.price_bucket_width, bool)
            or not isinstance(self.price_bucket_width, (int, float))
            or not math.isfinite(float(self.price_bucket_width))
            or self.price_bucket_width <= 0
        ):
            raise ValueError("price_bucket_width must be a positive finite number")

    @staticmethod
    def _validate_non_negative(values: Iterable[object], label: str) -> None:
        try:
            invalid = any(
                not math.isfinite(float(value)) or float(value) < 0 for value in values
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} must be finite numbers") from error
        if invalid:
            raise ValueError(f"{label} must be finite and non-negative")


class ProfileEvidenceAdapter:
    """Turn a possibly malformed aggregate profile into soft SearchQuery slots.

    Extraction delegates to Issue 1A's finite alias rules.  Purchase frequency,
    rating style, and arbitrary unknown tags are deliberately not interpreted as
    product constraints.
    """

    _SESSION_ID = "fallback-profile-adapter"

    def adapt(self, user_profile: object) -> object:
        profile = self._safe_profile(user_profile)
        manager = ConversationStateManager()
        manager.reset(self._SESSION_ID, profile)
        return manager.query_for(self._SESSION_ID)

    @staticmethod
    def _safe_profile(user_profile: object) -> dict[str, object]:
        if not isinstance(user_profile, Mapping):
            return {}

        safe: dict[str, object] = {}
        raw_tags = user_profile.get("preference_tags")
        if isinstance(raw_tags, (list, tuple, set, frozenset)):
            tags = raw_tags
            if isinstance(raw_tags, (set, frozenset)):
                tags = sorted(raw_tags, key=str)
            safe["preference_tags"] = [
                tag for tag in tags if isinstance(tag, (str, int, float))
            ]
        summary = user_profile.get("summary")
        if isinstance(summary, str):
            safe["summary"] = summary
        return safe


@dataclass(frozen=True)
class _Evidence:
    field: str
    value: object
    source: str
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class _FallbackRecord:
    parent_asin: str
    field_tokens: dict[str, frozenset[str]]
    price: float | None
    rating: float | None
    rating_count: float | None
    dimensions: dict[str, str]


@dataclass(frozen=True)
class _ScoredRecord:
    record: _FallbackRecord
    score: float


class FallbackCandidateGenerator:
    """Rank the catalog deterministically using only soft evidence and signals."""

    def __init__(
        self,
        products: Iterable[object],
        config: FallbackConfig | None = None,
        document_builder: CatalogDocumentBuilder | None = None,
        profile_adapter: ProfileEvidenceAdapter | None = None,
    ) -> None:
        self.config = config or FallbackConfig()
        self.document_builder = document_builder or CatalogDocumentBuilder()
        self.profile_adapter = profile_adapter or ProfileEvidenceAdapter()
        self._records = self._build_records(products)
        self._catalog_ids = frozenset(record.parent_asin for record in self._records)
        rating_counts = [
            record.rating_count
            for record in self._records
            if record.rating_count is not None
        ]
        self._maximum_log_rating_count = max(
            (math.log1p(value) for value in rating_counts),
            default=0.0,
        )

    @classmethod
    def from_jsonl(
        cls,
        catalog_path: str | Path,
        config: FallbackConfig | None = None,
        document_builder: CatalogDocumentBuilder | None = None,
        profile_adapter: ProfileEvidenceAdapter | None = None,
    ) -> FallbackCandidateGenerator:
        path = Path(catalog_path)

        def products() -> Iterator[object]:
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

        return cls(
            products(),
            config=config,
            document_builder=document_builder,
            profile_adapter=profile_adapter,
        )

    @property
    def catalog_ids(self) -> frozenset[str]:
        return self._catalog_ids

    def generate(
        self,
        query: object | None = None,
        user_profile: object | None = None,
        top_n: int = 10,
        removed_constraints: Iterable[str] | None = None,
    ) -> list[FallbackCandidate]:
        """Return up to ``top_n`` unique, catalog-valid fallback candidates.

        ``removed_constraints`` may be supplied from ``SessionState`` to prevent
        an explicitly removed profile slot from being reintroduced when a raw
        profile is also supplied.
        """

        if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n <= 0:
            return []
        if not self._records:
            return []

        profile_query = self.profile_adapter.adapt(user_profile)
        suppressed = self._suppressed_fields(removed_constraints)
        evidence = self._merge_evidence(query, profile_query, suppressed)
        exclusions = self._query_exclusions(query)
        scored = [
            _ScoredRecord(
                record=record,
                score=self._score(record, evidence, exclusions),
            )
            for record in self._records
        ]
        scored.sort(key=lambda item: (-item.score, item.record.parent_asin))
        selected = self._select_diverse(scored, min(top_n, len(scored)))
        return [
            FallbackCandidate(
                parent_asin=item.record.parent_asin,
                fallback_score=round(item.score, 8),
                rank=rank,
            )
            for rank, item in enumerate(selected, start=1)
        ]

    def _build_records(self, products: Iterable[object]) -> tuple[_FallbackRecord, ...]:
        records: list[_FallbackRecord] = []
        seen: set[str] = set()
        for product in products:
            document = self.document_builder.build(product)
            if document is None or document.parent_asin in seen:
                continue
            seen.add(document.parent_asin)
            source = product if isinstance(product, Mapping) else {}
            field_tokens = {
                name: frozenset(tokenize(value))
                for name, value in document.fields.items()
            }
            records.append(
                _FallbackRecord(
                    parent_asin=document.parent_asin,
                    field_tokens=field_tokens,
                    price=document.price,
                    rating=self._optional_number(
                        source.get("average_rating"), minimum=0.0, maximum=5.0
                    ),
                    rating_count=self._optional_number(
                        source.get("rating_number"), minimum=0.0
                    ),
                    dimensions=self._dimensions(document, field_tokens),
                )
            )
        return tuple(records)

    @staticmethod
    def _optional_number(
        value: object,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed):
            return None
        if minimum is not None and parsed < minimum:
            return None
        if maximum is not None and parsed > maximum:
            return None
        return parsed

    def _dimensions(
        self,
        document: CatalogDocument,
        field_tokens: Mapping[str, frozenset[str]],
    ) -> dict[str, str]:
        dimensions: dict[str, str] = {}
        for name in ("category", "brand", "style"):
            values = document.metadata.get(name, ())
            normalised = [" ".join(tokenize(value)) for value in values]
            normalised = [value for value in normalised if value]
            if normalised:
                dimensions[name] = normalised[-1] if name == "category" else normalised[0]

        if document.price is not None:
            bucket = int(document.price // float(self.config.price_bucket_width))
            dimensions["price_range"] = str(bucket)

        variant_tokens = set()
        for field_name in ("color", "material", "style"):
            variant_tokens.update(field_tokens.get(field_name, ()))
        family_tokens = [
            token
            for token in tokenize(document.fields.get("title", ""))
            if token not in variant_tokens and not token.isdigit()
        ]
        if family_tokens:
            dimensions["product_family"] = " ".join(family_tokens[:8])
        return dimensions

    @staticmethod
    def _suppressed_fields(removed_constraints: Iterable[str] | None) -> set[str]:
        if removed_constraints is None or isinstance(removed_constraints, str):
            return set()
        suppressed: set[str] = set()
        try:
            for item in removed_constraints:
                if not isinstance(item, str):
                    continue
                field_name = item.split(":", 1)[0]
                if field_name in {*EVIDENCE_FIELDS, "price"}:
                    suppressed.add(field_name)
        except TypeError:
            return set()
        return suppressed

    def _merge_evidence(
        self,
        query: object | None,
        profile_query: object,
        suppressed: set[str],
    ) -> tuple[_Evidence, ...]:
        evidence: list[_Evidence] = []
        exclusions = self._query_exclusions(query)

        for field_name in EVIDENCE_FIELDS:
            active = self._safe_constraint(getattr(query, field_name, None))
            profile = self._safe_constraint(getattr(profile_query, field_name, None))
            chosen = active
            if chosen is None and field_name not in suppressed:
                chosen = profile
            if chosen is None:
                continue
            value, source = chosen
            if source == "profile" and self._value_is_excluded(
                field_name, value, exclusions
            ):
                continue
            evidence.append(_Evidence(field=field_name, value=value, source=source))

        active_price = self._safe_price(getattr(query, "price", None))
        profile_price = self._safe_price(getattr(profile_query, "price", None))
        chosen_price = active_price
        if chosen_price is None and "price" not in suppressed:
            chosen_price = profile_price
        if chosen_price is not None:
            minimum, maximum, source = chosen_price
            evidence.append(
                _Evidence(
                    field="price",
                    value="price",
                    source=source,
                    minimum=minimum,
                    maximum=maximum,
                )
            )
        return tuple(evidence)

    @staticmethod
    def _safe_constraint(value: object) -> tuple[object, str] | None:
        if value is None:
            return None
        strength = getattr(value, "strength", None)
        source = getattr(value, "source", None)
        raw_value = getattr(value, "value", None)
        if strength not in {"hard", "soft"}:
            return None
        if source not in {"current_turn", "conversation", "profile"}:
            return None
        if not tokenize(raw_value):
            return None
        return raw_value, source

    @classmethod
    def _safe_price(cls, value: object) -> tuple[float | None, float | None, str] | None:
        if value is None:
            return None
        if getattr(value, "strength", None) not in {"hard", "soft"}:
            return None
        source = getattr(value, "source", None)
        if source not in {"current_turn", "conversation", "profile"}:
            return None
        minimum = cls._optional_number(getattr(value, "minimum", None), minimum=0.0)
        maximum = cls._optional_number(getattr(value, "maximum", None), minimum=0.0)
        if minimum is None and maximum is None:
            return None
        if minimum is not None and maximum is not None and minimum > maximum:
            return None
        return minimum, maximum, source

    @staticmethod
    def _query_exclusions(query: object | None) -> dict[str, tuple[object, ...]]:
        exclusions = getattr(query, "exclusions", None)
        if not isinstance(exclusions, Mapping):
            return {}
        safe: dict[str, tuple[object, ...]] = {}
        aliases = {"categories": "category", "colour": "color", "occasion": "use_case"}
        for raw_field, raw_values in exclusions.items():
            field_name = aliases.get(str(raw_field).lower(), str(raw_field).lower())
            if field_name not in {*EVIDENCE_FIELDS, "brand", "features"}:
                continue
            if isinstance(raw_values, str):
                values: Sequence[object] = (raw_values,)
            elif isinstance(raw_values, (list, tuple, set, frozenset)):
                values = tuple(raw_values)
            else:
                continue
            safe_values = tuple(value for value in values if tokenize(value))
            if safe_values:
                safe[field_name] = safe_values
        return safe

    @staticmethod
    def _value_is_excluded(
        field_name: str,
        value: object,
        exclusions: Mapping[str, tuple[object, ...]],
    ) -> bool:
        value_tokens = frozenset(tokenize(value))
        return any(
            frozenset(tokenize(excluded)) == value_tokens
            for excluded in exclusions.get(field_name, ())
        )

    def _score(
        self,
        record: _FallbackRecord,
        evidence: tuple[_Evidence, ...],
        exclusions: Mapping[str, tuple[object, ...]],
    ) -> float:
        score = 0.0
        if record.rating is not None:
            score += self.config.rating_weight * (record.rating / 5.0)
        if record.rating_count is not None and self._maximum_log_rating_count > 0:
            score += self.config.popularity_weight * (
                math.log1p(record.rating_count) / self._maximum_log_rating_count
            )

        for item in evidence:
            match = self._evidence_match(record, item)
            score += (
                float(self.config.evidence_weights.get(item.field, 0.0))
                * float(self.config.source_weights.get(item.source, 0.0))
                * match
            )

        for field_name, values in exclusions.items():
            for value in values:
                item = _Evidence(field=field_name, value=value, source="current_turn")
                score -= self.config.exclusion_penalty * self._evidence_match(record, item)
        return score

    def _evidence_match(self, record: _FallbackRecord, evidence: _Evidence) -> float:
        if evidence.field == "price":
            if record.price is None:
                return 0.0
            if evidence.minimum is not None and record.price < evidence.minimum:
                return 0.0
            if evidence.maximum is not None and record.price > evidence.maximum:
                return 0.0
            return 1.0

        value_tokens = frozenset(tokenize(evidence.value))
        if not value_tokens:
            return 0.0
        if evidence.field == "category":
            catalog_tokens = record.field_tokens.get("category", frozenset())
        else:
            catalog_tokens = frozenset().union(
                record.field_tokens.get(evidence.field, frozenset()),
                record.field_tokens.get("title", frozenset()),
                record.field_tokens.get("features", frozenset()),
                record.field_tokens.get("attributes", frozenset()),
                record.field_tokens.get("description", frozenset()),
            )
        return len(value_tokens & catalog_tokens) / len(value_tokens)

    def _select_diverse(
        self,
        candidates: list[_ScoredRecord],
        limit: int,
    ) -> list[_ScoredRecord]:
        if not self.config.diversity_dimensions:
            return candidates[:limit]

        remaining = list(candidates)
        selected: list[_ScoredRecord] = []
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        while remaining and len(selected) < limit:
            allowed = [
                item
                for item in remaining
                if self._within_diversity_caps(item.record, counts)
            ]
            pool = allowed or remaining
            choice = min(
                pool,
                key=lambda item: (
                    -self._diversity_adjusted_score(item, counts),
                    -item.score,
                    item.record.parent_asin,
                ),
            )
            selected.append(choice)
            remaining.remove(choice)
            for dimension in self.config.diversity_dimensions:
                value = choice.record.dimensions.get(dimension)
                if value:
                    counts[dimension][value] += 1
        return selected

    def _within_diversity_caps(
        self,
        record: _FallbackRecord,
        counts: Mapping[str, Counter[str]],
    ) -> bool:
        for dimension in self.config.diversity_dimensions:
            value = record.dimensions.get(dimension)
            cap = self.config.diversity_caps.get(dimension)
            if value and cap is not None and counts[dimension][value] >= cap:
                return False
        return True

    def _diversity_adjusted_score(
        self,
        item: _ScoredRecord,
        counts: Mapping[str, Counter[str]],
    ) -> float:
        penalty = 0.0
        for dimension in self.config.diversity_dimensions:
            value = item.record.dimensions.get(dimension)
            if value:
                penalty += (
                    float(self.config.diversity_penalties.get(dimension, 0.0))
                    * counts[dimension][value]
                )
        return item.score - penalty
