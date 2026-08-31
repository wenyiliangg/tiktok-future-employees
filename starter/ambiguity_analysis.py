"""Deterministic candidate-pool ambiguity and clarification analysis.

The analyzer inspects candidates and catalog metadata only.  It neither writes
user-facing questions nor changes retrieval, ranking, routing, or agent output.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from starter.conversation_state import SessionState


ANALYZED_ATTRIBUTES = (
    "category",
    "price",
    "color",
    "material",
    "style",
    "use_case",
    "feature",
)

ATTRIBUTE_ALIASES: dict[str, tuple[str, ...]] = {
    "category": ("categories", "category", "category_hierarchy"),
    "color": ("color", "colour"),
    "material": ("material", "materials", "fabric", "textile"),
    "style": ("style", "styles", "fit", "pattern", "cut", "theme"),
    "use_case": (
        "use_case",
        "use_cases",
        "occasion",
        "activity",
        "sport",
        "season",
        "purpose",
    ),
    "feature": ("features", "feature", "product_features"),
    "price": ("price", "prices"),
}


@dataclass(frozen=True, slots=True)
class AttributeValueStatistics:
    attribute: str
    candidate_count: int
    usable_count: int
    coverage: float
    value_counts: tuple[tuple[str, int], ...]
    dominant_share: float
    normalized_entropy: float
    expected_reduction: float

    @property
    def options(self) -> tuple[str, ...]:
        return tuple(value for value, _count in self.value_counts)


@dataclass(frozen=True, slots=True)
class ClarificationOpportunity:
    should_ask: bool
    attribute: str | None
    options: tuple[str, ...]
    expected_reduction: float
    reason: str


@dataclass(frozen=True, slots=True)
class AmbiguityConfig:
    min_candidate_count: int = 4
    min_usable_count: int = 3
    min_metadata_coverage: float = 0.65
    min_distinct_values: int = 2
    min_expected_reduction: float = 0.20
    max_dominant_share: float = 0.85
    max_options: int = 4
    max_feature_length: int = 80
    price_boundaries: tuple[float, ...] = (25.0, 50.0, 100.0, 200.0)
    attribute_priority: tuple[str, ...] = ANALYZED_ATTRIBUTES

    def __post_init__(self) -> None:
        for name in ("min_candidate_count", "min_usable_count", "min_distinct_values", "max_options"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.max_feature_length, int) or self.max_feature_length < 1:
            raise ValueError("max_feature_length must be a positive integer")
        for name in ("min_metadata_coverage", "min_expected_reduction", "max_dominant_share"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if not self.price_boundaries:
            raise ValueError("price_boundaries must not be empty")
        if any(
            isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0
            for value in self.price_boundaries
        ):
            raise ValueError("price_boundaries must contain positive finite numbers")
        if tuple(sorted(set(self.price_boundaries))) != self.price_boundaries:
            raise ValueError("price_boundaries must be strictly increasing")
        if (
            len(self.attribute_priority) != len(ANALYZED_ATTRIBUTES)
            or set(self.attribute_priority) != set(ANALYZED_ATTRIBUTES)
        ):
            raise ValueError("attribute_priority must contain every analyzed attribute exactly once")


def _normalise_key(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _normalise_value(value: object, *, limit: int | None = None) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"\s+", " ", text).strip(" ,;:-\t\n").lower()
    if not text or text in {"none", "null", "n/a", "unknown", "not available"}:
        return None
    if limit is not None and len(text) > limit:
        return None
    return text


def _flatten_values(value: object) -> tuple[str, ...]:
    values: list[str] = []
    if isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: (_normalise_key(item), str(item))):
            values.extend(_flatten_values(value[key]))
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=str) if isinstance(value, (set, frozenset)) else value
        for item in items:
            values.extend(_flatten_values(item))
    else:
        normalised = _normalise_value(value)
        if normalised is not None:
            values.append(normalised)
    return tuple(values)


def _mapping_value(mapping: Mapping[object, object], aliases: tuple[str, ...]) -> object | None:
    normalised_aliases = {_normalise_key(alias) for alias in aliases}
    for key in sorted(mapping, key=lambda item: (_normalise_key(item), str(item))):
        if _normalise_key(key) in normalised_aliases:
            return mapping[key]
    return None


def _catalog_value(product: Mapping[object, object], attribute: str) -> object | None:
    aliases = ATTRIBUTE_ALIASES[attribute]
    direct = _mapping_value(product, aliases)
    if direct is not None:
        return direct
    details = _mapping_value(product, ("details", "attributes"))
    if isinstance(details, Mapping):
        return _mapping_value(details, aliases)
    return None


def _candidate_id(candidate: object) -> str | None:
    if isinstance(candidate, str):
        value = candidate
    elif isinstance(candidate, Mapping):
        value = candidate.get("parent_asin")
    else:
        value = getattr(candidate, "parent_asin", None)
    text = str(value).strip() if value is not None else ""
    return text or None


def _price_number(value: object) -> float | None:
    values = _flatten_values(value)
    parsed: list[float] = []
    for item in values:
        for match in re.findall(r"-?\d[\d,]*(?:\.\d+)?", item):
            try:
                number = float(match.replace(",", ""))
            except ValueError:
                continue
            if math.isfinite(number) and number >= 0:
                parsed.append(number)
    return min(parsed) if parsed else None


def _format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _price_bucket(value: float, boundaries: tuple[float, ...]) -> str:
    lower = 0.0
    for upper in boundaries:
        if value < upper:
            return f"{_format_number(lower)}-{_format_number(upper)}"
        lower = upper
    return f"{_format_number(boundaries[-1])}+"


def _scalar_attribute_value(product: Mapping[object, object], attribute: str) -> str | None:
    values = _flatten_values(_catalog_value(product, attribute))
    if not values:
        return None
    return values[-1] if attribute == "category" else values[0]


def _known_attributes(state: object | None) -> frozenset[str]:
    if state is None:
        return frozenset()
    return frozenset(
        attribute
        for attribute in ("category", "price", "color", "material", "style", "use_case")
        if getattr(state, attribute, None) is not None
    )


def _statistics(
    attribute: str,
    values: list[str | None],
    candidate_count: int,
) -> AttributeValueStatistics:
    usable = [value for value in values if value is not None]
    counts = Counter(usable)
    ordered_counts = tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    usable_count = len(usable)
    coverage = usable_count / candidate_count if candidate_count else 0.0
    if usable_count:
        shares = tuple(count / usable_count for _value, count in ordered_counts)
        dominant_share = max(shares)
        expected_reduction = coverage * (1.0 - sum(share * share for share in shares))
    else:
        shares = ()
        dominant_share = 0.0
        expected_reduction = 0.0
    if len(shares) > 1:
        entropy = -sum(share * math.log(share) for share in shares if share > 0)
        normalized_entropy = entropy / math.log(len(shares))
    else:
        normalized_entropy = 0.0
    return AttributeValueStatistics(
        attribute=attribute,
        candidate_count=candidate_count,
        usable_count=usable_count,
        coverage=round(coverage, 6),
        value_counts=ordered_counts,
        dominant_share=round(dominant_share, 6),
        normalized_entropy=round(normalized_entropy, 6),
        expected_reduction=round(expected_reduction, 6),
    )


class AmbiguityAnalyzer:
    """Select the single missing attribute with the strongest useful split."""

    def __init__(self, config: AmbiguityConfig | None = None) -> None:
        self.config = config or AmbiguityConfig()

    def attribute_statistics(
        self,
        candidates: Iterable[object],
        catalog: Mapping[str, Mapping[object, object]],
        state: SessionState | object | None = None,
    ) -> tuple[AttributeValueStatistics, ...]:
        candidate_ids = self._candidate_ids(candidates)
        products = [catalog.get(parent_asin) for parent_asin in candidate_ids]
        known = _known_attributes(state)
        statistics: list[AttributeValueStatistics] = []

        for attribute in self.config.attribute_priority:
            if attribute in known:
                continue
            if attribute == "feature":
                feature_statistic = self._best_feature_statistic(products, len(candidate_ids))
                if feature_statistic is not None:
                    statistics.append(feature_statistic)
                continue

            values: list[str | None] = []
            for product in products:
                if not isinstance(product, Mapping):
                    values.append(None)
                    continue
                if attribute == "price":
                    price = _price_number(_catalog_value(product, "price"))
                    values.append(
                        _price_bucket(price, self.config.price_boundaries)
                        if price is not None
                        else None
                    )
                else:
                    values.append(_scalar_attribute_value(product, attribute))
            statistics.append(_statistics(attribute, values, len(candidate_ids)))
        return tuple(statistics)

    def analyze(
        self,
        candidates: Iterable[object],
        catalog: Mapping[str, Mapping[object, object]],
        state: SessionState | object | None = None,
    ) -> ClarificationOpportunity:
        candidate_ids = self._candidate_ids(candidates)
        candidate_count = len(candidate_ids)
        if candidate_count == 0:
            return ClarificationOpportunity(False, None, (), 0.0, "candidate_pool_is_empty")
        if candidate_count < self.config.min_candidate_count:
            return ClarificationOpportunity(
                False,
                None,
                (),
                0.0,
                f"candidate_pool_too_small:{candidate_count}<{self.config.min_candidate_count}",
            )

        viable = [
            statistic
            for statistic in self.attribute_statistics(candidate_ids, catalog, state)
            if self._is_viable(statistic)
        ]
        if not viable:
            return ClarificationOpportunity(
                False,
                None,
                (),
                0.0,
                "no_missing_attribute_meets_coverage_and_reduction_thresholds",
            )

        priority = {attribute: index for index, attribute in enumerate(self.config.attribute_priority)}
        best = min(
            viable,
            key=lambda statistic: (
                -statistic.expected_reduction,
                -statistic.coverage,
                priority[statistic.attribute],
                statistic.attribute,
            ),
        )
        options = best.options[: self.config.max_options]
        reason = (
            f"selected:{best.attribute};coverage={best.coverage:.3f};"
            f"dominant_share={best.dominant_share:.3f};"
            f"expected_reduction={best.expected_reduction:.3f}"
        )
        return ClarificationOpportunity(
            should_ask=True,
            attribute=best.attribute,
            options=options,
            expected_reduction=best.expected_reduction,
            reason=reason,
        )

    def _candidate_ids(self, candidates: Iterable[object]) -> tuple[str, ...]:
        identifiers: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            identifier = _candidate_id(candidate)
            if identifier is None or identifier in seen:
                continue
            seen.add(identifier)
            identifiers.append(identifier)
        return tuple(identifiers)

    def _best_feature_statistic(
        self,
        products: list[Mapping[object, object] | None],
        candidate_count: int,
    ) -> AttributeValueStatistics | None:
        feature_sets: list[set[str] | None] = []
        all_features: set[str] = set()
        for product in products:
            if not isinstance(product, Mapping):
                feature_sets.append(None)
                continue
            raw_features = _catalog_value(product, "feature")
            if raw_features is None:
                feature_sets.append(None)
                continue
            features = {
                value
                for value in _flatten_values(raw_features)
                if _normalise_value(value, limit=self.config.max_feature_length) is not None
            }
            feature_sets.append(features)
            all_features.update(features)

        best: AttributeValueStatistics | None = None
        for feature in sorted(all_features):
            values = [
                None if features is None else (feature if feature in features else f"not:{feature}")
                for features in feature_sets
            ]
            statistic = _statistics("feature", values, candidate_count)
            if best is None or (
                statistic.expected_reduction,
                statistic.coverage,
                tuple(statistic.value_counts),
            ) > (
                best.expected_reduction,
                best.coverage,
                tuple(best.value_counts),
            ):
                best = statistic
        return best

    def _is_viable(self, statistic: AttributeValueStatistics) -> bool:
        return (
            statistic.usable_count >= self.config.min_usable_count
            and statistic.coverage >= self.config.min_metadata_coverage
            and len(statistic.value_counts) >= self.config.min_distinct_values
            and statistic.dominant_share <= self.config.max_dominant_share
            and statistic.expected_reduction >= self.config.min_expected_reduction
        )


ambiguity_analyzer = AmbiguityAnalyzer()
