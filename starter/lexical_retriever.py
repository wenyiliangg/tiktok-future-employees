"""Deterministic field-aware lexical catalog retrieval.

The retriever uses SQLite FTS5's BM25 implementation with one index column per
catalog field.  Field weights, candidate-pool size, and soft preference boosts
are configured in :class:`LexicalRetrievalConfig`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
import sqlite3
import unicodedata

from .search_models import Constraint, RetrievalResult, SearchQuery


TOKEN_RE = re.compile(r"[a-z0-9]+")
PRICE_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "please",
        "some",
        "that",
        "the",
        "this",
        "to",
        "want",
        "with",
        "would",
        "you",
        "looking",
    }
)

INDEX_FIELDS = (
    "title",
    "category",
    "features",
    "color",
    "material",
    "style",
    "use_case",
    "attributes",
    "description",
    "brand",
)

CONSTRAINT_FIELDS = ("category", "color", "style", "material", "use_case")

DEFAULT_FIELD_WEIGHTS = {
    "title": 5.0,
    "category": 4.0,
    "features": 3.0,
    "color": 3.0,
    "material": 3.0,
    "style": 3.0,
    "use_case": 3.0,
    "attributes": 2.5,
    "description": 1.0,
    "brand": 1.5,
}

DEFAULT_SOFT_MATCH_BOOSTS = {
    "category": 1.0,
    "color": 0.8,
    "style": 0.8,
    "material": 0.8,
    "use_case": 0.8,
    "price": 0.5,
}

DETAIL_CLASSIFIERS = {
    "features": frozenset({"feature", "features"}),
    "color": frozenset({"color", "colour"}),
    "material": frozenset({"material", "fabric", "textile"}),
    "style": frozenset(
        {
            "style",
            "fit",
            "pattern",
            "neck",
            "sleeve",
            "closure",
            "shape",
            "rise",
            "cut",
            "theme",
        }
    ),
    "use_case": frozenset(
        {
            "occasion",
            "activity",
            "sport",
            "season",
            "use",
            "purpose",
            "recommended",
        }
    ),
}

TOP_LEVEL_ALIASES = {
    "title": ("title", "name"),
    "category": ("categories", "category", "category_hierarchy"),
    "features": ("features", "feature", "product_features"),
    "description": ("description", "descriptions"),
    "brand": ("store", "brand"),
    "color": ("color", "colour"),
    "material": ("material", "materials", "fabric"),
    "style": ("style", "styles"),
    "use_case": ("use_case", "use_cases", "occasion", "activity"),
}

PREFERENCE_FIELDS = frozenset({"color", "style", "material", "use_case"})
PREFERENCE_EVIDENCE_FIELDS = (
    "title",
    "features",
    "color",
    "material",
    "style",
    "use_case",
    "attributes",
    "description",
)

EXCLUSION_ALIASES = {
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


def _ascii_lower(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    return text.encode("ascii", "ignore").decode("ascii").lower()


def tokenize(value: object) -> tuple[str, ...]:
    """Return deterministic query/index terms for a possibly malformed value."""

    if value is None or isinstance(value, bool):
        return ()
    return tuple(
        token
        for token in TOKEN_RE.findall(_ascii_lower(value))
        if len(token) > 1 and token not in STOPWORDS
    )


def _normalise_key(value: object) -> str:
    return " ".join(tokenize(value))


def _ordered_mapping_items(value: Mapping[object, object]) -> list[tuple[object, object]]:
    return sorted(value.items(), key=lambda item: (_normalise_key(item[0]), str(item[0])))


def _flatten_scalars(value: object) -> Iterator[str]:
    """Flatten JSON-like values while safely ignoring missing/boolean values."""

    if value is None or isinstance(value, bool):
        return
    if isinstance(value, Mapping):
        for key, item in _ordered_mapping_items(value):
            key_text = str(key).strip()
            if key_text:
                yield key_text
            yield from _flatten_scalars(item)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        values = value
        if isinstance(value, (set, frozenset)):
            values = sorted(value, key=str)
        for item in values:
            yield from _flatten_scalars(item)
        return
    text = str(value).strip()
    if text:
        yield text


def _detail_leaves(value: object, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], str]]:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, Mapping):
        for key, item in _ordered_mapping_items(value):
            key_text = str(key).strip()
            next_path = path + ((key_text,) if key_text else ())
            yield from _detail_leaves(item, next_path)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        values = value
        if isinstance(value, (set, frozenset)):
            values = sorted(value, key=str)
        for item in values:
            yield from _detail_leaves(item, path)
        return
    text = str(value).strip()
    if text:
        yield path, text


def _values_for_aliases(product: Mapping[object, object], aliases: tuple[str, ...]) -> list[str]:
    normalised = {_normalise_key(key): value for key, value in product.items()}
    values: list[str] = []
    for alias in aliases:
        values.extend(_flatten_scalars(normalised.get(_normalise_key(alias))))
    return values


def _price_values(value: object) -> Iterator[float]:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, Mapping):
        for _, item in _ordered_mapping_items(value):
            yield from _price_values(item)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        values = value
        if isinstance(value, (set, frozenset)):
            values = sorted(value, key=str)
        for item in values:
            yield from _price_values(item)
        return
    candidates: list[float] = []
    if isinstance(value, (int, float)):
        candidates.append(float(value))
    elif isinstance(value, str):
        for match in PRICE_RE.findall(value):
            try:
                candidates.append(float(match.replace(",", "")))
            except ValueError:
                continue
    for candidate in candidates:
        if math.isfinite(candidate) and candidate >= 0:
            yield candidate


def parse_prices(value: object) -> tuple[float, ...]:
    """Parse, de-duplicate, and sort all valid non-negative prices."""

    return tuple(sorted(set(_price_values(value))))


@dataclass(frozen=True)
class CatalogDocument:
    parent_asin: str
    fields: dict[str, str]
    metadata: dict[str, tuple[str, ...]]
    available_prices: tuple[float, ...]
    price: float | None


class CatalogDocumentBuilder:
    """Build a safe, field-aware view without mutating the source product."""

    def build(self, product: object) -> CatalogDocument | None:
        if not isinstance(product, Mapping):
            return None
        raw_asin = product.get("parent_asin")
        if raw_asin is None or isinstance(raw_asin, (bool, Mapping, list, tuple, set)):
            return None
        parent_asin = str(raw_asin).strip()
        if not parent_asin:
            return None

        field_values: dict[str, list[str]] = {name: [] for name in INDEX_FIELDS}
        metadata_values: defaultdict[str, list[str]] = defaultdict(list)

        for field_name, aliases in TOP_LEVEL_ALIASES.items():
            values = _values_for_aliases(product, aliases)
            field_values[field_name].extend(values)
            if field_name not in {"title", "description"}:
                metadata_values[field_name].extend(values)

        details = product.get("details")
        for path, value in _detail_leaves(details):
            key = _normalise_key(" ".join(path))
            attribute_text = " ".join((*path, value)).strip()
            if attribute_text:
                field_values["attributes"].append(attribute_text)
            metadata_values["attributes"].append(value)
            if key:
                metadata_values[key].append(value)
            key_terms = set(tokenize(key))
            for field_name, classifiers in DETAIL_CLASSIFIERS.items():
                if key_terms & classifiers:
                    field_values[field_name].append(value)
                    metadata_values[field_name].append(value)

        fields = {name: " ".join(field_values[name]) for name in INDEX_FIELDS}
        metadata = {
            name: tuple(values)
            for name, values in sorted(metadata_values.items())
            if values
        }
        available_prices = parse_prices(product.get("price"))
        selected_price = available_prices[0] if available_prices else None
        return CatalogDocument(
            parent_asin=parent_asin,
            fields=fields,
            metadata=metadata,
            available_prices=available_prices,
            price=selected_price,
        )


@dataclass(frozen=True)
class LexicalRetrievalConfig:
    field_weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_FIELD_WEIGHTS))
    soft_match_boosts: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SOFT_MATCH_BOOSTS)
    )
    candidate_pool_size: int = 1_000
    max_query_terms: int = 40

    def __post_init__(self) -> None:
        if isinstance(self.candidate_pool_size, bool) or self.candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive")
        if isinstance(self.max_query_terms, bool) or self.max_query_terms <= 0:
            raise ValueError("max_query_terms must be positive")
        unknown_fields = set(self.field_weights) - set(INDEX_FIELDS)
        if unknown_fields:
            raise ValueError(f"unknown field weights: {sorted(unknown_fields)}")
        try:
            invalid_field_weight = any(
                not math.isfinite(float(weight)) or float(weight) < 0
                for weight in self.field_weights.values()
            )
            invalid_soft_boost = any(
                not math.isfinite(float(boost)) or float(boost) < 0
                for boost in self.soft_match_boosts.values()
            )
        except (TypeError, ValueError) as error:
            raise ValueError("weights and boosts must be finite numbers") from error
        if invalid_field_weight:
            raise ValueError("field weights must be finite and non-negative")
        if invalid_soft_boost:
            raise ValueError("soft boosts must be finite and non-negative")


@dataclass(frozen=True)
class _CatalogRecord:
    parent_asin: str
    metadata_tokens: dict[str, frozenset[str]]
    evidence_tokens: dict[str, frozenset[str]]
    price: float | None


class LexicalRetriever:
    """A deterministic, field-weighted BM25 catalog retriever."""

    def __init__(
        self,
        products: Iterable[object],
        config: LexicalRetrievalConfig | None = None,
        document_builder: CatalogDocumentBuilder | None = None,
    ) -> None:
        self.config = config or LexicalRetrievalConfig()
        self.document_builder = document_builder or CatalogDocumentBuilder()
        self.connection = sqlite3.connect(":memory:")
        self._records: dict[str, _CatalogRecord] = {}
        self._build_index(products)

    @classmethod
    def from_jsonl(
        cls,
        catalog_path: str | Path,
        config: LexicalRetrievalConfig | None = None,
        document_builder: CatalogDocumentBuilder | None = None,
    ) -> LexicalRetriever:
        path = Path(catalog_path)

        def products() -> Iterator[object]:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"invalid catalog JSON on line {line_number}") from error

        return cls(products(), config=config, document_builder=document_builder)

    @property
    def catalog_ids(self) -> frozenset[str]:
        return frozenset(self._records)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> LexicalRetriever:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _build_index(self, products: Iterable[object]) -> None:
        columns = ", ".join(INDEX_FIELDS)
        self.connection.execute(
            f"CREATE VIRTUAL TABLE products USING fts5("
            f"parent_asin UNINDEXED, {columns}, "
            "tokenize='unicode61 remove_diacritics 2', "
            f"detail=full, columnsize=1)"
        )
        placeholders = ", ".join("?" for _ in range(len(INDEX_FIELDS) + 1))
        insert_sql = f"INSERT INTO products VALUES ({placeholders})"
        batch: list[tuple[str, ...]] = []

        for product in products:
            document = self.document_builder.build(product)
            if document is None or document.parent_asin in self._records:
                continue
            self._records[document.parent_asin] = self._make_record(document)
            batch.append(
                (document.parent_asin, *(document.fields[name] for name in INDEX_FIELDS))
            )
            if len(batch) >= 1_000:
                self.connection.executemany(insert_sql, batch)
                batch.clear()
        if batch:
            self.connection.executemany(insert_sql, batch)
        self.connection.commit()

    @staticmethod
    def _make_record(document: CatalogDocument) -> _CatalogRecord:
        metadata_tokens = {
            name: frozenset(
                token
                for value in values
                for token in tokenize(value)
            )
            for name, values in document.metadata.items()
            if name not in {"attributes", "brand", "features"}
        }
        field_tokens = {
            name: frozenset(tokenize(text))
            for name, text in document.fields.items()
        }
        evidence_tokens = {
            "preference": frozenset(
                token
                for evidence_field in PREFERENCE_EVIDENCE_FIELDS
                for token in field_tokens.get(evidence_field, ())
            ),
            "category": metadata_tokens.get("category", frozenset()),
            "features": field_tokens["features"],
            "brand": field_tokens["brand"],
            "attributes": field_tokens["attributes"],
        }
        return _CatalogRecord(
            parent_asin=document.parent_asin,
            metadata_tokens=metadata_tokens,
            evidence_tokens=evidence_tokens,
            price=document.price,
        )

    def retrieve(self, query: SearchQuery, top_n: int = 200) -> list[RetrievalResult]:
        if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n <= 0:
            return []
        if not self._is_compatible_query(query):
            return []

        price_bounds = self._price_bounds(query)
        if price_bounds is None:
            return []

        constraints = self._constraints(query)
        for _, constraint in constraints:
            if constraint.strength == "hard" and not tokenize(constraint.value):
                return []
        if query.price is not None and query.price.strength not in {"hard", "soft"}:
            return []
        if query.exclusions is not None and not isinstance(query.exclusions, Mapping):
            return []

        terms = self._query_terms(query, constraints)
        if not terms:
            return []

        expression = " OR ".join(f'"{term}"' for term in terms)
        weights = [0.0, *(float(self.config.field_weights.get(name, 0.0)) for name in INDEX_FIELDS)]
        weight_sql = ", ".join("?" for _ in weights)
        sql = (
            "SELECT parent_asin, bm25(products, "
            f"{weight_sql}) AS lexical_score "
            "FROM products WHERE products MATCH ? "
            "ORDER BY lexical_score ASC, parent_asin ASC"
        )
        parameters = (*weights, expression)

        candidates: list[tuple[_CatalogRecord, float, tuple[str, ...], tuple[str, ...]]] = []
        pool_size = self.config.candidate_pool_size
        for parent_asin, raw_score in self.connection.execute(sql, parameters):
            record = self._records[str(parent_asin)]
            hard_matches = self._hard_matches(record, query, constraints, price_bounds)
            if hard_matches is None:
                continue
            soft_score, soft_matches, soft_failures = self._soft_score(
                record, constraints, query, price_bounds
            )
            lexical_score = max(0.0, -float(raw_score))
            candidates.append(
                (
                    record,
                    lexical_score + soft_score,
                    (*hard_matches, *soft_matches),
                    soft_failures,
                )
            )
            if len(candidates) >= pool_size:
                break

        candidates.sort(key=lambda item: (-item[1], item[0].parent_asin))
        return [
            RetrievalResult(
                parent_asin=record.parent_asin,
                score=score,
                rank=rank,
                matched_constraints=matched,
                failed_constraints=failed,
            )
            for rank, (record, score, matched, failed) in enumerate(candidates[:top_n], start=1)
        ]

    @staticmethod
    def _is_compatible_query(query: object) -> bool:
        """Validate the shared contract without requiring module/class identity.

        Issue 1A originally committed equivalent frozen dataclasses in its own
        module. Structural validation keeps that producer interoperable while
        still rejecting malformed query objects deterministically.
        """

        query_fields = (
            "text",
            "category",
            "color",
            "style",
            "material",
            "use_case",
            "price",
            "exclusions",
        )
        if not all(hasattr(query, name) for name in query_fields):
            return False
        if not isinstance(getattr(query, "text"), str):
            return False
        for name in CONSTRAINT_FIELDS:
            constraint = getattr(query, name, None)
            if constraint is None:
                continue
            if not all(
                hasattr(constraint, attribute)
                for attribute in ("value", "strength", "source", "updated_turn")
            ):
                return False
            if constraint.strength not in {"hard", "soft"}:
                return False
            if constraint.source not in {"current_turn", "conversation", "profile"}:
                return False

        price = getattr(query, "price", None)
        if price is not None:
            if not all(
                hasattr(price, attribute)
                for attribute in ("minimum", "maximum", "strength", "source")
            ):
                return False
            if price.strength not in {"hard", "soft"}:
                return False
            if price.source not in {"current_turn", "conversation", "profile"}:
                return False

        exclusions = getattr(query, "exclusions", None)
        return exclusions is None or isinstance(exclusions, Mapping)

    def _query_terms(
        self,
        query: SearchQuery,
        constraints: list[tuple[str, Constraint]],
    ) -> tuple[str, ...]:
        terms: list[str] = list(tokenize(query.text))
        for _, constraint in constraints:
            terms.extend(tokenize(constraint.value))
        return tuple(dict.fromkeys(terms))[: self.config.max_query_terms]

    @staticmethod
    def _constraints(query: SearchQuery) -> list[tuple[str, Constraint]]:
        return [
            (name, constraint)
            for name in CONSTRAINT_FIELDS
            if (constraint := getattr(query, name)) is not None
        ]

    @staticmethod
    def _coerce_bound(value: object) -> tuple[bool, float | None]:
        if value is None:
            return True, None
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            return False, None
        try:
            bound = float(value)
        except (TypeError, ValueError):
            return False, None
        if not math.isfinite(bound) or bound < 0:
            return False, None
        return True, bound

    def _price_bounds(
        self,
        query: SearchQuery,
    ) -> tuple[float | None, float | None] | None:
        if query.price is None:
            return (None, None)
        minimum_valid, minimum = self._coerce_bound(query.price.minimum)
        maximum_valid, maximum = self._coerce_bound(query.price.maximum)
        if not minimum_valid or not maximum_valid:
            return None
        if minimum is not None and maximum is not None and minimum > maximum:
            return None
        return minimum, maximum

    @staticmethod
    def _tokens_match(terms: tuple[str, ...], evidence: frozenset[str]) -> bool:
        return bool(terms) and set(terms).issubset(evidence)

    @staticmethod
    def _evidence_for(record: _CatalogRecord, name: str) -> frozenset[str]:
        if name in PREFERENCE_FIELDS:
            return record.evidence_tokens["preference"]
        return record.evidence_tokens.get(
            name,
            record.metadata_tokens.get(name, frozenset()),
        )

    def _hard_matches(
        self,
        record: _CatalogRecord,
        query: SearchQuery,
        constraints: list[tuple[str, Constraint]],
        price_bounds: tuple[float | None, float | None],
    ) -> tuple[str, ...] | None:
        matched: list[str] = []
        for name, constraint in constraints:
            if constraint.strength != "hard":
                continue
            terms = tokenize(constraint.value)
            if not self._tokens_match(terms, record.metadata_tokens.get(name, frozenset())):
                return None
            matched.append(f"{name}:{constraint.value}")

        if (
            query.price is not None
            and query.price.strength == "hard"
            and price_bounds != (None, None)
        ):
            if not self._price_matches(record.price, price_bounds):
                return None
            matched.append(self._price_label(price_bounds))

        if self._violates_exclusions(record, query.exclusions):
            return None
        return tuple(matched)

    def _soft_score(
        self,
        record: _CatalogRecord,
        constraints: list[tuple[str, Constraint]],
        query: SearchQuery,
        price_bounds: tuple[float | None, float | None],
    ) -> tuple[float, tuple[str, ...], tuple[str, ...]]:
        score = 0.0
        matched: list[str] = []
        failed: list[str] = []
        for name, constraint in constraints:
            if constraint.strength != "soft":
                continue
            label = f"{name}:{constraint.value}"
            evidence = self._evidence_for(record, name)
            if self._tokens_match(tokenize(constraint.value), evidence):
                score += float(self.config.soft_match_boosts.get(name, 0.0))
                matched.append(label)
            else:
                failed.append(label)

        if (
            query.price is not None
            and query.price.strength == "soft"
            and price_bounds != (None, None)
        ):
            label = self._price_label(price_bounds)
            if self._price_matches(record.price, price_bounds):
                score += float(self.config.soft_match_boosts.get("price", 0.0))
                matched.append(label)
            else:
                failed.append(label)
        return score, tuple(matched), tuple(failed)

    @staticmethod
    def _price_matches(
        price: float | None,
        bounds: tuple[float | None, float | None],
    ) -> bool:
        if price is None:
            return False
        minimum, maximum = bounds
        return (minimum is None or price >= minimum) and (maximum is None or price <= maximum)

    @staticmethod
    def _price_label(bounds: tuple[float | None, float | None]) -> str:
        minimum, maximum = bounds
        if minimum is not None and maximum is not None:
            return f"price:{minimum:g}-{maximum:g}"
        if minimum is not None:
            return f"price:>={minimum:g}"
        if maximum is not None:
            return f"price:<={maximum:g}"
        return "price:any"

    def _violates_exclusions(
        self,
        record: _CatalogRecord,
        exclusions: dict[str, set[str]] | None,
    ) -> bool:
        if not exclusions:
            return False
        for raw_field, raw_values in sorted(exclusions.items(), key=lambda item: str(item[0])):
            field_name = _normalise_key(raw_field)
            canonical = EXCLUSION_ALIASES.get(field_name, field_name)
            evidence = self._evidence_for(record, canonical)
            if not evidence:
                continue
            values = raw_values if isinstance(raw_values, (set, frozenset, list, tuple)) else (raw_values,)
            for value in sorted(values, key=str):
                if self._tokens_match(tokenize(value), evidence):
                    return True
        return False
