"""Catalog-derived candidate generation and current-evidence ranking.

This experiment is isolated behind a contextual retrieval policy.  It uses only
participant-visible catalog fields and runtime messages, and keeps every component
separately switchable for ablation.
"""

from __future__ import annotations

import json
import math
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .conversation_state import SearchQuery
from .hybrid_retrieval import Candidate, RankedResult
from .lexical_retriever import CatalogDocument, tokenize

VISIBLE_FIELDS = (
    "title",
    "features",
    "details",
    "description",
    "categories",
    "store",
)
GENERIC_CATEGORIES = frozenset(
    {"clothing", "clothing shoes jewelry", "clothing shoes and jewelry"}
)
DIALOGUE_TOKENS = frozenset(
    {
        "additional",
        "actually",
        "after",
        "ask",
        "earlier",
        "exploring",
        "ignore",
        "judgment",
        "matters",
        "preference",
        "quite",
        "specific",
        "those",
        "what",
        "yet",
    }
)


class CatalogView(Protocol):
    def get(self, parent_asin: str) -> CatalogDocument | None: ...


def _normalise(value: object) -> str:
    return " ".join(tokenize(value))


def _flatten(value: object) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            if item not in (None, "", []):
                yield f"{key} {item}"
                yield from _flatten(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=str) if isinstance(value, (set, frozenset)) else value
        for item in items:
            yield from _flatten(item)
    elif value not in (None, ""):
        yield str(value)


def _visible_text(product: Mapping[str, object]) -> str:
    return " ".join(
        part for field in VISIBLE_FIELDS for part in _flatten(product.get(field))
    )


def _category_parts(value: object) -> tuple[str, ...]:
    raw: list[str] = []
    if isinstance(value, (list, tuple)):
        for item in value:
            # A sequence already represents hierarchy levels.  Commas inside a
            # level are part of its label (for example a marketplace root), not
            # additional hierarchy separators.
            raw.append(str(item))
    elif value not in (None, ""):
        raw.extend(str(value).split(","))
    cleaned = []
    for item in raw:
        name = _normalise(item)
        if name and name not in GENERIC_CATEGORIES:
            cleaned.append(name)
    return tuple(cleaned)


def _category_aliases(value: object) -> tuple[str, tuple[str, ...]]:
    parts = _category_parts(value)
    if not parts:
        return "", ()
    canonical = " ".join(parts[-2:])
    aliases = {canonical, parts[-1]}
    aliases.update(parts)
    if len(parts) >= 2:
        aliases.add(" ".join(parts[-2:]))
    return canonical, tuple(
        sorted(aliases, key=lambda item: (-len(item.split()), item))
    )


def _source_phrases(product: Mapping[str, object]) -> frozenset[str]:
    phrases: set[str] = set()
    for field in ("features", "details"):
        for raw in _flatten(product.get(field)):
            phrase = _normalise(raw)
            if 2 <= len(phrase.split()) <= 40:
                phrases.add(phrase)
    return frozenset(phrases)


@dataclass(frozen=True, slots=True)
class CategoryEvidencePolicy:
    policy_id: str
    retrieval_policy_id: str
    enabled: bool = False
    use_category: bool = True
    use_phrases: bool = True
    use_rare_tokens: bool = True
    use_structured: bool = True
    use_popularity: bool = True
    use_conjunction: bool = True
    monotonic_constraint_coverage: bool = False
    category_candidate_limit: int = 1600
    total_candidate_limit: int = 4000
    weak_evidence_anchor_floor: int = 6
    rare_document_fraction: float = 0.05
    strong_evidence_document_fraction: float = 0.01
    phrase_weight: float = 6.25
    rare_weight: float = 3.75
    structured_weight: float = 2.75
    category_weight: float = 1.25
    anchor_weight: float = 0.90
    popularity_weight: float = 0.60
    history_weight: float = 0.45
    conjunction_weight: float = 1.35
    contradiction_penalty: float = 5.50

    def __post_init__(self) -> None:
        if not self.policy_id.strip() or not self.retrieval_policy_id.strip():
            raise ValueError("policy identifiers must not be empty")
        for name in (
            "use_category",
            "use_phrases",
            "use_rare_tokens",
            "use_structured",
            "use_popularity",
            "use_conjunction",
            "monotonic_constraint_coverage",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if self.category_candidate_limit < 10 or self.total_candidate_limit < 10:
            raise ValueError("candidate limits must be at least ten")
        if self.category_candidate_limit > self.total_candidate_limit:
            raise ValueError("category limit cannot exceed total candidate limit")
        if not 0 <= self.weak_evidence_anchor_floor <= 10:
            raise ValueError("weak_evidence_anchor_floor must be between zero and ten")
        for name in (
            "rare_document_fraction",
            "strong_evidence_document_fraction",
        ):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must be between zero and one")
        if self.strong_evidence_document_fraction > self.rare_document_fraction:
            raise ValueError(
                "strong evidence frequency cannot exceed rare candidate frequency"
            )
        for name in (
            "phrase_weight",
            "rare_weight",
            "structured_weight",
            "category_weight",
            "anchor_weight",
            "popularity_weight",
            "history_weight",
            "conjunction_weight",
            "contradiction_penalty",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


def category_evidence_policy_for_retrieval(
    retrieval_policy_id: str,
    *,
    monotonic_constraint_coverage: bool = False,
) -> CategoryEvidencePolicy:
    if retrieval_policy_id == "contextual.category-evidence.v1":
        return CategoryEvidencePolicy(
            policy_id=(
                "category-evidence.constraint-coverage.v1"
                if monotonic_constraint_coverage
                else "category-evidence.cohesive.v1"
            ),
            retrieval_policy_id=retrieval_policy_id,
            enabled=True,
            monotonic_constraint_coverage=monotonic_constraint_coverage,
        )
    return CategoryEvidencePolicy(
        policy_id="category-evidence.disabled.v1",
        retrieval_policy_id=retrieval_policy_id,
        enabled=False,
    )


@dataclass(slots=True)
class _SessionMessages:
    current: list[str]
    historical: list[str]
    raw: list[str]


class EvidenceMessageStore:
    """Small per-session raw evidence ledger for the experimental ranker."""

    def __init__(self) -> None:
        self._sessions: dict[str, _SessionMessages] = {}

    def reset(self, session_id: str) -> None:
        self._sessions[session_id] = _SessionMessages([], [], [])

    def observe(
        self,
        session_id: str,
        message: str,
        *,
        override: bool,
        non_evidence_reply: bool,
    ) -> None:
        state = self._sessions.setdefault(session_id, _SessionMessages([], [], []))
        text = message.strip()
        state.raw.append(text)
        if not text:
            return
        if override:
            state.historical = list(state.current)
            state.current = [text]
        elif not non_evidence_reply:
            state.current.append(text)

    def current_text(self, session_id: str) -> str:
        state = self._sessions.get(session_id)
        return " ".join(state.current) if state is not None else ""

    def historical_text(self, session_id: str) -> str:
        state = self._sessions.get(session_id)
        return " ".join(state.historical) if state is not None else ""


class CategoryEvidenceIndex:
    """Immutable label-free catalog evidence and deterministic ranking."""

    def __init__(
        self, products: Iterable[object], policy: CategoryEvidencePolicy
    ) -> None:
        self.policy = policy
        self.ids: list[str] = []
        self.id_to_doc: dict[str, int] = {}
        self.doc_category: list[str] = []
        self.popularity: list[float] = []
        self.prices: list[tuple[float, ...]] = []
        self._category_docs: dict[str, array] = defaultdict(lambda: array("I"))
        self._alias_categories: dict[str, set[str]] = defaultdict(set)
        phrase_docs: dict[str, array] = defaultdict(lambda: array("I"))
        token_docs: dict[str, array] = defaultdict(lambda: array("I"))
        raw_popularity: list[float] = []

        for product in products:
            if not isinstance(product, Mapping):
                continue
            raw_id = product.get("parent_asin")
            parent_asin = str(raw_id).strip() if raw_id is not None else ""
            if not parent_asin or parent_asin in self.id_to_doc:
                continue
            doc_id = len(self.ids)
            self.ids.append(parent_asin)
            self.id_to_doc[parent_asin] = doc_id

            category, aliases = _category_aliases(product.get("categories"))
            self.doc_category.append(category)
            if category:
                self._category_docs[category].append(doc_id)
                for alias in aliases:
                    self._alias_categories[alias].add(category)

            for phrase in _source_phrases(product):
                phrase_docs[phrase].append(doc_id)

            for token in frozenset(tokenize(_visible_text(product))):
                token_docs[token].append(doc_id)

            try:
                rating_count = max(0, int(product.get("rating_number") or 0))
            except (TypeError, ValueError):
                rating_count = 0
            try:
                rating = min(5.0, max(0.0, float(product.get("average_rating") or 0.0)))
            except (TypeError, ValueError):
                rating = 0.0
            raw_popularity.append(math.log1p(rating_count) + math.log1p(rating))
            raw_price = product.get("price")
            prices: list[float] = []
            for item in _flatten(raw_price):
                try:
                    value = float(str(item).replace("$", "").replace(",", ""))
                except ValueError:
                    continue
                if math.isfinite(value) and value >= 0:
                    prices.append(value)
            self.prices.append(tuple(sorted(set(prices))))

        self.count = len(self.ids)
        maximum = max(raw_popularity, default=1.0) or 1.0
        self.popularity = [value / maximum for value in raw_popularity]
        self._category_order = {
            category: tuple(
                sorted(docs, key=lambda doc: (-self.popularity[doc], self.ids[doc]))
            )
            for category, docs in self._category_docs.items()
        }
        self._category_percentile = [0.0] * self.count
        for docs in self._category_order.values():
            denominator = max(1, len(docs) - 1)
            for rank, doc_id in enumerate(docs):
                self._category_percentile[doc_id] = 1.0 - (rank / denominator)

        self.phrase_docs = dict(phrase_docs)
        self.phrase_lengths = frozenset(len(phrase.split()) for phrase in phrase_docs)
        maximum_document_frequency = max(
            1, int(self.count * self.policy.rare_document_fraction)
        )
        self.all_postings = dict(token_docs)
        self.rare_postings = {
            token: docs
            for token, docs in token_docs.items()
            if len(docs) <= maximum_document_frequency
        }
        self.token_document_frequency = {
            token: len(docs) for token, docs in token_docs.items()
        }
        self._posting_sets: dict[str, frozenset[int]] = {}
        self._maximum_category_words = max(
            (len(alias.split()) for alias in self._alias_categories), default=1
        )

    @classmethod
    def from_jsonl(
        cls, catalog_path: str | Path, policy: CategoryEvidencePolicy
    ) -> CategoryEvidenceIndex:
        def products() -> Iterable[object]:
            with Path(catalog_path).open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)

        return cls(products(), policy)

    def extract_category(self, text: str) -> str | None:
        words = _normalise(text).split()
        matches: list[tuple[int, int, str]] = []
        for size in range(min(len(words), self._maximum_category_words), 0, -1):
            canonical_matches: list[str] = []
            for start in range(len(words) - size + 1):
                alias = " ".join(words[start : start + size])
                categories = self._alias_categories.get(alias)
                if categories is None:
                    continue
                if alias in self._category_docs:
                    canonical_matches.append(alias)
                for category in categories:
                    matches.append(
                        (size, -len(self._category_docs[category]), category)
                    )
            if canonical_matches:
                return min(
                    canonical_matches,
                    key=lambda category: (len(self._category_docs[category]), category),
                )
            if matches:
                break
        return max(matches)[-1] if matches else None

    def matching_phrases(self, text: str) -> tuple[str, ...]:
        tokens = tokenize(text)
        found: set[str] = set()
        for size in self.phrase_lengths:
            if size > len(tokens):
                continue
            for start in range(len(tokens) - size + 1):
                phrase = " ".join(tokens[start : start + size])
                if phrase in self.phrase_docs:
                    found.add(phrase)
        return tuple(
            sorted(
                found,
                key=lambda phrase: (
                    len(self.phrase_docs[phrase]),
                    -len(phrase),
                    phrase,
                ),
            )
        )

    def _idf(self, document_frequency: int) -> float:
        return math.log((self.count + 1.0) / (document_frequency + 1.0))

    def _token_support(
        self, text: str, excluded: frozenset[str]
    ) -> tuple[dict[int, float], float]:
        support: dict[int, float] = defaultdict(float)
        total = 0.0
        for token in dict.fromkeys(tokenize(text)):
            if token in excluded or token in DIALOGUE_TOKENS:
                continue
            postings = self.rare_postings.get(token)
            if postings is None:
                continue
            weight = self._idf(len(postings))
            total += self._idf(0)
            for doc_id in postings:
                support[doc_id] += weight
        return dict(support), total

    def _phrase_support(self, phrases: Sequence[str]) -> tuple[dict[int, float], float]:
        support: dict[int, float] = defaultdict(float)
        total = 0.0
        for phrase in phrases:
            postings = self.phrase_docs[phrase]
            weight = self._idf(len(postings))
            total += self._idf(0)
            for doc_id in postings:
                support[doc_id] += weight
        return dict(support), total

    @staticmethod
    def _constraint_values(query: SearchQuery) -> tuple[tuple[str, object, str], ...]:
        values: list[tuple[str, object, str]] = []
        for name in ("color", "style", "material", "use_case"):
            item = getattr(query, name, None)
            if item is not None:
                values.append((name, item.value, item.strength))
        return tuple(values)

    def _posting_set(self, token: str) -> frozenset[int]:
        cached = self._posting_sets.get(token)
        if cached is None:
            cached = frozenset(self.all_postings.get(token, ()))
            self._posting_sets[token] = cached
        return cached

    def _matching_docs(self, value: object) -> frozenset[int] | None:
        terms = tuple(dict.fromkeys(tokenize(value)))
        if not terms or any(term not in self.all_postings for term in terms):
            return None
        ordered = sorted((self._posting_set(term) for term in terms), key=len)
        result = set(ordered[0])
        for postings in ordered[1:]:
            result.intersection_update(postings)
            if not result:
                break
        return frozenset(result)

    def _structured_evidence(
        self, query: SearchQuery
    ) -> tuple[tuple[tuple[frozenset[int], bool], ...], tuple[frozenset[int], ...]]:
        constraints: list[tuple[frozenset[int], bool]] = []
        for _name, value, strength in self._constraint_values(query):
            matches = self._matching_docs(value)
            if matches is not None:
                constraints.append((matches, strength == "hard"))
        exclusions: list[frozenset[int]] = []
        for values in (query.exclusions or {}).values():
            for value in values:
                matches = self._matching_docs(value)
                if matches is not None:
                    exclusions.append(matches)
        return tuple(constraints), tuple(exclusions)

    def _structured_support(
        self,
        query: SearchQuery,
        doc_id: int,
        constraints: tuple[tuple[frozenset[int], bool], ...],
        exclusions: tuple[frozenset[int], ...],
    ) -> tuple[float, int, int]:
        checked = len(constraints)
        matches = sum(doc_id in docs for docs, _hard in constraints)
        contradictions = sum(
            hard and doc_id not in docs for docs, hard in constraints
        ) + sum(doc_id in docs for docs in exclusions)
        price = query.price
        if price is not None and (
            price.minimum is not None or price.maximum is not None
        ):
            values = self.prices[doc_id]
            if values:
                checked += 1
                price_match = any(
                    (price.minimum is None or value >= price.minimum)
                    and (price.maximum is None or value <= price.maximum)
                    for value in values
                )
                matches += int(price_match)
                if not price_match and price.strength == "hard":
                    contradictions += 1
        return (matches / checked if checked else 0.0), checked, contradictions

    def rank(
        self,
        *,
        query: SearchQuery,
        current_text: str,
        historical_text: str,
        base_results: Sequence[RankedResult],
        history_results: Sequence[RankedResult],
        catalog: CatalogView,
        known_negative_ids: set[str] | frozenset[str],
        limit: int,
    ) -> list[Candidate]:
        if limit <= 0:
            return []
        category = (
            self.extract_category(current_text) if self.policy.use_category else None
        )
        category_tokens = frozenset(tokenize(category or ""))
        current_phrases = (
            self.matching_phrases(current_text) if self.policy.use_phrases else ()
        )
        history_phrases = (
            self.matching_phrases(historical_text)
            if self.policy.use_phrases and historical_text
            else ()
        )
        phrase_support, phrase_total = self._phrase_support(current_phrases)
        history_phrase_support, history_phrase_total = self._phrase_support(
            history_phrases
        )
        if self.policy.use_rare_tokens:
            rare_support, rare_total = self._token_support(
                current_text, category_tokens
            )
            history_rare_support, history_rare_total = self._token_support(
                historical_text, category_tokens
            )
        else:
            rare_support, rare_total = {}, 0.0
            history_rare_support, history_rare_total = {}, 0.0

        anchor_by_id = {result.parent_asin: result for result in base_results}
        history_by_id = {result.parent_asin: result for result in history_results}
        protected_ids = set(anchor_by_id) | set(history_by_id)
        phrase_ids = set(phrase_support) | set(history_phrase_support)
        rare_order = sorted(
            set(rare_support) | set(history_rare_support),
            key=lambda doc: (
                -(
                    rare_support.get(doc, 0.0)
                    + self.policy.history_weight * history_rare_support.get(doc, 0.0)
                ),
                self.ids[doc],
            ),
        )
        candidate_docs = {
            self.id_to_doc[parent_asin]
            for parent_asin in protected_ids
            if parent_asin in self.id_to_doc
        }
        candidate_docs.update(phrase_ids)
        remaining = max(0, self.policy.total_candidate_limit - len(candidate_docs))
        candidate_docs.update(rare_order[:remaining])
        if category:
            remaining = max(0, self.policy.total_candidate_limit - len(candidate_docs))
            category_docs = self._category_order.get(category, ())[
                : min(self.policy.category_candidate_limit, remaining)
            ]
            candidate_docs.update(category_docs)
        if not candidate_docs:
            candidate_docs.update(
                range(min(self.count, self.policy.category_candidate_limit))
            )

        structured_constraints, structured_exclusions = self._structured_evidence(query)

        ranked: list[tuple[float, int, float, int, int, Candidate]] = []
        for doc_id in candidate_docs:
            parent_asin = self.ids[doc_id]
            if parent_asin in known_negative_ids:
                continue
            anchor = anchor_by_id.get(parent_asin)
            anchor_component = 0.0 if anchor is None else 1.0 / float(anchor.rank)
            current_phrase = (
                phrase_support.get(doc_id, 0.0) / phrase_total if phrase_total else 0.0
            )
            current_rare = (
                rare_support.get(doc_id, 0.0) / rare_total if rare_total else 0.0
            )
            history_phrase = (
                history_phrase_support.get(doc_id, 0.0) / history_phrase_total
                if history_phrase_total
                else 0.0
            )
            history_rare = (
                history_rare_support.get(doc_id, 0.0) / history_rare_total
                if history_rare_total
                else 0.0
            )
            category_match = (
                1.0 if category and self.doc_category[doc_id] == category else 0.0
            )
            if self.policy.use_structured:
                structured, checked, contradictions = self._structured_support(
                    query,
                    doc_id,
                    structured_constraints,
                    structured_exclusions,
                )
            else:
                structured, checked, contradictions = 0.0, 0, 0
            structured_matches = int(round(structured * checked))
            phrase_matches = sum(
                doc_id in self.phrase_docs[phrase] for phrase in current_phrases
            )
            coverage_count = (
                int(category_match) + structured_matches + phrase_matches
            )
            independent = sum(
                (
                    current_phrase > 0.0,
                    current_rare >= 0.20,
                    checked > 0 and structured >= 0.50,
                )
            )
            conjunction = max(0, independent - 1) if self.policy.use_conjunction else 0
            prior = (
                self._category_percentile[doc_id]
                if category and self.doc_category[doc_id] == category
                else self.popularity[doc_id]
            )
            history_component = max(history_phrase, history_rare)
            score = (
                self.policy.phrase_weight * current_phrase
                + self.policy.rare_weight * current_rare
                + self.policy.structured_weight * structured
                + self.policy.category_weight * category_match
                + self.policy.anchor_weight * anchor_component
                + (
                    self.policy.popularity_weight * prior
                    if self.policy.use_popularity
                    else 0.0
                )
                + self.policy.history_weight * history_component
                + self.policy.conjunction_weight * conjunction
                - self.policy.contradiction_penalty * contradictions
            )
            candidate = Candidate(
                parent_asin=parent_asin,
                lexical_score=float(anchor.score) if anchor is not None else 0.0,
                lexical_rank=anchor.rank if anchor is not None else None,
                sources={
                    source
                    for source, present in (
                        ("bm25", anchor is not None),
                        ("category", bool(category_match)),
                        ("phrase", bool(current_phrase)),
                        ("rare_token", bool(current_rare)),
                        ("history", bool(history_component)),
                        ("structured", bool(checked)),
                        ("popularity", self.policy.use_popularity),
                    )
                    if present
                },
                fusion_score=score,
                component_scores={
                    "category": category_match,
                    "phrase": current_phrase,
                    "rare_token": current_rare,
                    "structured": structured,
                    "anchor": anchor_component,
                    "popularity": prior,
                    "history": history_component,
                    "conjunction": float(conjunction),
                    "contradictions": float(contradictions),
                    "constraint_coverage": float(coverage_count),
                },
            )
            anchor_rank = anchor.rank if anchor is not None else 2**31 - 1
            ranked.append(
                (
                    score,
                    anchor_rank,
                    prior,
                    coverage_count,
                    contradictions,
                    candidate,
                )
            )

        ranked.sort(key=lambda item: (-item[0], item[1], -item[2], item[-1].parent_asin))
        if self.policy.monotonic_constraint_coverage:
            # P7 is deliberately bounded to the same list P5 would return.  A
            # stable adjacent move is allowed only for strict extra coverage
            # with no additional explicit contradiction.  This preserves the
            # complete P5 ordering for ties and incomparable pairs.
            boundary = min(limit, len(ranked))
            for right in range(1, boundary):
                position = right
                while position > 0:
                    previous = ranked[position - 1]
                    current = ranked[position]
                    if current[3] <= previous[3] or current[4] > previous[4]:
                        break
                    ranked[position - 1], ranked[position] = current, previous
                    position -= 1
        ordered = [item[-1] for item in ranked]
        price = query.price
        maximum_specific_document_frequency = max(
            1, int(self.count * self.policy.strong_evidence_document_fraction)
        )
        strong_phrase_evidence = bool(
            len(current_phrases) >= 2
            or any(
                len(self.phrase_docs[phrase]) <= maximum_specific_document_frequency
                for phrase in current_phrases
            )
        )
        strong_structured_evidence = bool(
            len(structured_constraints) >= 2
            or any(
                len(documents) <= maximum_specific_document_frequency
                for documents, _hard in structured_constraints
            )
        )
        has_current_specific_evidence = bool(
            strong_phrase_evidence
            or strong_structured_evidence
            or structured_exclusions
            or (
                price is not None
                and (price.minimum is not None or price.maximum is not None)
            )
        )
        if has_current_specific_evidence or self.policy.weak_evidence_anchor_floor <= 0:
            return ordered[:limit]

        evidence_slots = max(0, limit - self.policy.weak_evidence_anchor_floor)
        selected = list(ordered[:evidence_slots])
        selected_ids = {candidate.parent_asin for candidate in selected}
        for result in sorted(
            base_results, key=lambda item: (item.rank, item.parent_asin)
        ):
            if len(selected) >= limit:
                break
            if (
                result.parent_asin in known_negative_ids
                or result.parent_asin in selected_ids
            ):
                continue
            candidate = next(
                (item for item in ordered if item.parent_asin == result.parent_asin),
                None,
            )
            if (
                candidate is not None
                and candidate.component_scores.get("contradictions", 0.0) == 0.0
            ):
                selected.append(candidate)
                selected_ids.add(candidate.parent_asin)
        for candidate in ordered:
            if len(selected) >= limit:
                break
            if candidate.parent_asin not in selected_ids:
                selected.append(candidate)
                selected_ids.add(candidate.parent_asin)
        return selected


def catalog_statistics(index: CategoryEvidenceIndex) -> dict[str, object]:
    """Return label-free diagnostics used by tests and experiment records."""

    bucket_sizes = sorted(len(docs) for docs in index._category_docs.values())
    phrase_sizes = sorted(len(docs) for docs in index.phrase_docs.values())
    phrase_frequencies = Counter(len(docs) for docs in index.phrase_docs.values())

    def percentile(values: Sequence[int], fraction: float) -> int:
        if not values:
            return 0
        position = min(len(values) - 1, int((len(values) - 1) * fraction))
        return values[position]

    return {
        "catalog_size": index.count,
        "category_count": len(index._category_docs),
        "category_median_size": (
            bucket_sizes[len(bucket_sizes) // 2] if bucket_sizes else 0
        ),
        "category_p95_size": percentile(bucket_sizes, 0.95),
        "category_max_size": bucket_sizes[-1] if bucket_sizes else 0,
        "category_alias_count": len(index._alias_categories),
        "ambiguous_category_alias_count": sum(
            len(categories) > 1 for categories in index._alias_categories.values()
        ),
        "phrase_count": len(index.phrase_docs),
        "unique_phrase_count": phrase_frequencies.get(1, 0),
        "phrase_p95_document_frequency": percentile(phrase_sizes, 0.95),
        "rare_token_count": len(index.rare_postings),
        "token_count": len(index.all_postings),
    }


def category_recovery_statistics(
    index: CategoryEvidenceIndex,
    products: Iterable[object],
) -> dict[str, object]:
    """Check taxonomy recovery without using relevance labels or evaluator code."""

    checked = 0
    recovered = 0
    missing = 0
    for product in products:
        if not isinstance(product, Mapping):
            continue
        canonical, _aliases = _category_aliases(product.get("categories"))
        if not canonical:
            missing += 1
            continue
        checked += 1
        observed = index.extract_category(f"show me {canonical} options")
        recovered += int(observed == canonical)
    return {
        "checked_product_count": checked,
        "recovered_product_count": recovered,
        "missing_category_count": missing,
        "recovery_rate": (recovered / checked if checked else 1.0),
    }
