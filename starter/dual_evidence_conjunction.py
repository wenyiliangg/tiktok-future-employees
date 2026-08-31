"""Strict current/history conjunction over an existing recommendation list."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from .lexical_retriever import CatalogDocument

TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?", re.IGNORECASE)
STOPWORDS = frozenset(
    {
        "actually",
        "and",
        "are",
        "comparing",
        "earlier",
        "favor",
        "for",
        "from",
        "have",
        "ignore",
        "into",
        "looking",
        "need",
        "now",
        "preference",
        "some",
        "that",
        "the",
        "this",
        "use",
        "what",
        "with",
    }
)
VISIBLE_FIELDS = ("title", "features", "details", "categories", "store", "description")
T = TypeVar("T")


class CatalogView(Protocol):
    def get(self, parent_asin: str) -> CatalogDocument | None: ...


@dataclass(frozen=True, slots=True)
class DualEvidencePolicy:
    policy_id: str
    retrieval_policy_id: str
    enabled: bool = False
    max_document_frequency: int = 1000
    minimum_side_support: float = 2.5
    minimum_margin: float = 0.25

    def __post_init__(self) -> None:
        if not self.policy_id.strip() or not self.retrieval_policy_id.strip():
            raise ValueError("policy ids must not be empty")
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean")
        if self.max_document_frequency < 1:
            raise ValueError("max_document_frequency must be positive")
        for name in ("minimum_side_support", "minimum_margin"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def fingerprint_sha256(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def dual_evidence_policy_for_retrieval(retrieval_policy_id: str) -> DualEvidencePolicy:
    if retrieval_policy_id == "contextual.override-history-conjunction.v1":
        return DualEvidencePolicy(
            policy_id="dual-evidence.unique-conjunction.v1",
            retrieval_policy_id=retrieval_policy_id,
            enabled=True,
        )
    return DualEvidencePolicy(
        policy_id="dual-evidence.disabled.v1",
        retrieval_policy_id=retrieval_policy_id,
    )


def _flatten(value: object) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            yield str(key)
            yield from _flatten(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _flatten(item)
    elif value not in (None, ""):
        yield str(value)


def evidence_tokens(value: object) -> frozenset[str]:
    return frozenset(
        token.lower()
        for token in TOKEN_RE.findall(" ".join(_flatten(value)))
        if len(token) >= 3 and token.lower() not in STOPWORDS
    )


def visible_product_tokens(product: Mapping[str, object]) -> frozenset[str]:
    return evidence_tokens([product.get(field) for field in VISIBLE_FIELDS])


def _document_tokens(document: CatalogDocument) -> frozenset[str]:
    return evidence_tokens([document.fields, document.metadata])


def promote_unique_conjunction(
    ranked: Sequence[T],
    identifiers: Sequence[str],
    scores: Mapping[str, tuple[float, float]],
    policy: DualEvidencePolicy,
) -> list[T]:
    qualified = [
        (min(left, right), left + right, -rank, asin)
        for rank, asin in enumerate(identifiers)
        for left, right in (scores.get(asin, (0.0, 0.0)),)
        if left >= policy.minimum_side_support and right >= policy.minimum_side_support
    ]
    qualified.sort(reverse=True)
    if not qualified:
        return list(ranked)
    best = qualified[0]
    if len(qualified) > 1 and best[0] - qualified[1][0] < policy.minimum_margin:
        return list(ranked)
    selected = best[-1]
    by_id = dict(zip(identifiers, ranked))
    return [
        by_id[selected],
        *(item for asin, item in zip(identifiers, ranked) if asin != selected),
    ]


class DualEvidenceConjunctionRanker:
    def __init__(
        self,
        document_frequency: Mapping[str, int],
        catalog_size: int,
        policy: DualEvidencePolicy,
    ) -> None:
        self.document_frequency = dict(document_frequency)
        self.catalog_size = catalog_size
        self.policy = policy

    @classmethod
    def from_jsonl(
        cls, catalog_path: str | Path, policy: DualEvidencePolicy
    ) -> DualEvidenceConjunctionRanker:
        frequency: Counter[str] = Counter()
        catalog_size = 0
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                if not isinstance(product, Mapping):
                    continue
                catalog_size += 1
                frequency.update(visible_product_tokens(product))
        return cls(frequency, catalog_size, policy)

    def rerank(
        self,
        candidates: Sequence[T],
        identifiers: Sequence[str],
        current_text: str,
        history_text: str,
        catalog: CatalogView,
    ) -> list[T]:
        if not self.policy.enabled or len(candidates) != len(identifiers):
            return list(candidates)
        current = evidence_tokens(current_text)
        history = evidence_tokens(history_text)
        shared = current & history
        current_distinctive = {
            token
            for token in current - shared
            if self.document_frequency.get(token, 0)
            <= self.policy.max_document_frequency
        }
        history_distinctive = {
            token
            for token in history - shared
            if self.document_frequency.get(token, 0)
            <= self.policy.max_document_frequency
        }
        scores: dict[str, tuple[float, float]] = {}
        for asin in identifiers:
            document = catalog.get(asin)
            if document is None:
                continue
            values = _document_tokens(document)
            current_support = sum(
                math.log(
                    (self.catalog_size + 1)
                    / (self.document_frequency.get(token, 0) + 1)
                )
                for token in current_distinctive & values
            )
            history_support = sum(
                math.log(
                    (self.catalog_size + 1)
                    / (self.document_frequency.get(token, 0) + 1)
                )
                for token in history_distinctive & values
            )
            scores[asin] = (current_support, history_support)
        return promote_unique_conjunction(candidates, identifiers, scores, self.policy)
