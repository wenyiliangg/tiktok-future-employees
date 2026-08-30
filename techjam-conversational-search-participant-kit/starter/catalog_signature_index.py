"""Deterministic unique catalog-phrase lookup over participant-visible fields."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?", re.IGNORECASE)
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
        "you",
        "your",
    }
)


@dataclass(frozen=True, slots=True)
class CatalogSignaturePolicy:
    policy_id: str
    retrieval_policy_id: str
    compatible_retrieval_policy_id: str
    enabled: bool = False
    min_tokens: int = 3
    max_tokens: int = 24
    min_characters: int = 16
    rank_head_count: int = 1
    fields: tuple[str, ...] = ("title", "features", "details", "categories")

    def __post_init__(self) -> None:
        if not self.policy_id.strip() or not self.retrieval_policy_id.strip():
            raise ValueError("signature and retrieval policy ids must not be empty")
        if not self.compatible_retrieval_policy_id.strip():
            raise ValueError("compatible_retrieval_policy_id must not be empty")
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean")
        if not 2 <= self.min_tokens <= self.max_tokens:
            raise ValueError("signature token limits are invalid")
        if self.min_characters < 1:
            raise ValueError("min_characters must be positive")
        if self.rank_head_count != 1:
            raise ValueError(
                "the isolated signature policy supports one head candidate"
            )
        if not self.fields or any(
            field not in {"title", "features", "details", "categories"}
            for field in self.fields
        ):
            raise ValueError("fields contains an unsupported catalog field")

    @property
    def fingerprint_sha256(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def clarification_is_compatible(self, required_policy_id: str) -> bool:
        return required_policy_id in {
            self.retrieval_policy_id,
            self.compatible_retrieval_policy_id,
        }


def catalog_signature_policy_for_retrieval(
    retrieval_policy_id: str,
) -> CatalogSignaturePolicy:
    if retrieval_policy_id == "contextual.signature-head.v1":
        return CatalogSignaturePolicy(
            policy_id="catalog-signature.unique-head.v1",
            retrieval_policy_id=retrieval_policy_id,
            compatible_retrieval_policy_id="contextual.feedback-memory.v1",
            enabled=True,
        )
    return CatalogSignaturePolicy(
        policy_id="catalog-signature.disabled.v1",
        retrieval_policy_id=retrieval_policy_id,
        compatible_retrieval_policy_id=retrieval_policy_id,
    )


@dataclass(frozen=True, slots=True)
class CatalogSignatureMatch:
    parent_asin: str
    normalized_signature: str
    token_count: int


def normalize_tokens(value: object) -> tuple[str, ...]:
    return tuple(
        token
        for token in TOKEN_RE.findall(str(value or "").lower().replace("’", "'"))
        if token not in STOPWORDS
    )


def _phrases(product: Mapping[str, object], fields: tuple[str, ...]) -> tuple[str, ...]:
    values: list[str] = []
    if "title" in fields and isinstance(product.get("title"), str):
        values.append(str(product["title"]))
    for field in ("features", "categories"):
        raw = product.get(field)
        if field in fields and isinstance(raw, list):
            values.extend(str(value) for value in raw if value not in (None, ""))
    details = product.get("details")
    if "details" in fields and isinstance(details, Mapping):
        values.extend(
            f"{key}: {value}"
            for key, value in details.items()
            if value not in (None, "", [])
        )
    return tuple(dict.fromkeys(values))


class CatalogSignatureIndex:
    """Keep only phrase prefixes that have exactly one catalog owner."""

    def __init__(
        self,
        unique_owners: Mapping[str, str],
        token_counts: Mapping[str, int],
        policy: CatalogSignaturePolicy,
    ) -> None:
        self._unique_owners = dict(unique_owners)
        self._token_counts = dict(token_counts)
        self.policy = policy
        self._lengths = tuple(sorted(set(token_counts.values()), reverse=True))

    @classmethod
    def from_jsonl(
        cls, path: str | Path, policy: CatalogSignaturePolicy
    ) -> CatalogSignatureIndex:
        owners: dict[str, str | None] = {}
        token_counts: dict[str, int] = {}
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                parent_asin = product.get("parent_asin")
                if not isinstance(parent_asin, str) or not parent_asin:
                    continue
                for phrase in _phrases(product, policy.fields):
                    phrase_tokens = normalize_tokens(phrase)[: policy.max_tokens]
                    normalized = " ".join(phrase_tokens)
                    if (
                        len(phrase_tokens) < policy.min_tokens
                        or len(normalized) < policy.min_characters
                    ):
                        continue
                    previous = owners.get(normalized)
                    if previous is None and normalized not in owners:
                        owners[normalized] = parent_asin
                        token_counts[normalized] = len(phrase_tokens)
                    elif previous != parent_asin:
                        owners[normalized] = None
                        token_counts.pop(normalized, None)
        unique = {
            signature: owner for signature, owner in owners.items() if owner is not None
        }
        return cls(unique, token_counts, policy)

    @property
    def unique_signature_count(self) -> int:
        return len(self._unique_owners)

    def match(self, user_message: object) -> CatalogSignatureMatch | None:
        query_tokens = normalize_tokens(user_message)
        if len(query_tokens) < self.policy.min_tokens:
            return None
        matches: list[CatalogSignatureMatch] = []
        for length in self._lengths:
            if length > len(query_tokens):
                continue
            for start in range(len(query_tokens) - length + 1):
                normalized = " ".join(query_tokens[start : start + length])
                owner = self._unique_owners.get(normalized)
                if owner is not None:
                    matches.append(CatalogSignatureMatch(owner, normalized, length))
            if matches:
                owners = {match.parent_asin for match in matches}
                return matches[0] if len(owners) == 1 else None
        return None
