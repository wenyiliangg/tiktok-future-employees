"""Measure non-public catalog-owner recovery from normalized field signatures."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from benchmarks.shadow_clarification_suite import (
    build_shadow_samples,
    load_jsonl,
    public_targets,
)

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
SIGNATURE_TOKEN_LIMIT = 24
RARE_TOKEN_MAX_DOCUMENTS = 500


def tokens(value: object) -> tuple[str, ...]:
    return tuple(
        token
        for token in TOKEN_RE.findall(str(value or "").lower().replace("’", "'"))
        if token not in STOPWORDS
    )


def signature_key(value: object) -> str:
    return " ".join(tokens(value)[:SIGNATURE_TOKEN_LIMIT])


def catalog_phrases(product: Mapping[str, object]) -> tuple[str, ...]:
    result: list[str] = []
    title = product.get("title")
    if isinstance(title, str):
        result.append(title)
    for field in ("features", "description", "categories"):
        values = product.get(field)
        if isinstance(values, list):
            result.extend(str(value) for value in values if value not in (None, ""))
    details = product.get("details")
    if isinstance(details, Mapping):
        result.extend(
            f"{key}: {value}"
            for key, value in details.items()
            if value not in (None, "", [])
        )
    return tuple(dict.fromkeys(value for value in result if len(tokens(value)) >= 2))


class SignatureDiagnosticIndex:
    """Benchmark-only exact-prefix and rare-token owner index."""

    def __init__(self, products: Mapping[str, Mapping[str, object]]) -> None:
        phrase_owners: dict[str, set[str]] = defaultdict(set)
        token_owners: dict[str, set[str]] = defaultdict(set)
        for parent_asin, product in products.items():
            product_tokens: set[str] = set()
            for phrase in catalog_phrases(product):
                key = signature_key(phrase)
                if len(key) >= 8:
                    phrase_owners[key].add(parent_asin)
                product_tokens.update(tokens(phrase))
            for token in product_tokens:
                token_owners[token].add(parent_asin)
        self.phrase_owners = dict(phrase_owners)
        self.token_owners = dict(token_owners)

    def exact_owners(self, phrase: object) -> frozenset[str]:
        return frozenset(self.phrase_owners.get(signature_key(phrase), ()))

    def rare_token_owners(self, phrase: object) -> frozenset[str]:
        postings = [
            owners
            for token in dict.fromkeys(tokens(phrase))
            if 0
            < len(owners := self.token_owners.get(token, set()))
            <= RARE_TOKEN_MAX_DOCUMENTS
        ]
        if not postings:
            return frozenset()
        postings.sort(key=len)
        result = set(postings[0])
        for owners in postings[1:]:
            narrowed = result & owners
            if narrowed:
                result = narrowed
        return frozenset(result)

    def owners(self, phrase: object) -> frozenset[str]:
        exact = self.exact_owners(phrase)
        return exact or self.rare_token_owners(phrase)


def _bucket(size: int) -> str:
    if size == 0:
        return "no_candidates"
    if size == 1:
        return "unique"
    if size <= 10:
        return "two_to_ten"
    if size <= 100:
        return "eleven_to_hundred"
    return "over_hundred"


def diagnose(
    index: SignatureDiagnosticIndex,
    samples: Iterable[object],
) -> dict[str, object]:
    phrase_buckets: Counter[str] = Counter()
    first_buckets: Counter[str] = Counter()
    two_phrase_buckets: Counter[str] = Counter()
    owner_recovered = Counter()
    candidate_sizes: list[int] = []
    sample_count = 0
    for sample in samples:
        sample_count += 1
        target = str(getattr(sample, "target"))
        constraints = tuple(getattr(sample, "constraints"))[:4]
        owner_sets: list[frozenset[str]] = []
        for constraint in constraints:
            owners = index.owners(constraint)
            owner_sets.append(owners)
            phrase_buckets[_bucket(len(owners))] += 1
            owner_recovered["phrase"] += target in owners
            candidate_sizes.append(len(owners))
        first = owner_sets[0]
        first_buckets[_bucket(len(first))] += 1
        owner_recovered["first_phrase"] += target in first
        combined = set(owner_sets[0])
        for owners in owner_sets[1:2]:
            intersection = combined & owners
            if intersection:
                combined = intersection
        two_phrase_buckets[_bucket(len(combined))] += 1
        owner_recovered["two_phrase"] += target in combined
    return {
        "sample_count": sample_count,
        "phrases_evaluated": sum(phrase_buckets.values()),
        "all_phrase_candidate_buckets": dict(sorted(phrase_buckets.items())),
        "first_phrase_candidate_buckets": dict(sorted(first_buckets.items())),
        "two_phrase_candidate_buckets": dict(sorted(two_phrase_buckets.items())),
        "owner_recovery": dict(sorted(owner_recovered.items())),
        "owner_recovery_rates": {
            "first_phrase": round(owner_recovered["first_phrase"] / sample_count, 6),
            "two_phrase": round(owner_recovered["two_phrase"] / sample_count, 6),
        },
        "candidate_size": {
            "mean": round(statistics.fmean(candidate_sizes), 6),
            "median": statistics.median(candidate_sizes),
            "maximum": max(candidate_sizes),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public-set", default="data/public_set.jsonl")
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    catalog_rows = load_jsonl(catalog_path)
    products = {
        str(row["parent_asin"]): row
        for row in catalog_rows
        if isinstance(row.get("parent_asin"), str)
    }
    excluded = public_targets(load_jsonl(args.public_set))
    samples = build_shadow_samples(products, excluded, args.sample_count)
    index = SignatureDiagnosticIndex(products)
    result = {
        "schema_version": 1,
        "experiment_id": "H2-catalog-signature-index-v1",
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "public_target_overlap": len({sample.target for sample in samples} & excluded),
        "configuration": {
            "signature_token_limit": SIGNATURE_TOKEN_LIMIT,
            "rare_token_max_documents": RARE_TOKEN_MAX_DOCUMENTS,
            "fields": ["title", "features", "description", "categories", "details"],
        },
        "diagnostic": diagnose(index, samples),
    }
    rendered = json.dumps(result, indent=2) + "\n"
    Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
