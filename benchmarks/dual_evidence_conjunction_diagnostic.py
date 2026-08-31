"""Evaluate strict dual-evidence conjunction over non-public H3 Top-10 pools."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

from benchmarks.override_history_diagnostic import comparison_label, rank_of
from benchmarks.shadow_clarification_suite import (
    _perturb,
    build_shadow_samples,
    initial_message,
    load_jsonl,
    public_targets,
)
from starter.bm25_anchor import BM25AnchorRetriever
from starter.contextual_retrieval import policy_by_id, rank_contextual_candidates

SEED = 20260830
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


def tokens(value: object) -> frozenset[str]:
    return frozenset(
        token.lower()
        for token in TOKEN_RE.findall(" ".join(_flatten(value)))
        if len(token) >= 3 and token.lower() not in STOPWORDS
    )


def product_tokens(product: Mapping[str, object]) -> frozenset[str]:
    return tokens([product.get(field) for field in VISIBLE_FIELDS])


def promote_unique_conjunction(
    ranked_ids: list[str],
    scores: Mapping[str, tuple[float, float]],
    *,
    minimum_side_support: float,
    minimum_margin: float,
) -> tuple[list[str], str | None]:
    qualified = [
        (min(left, right), left + right, -rank, asin)
        for rank, asin in enumerate(ranked_ids)
        for left, right in (scores.get(asin, (0.0, 0.0)),)
        if left >= minimum_side_support and right >= minimum_side_support
    ]
    qualified.sort(reverse=True)
    if not qualified:
        return list(ranked_ids), None
    best = qualified[0]
    runner_up = qualified[1] if len(qualified) > 1 else None
    if runner_up is not None and best[0] - runner_up[0] < minimum_margin:
        return list(ranked_ids), None
    selected = best[-1]
    if ranked_ids and selected == ranked_ids[0]:
        return list(ranked_ids), selected
    return [selected, *(asin for asin in ranked_ids if asin != selected)], selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public-set", default="data/public_set.jsonl")
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--max-document-frequency", type=int, default=500)
    parser.add_argument("--minimum-side-support", type=float, default=3.5)
    parser.add_argument("--minimum-margin", type=float, default=0.5)
    parser.add_argument("--output")
    args = parser.parse_args()

    products = {
        str(row["parent_asin"]): row
        for row in load_jsonl(args.catalog)
        if isinstance(row.get("parent_asin"), str)
    }
    excluded = public_targets(load_jsonl(args.public_set))
    samples = [
        sample
        for sample in build_shadow_samples(products, excluded, args.sample_count)
        if sample.scenario_type == "intent_override"
    ]
    token_sets = {asin: product_tokens(product) for asin, product in products.items()}
    document_frequency: Counter[str] = Counter(
        token for values in token_sets.values() for token in values
    )
    catalog_size = len(products)
    retriever = BM25AnchorRetriever(args.catalog)
    policy = policy_by_id("contextual.override-history-tail.v1")
    catalog_ids = frozenset(products)
    counts: Counter[str] = Counter()
    selected_count = 0
    selected_target_count = 0
    reciprocal_rank_deltas: list[float] = []
    rows: list[dict[str, object]] = []
    for sample in samples:
        history_text = initial_message(sample, set())
        current_text = (
            "Actually, ignore my earlier preference. I now need "
            f"{_perturb(sample.new_override_value or '', sample)}."
        )
        current_results = list(retriever.retrieve(current_text, top_n=100))
        history_results = list(retriever.retrieve(history_text, top_n=100))
        h3 = rank_contextual_candidates(
            current_results,
            history_results,
            [],
            catalog_ids,
            set(),
            policy,
            limit=10,
        )
        baseline_ids = [item.parent_asin for item in h3]
        current_tokens = tokens(current_text)
        history_tokens = tokens(history_text)
        shared = current_tokens & history_tokens
        current_distinctive = {
            token
            for token in current_tokens - shared
            if document_frequency[token] <= args.max_document_frequency
        }
        history_distinctive = {
            token
            for token in history_tokens - shared
            if document_frequency[token] <= args.max_document_frequency
        }
        scores: dict[str, tuple[float, float]] = {}
        for asin in baseline_ids:
            values = token_sets[asin]
            current_support = sum(
                math.log((catalog_size + 1) / (document_frequency[token] + 1))
                for token in current_distinctive & values
            )
            history_support = sum(
                math.log((catalog_size + 1) / (document_frequency[token] + 1))
                for token in history_distinctive & values
            )
            scores[asin] = (current_support, history_support)
        candidate_ids, selected = promote_unique_conjunction(
            baseline_ids,
            scores,
            minimum_side_support=args.minimum_side_support,
            minimum_margin=args.minimum_margin,
        )
        before = rank_of(sample.target, baseline_ids)
        after = rank_of(sample.target, candidate_ids)
        label = comparison_label(before, after)
        counts[label] += 1
        selected_count += selected is not None
        selected_target_count += selected == sample.target
        reciprocal_rank_deltas.append(
            (0.0 if after is None else 1.0 / after)
            - (0.0 if before is None else 1.0 / before)
        )
        rows.append(
            {
                "sample_id": sample.sample_id,
                "comparison": label,
                "baseline_rank": before,
                "candidate_rank": after,
                "selected_as_target": selected == sample.target,
                "promotion_applied": selected is not None,
            }
        )

    result = {
        "schema_version": 1,
        "experiment_id": "H4-dual-evidence-conjunction-v1",
        "seed": SEED,
        "sample_count": len(samples),
        "public_target_overlap": len({sample.target for sample in samples} & excluded),
        "configuration": {
            "max_document_frequency": args.max_document_frequency,
            "minimum_side_support": args.minimum_side_support,
            "minimum_margin": args.minimum_margin,
            "shared_evidence_tokens_removed": True,
            "candidate_scope": "H3 Top 10",
        },
        "counts": dict(sorted(counts.items())),
        "promotions_applied": selected_count,
        "promotions_selecting_target": selected_target_count,
        "mean_reciprocal_rank_delta": round(
            statistics.fmean(reciprocal_rank_deltas), 9
        ),
        "sessions": rows,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
