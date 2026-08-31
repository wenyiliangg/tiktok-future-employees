"""Measure bounded pre-override BM25 evidence on non-public shadow targets."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from benchmarks.shadow_clarification_suite import (
    _perturb,
    build_shadow_samples,
    initial_message,
    load_jsonl,
    public_targets,
)
from starter.bm25_anchor import BM25AnchorRetriever
from starter.contextual_retrieval import (
    ContextualRetrievalPolicy,
    rank_contextual_candidates,
)

SEED = 20260830
POLICY = ContextualRetrievalPolicy(
    policy_id="diagnostic.override-history-tail.v1",
    protected_lexical_count=8,
    candidate_count=100,
    state_lexical_weight=0.5,
)


def rank_of(target: str, values: list[str]) -> int | None:
    try:
        return values.index(target) + 1
    except ValueError:
        return None


def comparison_label(before: int | None, after: int | None) -> str:
    if before is None:
        return "gained_hit" if after is not None else "unchanged_miss"
    if after is None:
        return "lost_hit"
    if after < before:
        return "better_rank"
    if after > before:
        return "worse_rank"
    return "unchanged_hit"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public-set", default="data/public_set.jsonl")
    parser.add_argument("--sample-count", type=int, default=256)
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
    retriever = BM25AnchorRetriever(args.catalog)
    catalog_ids = frozenset(products)
    counts: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    reciprocal_rank_deltas: list[float] = []
    protected_prefix_changes = 0
    for sample in samples:
        history_text = initial_message(sample, set())
        current_text = (
            "Actually, ignore my earlier preference. I now need "
            f"{_perturb(sample.new_override_value or '', sample)}."
        )
        current = list(retriever.retrieve(current_text, top_n=100))
        history = list(retriever.retrieve(history_text, top_n=100))
        baseline_ids = [item.parent_asin for item in current[:10]]
        candidate = rank_contextual_candidates(
            current,
            history,
            [],
            catalog_ids,
            set(),
            POLICY,
            limit=10,
        )
        candidate_ids = [item.parent_asin for item in candidate]
        before = rank_of(sample.target, baseline_ids)
        after = rank_of(sample.target, candidate_ids)
        label = comparison_label(before, after)
        counts[label] += 1
        protected_prefix_changes += baseline_ids[:8] != candidate_ids[:8]
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
                "target_in_current_top_100": any(
                    item.parent_asin == sample.target for item in current
                ),
                "target_in_history_top_100": any(
                    item.parent_asin == sample.target for item in history
                ),
            }
        )

    result = {
        "schema_version": 1,
        "experiment_id": "H3-override-history-tail-v1",
        "seed": SEED,
        "sample_count": len(samples),
        "public_target_overlap": len({sample.target for sample in samples} & excluded),
        "configuration": {
            "protected_current_prefix": POLICY.protected_lexical_count,
            "history_tail_weight": POLICY.state_lexical_weight,
            "candidate_count": POLICY.candidate_count,
        },
        "counts": dict(sorted(counts.items())),
        "protected_prefix_changes": protected_prefix_changes,
        "mean_reciprocal_rank_delta": round(
            statistics.fmean(reciprocal_rank_deltas), 9
        ),
        "target_recovery": {
            "current_top_100": sum(
                bool(row["target_in_current_top_100"]) for row in rows
            ),
            "history_top_100": sum(
                bool(row["target_in_history_top_100"]) for row in rows
            ),
        },
        "sessions": rows,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
