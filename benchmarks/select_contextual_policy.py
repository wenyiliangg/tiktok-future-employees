"""Select a contextual retrieval policy under strict BM25-retention gates."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent
from starter.contextual_retrieval import (
    ContextualRetrievalPolicy,
    contextual_policy_candidates,
)
from starter.hybrid_retrieval import HybridRetrievalConfig, RetrievalMode

METRIC_KEYS = (
    "sample_count",
    "hit_rate_at_10",
    "mrr",
    "mttc",
    "efficiency",
    "recommended_technical_score",
)


def _metrics(result: dict) -> dict[str, object]:
    return {key: result[key] for key in METRIC_KEYS}


def assess_policy(baseline: dict, challenger: dict) -> dict[str, object]:
    """Apply promotion gates without depending on benchmark runtime details."""

    baseline_hits = {
        str(session["sample_id"]): session
        for session in baseline["sessions"]
        if session["hit"]
    }
    challenger_hits = {
        str(session["sample_id"]): session
        for session in challenger["sessions"]
        if session["hit"]
    }
    lost = sorted(set(baseline_hits) - set(challenger_hits))
    gained = sorted(set(challenger_hits) - set(baseline_hits))
    regressions: dict[str, dict[str, object]] = {}
    for sample_id in sorted(set(baseline_hits) & set(challenger_hits)):
        before = baseline_hits[sample_id]
        after = challenger_hits[sample_id]
        before_turn = int(before["first_hit_turn"])
        after_turn = int(after["first_hit_turn"])
        before_rank = int(before["best_rank"])
        after_rank = int(after["best_rank"])
        if after_turn > before_turn or (
            after_turn == before_turn and after_rank > before_rank
        ):
            regressions[sample_id] = {
                "baseline": {"turn": before_turn, "rank": before_rank},
                "challenger": {"turn": after_turn, "rank": after_rank},
            }

    gates = {
        "zero_response_exceptions": (
            challenger["evaluation_diagnostics"]["response_exception_count"] == 0
        ),
        "retains_all_bm25_successes": not lost,
        "no_bm25_hit_turn_or_rank_regression": not regressions,
        "gains_at_least_one_session": bool(gained),
        "technical_score_improves": (
            challenger["recommended_technical_score"]
            > baseline["recommended_technical_score"]
        ),
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "lost_session_ids": lost,
        "gained_session_ids": gained,
        "baseline_hit_regressions": regressions,
    }


def _run(
    catalog: Path,
    dense_cache: Path,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    *,
    mode: RetrievalMode,
    policy: ContextualRetrievalPolicy | None = None,
) -> tuple[dict, float]:
    started = time.perf_counter()
    agent = Agent(
        catalog,
        config=HybridRetrievalConfig(mode=mode),
        dense_cache_path=dense_cache,
        contextual_policy=policy,
    )
    result = evaluate(agent, samples, catalog_ids, categories, products)
    return result, time.perf_counter() - started


def select_policy(
    baseline: dict,
    evaluated: Iterable[tuple[ContextualRetrievalPolicy, dict, float]],
) -> dict[str, object]:
    reports: list[dict[str, Any]] = []
    passing: list[dict[str, Any]] = []
    for policy, result, elapsed_seconds in evaluated:
        assessment = assess_policy(baseline, result)
        report = {
            "policy": {
                "policy_id": policy.policy_id,
                "protected_lexical_count": policy.protected_lexical_count,
                "candidate_count": policy.candidate_count,
                "state_lexical_weight": policy.state_lexical_weight,
                "dense_weight": policy.dense_weight,
                "dense_routes": list(policy.dense_routes),
                "rrf_k": policy.rrf_k,
            },
            "metrics": _metrics(result),
            "elapsed_seconds": round(elapsed_seconds, 6),
            **assessment,
        }
        reports.append(report)
        if report["passed"]:
            passing.append(report)

    passing.sort(
        key=lambda report: (
            -float(report["metrics"]["recommended_technical_score"]),
            str(report["policy"]["policy_id"]),
        )
    )
    selected = passing[0]["policy"]["policy_id"] if passing else None
    return {
        "schema_version": 1,
        "baseline_mode": RetrievalMode.BM25.value,
        "baseline_metrics": _metrics(baseline),
        "promotion_rule": (
            "zero response exceptions; retain every BM25 hit; no later/worse "
            "BM25 hit; gain at least one session; improve TechnicalScore"
        ),
        "challengers": reports,
        "selected_policy_id": selected,
        "promotion_passed": selected is not None,
        "default_mode_after_selection": (
            RetrievalMode.CONTEXTUAL.value if selected else RetrievalMode.BM25.value
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument(
        "--dense-cache",
        type=Path,
        default=Path("data/.dense-retrieval/catalog-minilm.npz"),
    )
    parser.add_argument(
        "--policy",
        action="append",
        dest="policy_ids",
        help="Policy id to evaluate; repeat to limit the frozen candidate set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/results/recovery/contextual_policy_selection.json"),
    )
    args = parser.parse_args()
    candidates = list(contextual_policy_candidates())
    if args.policy_ids:
        requested = set(args.policy_ids)
        candidates = [policy for policy in candidates if policy.policy_id in requested]
        missing = sorted(requested - {policy.policy_id for policy in candidates})
        if missing:
            raise SystemExit(f"unknown policy id(s): {', '.join(missing)}")

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    baseline, baseline_seconds = _run(
        args.catalog,
        args.dense_cache,
        samples,
        catalog_ids,
        categories,
        products,
        mode=RetrievalMode.BM25,
    )
    evaluated: list[tuple[ContextualRetrievalPolicy, dict, float]] = []
    for policy in candidates:
        result, elapsed_seconds = _run(
            args.catalog,
            args.dense_cache,
            samples,
            catalog_ids,
            categories,
            products,
            mode=RetrievalMode.CONTEXTUAL,
            policy=policy,
        )
        evaluated.append((policy, result, elapsed_seconds))
        print(
            json.dumps(
                {
                    "policy_id": policy.policy_id,
                    "metrics": _metrics(result),
                    **assess_policy(baseline, result),
                },
                sort_keys=True,
            )
        )

    report = select_policy(baseline, evaluated)
    report["baseline_elapsed_seconds"] = round(baseline_seconds, 6)
    report["dataset"] = str(args.dataset)
    report["catalog"] = str(args.catalog)
    report["dense_cache"] = str(args.dense_cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_policy_id": report["selected_policy_id"],
                "promotion_passed": report["promotion_passed"],
                "default_mode_after_selection": report["default_mode_after_selection"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
