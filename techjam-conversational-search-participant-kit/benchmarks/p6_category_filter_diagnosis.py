"""Observational category/filter residual diagnosis for the frozen P5 shadow policy."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

from starter.agent import Agent
from starter.clarification_policies import clarification_policy_by_id

from . import p4_residual_diagnosis as residual
from .shadow_clarification_suite import (
    SEED,
    _agent,
    build_shadow_samples,
    evaluate_shadow,
    load_jsonl,
    metric_summary,
    public_targets,
)

P5_CONTEXTUAL_POLICY = "contextual.category-evidence.v1"
P5_CLARIFICATION_POLICY = "clarification.category-evidence-utility-buying.v1"
EXPECTED_P5_TRANSCRIPT_SHA256 = (
    "33f51cc6f6eb7eff627c116ff919b02a7bbb50fe6b24ddaef7efc9a00e693e92"
)


def _p5_agent(catalog: Path, dense_cache: Path) -> Agent:
    policy = clarification_policy_by_id(P5_CLARIFICATION_POLICY)
    return _agent(
        catalog,
        dense_cache,
        policy.clarification,
        policy.controller,
        P5_CONTEXTUAL_POLICY,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public-set", default="data/public_set.jsonl")
    parser.add_argument(
        "--dense-cache", default="data/.dense-retrieval/catalog-minilm.npz"
    )
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--output")
    args = parser.parse_args()

    catalog = Path(args.catalog)
    dense_cache = Path(args.dense_cache)
    products = {
        str(row["parent_asin"]): row
        for row in load_jsonl(catalog)
        if isinstance(row.get("parent_asin"), str)
    }
    excluded = public_targets(load_jsonl(args.public_set))
    samples = build_shadow_samples(products, excluded, args.sample_count)
    targets = {sample.target for sample in samples}

    baseline_agent = _p5_agent(catalog, dense_cache)
    baseline = evaluate_shadow(baseline_agent, samples, frozenset(products))
    del baseline_agent
    gc.collect()

    traced_agent = _p5_agent(catalog, dense_cache)
    sessions, correctness, trace_hash = residual._replay_with_traces(
        traced_agent, samples, products
    )
    original_factory = residual._p2_agent
    residual._p2_agent = _p5_agent
    try:
        diagnosis = residual._diagnose(
            sessions, samples, products, catalog, dense_cache
        )
    finally:
        residual._p2_agent = original_factory

    traced_metrics = metric_summary(sessions)
    baseline_outcomes = [
        {
            key: row[key]
            for key in (
                "sample_id",
                "scenario_type",
                "hit",
                "first_hit_turn",
                "best_rank",
                "reciprocal_rank",
            )
        }
        for row in sessions
    ]
    category_c = diagnosis["categories"]["C"]
    assignments = {
        str(row["sample_id"]): row for row in diagnosis["session_assignments"]
    }
    samples_by_id = {sample.sample_id: sample for sample in samples}
    residual_details: list[dict[str, object]] = []
    for session in sessions:
        sample_id = str(session["sample_id"])
        if sample_id not in assignments:
            continue
        assignment = assignments[sample_id]
        sample = samples_by_id[sample_id]
        traces = session["traces"]
        categories = products[sample.target].get("categories")
        if categories in (None, "", []):
            metadata_status = "missing"
        elif not isinstance(categories, (str, list, tuple)):
            metadata_status = "malformed"
        elif any(
            isinstance(trace.get("target_contradictions"), int)
            and trace["target_contradictions"] > 0
            for trace in traces
        ):
            metadata_status = "contradictory_to_structured_runtime_evidence"
        else:
            metadata_status = "explicit"
        category = str(assignment["category"])
        first_problem = next(
            (
                trace
                for trace in traces
                if (
                    trace["pool_positions"]["lexical"] is not None
                    or trace["pool_positions"]["fused"] is not None
                )
                and (
                    trace["pool_positions"]["reranked"] is None
                    or (
                        isinstance(trace.get("target_contradictions"), int)
                        and trace["target_contradictions"] > 0
                    )
                )
            ),
            None,
        )
        if category == "C" and first_problem is not None:
            if (
                isinstance(first_problem.get("target_contradictions"), int)
                and first_problem["target_contradictions"] > 0
            ):
                cause = "structured contradiction penalty"
            elif first_problem["pool_positions"]["filtered"] is not None:
                cause = "bounded category-evidence ordering dropped the lexical target before the top-50 analysis pool"
            else:
                cause = "candidate eligibility removal after bounded lexical or fused retrieval"
        else:
            cause = {
                "A": "disclosed structured evidence missing or stale in active state",
                "B": "target absent from all bounded generation pools",
                "D": "disclosed-constraint coverage underweighted below weaker candidates",
                "E": "coverage tie resolved by existing deterministic prior",
                "F": "insufficient disclosed dialogue information",
                "G": "override evidence delayed after the override turn",
                "H": "no supported primary cause",
            }.get(category, "not applicable")
        category_scores = [
            float(trace["target_component_scores"].get("category", 0.0))
            for trace in traces
        ]
        residual_details.append(
            {
                "sample_id": sample_id,
                "scenario_type": session["scenario_type"],
                "assigned_category": category,
                "exact_filter_or_penalty": cause,
                "catalog_metadata_status": metadata_status,
                "category_confidence": (
                    "high_explicit_match"
                    if any(score == 1.0 for score in category_scores)
                    else "uncertain_or_no_explicit_match"
                ),
                "raw_current_message_lexical_support": any(
                    trace["pool_positions"]["lexical"] is not None for trace in traces
                ),
                "observed": assignment["observed"],
                "oracle_gain_contribution_diagnostic_only": assignment[
                    "oracle_gain_contribution"
                ],
                "conservative_realizable_gain_contribution": assignment[
                    "conservative_gain_contribution"
                ],
                "stage_trace": [
                    {
                        "turn": trace["turn"],
                        "route": trace["observable_route"],
                        "lexical": trace["pool_positions"]["lexical"],
                        "fused": trace["pool_positions"]["fused"],
                        "filtered": trace["pool_positions"]["filtered"],
                        "reranked": trace["pool_positions"]["reranked"],
                        "final": trace["pool_positions"]["published"],
                        "target_contradictions": trace["target_contradictions"],
                    }
                    for trace in traces
                ],
            }
        )
    del traced_agent
    gc.collect()
    continuation_gate = {
        "affected_sessions_at_least_5": category_c["session_count"] >= 5,
        "conservative_gain_at_least_0_020": (
            category_c["conservative_realizable_technical_score_gain"] >= 0.020
        ),
    }
    continuation_gate["threshold_met"] = any(continuation_gate.values())
    result = {
        "schema_version": 1,
        "diagnosis_id": "P6A-p5-category-filter-residual-diagnosis-v1",
        "predeclaration_commit": "9229aa68c3696eb02ecf51e521ef3ea5bb993548",
        "seed": SEED,
        "inputs": {
            "sample_count": len(samples),
            "public_target_overlap": len(targets & excluded),
            "target_selection_sha256": hashlib.sha256(
                "\n".join(sorted(targets)).encode()
            ).hexdigest(),
            "p5_contextual_policy": P5_CONTEXTUAL_POLICY,
            "p5_clarification_policy": P5_CLARIFICATION_POLICY,
            "p3_exposure_enabled": False,
        },
        "trace_integrity": {
            "expected_p5_transcript_sha256": EXPECTED_P5_TRANSCRIPT_SHA256,
            "baseline_transcript_sha256": baseline["normalized_transcript_sha256"],
            "observed_trace_transcript_sha256": trace_hash,
            "baseline_matches_checkpoint": baseline["normalized_transcript_sha256"]
            == EXPECTED_P5_TRANSCRIPT_SHA256,
            "observer_matches_baseline": (
                trace_hash == baseline["normalized_transcript_sha256"]
                and baseline_outcomes == baseline["sessions"]
                and traced_metrics["technical_score"] == baseline["technical_score"]
            ),
        },
        "p5_shadow_metrics": traced_metrics,
        "correctness_counters": {
            name: correctness.get(name, 0)
            for name in (
                "response_exceptions",
                "invalid_responses",
                "invalid_ask_attributes",
                "invalid_asins",
                "duplicate_recommendations",
            )
        },
        "diagnosis": diagnosis,
        "residual_details": residual_details,
        "category_c_continuation_gate": continuation_gate,
        "p6_scope_decision": {
            "category_c_selected": continuation_gate["threshold_met"],
            "out_of_scope_categories_not_selected": True,
            "decision": (
                "continue_to_one_p6_correction"
                if continuation_gate["threshold_met"]
                else "reject_p6_retain_p5_package_immediately"
            ),
        },
        "official_runs_consumed": 0,
        "official_runs_remaining": 2,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
