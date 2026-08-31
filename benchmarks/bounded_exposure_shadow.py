"""Fixed-matrix shadow evaluation for predeclared bounded exposure policies."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import cast

from benchmarks.shadow_clarification_suite import (
    SEED,
    _agent,
    build_shadow_samples,
    comparison,
    evaluate_shadow,
    exposure_robustness_checks,
    load_jsonl,
    public_targets,
    robustness_checks,
    select_shadow_products,
)
from starter.clarification_policies import clarification_policy_by_id
from starter.recommendation_exposure import (
    disabled_exposure_policy,
    exposure_policy_by_id,
)

PRIMARY_POLICY_ID = "exposure.constraint-release-cap2.v1"
SANITY_POLICY_IDS = (
    "exposure.constraint-release-cap1.v1",
    "exposure.constraint-release-min3.v1",
)
P2_CONTEXTUAL_POLICY_ID = "contextual.category-evidence.v1"
P2_CLARIFICATION_POLICY_ID = "clarification.category-evidence-utility.v1"


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("expected a numeric metric")
    return float(value)


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("expected an integer counter")
    return value


def _scenario_hit_regressions(
    champion: dict[str, object], candidate: dict[str, object]
) -> dict[str, float]:
    before = cast(dict[str, dict[str, object]], champion["scenario_metrics"])
    after = cast(dict[str, dict[str, object]], candidate["scenario_metrics"])
    return {
        scenario: round(
            _number(after[scenario]["hit_rate_at_10"])
            - _number(before[scenario]["hit_rate_at_10"]),
            6,
        )
        for scenario in sorted(before)
    }


def _all_zero(values: object) -> bool:
    return isinstance(values, dict) and all(
        _integer(value) == 0 for value in values.values()
    )


def qualification_gates(
    champion: dict[str, object],
    candidate: dict[str, object],
    paired: dict[str, object],
    *,
    deterministic: bool,
    disabled_parity: bool,
    robustness: dict[str, bool],
) -> dict[str, bool]:
    bootstrap = cast(dict[str, object], paired["paired_bootstrap"])
    exposure = cast(dict[str, object], candidate["exposure_diagnostics"])
    fallback = cast(dict[str, object], candidate["fallback_diagnostics"])
    scenario_deltas = _scenario_hit_regressions(champion, candidate)
    return {
        "technical_score_delta_at_least_0_005": (
            _number(paired["technical_score_delta"]) >= 0.005
        ),
        "mrr_delta_at_least_0_010": _number(paired["mrr_delta"]) >= 0.01,
        "paired_interval_excludes_zero": _number(bootstrap["lower"]) > 0.0,
        "probability_positive_at_least_0_975": (
            _number(bootstrap["probability_delta_positive"]) >= 0.975
        ),
        "no_lost_p2_hits": _integer(paired["lost_hits"]) == 0,
        "no_overall_hit_rate_regression": (_number(paired["hit_rate_delta"]) >= 0.0),
        "no_scenario_hit_rate_regression": all(
            delta >= 0.0 for delta in scenario_deltas.values()
        ),
        "no_permanently_gated_sessions": (
            _integer(exposure["permanently_gated_session_count"]) == 0
        ),
        "no_policy_zero_result_sessions": (
            _integer(exposure["zero_result_session_count"]) == 0
        ),
        "zero_exposure_failures": (_integer(exposure["exposure_failure_count"]) == 0),
        "zero_response_correctness_failures": _all_zero(
            candidate["correctness_counters"]
        ),
        "zero_runtime_fallbacks": (
            _integer(fallback["fallback_attempt_count"]) == 0
            and _integer(fallback["routing_failure_count"]) == 0
            and _all_zero(fallback["component_failure_counts"])
            and _all_zero(fallback["initialization_fallback_counts"])
        ),
        "deterministic_replay": deterministic,
        "disabled_exact_p2_parity": disabled_parity,
        "target_independent_robustness": all(robustness.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public-set", default="data/public_set.jsonl")
    parser.add_argument(
        "--dense-cache", default="data/.dense-retrieval/catalog-minilm.npz"
    )
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    public_path = Path(args.public_set)
    products = {
        str(row["parent_asin"]): row
        for row in load_jsonl(catalog_path)
        if isinstance(row.get("parent_asin"), str)
    }
    catalog_ids = frozenset(products)
    excluded = public_targets(load_jsonl(public_path))
    samples = build_shadow_samples(products, excluded, args.sample_count)
    selected_targets = {sample.target for sample in samples}
    reordered = dict(reversed(list(products.items())))
    reorder_targets = {
        item[0]
        for item in select_shadow_products(reordered, excluded, args.sample_count)
    }

    p2 = clarification_policy_by_id(P2_CLARIFICATION_POLICY_ID)
    champion_agent = _agent(
        catalog_path,
        Path(args.dense_cache),
        p2.clarification,
        p2.controller,
        P2_CONTEXTUAL_POLICY_ID,
        disabled_exposure_policy(),
    )
    champion = evaluate_shadow(champion_agent, samples, catalog_ids)
    del champion_agent
    gc.collect()

    disabled_agent = _agent(
        catalog_path,
        Path(args.dense_cache),
        p2.clarification,
        p2.controller,
        P2_CONTEXTUAL_POLICY_ID,
        disabled_exposure_policy(),
    )
    disabled_repeat = evaluate_shadow(disabled_agent, samples, catalog_ids)
    del disabled_agent
    gc.collect()
    disabled_parity = (
        champion["normalized_transcript_sha256"]
        == disabled_repeat["normalized_transcript_sha256"]
        and champion["sessions"] == disabled_repeat["sessions"]
    )

    general_robustness = robustness_checks()
    primary_policy = exposure_policy_by_id(PRIMARY_POLICY_ID)
    exposure_robustness = exposure_robustness_checks(primary_policy)
    combined_robustness = {**general_robustness, **exposure_robustness}

    policy_results: dict[str, dict[str, object]] = {}
    for policy_id in (PRIMARY_POLICY_ID, *SANITY_POLICY_IDS):
        policy = exposure_policy_by_id(policy_id)
        candidate_agent = _agent(
            catalog_path,
            Path(args.dense_cache),
            p2.clarification,
            p2.controller,
            P2_CONTEXTUAL_POLICY_ID,
            policy,
        )
        candidate = evaluate_shadow(candidate_agent, samples, catalog_ids)
        del candidate_agent
        gc.collect()

        repeat_agent = _agent(
            catalog_path,
            Path(args.dense_cache),
            p2.clarification,
            p2.controller,
            P2_CONTEXTUAL_POLICY_ID,
            policy,
        )
        repeat = evaluate_shadow(repeat_agent, samples, catalog_ids)
        del repeat_agent
        gc.collect()
        deterministic = (
            candidate["normalized_transcript_sha256"]
            == repeat["normalized_transcript_sha256"]
            and candidate["sessions"] == repeat["sessions"]
            and candidate["exposure_diagnostics"] == repeat["exposure_diagnostics"]
        )
        paired = comparison(champion, candidate)
        policy_results[policy_id] = {
            "promotion_eligible": policy_id == PRIMARY_POLICY_ID,
            "configuration_fingerprint_sha256": policy.fingerprint_sha256,
            "candidate": candidate,
            "comparison": paired,
            "deterministic_replay": deterministic,
            "repeat_normalized_transcript_sha256": repeat[
                "normalized_transcript_sha256"
            ],
            "scenario_hit_rate_deltas": _scenario_hit_regressions(champion, candidate),
        }

    primary = cast(dict[str, object], policy_results[PRIMARY_POLICY_ID])
    gates = qualification_gates(
        champion,
        cast(dict[str, object], primary["candidate"]),
        cast(dict[str, object], primary["comparison"]),
        deterministic=bool(primary["deterministic_replay"]),
        disabled_parity=disabled_parity,
        robustness=combined_robustness,
    )
    result = {
        "schema_version": 1,
        "experiment_id": "P3-bounded-confidence-exposure-v1",
        "seed": SEED,
        "official_runs_consumed": 0,
        "inputs": {
            "catalog_sha256": _sha256(catalog_path),
            "public_set_sha256": _sha256(public_path),
            "predeclaration_sha256": _sha256(
                "docs/results/autonomous_optimization/shadow_results/"
                "p3_exposure_predeclaration.json"
            ),
        },
        "suite": {
            "sample_count": len(samples),
            "scenario_counts": dict(
                sorted(Counter(sample.scenario_type for sample in samples).items())
            ),
            "public_target_overlap": len(selected_targets & excluded),
            "catalog_reorder_invariant": selected_targets == reorder_targets,
            "target_selection_sha256": hashlib.sha256(
                "\n".join(sorted(selected_targets)).encode()
            ).hexdigest(),
        },
        "robustness_checks": combined_robustness,
        "p2_champion": champion,
        "disabled_exact_p2_parity": disabled_parity,
        "disabled_replay_hash": disabled_repeat["normalized_transcript_sha256"],
        "policies": policy_results,
        "primary_qualification_gates": gates,
        "primary_qualification_verdict": (
            "qualified_for_separate_official_authorization"
            if all(gates.values())
            else "rejected_retain_p2"
        ),
        "user_utility_cost": "The primary hides positions 2-10 for at most two low-confidence turns. This reduces immediate choice breadth and can delay a relevant non-rank-1 item until clarification, sufficient normalized constraints, an ineligible route, or the turn-3 release.",
    }
    rendered = json.dumps(result, indent=2) + "\n"
    Path(args.output).write_text(rendered, encoding="utf-8")
    policy_summaries: dict[str, object] = {}
    for policy_id, value in policy_results.items():
        candidate_summary = cast(dict[str, object], value["candidate"])
        policy_summaries[policy_id] = {
            "promotion_eligible": value["promotion_eligible"],
            "metrics": {
                key: candidate_summary[key]
                for key in (
                    "hit_rate_at_10",
                    "mrr",
                    "mttc",
                    "efficiency",
                    "technical_score",
                )
            },
            "scenario_metrics": candidate_summary["scenario_metrics"],
            "comparison": value["comparison"],
            "exposure_diagnostics": candidate_summary["exposure_diagnostics"],
            "deterministic_replay": value["deterministic_replay"],
        }

    summary = {
        "experiment_id": result["experiment_id"],
        "official_runs_consumed": 0,
        "suite": result["suite"],
        "disabled_exact_p2_parity": disabled_parity,
        "robustness_checks": combined_robustness,
        "policy_summaries": policy_summaries,
        "primary_qualification_gates": gates,
        "primary_qualification_verdict": result["primary_qualification_verdict"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
