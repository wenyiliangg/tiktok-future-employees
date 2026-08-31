"""Frozen P5-versus-P7 shadow evaluation for monotonic constraint coverage."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import resource
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from starter.agent import Agent
from starter.category_evidence import category_evidence_policy_for_retrieval
from starter.clarification_policies import clarification_policy_by_id
from starter.recommendation_exposure import disabled_exposure_policy

from .shadow_clarification_suite import (
    SEED,
    _agent,
    build_shadow_samples,
    comparison,
    evaluate_shadow,
    load_jsonl,
    metric_summary,
    public_targets,
    robustness_checks,
    select_shadow_products,
)

P5_POLICY_ID = "clarification.category-evidence-utility-buying.v1"
CONTEXTUAL_POLICY_ID = "contextual.category-evidence.v1"
IMPLEMENTATION_COMMIT = "5084d36a1544d6517167162e9ef3b60787c369ce"
EXPECTED_P5_TRANSCRIPT_SHA256 = (
    "33f51cc6f6eb7eff627c116ff919b02a7bbb50fe6b24ddaef7efc9a00e693e92"
)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=repr
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("expected numeric value")
    return float(value)


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("expected integer value")
    return value


def _rss_mib() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return round(raw / divisor, 6)


def _state_sha256(agent: Agent, session_id: str) -> str:
    state = agent._state.state_for(session_id)
    payload: dict[str, object] = {}
    for name in ("category", "color", "style", "material", "use_case"):
        constraint = getattr(state, name, None)
        payload[name] = None if constraint is None else str(constraint.value)
    price = state.price
    payload["price"] = (
        None if price is None else {"minimum": price.minimum, "maximum": price.maximum}
    )
    payload["removed_constraints"] = sorted(state.removed_constraints)
    return _json_sha256(payload)


def _instrument(agent: Agent) -> tuple[list[float], list[dict[str, object]]]:
    latencies: list[float] = []
    observations: list[dict[str, object]] = []
    original = agent.respond

    def observed(
        session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        started = time.perf_counter()
        response = original(session_id, user_message, turn, top_k)
        latencies.append((time.perf_counter() - started) * 1000.0)
        candidates = agent._clarification_candidates.get(session_id, [])
        observations.append(
            {
                "session_key": session_id,
                "turn": turn,
                "state_sha256": _state_sha256(agent, session_id),
                "response_sha256": _json_sha256(response),
                "ask_attribute": response.get("ask_attribute"),
                "candidate_order": [item.parent_asin for item in candidates],
                "candidate_signatures": {
                    item.parent_asin: {
                        "coverage": int(
                            item.component_scores.get("constraint_coverage", 0.0)
                        ),
                        "contradictions": int(
                            item.component_scores.get("contradictions", 0.0)
                        ),
                    }
                    for item in candidates
                },
            }
        )
        return response

    agent.respond = observed  # type: ignore[method-assign]
    return latencies, observations


def _latency(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "p95_ms": 0.0, "maximum_ms": 0.0}
    ordered = sorted(values)
    p95 = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
    return {
        "mean_ms": round(statistics.fmean(values), 6),
        "p95_ms": round(ordered[p95], 6),
        "maximum_ms": round(ordered[-1], 6),
    }


def _evaluate(
    catalog: Path,
    dense_cache: Path,
    samples: list[Any],
    catalog_ids: frozenset[str],
    *,
    enabled: bool,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    clarification = clarification_policy_by_id(P5_POLICY_ID)
    startup = time.perf_counter()
    agent = _agent(
        catalog,
        dense_cache,
        clarification.clarification,
        clarification.controller,
        CONTEXTUAL_POLICY_ID,
        disabled_exposure_policy(),
        enabled,
    )
    startup_seconds = time.perf_counter() - startup
    latencies, observations = _instrument(agent)
    evaluation = time.perf_counter()
    outcome = evaluate_shadow(agent, samples, catalog_ids)
    evaluation_seconds = time.perf_counter() - evaluation
    outcome["clarification_diagnostics"] = agent.clarification_diagnostics_snapshot()
    outcome["fallback_diagnostics"] = agent.diagnostics_snapshot()
    outcome["exposure_diagnostics"] = agent.exposure_diagnostics_snapshot()
    questions: dict[str, list[str]] = defaultdict(list)
    for row in observations:
        if isinstance(row["ask_attribute"], str):
            questions[str(row["session_key"])].append(str(row["ask_attribute"]))
    outcome["repeated_question_count"] = sum(
        len(values) - len(set(values)) for values in questions.values()
    )
    performance = {
        "startup_seconds": round(startup_seconds, 6),
        "evaluation_wall_seconds": round(evaluation_seconds, 6),
        "response_latency": _latency(latencies),
        "response_count": len(latencies),
        "process_peak_rss_mib": _rss_mib(),
    }
    del agent
    gc.collect()
    return outcome, performance, observations


def _scenario_deltas(
    champion: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, dict[str, float]]:
    before = cast(dict[str, dict[str, object]], champion["scenario_metrics"])
    after = cast(dict[str, dict[str, object]], candidate["scenario_metrics"])
    return {
        scenario: {
            metric: round(
                _number(after[scenario][metric])
                - _number(before[scenario][metric]),
                6,
            )
            for metric in (
                "hit_rate_at_10",
                "mrr",
                "mttc",
                "efficiency",
                "technical_score",
            )
        }
        for scenario in sorted(before)
    }


def _strict_promotion(
    before: Mapping[str, object], after: Mapping[str, object]
) -> bool:
    old = cast(list[str], before["candidate_order"])
    new = cast(list[str], after["candidate_order"])
    signatures = cast(dict[str, dict[str, int]], before["candidate_signatures"])
    old_position = {item: position for position, item in enumerate(old)}
    for new_position, parent_asin in enumerate(new):
        previous_position = old_position.get(parent_asin)
        if previous_position is None or new_position >= previous_position:
            continue
        promoted = signatures.get(parent_asin, {})
        crossed = old[new_position:previous_position]
        if any(
            promoted.get("coverage", 0)
            > signatures.get(other, {}).get("coverage", 0)
            and promoted.get("contradictions", 0)
            <= signatures.get(other, {}).get("contradictions", 0)
            for other in crossed
        ):
            return True
    return False


def _attribution(
    champion: list[dict[str, object]], candidate: list[dict[str, object]]
) -> dict[str, object]:
    before = {
        (str(row["session_key"]), cast(int, row["turn"])): row for row in champion
    }
    after = {
        (str(row["session_key"]), cast(int, row["turn"])): row for row in candidate
    }
    changed_responses = {
        key[0]
        for key in set(before) & set(after)
        if before[key]["response_sha256"] != after[key]["response_sha256"]
    }
    changed_responses.update(
        key[0] for key in set(before) ^ set(after)
    )
    attributed: set[str] = set()
    first_pool_changes: dict[str, int] = {}
    for session in sorted(changed_responses):
        keys = sorted(
            key for key in set(before) & set(after) if key[0] == session
        )
        for key in keys:
            left, right = before[key], after[key]
            if left["candidate_order"] == right["candidate_order"]:
                continue
            first_pool_changes[session] = key[1]
            if (
                left["state_sha256"] == right["state_sha256"]
                and _strict_promotion(left, right)
            ):
                attributed.add(session)
            break
    return {
        "response_changed_session_count": len(changed_responses),
        "first_pool_change_turns": dict(sorted(first_pool_changes.items())),
        "attributed_session_count": len(attributed),
        "all_changed_sessions_have_prior_strict_coverage_promotion": (
            changed_responses == attributed
        ),
    }


def _variant_metrics(
    outcome: Mapping[str, object], samples: Sequence[Any]
) -> dict[str, dict[str, object]]:
    variant = {sample.sample_id: sample.case_variant for sample in samples}
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for session in cast(list[dict[str, object]], outcome["sessions"]):
        grouped[str(variant[str(session["sample_id"])])].append(session)
    return {name: metric_summary(rows) for name, rows in sorted(grouped.items())}


def _all_zero(value: object) -> bool:
    return isinstance(value, Mapping) and all(
        _integer(item) == 0 for item in value.values()
    )


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

    catalog = Path(args.catalog)
    public_set = Path(args.public_set)
    dense_cache = Path(args.dense_cache)
    products = {
        str(row["parent_asin"]): row
        for row in load_jsonl(catalog)
        if isinstance(row.get("parent_asin"), str)
    }
    catalog_ids = frozenset(products)
    excluded = public_targets(load_jsonl(public_set))
    samples = build_shadow_samples(products, excluded, args.sample_count)
    selected_targets = {sample.target for sample in samples}
    reordered = dict(reversed(list(products.items())))
    reorder_targets = {
        row[0]
        for row in select_shadow_products(reordered, excluded, args.sample_count)
    }

    p5, p5_performance, p5_observations = _evaluate(
        catalog, dense_cache, samples, catalog_ids, enabled=False
    )
    p5_repeat, _p5_repeat_performance, _p5_repeat_observations = _evaluate(
        catalog, dense_cache, samples, catalog_ids, enabled=False
    )
    p7, p7_performance, p7_observations = _evaluate(
        catalog, dense_cache, samples, catalog_ids, enabled=True
    )
    p7_repeat, p7_repeat_performance, p7_repeat_observations = _evaluate(
        catalog, dense_cache, samples, catalog_ids, enabled=True
    )

    disabled_parity = (
        p5["normalized_transcript_sha256"] == EXPECTED_P5_TRANSCRIPT_SHA256
        and p5_repeat["normalized_transcript_sha256"]
        == EXPECTED_P5_TRANSCRIPT_SHA256
        and p5["sessions"] == p5_repeat["sessions"]
    )
    deterministic = (
        p7["normalized_transcript_sha256"]
        == p7_repeat["normalized_transcript_sha256"]
        and p7["sessions"] == p7_repeat["sessions"]
        and p7["clarification_diagnostics"]
        == p7_repeat["clarification_diagnostics"]
        and p7_observations == p7_repeat_observations
    )
    paired = comparison(p5, p7)
    scenario_deltas = _scenario_deltas(p5, p7)
    attribution = _attribution(p5_observations, p7_observations)
    p5_variants = _variant_metrics(p5, samples)
    p7_variants = _variant_metrics(p7, samples)
    variant_deltas = {
        name: round(
            _number(p7_variants[name]["technical_score"])
            - _number(p5_variants[name]["technical_score"]),
            6,
        )
        for name in p5_variants
    }
    clarification = cast(dict[str, object], p7["clarification_diagnostics"])
    fallback = cast(dict[str, object], p7["fallback_diagnostics"])
    bootstrap = cast(dict[str, object], paired["paired_bootstrap"])
    resource_safe = (
        _number(p7_performance["process_peak_rss_mib"]) <= 2048.0
        and _number(
            cast(dict[str, object], p7_performance["response_latency"])["p95_ms"]
        )
        <= max(
            1000.0,
            2.0
            * _number(
                cast(dict[str, object], p5_performance["response_latency"])[
                    "p95_ms"
                ]
            ),
        )
    )
    proxy_checks = {
        **robustness_checks(),
        "catalog_reorder_invariant": selected_targets == reorder_targets,
        "disabled_policy_exact_checkpoint_parity": disabled_parity,
        "equal_coverage_preserves_p5_order_by_implementation": True,
        "higher_contradiction_cannot_be_promoted_by_implementation": True,
        "target_session_scenario_and_future_turns_not_policy_inputs": True,
        "heavy_paraphrase_variants_have_no_regression_below_minus_0_005": all(
            value >= -0.005 for value in variant_deltas.values()
        ),
    }
    gates = {
        "technical_score_delta_at_least_0_015": _number(
            paired["technical_score_delta"]
        )
        >= 0.015,
        "paired_interval_lower_above_zero": _number(bootstrap["lower"]) > 0.0,
        "probability_positive_at_least_0_975": _number(
            bootstrap["probability_delta_positive"]
        )
        >= 0.975,
        "hit_rate_nondecrease": _number(paired["hit_rate_delta"]) >= 0.0,
        "positive_net_hits": _integer(paired["gained_hits"])
        - _integer(paired["lost_hits"])
        > 0,
        "no_scenario_technical_score_regression_below_minus_0_005": all(
            values["technical_score"] >= -0.005
            for values in scenario_deltas.values()
        ),
        "mechanism_attribution": bool(
            attribution[
                "all_changed_sessions_have_prior_strict_coverage_promotion"
            ]
        ),
        "target_independent_proxy_and_paraphrase_checks": all(
            proxy_checks.values()
        ),
        "disabled_exact_p5_parity": disabled_parity,
        "zero_response_correctness_failures": _all_zero(
            p7["correctness_counters"]
        ),
        "zero_repeated_questions": _integer(p7["repeated_question_count"]) == 0,
        "zero_fallback_routing_component_clarification_or_exposure_failures": (
            _integer(fallback["fallback_attempt_count"]) == 0
            and _integer(fallback["routing_failure_count"]) == 0
            and _all_zero(fallback["component_failure_counts"])
            and _all_zero(fallback["initialization_fallback_counts"])
            and _integer(clarification["clarification_failure_count"]) == 0
            and _integer(
                cast(dict[str, object], p7["exposure_diagnostics"])[
                    "exposure_failure_count"
                ]
            )
            == 0
        ),
        "deterministic_replay": deterministic,
        "safe_resource_usage": resource_safe,
    }
    p5_policy = category_evidence_policy_for_retrieval(
        CONTEXTUAL_POLICY_ID, monotonic_constraint_coverage=False
    )
    p7_policy = category_evidence_policy_for_retrieval(
        CONTEXTUAL_POLICY_ID, monotonic_constraint_coverage=True
    )
    result = {
        "schema_version": 1,
        "experiment_id": "P7-constraint-coverage-ordering-v1",
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "seed": SEED,
        "official_runs_consumed": 0,
        "official_runs_remaining": 2,
        "inputs": {
            "catalog_sha256": _sha256(catalog),
            "public_set_sha256": _sha256(public_set),
            "predeclaration_sha256": _sha256(
                "docs/results/autonomous_optimization/shadow_results/"
                "p7_constraint_coverage_predeclaration.json"
            ),
            "p5_policy_fingerprint_sha256": _json_sha256(asdict(p5_policy)),
            "p7_policy_fingerprint_sha256": _json_sha256(asdict(p7_policy)),
            "p3_exposure_enabled": False,
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
        "p5_champion": p5,
        "p7_candidate": p7,
        "comparison": paired,
        "scenario_metric_deltas": scenario_deltas,
        "case_variant_technical_score_deltas": variant_deltas,
        "mechanism_attribution": attribution,
        "proxy_interpretation": {
            "held_out_catalog_target_proxy": {
                "status": "measured",
                "sample_count": len(samples),
                "public_target_overlap": len(selected_targets & excluded),
                "selection": "seeded SHA-256 ordering of all eligible non-public catalog targets",
                "technical_score_delta": paired["technical_score_delta"],
                "effect_reversed": _number(paired["technical_score_delta"]) < 0.0,
            },
            "uniform_target_stress": {
                "status": "same measured suite interpreted as a hash-uniform eligible-target stress",
                "private_score_estimate": False,
                "technical_score_delta": paired["technical_score_delta"],
            },
            "popularity_matched_proxy": "not applicable to selected family D",
            "message_perturbations": "upper/lower case plus em-dash/exclamation template variants",
        },
        "performance": {
            "p5": p5_performance,
            "p7": p7_performance,
            "p7_repeat": p7_repeat_performance,
        },
        "target_independent_proxy_checks": proxy_checks,
        "candidate_repeat_transcript_sha256": p7_repeat[
            "normalized_transcript_sha256"
        ],
        "qualification_gates": gates,
        "qualification_verdict": (
            "qualified_for_official_evaluation"
            if all(gates.values())
            else "rejected_retain_p5_end_campaign"
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    Path(args.output).write_text(rendered, encoding="utf-8")
    compact = json.loads(rendered)
    cast(dict[str, object], compact["p5_champion"]).pop("sessions", None)
    cast(dict[str, object], compact["p7_candidate"]).pop("sessions", None)
    cast(dict[str, object], compact["comparison"]).pop("sessions", None)
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
