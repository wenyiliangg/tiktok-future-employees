"""Frozen P2-versus-P4 shadow evaluation for buying-route clarification."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import resource
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from starter.agent import Agent
from starter.clarification_policies import clarification_policy_by_id
from starter.recommendation_exposure import disabled_exposure_policy

from .shadow_clarification_suite import (
    SEED,
    _agent,
    build_shadow_samples,
    comparison,
    evaluate_shadow,
    load_jsonl,
    public_targets,
    robustness_checks,
    select_shadow_products,
)

P2_POLICY_ID = "clarification.category-evidence-utility.v1"
P4_POLICY_ID = "clarification.category-evidence-utility-buying.v1"
CONTEXTUAL_POLICY_ID = "contextual.category-evidence.v1"
IMPLEMENTATION_COMMIT = "accda5bf3e1755730868f8e50cf2c2a33c3ef492"


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    rendered = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=repr
    ).encode()
    return hashlib.sha256(rendered).hexdigest()


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


def _state_payload(agent: Agent, session_id: str) -> dict[str, object]:
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
    return payload


def _instrument(agent: Agent) -> tuple[list[float], list[dict[str, object]]]:
    latencies_ms: list[float] = []
    observations: list[dict[str, object]] = []
    original = agent.respond

    def observed_respond(
        session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        started = time.perf_counter()
        response = original(session_id, user_message, turn, top_k)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        analysis_candidates = agent._clarification_candidates.get(session_id, [])
        published = response.get("recommendations", [])
        observations.append(
            {
                "session_key": session_id,
                "turn": turn,
                "observable_route": agent._contextual_routes.get(
                    session_id, "uncertain"
                ),
                "ask_attribute": response.get("ask_attribute"),
                "analysis_candidate_count": len(analysis_candidates),
                "analysis_candidate_sha256": _json_sha256(
                    [item.parent_asin for item in analysis_candidates]
                ),
                "published_candidate_sha256": _json_sha256(published),
                "state_sha256": _json_sha256(_state_payload(agent, session_id)),
                "response_sha256": _json_sha256(response),
            }
        )
        return response

    agent.respond = observed_respond  # type: ignore[method-assign]
    return latencies_ms, observations


def _latency(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "p95_ms": 0.0, "maximum_ms": 0.0}
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
    return {
        "mean_ms": round(statistics.fmean(values), 6),
        "p95_ms": round(ordered[index], 6),
        "maximum_ms": round(ordered[-1], 6),
    }


def _evaluate_policy(
    policy_id: str,
    catalog: Path,
    dense_cache: Path,
    samples: list[Any],
    catalog_ids: frozenset[str],
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    policy = clarification_policy_by_id(policy_id)
    startup_started = time.perf_counter()
    agent = _agent(
        catalog,
        dense_cache,
        policy.clarification,
        policy.controller,
        CONTEXTUAL_POLICY_ID,
        disabled_exposure_policy(),
    )
    startup_seconds = time.perf_counter() - startup_started
    latencies, observations = _instrument(agent)
    evaluation_started = time.perf_counter()
    outcome = evaluate_shadow(agent, samples, catalog_ids)
    evaluation_seconds = time.perf_counter() - evaluation_started
    clarification = agent.clarification_diagnostics_snapshot()
    fallback = agent.diagnostics_snapshot()
    exposure = agent.exposure_diagnostics_snapshot()
    duplicate_questions = 0
    questions_by_session: dict[str, list[str]] = defaultdict(list)
    for item in observations:
        attribute = item["ask_attribute"]
        if isinstance(attribute, str):
            questions_by_session[str(item["session_key"])].append(attribute)
    for attributes in questions_by_session.values():
        duplicate_questions += len(attributes) - len(set(attributes))
    performance = {
        "startup_seconds": round(startup_seconds, 6),
        "evaluation_wall_seconds": round(evaluation_seconds, 6),
        "response_latency": _latency(latencies),
        "response_count": len(latencies),
        "process_peak_rss_mib": _rss_mib(),
    }
    outcome["clarification_diagnostics"] = clarification
    outcome["fallback_diagnostics"] = fallback
    outcome["exposure_diagnostics"] = exposure
    outcome["repeated_question_count"] = duplicate_questions
    del agent
    gc.collect()
    return outcome, performance, observations


def _observation_delta(
    champion: list[dict[str, object]], candidate: list[dict[str, object]]
) -> dict[str, object]:
    before = {
        (str(item["session_key"]), cast(int, item["turn"])): item for item in champion
    }
    after = {
        (str(item["session_key"]), cast(int, item["turn"])): item for item in candidate
    }
    keys = sorted(set(before) | set(after))
    response_changed = []
    pool_changed = []
    state_changed = []
    question_changed = []
    for key in keys:
        left, right = before.get(key), after.get(key)
        if left is None or right is None:
            response_changed.append(key)
            if right is not None and right.get("ask_attribute") is not None:
                question_changed.append(key)
            continue
        if left["response_sha256"] != right["response_sha256"]:
            response_changed.append(key)
        if (
            left["analysis_candidate_sha256"] != right["analysis_candidate_sha256"]
            or left["published_candidate_sha256"] != right["published_candidate_sha256"]
        ):
            pool_changed.append(key)
        if left["state_sha256"] != right["state_sha256"]:
            state_changed.append(key)
        if left["ask_attribute"] != right["ask_attribute"]:
            question_changed.append(key)
    changed_sessions = {session for session, _turn in response_changed}
    candidate_routes = {
        key: str(value["observable_route"]) for key, value in after.items()
    }
    attribution = all(
        candidate_routes.get(key, "buying") == "buying" for key in response_changed
    )
    return {
        "response_changed_turn_count": len(response_changed),
        "response_changed_session_count": len(changed_sessions),
        "question_changed_turn_count": len(question_changed),
        "analysis_or_published_pool_changed_turn_count": len(pool_changed),
        "state_changed_turn_count": len(state_changed),
        "all_changed_candidate_turns_observable_buying_route": attribution,
    }


def _scenario_deltas(
    champion: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, dict[str, float]]:
    before = cast(dict[str, dict[str, object]], champion["scenario_metrics"])
    after = cast(dict[str, dict[str, object]], candidate["scenario_metrics"])
    return {
        name: {
            metric: round(
                _number(after[name][metric]) - _number(before[name][metric]), 6
            )
            for metric in (
                "hit_rate_at_10",
                "mrr",
                "mttc",
                "efficiency",
                "technical_score",
            )
        }
        for name in sorted(before)
    }


def _all_zero(value: object) -> bool:
    return isinstance(value, Mapping) and all(
        _integer(item) == 0 for item in value.values()
    )


def _policy_robustness() -> dict[str, bool]:
    p2 = clarification_policy_by_id(P2_POLICY_ID)
    p4 = clarification_policy_by_id(P4_POLICY_ID)
    p2_payload = p2.fingerprint_payload()
    p4_payload = p4.fingerprint_payload()
    p2_payload["policy_id"] = P4_POLICY_ID
    p2_payload["clarification"]["eligible_routes"] = (  # type: ignore[index]
        "browsing",
        "boundary",
        "buying",
    )
    non_buying_parity = all(
        p2.clarification.utility_is_eligible(route, count)
        == p4.clarification.utility_is_eligible(route, count)
        for route in ("browsing", "boundary", "uncertain")
        for count in (0, 3, 4, 50)
    )
    return {
        "configuration_diff_only_policy_id_and_buying_route": p2_payload == p4_payload,
        "non_buying_route_eligibility_exact_p2_parity": non_buying_parity,
        "buying_route_subthreshold_is_ineligible": not p4.clarification.utility_is_eligible(
            "buying", 3
        ),
        "buying_route_threshold_is_eligible": p4.clarification.utility_is_eligible(
            "buying", 4
        ),
        "paraphrase_independent_after_route_resolution": (
            p4.clarification.utility_is_eligible("buying", 50)
            == p4.clarification.utility_is_eligible("buying", 50)
        ),
        "target_and_session_identifiers_are_not_policy_inputs": (
            set(p4.clarification.__dataclass_fields__)
            == set(p2.clarification.__dataclass_fields__)
        ),
    }


def _qualification_gates(
    champion: dict[str, object],
    candidate: dict[str, object],
    paired: dict[str, object],
    scenario_deltas: dict[str, dict[str, float]],
    observation_delta: dict[str, object],
    robustness: dict[str, bool],
    *,
    deterministic: bool,
    disabled_parity: bool,
) -> dict[str, bool]:
    bootstrap = cast(dict[str, object], paired["paired_bootstrap"])
    clarification = cast(dict[str, object], candidate["clarification_diagnostics"])
    fallback = cast(dict[str, object], candidate["fallback_diagnostics"])
    selected = ("boundary", "browsing", "intent_override")
    selected_delta = statistics.fmean(
        scenario_deltas[name]["technical_score"] for name in selected
    )
    resolution = cast(dict[str, int], clarification["resolution_counts"])
    buying_questions = cast(
        dict[str, int], clarification["question_counts_by_observable_route"]
    )
    return {
        "technical_score_delta_at_least_0_015": _number(paired["technical_score_delta"])
        >= 0.015,
        "paired_interval_excludes_zero": _number(bootstrap["lower"]) > 0.0,
        "probability_positive_at_least_0_975": _number(
            bootstrap["probability_delta_positive"]
        )
        >= 0.975,
        "selected_scenario_aggregate_delta_at_least_0_015": selected_delta >= 0.015,
        "no_overall_hit_rate_regression": _number(paired["hit_rate_delta"]) >= 0.0,
        "no_lost_p2_hits": _integer(paired["lost_hits"]) == 0,
        "no_scenario_hit_rate_regression": all(
            values["hit_rate_at_10"] >= 0.0 for values in scenario_deltas.values()
        ),
        "nonselected_buying_scenario_regression_within_0_005": scenario_deltas[
            "buying"
        ]["technical_score"]
        >= -0.005,
        "new_buying_route_questions_present": buying_questions.get("buying", 0) > 0,
        "answered_new_questions_present": resolution.get("answered", 0) > 0,
        "zero_repeated_questions": _integer(candidate["repeated_question_count"]) == 0,
        "zero_response_correctness_failures": _all_zero(
            candidate["correctness_counters"]
        ),
        "zero_fallback_routing_or_component_failures": (
            _integer(fallback["fallback_attempt_count"]) == 0
            and _integer(fallback["routing_failure_count"]) == 0
            and _all_zero(fallback["component_failure_counts"])
            and _all_zero(fallback["initialization_fallback_counts"])
            and _integer(clarification["clarification_failure_count"]) == 0
        ),
        "mechanism_attribution_to_buying_route": bool(
            observation_delta["all_changed_candidate_turns_observable_buying_route"]
        ),
        "disabled_exact_p2_parity": disabled_parity,
        "deterministic_replay": deterministic,
        "paraphrase_and_target_independent_robustness": all(robustness.values()),
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
        item[0]
        for item in select_shadow_products(reordered, excluded, args.sample_count)
    }

    champion, champion_performance, champion_observations = _evaluate_policy(
        P2_POLICY_ID, catalog, dense_cache, samples, catalog_ids
    )
    p2_repeat, _p2_repeat_performance, _p2_repeat_observations = _evaluate_policy(
        P2_POLICY_ID, catalog, dense_cache, samples, catalog_ids
    )
    disabled_parity = (
        champion["normalized_transcript_sha256"]
        == p2_repeat["normalized_transcript_sha256"]
        and champion["sessions"] == p2_repeat["sessions"]
    )

    candidate, candidate_performance, candidate_observations = _evaluate_policy(
        P4_POLICY_ID, catalog, dense_cache, samples, catalog_ids
    )
    repeat, repeat_performance, repeat_observations = _evaluate_policy(
        P4_POLICY_ID, catalog, dense_cache, samples, catalog_ids
    )
    deterministic = (
        candidate["normalized_transcript_sha256"]
        == repeat["normalized_transcript_sha256"]
        and candidate["sessions"] == repeat["sessions"]
        and candidate["clarification_diagnostics"]
        == repeat["clarification_diagnostics"]
        and candidate_observations == repeat_observations
    )

    paired = comparison(champion, candidate)
    scenario_deltas = _scenario_deltas(champion, candidate)
    observation_delta = _observation_delta(
        champion_observations, candidate_observations
    )
    robustness = {**robustness_checks(), **_policy_robustness()}
    gates = _qualification_gates(
        champion,
        candidate,
        paired,
        scenario_deltas,
        observation_delta,
        robustness,
        deterministic=deterministic,
        disabled_parity=disabled_parity,
    )
    result = {
        "schema_version": 1,
        "experiment_id": "P4-buying-route-utility-clarification-v1",
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "seed": SEED,
        "official_runs_consumed": 0,
        "official_runs_remaining": 4,
        "inputs": {
            "catalog_sha256": _sha256(catalog),
            "public_set_sha256": _sha256(public_set),
            "predeclaration_sha256": _sha256(
                "docs/results/autonomous_optimization/shadow_results/"
                "p4_buying_route_clarification_predeclaration.json"
            ),
            "clarification_registry_sha256": _sha256(
                "config/clarification_policies.json"
            ),
            "p2_policy_fingerprint_sha256": clarification_policy_by_id(
                P2_POLICY_ID
            ).fingerprint_sha256,
            "p4_policy_fingerprint_sha256": clarification_policy_by_id(
                P4_POLICY_ID
            ).fingerprint_sha256,
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
        "p2_champion": champion,
        "p4_candidate": candidate,
        "comparison": paired,
        "scenario_metric_deltas": scenario_deltas,
        "candidate_pool_and_state_changes": observation_delta,
        "performance": {
            "p2": champion_performance,
            "p4": candidate_performance,
            "p4_repeat": repeat_performance,
        },
        "robustness_checks": robustness,
        "disabled_exact_p2_parity": disabled_parity,
        "disabled_p2_transcript_sha256": p2_repeat["normalized_transcript_sha256"],
        "candidate_deterministic_replay": deterministic,
        "candidate_repeat_transcript_sha256": repeat["normalized_transcript_sha256"],
        "qualification_gates": gates,
        "qualification_verdict": (
            "qualified_for_separate_official_authorization"
            if all(gates.values())
            else "rejected_retain_p2"
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    Path(args.output).write_text(rendered, encoding="utf-8")
    compact = copy.deepcopy(result)
    cast(dict[str, object], compact["p2_champion"]).pop("sessions", None)
    cast(dict[str, object], compact["p4_candidate"]).pop("sessions", None)
    cast(dict[str, object], compact["comparison"]).pop("sessions", None)
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
