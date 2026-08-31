"""Prospective first-divergence attribution for frozen P2 and P5 shadow replays."""

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
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from starter.agent import Agent
from starter.clarification_policies import clarification_policy_by_id
from starter.recommendation_exposure import disabled_exposure_policy

from .shadow_clarification_suite import (
    MAX_TURNS,
    SEED,
    TOP_K,
    ShadowSample,
    _agent,
    _contract_counts,
    _perturb,
    _profile,
    _ranked,
    build_shadow_samples,
    customer_reply,
    initial_message,
    load_jsonl,
    metric_summary,
    public_targets,
    robustness_checks,
    scenario_metrics,
    select_shadow_products,
)

P2_POLICY_ID = "clarification.category-evidence-utility.v1"
P5_POLICY_ID = "clarification.category-evidence-utility-buying.v1"
CONTEXTUAL_POLICY_ID = "contextual.category-evidence.v1"
P2_EXPECTED_HASH = "765c2e05e4bf8dc6f87ab5f099d5ce1144dc1405a39850151e3bff8e60c467f4"
P5_EXPECTED_HASH = "33f51cc6f6eb7eff627c116ff919b02a7bbb50fe6b24ddaef7efc9a00e693e92"


def _sha(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=repr
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rss_mib() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return round(raw / divisor, 6)


def _state(agent: Agent, session_id: str) -> dict[str, object]:
    state = agent._state.state_for(session_id)
    payload: dict[str, object] = {}
    for name in ("category", "color", "style", "material", "use_case"):
        value = getattr(state, name, None)
        payload[name] = None if value is None else str(value.value)
    payload["price"] = (
        None
        if state.price is None
        else {"minimum": state.price.minimum, "maximum": state.price.maximum}
    )
    payload["removed_constraints"] = sorted(state.removed_constraints)
    return payload


def _response_without_question(response: Mapping[str, object]) -> dict[str, object]:
    return {
        key: copy.deepcopy(value)
        for key, value in response.items()
        if key not in {"message", "ask_attribute"}
    }


def _changed_slots(
    left: Mapping[str, object], right: Mapping[str, object]
) -> list[str]:
    return sorted(
        key for key in set(left) | set(right) if left.get(key) != right.get(key)
    )


def _latency(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "p95_ms": 0.0, "maximum_ms": 0.0}
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
    return {
        "mean_ms": round(statistics.fmean(values), 6),
        "p95_ms": round(ordered[index], 6),
        "maximum_ms": round(ordered[-1], 6),
    }


def _build_agent(policy_id: str, catalog: Path, dense_cache: Path) -> Agent:
    policy = clarification_policy_by_id(policy_id)
    return _agent(
        catalog,
        dense_cache,
        policy.clarification,
        policy.controller,
        CONTEXTUAL_POLICY_ID,
        disabled_exposure_policy(),
    )


def _trace_replay(
    policy_id: str,
    samples: Sequence[ShadowSample],
    catalog: Path,
    dense_cache: Path,
    catalog_ids: frozenset[str],
) -> dict[str, object]:
    startup_started = time.perf_counter()
    agent = _build_agent(policy_id, catalog, dense_cache)
    startup_seconds = time.perf_counter() - startup_started
    sessions: list[dict[str, object]] = []
    correctness: Counter[str] = Counter()
    transcript_hasher = hashlib.sha256()
    latencies: list[float] = []
    evaluation_started = time.perf_counter()
    for sample in samples:
        session_id = f"{sample.sample_id}_{sample.template_variant}"
        agent.reset(session_id, _profile(sample))
        disclosed: set[str] = set()
        boundary_declined = False
        override_applied = sample.scenario_type != "intent_override"
        user_message = initial_message(sample, disclosed)
        first_hit_turn: int | None = None
        best_rank: int | None = None
        turns: list[dict[str, object]] = []
        for turn in range(1, MAX_TURNS + 1):
            pre_state = _state(agent, session_id)
            started = time.perf_counter()
            response = agent.respond(session_id, user_message, turn, TOP_K)
            latencies.append((time.perf_counter() - started) * 1000.0)
            correctness.update(_contract_counts(response, catalog_ids))
            transcript_hasher.update(
                json.dumps(
                    {
                        "sample_id": sample.sample_id,
                        "turn": turn,
                        "user_message": user_message,
                        "response": response,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    default=repr,
                ).encode()
            )
            route = agent._contextual_routes.get(session_id, "uncertain")
            analysis = [
                candidate.parent_asin
                for candidate in agent._clarification_candidates.get(session_id, [])
            ]
            ranked = _ranked(response, catalog_ids)
            turns.append(
                {
                    "turn": turn,
                    "user_message_sha256": _sha(user_message),
                    "pre_state": pre_state,
                    "pre_state_sha256": _sha(pre_state),
                    "route": route,
                    "analysis_candidate_count": len(analysis),
                    "analysis_candidate_sha256": _sha(analysis),
                    "published_recommendations": ranked,
                    "published_recommendations_sha256": _sha(ranked),
                    "ask_attribute": response.get("ask_attribute"),
                    "usage": copy.deepcopy(response.get("usage")),
                    "usage_sha256": _sha(response.get("usage")),
                    "response_without_question_sha256": _sha(
                        _response_without_question(response)
                    ),
                    "response_sha256": _sha(response),
                    "post_state": _state(agent, session_id),
                    "post_state_sha256": _sha(_state(agent, session_id)),
                }
            )
            if override_applied and sample.target in ranked:
                first_hit_turn = turn
                best_rank = ranked.index(sample.target) + 1
                break
            if turn == MAX_TURNS:
                break
            ask_attribute = response.get("ask_attribute")
            next_turn = turn + 1
            if (
                not override_applied
                and sample.override_turn is not None
                and next_turn == sample.override_turn
            ):
                override_applied = True
                if sample.new_override_value:
                    disclosed.add(sample.new_override_value)
                user_message = (
                    "Actually, ignore my earlier preference. I now need "
                    f"{_perturb(sample.new_override_value or '', sample)}."
                )
            else:
                user_message, boundary_declined, _revealed = customer_reply(
                    sample, ask_attribute, disclosed, boundary_declined
                )
        sessions.append(
            {
                "sample_id": sample.sample_id,
                "scenario_type": sample.scenario_type,
                "hit": first_hit_turn is not None,
                "first_hit_turn": first_hit_turn,
                "best_rank": best_rank,
                "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
                "turns": turns,
            }
        )
    evaluation_seconds = time.perf_counter() - evaluation_started
    outcome_sessions = [
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
    clarification = agent.clarification_diagnostics_snapshot()
    fallback = agent.diagnostics_snapshot()
    exposure = agent.exposure_diagnostics_snapshot()
    result = {
        **metric_summary(outcome_sessions),
        "scenario_metrics": scenario_metrics(outcome_sessions),
        "normalized_transcript_sha256": transcript_hasher.hexdigest(),
        "sessions": sessions,
        "outcome_sessions": outcome_sessions,
        "correctness_counters": {
            name: correctness[name]
            for name in (
                "response_exceptions",
                "invalid_responses",
                "invalid_ask_attributes",
                "invalid_asins",
                "duplicate_recommendations",
            )
        },
        "clarification_diagnostics": clarification,
        "fallback_diagnostics": fallback,
        "exposure_diagnostics": exposure,
        "performance": {
            "startup_seconds": round(startup_seconds, 6),
            "evaluation_wall_seconds": round(evaluation_seconds, 6),
            "response_count": len(latencies),
            "response_latency": _latency(latencies),
            "process_peak_rss_mib": _rss_mib(),
        },
    }
    del agent
    gc.collect()
    return result


def _turn_map(session: Mapping[str, object]) -> dict[int, dict[str, object]]:
    return {
        cast(int, turn["turn"]): turn
        for turn in cast(list[dict[str, object]], session["turns"])
    }


def _outcome_delta(
    left: Mapping[str, object], right: Mapping[str, object]
) -> dict[str, object]:
    left_hit, right_hit = bool(left["hit"]), bool(right["hit"])
    return {
        "hit_delta": int(right_hit) - int(left_hit),
        "rank_delta": (
            None
            if left["best_rank"] is None or right["best_rank"] is None
            else cast(int, right["best_rank"]) - cast(int, left["best_rank"])
        ),
        "turn_delta": (
            None
            if left["first_hit_turn"] is None or right["first_hit_turn"] is None
            else cast(int, right["first_hit_turn"]) - cast(int, left["first_hit_turn"])
        ),
        "P2": {key: left[key] for key in ("hit", "first_hit_turn", "best_rank")},
        "P5": {key: right[key] for key in ("hit", "first_hit_turn", "best_rank")},
    }


def _lineage(p2: Mapping[str, object], p5: Mapping[str, object]) -> dict[str, object]:
    before = {
        str(row["sample_id"]): row
        for row in cast(list[dict[str, object]], p2["sessions"])
    }
    after = {
        str(row["sample_id"]): row
        for row in cast(list[dict[str, object]], p5["sessions"])
    }
    records: list[dict[str, object]] = []
    nonintervention_identical = 0
    changed_without_qualifying = 0
    qualifying_count = 0
    all_changed_pass = True
    recommendation_parity_count = 0
    usage_parity_count = 0
    for sample_id in sorted(before):
        left_session, right_session = before[sample_id], after[sample_id]
        left_turns, right_turns = _turn_map(left_session), _turn_map(right_session)
        common_turns = sorted(set(left_turns) & set(right_turns))
        first = next(
            (
                turn
                for turn in common_turns
                if left_turns[turn]["response_sha256"]
                != right_turns[turn]["response_sha256"]
                or left_turns[turn]["pre_state_sha256"]
                != right_turns[turn]["pre_state_sha256"]
                or left_turns[turn]["user_message_sha256"]
                != right_turns[turn]["user_message_sha256"]
                or left_turns[turn]["route"] != right_turns[turn]["route"]
                or left_turns[turn]["analysis_candidate_sha256"]
                != right_turns[turn]["analysis_candidate_sha256"]
            ),
            None,
        )
        exact_session = (
            left_session["turns"] == right_session["turns"]
            and _outcome_delta(left_session, right_session)["hit_delta"] == 0
            and left_session["best_rank"] == right_session["best_rank"]
            and left_session["first_hit_turn"] == right_session["first_hit_turn"]
        )
        if first is None:
            if exact_session:
                nonintervention_identical += 1
                continue
            all_changed_pass = False
            changed_without_qualifying += 1
            records.append({"sample_id": sample_id, "qualifying": False})
            continue
        left, right = left_turns[first], right_turns[first]
        pre_turns_identical = all(
            left_turns[turn] == right_turns[turn]
            for turn in common_turns
            if turn < first
        )
        recommendation_parity = (
            left["published_recommendations"] == right["published_recommendations"]
        )
        usage_parity = left["usage"] == right["usage"]
        response_rest_parity = (
            left["response_without_question_sha256"]
            == right["response_without_question_sha256"]
        )
        qualifying = bool(
            pre_turns_identical
            and left["user_message_sha256"] == right["user_message_sha256"]
            and left["pre_state_sha256"] == right["pre_state_sha256"]
            and left["post_state_sha256"] == right["post_state_sha256"]
            and left["route"] == "buying"
            and right["route"] == "buying"
            and left["ask_attribute"] is None
            and right["ask_attribute"] in {"other", "feature"}
            and recommendation_parity
            and usage_parity
            and response_rest_parity
        )
        if qualifying:
            qualifying_count += 1
            recommendation_parity_count += int(recommendation_parity)
            usage_parity_count += int(usage_parity)
        else:
            changed_without_qualifying += 1
            all_changed_pass = False
        later_turns = sorted(
            (set(left_turns) | set(right_turns))
            - {turn for turn in common_turns if turn <= first}
        )
        route_transitions = [
            {
                "turn": turn,
                "P2": left_turns.get(turn, {}).get("route"),
                "P5": right_turns.get(turn, {}).get("route"),
            }
            for turn in later_turns
            if left_turns.get(turn, {}).get("route")
            != right_turns.get(turn, {}).get("route")
        ]
        pool_changes = sum(
            left_turns.get(turn, {}).get("analysis_candidate_sha256")
            != right_turns.get(turn, {}).get("analysis_candidate_sha256")
            or left_turns.get(turn, {}).get("published_recommendations_sha256")
            != right_turns.get(turn, {}).get("published_recommendations_sha256")
            for turn in later_turns
        )
        next_turn = first + 1
        left_after = left_turns.get(next_turn)
        right_after = right_turns.get(next_turn)
        records.append(
            {
                "sample_id": sample_id,
                "scenario_type": left_session["scenario_type"],
                "qualifying": qualifying,
                "first_divergence_turn": first,
                "route_at_intervention": left["route"],
                "P2_question_field": left["ask_attribute"],
                "P5_question_field": right["ask_attribute"],
                "recommendation_parity_at_intervention": recommendation_parity,
                "usage_parity_at_intervention": usage_parity,
                "response_fields_other_than_question_parity": response_rest_parity,
                "state_before_answer_sha256": left["post_state_sha256"],
                "P2_state_after_answer_sha256": None
                if left_after is None
                else left_after["post_state_sha256"],
                "P5_state_after_answer_sha256": None
                if right_after is None
                else right_after["post_state_sha256"],
                "state_slots_changed_after_answer": (
                    []
                    if left_after is None or right_after is None
                    else _changed_slots(
                        cast(dict[str, object], left_after["post_state"]),
                        cast(dict[str, object], right_after["post_state"]),
                    )
                ),
                "subsequent_route_transitions": route_transitions,
                "candidate_pool_or_ranking_changed_turn_count": pool_changes,
                "final_outcome_delta": _outcome_delta(left_session, right_session),
            }
        )
    return {
        "paired_session_count": len(before),
        "qualifying_intervention_session_count": qualifying_count,
        "nonintervention_exact_session_count": nonintervention_identical,
        "changed_without_qualifying_intervention_count": changed_without_qualifying,
        "recommendation_parity_intervention_count": recommendation_parity_count,
        "usage_parity_intervention_count": usage_parity_count,
        "all_changed_sessions_pass_first_divergence_attribution": all_changed_pass,
        "all_nonintervention_sessions_exact_P2": (
            qualifying_count + nonintervention_identical == len(before)
            and changed_without_qualifying == 0
        ),
        "affected_sessions": records,
    }


def _policy_parity() -> dict[str, bool]:
    p2 = clarification_policy_by_id(P2_POLICY_ID)
    p5 = clarification_policy_by_id(P5_POLICY_ID)
    left, right = p2.fingerprint_payload(), p5.fingerprint_payload()
    left["policy_id"] = P5_POLICY_ID
    left["clarification"]["eligible_routes"] = ("browsing", "boundary", "buying")  # type: ignore[index]
    return {
        "P5_configuration_exact_frozen_P4_fingerprint": p5.fingerprint_sha256
        == "6db00179643c355adf1ecfbef5fee680ce50ce316f6f5c272da9aa53ab8bf62e",
        "only_policy_difference_from_P2_is_buying_route": left == right,
        "no_target_or_session_policy_fields": set(p2.clarification.__dataclass_fields__)
        == set(p5.clarification.__dataclass_fields__),
    }


def _question_safety(p5: Mapping[str, object]) -> dict[str, object]:
    turn_10 = 0
    repeated = 0
    for session in cast(list[dict[str, object]], p5["sessions"]):
        asked: set[str] = set()
        for turn in cast(list[dict[str, object]], session["turns"]):
            attribute = turn["ask_attribute"]
            if not isinstance(attribute, str):
                continue
            turn_10 += int(turn["turn"] == 10)
            repeated += int(attribute in asked)
            asked.add(attribute)
    return {
        "turn_10_question_count": turn_10,
        "repeated_resolved_or_declined_attribute_count": repeated,
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
    catalog, public_set, dense_cache = (
        Path(args.catalog),
        Path(args.public_set),
        Path(args.dense_cache),
    )
    products = {
        str(row["parent_asin"]): row
        for row in load_jsonl(catalog)
        if isinstance(row.get("parent_asin"), str)
    }
    catalog_ids = frozenset(products)
    excluded = public_targets(load_jsonl(public_set))
    samples = build_shadow_samples(products, excluded, args.sample_count)
    targets = {sample.target for sample in samples}
    reordered = dict(reversed(list(products.items())))
    reordered_targets = {
        item[0]
        for item in select_shadow_products(reordered, excluded, args.sample_count)
    }

    p2 = _trace_replay(P2_POLICY_ID, samples, catalog, dense_cache, catalog_ids)
    p5 = _trace_replay(P5_POLICY_ID, samples, catalog, dense_cache, catalog_ids)
    p5_repeat = _trace_replay(P5_POLICY_ID, samples, catalog, dense_cache, catalog_ids)
    lineage = _lineage(p2, p5)
    question_safety = _question_safety(p5)
    robustness = {**robustness_checks(), **_policy_parity()}
    deterministic = (
        p5["normalized_transcript_sha256"] == p5_repeat["normalized_transcript_sha256"]
        and p5["outcome_sessions"] == p5_repeat["outcome_sessions"]
        and p5["sessions"] == p5_repeat["sessions"]
        and p5["clarification_diagnostics"] == p5_repeat["clarification_diagnostics"]
    )
    fallback = cast(dict[str, object], p5["fallback_diagnostics"])
    clarification = cast(dict[str, object], p5["clarification_diagnostics"])
    exposure = cast(dict[str, object], p5["exposure_diagnostics"])
    counters_zero = (
        all(
            value == 0
            for value in cast(dict[str, int], p5["correctness_counters"]).values()
        )
        and fallback["fallback_attempt_count"] == 0
        and fallback["routing_failure_count"] == 0
        and not fallback["component_failure_counts"]
        and not fallback["initialization_fallback_counts"]
        and clarification["clarification_failure_count"] == 0
        and exposure["exposure_failure_count"] == 0
    )
    gates = {
        "instrumented_P2_matches_frozen_hash": p2["normalized_transcript_sha256"]
        == P2_EXPECTED_HASH,
        "instrumented_P5_matches_frozen_P4_hash": p5["normalized_transcript_sha256"]
        == P5_EXPECTED_HASH,
        "all_changed_sessions_pass_first_divergence_attribution": lineage[
            "all_changed_sessions_pass_first_divergence_attribution"
        ],
        "all_nonintervention_sessions_exact_P2": lineage[
            "all_nonintervention_sessions_exact_P2"
        ],
        "recommendation_parity_on_every_intervention": lineage[
            "recommendation_parity_intervention_count"
        ]
        == lineage["qualifying_intervention_session_count"],
        "usage_parity_on_every_intervention": lineage["usage_parity_intervention_count"]
        == lineage["qualifying_intervention_session_count"],
        "zero_correctness_and_failure_counters": counters_zero,
        "no_turn_10_questions": question_safety["turn_10_question_count"] == 0,
        "no_repeated_resolved_or_declined_attributes": question_safety[
            "repeated_resolved_or_declined_attribute_count"
        ]
        == 0,
        "deterministic_replay": deterministic,
        "target_independent_robustness": all(robustness.values()),
        "public_target_overlap_zero": len(targets & excluded) == 0,
        "catalog_reorder_invariant": targets == reordered_targets,
    }
    result = {
        "schema_version": 1,
        "experiment_id": "P5-prospective-causal-clarification-confirmation-v1",
        "seed": SEED,
        "inputs": {
            "catalog_sha256": _file_sha(catalog),
            "public_set_sha256": _file_sha(public_set),
            "predeclaration_sha256": _file_sha(
                "docs/results/autonomous_optimization/shadow_results/p5_causal_confirmation_predeclaration.json"
            ),
            "clarification_registry_sha256": _file_sha(
                "config/clarification_policies.json"
            ),
            "P2_policy_fingerprint": clarification_policy_by_id(
                P2_POLICY_ID
            ).fingerprint_sha256,
            "P5_policy_fingerprint": clarification_policy_by_id(
                P5_POLICY_ID
            ).fingerprint_sha256,
            "P5_behavior_equals_frozen_P4": True,
            "P3_exposure_enabled": False,
        },
        "suite": {
            "sample_count": len(samples),
            "public_target_overlap": len(targets & excluded),
            "catalog_reorder_invariant": targets == reordered_targets,
            "target_selection_sha256": hashlib.sha256(
                "\n".join(sorted(targets)).encode()
            ).hexdigest(),
        },
        "P2": p2,
        "P5": p5,
        "P5_repeat_hash": p5_repeat["normalized_transcript_sha256"],
        "lineage": lineage,
        "question_safety": question_safety,
        "robustness_checks": robustness,
        "preflight_gates": gates,
        "preflight_verdict": "pass" if all(gates.values()) else "fail_retain_P2",
        "official_runs_consumed": 0,
        "official_runs_remaining": 4,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    Path(args.output).write_text(rendered, encoding="utf-8")
    compact = copy.deepcopy(result)
    cast(dict[str, object], compact["P2"]).pop("sessions", None)
    cast(dict[str, object], compact["P5"]).pop("sessions", None)
    cast(dict[str, object], compact["lineage"]).pop("affected_sessions", None)
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
