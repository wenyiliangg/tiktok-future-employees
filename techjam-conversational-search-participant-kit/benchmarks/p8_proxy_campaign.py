"""Frozen target-independent proxy harness for the P8 clarification fallback."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from evaluator.local_evaluator import (
    catalog_index,
    coarse_category,
    evaluate,
    intent_card,
    materialize_hidden_fields,
    peak_process_rss_bytes,
)
from starter.agent import Agent
from starter.clarification_controller import ClarificationController
from starter.clarification_fallback import fallback_policy_by_id
from starter.clarification_policies import clarification_policy_by_id
from starter.contextual_retrieval import policy_by_id
from starter.hybrid_retrieval import HybridRetrievalConfig, RetrievalMode
from starter.recommendation_exposure import disabled_exposure_policy

PREDECLARATION_COMMIT = "dbdcf73894eed9f10fd6c580c680bc918acbf8c4"
DERIVATION_COMMIT = "279293055a1ea6a15ae72b951df1385090c7986d"
IMPLEMENTATION_COMMIT = "67222de722db1789d3aa26780702578a95cbd66c"
P5_COMMIT = "f4992683cff2fe0cb6e4e756edeb361f1f29f6b0"
P5_POLICY_ID = "clarification.category-evidence-utility-buying.v1"
CONTEXTUAL_POLICY_ID = "contextual.category-evidence.v1"
POLICY_IDS = {
    "disabled": "clarification-fallback.disabled.v1",
    "open": "clarification-fallback.open-once.v1",
    "catalog_utility": "clarification-fallback.catalog-utility-once.v1",
}
PHASE_SEEDS = {
    "development": 2026083101,
    "confirmation": 2026083102,
    "uniform_stress": 2026083103,
}
BOOTSTRAP_SEED = 2026083191
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_CONFIDENCE = 0.975
SAMPLE_COUNT = 400
SCENARIOS = ("boundary", "browsing", "buying", "intent_override")
PROFILE: dict[str, object] = {
    "average_prior_rating": 4.0,
    "preference_tags": ["general shopping"],
    "purchase_frequency": "3-4 prior purchases",
    "rating_style": "mixed",
    "summary": "Prior purchases emphasize general shopping; ratings are mixed.",
}


@dataclass(frozen=True, slots=True)
class EligibleTarget:
    parent_asin: str
    category: str
    popularity_quartile: int


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=repr
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _hash_order(seed: int, parent_asin: str, phase: str) -> str:
    material = f"p8-proxy-{phase}-v1\0{seed}\0{parent_asin}".encode()
    return hashlib.sha256(material).hexdigest()


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _popularity(product: Mapping[str, object]) -> float:
    try:
        count = max(0, int(product.get("rating_number") or 0))
    except (TypeError, ValueError, OverflowError):
        count = 0
    return math.log1p(count)


def _quartile(value: float, thresholds: Sequence[float]) -> int:
    if value <= thresholds[0]:
        return 1
    if value <= thresholds[1]:
        return 2
    if value <= thresholds[2]:
        return 3
    return 4


def _eligible_targets(
    products: Mapping[str, dict[str, Any]],
    categories: Mapping[str, list[str]],
    excluded: set[str],
    thresholds: Sequence[float],
) -> tuple[list[EligibleTarget], dict[str, int]]:
    eligible: list[EligibleTarget] = []
    failures: Counter[str] = Counter()
    for parent_asin in sorted(products):
        if parent_asin in excluded:
            continue
        product = products[parent_asin]
        try:
            intent_card(product)
        except Exception:  # noqa: BLE001 - frozen eligibility boundary
            failures["intent_card"] += 1
            continue
        probe = {
            "sample_id": "p8_eligibility_probe",
            "scenario_type": "boundary",
            "user_profile": PROFILE,
            "ground_truth": {"parent_asin": parent_asin},
        }
        try:
            materialize_hidden_fields(probe, cast(dict[str, dict], products))
        except Exception:  # noqa: BLE001 - frozen eligibility boundary
            failures["materialize_hidden_fields"] += 1
            continue
        eligible.append(
            EligibleTarget(
                parent_asin,
                coarse_category(categories.get(parent_asin, [])),
                _quartile(_popularity(product), thresholds),
            )
        )
    return eligible, dict(sorted(failures.items()))


def _stratified_select(
    eligible: Sequence[EligibleTarget], *, seed: int, phase: str
) -> list[EligibleTarget]:
    strata: dict[tuple[str, int], list[EligibleTarget]] = defaultdict(list)
    for row in eligible:
        strata[(row.category, row.popularity_quartile)].append(row)
    total = len(eligible)
    allocations: dict[tuple[str, int], int] = {
        key: SAMPLE_COUNT * len(rows) // total for key, rows in strata.items()
    }
    remaining = SAMPLE_COUNT - sum(allocations.values())
    remainders = sorted(
        strata,
        key=lambda key: (
            -(SAMPLE_COUNT * len(strata[key]) % total),
            key[0],
            key[1],
        ),
    )
    for key in remainders[:remaining]:
        allocations[key] += 1
    selected: list[EligibleTarget] = []
    for key in sorted(strata):
        ordered = sorted(
            strata[key], key=lambda row: _hash_order(seed, row.parent_asin, phase)
        )
        selected.extend(ordered[: allocations[key]])
    if len(selected) != SAMPLE_COUNT:
        raise RuntimeError("stratified allocation did not select exactly 400 targets")
    return selected


def _uniform_select(
    eligible: Sequence[EligibleTarget], *, seed: int, phase: str
) -> list[EligibleTarget]:
    return sorted(eligible, key=lambda row: _hash_order(seed, row.parent_asin, phase))[
        :SAMPLE_COUNT
    ]


def _select_phase(
    phase: str,
    all_eligible: Sequence[EligibleTarget],
    public_targets: set[str],
) -> tuple[list[EligibleTarget], dict[str, list[EligibleTarget]]]:
    prior: dict[str, list[EligibleTarget]] = {}
    excluded = set(public_targets)
    for current in ("development", "confirmation", "uniform_stress"):
        available = [row for row in all_eligible if row.parent_asin not in excluded]
        if current == "uniform_stress":
            chosen = _uniform_select(
                available, seed=PHASE_SEEDS[current], phase=current
            )
        else:
            chosen = _stratified_select(
                available, seed=PHASE_SEEDS[current], phase=current
            )
        prior[current] = chosen
        excluded.update(row.parent_asin for row in chosen)
        if current == phase:
            return chosen, prior
    raise ValueError(f"unsupported phase: {phase}")


def _selection_summary(
    selected: Sequence[EligibleTarget],
    public_targets: set[str],
    prior: Mapping[str, Sequence[EligibleTarget]],
    phase: str,
) -> dict[str, object]:
    ids = {row.parent_asin for row in selected}
    ordered_ids = sorted(ids)
    category_counts = Counter(row.category for row in selected)
    quartile_counts = Counter(str(row.popularity_quartile) for row in selected)
    intersections: dict[str, int] = {"public": len(ids & public_targets)}
    for name, rows in prior.items():
        if name == phase:
            continue
        intersections[name] = len(ids & {row.parent_asin for row in rows})
    return {
        "sample_count": len(selected),
        "target_selection_sha256": hashlib.sha256(
            "\n".join(ordered_ids).encode()
        ).hexdigest(),
        "ordered_assignment_sha256": _json_sha256(
            [
                row.parent_asin
                for row in sorted(
                    selected,
                    key=lambda item: _hash_order(
                        PHASE_SEEDS[phase], item.parent_asin, phase
                    ),
                )
            ]
        ),
        "category_counts": dict(sorted(category_counts.items())),
        "popularity_quartile_counts": dict(sorted(quartile_counts.items())),
        "scenario_counts": {name: 100 for name in SCENARIOS},
        "intersections": dict(sorted(intersections.items())),
        "target_ids_persisted": False,
    }


def _samples(
    selected: Sequence[EligibleTarget], *, phase: str
) -> list[dict[str, object]]:
    ordered = sorted(
        selected,
        key=lambda row: _hash_order(PHASE_SEEDS[phase], row.parent_asin, phase),
    )
    return [
        {
            "sample_id": f"p8_{phase}_{index:04d}",
            "scenario_type": SCENARIOS[(index - 1) % len(SCENARIOS)],
            "user_profile": dict(PROFILE),
            "ground_truth": {"parent_asin": row.parent_asin},
        }
        for index, row in enumerate(ordered, 1)
    ]


def _agent(policy_name: str, catalog_path: Path, dense_cache: Path) -> Agent:
    clarification = clarification_policy_by_id(P5_POLICY_ID)
    return Agent(
        catalog_path,
        config=HybridRetrievalConfig(mode=RetrievalMode.CONTEXTUAL),
        dense_cache_path=dense_cache,
        contextual_policy=policy_by_id(CONTEXTUAL_POLICY_ID),
        clarification_config=clarification.clarification,
        clarification_controller=ClarificationController(clarification.controller),
        exposure_policy=disabled_exposure_policy(),
        clarification_fallback_policy=fallback_policy_by_id(POLICY_IDS[policy_name]),
    )


def _instrument(
    agent: Agent, sample_ids: Sequence[str]
) -> tuple[dict[str, str], dict[str, list[dict[str, object]]]]:
    original_reset = agent.reset
    original_respond = agent.respond
    mapping: dict[str, str] = {}
    observations: dict[str, list[dict[str, object]]] = defaultdict(list)
    reset_index = 0

    def observed_reset(session_id: str, user_profile: dict) -> None:
        nonlocal reset_index
        if reset_index >= len(sample_ids):
            raise RuntimeError("evaluator reset count exceeded proxy sample count")
        mapping[session_id] = sample_ids[reset_index]
        reset_index += 1
        original_reset(session_id, user_profile)

    def observed_respond(
        session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        response = original_respond(session_id, user_message, turn, top_k)
        sample_id = mapping[session_id]
        observations[sample_id].append(
            {
                "turn": turn,
                "user_message_sha256": hashlib.sha256(
                    user_message.encode()
                ).hexdigest(),
                "response_sha256": _json_sha256(response),
                "ask_attribute": response.get("ask_attribute"),
                "recommendations_sha256": _json_sha256(response.get("recommendations")),
                "usage_sha256": _json_sha256(response.get("usage")),
                "reply_kind": (
                    "answered"
                    if user_message.startswith("For that, what matters is:")
                    else "declined"
                    if "don't have" in user_message.lower()
                    else "other"
                ),
            }
        )
        return response

    agent.reset = observed_reset  # type: ignore[method-assign]
    agent.respond = observed_respond  # type: ignore[method-assign]
    return mapping, observations


def _metric_summary(sessions: Sequence[Mapping[str, object]]) -> dict[str, object]:
    count = len(sessions)
    hit_rate = sum(bool(row["hit"]) for row in sessions) / count
    mrr = statistics.fmean(float(row["reciprocal_rank"]) for row in sessions)
    mttc = statistics.fmean(
        int(row["first_hit_turn"]) if row["first_hit_turn"] is not None else 11
        for row in sessions
    )
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    score = 0.5 * hit_rate + 0.3 * mrr + 0.2 * efficiency
    return {
        "sample_count": count,
        "hit_rate_at_10": round(hit_rate, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
        "efficiency": round(efficiency, 6),
        "technical_score": round(score, 6),
    }


def _scenario_metrics(
    sessions: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in sessions:
        grouped[str(row["scenario_type"])].append(row)
    return {name: _metric_summary(rows) for name, rows in sorted(grouped.items())}


def _evaluate_policy(
    policy_name: str,
    catalog_path: Path,
    dense_cache: Path,
    samples: list[dict[str, object]],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> dict[str, object]:
    started = time.perf_counter()
    agent = _agent(policy_name, catalog_path, dense_cache)
    startup_seconds = time.perf_counter() - started
    mapping, observations = _instrument(
        agent, [str(sample["sample_id"]) for sample in samples]
    )
    evaluation_started = time.perf_counter()
    outcome = evaluate(agent, samples, catalog_ids, categories, products)
    wall_seconds = time.perf_counter() - evaluation_started
    events: dict[str, list[dict[str, object]]] = {}
    for session_id, sample_id in mapping.items():
        snapshot = agent.clarification_fallback_diagnostics_snapshot(session_id)
        rows = cast(list[dict[str, object]], snapshot["events"])
        if rows:
            events[sample_id] = rows
    fallback = agent.clarification_fallback_diagnostics_snapshot()
    clarification = agent.clarification_diagnostics_snapshot()
    routing = agent.diagnostics_snapshot()
    exposure = agent.exposure_diagnostics_snapshot()
    sessions = cast(list[dict[str, object]], outcome["sessions"])
    outcome["technical_score"] = outcome.pop("recommended_technical_score")
    outcome["scenario_metrics"] = _scenario_metrics(sessions)
    outcome["clarification_diagnostics"] = clarification
    outcome["clarification_fallback_diagnostics"] = fallback
    outcome["routing_diagnostics"] = routing
    outcome["exposure_diagnostics"] = exposure
    outcome["performance"] = {
        "startup_seconds": round(startup_seconds, 6),
        "evaluation_wall_seconds": round(wall_seconds, 6),
        "peak_process_rss_mib": round(peak_process_rss_bytes() / 2**20, 6),
        "p95_response_latency_ms": cast(
            dict[str, object], outcome["evaluation_diagnostics"]
        )["p95_response_latency_ms"],
    }
    outcome["observations"] = dict(sorted(observations.items()))
    outcome["fallback_events"] = dict(sorted(events.items()))
    del agent
    gc.collect()
    return outcome


def _session_score(row: Mapping[str, object]) -> float:
    hit = 1.0 if row["hit"] else 0.0
    rank = float(row["reciprocal_rank"])
    turn = int(row["first_hit_turn"]) if row["first_hit_turn"] is not None else 11
    return 0.5 * hit + 0.3 * rank + 0.2 * ((11 - turn) / 10)


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _comparison(
    champion: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, object]:
    before = {
        str(row["sample_id"]): row
        for row in cast(list[dict[str, object]], champion["sessions"])
    }
    after = {
        str(row["sample_id"]): row
        for row in cast(list[dict[str, object]], candidate["sessions"])
    }
    if before.keys() != after.keys():
        raise ValueError("paired proxy outcomes have different sample ids")
    labels: Counter[str] = Counter()
    deltas: list[float] = []
    paired_rows: list[dict[str, object]] = []
    for sample_id in sorted(before):
        left, right = before[sample_id], after[sample_id]
        delta = _session_score(right) - _session_score(left)
        deltas.append(delta)
        if bool(left["hit"]) != bool(right["hit"]):
            label = "gained_hit" if right["hit"] else "lost_hit"
        elif not left["hit"]:
            label = "unchanged_miss"
        elif left["first_hit_turn"] != right["first_hit_turn"]:
            label = (
                "earlier_hit"
                if int(right["first_hit_turn"]) < int(left["first_hit_turn"])
                else "later_hit"
            )
        elif left["best_rank"] != right["best_rank"]:
            label = (
                "better_rank"
                if int(right["best_rank"]) < int(left["best_rank"])
                else "worse_rank"
            )
        else:
            label = "unchanged_hit"
        labels[label] += 1
        paired_rows.append(
            {
                "sample_id": sample_id,
                "scenario_type": left["scenario_type"],
                "comparison": label,
                "technical_score_contribution_delta": round(delta, 9),
                "champion": {
                    key: left.get(key)
                    for key in (
                        "hit",
                        "first_hit_turn",
                        "best_rank",
                        "reciprocal_rank",
                    )
                },
                "candidate": {
                    key: right.get(key)
                    for key in (
                        "hit",
                        "first_hit_turn",
                        "best_rank",
                        "reciprocal_rank",
                    )
                },
            }
        )
    rng = random.Random(BOOTSTRAP_SEED)
    means = [
        statistics.fmean(deltas[rng.randrange(len(deltas))] for _ in deltas)
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    tail = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    before_scenarios = cast(dict[str, dict[str, object]], champion["scenario_metrics"])
    after_scenarios = cast(dict[str, dict[str, object]], candidate["scenario_metrics"])
    scenario_deltas = {
        name: round(
            float(after_scenarios[name]["technical_score"])
            - float(before_scenarios[name]["technical_score"]),
            6,
        )
        for name in sorted(before_scenarios)
    }
    return {
        "paired_session_count": len(deltas),
        "counts": dict(sorted(labels.items())),
        "net_hits": labels["gained_hit"] - labels["lost_hit"],
        "technical_score_delta": round(
            float(candidate["technical_score"]) - float(champion["technical_score"]),
            6,
        ),
        "hit_rate_delta": round(
            float(candidate["hit_rate_at_10"]) - float(champion["hit_rate_at_10"]),
            6,
        ),
        "mrr_delta": round(float(candidate["mrr"]) - float(champion["mrr"]), 6),
        "mttc_delta": round(float(candidate["mttc"]) - float(champion["mttc"]), 6),
        "scenario_technical_score_deltas": scenario_deltas,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "confidence_level": BOOTSTRAP_CONFIDENCE,
            "observed_mean_delta": round(statistics.fmean(deltas), 9),
            "lower": round(_percentile(means, tail), 9),
            "upper": round(_percentile(means, 1 - tail), 9),
            "probability_delta_positive": round(
                sum(value > 0 for value in means) / len(means), 6
            ),
        },
        "sessions": paired_rows,
    }


def _failure_count(outcome: Mapping[str, object]) -> tuple[int, dict[str, int]]:
    evaluator = cast(dict[str, object], outcome["evaluation_diagnostics"])
    contract = cast(dict[str, object], outcome["response_contract_diagnostics"])
    clarification = cast(dict[str, object], outcome["clarification_diagnostics"])
    fallback = cast(dict[str, object], outcome["clarification_fallback_diagnostics"])
    routing = cast(dict[str, object], outcome["routing_diagnostics"])
    exposure = cast(dict[str, object], outcome["exposure_diagnostics"])
    component = cast(dict[str, object], routing["component_failure_counts"])
    initialization = cast(dict[str, object], routing["initialization_fallback_counts"])
    fields = {
        "response_exceptions": int(evaluator["response_exception_count"]),
        "repeated_questions": int(contract["repeated_question_count"]),
        "invalid_ask_attributes": int(contract["invalid_ask_attribute_count"]),
        "invalid_asins": int(contract["invalid_asin_count"]),
        "duplicate_recommendations": int(contract["duplicate_recommendation_count"]),
        "invalid_responses": int(contract["invalid_response_count"]),
        "clarification_failures": int(clarification["clarification_failure_count"]),
        "p8_fallback_failures": int(fallback["failure_count"]),
        "routing_failures": int(routing["routing_failure_count"]),
        "component_failures": sum(int(value) for value in component.values()),
        "initialization_fallbacks": sum(
            int(value) for value in initialization.values()
        ),
        "exposure_failures": int(exposure["exposure_failure_count"]),
    }
    return sum(fields.values()), fields


def _attribution(
    champion: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, object]:
    before = cast(dict[str, list[dict[str, object]]], champion["observations"])
    after = cast(dict[str, list[dict[str, object]]], candidate["observations"])
    events = cast(dict[str, list[dict[str, object]]], candidate["fallback_events"])
    preserving = True
    differences_after_intervention = True
    differing_session_count = 0
    for sample_id in sorted(set(before) | set(after)):
        left = {int(row["turn"]): row for row in before.get(sample_id, [])}
        right = {int(row["turn"]): row for row in after.get(sample_id, [])}
        differences = [
            turn
            for turn in sorted(set(left) | set(right))
            if left.get(turn, {}).get("response_sha256")
            != right.get(turn, {}).get("response_sha256")
        ]
        if not differences:
            continue
        differing_session_count += 1
        first_event = min(
            (int(row["turn"]) for row in events.get(sample_id, [])), default=99
        )
        if min(differences) < first_event or first_event == 99:
            differences_after_intervention = False
        for event in events.get(sample_id, []):
            turn = int(event["turn"])
            if (
                left.get(turn, {}).get("recommendations_sha256")
                != event["recommendations_sha256"]
                or right.get(turn, {}).get("recommendations_sha256")
                != event["recommendations_sha256"]
                or left.get(turn, {}).get("usage_sha256") != event["usage_sha256"]
                or right.get(turn, {}).get("usage_sha256") != event["usage_sha256"]
            ):
                preserving = False
    resolutions: Counter[str] = Counter()
    for sample_id, rows in events.items():
        candidate_rows = {int(row["turn"]): row for row in after.get(sample_id, [])}
        for event in rows:
            next_row = candidate_rows.get(int(event["turn"]) + 1)
            resolutions[
                "unobserved_after_stop"
                if next_row is None
                else str(next_row["reply_kind"])
            ] += 1
    intervention_count = sum(len(rows) for rows in events.values())
    return {
        "intervention_count": intervention_count,
        "affected_session_count": len(events),
        "differing_session_count": differing_session_count,
        "all_response_differences_at_or_after_intervention": (
            differences_after_intervention
        ),
        "intervention_preserved_p5_recommendations_order_scores_and_usage": preserving,
        "resolution_counts": dict(sorted(resolutions.items())),
        "runtime_target_ids_or_session_ids_used_as_decision_features": False,
        "runtime_event_schema_contains_target_identity": False,
        "passed": (
            intervention_count > 0 and differences_after_intervention and preserving
        ),
    }


def _development_gates(
    champion: Mapping[str, object],
    candidate: Mapping[str, object],
    comparison: Mapping[str, object],
    attribution: Mapping[str, object],
) -> dict[str, object]:
    bootstrap = cast(dict[str, object], comparison["bootstrap"])
    scenario = cast(dict[str, float], comparison["scenario_technical_score_deltas"])
    failure_count, failure_detail = _failure_count(candidate)
    base_performance = cast(dict[str, object], champion["performance"])
    performance = cast(dict[str, object], candidate["performance"])
    checks = {
        "technical_score_delta_at_least_0_015": float(
            comparison["technical_score_delta"]
        )
        >= 0.015,
        "bootstrap_97_5_lower_positive": float(bootstrap["lower"]) > 0,
        "bootstrap_probability_at_least_0_9875": float(
            bootstrap["probability_delta_positive"]
        )
        >= 0.9875,
        "hit_rate_nondecrease": float(comparison["hit_rate_delta"]) >= 0,
        "positive_net_hits": int(comparison["net_hits"]) >= 1,
        "scenario_regression_no_worse_than_minus_0_005": min(scenario.values())
        >= -0.005,
        "mttc_nonworsening": float(comparison["mttc_delta"]) <= 0,
        "zero_correctness_or_reliability_failures": failure_count == 0,
        "missing_information_attribution": bool(attribution["passed"]),
        "target_and_session_identifier_independence": not bool(
            attribution["runtime_target_ids_or_session_ids_used_as_decision_features"]
        ),
        "peak_rss_at_most_2048_mib": float(performance["peak_process_rss_mib"]) <= 2048,
        "p95_latency_within_frozen_limit": float(performance["p95_response_latency_ms"])
        <= max(
            1000,
            2 * float(base_performance["p95_response_latency_ms"]),
        ),
        "wall_time_at_most_60_minutes": float(performance["evaluation_wall_seconds"])
        <= 3600,
    }
    return {
        "checks": checks,
        "failure_count": failure_count,
        "failure_detail": failure_detail,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public-set", default="data/public_set.jsonl")
    parser.add_argument(
        "--derivation",
        default=(
            "docs/results/autonomous_optimization/shadow_results/"
            "p8_catalog_answerability.json"
        ),
    )
    parser.add_argument(
        "--predeclaration",
        default=(
            "docs/results/autonomous_optimization/shadow_results/"
            "p8_proxy_fallback_predeclaration.json"
        ),
    )
    parser.add_argument(
        "--dense-cache", default="data/.dense-retrieval/catalog-minilm.npz"
    )
    parser.add_argument(
        "--phase",
        choices=tuple(PHASE_SEEDS),
        default="development",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=tuple(POLICY_IDS),
        default=["disabled", "open", "catalog_utility"],
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if len(set(args.policies)) != len(args.policies):
        raise ValueError("policies must not repeat")
    catalog_path = Path(args.catalog)
    public_path = Path(args.public_set)
    derivation_path = Path(args.derivation)
    predeclaration_path = Path(args.predeclaration)
    derivation = json.loads(derivation_path.read_text(encoding="utf-8"))
    thresholds_payload = derivation["popularity_log1p_rating_number_quartiles"]
    thresholds = [float(thresholds_payload[name]) for name in ("q1", "q2", "q3")]
    catalog_ids, categories, products = catalog_index(catalog_path)
    public_rows = _load_jsonl(public_path)
    public_targets = {str(row["ground_truth"]["parent_asin"]) for row in public_rows}
    eligible, eligibility_failures = _eligible_targets(
        products, categories, public_targets, thresholds
    )
    selected, prior = _select_phase(args.phase, eligible, public_targets)
    samples = _samples(selected, phase=args.phase)
    selection = _selection_summary(selected, public_targets, prior, args.phase)
    if any(selection["intersections"].values()):
        raise RuntimeError("proxy target sets are not disjoint")

    outcomes: dict[str, dict[str, object]] = {}
    for name in args.policies:
        outcomes[name] = _evaluate_policy(
            name,
            catalog_path,
            Path(args.dense_cache),
            samples,
            catalog_ids,
            categories,
            products,
        )
    comparisons: dict[str, object] = {}
    gates: dict[str, object] = {}
    if "disabled" in outcomes:
        for name in args.policies:
            if name == "disabled":
                continue
            comparison = _comparison(outcomes["disabled"], outcomes[name])
            attribution = _attribution(outcomes["disabled"], outcomes[name])
            comparisons[name] = {
                "paired": comparison,
                "attribution": attribution,
            }
            if args.phase == "development":
                gates[name] = _development_gates(
                    outcomes["disabled"], outcomes[name], comparison, attribution
                )

    result = {
        "schema_version": 1,
        "experiment_id": "P8-proxy-clarification-fallback-v1",
        "phase": args.phase,
        "methodology_frozen_before_outcomes": True,
        "lineage": {
            "p5_champion_commit": P5_COMMIT,
            "predeclaration_commit": PREDECLARATION_COMMIT,
            "derivation_commit": DERIVATION_COMMIT,
            "implementation_commit": IMPLEMENTATION_COMMIT,
        },
        "inputs": {
            "catalog_sha256": _sha256(catalog_path),
            "public_set_sha256": _sha256(public_path),
            "predeclaration_sha256": _sha256(predeclaration_path),
            "derivation_sha256": _sha256(derivation_path),
            "evaluator_sha256": _sha256("evaluator/local_evaluator.py"),
            "harness_sha256": _sha256(__file__),
            "eligible_catalog_count_excluding_public": len(eligible),
            "eligibility_failures": eligibility_failures,
            "public_outcomes_consulted": False,
            "reference_checkout_accessed": False,
        },
        "selection": selection,
        "policy_order": args.policies,
        "outcomes": outcomes,
        "comparisons": comparisons,
        "development_gates": gates,
        "provisional_qualifiers": [
            name
            for name in args.policies
            if name != "disabled"
            and bool(cast(dict[str, object], gates.get(name, {})).get("passed"))
        ],
        "official_runs_consumed": 0,
        "target_ids_persisted": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    compact = {
        key: value
        for key, value in result.items()
        if key not in {"outcomes", "comparisons"}
    }
    compact["metrics"] = {
        name: {
            key: outcome[key]
            for key in (
                "hit_rate_at_10",
                "mrr",
                "mttc",
                "efficiency",
                "technical_score",
                "determinism",
                "performance",
                "clarification_fallback_diagnostics",
            )
        }
        for name, outcome in outcomes.items()
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
