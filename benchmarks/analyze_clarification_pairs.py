"""Reconstruct paired Issue 5C clarification diagnostics without policy tuning.

This benchmark-only utility replays the immutable champion and exact Issue 5C
configuration to recover question-session and turn-level evidence omitted from
the stored aggregate artifacts. Public sample identifiers, scenario labels,
and targets remain in this scoring/analysis layer and never enter agent config.
"""

from __future__ import annotations

import argparse
import gc
import json
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import Agent
from starter.ambiguity_analysis import AmbiguityAnalyzer, ClarificationOpportunity
from starter.contextual_retrieval import policy_by_id
from starter.hybrid_retrieval import HybridRetrievalConfig, RetrievalMode
from starter.selective_clarification import SelectiveClarificationConfig

CHAMPION_POLICY_ID = "contextual.browsing-dense.v1"
ISSUE_5C_CONFIG = SelectiveClarificationConfig(enabled=True)


class RecordingAmbiguityAnalyzer:
    """Capture bounded opportunity statistics without affecting selection."""

    def __init__(self) -> None:
        self._delegate = AmbiguityAnalyzer()
        self.last_record: dict[str, object] | None = None

    def analyze(
        self,
        candidates: Iterable[object],
        catalog: Mapping[str, Mapping[object, object]],
        state: object | None = None,
    ) -> ClarificationOpportunity:
        pool = list(candidates)
        opportunity = self._delegate.analyze(pool, catalog, state)
        selected = next(
            (
                item
                for item in self._delegate.attribute_statistics(pool, catalog, state)
                if item.attribute == opportunity.attribute
            ),
            None,
        )
        selected_payload = asdict(selected) if selected is not None else None
        if selected_payload is not None:
            selected_payload.pop("value_counts", None)
        self.last_record = {
            "candidate_count": len(pool),
            "opportunity": asdict(opportunity),
            "selected_attribute_statistics": selected_payload,
        }
        return opportunity


def _outcome_for_reply(reply: str, *, interrupted_by_override: bool) -> str:
    if interrupted_by_override:
        return "unresolved_override"
    lowered = reply.lower()
    if "don't have" in lowered or "do not have" in lowered:
        return "decline"
    if "what matters is:" in lowered:
        return "explicit_answer"
    return "unresolved_follow_up"


def _trace_policy(
    *,
    catalog_path: Path,
    dense_cache_path: Path,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    clarification_config: SelectiveClarificationConfig,
) -> dict[str, dict[str, Any]]:
    recorder = RecordingAmbiguityAnalyzer()
    agent = Agent(
        catalog_path,
        config=HybridRetrievalConfig(mode=RetrievalMode.CONTEXTUAL),
        dense_cache_path=dense_cache_path,
        contextual_policy=policy_by_id(CHAMPION_POLICY_ID),
        clarification_config=clarification_config,
        ambiguity_analyzer=recorder,
    )
    traces: dict[str, dict[str, Any]] = {}
    for sample in samples:
        runtime_session_id = f"diagnostic_{uuid.uuid4().hex}"
        agent.reset(runtime_session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        intent_card, behavior = materialize_hidden_fields(sample, products)
        effective_sample = {**sample, "intent_card": intent_card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(
            effective_sample,
            coarse_category(categories.get(target, [])),
            disclosed,
        )
        first_hit_turn: int | None = None
        best_rank: int | None = None
        turns: list[dict[str, object]] = []
        question_events: list[dict[str, object]] = []
        pending_question_event: dict[str, object] | None = None
        for turn in range(1, MAX_TURNS + 1):
            recorder.last_record = None
            response = agent.respond(runtime_session_id, user_message, turn, TOP_K)
            if pending_question_event is not None:
                controller = vars(agent)["_clarification_controller"]
                state = controller.state_for(runtime_session_id)
                attribute = str(pending_question_event["attribute"])
                if state is not None and attribute in state.answered_attributes:
                    runtime_resolution = "answered"
                elif state is not None and attribute in state.declined_attributes:
                    runtime_resolution = "declined"
                else:
                    runtime_resolution = "unresolved"
                pending_question_event["runtime_resolution_after_follow_up"] = (
                    runtime_resolution
                )
                pending_question_event = None
            recommendations = normalize_recommendations(
                response.get("recommendations"), catalog_ids
            )
            route_map = vars(agent)["_contextual_routes"]
            route = str(route_map.get(runtime_session_id, "uncertain"))
            target_rank = (
                recommendations.index(target) + 1
                if override_applied and target in recommendations
                else None
            )
            turn_record: dict[str, object] = {
                "turn": turn,
                "user_message": user_message,
                "observable_route": route,
                "ask_attribute": response.get("ask_attribute"),
                "recommendations": recommendations,
                "target_rank": target_rank,
            }
            if response.get("ask_attribute") is not None:
                turn_record["ambiguity"] = recorder.last_record
                question_events.append(
                    {
                        "turn": turn,
                        "observable_route": route,
                        "attribute": response.get("ask_attribute"),
                        "ambiguity": recorder.last_record,
                        "evaluator_follow_up": "not_observed_session_ended",
                        "runtime_resolution_after_follow_up": (
                            "not_observed_session_ended"
                        ),
                    }
                )
            turns.append(turn_record)
            if target_rank is not None:
                first_hit_turn = turn
                best_rank = target_rank
                break
            if turn == MAX_TURNS:
                break
            override = effective_sample.get("behavior", {}).get("override") or {}
            interrupted_by_override = not override_applied and turn + 1 == int(
                override.get("turn", 3)
            )
            if interrupted_by_override:
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                next_user_message = str(
                    override.get(
                        "message", "Actually, please ignore my earlier preference."
                    )
                )
            else:
                next_user_message, boundary_used = customer_reply(
                    effective_sample,
                    response.get("ask_attribute"),
                    disclosed,
                    boundary_used,
                )
            if response.get("ask_attribute") is not None:
                question_events[-1]["evaluator_follow_up"] = _outcome_for_reply(
                    next_user_message,
                    interrupted_by_override=interrupted_by_override,
                )
                pending_question_event = question_events[-1]
            user_message = next_user_message
        traces[str(sample["sample_id"])] = {
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "hit": first_hit_turn is not None,
            "first_hit_turn": first_hit_turn,
            "best_rank": best_rank,
            "turns": turns,
            "questions": question_events,
        }
    return traces


def _comparison_label(before: Mapping[str, Any], after: Mapping[str, Any]) -> str:
    before_hit = bool(before["hit"])
    after_hit = bool(after["hit"])
    if before_hit != after_hit:
        return "improved" if after_hit else "regressed"
    if not before_hit:
        return "unchanged"
    before_turn = int(before["first_hit_turn"])
    after_turn = int(after["first_hit_turn"])
    if before_turn != after_turn:
        return "improved" if after_turn < before_turn else "regressed"
    before_rank = int(before["best_rank"])
    after_rank = int(after["best_rank"])
    if before_rank != after_rank:
        return "improved" if after_rank < before_rank else "regressed"
    return "unchanged"


def _change_flags(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, bool]:
    before_hit = bool(before["hit"])
    after_hit = bool(after["hit"])
    shared_hit = before_hit and after_hit
    return {
        "new_hit": after_hit and not before_hit,
        "lost_hit": before_hit and not after_hit,
        "earlier_hit": shared_hit
        and int(after["first_hit_turn"]) < int(before["first_hit_turn"]),
        "later_hit": shared_hit
        and int(after["first_hit_turn"]) > int(before["first_hit_turn"]),
        "better_rank": shared_hit
        and int(after["first_hit_turn"]) == int(before["first_hit_turn"])
        and int(after["best_rank"]) < int(before["best_rank"]),
        "worse_rank": shared_hit
        and int(after["first_hit_turn"]) == int(before["first_hit_turn"])
        and int(after["best_rank"]) > int(before["best_rank"]),
    }


def _group_summary(rows: list[dict[str, Any]], key: str) -> dict[str, object]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    result: dict[str, object] = {}
    for name, items in sorted(grouped.items()):
        labels = Counter(str(item["comparison"]) for item in items)
        flags = Counter(
            flag
            for item in items
            for flag, present in dict(item["flags"]).items()
            if present
        )
        result[name] = {
            "questioned_sessions": len(items),
            "improved": labels["improved"],
            "regressed": labels["regressed"],
            "unchanged": labels["unchanged"],
            **{flag: flags[flag] for flag in sorted(flags)},
        }
    return result


def paired_diagnosis(
    champion: Mapping[str, Mapping[str, Any]],
    issue_5c: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(issue_5c):
        after = issue_5c[sample_id]
        questions = list(after["questions"])  # type: ignore[arg-type]
        if not questions:
            continue
        before = champion[sample_id]
        question = dict(questions[0])
        question_turn = int(question["turn"])
        later_changes: list[dict[str, object]] = []
        before_turns = {
            int(item["turn"]): item
            for item in before["turns"]  # type: ignore[index]
        }
        for after_turn in after["turns"]:  # type: ignore[index]
            turn = int(after_turn["turn"])
            if turn <= question_turn or turn not in before_turns:
                continue
            champion_recommendations = before_turns[turn]["recommendations"]
            issue_5c_recommendations = after_turn["recommendations"]
            if champion_recommendations != issue_5c_recommendations:
                champion_set = set(champion_recommendations)
                issue_5c_set = set(issue_5c_recommendations)
                later_changes.append(
                    {
                        "turn": turn,
                        "same_recommendation_set": champion_set == issue_5c_set,
                        "shared_recommendation_count": len(champion_set & issue_5c_set),
                        "champion_target_rank": before_turns[turn]["target_rank"],
                        "issue_5c_target_rank": after_turn["target_rank"],
                    }
                )
        session_flags = _change_flags(before, after)
        rows.append(
            {
                "sample_id": sample_id,
                "scenario_type": after["scenario_type"],
                "observable_route": question["observable_route"],
                "attribute": question["attribute"],
                "question_turn": question_turn,
                "evaluator_follow_up": question["evaluator_follow_up"],
                "runtime_resolution_after_follow_up": question[
                    "runtime_resolution_after_follow_up"
                ],
                "ambiguity": question["ambiguity"],
                "comparison": _comparison_label(before, after),
                "flags": session_flags,
                "champion": {
                    "hit": before["hit"],
                    "first_hit_turn": before["first_hit_turn"],
                    "best_rank": before["best_rank"],
                },
                "issue_5c": {
                    "hit": after["hit"],
                    "first_hit_turn": after["first_hit_turn"],
                    "best_rank": after["best_rank"],
                },
                "later_recommendation_changes": later_changes,
                "mechanism_trace": (
                    {
                        "champion_turns_after_question": [
                            {
                                "turn": item["turn"],
                                "user_message": item["user_message"],
                                "target_rank": item["target_rank"],
                            }
                            for item in before["turns"]  # type: ignore[index]
                            if int(item["turn"]) > question_turn
                        ],
                        "issue_5c_turns_after_question": [
                            {
                                "turn": item["turn"],
                                "user_message": item["user_message"],
                                "target_rank": item["target_rank"],
                            }
                            for item in after["turns"]  # type: ignore[index]
                            if int(item["turn"]) > question_turn
                        ],
                    }
                    if session_flags["new_hit"] or session_flags["lost_hit"]
                    else None
                ),
            }
        )

    labels = Counter(str(row["comparison"]) for row in rows)
    change_counts = Counter(
        flag for row in rows for flag, present in dict(row["flags"]).items() if present
    )
    evaluator_follow_ups = Counter(str(row["evaluator_follow_up"]) for row in rows)
    runtime_resolutions = Counter(
        str(row["runtime_resolution_after_follow_up"]) for row in rows
    )
    question_turns = Counter(str(row["question_turn"]) for row in rows)
    later_change_sessions = sum(
        bool(row["later_recommendation_changes"]) for row in rows
    )
    return {
        "questioned_session_count": len(rows),
        "improved_session_count": labels["improved"],
        "regressed_session_count": labels["regressed"],
        "unchanged_session_count": labels["unchanged"],
        "change_counts": {flag: change_counts[flag] for flag in sorted(change_counts)},
        "question_turn_counts": dict(
            sorted(question_turns.items(), key=lambda item: int(item[0]))
        ),
        "evaluator_follow_up_counts": dict(sorted(evaluator_follow_ups.items())),
        "runtime_resolution_counts": dict(sorted(runtime_resolutions.items())),
        "sessions_with_later_recommendation_changes": later_change_sessions,
        "by_observable_route": _group_summary(rows, "observable_route"),
        "by_attribute": _group_summary(rows, "attribute"),
        "by_scenario": _group_summary(rows, "scenario_type"),
        "questioned_sessions": rows,
    }


def _session_summaries(
    traces: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, object]]:
    return [
        {
            "sample_id": sample_id,
            "scenario_type": trace["scenario_type"],
            "hit": trace["hit"],
            "first_hit_turn": trace["first_hit_turn"],
            "best_rank": trace["best_rank"],
        }
        for sample_id, trace in sorted(traces.items())
    ]


def _validate_replay(
    champion: Mapping[str, Mapping[str, Any]],
    issue_5c: Mapping[str, Mapping[str, Any]],
    stored_issue_5c: Mapping[str, Any],
) -> dict[str, object]:
    stored_sessions = {
        str(item["sample_id"]): item for item in stored_issue_5c["sessions"]
    }
    replay_summaries = {
        str(item["sample_id"]): item for item in _session_summaries(issue_5c)
    }
    mismatches = {
        sample_id: {"stored": stored_sessions[sample_id], "replay": replay}
        for sample_id, replay in replay_summaries.items()
        if any(
            stored_sessions[sample_id][key] != replay[key]
            for key in ("hit", "first_hit_turn", "best_rank")
        )
    }
    replay_hit_ids = sorted(
        sample_id for sample_id, item in champion.items() if item["hit"]
    )
    expected_champion_hits = [
        "public_0010",
        "public_0044",
        "public_0046",
        "public_0061",
        "public_0067",
        "public_0070",
        "public_0081",
        "public_0082",
        "public_0085",
        "public_0088",
        "public_0090",
        "public_0107",
        "public_0119",
        "public_0129",
        "public_0134",
        "public_0137",
        "public_0142",
        "public_0143",
        "public_0148",
        "public_0155",
        "public_0156",
        "public_0160",
        "public_0166",
        "public_0168",
        "public_0185",
        "public_0190",
        "public_0193",
        "public_0197",
    ]
    question_count = sum(len(item["questions"]) for item in issue_5c.values())
    return {
        "issue_5c_session_metric_mismatch_count": len(mismatches),
        "issue_5c_session_metric_mismatches": mismatches,
        "issue_5c_expected_question_count": 38,
        "issue_5c_replayed_question_count": question_count,
        "champion_hit_ids_match_stored": replay_hit_ids == expected_champion_hits,
        "champion_replayed_hit_ids": replay_hit_ids,
        "passed": not mismatches
        and question_count == 38
        and replay_hit_ids == expected_champion_hits,
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
        "--stored-issue-5c",
        type=Path,
        default=Path(
            "diagnostics/retrieval_regression/issue_5c_clarification_enabled_200.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/results/issue_6a/paired_diagnosis.json"),
    )
    args = parser.parse_args()
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    champion = _trace_policy(
        catalog_path=args.catalog,
        dense_cache_path=args.dense_cache,
        samples=samples,
        catalog_ids=catalog_ids,
        categories=categories,
        products=products,
        clarification_config=SelectiveClarificationConfig(enabled=False),
    )
    gc.collect()
    issue_5c = _trace_policy(
        catalog_path=args.catalog,
        dense_cache_path=args.dense_cache,
        samples=samples,
        catalog_ids=catalog_ids,
        categories=categories,
        products=products,
        clarification_config=ISSUE_5C_CONFIG,
    )
    stored_issue_5c = json.loads(args.stored_issue_5c.read_text(encoding="utf-8"))
    validation = _validate_replay(champion, issue_5c, stored_issue_5c)
    if not validation["passed"]:
        raise SystemExit(json.dumps(validation, indent=2))
    output: dict[str, Any] = {
        "schema_version": 1,
        "issue": "6A",
        "analysis_kind": "benchmark-only deterministic reconstruction of stored policies",
        "runtime_leakage": False,
        "champion_policy_id": CHAMPION_POLICY_ID,
        "issue_5c_configuration": asdict(ISSUE_5C_CONFIG),
        "validation": validation,
        "diagnosis": paired_diagnosis(champion, issue_5c),
        "reproduction_command": (
            "python3 -m benchmarks.analyze_clarification_pairs "
            "--output docs/results/issue_6a/paired_diagnosis.json"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    diagnosis = dict(output["diagnosis"])
    diagnosis.pop("questioned_sessions", None)
    print(json.dumps({"output": str(args.output), **diagnosis}, indent=2))


if __name__ == "__main__":
    main()
