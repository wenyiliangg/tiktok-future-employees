"""Observational P2 residual-cause diagnosis under the frozen P4A contract.

This benchmark consults shadow targets only to label offline causes. It does not
alter Agent responses, runtime policy, the public set, or any evaluator code.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from starter.agent import NEGATIVE_FEEDBACK_RE, Agent
from starter.category_evidence import DIALOGUE_TOKENS, _visible_text
from starter.clarification_policies import clarification_policy_by_id
from starter.contextual_retrieval import rank_contextual_candidates
from starter.conversation_state import SearchQuery
from starter.hybrid_retrieval import RankedResult
from starter.lexical_retriever import tokenize

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
    classify_constraint,
    customer_reply,
    evaluate_shadow,
    initial_message,
    load_jsonl,
    metric_summary,
    public_targets,
)

P2_CONTEXTUAL_POLICY = "contextual.category-evidence.v1"
P2_CLARIFICATION_POLICY = "clarification.category-evidence-utility.v1"
EXPECTED_P2_TRANSCRIPT_SHA256 = (
    "765c2e05e4bf8dc6f87ab5f099d5ce1144dc1405a39850151e3bff8e60c467f4"
)
STRUCTURED_KINDS = frozenset({"budget", "material", "color", "style", "use_case"})
PRICE_RE = re.compile(r"\$\s*(\d+(?:\.\d+)?)")


def _p2_agent(catalog: Path, dense_cache: Path) -> Agent:
    policy = clarification_policy_by_id(P2_CLARIFICATION_POLICY)
    return _agent(
        catalog,
        dense_cache,
        policy.clarification,
        policy.controller,
        P2_CONTEXTUAL_POLICY,
    )


def _ids(values: Sequence[object]) -> list[str]:
    result: list[str] = []
    for value in values:
        parent_asin = getattr(value, "parent_asin", None)
        if isinstance(parent_asin, str) and parent_asin not in result:
            result.append(parent_asin)
    return result


def _state_snapshot(agent: Agent, session_id: str) -> dict[str, object]:
    state = agent._state.state_for(session_id)
    result: dict[str, object] = {}
    for name in ("category", "color", "style", "material", "use_case"):
        value = getattr(state, name, None)
        result[name] = None if value is None else str(value.value)
    price = state.price
    result["price"] = (
        None if price is None else {"minimum": price.minimum, "maximum": price.maximum}
    )
    result["removed_constraints"] = sorted(state.removed_constraints)
    return result


def _query_snapshot(agent: Agent, session_id: str) -> SearchQuery:
    state = agent._state.state_for(session_id)
    return SearchQuery(
        text=state.raw_current_turn_text,
        category=state.category,
        color=state.color,
        style=state.style,
        material=state.material,
        use_case=state.use_case,
        price=state.price,
        exclusions=copy.deepcopy(state.exclusions),
    )


def _state_represents(constraint: str, state: Mapping[str, object]) -> bool:
    kind = classify_constraint(constraint)
    if kind == "budget":
        match = PRICE_RE.search(constraint)
        price = state.get("price")
        if match is None or not isinstance(price, Mapping):
            return False
        expected = float(match.group(1))
        minimum = price.get("minimum")
        maximum = price.get("maximum")
        return (
            isinstance(minimum, (int, float))
            and not isinstance(minimum, bool)
            and float(minimum) <= expected
        ) or (
            isinstance(maximum, (int, float))
            and not isinstance(maximum, bool)
            and float(maximum) >= expected
        )
    if kind not in STRUCTURED_KINDS:
        return True
    raw = state.get(kind)
    if not isinstance(raw, str):
        return False
    expected_tokens = set(tokenize(constraint)) - DIALOGUE_TOKENS
    state_tokens = set(tokenize(raw))
    return bool(expected_tokens & state_tokens)


def _product_satisfies(constraint: str, product: Mapping[str, object]) -> bool:
    if classify_constraint(constraint) == "budget":
        match = PRICE_RE.search(constraint)
        if match is None:
            return False
        expected = float(match.group(1))
        try:
            actual = float(cast(Any, product.get("price")))
        except (TypeError, ValueError):
            return False
        return abs(actual - expected) <= 0.011
    required = set(tokenize(constraint)) - DIALOGUE_TOKENS
    visible = set(tokenize(_visible_text(product)))
    return bool(required) and required <= visible


def _coverage(
    parent_asin: str,
    disclosed: Sequence[str],
    products: Mapping[str, Mapping[str, object]],
) -> int:
    product = products.get(parent_asin)
    if product is None:
        return 0
    return sum(_product_satisfies(value, product) for value in disclosed)


def _filtered_pool(
    agent: Agent,
    query: SearchQuery,
    session_id: str,
    anchor: Sequence[RankedResult],
    history: Sequence[RankedResult],
) -> tuple[set[str], dict[str, int]]:
    """Mirror only category-evidence candidate eligibility for observation."""

    index = agent._category_evidence_index
    if index is None:
        return set(), {}
    current_text = agent._evidence_messages.current_text(
        session_id
    ) or agent._active_raw_intent.get(session_id, "")
    historical_text = agent._evidence_messages.historical_text(
        session_id
    ) or agent._historical_raw_evidence.get(session_id, "")
    category = index.extract_category(current_text)
    category_tokens = frozenset(tokenize(category or ""))
    current_phrases = index.matching_phrases(current_text)
    history_phrases = index.matching_phrases(historical_text) if historical_text else ()
    phrase_support, _ = index._phrase_support(current_phrases)
    history_phrase_support, _ = index._phrase_support(history_phrases)
    rare_support, _ = index._token_support(current_text, category_tokens)
    history_rare_support, _ = index._token_support(historical_text, category_tokens)
    protected_ids = {item.parent_asin for item in anchor} | {
        item.parent_asin for item in history
    }
    candidate_docs = {
        index.id_to_doc[parent_asin]
        for parent_asin in protected_ids
        if parent_asin in index.id_to_doc
    }
    candidate_docs.update(set(phrase_support) | set(history_phrase_support))
    rare_order = sorted(
        set(rare_support) | set(history_rare_support),
        key=lambda doc: (
            -(
                rare_support.get(doc, 0.0)
                + index.policy.history_weight * history_rare_support.get(doc, 0.0)
            ),
            index.ids[doc],
        ),
    )
    remaining = max(0, index.policy.total_candidate_limit - len(candidate_docs))
    candidate_docs.update(rare_order[:remaining])
    if category:
        remaining = max(0, index.policy.total_candidate_limit - len(candidate_docs))
        candidate_docs.update(
            index._category_order.get(category, ())[
                : min(index.policy.category_candidate_limit, remaining)
            ]
        )
    if not candidate_docs:
        candidate_docs.update(
            range(min(index.count, index.policy.category_candidate_limit))
        )
    known_negative = agent._known_negative_ids.get(session_id, set())
    constraints, exclusions = index._structured_evidence(query)
    identifiers: set[str] = set()
    contradictions: dict[str, int] = {}
    for doc_id in candidate_docs:
        parent_asin = index.ids[doc_id]
        if parent_asin in known_negative:
            continue
        identifiers.add(parent_asin)
        _support, _checked, count = index._structured_support(
            query, doc_id, constraints, exclusions
        )
        contradictions[parent_asin] = count
    return identifiers, contradictions


def _turn_trace(
    agent: Agent,
    session_id: str,
    query: SearchQuery,
    response: Mapping[str, object],
    disclosed: set[str],
    products: Mapping[str, Mapping[str, object]],
    target: str,
    turn: int,
) -> dict[str, object]:
    policy = agent._contextual_policy
    raw_text = agent._raw_turn_text(session_id, query)
    active_text = agent._active_raw_intent.get(session_id) or raw_text
    anchor_text = (
        active_text
        if policy.negative_feedback_uses_active_intent
        and NEGATIVE_FEEDBACK_RE.search(raw_text)
        else raw_text
    )
    anchor = list(agent._anchor.retrieve(anchor_text, top_n=policy.candidate_count))  # type: ignore[union-attr]
    history_text = agent._historical_raw_evidence.get(session_id, "")
    history: list[RankedResult] = []
    if agent._override_history_policy.enabled and history_text:
        history = list(
            agent._anchor.retrieve(history_text, top_n=policy.candidate_count)  # type: ignore[union-attr]
        )
    fused = rank_contextual_candidates(
        anchor,
        history,
        (),
        agent._catalog_ids,
        agent._known_negative_ids.get(session_id, set()),
        policy,
        limit=50,
    )
    filtered, contradictions = _filtered_pool(agent, query, session_id, anchor, history)
    reranked = copy.deepcopy(agent._clarification_candidates.get(session_id, []))
    reranked_ids = _ids(reranked)
    published = _ranked(response, frozenset(products))
    useful = [
        value
        for value in sorted(disclosed)
        if _product_satisfies(value, products[target])
        and any(
            not _product_satisfies(value, products[item])
            for item in (reranked_ids[:10] or published)
            if item in products and item != target
        )
    ]
    coverage = {
        parent_asin: _coverage(parent_asin, useful, products)
        for parent_asin in reranked_ids
    }
    target_candidate = next(
        (item for item in reranked if item.parent_asin == target), None
    )
    return {
        "turn": turn,
        "observable_route": agent._contextual_routes.get(session_id, "uncertain"),
        "ask_attribute": response.get("ask_attribute"),
        "state": _state_snapshot(agent, session_id),
        "disclosed": sorted(disclosed),
        "useful_disclosed": useful,
        "pool_positions": {
            "lexical": (_ids(anchor).index(target) + 1)
            if target in _ids(anchor)
            else None,
            "dense": "not_executed",
            "fused": (_ids(fused).index(target) + 1) if target in _ids(fused) else None,
            "filtered": 1 if target in filtered else None,
            "reranked": reranked_ids.index(target) + 1
            if target in reranked_ids
            else None,
            "published": published.index(target) + 1 if target in published else None,
        },
        "target_contradictions": contradictions.get(target),
        "target_component_scores": (
            {}
            if target_candidate is None
            else dict(sorted(target_candidate.component_scores.items()))
        ),
        "reranked_component_scores": {
            item.parent_asin: dict(sorted(item.component_scores.items()))
            for item in reranked
        },
        "reranked_ids": reranked_ids,
        "coverage": coverage,
    }


def _replay_with_traces(
    agent: Agent,
    samples: Sequence[ShadowSample],
    products: Mapping[str, Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int], str]:
    sessions: list[dict[str, object]] = []
    correctness: Counter[str] = Counter()
    transcript_hasher = hashlib.sha256()
    for sample in samples:
        session_id = f"{sample.sample_id}_{sample.template_variant}"
        agent.reset(session_id, _profile(sample))
        disclosed: set[str] = set()
        disclosed_turn: dict[str, int] = {}
        boundary_declined = False
        override_applied = sample.scenario_type != "intent_override"
        user_message = initial_message(sample, disclosed)
        for value in disclosed:
            disclosed_turn[value] = 1
        first_hit_turn: int | None = None
        best_rank: int | None = None
        traces: list[dict[str, object]] = []
        for turn in range(1, MAX_TURNS + 1):
            response = agent.respond(session_id, user_message, turn, TOP_K)
            correctness.update(_contract_counts(response, frozenset(products)))
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
            query = _query_snapshot(agent, session_id)
            traces.append(
                _turn_trace(
                    agent,
                    session_id,
                    query,
                    response,
                    disclosed,
                    products,
                    sample.target,
                    turn,
                )
            )
            ranked = _ranked(response, frozenset(products))
            if override_applied and sample.target in ranked:
                first_hit_turn = turn
                best_rank = ranked.index(sample.target) + 1
                break
            if turn == MAX_TURNS:
                break
            ask_attribute = (
                response.get("ask_attribute") if isinstance(response, Mapping) else None
            )
            next_turn = turn + 1
            before = set(disclosed)
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
            for value in disclosed - before:
                disclosed_turn[value] = next_turn
        sessions.append(
            {
                "sample_id": sample.sample_id,
                "scenario_type": sample.scenario_type,
                "hit": first_hit_turn is not None,
                "first_hit_turn": first_hit_turn,
                "best_rank": best_rank,
                "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
                "override_turn": sample.override_turn,
                "constraints": list(sample.constraints),
                "disclosed_turn": disclosed_turn,
                "traces": traces,
            }
        )
    return sessions, dict(correctness), transcript_hasher.hexdigest()


def _outcome_contribution(hit: bool, turn: int | None, rank: int | None) -> float:
    if not hit or turn is None or rank is None:
        return 0.0
    return 0.5 + 0.3 / rank + 0.2 * ((11 - turn) / 10)


def _observed_outcome(
    session: Mapping[str, object],
) -> tuple[bool, int | None, int | None]:
    return (
        bool(session["hit"]),
        cast(int | None, session["first_hit_turn"]),
        cast(int | None, session["best_rank"]),
    )


def _classify(session: Mapping[str, object]) -> tuple[str, int | None, int | None]:
    traces = cast(list[dict[str, object]], session["traces"])
    scenario = str(session["scenario_type"])
    override_turn = cast(int | None, session["override_turn"])
    disclosed_turn = cast(dict[str, int], session["disclosed_turn"])
    causal_traces = (
        [
            item
            for item in traces
            if override_turn is not None and cast(int, item["turn"]) >= override_turn
        ]
        if scenario == "intent_override"
        else traces
    )

    if scenario == "intent_override" and override_turn is not None:
        override_trace = next(
            (item for item in traces if item["turn"] == override_turn), None
        )
        later = [item for item in traces if cast(int, item["turn"]) > override_turn]
        if override_trace is not None:
            override_pos = cast(dict[str, object], override_trace["pool_positions"])
            if override_pos["reranked"] is None and any(
                cast(dict[str, object], item["pool_positions"])["reranked"] is not None
                for item in later
            ):
                recovery = min(
                    cast(int, item["turn"])
                    for item in later
                    if cast(dict[str, object], item["pool_positions"])["reranked"]
                    is not None
                )
                return "G", override_turn, recovery

    terminal = causal_traces[-1]
    useful_terminal = cast(list[str], terminal["useful_disclosed"])
    state = cast(dict[str, object], terminal["state"])
    missing_state = [
        value
        for value in useful_terminal
        if classify_constraint(value) in STRUCTURED_KINDS
        and not _state_represents(value, state)
    ]
    if missing_state:
        recovery = min(
            disclosed_turn.get(value, cast(int, terminal["turn"]))
            for value in missing_state
        )
        return "A", recovery, None

    evidence_traces = [item for item in causal_traces if item["useful_disclosed"]]
    if evidence_traces and all(
        all(
            cast(dict[str, object], item["pool_positions"])[stage] is None
            for stage in ("lexical", "fused", "filtered", "reranked")
        )
        for item in evidence_traces
    ):
        earliest = min(cast(int, item["turn"]) for item in evidence_traces)
        return "B", earliest, None

    for item in causal_traces:
        pools = cast(dict[str, object], item["pool_positions"])
        if (pools["lexical"] is not None or pools["fused"] is not None) and pools[
            "reranked"
        ] is None:
            return "C", cast(int, item["turn"]), None
        contradictions = item["target_contradictions"]
        if isinstance(contradictions, int) and contradictions > 0:
            return "C", cast(int, item["turn"]), None

    for item in causal_traces:
        pools = cast(dict[str, object], item["pool_positions"])
        reranked_position = pools["reranked"]
        if not isinstance(reranked_position, int):
            continue
        ids = cast(list[str], item["reranked_ids"])
        coverage = cast(dict[str, int], item["coverage"])
        target_coverage = coverage.get(ids[reranked_position - 1], 0)
        above = ids[: reranked_position - 1]
        if any(coverage.get(parent_asin, 0) < target_coverage for parent_asin in above):
            conservative_rank = min(
                10,
                1
                + sum(
                    coverage.get(parent_asin, 0) >= target_coverage
                    for parent_asin in above
                ),
            )
            return "D", cast(int, item["turn"]), conservative_rank

    for item in causal_traces:
        pools = cast(dict[str, object], item["pool_positions"])
        reranked_position = pools["reranked"]
        if not isinstance(reranked_position, int):
            continue
        ids = cast(list[str], item["reranked_ids"])
        coverage = cast(dict[str, int], item["coverage"])
        target_id = ids[reranked_position - 1]
        target_coverage = coverage.get(target_id, 0)
        above = ids[: reranked_position - 1]
        if all(
            coverage.get(parent_asin, 0) >= target_coverage for parent_asin in above
        ):
            scores = cast(dict[str, float], item["target_component_scores"])
            popularity = float(scores.get("popularity", 0.0))
            component_scores = cast(
                dict[str, dict[str, float]], item["reranked_component_scores"]
            )
            popularity_rank = 1 + sum(
                float(component_scores.get(parent_asin, {}).get("popularity", 0.0))
                > popularity
                for parent_asin in above
            )
            return "E", cast(int, item["turn"]), min(10, popularity_rank)

    disclosed = set(disclosed_turn)
    hidden = [
        value
        for value in cast(list[str], session["constraints"])
        if value not in disclosed
    ]
    if hidden:
        return "F", min(MAX_TURNS, max(2, len(traces) + 1)), None
    return "H", None, None


def _counterfactual_f_rank(
    sample: ShadowSample,
    products: Mapping[str, Mapping[str, object]],
    agent: Agent,
) -> tuple[int | None, int | None]:
    disclosed: set[str] = set()
    first_message = initial_message(sample, disclosed)
    hidden = [value for value in sample.constraints if value not in disclosed]
    if not hidden:
        return None, None
    session_id = f"p4_f_counterfactual_{sample.sample_id}"
    agent.reset(session_id, _profile(sample))
    agent.respond(session_id, first_message, 1, TOP_K)
    response = agent.respond(
        session_id, f"The deciding details are {hidden[0]}.", 2, TOP_K
    )
    ranked = _ranked(response, frozenset(products))
    result = (
        (2, ranked.index(sample.target) + 1)
        if sample.target in ranked
        else (None, None)
    )
    return result


def _diagnose(
    sessions: Sequence[dict[str, object]],
    samples: Sequence[ShadowSample],
    products: Mapping[str, Mapping[str, object]],
    catalog: Path,
    dense_cache: Path,
) -> dict[str, object]:
    samples_by_id = {sample.sample_id: sample for sample in samples}
    counterfactual_agent: Agent | None = None
    residual = [
        session
        for session in sessions
        if not bool(session["hit"]) or cast(int, session["best_rank"]) > 1
    ]
    rows: list[dict[str, object]] = []
    for session in residual:
        category, recovery_turn, conservative_rank = _classify(session)
        if category == "F":
            if counterfactual_agent is None:
                counterfactual_agent = _p2_agent(catalog, dense_cache)
            recovery_turn, conservative_rank = _counterfactual_f_rank(
                samples_by_id[str(session["sample_id"])],
                products,
                counterfactual_agent,
            )
        observed = _observed_outcome(session)
        oracle_turn = recovery_turn
        oracle_rank = 1 if oracle_turn is not None and category != "H" else None
        conservative_turn = recovery_turn
        if category == "B" and recovery_turn is not None:
            conservative_turn = min(MAX_TURNS, recovery_turn + 1)
            conservative_rank = 10
        elif category == "C" and recovery_turn is not None:
            trace = next(
                item
                for item in cast(list[dict[str, object]], session["traces"])
                if item["turn"] == recovery_turn
            )
            pools = cast(dict[str, object], trace["pool_positions"])
            raw_rank = pools["fused"] or pools["lexical"]
            if isinstance(raw_rank, int):
                conservative_rank = min(10, raw_rank)
                conservative_turn = min(MAX_TURNS, recovery_turn + int(raw_rank > 10))
        elif category == "A" and conservative_rank is None:
            best_observed = [
                cast(dict[str, object], item["pool_positions"])["reranked"]
                for item in cast(list[dict[str, object]], session["traces"])
                if cast(int, item["turn"]) >= cast(int, recovery_turn or 1)
                and isinstance(
                    cast(dict[str, object], item["pool_positions"])["reranked"], int
                )
            ]
            conservative_rank = min(10, min(cast(list[int], best_observed), default=10))
            if not best_observed and recovery_turn is not None:
                conservative_turn = min(MAX_TURNS, recovery_turn + 1)
        elif (
            category == "G" and conservative_rank is None and recovery_turn is not None
        ):
            trace = next(
                item
                for item in cast(list[dict[str, object]], session["traces"])
                if item["turn"] == recovery_turn
            )
            rank = cast(dict[str, object], trace["pool_positions"])["reranked"]
            conservative_rank = (
                min(10, cast(int, rank)) if isinstance(rank, int) else None
            )
            conservative_turn = max(
                cast(int, session["override_turn"]), recovery_turn - 1
            )

        observed_contribution = _outcome_contribution(*observed)
        oracle_contribution = _outcome_contribution(
            oracle_rank is not None, oracle_turn, oracle_rank
        )
        conservative_contribution = _outcome_contribution(
            conservative_rank is not None, conservative_turn, conservative_rank
        )
        impact_traces = cast(list[dict[str, object]], session["traces"])
        if session["scenario_type"] == "intent_override":
            impact_traces = [
                item
                for item in impact_traces
                if cast(int, item["turn"]) >= cast(int, session["override_turn"])
            ]
        rows.append(
            {
                "sample_id": session["sample_id"],
                "scenario_type": session["scenario_type"],
                "observed": {
                    "hit": observed[0],
                    "turn": observed[1],
                    "rank": observed[2],
                },
                "category": category,
                "earliest_recovery_turn": recovery_turn,
                "oracle_rank": oracle_rank,
                "conservative_turn": conservative_turn,
                "conservative_rank": conservative_rank,
                "oracle_gain_contribution": max(
                    0.0, oracle_contribution - observed_contribution
                ),
                "conservative_gain_contribution": max(
                    0.0, conservative_contribution - observed_contribution
                ),
                "pool_presence": {
                    stage: sum(
                        cast(dict[str, object], item["pool_positions"])[stage]
                        is not None
                        and cast(dict[str, object], item["pool_positions"])[stage]
                        != "not_executed"
                        for item in impact_traces
                    )
                    for stage in (
                        "lexical",
                        "dense",
                        "fused",
                        "filtered",
                        "reranked",
                        "published",
                    )
                },
                "question_count": sum(
                    isinstance(item["ask_attribute"], str) for item in impact_traces
                ),
                "question_attributes": [
                    item["ask_attribute"]
                    for item in impact_traces
                    if isinstance(item["ask_attribute"], str)
                ],
                "observable_routes": sorted(
                    {str(item["observable_route"]) for item in impact_traces}
                ),
            }
        )

    if counterfactual_agent is not None:
        del counterfactual_agent
        gc.collect()
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["category"])].append(row)
    summaries: dict[str, object] = {}
    patterns = {
        "A": "A disclosed structured value is absent or stale in active state.",
        "B": "Useful disclosed catalog text does not admit the target to any bounded generation pool.",
        "C": "The target is generated but loses eligibility or receives an unsupported contradiction/category treatment.",
        "D": "The target has higher disclosed-constraint coverage than at least one product above it.",
        "E": "Available dialogue evidence leaves the target coverage-tied with products above it.",
        "F": "A still-hidden catalog attribute is needed to distinguish the target and was not elicited.",
        "G": "Override evidence becomes usable only after the override turn.",
        "H": "Bounded traces do not establish a supported primary cause.",
    }
    for category in "ABCDEFGH":
        category_rows = grouped.get(category, [])
        scenario_counts = Counter(str(row["scenario_type"]) for row in category_rows)
        hit_rows = [
            row
            for row in category_rows
            if cast(dict[str, object], row["observed"])["hit"]
        ]
        ranks = [
            cast(int, cast(dict[str, object], row["observed"])["rank"])
            for row in hit_rows
        ]
        turns = [
            cast(int, cast(dict[str, object], row["observed"])["turn"])
            for row in hit_rows
        ]
        summaries[category] = {
            "session_count": len(category_rows),
            "scenario_counts": dict(sorted(scenario_counts.items())),
            "miss_count": len(category_rows) - len(hit_rows),
            "below_rank_one_hit_count": len(hit_rows),
            "terminal_rank_distribution": dict(sorted(Counter(ranks).items())),
            "mean_terminal_rank_for_hits": (
                round(sum(ranks) / len(ranks), 6) if ranks else None
            ),
            "mean_first_hit_turn_for_hits": (
                round(sum(turns) / len(turns), 6) if turns else None
            ),
            "pool_session_presence": {
                stage: sum(
                    cast(dict[str, int], row["pool_presence"])[stage] > 0
                    for row in category_rows
                )
                for stage in (
                    "lexical",
                    "dense",
                    "fused",
                    "filtered",
                    "reranked",
                    "published",
                )
            },
            "earliest_recovery_turn_distribution": dict(
                sorted(
                    Counter(
                        str(row["earliest_recovery_turn"])
                        for row in category_rows
                        if row["earliest_recovery_turn"] is not None
                    ).items()
                )
            ),
            "question_count_distribution": dict(
                sorted(
                    Counter(str(row["question_count"]) for row in category_rows).items()
                )
            ),
            "question_attribute_counts": dict(
                sorted(
                    Counter(
                        attribute
                        for row in category_rows
                        for attribute in cast(list[str], row["question_attributes"])
                    ).items()
                )
            ),
            "observable_route_counts": dict(
                sorted(
                    Counter(
                        route
                        for row in category_rows
                        for route in cast(list[str], row["observable_routes"])
                    ).items()
                )
            ),
            "oracle_technical_score_gain": round(
                sum(float(row["oracle_gain_contribution"]) for row in category_rows)
                / len(sessions),
                6,
            ),
            "conservative_realizable_technical_score_gain": round(
                sum(
                    float(row["conservative_gain_contribution"])
                    for row in category_rows
                )
                / len(sessions),
                6,
            ),
            "representative_generic_pattern": patterns[category],
        }
    eligible = [
        (
            category,
            cast(dict[str, object], summaries[category]),
        )
        for category in "ABCDEFG"
        if cast(int, cast(dict[str, object], summaries[category])["session_count"]) >= 8
        or float(
            cast(dict[str, object], summaries[category])[
                "conservative_realizable_technical_score_gain"
            ]
        )
        >= 0.020
    ]
    eligible.sort(
        key=lambda item: (
            -float(item[1]["conservative_realizable_technical_score_gain"]),
            -cast(int, item[1]["session_count"]),
            item[0],
        )
    )
    selected = eligible[0][0] if eligible else None
    return {
        "residual_session_count": len(residual),
        "assigned_session_count": len(rows),
        "assignment_is_exhaustive_and_exclusive": len(rows) == len(residual),
        "categories": summaries,
        "selected_category": selected,
        "selection_threshold_met": selected is not None,
        "session_assignments": rows,
    }


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

    baseline_agent = _p2_agent(catalog, dense_cache)
    baseline = evaluate_shadow(baseline_agent, samples, frozenset(products))
    del baseline_agent
    gc.collect()
    traced_agent = _p2_agent(catalog, dense_cache)
    sessions, correctness, trace_hash = _replay_with_traces(
        traced_agent, samples, products
    )
    del traced_agent
    gc.collect()
    traced_metrics = metric_summary(sessions)
    diagnosis = _diagnose(sessions, samples, products, catalog, dense_cache)
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
    result = {
        "schema_version": 1,
        "diagnosis_id": "P4A-p2-residual-causal-diagnosis-v1",
        "predeclaration_commit": "1363ad8",
        "seed": SEED,
        "inputs": {
            "sample_count": len(samples),
            "public_target_overlap": len(targets & excluded),
            "target_selection_sha256": hashlib.sha256(
                "\n".join(sorted(targets)).encode()
            ).hexdigest(),
            "p2_contextual_policy": P2_CONTEXTUAL_POLICY,
            "p2_clarification_policy": P2_CLARIFICATION_POLICY,
            "p3_exposure_enabled": False,
        },
        "trace_integrity": {
            "expected_p2_transcript_sha256": EXPECTED_P2_TRANSCRIPT_SHA256,
            "baseline_transcript_sha256": baseline["normalized_transcript_sha256"],
            "observed_trace_transcript_sha256": trace_hash,
            "baseline_matches_checkpoint": baseline["normalized_transcript_sha256"]
            == EXPECTED_P2_TRANSCRIPT_SHA256,
            "observer_matches_baseline": (
                trace_hash == baseline["normalized_transcript_sha256"]
                and baseline_outcomes == baseline["sessions"]
                and traced_metrics["technical_score"] == baseline["technical_score"]
            ),
        },
        "p2_shadow_metrics": traced_metrics,
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
        "official_runs_consumed": 0,
        "official_runs_remaining": 4,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
