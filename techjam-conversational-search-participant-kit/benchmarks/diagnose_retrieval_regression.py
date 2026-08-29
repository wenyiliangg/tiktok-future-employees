"""Targeted retrieval-regression diagnostics kept outside official results.

This harness replays only explicitly selected public sessions.  The
``--official-bm25-hits`` selector performs a lightweight anchor-only discovery
pass because the repository's official baseline artifact contains aggregate
metrics but no per-session records.  It does not call or modify the official
``evaluate`` function.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sqlite3
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, replace
from pathlib import Path

from evaluator.local_evaluator import (
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from starter.agent import Agent
from starter.hybrid_retrieval import (
    Candidate,
    HybridRetrievalConfig,
    RankedResult,
    RetrievalMode,
    default_route_policies,
    merge_candidates,
    reciprocal_rank_fusion,
)
from starter.lexical_retriever import INDEX_FIELDS, tokenize
from starter.route_aware_retrieval import (
    filter_candidates,
    merge_fallback_candidates,
    route_reciprocal_rank_fusion,
)

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "some",
    "that",
    "the",
    "this",
    "to",
    "want",
    "with",
    "would",
    "you",
    "looking",
}
OFFICIAL_FIELDS = ("title", "categories", "features", "details", "store", "description")
OFFICIAL_WEIGHTS = (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            token.lower()
            for token in TOKEN_RE.findall(text)
            if len(token) > 1 and token.lower() not in STOPWORDS
        )
    )[:40]


class OfficialWeakBM25:
    """Exact diagnostic copy of the official starter Agent's retrieval."""

    def __init__(self, catalog_path: Path) -> None:
        self.connection = sqlite3.connect(":memory:")
        self.connection.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, "
            "description, tokenize='unicode61 remove_diacritics 2')"
        )
        placeholders = ", ".join("?" for _ in range(len(OFFICIAL_FIELDS) + 1))
        batch: list[tuple[str, ...]] = []
        with catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        *(_text(product.get(field)) for field in OFFICIAL_FIELDS),
                    )
                )
                if len(batch) >= 1_000:
                    self.connection.executemany(
                        f"INSERT INTO products VALUES ({placeholders})", batch
                    )
                    batch.clear()
        if batch:
            self.connection.executemany(
                f"INSERT INTO products VALUES ({placeholders})", batch
            )
        self.connection.commit()
        self._cache: dict[tuple[str, int], list[RankedResult]] = {}

    def retrieve(self, user_message: str, top_n: int) -> list[RankedResult]:
        cache_key = (user_message, top_n)
        if cache_key in self._cache:
            return copy.deepcopy(self._cache[cache_key])
        terms = _terms(user_message)
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in terms)
        weights = ", ".join(str(value) for value in OFFICIAL_WEIGHTS)
        rows = self.connection.execute(
            "SELECT parent_asin, bm25(products, " + weights + ") AS score "
            "FROM products WHERE products MATCH ? ORDER BY score LIMIT ?",
            (expression, top_n),
        ).fetchall()
        results = [
            _DiagnosticRankedResult(str(parent_asin), -float(score), rank)
            for rank, (parent_asin, score) in enumerate(rows, start=1)
        ]
        self._cache[cache_key] = copy.deepcopy(results)
        return results


class _DiagnosticRankedResult:
    def __init__(self, parent_asin: str, score: float, rank: int) -> None:
        self.parent_asin = parent_asin
        self.score = score
        self.rank = rank


def _rank(results: Iterable[object], target: str) -> int | None:
    for position, item in enumerate(results, start=1):
        if getattr(item, "parent_asin", None) == target:
            value = getattr(item, "rank", None)
            return value if isinstance(value, int) else position
    return None


def _candidate_rank(results: Sequence[Candidate], target: str) -> int | None:
    for rank, item in enumerate(results, start=1):
        if item.parent_asin == target:
            return rank
    return None


def _session_messages(
    sample: dict,
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> list[dict[str, object]]:
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    message = initial_message(
        effective, coarse_category(categories.get(target, [])), disclosed
    )
    turns: list[dict[str, object]] = []
    for turn in range(1, 11):
        turns.append(
            {
                "turn": turn,
                "user_message": message,
                "scoring_eligible": override_applied,
            }
        )
        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            message = str(
                override.get(
                    "message", "Actually, please ignore my earlier preference."
                )
            )
        else:
            message, boundary_used = customer_reply(
                effective, None, disclosed, boundary_used
            )
    return turns


def _discover_official_hits(
    anchor: OfficialWeakBM25,
    samples: Sequence[dict],
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> list[str]:
    hit_ids: list[str] = []
    for sample in samples:
        target = str(sample["ground_truth"]["parent_asin"])
        for turn in _session_messages(sample, categories, products):
            official = anchor.retrieve(str(turn["user_message"]), TOP_K)
            if turn["scoring_eligible"] and _rank(official, target) is not None:
                hit_ids.append(str(sample["sample_id"]))
                break
    return hit_ids


def _constraint_snapshot(query: object) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {"hard": [], "soft": []}
    for name in ("category", "color", "style", "material", "use_case"):
        constraint = getattr(query, name, None)
        if constraint is None:
            continue
        strength = str(getattr(constraint, "strength", "hard"))
        result.setdefault(strength, []).append(
            {
                "name": name,
                "value": getattr(constraint, "value", None),
                "source": getattr(constraint, "source", None),
                "updated_turn": getattr(constraint, "updated_turn", None),
            }
        )
    price = getattr(query, "price", None)
    if price is not None:
        strength = str(getattr(price, "strength", "hard"))
        result.setdefault(strength, []).append(
            {
                "name": "price",
                "minimum": getattr(price, "minimum", None),
                "maximum": getattr(price, "maximum", None),
                "source": getattr(price, "source", None),
                "updated_turn": getattr(price, "updated_turn", None),
            }
        )
    return result


def _current_raw_lexical_results(lexical: object, query: object) -> list[RankedResult]:
    constraints = lexical._constraints(query)
    terms = lexical._query_terms(query, constraints)
    if not terms:
        return []
    expression = " OR ".join(f'"{term}"' for term in terms)
    weights = [
        0.0,
        *(float(lexical.config.field_weights.get(name, 0.0)) for name in INDEX_FIELDS),
    ]
    weight_sql = ", ".join("?" for _ in weights)
    rows = lexical.connection.execute(
        "SELECT parent_asin, bm25(products, " + weight_sql + ") AS score "
        "FROM products WHERE products MATCH ? "
        "ORDER BY score ASC, parent_asin ASC LIMIT ?",
        (*weights, expression, lexical.config.candidate_pool_size),
    ).fetchall()
    return [
        _DiagnosticRankedResult(str(parent_asin), -float(score), rank)
        for rank, (parent_asin, score) in enumerate(rows, start=1)
    ]


def _current_lexical_filter_audit(
    lexical: object, query: object, target: str
) -> dict[str, object]:
    record = lexical._records[target]
    violations: list[str] = []
    constraints = lexical._constraints(query)
    for name, constraint in constraints:
        if constraint.strength != "hard":
            continue
        if not lexical._tokens_match(
            tokenize(constraint.value), record.metadata_tokens.get(name, frozenset())
        ):
            violations.append(name)
    price_bounds = lexical._price_bounds(query)
    if (
        query.price is not None
        and query.price.strength == "hard"
        and price_bounds not in (None, (None, None))
        and not lexical._price_matches(record.price, price_bounds)
    ):
        violations.append("price")
    exclusion_violation = lexical._violates_exclusions(record, query.exclusions)
    return {
        "passes": not violations and not exclusion_violation,
        "hard_violations": violations,
        "exclusion_violation": exclusion_violation,
    }


def _route_pipeline(
    agent: Agent,
    query: object,
    sample_id: str,
    target: str,
) -> dict[str, object]:
    state = agent._state.state_for(sample_id)
    decision = agent._router.route(state, query)
    route = str(decision.route)
    policy = agent.config.policy_for(route)
    lexical = list(
        agent._require_lexical().retrieve(query, top_n=policy.lexical_candidate_count)
    )
    dense = (
        list(
            agent._require_dense().retrieve(
                query.text, top_n=policy.dense_candidate_count
            )
        )
        if query.text.strip()
        else []
    )
    merged = merge_candidates(lexical, dense, agent._catalog_ids)
    fallback_results: list[object] = []
    fallback_attempted = (
        agent.config.enable_boundary_fallback
        and route == "boundary"
        and policy.fallback_candidate_count > 0
        and (
            policy.always_attempt_fallback
            or len(merged) < policy.fallback_trigger_count
        )
    )
    fallback_cache_hit = False
    if fallback_attempted and agent._fallback is not None:
        cache_key = agent._fallback_cache_key(
            query, state.removed_constraints, policy.fallback_candidate_count
        )
        cached = agent._fallback_cache.setdefault(sample_id, {}).get(cache_key)
        if cached is not None:
            fallback_cache_hit = True
            fallback_results = copy.deepcopy(cached)
        else:
            fallback_results = list(
                agent._fallback.generate(
                    query=query,
                    user_profile=agent._user_profiles.get(sample_id, {}),
                    top_n=policy.fallback_candidate_count,
                    removed_constraints=state.removed_constraints,
                )
            )
            agent._fallback_cache[sample_id][cache_key] = copy.deepcopy(
                fallback_results
            )
    merged = merge_fallback_candidates(merged, fallback_results, agent._catalog_ids)
    union_inclusion = any(item.parent_asin == target for item in merged)
    filtered, summary = filter_candidates(
        query, copy.deepcopy(merged), agent._catalog_view, policy
    )
    target_probe = Candidate(target)
    filter_candidates(query, [target_probe], agent._catalog_view, policy)
    fused_all = route_reciprocal_rank_fusion(
        copy.deepcopy(filtered), policy, limit=max(1, len(filtered))
    )
    final = fused_all[: min(TOP_K, policy.final_candidate_count)]
    return {
        "selected_route": route,
        "route_confidence": decision.confidence,
        "route_reasons": list(decision.reasons),
        "policy": asdict(policy),
        "lexical_target_rank": _rank(lexical, target),
        "dense_target_rank": _rank(dense, target),
        "hybrid_union_inclusion": union_inclusion,
        "post_filter_inclusion": any(item.parent_asin == target for item in filtered),
        "post_fusion_rank": _candidate_rank(fused_all, target),
        "final_rank": _candidate_rank(final, target),
        "target_filter_diagnostics": target_probe.filter_diagnostics,
        "filter_counts": asdict(summary),
        "fallback_attempted": fallback_attempted,
        "fallback_cache_hit": fallback_cache_hit,
        "fallback_candidate_count": len(fallback_results),
    }


def _loss_classification(turn: dict[str, object]) -> str:
    if turn["current_fixed_final_rank"] is not None:
        return "recovered_by_current_fixed"
    if not turn["scoring_eligible"]:
        return "interaction_or_override_eligibility"
    if not turn["hybrid_union_inclusion"]:
        return "target_not_retrieved"
    fusion_rank = turn["post_fusion_rank"]
    if isinstance(fusion_rank, int) and fusion_rank > TOP_K:
        return "target_present_but_demoted_by_fusion"
    if turn["feature_reranker_removal_reason"]:
        return "target_lost_after_reranking_filter"
    return "target_lost_after_reranking"


def _diagnose(
    anchor: OfficialWeakBM25,
    samples: Sequence[dict],
    catalog_path: Path,
    dense_cache: Path,
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> dict[str, object]:
    legacy_policies = default_route_policies()
    legacy_policies["boundary"] = replace(
        legacy_policies["boundary"], always_attempt_fallback=True
    )
    config = HybridRetrievalConfig(
        mode=RetrievalMode.ROUTE_AWARE,
        enable_feature_reranker=True,
        enable_boundary_fallback=True,
        route_policies=legacy_policies,
    )
    agent = Agent(catalog_path, config=config, dense_cache_path=dense_cache)
    sessions: list[dict[str, object]] = []
    loss_counts: Counter[str] = Counter()
    rrf_displacements = 0
    hard_filter_false_negatives = 0
    official_hit_turns_with_dropped_text = 0

    for sample in samples:
        sample_id = str(sample["sample_id"])
        target = str(sample["ground_truth"]["parent_asin"])
        agent.reset(sample_id, sample["user_profile"])
        turns: list[dict[str, object]] = []
        official_hit_turn: dict[str, object] | None = None
        for transcript_turn in _session_messages(sample, categories, products):
            turn_number = int(transcript_turn["turn"])
            user_message = str(transcript_turn["user_message"])
            query = agent._state.update(sample_id, user_message, turn_number)

            official = anchor.retrieve(user_message, 200)
            current_raw_lexical = _current_raw_lexical_results(
                agent._require_lexical(), query
            )
            current_lexical = list(agent._require_lexical().retrieve(query, top_n=200))
            current_dense = (
                list(agent._require_dense().retrieve(query.text, top_n=200))
                if query.text.strip()
                else []
            )
            raw_dense = list(agent._require_dense().retrieve(user_message, top_n=200))
            merged = merge_candidates(
                current_lexical, current_dense, agent._catalog_ids
            )
            fused_all = reciprocal_rank_fusion(
                copy.deepcopy(merged), config, limit=max(1, len(merged))
            )
            feature_pool = copy.deepcopy(fused_all[: config.rerank_candidate_count])
            fixed_final = agent._reranker.rerank(
                query, feature_pool, agent._catalog_view, top_k=TOP_K
            )
            reranker_diagnostic = agent._reranker.last_diagnostics.get(target, {})

            anchor_dense = merge_candidates(official, raw_dense, agent._catalog_ids)
            anchor_dense_fused = reciprocal_rank_fusion(
                anchor_dense, config, limit=max(1, len(anchor_dense))
            )
            anchor_rrf_rank = _candidate_rank(anchor_dense_fused, target)
            official_rank = _rank(official, target)
            dual_source_above = []
            if official_rank is not None and official_rank <= TOP_K and anchor_rrf_rank:
                dual_source_above = [
                    {
                        "parent_asin": item.parent_asin,
                        "lexical_rank": item.lexical_rank,
                        "dense_rank": item.dense_rank,
                        "fusion_score": item.fusion_score,
                    }
                    for item in anchor_dense_fused[: anchor_rrf_rank - 1]
                    if item.lexical_rank is not None and item.dense_rank is not None
                ]

            route = _route_pipeline(agent, query, sample_id, target)
            filter_audit = _current_lexical_filter_audit(
                agent._require_lexical(), query, target
            )
            raw_tokens = list(tokenize(user_message))
            structured_tokens = list(tokenize(query.text))
            dropped_tokens = [
                token for token in raw_tokens if token not in set(structured_tokens)
            ]
            turn_result: dict[str, object] = {
                **transcript_turn,
                "target": target,
                "official_bm25_target_rank": official_rank,
                "official_bm25_scored_rank": (
                    official_rank
                    if official_rank is not None and official_rank <= TOP_K
                    else None
                ),
                "structured_query_text": query.text,
                "raw_current_turn_tokens": raw_tokens,
                "structured_query_tokens": structured_tokens,
                "dropped_raw_tokens": dropped_tokens,
                "active_constraints": _constraint_snapshot(query),
                "current_raw_lexical_target_rank": _rank(current_raw_lexical, target),
                "current_lexical_target_rank": _rank(current_lexical, target),
                "current_dense_target_rank": _rank(current_dense, target),
                "raw_text_dense_target_rank": _rank(raw_dense, target),
                "hybrid_union_inclusion": any(
                    item.parent_asin == target for item in merged
                ),
                "post_fusion_rank": _candidate_rank(fused_all, target),
                "common_2b_final_rank": _candidate_rank(fused_all[:TOP_K], target),
                "current_fixed_final_rank": _candidate_rank(fixed_final, target),
                "feature_reranker_removal_reason": reranker_diagnostic.get(
                    "removal_reason"
                ),
                "current_lexical_filter_audit": filter_audit,
                "anchor_plus_raw_dense_rrf_rank": anchor_rrf_rank,
                "anchor_top10_displaced_by_equal_rrf": (
                    official_rank is not None
                    and official_rank <= TOP_K
                    and (anchor_rrf_rank is None or anchor_rrf_rank > TOP_K)
                ),
                "dual_source_candidates_above_anchor_target": dual_source_above,
                "route_aware": route,
                "selected_route": route["selected_route"],
                "post_filter_inclusion": route["post_filter_inclusion"],
                "route_post_fusion_rank": route["post_fusion_rank"],
                "route_final_rank": route["final_rank"],
            }
            turns.append(turn_result)
            if (
                official_hit_turn is None
                and transcript_turn["scoring_eligible"]
                and official_rank is not None
                and official_rank <= TOP_K
            ):
                turn_result["loss_classification"] = _loss_classification(turn_result)
                official_hit_turn = turn_result

        if official_hit_turn is not None:
            classification = str(official_hit_turn["loss_classification"])
            loss_counts[classification] += 1
            if official_hit_turn["anchor_top10_displaced_by_equal_rrf"]:
                rrf_displacements += 1
            if not official_hit_turn["current_lexical_filter_audit"]["passes"]:
                hard_filter_false_negatives += 1
            if official_hit_turn["dropped_raw_tokens"]:
                official_hit_turns_with_dropped_text += 1
        sessions.append(
            {
                "sample_id": sample_id,
                "scenario_type": sample["scenario_type"],
                "target": target,
                "official_hit_turn": (
                    official_hit_turn["turn"] if official_hit_turn is not None else None
                ),
                "loss_classification": (
                    official_hit_turn["loss_classification"]
                    if official_hit_turn is not None
                    else "official_baseline_miss"
                ),
                "turns": turns,
            }
        )

    return {
        "diagnostic_only": True,
        "official_evaluator_modified": False,
        "public_labels_modified": False,
        "configuration": {
            "official_bm25_fields": list(OFFICIAL_FIELDS),
            "official_bm25_weights": list(OFFICIAL_WEIGHTS),
            "current_fixed": {
                "lexical_candidates": config.lexical_candidate_count,
                "dense_candidates": config.dense_candidate_count,
                "rrf_k": config.rrf_k,
                "lexical_weight": config.lexical_weight,
                "dense_weight": config.dense_weight,
                "feature_rerank_candidates": config.rerank_candidate_count,
                "final_candidates": config.final_candidate_count,
                "feature_reranker_enabled": config.enable_feature_reranker,
                "boundary_fallback_enabled": config.enable_boundary_fallback,
            },
            "dense_cache": str(dense_cache),
        },
        "summary": {
            "selected_session_count": len(samples),
            "loss_classifications": dict(sorted(loss_counts.items())),
            "official_top10_targets_displaced_by_anchor_plus_raw_dense_equal_rrf": (
                rrf_displacements
            ),
            "official_hit_targets_failing_current_lexical_hard_filters": (
                hard_filter_false_negatives
            ),
            "official_hit_turns_with_dropped_raw_tokens": (
                official_hit_turns_with_dropped_text
            ),
        },
        "sessions": sessions,
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
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument(
        "--official-bm25-hits",
        action="store_true",
        help="diagnose exactly the sessions hit by the official weak BM25 anchor",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/retrieval_regression/selected_sessions.json"),
    )
    args = parser.parse_args()
    if args.sample_id and args.official_bm25_hits:
        parser.error("choose --sample-id or --official-bm25-hits, not both")
    if not args.catalog.is_file() or not args.dataset.is_file():
        parser.error("catalog and dataset must exist")

    samples = load_jsonl(args.dataset)
    _, categories, products = catalog_index(args.catalog)
    anchor = OfficialWeakBM25(args.catalog)
    discovered_hit_ids: list[str] | None = None
    if args.official_bm25_hits or not args.sample_id:
        discovered_hit_ids = _discover_official_hits(
            anchor, samples, categories, products
        )
        selected_ids = set(discovered_hit_ids)
    else:
        selected_ids = set(args.sample_id)
    available_ids = {str(sample["sample_id"]) for sample in samples}
    missing_ids = sorted(selected_ids - available_ids)
    if missing_ids:
        parser.error(f"unknown sample IDs: {', '.join(missing_ids)}")
    selected = [
        sample for sample in samples if str(sample["sample_id"]) in selected_ids
    ]
    result = _diagnose(
        anchor,
        selected,
        args.catalog,
        args.dense_cache,
        categories,
        products,
    )
    result["selection"] = {
        "mode": "official_bm25_hits" if discovered_hit_ids is not None else "explicit",
        "sample_ids": [str(sample["sample_id"]) for sample in selected],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
