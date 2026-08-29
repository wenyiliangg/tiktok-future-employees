"""Run the unchanged public evaluator while exporting agent-side diagnostics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import cast

from evaluator.local_evaluator import (
    catalog_index,
    evaluate,
    load_jsonl,
    peak_process_rss_bytes,
)
from starter.agent import Agent
from starter.hybrid_retrieval import HybridRetrievalConfig, RetrievalMode


class AuditedAgent:
    """Transparent proxy counting invalid/duplicate raw recommendation IDs."""

    def __init__(self, agent: Agent, catalog_ids: set[str]) -> None:
        self.agent = agent
        self.catalog_ids = catalog_ids
        self.response_count = 0
        self.raw_recommendation_count = 0
        self.invalid_asin_count = 0
        self.duplicate_asin_count = 0

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.agent.reset(session_id, user_profile)

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        response = self.agent.respond(session_id, user_message, turn, top_k)
        self.response_count += 1
        seen: set[str] = set()
        recommendations = response.get("recommendations")
        if not isinstance(recommendations, list):
            self.invalid_asin_count += 1
            return response
        self.raw_recommendation_count += len(recommendations)
        for item in recommendations:
            parent_asin = item.get("parent_asin") if isinstance(item, dict) else None
            if (
                not isinstance(parent_asin, str)
                or not parent_asin
                or parent_asin not in self.catalog_ids
            ):
                self.invalid_asin_count += 1
                continue
            if parent_asin in seen:
                self.duplicate_asin_count += 1
            seen.add(parent_asin)
        return response

    def snapshot(self) -> dict[str, int]:
        return {
            "response_count": self.response_count,
            "raw_recommendation_count": self.raw_recommendation_count,
            "invalid_asin_count": self.invalid_asin_count,
            "duplicate_asin_count": self.duplicate_asin_count,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Public evaluation with route/fallback diagnostics"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results.json")
    parser.add_argument(
        "--retrieval-mode",
        choices=[mode.value for mode in RetrievalMode],
        default=RetrievalMode.LEXICAL.value,
    )
    parser.add_argument("--lexical-candidates", type=int, default=200)
    parser.add_argument("--dense-candidates", type=int, default=200)
    parser.add_argument("--final-candidates", type=int, default=10)
    parser.add_argument("--lexical-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument(
        "--dense-cache", default="data/.dense-retrieval/catalog-minilm.npz"
    )
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    config = HybridRetrievalConfig(
        mode=args.retrieval_mode,
        lexical_candidate_count=args.lexical_candidates,
        dense_candidate_count=args.dense_candidates,
        final_candidate_count=args.final_candidates,
        lexical_weight=args.lexical_weight,
        dense_weight=args.dense_weight,
        rrf_k=args.rrf_k,
    )
    startup_started = time.perf_counter()
    agent = Agent(args.catalog, config=config, dense_cache_path=args.dense_cache)
    startup_seconds = time.perf_counter() - startup_started
    audited_agent = AuditedAgent(agent, catalog_ids)
    result = evaluate(
        cast(Agent, audited_agent), samples, catalog_ids, categories, products
    )
    result["retrieval_diagnostics"] = agent.diagnostics_snapshot()
    result["raw_output_audit"] = audited_agent.snapshot()
    result["performance"] = {
        "agent_startup_seconds": round(startup_seconds, 6),
        "peak_process_rss_bytes": peak_process_rss_bytes(),
        "startup_scope": (
            "Agent construction, including catalog views and mode-specific indexes/cache"
        ),
        "latency_scope": "wall-clock time inside Agent.respond",
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "sessions"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
