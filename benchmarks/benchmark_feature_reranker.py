"""Lightweight deterministic benchmark for 30-100 candidate reranking pools."""

from __future__ import annotations

import argparse
import json
import statistics
import time

from starter.feature_reranker import FeatureReranker, InMemoryCatalogView
from starter.hybrid_retrieval import Candidate
from starter.search_models import Constraint, PriceConstraint, SearchQuery


def percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, int(0.95 * len(ordered) + 0.999999) - 1)]


def fixture(pool_size: int) -> tuple[SearchQuery, list[Candidate], InMemoryCatalogView]:
    products: list[dict] = []
    candidates: list[Candidate] = []
    for index in range(pool_size):
        parent_asin = f"SYNTHETIC-{index:03d}"
        products.append(
            {
                "parent_asin": parent_asin,
                "title": f"{'red' if index % 3 == 0 else 'blue'} running shoe",
                "categories": ["sneakers" if index % 5 else "sandals"],
                "details": {
                    "Color": "red" if index % 3 == 0 else "blue",
                    "Material": "canvas" if index % 2 else "leather",
                    "Recommended Use": "running" if index % 4 else "walking",
                },
                "price": 20 + index,
            }
        )
        candidates.append(
            Candidate(
                parent_asin,
                lexical_score=float(pool_size - index),
                dense_score=1.0 - index / pool_size,
                lexical_rank=index + 1,
                dense_rank=pool_size - index,
                sources={"lexical", "dense"},
                fusion_score=1.0 / (61 + index) + 1.0 / (60 + pool_size - index),
            )
        )
    query = SearchQuery(
        text="red canvas running sneakers under 70",
        category=Constraint("sneakers", "soft", "current_turn", 1),
        color=Constraint("red", "soft", "current_turn", 1),
        material=Constraint("canvas", "soft", "current_turn", 1),
        use_case=Constraint("running", "soft", "current_turn", 1),
        price=PriceConstraint(maximum=70, strength="soft"),
    )
    return query, candidates, InMemoryCatalogView(products)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--runs", type=int, default=1_000)
    parser.add_argument("--warmup-runs", type=int, default=20)
    args = parser.parse_args()
    if not 30 <= args.pool_size <= 100:
        parser.error("--pool-size must be between 30 and 100")
    if args.runs <= 0 or args.warmup_runs < 0:
        parser.error("--runs must be positive and --warmup-runs non-negative")

    query, candidates, catalog = fixture(args.pool_size)
    reranker = FeatureReranker()
    for _ in range(args.warmup_runs):
        reranker.rerank(query, candidates, catalog, top_k=10)

    latencies_ms: list[float] = []
    expected_order: list[str] | None = None
    for _ in range(args.runs):
        started = time.perf_counter()
        result = reranker.rerank(query, candidates, catalog, top_k=10)
        latencies_ms.append((time.perf_counter() - started) * 1_000)
        order = [candidate.parent_asin for candidate in result]
        if expected_order is None:
            expected_order = order
        elif order != expected_order:
            raise RuntimeError("reranking output changed across identical benchmark runs")
        if len(order) != len(set(order)) or not set(order) <= {
            candidate.parent_asin for candidate in candidates
        }:
            raise RuntimeError("candidate-pool invariant failed")

    print(
        json.dumps(
            {
                "pool_size": args.pool_size,
                "top_k": 10,
                "runs": args.runs,
                "mean_latency_ms": round(statistics.fmean(latencies_ms), 6),
                "p95_latency_ms": round(percentile_95(latencies_ms), 6),
                "deterministic": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
