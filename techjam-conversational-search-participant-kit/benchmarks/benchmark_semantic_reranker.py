"""Run isolated official-set A/B evaluations for semantic reranking."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from dense_retrieval import DenseRetriever
from starter.semantic_reranker import (
    DEFAULT_RERANKER_MAX_LENGTH,
    DEFAULT_RERANKER_MODEL,
)


SCENARIOS = ("buying", "browsing", "intent_override", "boundary")


def _prepare_dense_cache(catalog: Path, dense_cache: Path) -> dict[str, Any]:
    """Validate or build the dense cache outside both measured variants."""

    existed = dense_cache.is_file()
    previous_mtime_ns = dense_cache.stat().st_mtime_ns if existed else None
    started = time.perf_counter()
    retriever = DenseRetriever.from_catalog(catalog, cache_path=dense_cache)
    elapsed = time.perf_counter() - started
    details = {
        "cache_existed_before_preparation": existed,
        "cache_rebuilt": (
            not existed or dense_cache.stat().st_mtime_ns != previous_mtime_ns
        ),
        "preparation_seconds": elapsed,
        "catalog_size": retriever.catalog_size,
        "embedding_dimension": retriever.embedding_dimension,
        "embedding_matrix_bytes": retriever.embedding_nbytes,
        "cache_file_bytes": dense_cache.stat().st_size,
    }
    del retriever
    gc.collect()
    return details


def _variant_summary(result: dict[str, Any]) -> dict[str, Any]:
    semantic = result["semantic_reranker"]
    performance = result["performance"]
    return {
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "efficiency": result["efficiency"],
        "technical_score": result["recommended_technical_score"],
        "scenario_metrics": {
            scenario: result["scenario_metrics"].get(scenario) for scenario in SCENARIOS
        },
        "agent_startup_seconds": performance["agent_startup_seconds"],
        "average_response_latency_ms": result["evaluation_diagnostics"][
            "average_response_latency_ms"
        ],
        "p95_response_latency_ms": result["evaluation_diagnostics"][
            "p95_response_latency_ms"
        ],
        "peak_process_rss_bytes": performance["peak_process_rss_bytes"],
        "model": semantic["model"],
        "configured_model_revision": semantic.get("configured_model_revision"),
        "resolved_model_revision": semantic.get("resolved_model_revision"),
        "candidate_count": semantic.get("candidate_count"),
        "batch_size": semantic.get("batch_size"),
        "model_size_bytes": semantic.get("model_size_bytes"),
        "cold_start_seconds": semantic.get("cold_start_seconds"),
        "average_reranking_latency_ms": semantic.get("average_reranking_latency_ms"),
        "p95_reranking_latency_ms": semantic.get("p95_reranking_latency_ms"),
        "max_scored_candidates_per_query": semantic.get(
            "max_scored_candidates_per_query", 0
        ),
        "failure_count": semantic["failure_count"],
    }


def _numeric_delta(semantic: dict[str, Any], baseline: dict[str, Any], key: str) -> Any:
    left = semantic.get(key)
    right = baseline.get(key)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left - right
    return None


def compare_results(
    baseline_result: dict[str, Any],
    semantic_result: dict[str, Any],
    *,
    minimum_technical_score_gain: float,
    maximum_average_reranking_latency_ms: float = 250.0,
    maximum_peak_memory_increase_bytes: int = 500_000_000,
) -> dict[str, Any]:
    baseline = _variant_summary(baseline_result)
    semantic = _variant_summary(semantic_result)
    delta_keys = (
        "hit_rate_at_10",
        "mrr",
        "mttc",
        "efficiency",
        "technical_score",
        "agent_startup_seconds",
        "average_response_latency_ms",
        "p95_response_latency_ms",
        "peak_process_rss_bytes",
    )
    deltas = {key: _numeric_delta(semantic, baseline, key) for key in delta_keys}
    score_gain = deltas["technical_score"]
    average_reranking_latency = semantic["average_reranking_latency_ms"]
    memory_increase = deltas["peak_process_rss_bytes"]
    measurable_gain = (
        isinstance(score_gain, (int, float))
        and score_gain >= minimum_technical_score_gain
        and semantic["hit_rate_at_10"] >= baseline["hit_rate_at_10"]
        and semantic["failure_count"] == 0
        and isinstance(average_reranking_latency, (int, float))
        and average_reranking_latency <= maximum_average_reranking_latency_ms
        and isinstance(memory_increase, (int, float))
        and memory_increase <= maximum_peak_memory_increase_bytes
    )
    return {
        "experiment": "bounded local semantic cross-encoder reranking",
        "baseline": baseline,
        "semantic_reranker": semantic,
        "deltas_semantic_minus_baseline": deltas,
        "decision_rule": {
            "minimum_technical_score_gain": minimum_technical_score_gain,
            "must_not_reduce_hit_rate_at_10": True,
            "requires_zero_failures": True,
            "maximum_average_reranking_latency_ms": (
                maximum_average_reranking_latency_ms
            ),
            "maximum_peak_memory_increase_bytes": (maximum_peak_memory_increase_bytes),
        },
        "recommendation": (
            "retain_as_optional_experiment"
            if measurable_gain
            else "do_not_retain_without_further_evidence"
        ),
        "default_enabled": False,
    }


def _run_variant(
    *,
    catalog: Path,
    dataset: Path,
    dense_cache: Path,
    output: Path,
    semantic: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "evaluator.local_evaluator",
        "--catalog",
        str(catalog),
        "--dataset",
        str(dataset),
        "--output",
        str(output),
        "--retrieval-mode",
        "hybrid",
        "--lexical-candidates",
        str(args.lexical_candidates),
        "--dense-candidates",
        str(args.dense_candidates),
        "--final-candidates",
        str(args.final_candidates),
        "--lexical-weight",
        str(args.lexical_weight),
        "--dense-weight",
        str(args.dense_weight),
        "--rrf-k",
        str(args.rrf_k),
        "--dense-cache",
        str(dense_cache),
        "--semantic-model",
        args.semantic_model,
        "--semantic-candidates",
        str(args.semantic_candidates),
        "--semantic-batch-size",
        str(args.semantic_batch_size),
        "--semantic-max-length",
        str(args.semantic_max_length),
    ]
    if args.semantic_model_revision:
        command.extend(["--semantic-model-revision", args.semantic_model_revision])
    if semantic:
        command.append("--semantic-reranker")
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare official metrics with and without semantic reranking"
    )
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument(
        "--dense-cache",
        type=Path,
        default=Path("data/.dense-retrieval/catalog-minilm.npz"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("docs/results/issue_4b")
    )
    parser.add_argument("--lexical-candidates", type=int, default=200)
    parser.add_argument("--dense-candidates", type=int, default=200)
    parser.add_argument("--final-candidates", type=int, default=10)
    parser.add_argument("--lexical-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--semantic-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--semantic-model-revision")
    parser.add_argument("--semantic-candidates", type=int, default=50)
    parser.add_argument("--semantic-batch-size", type=int, default=16)
    parser.add_argument(
        "--semantic-max-length", type=int, default=DEFAULT_RERANKER_MAX_LENGTH
    )
    parser.add_argument("--minimum-technical-score-gain", type=float, default=0.005)
    parser.add_argument(
        "--maximum-average-reranking-latency-ms", type=float, default=250.0
    )
    parser.add_argument(
        "--maximum-peak-memory-increase-bytes", type=int, default=500_000_000
    )
    args = parser.parse_args()

    if not args.catalog.is_file():
        parser.error(f"catalog not found: {args.catalog}")
    if not args.dataset.is_file():
        parser.error(f"dataset not found: {args.dataset}")
    if args.minimum_technical_score_gain < 0:
        parser.error("--minimum-technical-score-gain must be non-negative")
    if args.maximum_average_reranking_latency_ms < 0:
        parser.error("--maximum-average-reranking-latency-ms must be non-negative")
    if args.maximum_peak_memory_increase_bytes < 0:
        parser.error("--maximum-peak-memory-increase-bytes must be non-negative")

    catalog = args.catalog.resolve()
    dataset = args.dataset.resolve()
    dense_cache = args.dense_cache.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = output_dir / "hybrid_without_semantic.json"
    semantic_path = output_dir / "hybrid_with_semantic.json"
    comparison_path = output_dir / "comparison.json"

    dense_cache_preparation = _prepare_dense_cache(catalog, dense_cache)

    baseline = _run_variant(
        catalog=catalog,
        dataset=dataset,
        dense_cache=dense_cache,
        output=baseline_path,
        semantic=False,
        args=args,
    )
    semantic = _run_variant(
        catalog=catalog,
        dataset=dataset,
        dense_cache=dense_cache,
        output=semantic_path,
        semantic=True,
        args=args,
    )
    comparison = compare_results(
        baseline,
        semantic,
        minimum_technical_score_gain=args.minimum_technical_score_gain,
        maximum_average_reranking_latency_ms=(
            args.maximum_average_reranking_latency_ms
        ),
        maximum_peak_memory_increase_bytes=(args.maximum_peak_memory_increase_bytes),
    )
    comparison["dense_cache_preparation"] = dense_cache_preparation
    comparison_path.write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
