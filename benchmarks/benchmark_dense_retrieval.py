from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from dense_retrieval import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    DenseRetriever,
    SentenceTransformerEncoder,
)


REPRESENTATIVE_QUERIES = (
    "comfortable everyday clothes",
    "women's casual shoes",
    "warm winter accessories",
    "red cotton crew neck shirt for men",
    "blue waterproof trail running shoes with grippy sole",
    "lightweight hypoallergenic stainless steel hoop earrings",
)


class TimedEncoder:
    def __init__(self, encoder: object) -> None:
        self._encoder = encoder
        self.encode_seconds = 0.0

    @property
    def model_name(self) -> str:
        return str(getattr(self._encoder, "model_name"))

    @property
    def model_revision(self) -> str | None:
        value = getattr(self._encoder, "model_revision", None)
        return str(value) if value is not None else None

    @property
    def resolved_model_revision(self) -> str | None:
        value = getattr(self._encoder, "resolved_model_revision", None)
        return str(value) if value is not None else None

    @property
    def embedding_dimension(self) -> int | None:
        value = getattr(self._encoder, "embedding_dimension", None)
        return int(value) if value is not None else None

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        started = time.perf_counter()
        try:
            return np.asarray(getattr(self._encoder, "encode")(texts, batch_size))
        finally:
            self.encode_seconds += time.perf_counter() - started


class DeterministicBenchmarkEncoder:
    """Explicit offline benchmark fixture; never selected by production retrieval."""

    def __init__(self, dimension: int) -> None:
        self.model_name = "benchmark/deterministic-fake"
        self.model_revision = "1"
        self.embedding_dimension = dimension

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        del batch_size
        vectors: list[np.ndarray] = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
            vectors.append(
                np.random.default_rng(seed).standard_normal(
                    self.embedding_dimension, dtype=np.float32
                )
            )
        return np.stack(vectors)


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux and most BSDs report KiB.
    return value if sys.platform == "darwin" else value * 1024


def percentile_95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    index = max(0, int(np.ceil(0.95 * len(ordered))) - 1)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark exact dense catalog retrieval")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument(
        "--cache", type=Path, default=Path("data/.dense-retrieval/catalog.npz")
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument(
        "--deterministic-fake-dimension",
        type=int,
        help="explicitly use an offline fake encoder to verify benchmark plumbing",
    )
    args = parser.parse_args()
    if args.runs <= 0 or args.warmup_runs < 0:
        parser.error("--runs must be positive and --warmup-runs must be non-negative")

    def make_encoder() -> object:
        if args.deterministic_fake_dimension is not None:
            if args.deterministic_fake_dimension <= 0:
                parser.error("--deterministic-fake-dimension must be positive")
            return DeterministicBenchmarkEncoder(args.deterministic_fake_dimension)
        return SentenceTransformerEncoder(args.model, revision=args.model_revision)

    cold_encoder = TimedEncoder(make_encoder())
    cold_started = time.perf_counter()
    cold_retriever = DenseRetriever.from_catalog(
        args.catalog,
        cache_path=args.cache,
        encoder=cold_encoder,
        batch_size=args.batch_size,
        rebuild_cache=True,
    )
    cold_seconds = time.perf_counter() - cold_started

    cached_encoder = TimedEncoder(make_encoder())
    cached_started = time.perf_counter()
    retriever = DenseRetriever.from_catalog(
        args.catalog,
        cache_path=args.cache,
        encoder=cached_encoder,
        batch_size=args.batch_size,
    )
    cached_seconds = time.perf_counter() - cached_started
    cached_startup_encoder_seconds = cached_encoder.encode_seconds

    for _ in range(args.warmup_runs):
        for query in REPRESENTATIVE_QUERIES:
            retriever.retrieve(query, top_n=200)

    latencies_ms: list[float] = []
    for _ in range(args.runs):
        for query in REPRESENTATIVE_QUERIES:
            started = time.perf_counter()
            retriever.retrieve(query, top_n=200)
            latencies_ms.append((time.perf_counter() - started) * 1000.0)

    result = {
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "python": platform.python_version(),
            "logical_cpu_count": os.cpu_count(),
            "numpy": np.__version__,
        },
        "catalog_size": cold_retriever.catalog_size,
        "model": cold_encoder.model_name,
        "configured_model_revision": cold_encoder.model_revision,
        "resolved_model_revision": cold_encoder.resolved_model_revision,
        "embedding_dimension": cold_retriever.embedding_dimension,
        "batch_size": args.batch_size,
        "cold_startup_seconds": cold_seconds,
        "initial_embedding_generation_seconds": cold_encoder.encode_seconds,
        "cached_startup_seconds": cached_seconds,
        "cached_startup_encoder_calls_seconds": cached_startup_encoder_seconds,
        "embedding_matrix_bytes": cold_retriever.embedding_nbytes,
        "embedding_matrix_decimal_mb": cold_retriever.embedding_nbytes / 1_000_000,
        "embedding_matrix_mib": cold_retriever.embedding_nbytes / (1024 * 1024),
        "process_peak_rss_bytes": peak_rss_bytes(),
        "query_count_per_run": len(REPRESENTATIVE_QUERIES),
        "measured_runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "measured_query_count": len(latencies_ms),
        "average_query_latency_ms": statistics.fmean(latencies_ms),
        "p95_query_latency_ms": percentile_95(latencies_ms),
        "model_download_time_note": (
            "Cold startup includes download time if the configured real model was not cached."
        ),
        "benchmark_encoder": (
            "explicit deterministic fake (plumbing verification only)"
            if args.deterministic_fake_dimension is not None
            else "sentence-transformers production encoder"
        ),
        "queries": list(REPRESENTATIVE_QUERIES),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
