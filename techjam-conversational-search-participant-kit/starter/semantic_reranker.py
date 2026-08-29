"""Optional bounded local cross-encoder reranking.

The reranker is deliberately a post-processor: it can only reorder candidate
objects supplied by retrieval, and any load/inference/validation failure returns
those objects in their original order.  No network service is used for scoring.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from dense_retrieval import ProductTextBuilder

from .hybrid_retrieval import Candidate

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on Windows
    resource = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_RERANKER_MAX_LENGTH = 256


def _positive_integer(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class SemanticRerankerConfig:
    """Configuration for the opt-in reranking experiment."""

    enabled: bool = False
    model_name: str = DEFAULT_RERANKER_MODEL
    model_revision: str | None = None
    candidate_count: int = 50
    batch_size: int = 16
    max_length: int = DEFAULT_RERANKER_MAX_LENGTH
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be non-empty")
        if self.model_revision is not None and not isinstance(self.model_revision, str):
            raise ValueError("model_revision must be a string or None")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be non-empty")
        _positive_integer(self.candidate_count, "candidate_count")
        _positive_integer(self.batch_size, "batch_size")
        _positive_integer(self.max_length, "max_length")


class PairScorer(Protocol):
    """Minimal injectable boundary used by production and offline tests."""

    model_name: str
    model_revision: str | None

    @property
    def model_size_bytes(self) -> int | None: ...

    @property
    def cache_hit(self) -> bool | None: ...

    def ensure_loaded(self) -> None: ...

    def score(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
    ) -> Sequence[float]: ...


@dataclass(slots=True)
class _CachedModel:
    model: Any
    inference_lock: threading.Lock
    model_size_bytes: int | None
    resolved_revision: str | None


_MODEL_CACHE: dict[tuple[str, str | None, str, int], _CachedModel] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _parameter_bytes(model: object) -> int | None:
    transformer = getattr(model, "model", None)
    parameters = getattr(transformer, "parameters", None)
    if not callable(parameters):
        return None
    total = 0
    try:
        for parameter in parameters():
            total += int(parameter.numel()) * int(parameter.element_size())
    except (AttributeError, TypeError, ValueError):
        return None
    return total


def _resolved_revision(model: object) -> str | None:
    transformer = getattr(model, "model", None)
    config = getattr(transformer, "config", None)
    value = getattr(config, "_commit_hash", None)
    return value if isinstance(value, str) and value else None


class CrossEncoderPairScorer:
    """Cached, CPU-local ``sentence-transformers`` CrossEncoder adapter."""

    def __init__(
        self,
        model_name: str,
        *,
        model_revision: str | None,
        device: str,
        max_length: int,
    ) -> None:
        self.model_name = model_name
        self.model_revision = model_revision
        self.device = device
        self.max_length = max_length
        self._entry: _CachedModel | None = None
        self._cache_hit: bool | None = None

    @property
    def cache_hit(self) -> bool | None:
        return self._cache_hit

    @property
    def model_size_bytes(self) -> int | None:
        return self._entry.model_size_bytes if self._entry is not None else None

    @property
    def resolved_model_revision(self) -> str | None:
        return self._entry.resolved_revision if self._entry is not None else None

    def ensure_loaded(self) -> None:
        if self._entry is not None:
            return
        key = (
            self.model_name,
            self.model_revision,
            self.device,
            self.max_length,
        )
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(key)
            if cached is not None:
                self._entry = cached
                self._cache_hit = True
                return
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as error:
                raise RuntimeError(
                    "sentence-transformers is required for semantic reranking; "
                    "install dependencies from requirements.txt"
                ) from error
            kwargs: dict[str, object] = {
                "device": self.device,
                "max_length": self.max_length,
                "trust_remote_code": False,
            }
            if self.model_revision is not None:
                kwargs["revision"] = self.model_revision
            model = CrossEncoder(self.model_name, **kwargs)
            cached = _CachedModel(
                model=model,
                inference_lock=threading.Lock(),
                model_size_bytes=_parameter_bytes(model),
                resolved_revision=_resolved_revision(model),
            )
            _MODEL_CACHE[key] = cached
            self._entry = cached
            self._cache_hit = False

    def score(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
    ) -> Sequence[float]:
        self.ensure_loaded()
        assert self._entry is not None
        with self._entry.inference_lock:
            values = self._entry.model.predict(
                list(pairs),
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        # Avoid a hard NumPy dependency in this optional module.  CrossEncoder
        # outputs expose ``tolist``; plain injected sequences work unchanged.
        tolist = getattr(values, "tolist", None)
        return tolist() if callable(tolist) else values


def _peak_process_rss_bytes() -> int:
    if resource is None:
        return 0
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def _p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


class SemanticReranker:
    """Score only a bounded prefix and preserve the remaining base ordering."""

    def __init__(
        self,
        product_texts: Mapping[str, str],
        *,
        config: SemanticRerankerConfig,
        scorer: PairScorer | None = None,
    ) -> None:
        self.config = config
        self._product_texts = dict(product_texts)
        self._scorer: PairScorer = scorer or CrossEncoderPairScorer(
            config.model_name,
            model_revision=config.model_revision,
            device=config.device,
            max_length=config.max_length,
        )
        self._load_attempted = False
        self._cold_start_seconds: float | None = None
        self._load_memory_delta_bytes: int | None = None
        self._latencies_ms: list[float] = []
        self._failure_count = 0
        self._query_count = 0
        self._scored_candidate_count = 0
        self._max_scored_candidates = 0
        self._peak_process_rss_bytes = _peak_process_rss_bytes()
        self._failure_logged = False

    @classmethod
    def from_catalog(
        cls,
        catalog_path: str | Path,
        *,
        config: SemanticRerankerConfig,
        scorer: PairScorer | None = None,
        text_builder: ProductTextBuilder | None = None,
    ) -> SemanticReranker:
        builder = text_builder or ProductTextBuilder()
        texts: dict[str, str] = {}
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                product = json.loads(line)
                parent_asin = str(product.get("parent_asin") or "").strip()
                if not parent_asin:
                    raise ValueError(
                        f"catalog row {line_number} has an empty parent_asin"
                    )
                if parent_asin in texts:
                    raise ValueError(f"duplicate catalog parent_asin: {parent_asin}")
                texts[parent_asin] = builder.build(product)
        if not texts:
            raise ValueError("catalog contains no products")
        return cls(texts, config=config, scorer=scorer)

    def rerank(
        self,
        query_text: str,
        candidates: Sequence[Candidate],
    ) -> list[Candidate]:
        original = list(candidates)
        window_size = min(self.config.candidate_count, len(original))
        if not self.config.enabled or window_size == 0 or not query_text.strip():
            return original

        started = time.perf_counter()
        excluded_load_seconds = 0.0
        self._query_count += 1
        try:
            window = original[:window_size]
            product_texts = [self._product_texts[item.parent_asin] for item in window]
            if not self._load_attempted:
                self._load_attempted = True
                before_rss = _peak_process_rss_bytes()
                load_started = time.perf_counter()
                try:
                    self._scorer.ensure_loaded()
                finally:
                    self._cold_start_seconds = time.perf_counter() - load_started
                    excluded_load_seconds = self._cold_start_seconds
                    after_rss = _peak_process_rss_bytes()
                    self._load_memory_delta_bytes = max(0, after_rss - before_rss)

            pairs = list(zip([query_text] * window_size, product_texts, strict=True))
            raw_scores = self._scorer.score(pairs, batch_size=self.config.batch_size)
            scores = self._validate_scores(raw_scores, window_size)
            ranked = sorted(
                zip(window, scores, range(window_size), strict=True),
                key=lambda item: (-item[1], item[2]),
            )
            for candidate, score, _ in ranked:
                candidate.semantic_score = score
            self._scored_candidate_count += window_size
            self._max_scored_candidates = max(self._max_scored_candidates, window_size)
            return [item[0] for item in ranked] + original[window_size:]
        except Exception as error:
            self._failure_count += 1
            if not self._failure_logged:
                self._failure_logged = True
                LOGGER.warning(
                    "semantic reranking failed; preserving candidate order: %s", error
                )
            return original
        finally:
            query_seconds = max(
                0.0, time.perf_counter() - started - excluded_load_seconds
            )
            self._latencies_ms.append(query_seconds * 1000.0)
            self._peak_process_rss_bytes = max(
                self._peak_process_rss_bytes, _peak_process_rss_bytes()
            )

    @staticmethod
    def _validate_scores(values: object, expected_count: int) -> list[float]:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError("model scores must be a sequence")
        scores: list[float] = []
        for value in values:
            # Some cross-encoders return an N x 1 sequence.
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                if len(value) != 1:
                    raise ValueError("model must return one score per pair")
                value = value[0]
            if isinstance(value, bool):
                raise ValueError("model returned a non-numeric score")
            score = float(value)
            if not math.isfinite(score):
                raise ValueError("model returned a non-finite score")
            scores.append(score)
        if len(scores) != expected_count:
            raise ValueError(
                f"model returned {len(scores)} scores for {expected_count} pairs"
            )
        return scores

    def metrics_snapshot(self) -> dict[str, object]:
        resolved_revision = getattr(self._scorer, "resolved_model_revision", None)
        p95 = _p95(self._latencies_ms)
        return {
            "enabled": self.config.enabled,
            "model": self._scorer.model_name,
            "configured_model_revision": self._scorer.model_revision,
            "resolved_model_revision": resolved_revision,
            "candidate_count": self.config.candidate_count,
            "batch_size": self.config.batch_size,
            "max_length": self.config.max_length,
            "model_size_bytes": self._scorer.model_size_bytes,
            "model_size_scope": "in-memory parameter tensors",
            "cold_start_seconds": self._cold_start_seconds,
            "model_cache_hit": self._scorer.cache_hit,
            "model_load_peak_rss_delta_bytes": self._load_memory_delta_bytes,
            "query_count": self._query_count,
            "scored_candidate_count": self._scored_candidate_count,
            "max_scored_candidates_per_query": self._max_scored_candidates,
            "average_reranking_latency_ms": (
                statistics.fmean(self._latencies_ms) if self._latencies_ms else None
            ),
            "p95_reranking_latency_ms": p95,
            "peak_process_rss_bytes_observed": self._peak_process_rss_bytes,
            "failure_count": self._failure_count,
            "latency_scope": (
                "candidate text lookup, batched inference, score validation, and "
                "sorting; model loading is excluded and reported as cold start"
            ),
            "product_text_builder": ProductTextBuilder.version,
        }
