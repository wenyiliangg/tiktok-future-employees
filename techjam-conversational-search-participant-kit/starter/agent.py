"""Evaluator-facing agent with fixed and route-aware retrieval modes."""

from __future__ import annotations

import copy
import json
import logging
import math
import statistics
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from dense_retrieval import DEFAULT_MODEL_NAME, DEFAULT_MODEL_REVISION, DenseRetriever

from .conversation_state import ConversationStateManager, SearchQuery
from .fallback_candidates import FallbackCandidateGenerator
from .feature_reranker import (
    CatalogView,
    FeatureReranker,
    InMemoryCatalogView,
    RerankerConfig,
)
from .hybrid_retrieval import (
    Candidate,
    HybridRetrievalConfig,
    RankedResult,
    RetrievalMode,
    merge_candidates,
    rank_single_source,
    reciprocal_rank_fusion,
)
from .intent_router import IntentRouter, RoutingDecision
from .lexical_retriever import LexicalRetriever
from .route_aware_retrieval import (
    filter_candidates,
    merge_fallback_candidates,
    route_reciprocal_rank_fusion,
)

LOGGER = logging.getLogger(__name__)
DenseFactory = Callable[[], Any]


class Agent:
    """Stateful search agent preserving fixed modes and adding route awareness."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        config: HybridRetrievalConfig | None = None,
        dense_cache_path: str | Path = "data/.dense-retrieval/catalog-minilm.npz",
        model_name: str = DEFAULT_MODEL_NAME,
        model_revision: str | None = DEFAULT_MODEL_REVISION,
        lexical_retriever: Any | None = None,
        dense_retriever: Any | None = None,
        dense_factory: DenseFactory | None = None,
        reranker: FeatureReranker | None = None,
        reranker_config: RerankerConfig | None = None,
        catalog_view: CatalogView | None = None,
        router: Any | None = None,
        fallback_generator: Any | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.config = config or HybridRetrievalConfig()
        self._state = ConversationStateManager()
        self._lexical: Any | None = lexical_retriever
        self._dense: Any | None = dense_retriever
        self._dense_unavailable = False
        self._dense_failure_logged = False
        self._router = router or IntentRouter()
        self._fallback: Any | None = fallback_generator
        self._fallback_init_error: str | None = None
        self._user_profiles: dict[str, dict] = {}
        self._session_diagnostics: dict[str, list[dict[str, object]]] = {}
        self._last_candidates: dict[str, list[Candidate]] = {}
        self._fallback_cache: dict[str, dict[tuple[object, ...], list[object]]] = {}
        self._route_counts: Counter[str] = Counter()
        self._component_failure_counts: Counter[str] = Counter()
        self._fallback_reasons: Counter[str] = Counter()
        self._routing_failure_count = 0
        self._fallback_attempt_count = 0
        self._fallback_success_count = 0
        self._routing_latencies_ms: list[float] = []
        self._retrieval_latencies_ms: list[float] = []

        self._catalog_ids = self._load_catalog_ids(self.catalog_path)
        self._catalog_view = catalog_view or InMemoryCatalogView.from_jsonl(
            self.catalog_path
        )
        # Issue 2B's legacy modes already used this reranker. Route-aware mode
        # intentionally stops before reranking, as required by Issue 3C.
        self._reranker = reranker or FeatureReranker(reranker_config)

        if dense_factory is None:
            cache_path = Path(dense_cache_path)

            def build_dense() -> DenseRetriever:
                return DenseRetriever.from_catalog(
                    self.catalog_path,
                    cache_path=cache_path,
                    model_name=model_name,
                    model_revision=model_revision,
                )

            self._dense_factory = build_dense
        else:
            self._dense_factory = dense_factory

        if (
            self.config.mode
            in {RetrievalMode.LEXICAL, RetrievalMode.HYBRID, RetrievalMode.ROUTE_AWARE}
            and self._lexical is None
        ):
            self._lexical = LexicalRetriever.from_jsonl(self.catalog_path)
        if self.config.mode is RetrievalMode.DENSE and self._dense is None:
            self._dense = self._dense_factory()
        elif (
            self.config.mode in {RetrievalMode.HYBRID, RetrievalMode.ROUTE_AWARE}
            and self._dense is None
        ):
            try:
                self._dense = self._dense_factory()
            except Exception as error:  # noqa: BLE001 - dense component boundary
                self._dense_unavailable = True
                self._log_dense_failure(error)

        if self.config.mode is RetrievalMode.ROUTE_AWARE and self._fallback is None:
            try:
                self._fallback = FallbackCandidateGenerator.from_jsonl(
                    self.catalog_path
                )
            except (OSError, TypeError, ValueError) as error:
                self._fallback_init_error = f"{type(error).__name__}: {error}"
                LOGGER.warning(
                    "boundary fallback initialization failed; continuing without it: %s",
                    error,
                )

    @staticmethod
    def _load_catalog_ids(path: Path) -> frozenset[str]:
        identifiers: set[str] = set()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line).get("parent_asin")
                if isinstance(value, str) and value:
                    identifiers.add(value)
        return frozenset(identifiers)

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._state.reset(session_id, user_profile)
        self._user_profiles[session_id] = copy.deepcopy(user_profile)
        self._session_diagnostics[session_id] = []
        self._last_candidates[session_id] = []
        self._fallback_cache[session_id] = {}

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        query = self._state.update(session_id, user_message, turn)
        limit = min(max(0, top_k), self.config.final_candidate_count)
        candidates = self._retrieve(query, limit, session_id, turn) if limit else []
        self._last_candidates[session_id] = copy.deepcopy(candidates)
        return {
            "message": "Here are the closest matches I found.",
            "ask_attribute": None,
            "recommendations": [
                {"parent_asin": item.parent_asin} for item in candidates
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def _retrieve(
        self,
        query: SearchQuery,
        limit: int,
        session_id: str,
        turn: int,
    ) -> list[Candidate]:
        mode = self.config.mode
        if mode is RetrievalMode.ROUTE_AWARE:
            return self._retrieve_route_aware(query, limit, session_id, turn)

        pool_limit = max(limit, self.config.rerank_candidate_count)
        if mode is RetrievalMode.LEXICAL:
            lexical = cast(
                list[RankedResult],
                self._require_lexical().retrieve(
                    query, top_n=self.config.lexical_candidate_count
                ),
            )
            pool = rank_single_source(lexical, mode, self._catalog_ids, pool_limit)
            return self._reranker.rerank(query, pool, self._catalog_view, top_k=limit)

        if mode is RetrievalMode.DENSE:
            if not query.text.strip():
                return []
            dense = cast(
                list[RankedResult],
                self._require_dense().retrieve(
                    query.text, top_n=self.config.dense_candidate_count
                ),
            )
            pool = rank_single_source(dense, mode, self._catalog_ids, pool_limit)
            return self._reranker.rerank(query, pool, self._catalog_view, top_k=limit)

        lexical = cast(
            list[RankedResult],
            self._require_lexical().retrieve(
                query, top_n=self.config.lexical_candidate_count
            ),
        )
        dense = self._safe_dense_results(query.text)
        merged = merge_candidates(lexical, dense, self._catalog_ids)
        pool = reciprocal_rank_fusion(merged, self.config, limit=pool_limit)
        return self._reranker.rerank(query, pool, self._catalog_view, top_k=limit)

    def _retrieve_route_aware(
        self,
        query: SearchQuery,
        limit: int,
        session_id: str,
        turn: int,
    ) -> list[Candidate]:
        retrieval_started = time.perf_counter()
        state = self._state.state_for(session_id)
        route_started = time.perf_counter()
        routing_failed = False
        routing_error: str | None = None
        try:
            decision = self._router.route(state, query)
            route = getattr(decision, "route", None)
            if not isinstance(route, str) or route not in self.config.route_policies:
                raise ValueError(f"unsupported router output: {route!r}")
        except Exception as error:  # noqa: BLE001 - router component boundary
            routing_failed = True
            routing_error = f"{type(error).__name__}: {error}"
            route = "uncertain"
            decision = RoutingDecision(
                route="uncertain",
                confidence=0.0,
                reasons=("router_failure_safe_default",),
                policy_id=self.config.policy_for("uncertain").policy_id,
            )
            LOGGER.warning(
                "intent routing failed for session %s turn %s; using uncertain policy: %s",
                session_id,
                turn,
                error,
            )
        routing_latency_ms = (time.perf_counter() - route_started) * 1000.0
        policy = self.config.policy_for(route)

        attempted: list[str] = []
        successful: list[str] = []
        component_failures: dict[str, str] = {}

        attempted.append("lexical")
        try:
            lexical = list(
                self._require_lexical().retrieve(
                    query, top_n=policy.lexical_candidate_count
                )
            )
            successful.append("lexical")
        except Exception as error:  # noqa: BLE001 - lexical component boundary
            lexical = []
            component_failures["lexical"] = f"{type(error).__name__}: {error}"
            LOGGER.warning(
                "lexical retrieval failed for session %s turn %s; continuing safely: %s",
                session_id,
                turn,
                error,
            )

        dense: list[RankedResult] = []
        if query.text.strip():
            attempted.append("dense")
            try:
                dense = list(
                    self._require_dense().retrieve(
                        query.text, top_n=policy.dense_candidate_count
                    )
                )
                successful.append("dense")
            except Exception as error:  # noqa: BLE001 - dense component boundary
                component_failures["dense"] = f"{type(error).__name__}: {error}"
                self._log_dense_failure(error)

        raw_candidate_count = len(lexical) + len(dense)
        merged = merge_candidates(lexical, dense, self._catalog_ids)
        normal_deduplicated_count = len(merged)

        fallback_attempted = False
        fallback_succeeded = False
        fallback_cache_hit = False
        fallback_reason: str | None = None
        fallback_results: list[object] = []
        should_attempt_fallback = (
            route == "boundary"
            and policy.fallback_candidate_count > 0
            and (
                policy.always_attempt_fallback
                or len(merged) < policy.fallback_trigger_count
            )
        )
        if should_attempt_fallback:
            fallback_attempted = True
            fallback_reason = (
                "boundary_route"
                if policy.always_attempt_fallback
                else "insufficient_boundary_candidates"
            )
            attempted.append("fallback")
            if self._fallback is None:
                component_failures["fallback"] = (
                    self._fallback_init_error or "fallback generator unavailable"
                )
            else:
                try:
                    cache_key = self._fallback_cache_key(
                        query,
                        state.removed_constraints,
                        policy.fallback_candidate_count,
                    )
                    cached = self._fallback_cache.setdefault(session_id, {}).get(
                        cache_key
                    )
                    if cached is not None:
                        fallback_cache_hit = True
                        fallback_results = copy.deepcopy(cached)
                    else:
                        fallback_results = list(
                            self._fallback.generate(
                                query=query,
                                user_profile=self._user_profiles.get(session_id, {}),
                                top_n=policy.fallback_candidate_count,
                                removed_constraints=state.removed_constraints,
                            )
                        )
                        self._fallback_cache[session_id][cache_key] = copy.deepcopy(
                            fallback_results
                        )
                    successful.append("fallback")
                    fallback_succeeded = bool(fallback_results)
                except Exception as error:  # noqa: BLE001 - fallback component boundary
                    component_failures["fallback"] = f"{type(error).__name__}: {error}"
                    LOGGER.warning(
                        "boundary fallback failed for session %s turn %s; continuing safely: %s",
                        session_id,
                        turn,
                        error,
                    )

        merged = merge_fallback_candidates(merged, fallback_results, self._catalog_ids)
        merged_with_fallback_count = len(merged)
        filtered, filter_summary = filter_candidates(
            query, merged, self._catalog_view, policy
        )
        candidates = route_reciprocal_rank_fusion(
            filtered, policy, limit=min(limit, policy.final_candidate_count)
        )
        retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000.0

        diagnostic: dict[str, object] = {
            "session_id": session_id,
            "turn": turn,
            "selected_route": route,
            "router_policy_id": getattr(decision, "policy_id", None),
            "applied_policy_id": policy.policy_id,
            "route_confidence": getattr(decision, "confidence", None),
            "route_reasons": tuple(getattr(decision, "reasons", ())),
            "routing_failed": routing_failed,
            "routing_error": routing_error,
            "retrievers_attempted": tuple(attempted),
            "retrievers_successful": tuple(successful),
            "component_failures": component_failures,
            "fallback_attempted": fallback_attempted,
            "fallback_reason": fallback_reason,
            "fallback_succeeded": fallback_succeeded,
            "fallback_cache_hit": fallback_cache_hit,
            "candidate_counts": {
                "lexical_raw": len(lexical),
                "dense_raw": len(dense),
                "fallback_raw": len(fallback_results),
                "before_deduplication": raw_candidate_count,
                "after_normal_deduplication": normal_deduplicated_count,
                "after_fallback_deduplication": merged_with_fallback_count,
                "before_filtering": filter_summary.before_count,
                "after_filtering": filter_summary.after_count,
                "final": len(candidates),
            },
            "filter_counts": {
                "hard_constraint_removals": filter_summary.hard_constraint_removals,
                "exclusion_removals": filter_summary.exclusion_removals,
                "missing_document_removals": filter_summary.missing_document_removals,
            },
            "routing_latency_ms": round(routing_latency_ms, 6),
            "retrieval_latency_ms": round(retrieval_latency_ms, 6),
        }
        self._record_diagnostic(session_id, diagnostic)
        return candidates

    @staticmethod
    def _fallback_cache_key(
        query: SearchQuery,
        removed_constraints: set[str],
        top_n: int,
    ) -> tuple[object, ...]:
        """Represent only deterministic active inputs to fallback generation."""

        constraints = tuple(
            (
                name,
                getattr(constraint, "value", None),
                getattr(constraint, "strength", None),
                getattr(constraint, "source", None),
                getattr(constraint, "updated_turn", None),
            )
            for name in ("category", "color", "style", "material", "use_case")
            if (constraint := getattr(query, name, None)) is not None
        )
        price = getattr(query, "price", None)
        price_key = (
            None
            if price is None
            else (
                getattr(price, "minimum", None),
                getattr(price, "maximum", None),
                getattr(price, "strength", None),
                getattr(price, "source", None),
                getattr(price, "updated_turn", None),
            )
        )
        exclusions = getattr(query, "exclusions", None)
        exclusion_key = (
            tuple(
                (str(name), tuple(sorted(str(value) for value in values)))
                for name, values in sorted(
                    exclusions.items(), key=lambda item: str(item[0])
                )
            )
            if isinstance(exclusions, dict)
            else ()
        )
        return (
            query.text,
            constraints,
            price_key,
            exclusion_key,
            tuple(sorted(removed_constraints)),
            top_n,
        )

    def _record_diagnostic(
        self, session_id: str, diagnostic: dict[str, object]
    ) -> None:
        self._session_diagnostics.setdefault(session_id, []).append(diagnostic)
        self._route_counts[str(diagnostic["selected_route"])] += 1
        if diagnostic["routing_failed"]:
            self._routing_failure_count += 1
        failures = diagnostic["component_failures"]
        if isinstance(failures, dict):
            self._component_failure_counts.update(failures)
        if diagnostic["fallback_attempted"]:
            self._fallback_attempt_count += 1
            reason = diagnostic.get("fallback_reason")
            if isinstance(reason, str):
                self._fallback_reasons[reason] += 1
        if diagnostic["fallback_succeeded"]:
            self._fallback_success_count += 1
        self._routing_latencies_ms.append(
            float(cast(float, diagnostic["routing_latency_ms"]))
        )
        self._retrieval_latencies_ms.append(
            float(cast(float, diagnostic["retrieval_latency_ms"]))
        )

    def diagnostics_snapshot(self, session_id: str | None = None) -> dict[str, object]:
        """Return JSON-safe route diagnostics without changing response payloads."""

        if session_id is not None:
            return {
                "session_id": session_id,
                "turns": copy.deepcopy(self._session_diagnostics.get(session_id, [])),
            }

        def latency_summary(values: list[float]) -> dict[str, float]:
            if not values:
                return {"average_ms": 0.0, "p95_ms": 0.0}
            ordered = sorted(values)
            index = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
            return {
                "average_ms": round(statistics.fmean(values), 6),
                "p95_ms": round(ordered[index], 6),
            }

        return {
            "route_counts": dict(sorted(self._route_counts.items())),
            "routing_failure_count": self._routing_failure_count,
            "component_failure_counts": dict(
                sorted(self._component_failure_counts.items())
            ),
            "fallback_attempt_count": self._fallback_attempt_count,
            "fallback_success_count": self._fallback_success_count,
            "fallback_reasons": dict(sorted(self._fallback_reasons.items())),
            "routing_latency": latency_summary(self._routing_latencies_ms),
            "retrieval_latency": latency_summary(self._retrieval_latencies_ms),
        }

    def last_candidates(self, session_id: str) -> list[Candidate]:
        """Expose copied candidate provenance for debugging and later reranking."""

        return copy.deepcopy(self._last_candidates.get(session_id, []))

    def _require_lexical(self) -> Any:
        if self._lexical is None:
            raise RuntimeError("lexical retriever is unavailable")
        return self._lexical

    def _require_dense(self) -> Any:
        if self._dense_unavailable:
            raise RuntimeError(
                "dense retriever is unavailable after an earlier initialization failure"
            )
        if self._dense is None:
            self._dense = self._dense_factory()
        return self._dense

    def _safe_dense_results(self, query_text: str) -> list[RankedResult]:
        if not query_text.strip():
            return []
        if self._dense_unavailable:
            return []
        try:
            results = cast(
                list[RankedResult],
                self._require_dense().retrieve(
                    query_text, top_n=self.config.dense_candidate_count
                ),
            )
        except Exception as error:  # noqa: BLE001 - dense component boundary
            if self._dense is None:
                self._dense_unavailable = True
            self._log_dense_failure(error)
            return []
        if not results:
            LOGGER.warning(
                "dense retrieval returned no candidates; using lexical candidates"
            )
        return results

    def _log_dense_failure(self, error: Exception) -> None:
        if self._dense_failure_logged:
            return
        self._dense_failure_logged = True
        LOGGER.warning(
            "dense retrieval unavailable; using lexical candidates: %s", error
        )
