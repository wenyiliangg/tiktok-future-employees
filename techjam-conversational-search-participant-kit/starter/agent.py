"""Evaluator-facing agent with fixed and route-aware retrieval modes."""

from __future__ import annotations

import copy
import json
import logging
import math
import re
import statistics
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from dense_retrieval import DEFAULT_MODEL_NAME, DEFAULT_MODEL_REVISION, DenseRetriever

from .ambiguity_analysis import AmbiguityAnalyzer
from .bm25_anchor import BM25AnchorRetriever
from .clarification_controller import (
    ClarificationController,
    compose_clarification_response,
    is_explicit_no_preference,
    normalize_attribute,
)
from .clarification_policies import load_clarification_policy_registry
from .contextual_retrieval import (
    ContextualRetrievalPolicy,
    policy_by_id,
    rank_contextual_candidates,
)
from .conversation_state import (
    OVERRIDE_CUE_RE,
    ConversationStateManager,
    SearchQuery,
    explicit_attribute_mentions,
)
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
from .selective_clarification import SelectiveClarificationConfig

LOGGER = logging.getLogger(__name__)
NEGATIVE_FEEDBACK_RE = re.compile(
    r"\b(?:not quite right|not right|none of (?:these|those)|do not like|don't like|not what i)\b",
    re.IGNORECASE,
)
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
        anchor_retriever: Any | None = None,
        contextual_policy: ContextualRetrievalPolicy | None = None,
        clarification_config: SelectiveClarificationConfig | None = None,
        ambiguity_analyzer: Any | None = None,
        clarification_controller: Any | None = None,
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
        self._anchor: Any | None = anchor_retriever
        self._contextual_policy = contextual_policy or policy_by_id(
            "contextual.browsing-dense.v1"
        )
        if clarification_config is None:
            default_clarification_policy = (
                load_clarification_policy_registry().runtime_default
            )
            self._clarification_config = default_clarification_policy.clarification
            default_controller_config = default_clarification_policy.controller
        else:
            self._clarification_config = clarification_config
            default_controller_config = None
        self._ambiguity_analyzer = ambiguity_analyzer or AmbiguityAnalyzer()
        self._clarification_controller = (
            clarification_controller
            or ClarificationController(default_controller_config)
        )
        self._known_negative_ids: dict[str, set[str]] = {}
        self._last_recommended_ids: dict[str, tuple[str, ...]] = {}
        self._active_raw_intent: dict[str, str] = {}
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
        self._clarification_candidates: dict[str, list[Candidate]] = {}
        self._contextual_routes: dict[str, str] = {}
        self._clarification_disabled_sessions: set[str] = set()
        self._clarification_question_counts: Counter[str] = Counter()
        self._clarification_route_counts: Counter[str] = Counter()
        self._clarification_resolution_counts: Counter[str] = Counter()
        self._clarification_failure_count = 0

        self._catalog_ids = self._load_catalog_ids(self.catalog_path)
        self._catalog_view = catalog_view or InMemoryCatalogView.from_jsonl(
            self.catalog_path
        )
        # Experimental reranking remains opt-in. Protected BM25 and contextual
        # modes never pass their prefix through this component.
        self._reranker = reranker or FeatureReranker(reranker_config)

        if (
            self.config.mode
            in {
                RetrievalMode.BM25,
                RetrievalMode.ANCHORED,
                RetrievalMode.CONTEXTUAL,
            }
            and self._anchor is None
        ):
            self._anchor = BM25AnchorRetriever(self.catalog_path)

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

        if (
            self.config.mode is RetrievalMode.ROUTE_AWARE
            and self.config.enable_boundary_fallback
            and self._fallback is None
        ):
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
        self._known_negative_ids[session_id] = set()
        self._last_recommended_ids[session_id] = ()
        self._active_raw_intent[session_id] = ""
        self._clarification_candidates[session_id] = []
        self._contextual_routes.pop(session_id, None)
        self._clarification_disabled_sessions.discard(session_id)
        if self._clarification_config.enabled:
            try:
                self._clarification_controller.reset(session_id)
            except Exception as error:  # noqa: BLE001 - clarification-only boundary
                self._clarification_disabled_sessions.add(session_id)
                self._record_clarification_failure(
                    session_id, "controller_reset", error
                )

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if self._clarification_config.enabled:
            self._resolve_pending_clarification(session_id, user_message)

        if self.config.mode is RetrievalMode.CONTEXTUAL:
            if OVERRIDE_CUE_RE.search(user_message):
                self._known_negative_ids.setdefault(session_id, set()).clear()
                self._last_recommended_ids[session_id] = ()
                self._active_raw_intent[session_id] = user_message
            elif NEGATIVE_FEEDBACK_RE.search(user_message):
                self._known_negative_ids.setdefault(session_id, set()).update(
                    self._last_recommended_ids.get(session_id, ())
                )
            else:
                self._active_raw_intent[session_id] = user_message

        query = self._state.update(session_id, user_message, turn)
        limit = min(max(0, top_k), self.config.final_candidate_count)
        if self._clarification_config.enabled and not limit:
            self._clarification_candidates[session_id] = []
        candidates = self._retrieve(query, limit, session_id, turn) if limit else []
        self._last_candidates[session_id] = copy.deepcopy(candidates)
        self._last_recommended_ids[session_id] = tuple(
            candidate.parent_asin for candidate in candidates
        )
        response: dict[str, object] = {
            "message": "Here are the closest matches I found.",
            "ask_attribute": None,
            "recommendations": [
                {"parent_asin": item.parent_asin} for item in candidates
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
        if not self._clarification_config.enabled:
            return response
        return self._attach_clarification(response, session_id=session_id, turn=turn)

    def _resolve_pending_clarification(
        self, session_id: str, user_message: str
    ) -> None:
        """Resolve only an explicit parser-recognized answer or explicit decline."""

        if session_id in self._clarification_disabled_sessions:
            return
        try:
            clarification_state = self._clarification_controller.state_for(session_id)
            pending = getattr(clarification_state, "pending_attribute", None)
            if pending is None:
                return
            if is_explicit_no_preference(user_message):
                if self._clarification_controller.record_resolution(
                    session_id, pending, "no_preference"
                ):
                    self._clarification_resolution_counts["declined"] += 1
                return
            recognized = {
                normalize_attribute(attribute)
                for attribute in explicit_attribute_mentions(user_message)
            }
            if (
                pending in recognized
                and self._clarification_controller.record_resolution(
                    session_id, pending, "answered"
                )
            ):
                self._clarification_resolution_counts["answered"] += 1
        except Exception as error:  # noqa: BLE001 - clarification-only boundary
            self._record_clarification_failure(session_id, "resolve_pending", error)

    def _attach_clarification(
        self,
        response: Mapping[str, object],
        *,
        session_id: str,
        turn: int,
    ) -> dict[str, object]:
        """Fail closed while composing one post-retrieval clarification."""

        config = self._clarification_config
        if (
            session_id in self._clarification_disabled_sessions
            or self.config.mode is not RetrievalMode.CONTEXTUAL
            or self._contextual_policy.policy_id != config.required_retrieval_policy_id
        ):
            return dict(response)
        try:
            candidates = self._clarification_candidates.get(session_id, [])[
                : config.analysis_candidate_limit
            ]
            catalog = self._clarification_catalog(candidates)
            active_state = self._state.state_for(session_id)
            opportunity = self._ambiguity_analyzer.analyze(
                candidates, catalog, active_state
            )
            route = self._contextual_routes.get(session_id, "uncertain")
            if not config.is_eligible(route, len(candidates), opportunity):
                return dict(response)
            prompt = self._clarification_controller.build_prompt(
                session_id, opportunity.attribute, active_state, turn
            )
            if prompt is None:
                return dict(response)
            composed = compose_clarification_response(response, prompt)
            attribute = str(prompt.ask_attribute)
            self._clarification_question_counts[attribute] += 1
            self._clarification_route_counts[route] += 1
            return composed
        except Exception as error:  # noqa: BLE001 - clarification-only boundary
            self._record_clarification_failure(session_id, "attach_question", error)
            return dict(response)

    def _clarification_catalog(
        self, candidates: list[Candidate]
    ) -> dict[str, Mapping[Any, object]]:
        """Adapt metadata already held by the bounded in-memory catalog view."""

        catalog: dict[str, Mapping[Any, object]] = {}
        for candidate in candidates:
            document = self._catalog_view.get(candidate.parent_asin)
            if document is None:
                continue
            if isinstance(document, Mapping):
                catalog[candidate.parent_asin] = document
                continue
            metadata = getattr(document, "metadata", None)
            product: dict[str, object] = (
                dict(metadata) if isinstance(metadata, Mapping) else {}
            )
            price = getattr(document, "price", None)
            if price is not None:
                product["price"] = price
            catalog[candidate.parent_asin] = product
        return catalog

    def _record_clarification_failure(
        self, session_id: str, operation: str, error: Exception
    ) -> None:
        self._clarification_failure_count += 1
        LOGGER.warning(
            "clarification %s failed for session %s; returning recommendations only: %s",
            operation,
            session_id,
            error,
        )

    def _retrieve(
        self,
        query: SearchQuery,
        limit: int,
        session_id: str,
        turn: int,
    ) -> list[Candidate]:
        mode = self.config.mode
        if mode is RetrievalMode.BM25:
            return self._retrieve_bm25(query, limit, session_id)
        if mode is RetrievalMode.ANCHORED:
            return self._retrieve_anchored(query, limit, session_id)
        if mode is RetrievalMode.CONTEXTUAL:
            return self._retrieve_contextual(query, limit, session_id)
        if mode is RetrievalMode.ROUTE_AWARE:
            return self._retrieve_route_aware(query, limit, session_id, turn)

        pool_limit = (
            max(limit, self.config.rerank_candidate_count)
            if self.config.enable_feature_reranker
            else limit
        )
        if mode is RetrievalMode.LEXICAL:
            lexical = cast(
                list[RankedResult],
                self._require_lexical().retrieve(
                    query, top_n=self.config.lexical_candidate_count
                ),
            )
            pool = rank_single_source(lexical, mode, self._catalog_ids, pool_limit)
            return self._finish_fixed(query, pool, limit)

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
            return self._finish_fixed(query, pool, limit)

        lexical = cast(
            list[RankedResult],
            self._require_lexical().retrieve(
                query, top_n=self.config.lexical_candidate_count
            ),
        )
        dense = self._safe_dense_results(query.text)
        merged = merge_candidates(lexical, dense, self._catalog_ids)
        pool = reciprocal_rank_fusion(merged, self.config, limit=pool_limit)
        return self._finish_fixed(query, pool, limit)

    def _finish_fixed(
        self, query: SearchQuery, pool: list[Candidate], limit: int
    ) -> list[Candidate]:
        if not self.config.enable_feature_reranker:
            return pool[:limit]
        return self._reranker.rerank(query, pool, self._catalog_view, top_k=limit)

    def _raw_turn_text(self, session_id: str, query: SearchQuery) -> str:
        return self._state.state_for(session_id).raw_current_turn_text or query.text

    def _retrieve_bm25(
        self, query: SearchQuery, limit: int, session_id: str
    ) -> list[Candidate]:
        if self._anchor is None:
            raise RuntimeError("BM25 retriever is unavailable")
        results = self._anchor.retrieve(
            self._raw_turn_text(session_id, query), top_n=limit
        )
        return rank_single_source(
            results, RetrievalMode.LEXICAL, self._catalog_ids, limit
        )

    @staticmethod
    def _soft_backfill_query(query: SearchQuery) -> SearchQuery:
        changes: dict[str, object] = {}
        for name in ("category", "color", "style", "material", "use_case"):
            constraint = getattr(query, name, None)
            if constraint is not None and constraint.strength == "hard":
                changes[name] = replace(constraint, strength="soft")
        price = query.price
        if price is not None and price.strength == "hard":
            changes["price"] = replace(price, strength="soft")
        return replace(query, **changes)

    def _retrieve_anchored(
        self, query: SearchQuery, limit: int, session_id: str
    ) -> list[Candidate]:
        if self._anchor is None:
            raise RuntimeError("BM25 anchor retriever is unavailable")
        raw_text = self._raw_turn_text(session_id, query)
        anchor_results = list(self._anchor.retrieve(raw_text, top_n=limit))
        anchor = rank_single_source(
            anchor_results, RetrievalMode.LEXICAL, self._catalog_ids, limit
        )
        if len(anchor) >= limit:
            return anchor
        soft_query = self._soft_backfill_query(query)
        if self._lexical is None:
            self._lexical = LexicalRetriever.from_jsonl(self.catalog_path)
        lexical = list(
            self._lexical.retrieve(
                soft_query, top_n=self.config.lexical_candidate_count
            )
        )
        dense = self._safe_dense_results(raw_text)
        backfill = reciprocal_rank_fusion(
            merge_candidates(lexical, dense, self._catalog_ids),
            self.config,
            limit=self.config.final_candidate_count,
        )
        seen = {item.parent_asin for item in anchor}
        for candidate in backfill:
            if candidate.parent_asin in seen:
                continue
            anchor.append(candidate)
            seen.add(candidate.parent_asin)
            if len(anchor) >= limit:
                break
        return anchor

    def _retrieve_contextual(
        self, query: SearchQuery, limit: int, session_id: str
    ) -> list[Candidate]:
        if self._anchor is None:
            raise RuntimeError("BM25 anchor retriever is unavailable")
        policy = self._contextual_policy
        raw_text = self._raw_turn_text(session_id, query)
        active_text = self._active_raw_intent.get(session_id) or raw_text
        anchor = list(self._anchor.retrieve(raw_text, top_n=policy.candidate_count))
        soft_query = self._soft_backfill_query(query)
        state_results: list[RankedResult] = []
        if policy.state_lexical_weight > 0:
            if self._lexical is None:
                self._lexical = LexicalRetriever.from_jsonl(self.catalog_path)
            state_results = list(
                self._lexical.retrieve(soft_query, top_n=policy.candidate_count)
            )
        route_query = replace(query, text=active_text)
        try:
            route = self._router.route(
                self._state.state_for(session_id), route_query
            ).route
        except Exception:  # noqa: BLE001 - router component boundary
            route = "uncertain"
        self._contextual_routes[session_id] = route
        dense_results: list[RankedResult] = []
        if policy.dense_weight > 0 and route in policy.dense_routes:
            dense_results = self._safe_dense_results(active_text)
        ranking_limit = limit
        if self._clarification_config.enabled:
            ranking_limit = min(
                policy.candidate_count,
                max(limit, self._clarification_config.analysis_candidate_limit),
            )
        ranked = rank_contextual_candidates(
            anchor,
            state_results,
            dense_results,
            self._catalog_ids,
            self._known_negative_ids.get(session_id, set()),
            policy,
            limit=ranking_limit,
        )
        if self._clarification_config.enabled:
            self._clarification_candidates[session_id] = copy.deepcopy(
                ranked[: self._clarification_config.analysis_candidate_limit]
            )
        return ranked[:limit]

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
            self.config.enable_boundary_fallback
            and route == "boundary"
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

    def clarification_diagnostics_snapshot(self) -> dict[str, object]:
        """Return aggregate feature-only diagnostics for benchmark reporting."""

        return {
            "enabled": self._clarification_config.enabled,
            "question_count": sum(self._clarification_question_counts.values()),
            "question_counts_by_attribute": dict(
                sorted(self._clarification_question_counts.items())
            ),
            "question_counts_by_observable_route": dict(
                sorted(self._clarification_route_counts.items())
            ),
            "resolution_counts": dict(
                sorted(self._clarification_resolution_counts.items())
            ),
            "clarification_failure_count": self._clarification_failure_count,
            "analysis_candidate_limit": (
                self._clarification_config.analysis_candidate_limit
            ),
        }

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
