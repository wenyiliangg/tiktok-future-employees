"""Evaluator-facing agent integrating active-state lexical and dense retrieval."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from dense_retrieval import DEFAULT_MODEL_NAME, DEFAULT_MODEL_REVISION, DenseRetriever

from .conversation_state import ConversationStateManager, SearchQuery
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
from .lexical_retriever import LexicalRetriever

LOGGER = logging.getLogger(__name__)


DenseFactory = Callable[[], Any]


class Agent:
    """Stateful search agent with lexical, dense, and fixed-hybrid modes."""

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
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.config = config or HybridRetrievalConfig()
        self._state = ConversationStateManager()
        self._lexical: Any | None = lexical_retriever
        self._dense: Any | None = dense_retriever
        self._dense_unavailable = False
        self._dense_failure_logged = False

        catalog_ids = self._load_catalog_ids(self.catalog_path)
        self._catalog_ids = catalog_ids
        self._catalog_view = catalog_view or InMemoryCatalogView.from_jsonl(
            self.catalog_path
        )
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
            self.config.mode in {RetrievalMode.LEXICAL, RetrievalMode.HYBRID}
            and self._lexical is None
        ):
            self._lexical = LexicalRetriever.from_jsonl(self.catalog_path)
        if self.config.mode is RetrievalMode.DENSE and self._dense is None:
            self._dense = self._dense_factory()
        elif self.config.mode is RetrievalMode.HYBRID and self._dense is None:
            try:
                self._dense = self._dense_factory()
            except Exception as error:
                self._dense_unavailable = True
                self._log_dense_failure(error)

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

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        query = self._state.update(session_id, user_message, turn)
        limit = min(max(0, top_k), self.config.final_candidate_count)
        candidates = self._retrieve(query, limit) if limit else []
        return {
            "message": "Here are the closest matches I found.",
            "ask_attribute": None,
            "recommendations": [
                {"parent_asin": item.parent_asin} for item in candidates
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def _retrieve(self, query: SearchQuery, limit: int) -> list[Candidate]:
        mode = self.config.mode
        pool_limit = max(limit, self.config.rerank_candidate_count)
        if mode is RetrievalMode.LEXICAL:
            lexical = cast(
                list[RankedResult],
                self._require_lexical().retrieve(
                    query, top_n=self.config.lexical_candidate_count
                ),
            )
            pool = rank_single_source(lexical, mode, self._catalog_ids, pool_limit)
            return self._reranker.rerank(
                query, pool, self._catalog_view, top_k=limit
            )

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
            return self._reranker.rerank(
                query, pool, self._catalog_view, top_k=limit
            )

        lexical = cast(
            list[RankedResult],
            self._require_lexical().retrieve(
                query, top_n=self.config.lexical_candidate_count
            ),
        )
        dense = self._safe_dense_results(query.text)
        merged = merge_candidates(lexical, dense, self._catalog_ids)
        pool = reciprocal_rank_fusion(merged, self.config, limit=pool_limit)
        return self._reranker.rerank(
            query, pool, self._catalog_view, top_k=limit
        )

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
        except Exception as error:
            # This boundary deliberately protects hybrid retrieval only. Dense-only
            # mode surfaces its configuration/runtime failure to the evaluator.
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
