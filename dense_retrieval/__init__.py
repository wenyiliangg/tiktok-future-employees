"""Standalone exact dense retrieval for the frozen product catalog."""

from .core import (
    CACHE_SCHEMA_VERSION,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    TEXT_BUILDER_VERSION,
    CatalogValidationError,
    DenseRetriever,
    Encoder,
    EmbeddingValidationError,
    ProductTextBuilder,
    RetrievalResult,
    SentenceTransformerEncoder,
)

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_MODEL_REVISION",
    "TEXT_BUILDER_VERSION",
    "CatalogValidationError",
    "DenseRetriever",
    "Encoder",
    "EmbeddingValidationError",
    "ProductTextBuilder",
    "RetrievalResult",
    "SentenceTransformerEncoder",
]
