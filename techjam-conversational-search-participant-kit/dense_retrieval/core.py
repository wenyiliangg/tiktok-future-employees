from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

TEXT_BUILDER_VERSION = "catalog-semantic-text-v1"
CACHE_SCHEMA_VERSION = 1
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
NORMALIZATION_TOLERANCE = 1e-4


class CatalogValidationError(ValueError):
    """Raised when the frozen catalog cannot be indexed one-to-one."""


class EmbeddingValidationError(ValueError):
    """Raised when an encoder returns an unsafe or incompatible matrix."""


class Encoder(Protocol):
    model_name: str
    model_revision: str | None
    embedding_dimension: int | None

    def encode(
        self, texts: Sequence[str], batch_size: int
    ) -> NDArray[np.floating[Any]]:
        """Encode texts into one vector per input text."""


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    parent_asin: str
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class _CatalogSnapshot:
    products: tuple[dict[str, Any], ...]
    parent_asins: tuple[str, ...]
    checksum: str


class ProductTextBuilder:
    """Build deterministic semantic text from participant-visible catalog fields."""

    version = TEXT_BUILDER_VERSION

    # These are the searchable, customer-facing keys that occur in the frozen
    # catalog's ``details`` dictionaries. Operational metadata (dates, ranks,
    # model/part numbers, dimensions, weight, shipping, and battery data) is
    # intentionally excluded.
    SEARCHABLE_DETAIL_KEYS = frozenset(
        {
            "Additional product features",
            "Age Range (Description)",
            "Age Range Description",
            "Band Color",
            "Band Material Type",
            "Band Size",
            "Best uses",
            "Brand",
            "Brand Name",
            "Care instructions",
            "Cartoon Character",
            "Chain Type",
            "Clasp Type",
            "Closure",
            "Closure Type",
            "Collection Name",
            "Collar Style",
            "Color",
            "Color Name",
            "Department",
            "Embellishment",
            "Fabric Type",
            "Fabric cleaning",
            "Fastener Material",
            "Fastener Type",
            "Finish",
            "Finish Type",
            "Finish types",
            "Fit Type",
            "Frame Material",
            "Gem Type",
            "Glove Type",
            "Hair Type",
            "Hand Orientation",
            "Import",
            "Import Designation",
            "Incontinence Protector Type",
            "Indoor/Outdoor Usage",
            "Inner Material",
            "Lifestyle",
            "Lining Description",
            "Material",
            "Material Composition",
            "Material Feature",
            "Material Type",
            "Metal Stamp",
            "Metal Type",
            "Neck Style",
            "Occasion",
            "Outer Material",
            "Pattern",
            "Pocket Description",
            "Primary Stone Gem Type",
            "Product Benefits",
            "Product Care Instructions",
            "Recommended Uses For Product",
            "Reusability",
            "Ring Size",
            "Seasons",
            "Shaft Material",
            "Shape",
            "Shell Type",
            "Shirt form type",
            "Size",
            "Size Map",
            "Skill Level",
            "Sleeve Type",
            "Sole Material",
            "Special Feature",
            "Special Features",
            "Special features",
            "Specific Uses For Product",
            "Specific instructions for use",
            "Sport",
            "Sport Type",
            "Stone Color",
            "Stone Cut",
            "Strap Type",
            "Style",
            "Suggested Users",
            "Target Audience",
            "Target gender",
            "Team Name",
            "Theme",
            "Top Style",
            "Use for",
            "Usage",
            "Water Resistance Depth",
            "Water Resistance Level",
            "material_composition",
        }
    )

    def build(self, product: Mapping[str, Any]) -> str:
        sections: list[tuple[str, object]] = [
            ("Title", product.get("title")),
            ("Category", product.get("categories")),
            ("Brand", product.get("store")),
            ("Features", product.get("features")),
            ("Description", product.get("description")),
        ]
        details = product.get("details")
        if isinstance(details, Mapping):
            searchable_details = {
                str(key): value
                for key, value in details.items()
                if str(key) in self.SEARCHABLE_DETAIL_KEYS
            }
            sections.append(("Attributes", searchable_details))

        rendered: list[str] = []
        for label, value in sections:
            normalized = self._normalize(value)
            if normalized:
                rendered.append(f"{label}: {normalized}")
        return "\n".join(rendered)

    @classmethod
    def _normalize(cls, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, Mapping):
            parts: list[str] = []
            for key in sorted(value, key=lambda item: str(item)):
                normalized_key = cls._clean_scalar(key)
                normalized_value = cls._normalize(value[key])
                if normalized_key and normalized_value:
                    parts.append(f"{normalized_key}: {normalized_value}")
            return "; ".join(parts)
        if isinstance(value, (list, tuple)):
            parts = [cls._normalize(item) for item in value]
            return "; ".join(item for item in parts if item)
        if isinstance(value, (set, frozenset)):
            parts = sorted(
                item for item in (cls._normalize(item) for item in value) if item
            )
            return "; ".join(parts)
        return cls._clean_scalar(value)

    @staticmethod
    def _clean_scalar(value: object) -> str:
        return " ".join(str(value).split()).strip()


class SentenceTransformerEncoder:
    """Lazy CPU sentence-transformers adapter used by production retrieval."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        revision: str | None = DEFAULT_MODEL_REVISION,
        device: str = "cpu",
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must be non-empty")
        self.model_name = model_name
        self.model_revision = revision
        self.resolved_model_revision: str | None = None
        self.embedding_dimension: int | None = None
        self._device = device
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "sentence-transformers is required for the production encoder; "
                    "install dependencies from requirements.txt"
                ) from error
            kwargs: dict[str, object] = {
                "device": self._device,
                "local_files_only": True,
                "trust_remote_code": False,
            }
            if self.model_revision is not None:
                kwargs["revision"] = self.model_revision
            self._model = SentenceTransformer(self.model_name, **kwargs)
            dimension_getter = getattr(self._model, "get_embedding_dimension", None)
            if dimension_getter is None:
                dimension_getter = self._model.get_sentence_embedding_dimension
            dimension = dimension_getter()
            self.embedding_dimension = int(dimension) if dimension is not None else None
            first_module = self._model[0] if len(self._model) else None
            config = getattr(getattr(first_module, "auto_model", None), "config", None)
            detected_revision = getattr(config, "_commit_hash", None)
            if isinstance(detected_revision, str) and detected_revision:
                self.resolved_model_revision = detected_revision
        return self._model

    def encode(
        self, texts: Sequence[str], batch_size: int
    ) -> NDArray[np.floating[Any]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        model = self._load_model()
        result = model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(result)


class DenseRetriever:
    """Exact in-memory cosine retrieval over a catalog-aligned matrix."""

    def __init__(
        self,
        embeddings: NDArray[np.float32],
        parent_asins: Sequence[str],
        encoder: Encoder,
    ) -> None:
        validated = _validate_normalized_embeddings(
            embeddings, expected_rows=len(parent_asins)
        )
        _validate_parent_asins(parent_asins)
        self._embeddings = np.array(validated, dtype=np.float32, order="C", copy=True)
        self._embeddings.flags.writeable = False
        self._parent_asins = tuple(parent_asins)
        self._parent_asin_set = frozenset(parent_asins)
        self._encoder = encoder

    @property
    def catalog_size(self) -> int:
        return len(self._parent_asins)

    @property
    def embedding_dimension(self) -> int:
        return int(self._embeddings.shape[1])

    @property
    def embedding_nbytes(self) -> int:
        return int(self._embeddings.nbytes)

    @classmethod
    def from_catalog(
        cls,
        catalog_path: str | Path,
        *,
        cache_path: str | Path,
        encoder: Encoder | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        model_revision: str | None = DEFAULT_MODEL_REVISION,
        batch_size: int = 64,
        text_builder: ProductTextBuilder | None = None,
        rebuild_cache: bool = False,
    ) -> DenseRetriever:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        snapshot = _load_catalog(catalog_path)
        selected_encoder: Encoder = encoder or SentenceTransformerEncoder(
            model_name, revision=model_revision
        )
        builder = text_builder or ProductTextBuilder()
        cache = Path(cache_path)
        expected = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "catalog_checksum": snapshot.checksum,
            "catalog_product_count": len(snapshot.parent_asins),
            "model_name": selected_encoder.model_name,
            "model_revision": selected_encoder.model_revision,
            "text_builder_version": builder.version,
            "dtype": "float32",
            "normalized": True,
        }

        loaded = (
            None
            if rebuild_cache
            else _load_cache(cache, expected, snapshot.parent_asins)
        )
        if loaded is None:
            texts = [builder.build(product) for product in snapshot.products]
            raw_embeddings = selected_encoder.encode(texts, batch_size=batch_size)
            embeddings = _normalize_embeddings(raw_embeddings, expected_rows=len(texts))
            expected["model_revision"] = selected_encoder.model_revision
            metadata = {
                **expected,
                "resolved_model_revision": getattr(
                    selected_encoder, "resolved_model_revision", None
                ),
                "embedding_dimension": int(embeddings.shape[1]),
            }
            _write_cache_atomic(cache, embeddings, snapshot.parent_asins, metadata)
        else:
            embeddings = loaded

        encoder_dimension = selected_encoder.embedding_dimension
        if encoder_dimension is not None and embeddings.shape[1] != encoder_dimension:
            raise EmbeddingValidationError(
                f"cached embedding dimension {embeddings.shape[1]} does not match "
                f"encoder dimension {encoder_dimension}"
            )
        return cls(embeddings, snapshot.parent_asins, selected_encoder)

    def retrieve(self, query_text: str, top_n: int = 200) -> list[RetrievalResult]:
        if not isinstance(query_text, str):
            raise TypeError("query_text must be a string")
        query = query_text.strip()
        if not query:
            raise ValueError("query_text must not be empty or whitespace-only")
        if not isinstance(top_n, int) or isinstance(top_n, bool):
            raise TypeError("top_n must be an integer")
        if top_n <= 0:
            return []

        raw_query = self._encoder.encode([query], batch_size=1)
        query_matrix = _normalize_embeddings(raw_query, expected_rows=1)
        if query_matrix.shape[1] != self.embedding_dimension:
            raise EmbeddingValidationError(
                f"query embedding dimension {query_matrix.shape[1]} does not match "
                f"catalog dimension {self.embedding_dimension}"
            )
        scores = np.matmul(self._embeddings, query_matrix[0])
        if not np.isfinite(scores).all():
            raise EmbeddingValidationError(
                "cosine similarity produced non-finite scores"
            )

        result_count = min(top_n, self.catalog_size)
        # Stable sorting makes the original catalog row order the secondary key.
        ordered_indices = np.argsort(-scores, kind="stable")[:result_count]
        results = [
            RetrievalResult(
                parent_asin=self._parent_asins[int(index)],
                score=float(scores[int(index)]),
                rank=rank,
            )
            for rank, index in enumerate(ordered_indices, start=1)
        ]
        if len({result.parent_asin for result in results}) != len(results):
            raise RuntimeError("retrieval produced duplicate parent_asin values")
        if any(result.parent_asin not in self._parent_asin_set for result in results):
            raise RuntimeError("retrieval produced an ASIN outside the indexed catalog")
        return results


def _load_catalog(catalog_path: str | Path) -> _CatalogSnapshot:
    path = Path(catalog_path)
    products: list[dict[str, Any]] = []
    parent_asins: list[str] = []
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                if not raw_line.strip():
                    continue
                try:
                    product = json.loads(raw_line)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    raise CatalogValidationError(
                        f"invalid catalog JSON on line {line_number}: {error}"
                    ) from error
                if not isinstance(product, dict):
                    raise CatalogValidationError(
                        f"catalog line {line_number} is not a JSON object"
                    )
                products.append(product)
                parent_asins.append(_catalog_asin(product, line_number))
    except OSError as error:
        raise CatalogValidationError(
            f"unable to read catalog {path}: {error}"
        ) from error
    if not products:
        raise CatalogValidationError("catalog contains no products")
    _validate_parent_asins(parent_asins)
    return _CatalogSnapshot(tuple(products), tuple(parent_asins), digest.hexdigest())


def _catalog_asin(product: Mapping[str, Any], line_number: int) -> str:
    value = product.get("parent_asin")
    if not isinstance(value, str) or not value.strip():
        raise CatalogValidationError(
            f"catalog line {line_number} has a missing or non-string parent_asin"
        )
    if value != value.strip():
        raise CatalogValidationError(
            f"catalog line {line_number} has surrounding whitespace in parent_asin"
        )
    return value


def _validate_parent_asins(parent_asins: Sequence[str]) -> None:
    seen: set[str] = set()
    for row_index, parent_asin in enumerate(parent_asins):
        if not isinstance(parent_asin, str) or not parent_asin:
            raise CatalogValidationError(
                f"indexed row {row_index} has an empty parent_asin"
            )
        if parent_asin in seen:
            raise CatalogValidationError(
                f"duplicate parent_asin {parent_asin!r} at indexed row {row_index}"
            )
        seen.add(parent_asin)


def _normalize_embeddings(values: object, *, expected_rows: int) -> NDArray[np.float32]:
    array = np.asarray(values)
    if array.ndim != 2:
        raise EmbeddingValidationError(
            f"encoder must return a 2D matrix, received shape {array.shape}"
        )
    if array.shape[0] != expected_rows:
        raise EmbeddingValidationError(
            f"encoder returned {array.shape[0]} rows for {expected_rows} texts"
        )
    if array.shape[1] <= 0:
        raise EmbeddingValidationError("embedding dimension must be positive")
    converted = np.asarray(array, dtype=np.float32)
    if not np.isfinite(converted).all():
        raise EmbeddingValidationError("embeddings contain non-finite values")
    norms = np.linalg.norm(converted, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= np.finfo(np.float32).eps):
        raise EmbeddingValidationError(
            "embeddings contain a zero-norm or invalid vector"
        )
    normalized = converted / norms[:, np.newaxis]
    return np.asarray(normalized, dtype=np.float32, order="C")


def _validate_normalized_embeddings(
    values: object, *, expected_rows: int
) -> NDArray[np.float32]:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[0] != expected_rows or array.shape[1] <= 0:
        raise EmbeddingValidationError(
            f"invalid cached embedding shape {array.shape}; expected {expected_rows} rows"
        )
    if array.dtype != np.dtype("float32"):
        raise EmbeddingValidationError(
            f"cached embeddings must be float32, received {array.dtype}"
        )
    if not np.isfinite(array).all():
        raise EmbeddingValidationError("cached embeddings contain non-finite values")
    norms = np.linalg.norm(array, axis=1)
    if not np.allclose(
        norms, 1.0, rtol=NORMALIZATION_TOLERANCE, atol=NORMALIZATION_TOLERANCE
    ):
        raise EmbeddingValidationError("cached embeddings are not unit-normalized")
    return np.asarray(array, dtype=np.float32)


def _load_cache(
    cache_path: Path,
    expected_metadata: Mapping[str, object],
    expected_parent_asins: Sequence[str],
) -> NDArray[np.float32] | None:
    try:
        with np.load(cache_path, allow_pickle=False) as archive:
            required = {"embeddings", "parent_asins", "metadata"}
            if set(archive.files) != required:
                return None
            metadata_value = archive["metadata"]
            if metadata_value.shape != () or metadata_value.dtype.kind not in {
                "U",
                "S",
            }:
                return None
            metadata = json.loads(str(metadata_value.item()))
            if not isinstance(metadata, dict):
                return None
            if any(
                metadata.get(key) != value for key, value in expected_metadata.items()
            ):
                return None
            embeddings = archive["embeddings"]
            stored_asins_array = archive["parent_asins"]
            if stored_asins_array.ndim != 1 or stored_asins_array.dtype.kind not in {
                "U",
                "S",
            }:
                return None
            stored_asins = tuple(str(value) for value in stored_asins_array.tolist())
            if stored_asins != tuple(expected_parent_asins):
                return None
            if metadata.get("catalog_product_count") != len(stored_asins):
                return None
            if metadata.get("embedding_dimension") != (
                int(embeddings.shape[1]) if embeddings.ndim == 2 else None
            ):
                return None
            return np.array(
                _validate_normalized_embeddings(
                    embeddings, expected_rows=len(stored_asins)
                ),
                dtype=np.float32,
                order="C",
                copy=True,
            )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_cache_atomic(
    cache_path: Path,
    embeddings: NDArray[np.float32],
    parent_asins: Sequence[str],
    metadata: Mapping[str, object],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            dir=cache_path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez(
                handle,
                embeddings=np.asarray(embeddings, dtype=np.float32),
                parent_asins=np.asarray(parent_asins, dtype=np.str_),
                metadata=np.asarray(
                    json.dumps(metadata, sort_keys=True, separators=(",", ":")),
                    dtype=np.str_,
                ),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, cache_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
