from __future__ import annotations

import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

import numpy as np

from dense_retrieval import (
    CatalogValidationError,
    DenseRetriever,
    EmbeddingValidationError,
    ProductTextBuilder,
)
from dense_retrieval import core


class FakeEncoder:
    def __init__(
        self,
        vectors: dict[str, Sequence[float]],
        *,
        model_name: str = "test/model",
        model_revision: str | None = "test-revision",
        embedding_dimension: int | None = 3,
    ) -> None:
        self.vectors = vectors
        self.model_name = model_name
        self.model_revision = model_revision
        self.embedding_dimension = embedding_dimension
        self.calls: list[tuple[tuple[str, ...], int]] = []

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        self.calls.append((tuple(texts), batch_size))
        return np.asarray([self.vectors[text] for text in texts])


class FailingEncoder(FakeEncoder):
    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        raise RuntimeError("intentional encoder failure")


class DenseRetrievalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.catalog_path = self.root / "catalog.jsonl"
        self.cache_path = self.root / "cache" / "catalog.npz"
        self.products = [
            {
                "parent_asin": "ASIN-A",
                "title": "Red cotton shirt",
                "features": ["Soft", "Machine washable"],
                "description": ["Everyday crew neck"],
                "categories": ["Clothing", "Men", "Shirts"],
                "details": {
                    "Material": "Cotton",
                    "Color": "Red",
                    "Date First Available": "yesterday",
                    "Item model number": "secret-123",
                },
                "store": "Example Brand",
                "price": 20.0,
                "average_rating": 4.8,
                "rating_number": 10,
            },
            {
                "parent_asin": "ASIN-B",
                "title": "Blue trail shoe",
                "features": ["Grippy sole"],
                "description": ["For hiking"],
                "categories": ["Clothing", "Shoes"],
                "details": {"Color": "Blue", "Sport Type": "Hiking"},
                "store": "Trail Brand",
                "price": 80.0,
                "average_rating": 4.5,
                "rating_number": 20,
            },
            {
                "parent_asin": "ASIN-C",
                "title": "Green wool hat",
                "features": [],
                "description": [],
                "categories": ["Clothing", "Hats"],
                "details": {"Material": "Wool", "Color": "Green"},
                "store": None,
                "price": None,
                "average_rating": 4.0,
                "rating_number": 2,
            },
        ]
        self._write_catalog(self.products)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_catalog(self, products: Sequence[dict]) -> None:
        self.catalog_path.write_text(
            "".join(json.dumps(product, sort_keys=True) + "\n" for product in products),
            encoding="utf-8",
        )

    def _vectors(
        self, query_vectors: dict[str, Sequence[float]] | None = None
    ) -> dict[str, Sequence[float]]:
        builder = ProductTextBuilder()
        vectors: dict[str, Sequence[float]] = {
            builder.build(self.products[0]): [3.0, 0.0, 0.0],
            builder.build(self.products[1]): [0.0, 4.0, 0.0],
            builder.build(self.products[2]): [1.0, 1.0, 0.0],
        }
        vectors.update(query_vectors or {})
        return vectors

    def _build(
        self,
        encoder: FakeEncoder | None = None,
        **kwargs: object,
    ) -> tuple[DenseRetriever, FakeEncoder]:
        selected_encoder = encoder or FakeEncoder(self._vectors({"query": [2.0, 0.0, 0.0]}))
        retriever = DenseRetriever.from_catalog(
            self.catalog_path,
            cache_path=self.cache_path,
            encoder=selected_encoder,
            batch_size=2,
            **kwargs,
        )
        return retriever, selected_encoder

    def _rewrite_cache(
        self,
        *,
        embeddings: np.ndarray | None = None,
        metadata_updates: dict[str, object] | None = None,
        omit: str | None = None,
    ) -> None:
        with np.load(self.cache_path, allow_pickle=False) as archive:
            values = {name: np.array(archive[name], copy=True) for name in archive.files}
        if embeddings is not None:
            values["embeddings"] = embeddings
        if metadata_updates is not None:
            metadata = json.loads(str(values["metadata"].item()))
            metadata.update(metadata_updates)
            values["metadata"] = np.asarray(json.dumps(metadata), dtype=np.str_)
        if omit is not None:
            values.pop(omit)
        with self.cache_path.open("wb") as handle:
            np.savez(handle, **values)

    def test_text_builder_is_deterministic_labelled_and_non_mutating(self) -> None:
        product = copy.deepcopy(self.products[0])
        original = copy.deepcopy(product)
        builder = ProductTextBuilder()

        first = builder.build(product)
        second = builder.build(product)

        self.assertEqual(first, second)
        self.assertEqual(product, original)
        self.assertEqual(
            first.splitlines(),
            [
                "Title: Red cotton shirt",
                "Category: Clothing; Men; Shirts",
                "Brand: Example Brand",
                "Features: Soft; Machine washable",
                "Description: Everyday crew neck",
                "Attributes: Color: Red; Material: Cotton",
            ],
        )
        self.assertNotIn("secret-123", first)
        self.assertNotIn("yesterday", first)
        self.assertNotIn("rating", first.lower())
        self.assertTrue(builder.version)

    def test_text_builder_handles_missing_and_complex_values_stably(self) -> None:
        builder = ProductTextBuilder()
        product = {
            "title": "  A\n title  ",
            "categories": ["Shoes", None, ""],
            "features": {"z": ["last", None], "a": {"b": "first"}},
            "details": {"Material": ["Wool", "Cotton"], "Color": None},
            "description": None,
        }
        self.assertEqual(
            builder.build(product),
            "Title: A title\n"
            "Category: Shoes\n"
            "Features: a: b: first; z: last\n"
            "Attributes: Material: Wool; Cotton",
        )

    def test_catalog_embeddings_are_aligned_normalized_float32_and_cached(self) -> None:
        retriever, encoder = self._build()
        self.assertEqual(retriever.catalog_size, 3)
        self.assertEqual(retriever.embedding_dimension, 3)
        self.assertEqual(retriever.embedding_nbytes, 36)
        self.assertEqual(len(encoder.calls), 1)
        with np.load(self.cache_path, allow_pickle=False) as archive:
            self.assertEqual(archive["parent_asins"].tolist(), ["ASIN-A", "ASIN-B", "ASIN-C"])
            self.assertEqual(archive["embeddings"].dtype, np.float32)
            np.testing.assert_allclose(
                np.linalg.norm(archive["embeddings"], axis=1), np.ones(3), atol=1e-6
            )

        cached_encoder = FakeEncoder(self._vectors({"query": [1.0, 0.0, 0.0]}))
        cached, _ = self._build(cached_encoder)
        self.assertEqual(cached.catalog_size, 3)
        self.assertEqual(cached_encoder.calls, [])

    def test_missing_and_duplicate_asins_are_rejected_without_encoding(self) -> None:
        for invalid_products in (
            [{**self.products[0], "parent_asin": ""}],
            [self.products[0], {**self.products[1], "parent_asin": "ASIN-A"}],
            [{**self.products[0], "parent_asin": None}],
        ):
            with self.subTest(invalid_products=invalid_products):
                self._write_catalog(invalid_products)
                encoder = FakeEncoder({})
                with self.assertRaises(CatalogValidationError):
                    self._build(encoder, rebuild_cache=True)
                self.assertEqual(encoder.calls, [])

    def test_cache_invalidates_for_catalog_model_and_builder_changes(self) -> None:
        _, initial = self._build()
        self.assertEqual(len(initial.calls), 1)

        changed_products = copy.deepcopy(self.products)
        changed_products[0]["title"] = "Changed title"
        self._write_catalog(changed_products)
        changed_vectors = self._vectors()
        changed_vectors[ProductTextBuilder().build(changed_products[0])] = [3.0, 0.0, 0.0]
        changed_catalog_encoder = FakeEncoder(changed_vectors)
        self._build(changed_catalog_encoder)
        self.assertEqual(len(changed_catalog_encoder.calls), 1)

        self._write_catalog(self.products)
        changed_model = FakeEncoder(self._vectors(), model_name="different/model")
        self._build(changed_model)
        self.assertEqual(len(changed_model.calls), 1)

        changed_revision = FakeEncoder(
            self._vectors(), model_name="different/model", model_revision="new-revision"
        )
        self._build(changed_revision)
        self.assertEqual(len(changed_revision.calls), 1)

        class ChangedBuilder(ProductTextBuilder):
            version = "changed-builder-version"

        changed_builder = ChangedBuilder()
        builder_encoder = FakeEncoder(
            {
                **self._vectors(),
                **{
                    changed_builder.build(product): vector
                    for product, vector in zip(
                        self.products,
                        ([3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [1.0, 1.0, 0.0]),
                    )
                },
            }
        )
        self._build(builder_encoder, text_builder=changed_builder)
        self.assertEqual(len(builder_encoder.calls), 1)

    def test_cache_invalidates_for_dimension_normalization_and_incomplete_data(self) -> None:
        cases = ("dimension", "normalization", "normalization_metadata", "incomplete")
        for case in cases:
            with self.subTest(case=case):
                self.cache_path.unlink(missing_ok=True)
                self._build()
                if case == "dimension":
                    self._rewrite_cache(metadata_updates={"embedding_dimension": 99})
                elif case == "normalization":
                    with np.load(self.cache_path, allow_pickle=False) as archive:
                        bad = np.array(archive["embeddings"], copy=True) * 2
                    self._rewrite_cache(embeddings=bad)
                elif case == "normalization_metadata":
                    self._rewrite_cache(metadata_updates={"normalized": False})
                else:
                    self._rewrite_cache(omit="parent_asins")
                encoder = FakeEncoder(self._vectors())
                self._build(encoder)
                self.assertEqual(len(encoder.calls), 1)

    def test_corrupt_cache_rebuilds_and_failed_rebuild_preserves_old_artifact(self) -> None:
        self.cache_path.parent.mkdir(parents=True)
        self.cache_path.write_bytes(b"not an npz archive")
        encoder = FakeEncoder(self._vectors())
        self._build(encoder)
        self.assertEqual(len(encoder.calls), 1)
        known_good = self.cache_path.read_bytes()

        changed_products = copy.deepcopy(self.products)
        changed_products[0]["title"] = "Forces a rebuild"
        self._write_catalog(changed_products)
        with self.assertRaisesRegex(RuntimeError, "intentional encoder failure"):
            self._build(FailingEncoder({}))
        self.assertEqual(self.cache_path.read_bytes(), known_good)
        self.assertEqual(list(self.cache_path.parent.glob("*.tmp")), [])

    def test_retrieval_normalizes_query_and_ranks_with_stable_ties(self) -> None:
        retriever, encoder = self._build(
            FakeEncoder(self._vectors({"query": [9.0, 9.0, 0.0]}))
        )
        results = retriever.retrieve("  query  ", top_n=3)
        self.assertEqual([item.parent_asin for item in results], ["ASIN-C", "ASIN-A", "ASIN-B"])
        self.assertEqual([item.rank for item in results], [1, 2, 3])
        self.assertAlmostEqual(results[0].score, 1.0, places=6)
        self.assertAlmostEqual(results[1].score, results[2].score, places=6)
        self.assertEqual(encoder.calls[-1], (("query",), 1))

    def test_top_n_bounds_uniqueness_and_catalog_membership(self) -> None:
        retriever, _ = self._build()
        for top_n, expected_count in ((1, 1), (3, 3), (30, 3), (0, 0), (-2, 0)):
            with self.subTest(top_n=top_n):
                results = retriever.retrieve("query", top_n=top_n)
                self.assertEqual(len(results), expected_count)
                self.assertEqual(len({item.parent_asin for item in results}), expected_count)
                self.assertTrue(
                    all(item.parent_asin in {"ASIN-A", "ASIN-B", "ASIN-C"} for item in results)
                )

    def test_invalid_queries_and_query_embeddings_are_rejected(self) -> None:
        retriever, _ = self._build()
        invalid_values = (
            (None, TypeError),
            (5, TypeError),
            ("", ValueError),
            (" \n", ValueError),
        )
        for value, exception in invalid_values:
            with self.subTest(value=value), self.assertRaises(exception):
                retriever.retrieve(value)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            retriever.retrieve("query", top_n=1.5)  # type: ignore[arg-type]

        for vector in ([0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [1.0, 0.0]):
            with self.subTest(vector=vector):
                invalid, _ = self._build(
                    FakeEncoder(self._vectors({"bad": vector})), rebuild_cache=True
                )
                with self.assertRaises(EmbeddingValidationError):
                    invalid.retrieve("bad")

    def test_invalid_catalog_embeddings_are_rejected(self) -> None:
        builder = ProductTextBuilder()
        for vector in ([0.0, 0.0, 0.0], [np.inf, 0.0, 0.0]):
            vectors = self._vectors()
            vectors[builder.build(self.products[1])] = vector
            with self.subTest(vector=vector), self.assertRaises(EmbeddingValidationError):
                self._build(FakeEncoder(vectors), rebuild_cache=True)

    def test_production_search_is_numpy_exact_without_vector_database(self) -> None:
        source = inspect.getsource(core.DenseRetriever.retrieve)
        self.assertIn("np.matmul", source)
        self.assertIn("np.argsort", source)
        module_source = inspect.getsource(core)
        self.assertNotIn("faiss", module_source.lower())


if __name__ == "__main__":
    unittest.main()
