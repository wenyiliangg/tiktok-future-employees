from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Literal
import unittest

from starter.lexical_retriever import (
    CatalogDocumentBuilder,
    DEFAULT_FIELD_WEIGHTS,
    LexicalRetrievalConfig,
    LexicalRetriever,
)
from starter.search_models import Constraint, PriceConstraint, SearchQuery


def constraint(value: str, strength: Literal["hard", "soft"] = "hard") -> Constraint:
    return Constraint(
        value=value,
        strength=strength,
        source="current_turn",
        updated_turn=1,
    )


class CatalogDocumentBuilderTest(unittest.TestCase):
    def test_constructs_separate_fields_from_nested_and_list_values(self) -> None:
        product = {
            "parent_asin": "A1",
            "title": "Trail Runner",
            "categories": ["Shoes", ["Athletic", None]],
            "features": ["Lightweight", {"Protection": "Waterproof"}],
            "description": {"summary": ["Built for long hikes"]},
            "store": "Acme",
            "details": {
                "Color": ["Cloud White", None],
                "Construction": {"Outer Material": "Canvas"},
                "Style": "Casual",
                "Recommended Use": {"primary": "Hiking"},
                "Department": "Women",
            },
            "price": {"offers": ["$89.50", 79.0, "unavailable"]},
        }

        document = CatalogDocumentBuilder().build(product)

        self.assertIsNotNone(document)
        assert document is not None
        self.assertEqual(document.parent_asin, "A1")
        self.assertIn("Trail Runner", document.fields["title"])
        self.assertIn("Athletic", document.fields["category"])
        self.assertIn("Waterproof", document.fields["features"])
        self.assertIn("Cloud White", document.fields["color"])
        self.assertIn("Canvas", document.fields["material"])
        self.assertIn("Casual", document.fields["style"])
        self.assertIn("Hiking", document.fields["use_case"])
        self.assertIn("Department Women", document.fields["attributes"])
        self.assertIn("Built for long hikes", document.fields["description"])
        self.assertEqual(document.available_prices, (79.0, 89.5))
        self.assertEqual(document.price, 79.0)

    def test_missing_and_malformed_fields_are_safe(self) -> None:
        builder = CatalogDocumentBuilder()
        document = builder.build(
            {
                "parent_asin": 123,
                "title": None,
                "features": False,
                "description": 42,
                "details": [None, "unstructured"],
                "price": "not available",
            }
        )

        self.assertIsNotNone(document)
        assert document is not None
        self.assertEqual(document.parent_asin, "123")
        self.assertEqual(document.fields["title"], "")
        self.assertIn("42", document.fields["description"])
        self.assertIsNone(document.price)
        self.assertIsNone(builder.build({"title": "missing id"}))
        self.assertIsNone(builder.build(["not", "a", "mapping"]))


class LexicalRetrieverTest(unittest.TestCase):
    def test_title_match_outranks_description_only_match(self) -> None:
        products = [
            {"parent_asin": "TITLE", "title": "opal footwear"},
            {"parent_asin": "DESCRIPTION", "description": "opal footwear"},
        ]
        with LexicalRetriever(products) as retriever:
            results = retriever.retrieve(SearchQuery(text="opal footwear"), top_n=2)

        self.assertEqual([result.parent_asin for result in results], ["TITLE", "DESCRIPTION"])
        self.assertGreater(results[0].score, results[1].score)

    def test_field_weights_are_configurable(self) -> None:
        weights = dict(DEFAULT_FIELD_WEIGHTS)
        weights["title"] = 0.1
        weights["description"] = 20.0
        config = LexicalRetrievalConfig(field_weights=weights)
        products = [
            {"parent_asin": "TITLE", "title": "opal footwear"},
            {"parent_asin": "DESCRIPTION", "description": "opal footwear"},
        ]
        with LexicalRetriever(products, config=config) as retriever:
            results = retriever.retrieve(SearchQuery(text="opal footwear"), top_n=2)

        self.assertEqual(results[0].parent_asin, "DESCRIPTION")

    def test_hard_category_constraint_uses_category_metadata(self) -> None:
        products = [
            {"parent_asin": "BOOTS", "title": "winter item", "categories": ["Shoes", "Boots"]},
            {"parent_asin": "OTHER", "title": "winter boots", "categories": ["Accessories"]},
        ]
        query = SearchQuery(text="winter boots", category=constraint("boots"))
        with LexicalRetriever(products) as retriever:
            results = retriever.retrieve(query)

        self.assertEqual([result.parent_asin for result in results], ["BOOTS"])
        self.assertEqual(results[0].matched_constraints, ("category:boots",))

    def test_structured_only_query_can_retrieve_category_matches(self) -> None:
        products = [
            {"parent_asin": "SHOE", "categories": ["Running Shoes"]},
            {"parent_asin": "HAT", "categories": ["Hats"]},
        ]
        query = SearchQuery(text="", category=constraint("running shoes"))
        with LexicalRetriever(products) as retriever:
            results = retriever.retrieve(query)

        self.assertEqual([result.parent_asin for result in results], ["SHOE"])

    def test_hard_structured_constraint_rejects_mismatch_and_missing_metadata(self) -> None:
        products = [
            {"parent_asin": "CANVAS", "title": "day bag", "material": "Canvas"},
            {"parent_asin": "LEATHER", "title": "day bag", "material": "Leather"},
            {"parent_asin": "UNKNOWN", "title": "day bag"},
        ]
        query = SearchQuery(text="day bag", material=constraint("canvas"))
        with LexicalRetriever(products) as retriever:
            results = retriever.retrieve(query)

        self.assertEqual([result.parent_asin for result in results], ["CANVAS"])

    def test_soft_color_style_material_and_use_case_preferences_boost(self) -> None:
        cases = {
            "color": "white",
            "style": "casual",
            "material": "canvas",
            "use_case": "hiking",
        }
        for field_name, value in cases.items():
            with self.subTest(field=field_name):
                products = [
                    {"parent_asin": "MATCH", "title": "common product", field_name: value},
                    {"parent_asin": "OTHER", "title": "common product", field_name: "different"},
                ]
                query = SearchQuery(
                    text="common product",
                    **{field_name: constraint(value, strength="soft")},
                )
                with LexicalRetriever(products) as retriever:
                    results = retriever.retrieve(query, top_n=2)

                self.assertEqual(results[0].parent_asin, "MATCH")
                self.assertIn(f"{field_name}:{value}", results[0].matched_constraints)
                self.assertIn(f"{field_name}:{value}", results[1].failed_constraints)

    def test_maximum_price_is_inclusive_and_filters_missing_or_malformed_prices(self) -> None:
        products = [
            {"parent_asin": "LOW", "title": "running shoe", "price": 50},
            {"parent_asin": "EDGE", "title": "running shoe", "price": "$80.00"},
            {"parent_asin": "HIGH", "title": "running shoe", "price": 81},
            {"parent_asin": "MISSING", "title": "running shoe"},
            {"parent_asin": "BAD", "title": "running shoe", "price": "call us"},
        ]
        query = SearchQuery(text="running shoe", price=PriceConstraint(maximum=80))
        with LexicalRetriever(products) as retriever:
            results = retriever.retrieve(query)

        self.assertEqual({result.parent_asin for result in results}, {"LOW", "EDGE"})

    def test_minimum_price_is_inclusive(self) -> None:
        products = [
            {"parent_asin": "LOW", "title": "running shoe", "price": 49.99},
            {"parent_asin": "EDGE", "title": "running shoe", "price": 50},
            {"parent_asin": "HIGH", "title": "running shoe", "price": 70},
        ]
        query = SearchQuery(text="running shoe", price=PriceConstraint(minimum=50))
        with LexicalRetriever(products) as retriever:
            results = retriever.retrieve(query)

        self.assertEqual({result.parent_asin for result in results}, {"EDGE", "HIGH"})

    def test_bounded_price_uses_lowest_valid_available_price(self) -> None:
        products = [
            {"parent_asin": "IN", "title": "running shoe", "price": [120, "$80"]},
            {"parent_asin": "BELOW", "title": "running shoe", "price": [20, 70]},
            {"parent_asin": "ABOVE", "title": "running shoe", "price": [101, 150]},
        ]
        query = SearchQuery(
            text="running shoe",
            price=PriceConstraint(minimum=50, maximum=100),
        )
        with LexicalRetriever(products) as retriever:
            results = retriever.retrieve(query)

        self.assertEqual([result.parent_asin for result in results], ["IN"])

    def test_soft_price_boosts_without_filtering(self) -> None:
        products = [
            {"parent_asin": "IN", "title": "running shoe", "price": 60},
            {"parent_asin": "OUT", "title": "running shoe", "price": 120},
            {"parent_asin": "UNKNOWN", "title": "running shoe"},
        ]
        query = SearchQuery(
            text="running shoe",
            price=PriceConstraint(maximum=80, strength="soft"),
        )
        with LexicalRetriever(products) as retriever:
            results = retriever.retrieve(query, top_n=3)

        self.assertEqual(results[0].parent_asin, "IN")
        self.assertEqual({result.parent_asin for result in results}, {"IN", "OUT", "UNKNOWN"})

    def test_explicit_exclusions_filter_known_matches_but_allow_missing_metadata(self) -> None:
        products = [
            {"parent_asin": "LEATHER", "title": "leather day bag"},
            {"parent_asin": "CANVAS", "title": "canvas day bag", "material": "Canvas"},
            {"parent_asin": "UNKNOWN", "title": "day bag"},
            {"parent_asin": "WATERPROOF", "title": "day bag", "features": ["Waterproof"]},
        ]
        query = SearchQuery(
            text="day bag",
            exclusions={"material": {"leather"}, "feature": {"waterproof"}},
        )
        with LexicalRetriever(products) as retriever:
            results = retriever.retrieve(query)

        self.assertEqual({result.parent_asin for result in results}, {"CANVAS", "UNKNOWN"})

    def test_deterministic_order_ranks_and_duplicate_prevention(self) -> None:
        products = [
            {"parent_asin": "B", "title": "same token"},
            {"parent_asin": "A", "title": "same token"},
            {"parent_asin": "A", "title": "same token duplicate"},
            {"parent_asin": "", "title": "same token"},
            {"title": "same token"},
        ]
        query = SearchQuery(text="same token")
        with LexicalRetriever(products) as retriever:
            first = retriever.retrieve(query, top_n=10)
            second = retriever.retrieve(query, top_n=10)
            catalog_ids = retriever.catalog_ids

        self.assertEqual(first, second)
        self.assertEqual([result.parent_asin for result in first], ["A", "B"])
        self.assertEqual([result.rank for result in first], [1, 2])
        self.assertTrue(all(result.parent_asin in catalog_ids for result in first))

    def test_top_n_is_enforced(self) -> None:
        products = [
            {"parent_asin": str(index), "title": "shared item"}
            for index in range(10)
        ]
        with LexicalRetriever(products) as retriever:
            results = retriever.retrieve(SearchQuery(text="shared item"), top_n=3)

        self.assertEqual(len(results), 3)
        self.assertEqual([result.rank for result in results], [1, 2, 3])

    def test_candidate_pool_size_is_configurable(self) -> None:
        products = [
            {"parent_asin": str(index), "title": "shared item"}
            for index in range(10)
        ]
        config = LexicalRetrievalConfig(candidate_pool_size=2)
        with LexicalRetriever(products, config=config) as retriever:
            results = retriever.retrieve(SearchQuery(text="shared item"), top_n=5)

        self.assertEqual(len(results), 2)

    def test_empty_catalog_query_and_filtered_candidate_sets_are_safe(self) -> None:
        with LexicalRetriever([]) as retriever:
            self.assertEqual(retriever.retrieve(SearchQuery(text="shoe")), [])

        products = [{"parent_asin": "A", "title": "shoe", "categories": ["Shoes"]}]
        with LexicalRetriever(products) as retriever:
            self.assertEqual(
                retriever.retrieve(SearchQuery(text="shoe", category=constraint("hats"))),
                [],
            )

    def test_empty_and_incompatible_queries_return_empty_results(self) -> None:
        products = [{"parent_asin": "A", "title": "shoe", "price": 20}]
        with LexicalRetriever(products) as retriever:
            self.assertEqual(retriever.retrieve(SearchQuery(text="")), [])
            self.assertEqual(retriever.retrieve(SearchQuery(text="the and please")), [])
            self.assertEqual(
                retriever.retrieve(
                    SearchQuery(text="shoe", price=PriceConstraint(minimum=100, maximum=50))
                ),
                [],
            )
            self.assertEqual(retriever.retrieve(SearchQuery(text="shoe"), top_n=0), [])

    def test_jsonl_loader_indexes_valid_catalog_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text(
                json.dumps({"parent_asin": "A", "title": "canvas bag"}) + "\n\n",
                encoding="utf-8",
            )
            with LexicalRetriever.from_jsonl(path) as retriever:
                results = retriever.retrieve(SearchQuery(text="canvas bag"))

        self.assertEqual([result.parent_asin for result in results], ["A"])


if __name__ == "__main__":
    unittest.main()
