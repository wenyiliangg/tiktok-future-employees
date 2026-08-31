"""Contract-boundary tests shared by conversation state and lexical retrieval.

These fixtures represent Issue 1A output. They intentionally do not parse user
messages or implement conversation-state precedence in this issue.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from importlib.util import find_spec
import unittest

from starter.lexical_retriever import LexicalRetriever
from starter.search_models import (
    Constraint,
    ConstraintSource,
    ConstraintStrength,
    PriceConstraint,
    SearchQuery,
)

HAS_ISSUE_1A = find_spec("starter.conversation_state") is not None


def slot(
    value: str,
    strength: ConstraintStrength,
    source: ConstraintSource,
    updated_turn: int,
) -> Constraint:
    return Constraint(
        value=value,
        strength=strength,
        source=source,
        updated_turn=updated_turn,
    )


class SearchQueryContractIntegrationTest(unittest.TestCase):
    def test_shared_contract_shape_stays_compatible_with_issue_1a(self) -> None:
        self.assertTrue(is_dataclass(Constraint))
        self.assertTrue(is_dataclass(PriceConstraint))
        self.assertTrue(is_dataclass(SearchQuery))
        self.assertEqual(
            [field.name for field in fields(Constraint)],
            ["value", "strength", "source", "updated_turn"],
        )
        self.assertEqual(
            [field.name for field in fields(PriceConstraint)],
            ["minimum", "maximum", "strength", "source", "updated_turn"],
        )
        self.assertEqual(
            [field.name for field in fields(SearchQuery)],
            [
                "text",
                "category",
                "color",
                "style",
                "material",
                "use_case",
                "price",
                "exclusions",
            ],
        )

        query = SearchQuery(text="shoes")
        self.assertIsNone(query.category)
        self.assertIsNone(query.price)
        self.assertIsNone(query.exclusions)

    def test_issue_1a_style_query_flows_directly_into_retrieval(self) -> None:
        products = [
            {
                "parent_asin": "WHITE_CANVAS",
                "title": "White casual canvas sneakers",
                "categories": ["Shoes", "Sneakers"],
                "color": "white",
                "style": "casual",
                "material": "canvas",
                "use_case": "everyday",
                "price": 75,
            },
            {
                "parent_asin": "BLACK_CANVAS",
                "title": "Black casual canvas sneakers",
                "categories": ["Shoes", "Sneakers"],
                "color": "black",
                "style": "casual",
                "material": "canvas",
                "use_case": "everyday",
                "price": 60,
            },
            {
                "parent_asin": "WHITE_LEATHER",
                "title": "White casual leather sneakers",
                "categories": ["Shoes", "Sneakers"],
                "color": "white",
                "style": "casual",
                "material": "leather",
                "use_case": "everyday",
                "price": 70,
            },
            {
                "parent_asin": "EXPENSIVE_WHITE_CANVAS",
                "title": "White casual canvas sneakers",
                "categories": ["Shoes", "Sneakers"],
                "color": "white",
                "style": "casual",
                "material": "canvas",
                "use_case": "everyday",
                "price": 120,
            },
            {
                "parent_asin": "WHITE_CANVAS_BAG",
                "title": "White casual canvas bag",
                "categories": ["Accessories", "Bags"],
                "color": "white",
                "style": "casual",
                "material": "canvas",
                "use_case": "everyday",
                "price": 50,
            },
        ]
        active_query_from_issue_1a = SearchQuery(
            text="casual sneakers",
            category=slot("sneakers", "hard", "current_turn", 3),
            color=slot("white", "soft", "current_turn", 3),
            style=slot("casual", "soft", "conversation", 2),
            material=slot("canvas", "soft", "current_turn", 3),
            use_case=slot("everyday", "soft", "profile", 0),
            price=PriceConstraint(
                maximum=80,
                strength="hard",
                source="current_turn",
            ),
            exclusions={"material": {"leather"}},
        )

        with LexicalRetriever(products) as retriever:
            results = retriever.retrieve(active_query_from_issue_1a, top_n=10)

        self.assertEqual(
            [result.parent_asin for result in results],
            ["WHITE_CANVAS", "BLACK_CANVAS"],
        )
        self.assertIn("category:sneakers", results[0].matched_constraints)
        self.assertIn("price:<=80", results[0].matched_constraints)
        self.assertIn("color:white", results[0].matched_constraints)
        self.assertIn("color:white", results[1].failed_constraints)

    def test_new_active_query_replaces_old_preference_without_retriever_state(self) -> None:
        products = [
            {
                "parent_asin": "RED",
                "title": "Red trail shoe",
                "categories": ["Shoes"],
                "color": "red",
            },
            {
                "parent_asin": "BLACK",
                "title": "Black trail shoe",
                "categories": ["Shoes"],
                "color": "black",
            },
        ]
        before_override = SearchQuery(
            text="trail shoe",
            category=slot("shoes", "hard", "conversation", 1),
            color=slot("red", "soft", "conversation", 1),
        )
        after_override = SearchQuery(
            text="trail shoe",
            category=slot("shoes", "hard", "conversation", 1),
            color=slot("black", "soft", "current_turn", 4),
        )
        after_negation = SearchQuery(
            text="trail shoe",
            category=slot("shoes", "hard", "conversation", 1),
            exclusions={"color": {"red"}},
        )

        with LexicalRetriever(products) as retriever:
            red_results = retriever.retrieve(before_override, top_n=2)
            black_results = retriever.retrieve(after_override, top_n=2)
            not_red_results = retriever.retrieve(after_negation, top_n=2)

        self.assertEqual(red_results[0].parent_asin, "RED")
        self.assertEqual(black_results[0].parent_asin, "BLACK")
        self.assertEqual([result.parent_asin for result in not_red_results], ["BLACK"])
        assert after_override.color is not None
        self.assertEqual(after_override.color.updated_turn, 4)
        self.assertEqual(after_override.color.source, "current_turn")

    @unittest.skipUnless(
        HAS_ISSUE_1A,
        "Issue 1A conversation_state.py is not present on this branch",
    )
    def test_real_issue_1a_query_producer_feeds_retriever_directly(self) -> None:
        from starter.conversation_state import ConversationStateManager

        products = [
            {
                "parent_asin": "MATCH",
                "title": "White casual canvas sneakers",
                "categories": ["Shoes", "Sneakers"],
                "color": "white",
                "style": "casual",
                "material": "canvas",
                "price": 75,
            },
            {
                "parent_asin": "OTHER_COLOR",
                "title": "Black casual canvas sneakers",
                "categories": ["Shoes", "Sneakers"],
                "color": "black",
                "style": "casual",
                "material": "canvas",
                "price": 70,
            },
            {
                "parent_asin": "EXCLUDED",
                "title": "White casual leather sneakers",
                "categories": ["Shoes", "Sneakers"],
                "color": "white",
                "style": "casual",
                "material": "leather",
                "price": 70,
            },
            {
                "parent_asin": "TOO_EXPENSIVE",
                "title": "White casual canvas sneakers",
                "categories": ["Shoes", "Sneakers"],
                "color": "white",
                "style": "casual",
                "material": "canvas",
                "price": 100,
            },
        ]
        manager = ConversationStateManager()
        manager.reset("real-1a", {})
        manager.update("real-1a", "I need sneakers under $80", 1)
        manager.update("real-1a", "I would prefer white casual canvas", 2)
        active_query = manager.update("real-1a", "Not leather", 3)

        with LexicalRetriever(products) as retriever:
            results = retriever.retrieve(active_query, top_n=10)  # type: ignore[arg-type]

        self.assertEqual(
            [result.parent_asin for result in results],
            ["MATCH", "OTHER_COLOR"],
        )
        self.assertEqual(active_query.exclusions, {"material": {"leather"}})
        self.assertEqual(active_query.price.maximum, 80.0)


if __name__ == "__main__":
    unittest.main()
