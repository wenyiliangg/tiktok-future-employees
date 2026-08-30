from __future__ import annotations

import unittest
from collections.abc import Iterator
from collections.abc import Set as AbstractSet
from types import SimpleNamespace

from starter.response_validation import (
    DEFAULT_RECOMMENDATION_MESSAGE,
    validate_response,
)


class MembershipOnlyCatalog(AbstractSet[str]):
    def __init__(self, values: set[str]) -> None:
        self.values = values
        self.membership_checks = 0

    def __contains__(self, value: object) -> bool:
        self.membership_checks += 1
        return value in self.values

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("validator must not scan the catalog")

    def __len__(self) -> int:
        raise AssertionError("validator must not size the catalog")


class ResponseValidationTest(unittest.TestCase):
    def test_empty_pool_and_nonpositive_top_k_are_structurally_valid(self) -> None:
        base = {"message": "ok", "ask_attribute": None, "recommendations": []}
        self.assertEqual(validate_response(base, [], {"A"}, 10)["recommendations"], [])
        self.assertEqual(
            validate_response(base, [SimpleNamespace(parent_asin="A")], {"A"}, 0)[
                "recommendations"
            ],
            [],
        )
        self.assertEqual(
            validate_response(base, [SimpleNamespace(parent_asin="A")], {"A"}, -1)[
                "recommendations"
            ],
            [],
        )

    def test_top_k_is_upper_bound_and_short_pool_is_not_padded(self) -> None:
        candidates = [SimpleNamespace(parent_asin=value) for value in "ABC"]
        base = {"message": "ok", "ask_attribute": None, "recommendations": []}
        self.assertEqual(
            [
                item["parent_asin"]
                for item in validate_response(base, candidates, set("ABC"), 2)[
                    "recommendations"
                ]
            ],
            ["A", "B"],
        )
        self.assertEqual(
            [
                item["parent_asin"]
                for item in validate_response(base, candidates, set("ABC"), 10)[
                    "recommendations"
                ]
            ],
            ["A", "B", "C"],
        )

    def test_invalid_and_duplicate_entries_are_replaced_from_bounded_ranking(
        self,
    ) -> None:
        response = {
            "message": "ok",
            "ask_attribute": None,
            "recommendations": [
                {"parent_asin": "A"},
                {"parent_asin": "INVALID"},
                {"parent_asin": "A"},
            ],
        }
        ranked = [
            SimpleNamespace(parent_asin="A"),
            SimpleNamespace(parent_asin="B"),
            SimpleNamespace(parent_asin="C"),
        ]
        validated = validate_response(response, ranked, {"A", "B", "C"}, 3)
        self.assertEqual(
            [item["parent_asin"] for item in validated["recommendations"]],
            ["A", "B", "C"],
        )

    def test_malformed_response_types_are_canonicalized(self) -> None:
        validated = validate_response(
            {
                "message": 42,
                "ask_attribute": [],
                "recommendations": "A",
                "usage": {"prompt_tokens": True, "completion_tokens": 0},
            },
            [SimpleNamespace(parent_asin="A")],
            {"A"},
            10,
        )
        self.assertEqual(validated["message"], DEFAULT_RECOMMENDATION_MESSAGE)
        self.assertIsNone(validated["ask_attribute"])
        self.assertEqual(validated["recommendations"], [{"parent_asin": "A"}])
        self.assertNotIn("usage", validated)

    def test_invalid_or_inconsistent_clarification_is_dropped_only(self) -> None:
        for response in (
            {
                "message": "Question?",
                "ask_attribute": "price",
                "recommendations": [{"parent_asin": "A"}],
            },
            {
                "message": "",
                "ask_attribute": "color",
                "recommendations": [{"parent_asin": "A"}],
            },
            {"ask_attribute": "color", "recommendations": [{"parent_asin": "A"}]},
        ):
            with self.subTest(response=response):
                validated = validate_response(response, [], {"A"}, 10)
                self.assertIsNone(validated["ask_attribute"])
                self.assertEqual(validated["recommendations"], [{"parent_asin": "A"}])
                self.assertEqual(validated["message"], DEFAULT_RECOMMENDATION_MESSAGE)

    def test_valid_order_scores_question_and_usage_are_preserved(self) -> None:
        response = {
            "message": "Preferred color?",
            "ask_attribute": "color",
            "recommendations": [
                {"parent_asin": "B", "score": 0.5},
                {"parent_asin": "A", "score": float("nan")},
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        }
        validated = validate_response(response, [], {"A", "B"}, 10)
        self.assertEqual(validated["message"], "Preferred color?")
        self.assertEqual(validated["ask_attribute"], "color")
        self.assertEqual(
            validated["recommendations"],
            [{"parent_asin": "B", "score": 0.5}, {"parent_asin": "A"}],
        )
        self.assertEqual(validated["usage"], response["usage"])

    def test_catalog_is_membership_only_and_ranked_pool_is_the_only_backfill(
        self,
    ) -> None:
        catalog = MembershipOnlyCatalog({"A", "B", "UNSCANNED"})
        validated = validate_response(
            {
                "message": "ok",
                "ask_attribute": None,
                "recommendations": [{"parent_asin": "BAD"}],
            },
            [SimpleNamespace(parent_asin="A"), SimpleNamespace(parent_asin="B")],
            catalog,
            10,
        )
        self.assertEqual(
            validated["recommendations"], [{"parent_asin": "A"}, {"parent_asin": "B"}]
        )
        self.assertGreater(catalog.membership_checks, 0)


if __name__ == "__main__":
    unittest.main()
