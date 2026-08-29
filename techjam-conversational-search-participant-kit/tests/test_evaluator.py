from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluator.local_evaluator import (
    catalog_index,
    evaluate,
    metric_summary,
    normalize_recommendations,
    recommendation_contract_issues,
    retrieval_configuration_fingerprint,
)
from starter.contextual_retrieval import policy_by_id
from starter.hybrid_retrieval import HybridRetrievalConfig


class EchoTargetAgent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        self.session_id = session_id

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        asin = "A"
        if "B" in user_message:
            asin = "B"
        return {
            "message": "ok",
            "ask_attribute": None,
            "recommendations": [{"parent_asin": asin}],
        }


class InvalidRepeatingAgent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        pass

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        return {
            "message": "question",
            "ask_attribute": "color",
            "recommendations": [
                {"parent_asin": "A"},
                {"parent_asin": "A"},
                {"parent_asin": "INVALID"},
            ],
        }


class EvaluatorTest(unittest.TestCase):
    def test_normalization_preserves_first_valid_unique_order(self) -> None:
        payload = [
            {"parent_asin": "A"},
            {"parent_asin": "bad"},
            {"parent_asin": "A"},
            "B",
            {"parent_asin": "C"},
        ]
        self.assertEqual(
            normalize_recommendations(payload, {"A", "B", "C"}), ["A", "B", "C"]
        )

    def test_metric_summary_assigns_turn_11_to_miss(self) -> None:
        sessions = [
            {"hit": True, "reciprocal_rank": 0.5, "first_hit_turn": 2},
            {"hit": False, "reciprocal_rank": 0.0, "first_hit_turn": None},
        ]
        self.assertEqual(
            metric_summary(sessions),
            {
                "sample_count": 2,
                "hit_rate_at_10": 0.5,
                "mrr": 0.25,
                "mttc": 6.5,
            },
        )

    def test_contract_diagnostics_count_raw_invalid_and_duplicate_asins(self) -> None:
        self.assertEqual(
            recommendation_contract_issues(
                [
                    {"parent_asin": "A"},
                    {"parent_asin": "A"},
                    {"parent_asin": "INVALID"},
                ],
                {"A"},
            ),
            (1, 1),
        )

    def test_retrieval_fingerprint_excludes_feature_flag_state(self) -> None:
        fingerprint, payload = retrieval_configuration_fingerprint(
            HybridRetrievalConfig(),
            policy_by_id("contextual.browsing-dense.v1"),
        )

        self.assertEqual(len(fingerprint), 64)
        self.assertNotIn("selective_clarification", payload)
        self.assertEqual(
            payload["contextual_policy"]["policy_id"],  # type: ignore[index]
            "contextual.browsing-dense.v1",
        )

    def test_evaluate_reports_repeated_question_and_recommendation_issues(self) -> None:
        product = {
            "parent_asin": "B",
            "title": "Blue shoe",
            "categories": ["Shoes"],
        }
        result = evaluate(
            InvalidRepeatingAgent(),
            [
                {
                    "sample_id": "public_contract",
                    "scenario_type": "browsing",
                    "user_profile": {},
                    "ground_truth": {"parent_asin": "B"},
                }
            ],
            {"A", "B"},
            {"B": ["Shoes"]},
            {"B": product},
        )

        diagnostics = result["response_contract_diagnostics"]
        self.assertEqual(diagnostics["clarification_question_count"], 10)
        self.assertEqual(diagnostics["repeated_question_count"], 9)
        self.assertEqual(diagnostics["invalid_asin_count"], 10)
        self.assertEqual(diagnostics["duplicate_recommendation_count"], 10)

    def test_evaluate_derives_hidden_fields_when_public_set_omits_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            catalog_rows = [
                {
                    "parent_asin": "A",
                    "title": "Blue running shoe",
                    "features": ["cotton"],
                    "details": {"department": "womens"},
                    "description": ["walking shoe"],
                    "categories": ["Clothing", "Shoes"],
                    "store": "Example",
                    "average_rating": 4.2,
                    "rating_number": 10,
                    "price": 49.0,
                },
                {
                    "parent_asin": "B",
                    "title": "Black winter boot",
                    "features": ["leather"],
                    "details": {"department": "womens"},
                    "description": ["winter boot"],
                    "categories": ["Clothing", "Boots"],
                    "store": "Example",
                    "average_rating": 4.4,
                    "rating_number": 12,
                    "price": 89.0,
                },
            ]
            catalog_path.write_text(
                "".join(json.dumps(row) + "\n" for row in catalog_rows),
                encoding="utf-8",
            )
            catalog_ids, categories, products = catalog_index(catalog_path)
            samples = [
                {
                    "sample_id": "public_v2_0001",
                    "scenario_type": "buying",
                    "user_profile": {"summary": "x"},
                    "ground_truth": {"parent_asin": "A"},
                }
            ]
            result = evaluate(
                EchoTargetAgent(), samples, catalog_ids, categories, products
            )
            self.assertEqual(result["hit_rate_at_10"], 1.0)


if __name__ == "__main__":
    unittest.main()
