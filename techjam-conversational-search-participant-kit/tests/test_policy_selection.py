from __future__ import annotations

import unittest

from benchmarks.select_contextual_policy import assess_policy


def result(*sessions: dict, score: float = 0.1, exceptions: int = 0) -> dict:
    return {
        "sessions": list(sessions),
        "recommended_technical_score": score,
        "evaluation_diagnostics": {"response_exception_count": exceptions},
    }


def session(sample_id: str, turn: int | None, rank: int | None) -> dict:
    return {
        "sample_id": sample_id,
        "hit": turn is not None,
        "first_hit_turn": turn,
        "best_rank": rank,
    }


class PolicyPromotionGateTest(unittest.TestCase):
    def test_passing_challenger_retains_and_improves(self) -> None:
        baseline = result(
            session("retained", 2, 5), session("gain", None, None), score=0.1
        )
        challenger = result(session("retained", 2, 4), session("gain", 3, 2), score=0.2)

        assessment = assess_policy(baseline, challenger)

        self.assertTrue(assessment["passed"])
        self.assertEqual(assessment["gained_session_ids"], ["gain"])

    def test_loss_or_hit_regression_blocks_promotion(self) -> None:
        baseline = result(session("lost", 1, 1), session("later", 2, 3), score=0.1)
        challenger = result(
            session("lost", None, None), session("later", 3, 1), score=0.2
        )

        assessment = assess_policy(baseline, challenger)

        self.assertFalse(assessment["passed"])
        self.assertEqual(assessment["lost_session_ids"], ["lost"])
        self.assertIn("later", assessment["baseline_hit_regressions"])


if __name__ == "__main__":
    unittest.main()
