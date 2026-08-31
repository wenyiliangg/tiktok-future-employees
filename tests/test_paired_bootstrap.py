from __future__ import annotations

import unittest

from benchmarks.paired_bootstrap import compare, paired_bootstrap, session_score


class PairedBootstrapTest(unittest.TestCase):
    @staticmethod
    def session(
        sample_id: str,
        *,
        hit: bool,
        turn: int | None,
        rank: int | None,
    ) -> dict[str, object]:
        return {
            "sample_id": sample_id,
            "scenario_type": "browsing",
            "hit": hit,
            "first_hit_turn": turn,
            "best_rank": rank,
            "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
        }

    def test_session_contribution_matches_metric_formula(self) -> None:
        self.assertAlmostEqual(
            session_score(self.session("a", hit=True, turn=1, rank=1)), 1.0
        )
        self.assertEqual(
            session_score(self.session("b", hit=False, turn=None, rank=None)), 0.0
        )

    def test_compare_is_paired_and_deterministic(self) -> None:
        champion = {
            "sessions": [
                self.session("a", hit=False, turn=None, rank=None),
                self.session("b", hit=True, turn=5, rank=4),
            ]
        }
        candidate = {
            "sessions": [
                self.session("a", hit=True, turn=2, rank=1),
                self.session("b", hit=True, turn=3, rank=2),
            ]
        }

        first = compare(champion, candidate, seed=7, resamples=100)
        second = compare(champion, candidate, seed=7, resamples=100)

        self.assertEqual(first, second)
        self.assertEqual(first["counts"], {"earlier_hit": 1, "gained_hit": 1})
        self.assertEqual(first["net_hits"], 1)
        interval = first["bootstrap_technical_score_delta"]
        self.assertGreater(interval["lower"], 0)  # type: ignore[index]

    def test_bootstrap_rejects_nonpositive_probability_for_zero_deltas(self) -> None:
        result = paired_bootstrap([0.0, 0.0], seed=3, resamples=20)

        self.assertEqual(result["probability_delta_positive"], 0.0)
        self.assertEqual(result["lower"], 0.0)
        self.assertEqual(result["upper"], 0.0)


if __name__ == "__main__":
    unittest.main()
