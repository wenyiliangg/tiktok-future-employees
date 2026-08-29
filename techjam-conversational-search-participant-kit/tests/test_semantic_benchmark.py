from __future__ import annotations

import unittest

from benchmarks.benchmark_semantic_reranker import compare_results


def _result(*, hr: float, mrr: float, score: float, failures: int) -> dict:
    return {
        "hit_rate_at_10": hr,
        "mrr": mrr,
        "mttc": 9.0,
        "efficiency": 0.2,
        "recommended_technical_score": score,
        "scenario_metrics": {
            name: {"hit_rate_at_10": hr, "mrr": mrr}
            for name in ("buying", "browsing", "intent_override", "boundary")
        },
        "evaluation_diagnostics": {
            "average_response_latency_ms": 10.0,
            "p95_response_latency_ms": 20.0,
        },
        "performance": {
            "agent_startup_seconds": 1.0,
            "peak_process_rss_bytes": 100,
        },
        "semantic_reranker": {
            "model": "test/model",
            "model_size_bytes": 40,
            "cold_start_seconds": 0.5,
            "average_reranking_latency_ms": 3.0,
            "p95_reranking_latency_ms": 4.0,
            "max_scored_candidates_per_query": 50,
            "failure_count": failures,
        },
    }


class SemanticBenchmarkComparisonTest(unittest.TestCase):
    def test_recommends_only_measurable_safe_gain_and_never_enables_default(
        self,
    ) -> None:
        baseline = _result(hr=0.10, mrr=0.05, score=0.10, failures=0)
        improved = _result(hr=0.11, mrr=0.06, score=0.106, failures=0)
        comparison = compare_results(
            baseline, improved, minimum_technical_score_gain=0.005
        )
        self.assertEqual(comparison["recommendation"], "retain_as_optional_experiment")
        self.assertFalse(comparison["default_enabled"])
        self.assertAlmostEqual(
            comparison["deltas_semantic_minus_baseline"]["technical_score"],
            0.006,
        )

    def test_failure_or_quality_regression_rejects_component(self) -> None:
        baseline = _result(hr=0.10, mrr=0.05, score=0.10, failures=0)
        for semantic in (
            _result(hr=0.11, mrr=0.06, score=0.11, failures=1),
            _result(hr=0.09, mrr=0.07, score=0.11, failures=0),
            _result(hr=0.10, mrr=0.05, score=0.102, failures=0),
        ):
            with self.subTest(semantic=semantic):
                comparison = compare_results(
                    baseline, semantic, minimum_technical_score_gain=0.005
                )
                self.assertEqual(
                    comparison["recommendation"],
                    "do_not_retain_without_further_evidence",
                )
                self.assertFalse(comparison["default_enabled"])

    def test_runtime_or_memory_over_budget_rejects_component(self) -> None:
        baseline = _result(hr=0.10, mrr=0.05, score=0.10, failures=0)
        semantic = _result(hr=0.11, mrr=0.06, score=0.11, failures=0)
        semantic["semantic_reranker"]["average_reranking_latency_ms"] = 251.0
        semantic["performance"]["peak_process_rss_bytes"] = 600_000_101
        comparison = compare_results(
            baseline,
            semantic,
            minimum_technical_score_gain=0.005,
            maximum_average_reranking_latency_ms=250.0,
            maximum_peak_memory_increase_bytes=500_000_000,
        )
        self.assertEqual(
            comparison["recommendation"],
            "do_not_retain_without_further_evidence",
        )


if __name__ == "__main__":
    unittest.main()
