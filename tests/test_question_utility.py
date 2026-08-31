from __future__ import annotations

from starter.ambiguity_analysis import AttributeValueStatistics
from starter.question_utility import rank_question_candidates
from starter.selective_clarification import SelectiveClarificationConfig


def _statistic(
    attribute: str, reduction: float, entropy: float
) -> AttributeValueStatistics:
    return AttributeValueStatistics(
        attribute=attribute,
        candidate_count=10,
        usable_count=10,
        coverage=1.0,
        value_counts=(("a", 5), ("b", 5)),
        dominant_share=0.5,
        normalized_entropy=entropy,
        expected_reduction=reduction,
    )


def _config() -> SelectiveClarificationConfig:
    return SelectiveClarificationConfig(
        enabled=True,
        required_retrieval_policy_id="contextual.category-evidence.v1",
        eligible_routes=("browsing", "boundary"),
        question_candidates=("other", "feature"),
        utility_min_candidates=4,
        answerability_rates=(("other", 1.0), ("feature", 0.99294)),
    )


def test_open_question_combines_two_candidate_reductions() -> None:
    ranked = rank_question_candidates(
        (_statistic("feature", 0.4, 1.0), _statistic("material", 0.5, 0.8)),
        _config(),
    )
    assert [item.attribute for item in ranked] == ["other", "feature"]
    assert ranked[0].expected_hit_probability_change == 0.7
    assert ranked[0].expected_utility > ranked[1].expected_utility


def test_utility_strategy_uses_route_and_pool_gates() -> None:
    config = _config()
    assert config.utility_is_eligible("browsing", 4)
    assert config.utility_is_eligible("boundary", 10)
    assert not config.utility_is_eligible("buying", 10)
    assert not config.utility_is_eligible("browsing", 3)
