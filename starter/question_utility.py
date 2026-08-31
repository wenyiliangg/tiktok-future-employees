"""Expected-utility ordering for answerable clarification questions."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from .ambiguity_analysis import AttributeValueStatistics
from .selective_clarification import SelectiveClarificationConfig


@dataclass(frozen=True, slots=True)
class QuestionUtility:
    attribute: str
    expected_hit_probability_change: float
    expected_reciprocal_rank_change: float
    expected_utility: float


def _proxies(
    attribute: str,
    statistics: dict[str, AttributeValueStatistics],
) -> tuple[float, float]:
    if attribute != "other":
        statistic = statistics.get(attribute)
        if statistic is None:
            return 0.0, 0.0
        reduction = statistic.expected_reduction
        return reduction, reduction * statistic.normalized_entropy

    useful = sorted(
        (
            statistic
            for name, statistic in statistics.items()
            if name != "category" and statistic.expected_reduction > 0
        ),
        key=lambda item: (-item.expected_reduction, item.attribute),
    )[:2]
    if not useful:
        return 0.0, 0.0
    residual = math.prod(1.0 - item.expected_reduction for item in useful)
    hit_change = 1.0 - residual
    rank_change = sum(
        item.expected_reduction * item.normalized_entropy for item in useful
    ) / len(useful)
    return hit_change, rank_change


def rank_question_candidates(
    statistics: Iterable[AttributeValueStatistics],
    config: SelectiveClarificationConfig,
) -> tuple[QuestionUtility, ...]:
    """Order configured questions using the declared approximate objective."""

    by_attribute = {item.attribute: item for item in statistics}
    answerability = dict(config.answerability_rates)
    ranked: list[QuestionUtility] = []
    for attribute in config.question_candidates:
        hit_change, rank_change = _proxies(attribute, by_attribute)
        rate = answerability.get(attribute, 0.0)
        hit_change *= rate
        rank_change *= rate
        utility = (
            config.hit_probability_weight * hit_change
            + config.reciprocal_rank_weight * rank_change
            - config.additional_turn_cost
        )
        if utility <= 0:
            continue
        ranked.append(
            QuestionUtility(
                attribute=attribute,
                expected_hit_probability_change=hit_change,
                expected_reciprocal_rank_change=rank_change,
                expected_utility=utility,
            )
        )
    return tuple(
        sorted(ranked, key=lambda item: (-item.expected_utility, item.attribute))
    )
