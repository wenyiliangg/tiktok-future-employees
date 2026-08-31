"""Pure evaluator-response validation over an already ranked candidate pool."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet

from .clarification_controller import OFFICIAL_ATTRIBUTES

DEFAULT_RECOMMENDATION_MESSAGE = "Here are the closest matches I found."
MAX_RESPONSE_RECOMMENDATIONS = 100


def _candidate_identifier(candidate: object) -> str | None:
    if isinstance(candidate, Mapping):
        value = candidate.get("parent_asin")
    else:
        value = getattr(candidate, "parent_asin", None)
    return value if isinstance(value, str) and value else None


def _valid_score(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _recommendation(candidate: object) -> dict[str, object] | None:
    identifier = _candidate_identifier(candidate)
    if identifier is None:
        return None
    item: dict[str, object] = {"parent_asin": identifier}
    if isinstance(candidate, Mapping) and "score" in candidate:
        score = candidate["score"]
        if _valid_score(score):
            item["score"] = score
    return item


def validate_response(
    response: object,
    ranked_candidates: Iterable[object],
    catalog_ids: AbstractSet[str],
    top_k: object,
) -> dict[str, object]:
    """Return one deterministic contract response without retrieval or catalog scans.

    ``top_k`` is an upper bound. Non-integer, boolean, and non-positive values
    produce no recommendations. Valid unique response entries are retained in
    order, then the supplied bounded ranked list is continued to replace any
    invalid or duplicate entries. Catalog membership is checked only against
    the immutable identifier index supplied by the caller.
    """

    source = response if isinstance(response, Mapping) else {}
    raw_message = source.get("message")
    message = (
        raw_message if isinstance(raw_message, str) else DEFAULT_RECOMMENDATION_MESSAGE
    )
    ask_attribute = source.get("ask_attribute")
    clarification_is_valid = (
        isinstance(ask_attribute, str)
        and ask_attribute in OFFICIAL_ATTRIBUTES
        and isinstance(raw_message, str)
        and bool(raw_message.strip())
    )
    if ask_attribute is not None and not clarification_is_valid:
        ask_attribute = None
        message = DEFAULT_RECOMMENDATION_MESSAGE
    elif ask_attribute is None and not message.strip():
        message = DEFAULT_RECOMMENDATION_MESSAGE

    limit = (
        min(max(0, top_k), MAX_RESPONSE_RECOMMENDATIONS)
        if isinstance(top_k, int) and not isinstance(top_k, bool)
        else 0
    )
    raw_recommendations = source.get("recommendations")
    response_candidates: Iterable[object] = (
        raw_recommendations if isinstance(raw_recommendations, (list, tuple)) else ()
    )
    recommendations: list[dict[str, object]] = []
    seen: set[str] = set()
    if limit == 0:
        ranked_candidates = ()
        response_candidates = ()
    for candidate in (*response_candidates, *ranked_candidates):
        item = _recommendation(candidate)
        if item is None:
            continue
        identifier = str(item["parent_asin"])
        if identifier not in catalog_ids or identifier in seen:
            continue
        recommendations.append(item)
        seen.add(identifier)
        if len(recommendations) >= limit:
            break

    validated: dict[str, object] = {
        "message": message,
        "ask_attribute": ask_attribute if clarification_is_valid else None,
        "recommendations": recommendations,
    }
    usage = source.get("usage")
    if (
        isinstance(usage, Mapping)
        and set(usage) == {"prompt_tokens", "completion_tokens"}
        and all(
            isinstance(usage.get(name), int)
            and not isinstance(usage.get(name), bool)
            and int(usage[name]) >= 0
            for name in ("prompt_tokens", "completion_tokens")
        )
    ):
        validated["usage"] = {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
        }
    return validated
