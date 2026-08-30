"""Deterministic, non-public shadow evaluation for campaign experiments.

The suite excludes every public target before deterministic target selection.
It uses independently written dialogue templates and perturbations, and keeps
all target identities inside this benchmark layer.

This general robustness screen is retained even when an individual candidate
fails an official promotion guardrail.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import re
import statistics
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from starter.agent import Agent
from starter.clarification_controller import (
    OFFICIAL_ATTRIBUTES,
    ClarificationController,
    ClarificationControllerConfig,
)
from starter.clarification_policies import clarification_policy_by_id
from starter.contextual_retrieval import policy_by_id
from starter.hybrid_retrieval import HybridRetrievalConfig, RetrievalMode
from starter.selective_clarification import SelectiveClarificationConfig

SEED = 20260830
MAX_TURNS = 10
TOP_K = 10
SCENARIOS = ("buying", "browsing", "intent_override", "boundary")
PUBLIC_ID_RE = re.compile(r"^public_", re.IGNORECASE)
PRICE_RE = re.compile(r"(?:budget|price|\$)", re.IGNORECASE)
MATERIAL_RE = re.compile(
    r"\b(?:cotton|polyester|leather|wool|silk|linen|nylon|canvas|spandex|rubber)\b",
    re.IGNORECASE,
)
COLOR_RE = re.compile(
    r"\b(?:black|white|blue|red|pink|green|brown|gray|grey|yellow|purple)\b",
    re.IGNORECASE,
)
SIZE_RE = re.compile(r"\b(?:size|sizing|width|wide|narrow|inch|cm|small|large)\b", re.I)
STYLE_RE = re.compile(r"\b(?:style|fit|sleeve|neck|department|casual|formal)\b", re.I)
USE_CASE_RE = re.compile(r"\b(?:hiking|running|gym|winter|outdoor|work|travel)\b", re.I)


@dataclass(frozen=True, slots=True)
class ShadowSample:
    sample_id: str
    scenario_type: str
    target: str
    category: str
    constraints: tuple[str, ...]
    initial_constraint: str | None
    old_override_value: str | None
    new_override_value: str | None
    override_turn: int | None
    template_variant: int
    case_variant: str
    partial_disclosure: bool


def load_jsonl(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def public_targets(rows: list[dict[str, object]]) -> frozenset[str]:
    result: set[str] = set()
    for row in rows:
        ground_truth = row.get("ground_truth")
        if isinstance(ground_truth, Mapping):
            value = ground_truth.get("parent_asin")
            if isinstance(value, str) and value:
                result.add(value)
    return frozenset(result)


def _clean(value: object, limit: int = 150) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" -;,.\t\n")[:limit].rstrip()


def _finite_price(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def shadow_constraints(product: Mapping[str, object]) -> tuple[str, ...]:
    """Derive benchmark-only evidence while tolerating malformed metadata."""

    raw: list[str] = []
    features = product.get("features")
    if isinstance(features, list):
        raw.extend(_clean(value) for value in features)
    details = product.get("details")
    if isinstance(details, Mapping):
        raw.extend(
            _clean(f"{key}: {value}")
            for key, value in sorted(details.items(), key=lambda item: str(item[0]))
            if value not in (None, "", [])
        )
    price = _finite_price(product.get("price"))
    if price is not None:
        raw.append(f"budget around ${price:.2f}")
    title = _clean(product.get("title"))
    if title:
        raw.append(title)
    return tuple(dict.fromkeys(value for value in raw if len(value) >= 3))[:6]


def shadow_category(product: Mapping[str, object]) -> str:
    categories = product.get("categories")
    if not isinstance(categories, list):
        return "clothing item"
    ignored = {"clothing", "clothing, shoes & jewelry", "clothing shoes & jewelry"}
    useful = [
        _clean(value, 60)
        for value in categories
        if _clean(value, 60).lower() not in ignored
    ]
    return " ".join(useful[-2:]) if useful else "clothing item"


def _stable_key(parent_asin: str) -> str:
    return hashlib.sha256(f"shadow-v1\0{SEED}\0{parent_asin}".encode()).hexdigest()


def select_shadow_products(
    products: Mapping[str, Mapping[str, object]],
    excluded_targets: frozenset[str],
    sample_count: int,
) -> list[tuple[str, Mapping[str, object], tuple[str, ...]]]:
    """Select usable targets independently of catalog row order."""

    eligible: list[tuple[str, Mapping[str, object], tuple[str, ...]]] = []
    for parent_asin, product in products.items():
        if parent_asin in excluded_targets or PUBLIC_ID_RE.search(parent_asin):
            continue
        constraints = shadow_constraints(product)
        if len(constraints) >= 4 and shadow_category(product) != "clothing item":
            eligible.append((parent_asin, product, constraints))
    eligible.sort(key=lambda item: (_stable_key(item[0]), item[0]))
    if len(eligible) < sample_count:
        raise ValueError(f"only {len(eligible)} usable non-public shadow targets")
    return eligible[:sample_count]


def build_shadow_samples(
    products: Mapping[str, Mapping[str, object]],
    excluded_targets: frozenset[str],
    sample_count: int = 64,
) -> list[ShadowSample]:
    selected = select_shadow_products(products, excluded_targets, sample_count)
    samples: list[ShadowSample] = []
    for index, (target, product, constraints) in enumerate(selected):
        scenario = SCENARIOS[index % len(SCENARIOS)]
        initial = constraints[0] if scenario == "buying" else None
        old_override = constraints[-1] if scenario == "intent_override" else None
        new_override = constraints[0] if scenario == "intent_override" else None
        samples.append(
            ShadowSample(
                sample_id=f"shadow_{index + 1:04d}",
                scenario_type=scenario,
                target=target,
                category=shadow_category(product),
                constraints=constraints,
                initial_constraint=initial,
                old_override_value=old_override,
                new_override_value=new_override,
                override_turn=3 + (index % 2)
                if scenario == "intent_override"
                else None,
                template_variant=index % 4,
                case_variant=(
                    "upper"
                    if index % 7 == 0
                    else "lower"
                    if index % 7 == 1
                    else "natural"
                ),
                partial_disclosure=index % 3 == 0,
            )
        )
    return samples


def classify_constraint(value: str) -> str:
    if PRICE_RE.search(value):
        return "budget"
    if MATERIAL_RE.search(value):
        return "material"
    if COLOR_RE.search(value):
        return "color"
    if SIZE_RE.search(value):
        return "size"
    if STYLE_RE.search(value):
        return "style"
    if USE_CASE_RE.search(value):
        return "use_case"
    return "feature"


def _perturb(value: str, sample: ShadowSample) -> str:
    result = value.upper() if sample.case_variant == "upper" else value
    result = result.lower() if sample.case_variant == "lower" else result
    if sample.template_variant == 1:
        return result.replace(",", " —")
    if sample.template_variant == 2:
        return result.rstrip(".") + "!"
    return result


def initial_message(sample: ShadowSample, disclosed: set[str]) -> str:
    templates = (
        "I'm comparing {category} and could use a few options.",
        "Help me explore {category}; I haven't settled on the details.",
        "Could you show me some {category}? I'm still deciding.",
        "Browsing {category} today — surprise me with plausible choices.",
    )
    if sample.scenario_type == "buying" and sample.initial_constraint:
        disclosed.add(sample.initial_constraint)
        return (
            f"I need {sample.category}. One requirement is "
            f"{_perturb(sample.initial_constraint, sample)}."
        )
    if sample.scenario_type == "intent_override" and sample.old_override_value:
        return (
            f"I'm comparing {sample.category}; for now favor "
            f"{_perturb(sample.old_override_value, sample)}."
        )
    return templates[sample.template_variant].format(category=sample.category)


def customer_reply(
    sample: ShadowSample,
    ask_attribute: object,
    disclosed: set[str],
    boundary_declined: bool,
) -> tuple[str, bool, int]:
    attribute = ask_attribute if isinstance(ask_attribute, str) else None
    if sample.scenario_type == "boundary" and attribute and not boundary_declined:
        return (
            f"I don't have a preference for {attribute}; use your judgement.",
            True,
            0,
        )
    if attribute is None:
        rejections = (
            "Those are not quite right yet; ask a useful follow-up.",
            "None of these works so far. What else should I clarify?",
            "The options are not quite right. Ask about one decision factor.",
            "None of those is right yet; a focused question may help.",
        )
        return rejections[sample.template_variant], boundary_declined, 0
    if attribute not in OFFICIAL_ATTRIBUTES:
        attribute = "other"
    hidden = [value for value in sample.constraints if value not in disclosed]
    matches = [
        value
        for value in hidden
        if attribute == "other" or classify_constraint(value) == attribute
    ]
    limit = 1 if sample.partial_disclosure else 2
    selected = matches[:limit]
    if not selected:
        return (
            f"No additional preference for {attribute}; please use your judgment.",
            boundary_declined,
            0,
        )
    disclosed.update(selected)
    evidence = "; ".join(_perturb(value, sample) for value in selected)
    answers = (
        f"The deciding details are {evidence}.",
        f"Please prioritize this: {evidence}.",
        f"What matters on my side is {evidence}.",
        f"My useful constraints are {evidence}.",
    )
    return answers[sample.template_variant], boundary_declined, len(selected)


def _profile(sample: ShadowSample) -> dict[str, object]:
    return {
        "preference_tags": ["fit", "comfort", "durability"],
        "summary": f"Shadow shopper profile {sample.template_variant}.",
        "rating_style": "mixed",
    }


def _contract_counts(response: object, catalog_ids: frozenset[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not isinstance(response, dict):
        counts["invalid_responses"] += 1
        return counts
    if not isinstance(response.get("message"), str):
        counts["invalid_responses"] += 1
    ask_attribute = response.get("ask_attribute")
    if ask_attribute is not None and ask_attribute not in OFFICIAL_ATTRIBUTES:
        counts["invalid_ask_attributes"] += 1
    recommendations = response.get("recommendations")
    if not isinstance(recommendations, list):
        counts["invalid_responses"] += 1
        return counts
    seen: set[str] = set()
    for item in recommendations:
        value = item.get("parent_asin") if isinstance(item, Mapping) else None
        if not isinstance(value, str) or value not in catalog_ids:
            counts["invalid_asins"] += 1
        elif value in seen:
            counts["duplicate_recommendations"] += 1
        else:
            seen.add(value)
    return counts


def _ranked(response: object, catalog_ids: frozenset[str]) -> list[str]:
    if not isinstance(response, Mapping) or not isinstance(
        response.get("recommendations"), list
    ):
        return []
    result: list[str] = []
    for item in response["recommendations"]:  # type: ignore[index]
        value = item.get("parent_asin") if isinstance(item, Mapping) else None
        if isinstance(value, str) and value in catalog_ids and value not in result:
            result.append(value)
        if len(result) == TOP_K:
            break
    return result


def evaluate_shadow(
    agent: Agent,
    samples: list[ShadowSample],
    catalog_ids: frozenset[str],
) -> dict[str, object]:
    sessions: list[dict[str, object]] = []
    correctness: Counter[str] = Counter()
    question_counts: Counter[str] = Counter()
    question_yield: Counter[str] = Counter()
    transcript_hasher = hashlib.sha256()
    for sample in samples:
        session_id = f"{sample.sample_id}_{sample.template_variant}"
        agent.reset(session_id, _profile(sample))
        disclosed: set[str] = set()
        boundary_declined = False
        override_applied = sample.scenario_type != "intent_override"
        user_message = initial_message(sample, disclosed)
        first_hit_turn: int | None = None
        best_rank: int | None = None
        for turn in range(1, MAX_TURNS + 1):
            try:
                response = agent.respond(session_id, user_message, turn, TOP_K)
            except Exception:  # noqa: BLE001 - shadow records the agent boundary
                correctness["response_exceptions"] += 1
                response = {}
            correctness.update(_contract_counts(response, catalog_ids))
            transcript_hasher.update(
                json.dumps(
                    {
                        "sample_id": sample.sample_id,
                        "turn": turn,
                        "user_message": user_message,
                        "response": response,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    default=repr,
                ).encode()
            )
            ranked = _ranked(response, catalog_ids)
            if override_applied and sample.target in ranked:
                first_hit_turn = turn
                best_rank = ranked.index(sample.target) + 1
                break
            if turn == MAX_TURNS:
                break
            ask_attribute = (
                response.get("ask_attribute") if isinstance(response, Mapping) else None
            )
            if isinstance(ask_attribute, str):
                question_counts[ask_attribute] += 1
            next_turn = turn + 1
            if (
                not override_applied
                and sample.override_turn is not None
                and next_turn == sample.override_turn
            ):
                override_applied = True
                if sample.new_override_value:
                    disclosed.add(sample.new_override_value)
                user_message = (
                    "Actually, ignore my earlier preference. I now need "
                    f"{_perturb(sample.new_override_value or '', sample)}."
                )
                if isinstance(ask_attribute, str):
                    question_yield[f"{ask_attribute}.interrupted"] += 1
            else:
                user_message, boundary_declined, revealed = customer_reply(
                    sample, ask_attribute, disclosed, boundary_declined
                )
                if isinstance(ask_attribute, str):
                    question_yield[f"{ask_attribute}.asked"] += 1
                    question_yield[f"{ask_attribute}.revealed"] += revealed
        sessions.append(
            {
                "sample_id": sample.sample_id,
                "scenario_type": sample.scenario_type,
                "hit": first_hit_turn is not None,
                "first_hit_turn": first_hit_turn,
                "best_rank": best_rank,
                "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
            }
        )
    return {
        **metric_summary(sessions),
        "scenario_metrics": scenario_metrics(sessions),
        "question_counts": dict(sorted(question_counts.items())),
        "question_yield": dict(sorted(question_yield.items())),
        "correctness_counters": {
            name: correctness[name]
            for name in (
                "response_exceptions",
                "invalid_responses",
                "invalid_ask_attributes",
                "invalid_asins",
                "duplicate_recommendations",
            )
        },
        "normalized_transcript_sha256": transcript_hasher.hexdigest(),
        "sessions": sessions,
    }


def metric_summary(sessions: list[dict[str, object]]) -> dict[str, object]:
    count = len(sessions)
    hit_rate = sum(bool(item["hit"]) for item in sessions) / count
    mrr = statistics.fmean(float(item["reciprocal_rank"]) for item in sessions)
    mttc = statistics.fmean(
        int(item["first_hit_turn"]) if item["first_hit_turn"] is not None else 11
        for item in sessions
    )
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    score = 0.5 * hit_rate + 0.3 * mrr + 0.2 * efficiency
    return {
        "sample_count": count,
        "hit_rate_at_10": round(hit_rate, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
        "efficiency": round(efficiency, 6),
        "technical_score": round(score, 6),
    }


def scenario_metrics(sessions: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for session in sessions:
        grouped[str(session["scenario_type"])].append(session)
    return {name: metric_summary(rows) for name, rows in sorted(grouped.items())}


def question_answerability(samples: list[ShadowSample]) -> dict[str, object]:
    totals: Counter[str] = Counter()
    for sample in samples:
        initially_disclosed = (
            {sample.initial_constraint} if sample.initial_constraint else set()
        )
        hidden = [
            value for value in sample.constraints if value not in initially_disclosed
        ]
        totals["other"] += min(2, len(hidden))
        for attribute in OFFICIAL_ATTRIBUTES:
            if attribute in {"other", "category", "brand"}:
                continue
            totals[attribute] += min(
                2, sum(classify_constraint(value) == attribute for value in hidden)
            )
    return {
        "sample_count": len(samples),
        "potential_constraints_revealed": dict(sorted(totals.items())),
        "mean_other_yield": round(totals["other"] / len(samples), 6),
        "category_target_derived_yield": 0,
        "brand_target_derived_yield": 0,
    }


def comparison(
    champion: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, object]:
    before = {str(row["sample_id"]): row for row in champion["sessions"]}  # type: ignore[index]
    after = {str(row["sample_id"]): row for row in candidate["sessions"]}  # type: ignore[index]
    gained = lost = earlier = later = 0
    for sample_id in sorted(before):
        left, right = before[sample_id], after[sample_id]
        if not left["hit"] and right["hit"]:
            gained += 1
        elif left["hit"] and not right["hit"]:
            lost += 1
        elif left["hit"] and right["hit"]:
            earlier += int(right["first_hit_turn"]) < int(left["first_hit_turn"])
            later += int(right["first_hit_turn"]) > int(left["first_hit_turn"])
    return {
        "technical_score_delta": round(
            float(candidate["technical_score"]) - float(champion["technical_score"]), 6
        ),
        "gained_hits": gained,
        "lost_hits": lost,
        "earlier_shared_hits": earlier,
        "later_shared_hits": later,
    }


class _FixedRetriever:
    def __init__(self, values: list[str]) -> None:
        self.values = values

    def retrieve(self, *_args: object, **kwargs: object) -> list[object]:
        limit = int(kwargs.get("top_n", len(self.values)))
        return [
            SimpleNamespace(parent_asin=value, score=1.0 / rank, rank=rank)
            for rank, value in enumerate(self.values[:limit], start=1)
        ]


class _FixedRouter:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def route(self, *_args: object) -> object:
        if self.fail:
            raise RuntimeError("injected shadow router failure")
        return SimpleNamespace(route="browsing")


def robustness_checks() -> dict[str, bool]:
    malformed = shadow_constraints(
        {"features": None, "details": "bad", "price": "NaN"}
    ) == () and shadow_constraints({"title": "valid item", "price": float("inf")}) == (
        "valid item",
    )
    with tempfile.TemporaryDirectory() as directory:
        catalog = Path(directory) / "catalog.jsonl"
        rows = [
            {"parent_asin": value, "title": f"item {value}", "price": price}
            for value, price in zip(("A", "B", "C", "D"), (None, "NaN", 10, 20))
        ]
        catalog.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        config = SelectiveClarificationConfig(
            enabled=True,
            eligible_routes=("browsing",),
            question_priority=("other", "feature"),
            priority_min_candidates=4,
        )
        missing_cache = Agent(
            catalog,
            config=HybridRetrievalConfig(mode=RetrievalMode.CONTEXTUAL),
            anchor_retriever=_FixedRetriever(["A", "B", "C", "D"]),
            dense_factory=lambda: (_ for _ in ()).throw(FileNotFoundError("missing")),
            router=_FixedRouter(),
            clarification_config=config,
            clarification_controller=ClarificationController(
                ClarificationControllerConfig(max_questions_per_session=2)
            ),
        )
        sequential_valid = True
        for session_id in ("shadow-consecutive-one", "shadow-consecutive-two"):
            missing_cache.reset(session_id, {})
            response = missing_cache.respond(session_id, "browse items", 1, 4)
            sequential_valid &= len(_ranked(response, frozenset("ABCD"))) == 4
        broken_router = Agent(
            catalog,
            config=HybridRetrievalConfig(mode=RetrievalMode.CONTEXTUAL),
            anchor_retriever=_FixedRetriever(["D", "C", "B", "A"]),
            dense_retriever=_FixedRetriever([]),
            router=_FixedRouter(fail=True),
            clarification_config=config,
            clarification_controller=ClarificationController(
                ClarificationControllerConfig(max_questions_per_session=2)
            ),
        )
        broken_router.reset("shadow-component-failure", {})
        failure_response = broken_router.respond(
            "shadow-component-failure", "browse items", 1, 4
        )
        component_fallback = _ranked(failure_response, frozenset("ABCD")) == [
            "D",
            "C",
            "B",
            "A",
        ]
    return {
        "malformed_price_and_metadata": malformed,
        "missing_dense_cache_fallback": sequential_valid,
        "component_failure_fallback": component_fallback,
        "consecutive_session_isolation": sequential_valid,
    }


def _agent(
    catalog_path: Path,
    dense_cache_path: Path,
    clarification: SelectiveClarificationConfig,
    controller: ClarificationControllerConfig,
) -> Agent:
    return Agent(
        catalog_path,
        config=HybridRetrievalConfig(mode=RetrievalMode.CONTEXTUAL),
        dense_cache_path=dense_cache_path,
        contextual_policy=policy_by_id("contextual.feedback-memory.v1"),
        clarification_config=clarification,
        clarification_controller=ClarificationController(controller),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public-set", default="data/public_set.jsonl")
    parser.add_argument(
        "--dense-cache", default="data/.dense-retrieval/catalog-minilm.npz"
    )
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--output")
    parser.add_argument("--diagnostic-only", action="store_true")
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    products = {
        str(row["parent_asin"]): row
        for row in load_jsonl(catalog_path)
        if isinstance(row.get("parent_asin"), str)
    }
    excluded = public_targets(load_jsonl(args.public_set))
    samples = build_shadow_samples(products, excluded, args.sample_count)
    selected_targets = {sample.target for sample in samples}
    reordered = dict(reversed(list(products.items())))
    reorder_targets = {
        item[0]
        for item in select_shadow_products(reordered, excluded, args.sample_count)
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "seed": SEED,
        "suite": {
            "sample_count": len(samples),
            "scenario_counts": dict(
                Counter(sample.scenario_type for sample in samples)
            ),
            "public_target_overlap": len(selected_targets & excluded),
            "catalog_reorder_invariant": selected_targets == reorder_targets,
            "case_variants": dict(Counter(sample.case_variant for sample in samples)),
            "partial_disclosure_count": sum(
                sample.partial_disclosure for sample in samples
            ),
            "target_selection_sha256": hashlib.sha256(
                "\n".join(sorted(selected_targets)).encode()
            ).hexdigest(),
        },
        "question_answerability": question_answerability(samples),
        "robustness_checks": robustness_checks(),
    }
    if not args.diagnostic_only:
        champion_config = SelectiveClarificationConfig(
            enabled=True,
            required_retrieval_policy_id="contextual.feedback-memory.v1",
            eligible_routes=("browsing",),
        )
        champion_agent = _agent(
            catalog_path,
            Path(args.dense_cache),
            champion_config,
            ClarificationControllerConfig(max_questions_per_session=1),
        )
        champion = evaluate_shadow(champion_agent, samples, frozenset(products))
        del champion_agent
        gc.collect()

        candidate_policy = clarification_policy_by_id(
            "clarification.feedback-memory.v1"
        )
        candidate_agent = _agent(
            catalog_path,
            Path(args.dense_cache),
            candidate_policy.clarification,
            candidate_policy.controller,
        )
        candidate = evaluate_shadow(candidate_agent, samples, frozenset(products))
        del candidate_agent
        gc.collect()

        repeat_agent = _agent(
            catalog_path,
            Path(args.dense_cache),
            candidate_policy.clarification,
            candidate_policy.controller,
        )
        repeat = evaluate_shadow(repeat_agent, samples, frozenset(products))
        result.update(
            {
                "champion": champion,
                "candidate": candidate,
                "comparison": comparison(champion, candidate),
                "candidate_deterministic": (
                    candidate["normalized_transcript_sha256"]
                    == repeat["normalized_transcript_sha256"]
                    and candidate["sessions"] == repeat["sessions"]
                ),
            }
        )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    random.seed(SEED)
    main()
