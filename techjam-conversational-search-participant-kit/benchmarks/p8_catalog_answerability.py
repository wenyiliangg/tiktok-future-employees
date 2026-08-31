"""Derive P8 answerability constants from catalog-wide official helpers only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluator.local_evaluator import (
    ALLOWED_ATTRIBUTES,
    classify_constraint,
    coarse_category,
    customer_reply,
    materialize_hidden_fields,
)
from starter.ambiguity_analysis import AmbiguityAnalyzer

DERIVATION_SEED = 2026083100
TARGETED_ATTRIBUTES = (
    "budget",
    "color",
    "material",
    "style",
    "use_case",
    "feature",
)
METADATA_FIELDS = (
    "title",
    "features",
    "details",
    "categories",
    "price",
    "store",
)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stable_key(parent_asin: str) -> str:
    return hashlib.sha256(
        f"p8-ambiguity-block\0{DERIVATION_SEED}\0{parent_asin}".encode()
    ).hexdigest()


def _present(value: object) -> bool:
    return value not in (None, "", [], {})


def _popularity(product: dict[str, Any]) -> float:
    try:
        count = max(0, int(product.get("rating_number") or 0))
    except (TypeError, ValueError):
        count = 0
    return math.log1p(count)


def _nearest_rank(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    position = max(0, min(len(values) - 1, math.ceil(fraction * len(values)) - 1))
    return values[position]


def _rounded_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    products: dict[str, dict[str, Any]] = {}
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            parent_asin = str(product.get("parent_asin") or "").strip()
            if parent_asin and parent_asin not in products:
                products[parent_asin] = product

    ordered_ids = sorted(products)
    answerable: Counter[str] = Counter()
    total_yield: Counter[str] = Counter()
    answerable_yield: Counter[str] = Counter()
    metadata_present: Counter[str] = Counter()
    category_rows: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "product_count": 0,
            "answerable": Counter(),
            "yield": Counter(),
            "metadata_present": Counter(),
        }
    )
    category_ids: dict[str, list[str]] = defaultdict(list)
    popularity_values: list[float] = []
    materialization_failures = 0
    helper_reply_failures = 0

    for parent_asin in ordered_ids:
        product = products[parent_asin]
        category = coarse_category(
            [str(value) for value in product.get("categories") or []]
        )
        category_ids[category].append(parent_asin)
        row = category_rows[category]
        row["product_count"] += 1
        popularity_values.append(_popularity(product))
        for field in METADATA_FIELDS:
            if _present(product.get(field)):
                metadata_present[field] += 1
                row["metadata_present"][field] += 1

        sample = {
            "sample_id": f"p8_derive_{parent_asin}",
            "scenario_type": "browsing",
            "ground_truth": {"parent_asin": parent_asin},
        }
        try:
            card, behavior = materialize_hidden_fields(sample, products)
        except Exception:  # noqa: BLE001 - aggregate derivation boundary
            materialization_failures += 1
            continue
        effective = {**sample, "intent_card": card, "behavior": behavior}
        constraints = [
            *[str(value) for value in card.get("hard_constraints", [])],
            *[str(value) for value in card.get("soft_preferences", [])],
        ]
        counts = Counter(classify_constraint(value) for value in constraints)
        counts["other"] = len(constraints)
        for attribute in sorted(ALLOWED_ATTRIBUTES):
            count = counts.get(attribute, 0)
            capped = min(2, count)
            total_yield[attribute] += capped
            row["yield"][attribute] += capped
            if count <= 0:
                continue
            try:
                reply, _boundary = customer_reply(
                    effective, attribute, set(), False
                )
                if not isinstance(reply, str) or not reply:
                    helper_reply_failures += 1
                    continue
            except Exception:  # noqa: BLE001 - aggregate derivation boundary
                helper_reply_failures += 1
                continue
            answerable[attribute] += 1
            answerable_yield[attribute] += capped
            row["answerable"][attribute] += 1

    retained = tuple(
        attribute for attribute in TARGETED_ATTRIBUTES if answerable[attribute] > 0
    )
    analyzer = AmbiguityAnalyzer()
    block_maxima: list[float] = []
    block_counts: Counter[str] = Counter()
    for category in sorted(category_ids):
        identifiers = sorted(
            category_ids[category], key=lambda item: (_stable_key(item), item)
        )
        for start in range(0, len(identifiers), 50):
            block = identifiers[start : start + 50]
            if len(block) < 4:
                continue
            statistics_by_attribute = {
                ("budget" if item.attribute == "price" else item.attribute): item
                for item in analyzer.attribute_statistics(block, products)
            }
            reductions = [
                statistics_by_attribute[attribute].expected_reduction
                for attribute in retained
                if attribute in statistics_by_attribute
                and statistics_by_attribute[attribute].expected_reduction > 0.0
            ]
            if reductions:
                block_maxima.append(max(reductions))
                block_counts[category] += 1

    raw_median = statistics.median(block_maxima) if block_maxima else 0.0
    ambiguity_threshold = math.floor(raw_median * 1000.0) / 1000.0
    catalog_count = len(ordered_ids)
    answerability = {
        attribute: {
            "answerable_product_count": answerable[attribute],
            "answerability_rate": _rounded_rate(
                answerable[attribute], catalog_count
            ),
            "mean_constraint_yield_all_products": round(
                total_yield[attribute] / catalog_count, 6
            )
            if catalog_count
            else 0.0,
            "mean_constraint_yield_when_answerable": round(
                answerable_yield[attribute] / answerable[attribute], 6
            )
            if answerable[attribute]
            else 0.0,
            "retained_for_variant_B": attribute in retained,
        }
        for attribute in sorted(ALLOWED_ATTRIBUTES)
    }
    category_coverage: dict[str, object] = {}
    for category in sorted(category_rows):
        row = category_rows[category]
        count = int(row["product_count"])
        category_coverage[category] = {
            "product_count": count,
            "answerability_rates": {
                attribute: _rounded_rate(row["answerable"][attribute], count)
                for attribute in ("other", *TARGETED_ATTRIBUTES)
            },
            "metadata_completeness": {
                field: _rounded_rate(row["metadata_present"][field], count)
                for field in METADATA_FIELDS
            },
            "ambiguity_block_count": block_counts[category],
        }

    popularity_values.sort()
    result = {
        "schema_version": 1,
        "derivation_id": "P8B-catalog-answerability-v1",
        "predeclaration_commit": "dbdcf73894eed9f10fd6c580c680bc918acbf8c4",
        "seed": DERIVATION_SEED,
        "command": (
            "python3 -m benchmarks.p8_catalog_answerability "
            "--catalog data/catalog.jsonl --output "
            "docs/results/autonomous_optimization/shadow_results/"
            "p8_catalog_answerability.json"
        ),
        "inputs": {
            "catalog_path": str(catalog_path),
            "catalog_sha256": _sha256(catalog_path),
            "catalog_product_count": catalog_count,
            "evaluator_helper_sha256": _sha256("evaluator/local_evaluator.py"),
            "ambiguity_analyzer_sha256": _sha256(
                "starter/ambiguity_analysis.py"
            ),
            "public_outcomes_consulted": False,
            "proxy_targets_generated_or_evaluated": False,
        },
        "answerability": answerability,
        "variant_B_retained_attributes": list(retained),
        "ambiguity_threshold": {
            "block_size": 50,
            "minimum_block_size": 4,
            "positive_block_count": len(block_maxima),
            "raw_median_maximum_expected_reduction": round(raw_median, 6),
            "runtime_floor_to_three_decimals": ambiguity_threshold,
        },
        "popularity_log1p_rating_number_quartiles": {
            "method": "nearest-rank over all catalog products",
            "q1": round(_nearest_rank(popularity_values, 0.25), 9),
            "q2": round(_nearest_rank(popularity_values, 0.50), 9),
            "q3": round(_nearest_rank(popularity_values, 0.75), 9),
        },
        "metadata_completeness": {
            field: {
                "present_count": metadata_present[field],
                "rate": _rounded_rate(metadata_present[field], catalog_count),
            }
            for field in METADATA_FIELDS
        },
        "category_count": len(category_coverage),
        "category_coverage": category_coverage,
        "derivation_failures": {
            "materialization": materialization_failures,
            "customer_reply": helper_reply_failures,
        },
        "runtime_export": {
            "ambiguity_threshold": ambiguity_threshold,
            "answerability_rates": {
                attribute: answerability[attribute]["answerability_rate"]
                for attribute in retained
            },
            "mean_yield_when_answerable": {
                attribute: answerability[attribute][
                    "mean_constraint_yield_when_answerable"
                ]
                for attribute in retained
            },
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    Path(args.output).write_text(rendered, encoding="utf-8")
    compact = dict(result)
    compact.pop("category_coverage", None)
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
