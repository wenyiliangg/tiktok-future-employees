"""Catalog-wide, relevance-label-free clarification answerability diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from benchmarks.shadow_clarification_suite import (
    classify_constraint,
    load_jsonl,
    shadow_constraints,
)

TARGETED_ATTRIBUTES = (
    "budget",
    "color",
    "feature",
    "material",
    "size",
    "style",
    "use_case",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def answerability(products: list[dict[str, object]]) -> dict[str, object]:
    product_count = len(products)
    answerable_products: Counter[str] = Counter()
    total_yield: Counter[str] = Counter()
    constraint_counts: list[int] = []

    for product in products:
        constraints = shadow_constraints(product)
        constraint_counts.append(len(constraints))
        per_attribute = Counter(classify_constraint(value) for value in constraints)
        for attribute in TARGETED_ATTRIBUTES:
            count = per_attribute[attribute]
            answerable_products[attribute] += int(count > 0)
            total_yield[attribute] += min(2, count)
        other_yield = min(2, len(constraints))
        answerable_products["other"] += int(other_yield > 0)
        total_yield["other"] += other_yield

    def attribute_result(attribute: str) -> dict[str, object]:
        answered = answerable_products[attribute]
        return {
            "answerable_product_count": answered,
            "answerability_rate": answered / product_count if product_count else 0.0,
            "mean_constraint_yield": (
                total_yield[attribute] / product_count if product_count else 0.0
            ),
        }

    attributes = {
        attribute: attribute_result(attribute)
        for attribute in (*TARGETED_ATTRIBUTES, "other")
    }
    for attribute in ("category", "brand"):
        attributes[attribute] = {
            "answerable_product_count": 0,
            "answerability_rate": 0.0,
            "mean_constraint_yield": 0.0,
            "reason": "official reply contract does not derive this attribute from the target",
        }
    return {
        "product_count": product_count,
        "mean_available_constraint_count": (
            sum(constraint_counts) / product_count if product_count else 0.0
        ),
        "attributes": dict(sorted(attributes.items())),
    }


def run(catalog_path: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "diagnostic": "question_answerability_label_free",
        "uses_relevance_labels": False,
        "catalog_sha256": _sha256(catalog_path),
        "answerability": answerability(load_jsonl(catalog_path)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run(Path(args.catalog))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
