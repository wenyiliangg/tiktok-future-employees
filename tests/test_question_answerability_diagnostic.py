from __future__ import annotations

from benchmarks.question_answerability_diagnostic import answerability


def test_answerability_is_label_free_and_excludes_category_and_brand() -> None:
    result = answerability(
        [
            {
                "parent_asin": "A",
                "title": "Blue trail shoe",
                "features": ["waterproof membrane", "rubber sole"],
                "details": {"Material": "nylon", "Color": "blue"},
                "categories": ["Shoes", "Trail Running"],
            },
            {
                "parent_asin": "B",
                "title": "Simple shoe",
                "features": [],
                "details": {},
                "categories": ["Shoes"],
            },
        ]
    )
    attributes = result["attributes"]
    assert attributes["category"]["answerability_rate"] == 0.0
    assert attributes["brand"]["answerability_rate"] == 0.0
    assert attributes["other"]["answerability_rate"] == 1.0
    assert attributes["other"]["mean_constraint_yield"] == 1.5
    assert attributes["material"]["answerable_product_count"] == 1
