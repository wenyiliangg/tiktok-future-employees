from __future__ import annotations

from starter.category_evidence import (
    CategoryEvidenceIndex,
    CategoryEvidencePolicy,
    EvidenceMessageStore,
    catalog_statistics,
    category_recovery_statistics,
)
from starter.conversation_state import Constraint, SearchQuery
from starter.hybrid_retrieval import Candidate
from starter.lexical_retriever import CatalogDocumentBuilder
from starter.search_models import RetrievalResult

PRODUCTS = [
    {
        "parent_asin": "A",
        "title": "Trail Runner",
        "features": ["storm shield heel counter", "reflective trim"],
        "details": {"Material": "nylon", "Color": "blue"},
        "categories": ["Clothing, Shoes & Jewelry", "Women", "Shoes", "Running"],
        "average_rating": 4.7,
        "rating_number": 50,
    },
    {
        "parent_asin": "B",
        "title": "Popular Road Runner",
        "features": ["everyday cushioned trainer"],
        "details": {"Material": "mesh", "Color": "black"},
        "categories": ["Clothing, Shoes & Jewelry", "Women", "Shoes", "Running"],
        "average_rating": 4.9,
        "rating_number": 9000,
    },
    {
        "parent_asin": "C",
        "title": "Winter Coat",
        "features": ["storm shield heel counter"],
        "details": {"Material": "wool", "Color": "blue"},
        "categories": ["Clothing, Shoes & Jewelry", "Women", "Clothing", "Coats"],
        "average_rating": 4.8,
        "rating_number": 12000,
    },
]


class _Catalog:
    def __init__(self) -> None:
        builder = CatalogDocumentBuilder()
        self.documents = {
            document.parent_asin: document
            for product in PRODUCTS
            if (document := builder.build(product)) is not None
        }

    def get(self, parent_asin: str):
        return self.documents.get(parent_asin)


def _policy(**changes: object) -> CategoryEvidencePolicy:
    values = {
        "policy_id": "test.category-evidence",
        "retrieval_policy_id": "test.contextual",
        "category_candidate_limit": 20,
        "total_candidate_limit": 30,
        "rare_document_fraction": 0.67,
        "strong_evidence_document_fraction": 0.67,
    }
    values.update(changes)
    return CategoryEvidencePolicy(**values)


def _query(category: str = "shoes running", material: str | None = None) -> SearchQuery:
    return SearchQuery(
        text=category,
        category=Constraint(category, "hard", "current_turn", 1),
        material=(
            Constraint(material, "hard", "current_turn", 1)
            if material is not None
            else None
        ),
    )


def test_catalog_category_is_recovered_from_any_message_position() -> None:
    index = CategoryEvidenceIndex(PRODUCTS, _policy())
    assert (
        index.extract_category("Could you show me Shoes Running for a trip?")
        == "shoes running"
    )
    assert index.extract_category("I need Clothing Coats in blue") == "women coats"


def test_exact_phrase_and_category_outrank_popularity() -> None:
    index = CategoryEvidenceIndex(PRODUCTS, _policy())
    ranked = index.rank(
        query=_query(),
        current_text="I need Shoes Running with storm shield heel counter",
        historical_text="",
        base_results=(RetrievalResult("B", 9.0, 1),),
        history_results=(),
        catalog=_Catalog(),
        known_negative_ids=set(),
        limit=3,
    )
    assert ranked[0].parent_asin == "A"
    assert ranked[0].fusion_score > next(
        candidate.fusion_score for candidate in ranked if candidate.parent_asin == "C"
    )
    assert ranked[0].component_scores["phrase"] > 0
    assert "phrase" in ranked[0].sources


def test_current_structured_evidence_dominates_history_and_prior() -> None:
    index = CategoryEvidenceIndex(PRODUCTS, _policy())
    ranked = index.rank(
        query=_query(material="nylon"),
        current_text="Shoes Running nylon",
        historical_text="everyday cushioned trainer",
        base_results=(RetrievalResult("B", 9.0, 1),),
        history_results=(RetrievalResult("B", 9.0, 1),),
        catalog=_Catalog(),
        known_negative_ids=set(),
        limit=3,
    )
    assert ranked[0].parent_asin == "A"
    by_id = {candidate.parent_asin: candidate for candidate in ranked}
    assert by_id["B"].component_scores["contradictions"] >= 1


def test_known_negative_is_not_returned() -> None:
    index = CategoryEvidenceIndex(PRODUCTS, _policy())
    ranked = index.rank(
        query=_query(),
        current_text="Shoes Running",
        historical_text="",
        base_results=(RetrievalResult("B", 9.0, 1),),
        history_results=(),
        catalog=_Catalog(),
        known_negative_ids={"B"},
        limit=3,
    )
    assert "B" not in {candidate.parent_asin for candidate in ranked}


def test_message_store_separates_current_and_historical_on_override() -> None:
    store = EvidenceMessageStore()
    store.reset("s")
    store.observe(
        "s", "old distinctive detail", override=False, non_evidence_reply=False
    )
    store.observe("s", "generic retry", override=False, non_evidence_reply=True)
    store.observe(
        "s", "actually use new detail", override=True, non_evidence_reply=False
    )
    assert store.current_text("s") == "actually use new detail"
    assert store.historical_text("s") == "old distinctive detail"


def test_label_free_statistics_and_candidate_shape_are_deterministic() -> None:
    first = CategoryEvidenceIndex(PRODUCTS, _policy())
    second = CategoryEvidenceIndex(reversed(PRODUCTS), _policy())
    assert catalog_statistics(first) == catalog_statistics(second)
    assert catalog_statistics(first)["catalog_size"] == 3
    assert catalog_statistics(first)["category_max_size"] == 2
    assert category_recovery_statistics(first, PRODUCTS) == {
        "checked_product_count": 3,
        "recovered_product_count": 3,
        "missing_category_count": 0,
        "recovery_rate": 1.0,
    }
    arguments = {
        "query": _query(),
        "current_text": "Shoes Running with storm shield heel counter",
        "historical_text": "",
        "base_results": (RetrievalResult("B", 9.0, 1),),
        "history_results": (),
        "catalog": _Catalog(),
        "known_negative_ids": set(),
        "limit": 3,
    }
    assert [item.parent_asin for item in first.rank(**arguments)] == [
        item.parent_asin for item in second.rank(**arguments)
    ]
    candidate = Candidate(parent_asin="A")
    assert candidate.parent_asin == "A"
