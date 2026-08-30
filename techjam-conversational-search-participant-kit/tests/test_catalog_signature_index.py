from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.catalog_signature_index import (
    CatalogSignatureIndex,
    CatalogSignaturePolicy,
    catalog_signature_policy_for_retrieval,
    normalize_tokens,
)


class CatalogSignatureIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.catalog = Path(self.temporary.name) / "catalog.jsonl"
        rows = [
            {
                "parent_asin": "A",
                "title": "blue trail shoe",
                "features": ["rare waterproof membrane alpha"],
                "details": {"Material": "canvas"},
            },
            {
                "parent_asin": "B",
                "title": "red road shoe",
                "features": ["breathable mesh beta", "shared comfort phrase"],
            },
            {
                "parent_asin": "C",
                "title": "green road shoe",
                "features": ["shared comfort phrase"],
            },
        ]
        self.catalog.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        self.policy = CatalogSignaturePolicy(
            policy_id="test",
            retrieval_policy_id="test-retrieval",
            compatible_retrieval_policy_id="test-retrieval",
            enabled=True,
            min_characters=8,
        )
        self.index = CatalogSignatureIndex.from_jsonl(self.catalog, self.policy)

    def test_unique_phrase_matches_inside_paraphrased_wrapper(self) -> None:
        match = self.index.match(
            "For me, what matters is: RARE waterproof membrane alpha!"
        )

        assert match is not None
        self.assertEqual(match.parent_asin, "A")
        self.assertEqual(match.normalized_signature, "rare waterproof membrane alpha")

    def test_duplicate_or_short_phrase_does_not_match(self) -> None:
        self.assertIsNone(self.index.match("I value shared comfort phrase"))
        self.assertIsNone(self.index.match("blue"))

    def test_catalog_reorder_does_not_change_unique_matches(self) -> None:
        rows = [json.loads(line) for line in self.catalog.read_text().splitlines()]
        reordered = Path(self.temporary.name) / "reordered.jsonl"
        reordered.write_text(
            "".join(json.dumps(row) + "\n" for row in reversed(rows)),
            encoding="utf-8",
        )
        second = CatalogSignatureIndex.from_jsonl(reordered, self.policy)

        self.assertEqual(
            self.index.match("rare waterproof membrane alpha"),
            second.match("rare waterproof membrane alpha"),
        )

    def test_normalization_policy_and_fingerprint_are_deterministic(self) -> None:
        self.assertEqual(
            normalize_tokens("The BLUE—trail shoe!"), ("blue", "trail", "shoe")
        )
        selected = catalog_signature_policy_for_retrieval(
            "contextual.signature-head.v1"
        )

        self.assertTrue(selected.enabled)
        self.assertEqual(selected.fingerprint_sha256, selected.fingerprint_sha256)
        self.assertTrue(
            selected.clarification_is_compatible("contextual.feedback-memory.v1")
        )


if __name__ == "__main__":
    unittest.main()
