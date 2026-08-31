from __future__ import annotations

import unittest

from benchmarks.catalog_signature_diagnostic import (
    SignatureDiagnosticIndex,
    catalog_phrases,
    diagnose,
    signature_key,
)
from benchmarks.shadow_clarification_suite import ShadowSample


class CatalogSignatureDiagnosticTest(unittest.TestCase):
    def products(self) -> dict[str, dict[str, object]]:
        return {
            "A": {
                "title": "blue trail shoe",
                "features": ["rare waterproof membrane alpha"],
                "details": {"Material": "canvas"},
            },
            "B": {
                "title": "red road shoe",
                "features": ["breathable mesh beta"],
                "details": {"Material": "mesh"},
            },
        }

    def sample(self) -> ShadowSample:
        return ShadowSample(
            sample_id="shadow_test",
            scenario_type="buying",
            target="A",
            category="shoes",
            constraints=(
                "rare waterproof membrane alpha",
                "Material: canvas",
                "blue trail shoe",
                "budget around $50",
            ),
            initial_constraint="rare waterproof membrane alpha",
            old_override_value=None,
            new_override_value=None,
            override_turn=None,
            template_variant=0,
            case_variant="natural",
            partial_disclosure=False,
        )

    def test_normalization_and_malformed_fields_are_deterministic(self) -> None:
        self.assertEqual(signature_key("The BLUE—trail shoe!"), "blue trail shoe")
        self.assertEqual(catalog_phrases({"features": None, "details": "bad"}), ())

    def test_exact_and_rare_token_lookup_recover_owner(self) -> None:
        index = SignatureDiagnosticIndex(self.products())

        self.assertEqual(
            index.owners("RARE waterproof membrane alpha!"), frozenset({"A"})
        )
        self.assertEqual(index.owners("breathable beta"), frozenset({"B"}))

    def test_diagnostic_reports_unique_first_phrase(self) -> None:
        result = diagnose(SignatureDiagnosticIndex(self.products()), [self.sample()])

        self.assertEqual(result["owner_recovery_rates"]["first_phrase"], 1.0)  # type: ignore[index]
        self.assertEqual(result["first_phrase_candidate_buckets"], {"unique": 1})


if __name__ == "__main__":
    unittest.main()
