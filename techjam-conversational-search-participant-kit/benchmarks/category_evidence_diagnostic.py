"""Label-free catalog diagnostic for the category-evidence experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from starter.category_evidence import (
    CategoryEvidenceIndex,
    catalog_statistics,
    category_evidence_policy_for_retrieval,
    category_recovery_statistics,
)


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(catalog_path: Path) -> dict[str, object]:
    policy = category_evidence_policy_for_retrieval("contextual.category-evidence.v1")
    started = perf_counter()
    index = CategoryEvidenceIndex.from_jsonl(catalog_path, policy)
    build_seconds = perf_counter() - started
    return {
        "schema_version": 1,
        "diagnostic": "category_evidence_label_free",
        "uses_relevance_labels": False,
        "catalog_sha256": _sha256(catalog_path),
        "policy_id": policy.policy_id,
        "index": catalog_statistics(index),
        "category_recovery": category_recovery_statistics(index, _jsonl(catalog_path)),
        "build_seconds": round(build_seconds, 6),
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
