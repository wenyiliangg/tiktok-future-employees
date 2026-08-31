"""Fast selected-session verification for the protected BM25 recovery path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent
from starter.hybrid_retrieval import HybridRetrievalConfig, RetrievalMode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("diagnostics/retrieval_regression/official_bm25_hits.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/retrieval_regression/recovery_verification.json"),
    )
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    expected = {
        str(session["sample_id"]): (
            session["official_hit_turn"],
            next(
                turn["official_bm25_target_rank"]
                for turn in session["turns"]
                if turn.get("loss_classification")
            ),
        )
        for session in selection["sessions"]
    }
    samples = [
        sample
        for sample in load_jsonl(args.dataset)
        if str(sample["sample_id"]) in expected
    ]
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(
        args.catalog,
        config=HybridRetrievalConfig(mode=RetrievalMode.ANCHORED),
    )
    result = evaluate(agent, samples, catalog_ids, categories, products)
    actual = {
        str(session["sample_id"]): (session["first_hit_turn"], session["best_rank"])
        for session in result["sessions"]
    }
    mismatches = {
        sample_id: {"expected": expected[sample_id], "actual": actual.get(sample_id)}
        for sample_id in expected
        if actual.get(sample_id) != expected[sample_id]
    }
    result["diagnostic_only"] = True
    result["selected_session_ids"] = sorted(expected)
    result["expected_anchor_turn_and_rank"] = {
        key: list(value) for key, value in sorted(expected.items())
    }
    result["recovery_mismatches"] = mismatches
    result["recovery_verified"] = not mismatches and len(actual) == len(expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_session_count": len(samples),
                "hit_rate_at_10": result["hit_rate_at_10"],
                "mrr": result["mrr"],
                "mttc": result["mttc"],
                "recovery_verified": result["recovery_verified"],
                "mismatch_count": len(mismatches),
            },
            indent=2,
        )
    )
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
