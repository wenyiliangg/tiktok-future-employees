"""Deterministic paired outcome comparison with a bootstrap confidence interval."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path

SEED = 20260830
RESAMPLES = 10_000


def _load(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def session_score(session: Mapping[str, object]) -> float:
    hit = 1.0 if session.get("hit") else 0.0
    reciprocal_rank = float(session.get("reciprocal_rank") or 0.0)
    first_hit_turn = session.get("first_hit_turn")
    turn_cost = 11 if first_hit_turn is None else int(first_hit_turn)
    efficiency = (11.0 - turn_cost) / 10.0
    return 0.5 * hit + 0.3 * reciprocal_rank + 0.2 * efficiency


def _label(before: Mapping[str, object], after: Mapping[str, object]) -> str:
    before_hit, after_hit = bool(before.get("hit")), bool(after.get("hit"))
    if before_hit != after_hit:
        return "gained_hit" if after_hit else "lost_hit"
    if not before_hit:
        return "unchanged_miss"
    before_turn = int(before["first_hit_turn"])
    after_turn = int(after["first_hit_turn"])
    if before_turn != after_turn:
        return "earlier_hit" if after_turn < before_turn else "later_hit"
    before_rank = int(before["best_rank"])
    after_rank = int(after["best_rank"])
    if before_rank != after_rank:
        return "better_rank" if after_rank < before_rank else "worse_rank"
    return "unchanged_hit"


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap(
    deltas: list[float], *, seed: int = SEED, resamples: int = RESAMPLES
) -> dict[str, object]:
    rng = random.Random(seed)
    count = len(deltas)
    means = [
        statistics.fmean(deltas[rng.randrange(count)] for _ in range(count))
        for _ in range(resamples)
    ]
    return {
        "seed": seed,
        "resamples": resamples,
        "observed_mean_delta": round(statistics.fmean(deltas), 9),
        "confidence_level": 0.95,
        "lower": round(percentile(means, 0.025), 9),
        "upper": round(percentile(means, 0.975), 9),
        "probability_delta_positive": round(
            sum(value > 0 for value in means) / resamples, 6
        ),
    }


def _summary(payload: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "sample_count",
        "hit_rate_at_10",
        "mrr",
        "mttc",
        "efficiency",
        "recommended_technical_score",
        "scenario_metrics",
        "response_contract_diagnostics",
        "determinism",
        "performance",
        "retrieval_configuration",
        "finalist_configuration",
        "clarification_diagnostics",
        "fallback_diagnostics",
    )
    return {key: payload.get(key) for key in keys if key in payload}


def compare(
    champion: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    seed: int = SEED,
    resamples: int = RESAMPLES,
) -> dict[str, object]:
    champion_sessions = champion.get("sessions")
    candidate_sessions = candidate.get("sessions")
    if not isinstance(champion_sessions, list) or not isinstance(
        candidate_sessions, list
    ):
        raise TypeError("both evaluator artifacts must contain sessions")
    before = {str(row["sample_id"]): row for row in champion_sessions}
    after = {str(row["sample_id"]): row for row in candidate_sessions}
    if set(before) != set(after):
        raise ValueError("paired artifacts do not contain the same sample ids")

    labels: Counter[str] = Counter()
    scenario_deltas: dict[str, list[float]] = defaultdict(list)
    rows: list[dict[str, object]] = []
    deltas: list[float] = []
    for sample_id in sorted(before):
        left, right = before[sample_id], after[sample_id]
        delta = session_score(right) - session_score(left)
        label = _label(left, right)
        scenario = str(left["scenario_type"])
        labels[label] += 1
        deltas.append(delta)
        scenario_deltas[scenario].append(delta)
        rows.append(
            {
                "sample_id": sample_id,
                "scenario_type": scenario,
                "comparison": label,
                "technical_score_contribution_delta": round(delta, 9),
                "champion": {
                    key: left.get(key)
                    for key in ("hit", "first_hit_turn", "best_rank", "reciprocal_rank")
                },
                "candidate": {
                    key: right.get(key)
                    for key in ("hit", "first_hit_turn", "best_rank", "reciprocal_rank")
                },
            }
        )
    return {
        "paired_session_count": len(rows),
        "counts": dict(sorted(labels.items())),
        "net_hits": labels["gained_hit"] - labels["lost_hit"],
        "scenario_mean_contribution_deltas": {
            name: round(statistics.fmean(values), 9)
            for name, values in sorted(scenario_deltas.items())
        },
        "bootstrap_technical_score_delta": paired_bootstrap(
            deltas, seed=seed, resamples=resamples
        ),
        "sessions": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    args = parser.parse_args()

    champion = _load(args.champion)
    candidate = _load(args.candidate)
    result = {
        "schema_version": 1,
        "inputs": {
            "champion_sha256": _sha256(args.champion),
            "candidate_sha256": _sha256(args.candidate),
        },
        "champion": _summary(champion),
        "candidate": _summary(candidate),
        "comparison": compare(
            champion, candidate, seed=args.seed, resamples=args.resamples
        ),
    }
    rendered = json.dumps(result, indent=2) + "\n"
    Path(args.output).write_text(rendered, encoding="utf-8")
    compact = dict(result)
    compact["comparison"] = {
        key: value for key, value in result["comparison"].items() if key != "sessions"
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
