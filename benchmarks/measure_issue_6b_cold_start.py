"""Measure Issue 6B cold Agent construction in independent processes."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from evaluator.local_evaluator import current_process_rss_bytes, peak_process_rss_bytes
from starter.agent import Agent


def child_measurement(catalog: str) -> dict[str, object]:
    rss_before = current_process_rss_bytes()
    started = time.perf_counter()
    Agent(catalog)
    duration = time.perf_counter() - started
    return {
        "duration_seconds": round(duration, 6),
        "rss_before_bytes": rss_before,
        "rss_after_bytes": current_process_rss_bytes(),
        "peak_rss_bytes": peak_process_rss_bytes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        print(json.dumps(child_measurement(args.catalog), sort_keys=True))
        return
    if args.runs < 3:
        raise ValueError("cold-start measurement requires at least three runs")
    measurements = []
    for _index in range(args.runs):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.measure_issue_6b_cold_start",
                "--child",
                "--catalog",
                args.catalog,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        measurements.append(json.loads(completed.stdout))
    durations = [float(item["duration_seconds"]) for item in measurements]
    result = {
        "method": "Agent construction in independent Python processes; no response warmup",
        "run_count": len(measurements),
        "median_duration_seconds": round(statistics.median(durations), 6),
        "maximum_duration_seconds": round(max(durations), 6),
        "runs": measurements,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
