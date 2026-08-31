# Shadow-suite results

This directory stores deterministic robustness-screen summaries for candidates
before they are permitted to consume an official public-set evaluation run.
The shadow suite is benchmark-only and is not loaded by the evaluator-facing
agent.

The terminal P7 evidence is in `p7_constraint_coverage_shadow.json`. It records
exact disabled P5 parity, deterministic P7 replay, paired and scenario deltas,
attribution, proxy checks, failure counters, and the frozen rejection verdict.
