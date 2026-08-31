# Issue 6A final ablation and clarification tuning

Date: 2026-08-30

Starting `main`: `a027eacec8cfc63a288afa2560f604d7c90bc58b`

Branch: `experiment/issue-6a-ablation-tuning`

## Decision

Select `clarification.browsing-only.v1` for Issue 6B. It exactly matches the
stored Issue 5C policy on every official aggregate/scenario metric and all 200
per-session hit/turn/rank outcomes, while asking 13 questions instead of 38.
The 25 removed Buying questions produced no public-session outcome change.

The selected full configuration fingerprint is
`405c3ff441211cc6073b3732e1bd60b7aa8e85698c8ceb7d7931fed8eeaeb6fd`.
The unchanged named rollback remains `contextual.browsing-dense.v1`; its
protected retrieval fingerprint remains
`972158c9e3905e4d0bb5390eb7224fe3fe00b9f5444337099b3422960bf0448a`.

## Evidence audit

The machine-readable manifest is
[`results/issue_6a/historical_evidence_manifest.json`](results/issue_6a/historical_evidence_manifest.json).
No unchanged historical system was rerun.

| Historical system | Source | Evidence status | Comparable official quality? |
| --- | --- | --- | --- |
| Official weak BM25 | `docs/baseline_results.json` at `166bae7` | Stored aggregate; exact reconstruction in `retrieval_regression_audit.md` | Yes |
| Issue 2B field-aware lexical | `docs/results/issue_2b/lexical.json` at `36a9974` | Stored raw 200-session output | Yes |
| Issue 2B dense-only | `docs/results/issue_2b/dense.json` at `36a9974` | Stored raw 200-session output | Yes |
| Issue 2B equal-weight fixed hybrid | `docs/results/issue_2b/hybrid.json` at `36a9974` | Stored raw 200-session output | Yes |
| Issue 3C fixed hybrid + deterministic feature reranking | `git:8fb7f27:.../docs/results/issue_3c/fixed_hybrid.json` | Historical Git-object raw output; metrics independently reconstructed in the current audit | Yes |
| Issue 3C route-aware + Boundary fallback | `git:8fb7f27:.../docs/results/issue_3c/route_aware.json` | Historical Git-object raw output; metrics independently reconstructed in the current audit | Yes |
| Issue 4B cross-encoder reranking | `git:3b21a08:.../docs/results/issue_4b/comparison.json` | Historical Git-object comparison; metrics independently reconstructed in the current audit | Yes |
| Stable contextual champion | `docs/results/recovery/contextual_policy_selection.json` | Stored aggregate/policy result; scenario metrics reconstructed below | Yes |
| Issue 5C clarification | `docs/results/issue_5c/selective_clarification.json` plus ignored local raw trace | Stored aggregate/scenario/per-session output | Yes |

Historical fingerprints and numeric/global seeds were not recorded except for
the promoted retrieval configuration. Champion latency, peak memory, and the
committed champion per-session trace were not recorded. Historical p50 latency
was not recorded. Historical latency/memory used different environments or
harnesses and is marked non-comparable with the fresh Issue 6A measurement.
There is no standalone deterministic-feature-reranker official result separate
from the Issue 3C fixed-hybrid-plus-feature pipeline.

## Paired diagnosis of the 38 Issue 5C questions

The committed aggregates omitted question-session identities and turn traces.
`benchmarks.analyze_clarification_pairs` therefore performed a benchmark-only,
read-only deterministic reconstruction of the two already evaluated policies.
It passed two validation gates: all 200 reconstructed Issue 5C hit/turn/rank
summaries exactly matched the stored raw output, and all 28 reconstructed
champion hit IDs matched the recovery artifact. The detailed machine output is
[`results/issue_6a/paired_diagnosis.json`](results/issue_6a/paired_diagnosis.json).

- Outcomes: 1 improved, 1 regressed, and 36 unchanged questioned sessions; 1
  new hit and 1 lost hit; no shared-hit earlier/later or better/worse-rank
  changes.
- Observable route: Browsing 13 questions (1 improved, 1 regressed, 11
  unchanged); Buying 25 (all unchanged).
- Attribute: category 28 (1 regression), feature 9 (1 improvement), style 1
  (unchanged).
- Question turn: turn 1 = 33, turn 2 = 4, turn 4 = 1.
- Evaluator follow-up: 25 declines, 7 attribute-bearing explicit answers, and 6
  sessions ending before a reply. Runtime controller state recorded the 25
  declines; the 7 answer messages remained unresolved by the finite explicit
  attribute parser even though their exact raw text still drove retrieval.
- Later recommendations changed in 32 questioned sessions: all 13 Browsing and
  19 of 25 Buying.
- Benchmark scenario: Buying 22 questioned (all unchanged), Browsing 11 (one
  gain/one loss), Intent Override 3 (all unchanged), Boundary 2 (all unchanged).
  No scenario HR@10 collapsed; every finalist has the same four scenario HR@10
  values.

Mechanism evidence is factual and remains benchmark-only. In `public_0070`, a
turn-1 Browsing/category question elicited a decline on turn 2. The champion
instead received the evaluator's standard negative-feedback message, so the
clarification stream's known-negative rotation lagged by one turn. The champion
found the target on turn 10 at rank 9; the clarification stream did not reach
that rotation before the turn limit. In `public_0102`, a turn-1
Browsing/feature question elicited feature text on turn 2. Preserving that exact
raw answer changed retrieval and placed the target at rank 1; the champion's
generic follow-up stream missed. No code or threshold names either session.

The 25 Buying questions showed no aggregate-value subset, so a stricter Buying
policy was not justified or evaluated.

## Predeclared finalists

All values live in `config/clarification_policies.json`; fingerprints include
the name, retrieval-policy reference, fixed seed, clarification gates, and
controller limits.

| Policy | Full fingerprint | Clarification routes | Questions/session | Seed |
| --- | --- | --- | ---: | ---: |
| `contextual.browsing-dense.v1` | `04a16f9cec5162ab8a3d6ecff098c0342d205a37d24b5665a36316ba4f64f8a6` | disabled | 0 | 20260830 |
| `clarification.issue-5c.v1` | `307822c299d9c3614f06215ecb5118107a5264f1d2efb7b704cf3787018ce1ed` | Browsing, Buying | at most 1 | 20260830 |
| `clarification.browsing-only.v1` | `405c3ff441211cc6073b3732e1bd60b7aa8e85698c8ceb7d7931fed8eeaeb6fd` | Browsing | at most 1 | 20260830 |

The Issue 5C values were not retrospectively changed. Browsing-only differs
only in `eligible_routes`. The rollback clarification configuration equals the
legacy disabled dataclass and retains the protected retrieval fingerprint.

## Finalist comparison

Evidence labels: **stored** = immutable committed artifact; **reconstructed** =
exact derivation from stored per-session records plus the validated paired
trace; **fresh** = the single Issue 6A official run; **missing** = not recorded;
**NC** = recorded performance but not comparable with the fresh method.

| Policy | Evidence | HR@10 | MRR | MTTC | Efficiency | TechnicalScore | Buying HR | Browsing HR | Intent HR | Boundary HR | Mean / p50 / p95 ms | Peak RSS | Questions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Stable champion | Overall stored; scenario reconstructed; performance missing | 0.140 | 0.070423 | 9.780 | 0.122 | 0.115527 | 0.2375 | 0.0625 | 0.133333 | 0.0000 | missing | missing | 0 |
| Exact Issue 5C | Quality stored; performance stored/NC | 0.140 | 0.074867 | 9.740 | 0.126 | 0.117660 | 0.2375 | 0.0625 | 0.133333 | 0.0000 | 27.252 / missing / 25.695 NC | 1,079,312,384 NC | 38 |
| Browsing-only | Fresh | 0.140 | 0.074867 | 9.740 | 0.126 | 0.117660 | 0.2375 | 0.0625 | 0.133333 | 0.0000 | 32.499 / 27.459 / 31.195 | 1,071,824,896 | 13 |

The reconstructed champion Browsing supplementary metrics are MRR `0.010486`
and MTTC `10.675`; the stored/fresh clarification rows are MRR `0.021597` and
MTTC `10.575`. The other scenario MRR/MTTC values are identical across all
three finalists. Relative to the stable champion, Browsing-only changes HR@10
by `0`, MRR by `+0.004444`, MTTC by `-0.040`, Efficiency by `+0.004`, and
TechnicalScore by `+0.002133`.

Issue 5C and Browsing-only have zero per-session metric mismatches. Both lose
`public_0070` and gain `public_0102` relative to the champion. This is a
robustness diagnostic, not an automatic veto; the aggregate official metrics
improve. Browsing-only removes 25 questions without changing any official or
per-session selection result.

## Correctness gates

All three finalists are eligible. Browsing-only freshly recorded zero response
exceptions, clarification failures, invalid ASINs, duplicates, invalid
question attributes, repeated questions, or response-contract violations. The
controller prevents turn-10 questions and the evaluator caps sessions at ten
turns. Runtime configuration contains no target ASINs, public IDs, hidden
labels, or stored hit lists. Fixed-seed deterministic tests pass. Clarification
failures return the unchanged recommendation response.

The stable control's zero-exception result is stored in the recovery gate;
invalid/duplicate safety follows the shared validated Agent output contract and
exact disabled response parity. Issue 5C's committed artifact records every
correctness count at zero.

## Verification and evaluator budget

| Check | Result | Wall time |
| --- | --- | ---: |
| Policy names/values/fingerprints, Browsing eligibility, rollback parity | 7 passed | 0.24 s |
| Existing 5A/5B/5C tests | 56 passed | 0.36 s |
| State/route/retrieval/response-contract checkpoint | 118 passed | 0.50 s |
| Final full unit suite | 208 passed | 0.74 s |
| Ruff on all Issue 6A Python files | passed | 0.08 s |
| Ruff format check on all Issue 6A Python files | passed | 0.06 s |
| Focused MyPy on six runtime/evaluator/analysis files | passed | 0.43 s |
| JSON parsing and `git diff --check` | passed | — |

Repository-wide Ruff remains non-clean with 23 pre-existing findings in
untouched dense, lexical, fallback, router, and historical test files. They were
not expanded into this issue or modified. No new Issue 6A Ruff/MyPy finding
remains.

Exactly one new official 200-session configuration was run, exactly once:

```bash
python3 -m evaluator.local_evaluator \
  --retrieval-mode contextual \
  --contextual-policy contextual.browsing-dense.v1 \
  --clarification-policy clarification.browsing-only.v1 \
  --output diagnostics/retrieval_regression/issue_6a_browsing_only_200.json
```

Seed: `20260830`. Duration: `69.70 s`. Warmup responses: `0`. Latency measures
every `Agent.respond` call including first-query model loading; memory is peak
process RSS. The official BM25 baseline, rejected historical systems, champion,
and Issue 5C were not rerun. The raw fresh trace is reproducible and ignored;
the concise committed result is
[`results/issue_6a/final_comparison.json`](results/issue_6a/final_comparison.json).

## Selection limits

TechnicalScore was primary. Exact Issue 5C and Browsing-only tie on every
official quality metric; Browsing-only wins the specified tie-break because it
is narrower and asks 25 fewer questions. Historical/fresh latency and memory
were not compared directly, so the selection does not claim a measured speed
or memory win over Issue 5C.

The public 200 sessions influenced development and are not a pristine holdout.
No paired-bootstrap facility existed, and scope was not expanded merely to add
one. This evidence supports a bounded deterministic engineering choice, not
statistical certainty or guaranteed private-set improvement. Issue 6B
hardening is intentionally not implemented here.
