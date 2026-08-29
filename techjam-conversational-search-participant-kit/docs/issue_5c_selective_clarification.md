# Issue 5C selective clarification

Date: 2026-08-30

## Decision

Issue 5C is experimentally complete but remains disabled. The enabled policy
improved aggregate MRR, MTTC, Efficiency, and TechnicalScore, but retained only
27 of the immutable champion's 28 hit sessions. It lost `public_0070` while
gaining `public_0102`, so the mandatory all-hit retention gate failed.

The default remains `contextual.browsing-dense.v1` with selective clarification
off. No merge into `main` was performed.

## Integration

`Agent.respond` resolves only explicit pending answers or declines, then uses
the existing Issue 1A state update and runs the promoted contextual retriever
once. When the feature is enabled, the already-ranked bounded pool retains at
most 50 candidates for Issue 5A analysis. Issue 5B may then attach one question
without changing recommendation identities, order, count, usage, or raw-message
evidence. Boundary and Uncertain routes do not ask; Buying uses a stricter
expected-reduction threshold than Browsing.

Clarification-only failures return the unchanged recommendation response. The
feature flag is `SelectiveClarificationConfig.enabled` and defaults to `False`.

## Configuration parity

The exact runtime policy matches the selected policy object in
`contextual_policy_selection.json`. The deterministic recommendation-affecting
configuration fingerprint is:

`972158c9e3905e4d0bb5390eb7224fe3fe00b9f5444337099b3422960bf0448a`

The legacy recovery artifacts do not contain a named fingerprint field, so the
value above was derived from their exact policy plus the documented evaluator
defaults. Disabled response parity passed, and the analyzer was not invoked.

## Results

| Metric | Champion | Clarification enabled | Gate |
| --- | ---: | ---: | --- |
| HR@10 | 0.140 | 0.140 | PASS |
| MRR | 0.070423 | 0.074867 | PASS |
| MTTC | 9.780 | 9.740 | PASS |
| Efficiency | 0.122 | 0.126 | PASS |
| TechnicalScore | 0.115527 | 0.117660 | PASS |

The enabled run asked 38 questions: 28 `category`, 9 `feature`, and 1 `style`.
By observable runtime route, 13 were Browsing and 25 were Buying. It produced
zero repeated questions, evaluator exceptions, clarification failures, invalid
attributes, invalid ASINs, and duplicate recommendations.

Champion retention was 27/28. `public_0070` became a miss. The new hit was
`public_0102` at turn 2, rank 1. The 25 stored exact-BM25 anchor hits had no
regression relative to their stored BM25 turn/rank details.

Exact champion per-scenario metrics and the complete 28-session contextual
turn/rank trace are not present in the committed recovery artifacts; the raw
trace named in the recovery document was intentionally uncommitted and absent.
Those gates therefore cannot be certified as passing, and no values were
invented or approximated. The lost champion hit independently blocks promotion.

## Verification

- Focused 5C: 15 tests passed.
- Existing 5A/5B: 41 tests passed after adding the decline-recognition assertion.
- Relevant state/retrieval/contract checkpoint: 38 tests passed.
- Full suite: 202 tests passed.
- Ruff formatting and lint: passed on every changed Python file.
- Focused MyPy: passed for the new policy, controller, and state-parser surface.
- Broader MyPy remains non-clean with 24 pre-existing errors in unrelated
  ambiguity/reranker/fallback and legacy Agent typing paths; none were expanded
  into scope.
- Disabled parity: passed with the unchanged champion fingerprint and exact
  response equality.
- Enabled official evaluator: run exactly once.

The concise machine-readable evidence is
[`results/issue_5c/selective_clarification.json`](results/issue_5c/selective_clarification.json).
The reproducible raw trace remains local and ignored at
`diagnostics/retrieval_regression/issue_5c_clarification_enabled_200.json`.
