# TechnicalScore iteration 01: feedback-memory BM25 rotation

Date: 2026-08-30

## Decision

**Promising.** The official 200-session evaluator improved TechnicalScore from
`0.117660` to `0.305993`. The result reproduced exactly in a second run. This
branch remains an isolated experiment: the existing runtime default and its
rollback fingerprints are unchanged, and nothing has been merged.

## Experiment identity

- Branch: `experiment/technical-score-iter-01`
- Parent commit: `c57dfce8f7aeb44cb366ab3a90c11e9ac613cf07`
- Retrieval policy: `contextual.feedback-memory.v1`
- Clarification policy: `clarification.feedback-memory.v1`
- Retrieval fingerprint: `9feee454891b5557c268bfe1e5942f04047a83896ba9ffdad5b59ed2b887a9be`
- Clarification fingerprint: `56550e0f09f152db8be1a3988bacda499955e8b5683e95eedad44b3ce19fb7a5`
- Dataset SHA-256: `857259f7a438e6188ac63e18995b6ff4489bfcfc4a716a798b9a2aa0ee8f7579`
- Catalog SHA-256: `da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`
- Evaluation seed: `20260830`

The local dense cache passed the implementation's metadata validation. Its
container SHA-256 was
`12eca9d32623a0f7bc3790799e197911089546d50d4a7a8e081c91d576937bfd`;
the official output and configuration hashes, rather than the container byte
hash, were used for determinism checks.

## Hypothesis

The evaluator emits a generic negative-feedback turn when the agent does not
ask a clarification question:

```text
Those options are not quite right yet. Ask me about one specific attribute.
```

The selected production policy already keeps the last informative user turn
for dense retrieval, but sends the generic rejection text to raw-turn BM25.
That can remove the lexical signal precisely when known-negative rotation
needs it.

Iteration 01 changes only this behavior behind a named policy: on recognized
negative feedback, BM25 reuses the last informative user turn while continuing
to exclude every previously rejected recommendation. All non-negative turns
still use exact raw-turn BM25. The eight-item BM25 protection, selective
Browsing dense tail, override clearing, clarification gates, candidate sizes,
and ranking weights are unchanged.

## Audit disposition

Kept enabled in the experiment:

- exact raw-turn BM25 on every normal turn;
- protected BM25 anchors and known-negative exclusion;
- conversation state and intent-override clearing;
- validated MiniLM cache and Browsing-only dense residual evidence;
- the selected Browsing-only clarification thresholds and one-question limit.

Kept disabled by default:

- global field-aware lexical filtering;
- global equal-weight RRF;
- route-aware fallback;
- feature and semantic reranking;
- any LLM or token-consuming generation.

## Files changed

- `starter/contextual_retrieval.py`: add the opt-in query-memory setting and
  named retrieval policy.
- `starter/agent.py`: select the remembered informative text for BM25 only
  when the policy is enabled and the current turn is recognized negative
  feedback.
- `evaluator/local_evaluator.py`: preserve historical policy fingerprints and
  permit the named experimental clarification policy.
- `config/clarification_policies.json`: declare the matching fixed-seed
  clarification policy without changing the runtime default.
- `tests/test_contextual_retrieval.py`: cover opt-in memory reuse and exact
  default-policy compatibility.
- `tests/test_clarification_policies.py`: lock the new policy fingerprint.
- this report.

## Exact validation commands

All commands ran from `techjam-conversational-search-participant-kit`.

```bash
/private/tmp/tiktok-techscore-venv-20260830/bin/python -m unittest discover -s tests -v

/private/tmp/tiktok-techscore-venv-20260830/bin/python -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --retrieval-mode contextual \
  --contextual-policy contextual.browsing-dense.v1 \
  --clarification-policy clarification.browsing-only.v1 \
  --dense-cache data/.dense-retrieval/catalog-minilm.npz \
  --output /private/tmp/technical-score-baseline-c57dfce.json

/private/tmp/tiktok-techscore-venv-20260830/bin/python -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --retrieval-mode contextual \
  --contextual-policy contextual.feedback-memory.v1 \
  --clarification-policy clarification.feedback-memory.v1 \
  --dense-cache data/.dense-retrieval/catalog-minilm.npz \
  --output /private/tmp/technical-score-iter-01-feedback-memory.json

/private/tmp/tiktok-techscore-venv-20260830/bin/python -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --retrieval-mode contextual \
  --contextual-policy contextual.feedback-memory.v1 \
  --clarification-policy clarification.feedback-memory.v1 \
  --dense-cache data/.dense-retrieval/catalog-minilm.npz \
  --output /private/tmp/technical-score-iter-01-feedback-memory-rerun.json

/private/tmp/tiktok-techscore-venv-20260830/bin/python -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --retrieval-mode contextual \
  --contextual-policy contextual.browsing-dense.v1 \
  --clarification-policy clarification.browsing-only.v1 \
  --dense-cache data/.dense-retrieval/catalog-minilm.npz \
  --output /private/tmp/technical-score-default-parity-iter-01.json

ruff check starter/agent.py starter/contextual_retrieval.py \
  evaluator/local_evaluator.py tests/test_contextual_retrieval.py \
  tests/test_clarification_policies.py

ruff format --check starter/agent.py starter/contextual_retrieval.py \
  evaluator/local_evaluator.py tests/test_contextual_retrieval.py \
  tests/test_clarification_policies.py
```

## Official metrics

| Metric | BM25 | Current solution | Iteration 01 | Delta vs current |
| --- | ---: | ---: | ---: | ---: |
| HitRate@10 | 0.125 | 0.140 | **0.405** | **+0.265** |
| MRR | 0.068034 | 0.074867 | **0.153310** | **+0.078443** |
| MTTC | 9.810 | 9.740 | **8.125** | **-1.615** |
| Efficiency | 0.1190 | 0.1260 | **0.2875** | **+0.1615** |
| TechnicalScore | 0.106710 | 0.117660 | **0.305993** | **+0.188333** |

TechnicalScore improved by `160.07%` relative to the current solution,
`164.87%` relative to the stable contextual retrieval-only score `0.115527`,
and `186.75%` relative to BM25.

## Scenario metrics

| Scenario | Current HR@10 | Iteration 01 HR@10 | Absolute delta |
| --- | ---: | ---: | ---: |
| Buying | 0.2375 | **0.5875** | **+0.3500** |
| Browsing | 0.0625 | **0.3125** | **+0.2500** |
| Intent Override | 0.133333 | **0.166667** | **+0.033334** |
| Boundary | 0.0000 | **0.4000** | **+0.4000** |

## Paired accuracy analysis

Against the current `0.117660` solution:

- baseline hits: 28;
- iteration hits: 81;
- new hits: 54;
- lost hits: 1 (`public_0137`, Browsing);
- net additional hits: 53;
- shared-hit later-turn/worse-rank regressions under the existing promotion
  rule: 0;
- one shared hit (`public_0085`) moved earlier from turn 9/rank 5 to turn
  5/rank 10.

Against exact BM25, all 25 baseline-success sessions were retained and 56
sessions were gained. Recommendation identities remained catalog-valid and
unique, and both full runs reported zero response exceptions, invalid ASINs,
duplicates, invalid question attributes, repeated questions, or response
contract violations.

The one lost current-only hit is explainable: the remembered BM25 query
repopulates the protected prefix on a later Browsing turn that previously had
a nearly empty lexical anchor, displacing a dense-only success. The aggregate
gain is large, but this tradeoff should remain visible rather than being
described as strict current-solution retention.

## Efficiency and feasibility

| Measure | Current fresh run | Iteration run 1 | Iteration run 2 |
| --- | ---: | ---: | ---: |
| Responses | 1,776 | 1,506 | 1,506 |
| Mean response latency | 29.607 ms | 15.372 ms | 15.924 ms |
| p95 response latency | 17.008 ms | 22.207 ms | 22.916 ms |
| Maximum response latency | 26,579.753 ms | 4,039.481 ms | 4,091.846 ms |
| Full evaluator wall time | 52.701 s | 23.257 s | 24.102 s |
| Peak RSS | 1,053.891 MiB | 1,108.984 MiB | 1,107.641 MiB |
| Clarification questions | 13 | 13 | 13 |
| Token usage | 0 | 0 | 0 |

The p95 cost increased by about 5-6 ms because negative-feedback turns now run
BM25 instead of querying generic text that usually matches nothing. Peak RSS
increased by about 55 MiB. Mean latency, maximum latency, response count, and
wall time decreased, primarily because many sessions terminate earlier. Model
loading and OS page-cache state make the maximum-latency comparison
environment-sensitive.

Candidate limits are unchanged: BM25 Top-100, dense Top-200, final Top-10, and
clarification analysis Top-50.

## Determinism and quality gates

- Full unit suite: 230 passed.
- Ruff lint on changed Python files: passed.
- Ruff format check on changed Python files: passed.
- Two official evaluator runs had identical session records, metrics,
  retrieval configuration, normalized-response hash, and session-outcome hash.
- A post-change parity run of the unchanged runtime default reproduced
  TechnicalScore `0.117660` and both historical output hashes exactly.
- Normalized-response SHA-256:
  `9e50a9e37aea0e149bfaff346266d24628679558c1999bde36146681d67bca6d`.
- Session-outcome SHA-256:
  `ee2d7a3a683273d17dcb2e8a9d0dcc1ab7c05816c4c611a43be9070133e45e67`.

## Recommended next experiment

Evaluate a confidence-aware Browsing residual guard that preserves a strong
dense-only tail candidate when the feedback-memory BM25 anchor is repopulated.
The narrow objective should be recovering `public_0137` or an equivalent
held-out pattern without losing the 54 gains from this iteration. It must be a
separate branch from this commit and must retain this branch as the measured
control.
