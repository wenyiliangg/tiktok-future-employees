# Issue 6B evaluator-facing agent hardening

Date: 2026-08-30

Starting `main`: `7b8cf08076b8198a616de3ee74448b0e3f2c722d`

Branch: `feature/issue-6b-agent-hardening`

## Verdict

**PASS.** Two identical post-hardening 200-session runs preserve every Issue
6A metric, scenario result and all 13 Browsing questions. Their normalized
responses, recommendation order, question fields and per-session hit/turn/rank
outcomes have identical SHA-256 hashes. Both runs record zero exceptions,
invalid responses/ASINs, duplicates, repeated questions, contract violations,
clarification failures and fallback activations.

## Configuration gate

- Selected: `clarification.browsing-only.v1`, fingerprint
  `405c3ff441211cc6073b3732e1bd60b7aa8e85698c8ceb7d7931fed8eeaeb6fd`.
- Rollback: `contextual.browsing-dense.v1`, full disabled-clarification
  fingerprint
  `04a16f9cec5162ab8a3d6ecff098c0342d205a37d24b5665a36316ba4f64f8a6`
  and protected retrieval fingerprint
  `972158c9e3905e4d0bb5390eb7224fe3fe00b9f5444337099b3422960bf0448a`.
- Seed: `20260830`.

The external registry is unchanged. Runtime loads the selected policy by stable
name and fingerprint. If only that entry is invalid, it loads the rollback by
its stable name and independently validated fingerprint. If both are invalid,
construction fails clearly.

## Central response boundary

`starter.response_validation.validate_response` is pure over the proposed
response, the already-ranked bounded candidates and the immutable catalog-ID
set. It performs membership checks only; it cannot retrieve or scan the
catalog.

`top_k` is an upper bound capped by the response contract's 100-item maximum.
A non-integer, boolean or non-positive value returns zero recommendations.
Valid unique response entries keep their order, followed by valid unique
entries from the same bounded ranking when invalid or duplicate entries need
replacement. Fewer than `top_k` are returned when the bounded pool is
exhausted. The validator never pads, reruns retrieval or reorders valid items.
Malformed clarification fields are dropped without dropping recommendations.
Valid finite scores and exact non-negative usage fields are preserved.

The official turn request fixes `top_k` at 10; zero is supported as a safe
direct-call boundary consistent with the repository's existing behavior.
Empty/whitespace messages update state deterministically but skip retrieval and
model work.

## Failure matrix

| Injected failure | Deterministic outcome |
| --- | --- |
| Missing/corrupt dense cache | Protected raw-message BM25 |
| Dense initialization/query | Protected raw-message BM25 |
| Router exception/invalid route | Protected raw-message BM25 order |
| Contextual fusion | Already retrieved BM25 order; no second retrieval |
| Analyzer, eligibility, attribute, controller or question composition | Same recommendations; clarification omitted |
| Selected config invalid | Verified named rollback |
| Selected and rollback invalid | Clear initialization failure |
| Unexpected ordinary response exception | One non-recursive protected BM25 attempt |
| Protected BM25 attempt also fails | Smallest valid empty recommendation response |

All component counters are internal snapshots and never enter the response or
stdout stream. Dense model loading is pinned, offline-only and rejects remote
code. A missing local model therefore activates the existing BM25 fallback
instead of contacting a hosted service.

## State and turn isolation

Focused tests verify idempotent repeated reset, fresh-agent/reset-agent parity,
consecutive session isolation, pending clarification clearing, unrelated
follow-up handling, turn 9/10 behavior, deterministic identical transcripts
and failure isolation. Preferences, exclusions, raw intent, known negatives,
last recommendations, bounded candidates, route state, clarification state and
session caches reset together. Immutable catalog and embedding indexes persist.

The two official processes finish with similar post-session RSS (1,098,399,744
and 1,126,301,696 bytes), with no continuing cross-run growth signal. The
increase from post-construction RSS reflects the lazily loaded local dense model
and embedding working set, not customer/session state.

## Verification

Commands were run from `techjam-conversational-search-participant-kit`:

```bash
python3 -m unittest tests.test_response_validation tests.test_agent_reliability
python3 -m unittest tests.test_ambiguity_analysis tests.test_clarification_controller tests.test_selective_clarification tests.test_clarification_policies
python3 -m unittest tests.test_conversation_state tests.test_intent_router tests.test_bm25_anchor tests.test_contextual_retrieval tests.test_dense_retrieval tests.test_hybrid_retrieval tests.test_evaluator tests.test_response_validation tests.test_agent_reliability
python3 -m unittest discover -s tests
python3 -m ruff check <changed Python files>
python3 -m ruff format --check <changed Python files>
python3 -m mypy --follow-imports=skip starter/clarification_policies.py starter/response_validation.py benchmarks/measure_issue_6b_cold_start.py evaluator/local_evaluator.py
python3 -m benchmarks.measure_issue_6b_cold_start --runs 3 --output diagnostics/retrieval_regression/issue_6b_cold_start.json
python3 -m evaluator.local_evaluator --retrieval-mode contextual --contextual-policy contextual.browsing-dense.v1 --clarification-policy clarification.browsing-only.v1 --output diagnostics/retrieval_regression/issue_6b_run_1.json
python3 -m evaluator.local_evaluator --retrieval-mode contextual --contextual-policy contextual.browsing-dense.v1 --clarification-policy clarification.browsing-only.v1 --output diagnostics/retrieval_regression/issue_6b_run_2.json
```

Results: 20 new focused tests, 62 5A/5B/5C/6A checkpoint tests, 94
state/routing/retrieval/contract tests and 229 full-suite tests pass. Ruff check
and format pass on changed files. Focused MyPy passes on the new boundary,
policy loader, evaluator and measurement code. Existing `agent.py` query-model
versus reranker type findings were not expanded into this issue.

## Official result and determinism

Both runs produced:

| Metric | Value |
| --- | ---: |
| HR@10 | 0.140 |
| MRR | 0.074867 |
| MTTC | 9.740 |
| Efficiency | 0.126 |
| TechnicalScore | 0.117660 |
| Buying HR@10 | 0.2375 |
| Browsing HR@10 | 0.0625 |
| Intent Override HR@10 | 0.133333 |
| Boundary HR@10 | 0.000 |
| Browsing questions | 13 |

Normalized response hash:
`318f069ef485d23d74d689ff25a3a58db874bfa3f412fcdb541c38cfaae4f91c`.

Session outcome hash:
`b5e4c9d3643d03aac697747e9e37cb202af2fee550ba0e2db235f7e126f4f165`.

Timing is excluded from determinism. The first diagnostic run revealed an
upstream model Hub probe and was invalidated before the final pair. After
enforcing offline-only local model loading, Final Run 1 took 56.022 s and Final
Run 2 took 54.569 s.

## Runtime and memory

Cold construction was measured in three independent Python processes without a
warmup response: 9.968, 9.842 and 9.966 s; median 9.966 s and maximum 9.968 s.

| Run | Mean ms | p50 ms | p95 ms | Max ms | Peak RSS bytes | Peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 31.426 | 27.085 | 30.691 | 8881.442 | 1,115,684,864 | 1064.000 |
| 2 | 30.614 | 26.805 | 29.283 | 8060.185 | 1,137,213,440 | 1084.531 |

Latency measures every `Agent.respond` wall time with zero warmups, so the
maximum includes first local model load. Peak RSS uses `resource.getrusage` and
converts platform units to bytes; current RSS uses `ps` KiB multiplied by 1024.
The Issue 6A environment/method is not sufficiently controlled for a direct
performance claim, so the comparison is marked non-comparable.

The concise machine-readable evidence is
[`results/issue_6b/reliability.json`](results/issue_6b/reliability.json). Raw
traces remain ignored and are not committed.
