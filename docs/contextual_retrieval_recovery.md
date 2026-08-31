# BM25 contextual retrieval recovery

Date: 2026-08-30

## Decision

Promote `contextual.browsing-dense.v1` behind the explicit `contextual`
retrieval mode and make that mode the default. Exact raw-turn BM25 remains the
lexical foundation. The selected policy adds only two contextual behaviors:

- products explicitly rejected by the shopper are excluded on later turns;
- dense evidence may fill two unprotected positions only on routed Browsing
  turns.

An intent-override cue clears the known-negative product set. BM25 occupies the
first eight non-negative positions unchanged, and all remaining evidence is
ranked with deterministic score and identity tie-breaks. Feature reranking,
route-aware hard filtering, Boundary fallback, and state-aware lexical
reranking are disabled in the selected policy.

## Promotion gates

The offline selector compared exact `bm25` against four frozen challengers on
the 200-session public set. A challenger had to produce zero response
exceptions, retain every BM25-success session, introduce no later-turn or
worse-rank BM25 hit, gain at least one session, and improve TechnicalScore.

| Policy | BM25 hits lost | New hits | HR@10 | TechnicalScore | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Exact BM25 | — | — | 0.125 | 0.106710 | Baseline |
| Negative rotation | 0 | 2 | 0.135 | 0.112760 | Pass |
| State-aware lexical | 1 | 3 | 0.135 | 0.112744 | Reject |
| Browsing-only dense | 0 | 3 | 0.140 | 0.115527 | **Selected** |
| State + Browsing dense | 1 | 4 | 0.140 | 0.115910 | Reject |

The state-aware variants lost `public_0143`, so their higher raw gain count was
not safe incremental value. The selected policy gained `public_0070`,
`public_0085`, and `public_0137` while retaining all 25 exact-BM25 successes.
The concise machine-readable evidence is
[`results/recovery/contextual_policy_selection.json`](results/recovery/contextual_policy_selection.json).

## Official evaluator confirmation

After the offline gate passed, the full evaluator was run for the selected
policy. It reproduced the selector metrics with zero response exceptions:

| HR@10 | MRR | MTTC | Efficiency | TechnicalScore |
| ---: | ---: | ---: | ---: | ---: |
| 0.140 | 0.070423 | 9.780 | 0.122 | 0.115527 |

The raw 200-session evaluator output is intentionally not committed. It is a
reproducible 38 KB local trace at
`diagnostics/retrieval_regression/contextual_full_200.json`.

## Reproduction

From the repository kit directory:

```bash
python3 -m benchmarks.verify_anchor_recovery \
  --selection diagnostics/retrieval_regression/official_bm25_hits.json \
  --output diagnostics/retrieval_regression/recovery_verification.json

python3 -m benchmarks.select_contextual_policy \
  --output docs/results/recovery/contextual_policy_selection.json

python3 -m evaluator.local_evaluator \
  --retrieval-mode contextual \
  --contextual-policy contextual.browsing-dense.v1 \
  --output diagnostics/retrieval_regression/contextual_full_200.json
```

Inputs used for the recorded result:

- catalog SHA-256: `da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`
- public set SHA-256: `857259f7a438e6188ac63e18995b6ff4489bfcfc4a716a798b9a2aa0ee8f7579`
- dense cache SHA-256: `dffc05e41866913f169c3a51ef295b47f43859a0da6bb0c6a4c19de77fd1a5f6`

The catalog, dense cache, raw official-hit diagnostic trace, and raw full
evaluator output are local reproducible artifacts and must not be committed.
