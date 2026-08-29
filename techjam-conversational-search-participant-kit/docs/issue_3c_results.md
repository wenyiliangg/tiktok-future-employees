# Issue 3C route-aware retrieval results

## Reproduction environment

- Date: 2026-08-29
- Branch: `feature/issue-3c-route-aware-retrieval`
- Base commit: `f9d668937bac5157795e9b30bf24a6f6069d87a0`
- Host: Apple arm64, macOS 26.5.2
- Runtime: Python 3.12.2, NumPy 2.5.2, sentence-transformers 5.7.0,
  PyTorch 2.13.0, CPU inference
- Dense model/cache: `sentence-transformers/all-MiniLM-L6-v2`, pinned revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, existing compatible
  `data/.dense-retrieval/catalog-minilm.npz`
- Catalog SHA-256:
  `da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`
- Public-set SHA-256:
  `857259f7a438e6188ac63e18995b6ff4489bfcfc4a716a798b9a2aa0ee8f7579`

The diagnostic runner calls the unchanged
`evaluator.local_evaluator.evaluate` function. Both modes used the same frozen
catalog, 200-session public set, cache, top-10 limit, 10-turn limit, process,
and evaluator settings. Raw artifacts are
[`fixed_hybrid.json`](results/issue_3c/fixed_hybrid.json) and
[`route_aware.json`](results/issue_3c/route_aware.json).

```bash
python3 -m benchmarks.evaluate_route_aware --retrieval-mode hybrid \
  --lexical-candidates 200 --dense-candidates 200 --final-candidates 10 \
  --lexical-weight 1.0 --dense-weight 1.0 --rrf-k 60 \
  --output docs/results/issue_3c/fixed_hybrid.json

python3 -m benchmarks.evaluate_route_aware --retrieval-mode route-aware \
  --output docs/results/issue_3c/route_aware.json
```

## Overall comparison

Positive relative changes mean a numeric increase; lower MTTC and latency are
better, so their positive changes are regressions.

| Metric | Fixed hybrid | Route-aware | Absolute change | Relative change |
| --- | ---: | ---: | ---: | ---: |
| HR@10 | 0.020000 | 0.015000 | -0.005000 | -25.00% |
| MRR | 0.007500 | 0.006833 | -0.000667 | -8.89% |
| MTTC | 10.815000 | 10.865000 | +0.050000 | +0.46% |
| Efficiency | 0.018500 | 0.013500 | -0.005000 | -27.03% |
| TechnicalScore | 0.015950 | 0.012250 | -0.003700 | -23.20% |
| Average response latency (ms) | 61.647544 | 235.433495 | +173.785951 | +281.90% |
| p95 response latency (ms) | 105.677583 | 130.108708 | +24.431125 | +23.12% |
| Agent startup (s) | 23.912868 | 37.606661 | +13.693793 | +57.27% |
| Peak RSS (GB, decimal) | 1.911570 | 2.575254 | +0.663684 | +34.72% |
| Completed sessions | 200 | 200 | 0 | 0% |
| Response exceptions | 0 | 0 | 0 | n/a |

Average route-aware latency exceeds p95 because uncached first fallback calls
are a small set of large catalog-scoring outliers. Repeated identical active
Boundary state is cached within a session; cache state is cleared by `reset()`.

## Scenario metrics

| Scenario | Mode | HR@10 | MRR | MTTC |
| --- | --- | ---: | ---: | ---: |
| Buying | Fixed hybrid | 0.037500 | 0.016250 | 10.625000 |
| Buying | Route-aware | 0.025000 | 0.015000 | 10.750000 |
| Browsing | Fixed hybrid | 0.000000 | 0.000000 | 11.000000 |
| Browsing | Route-aware | 0.000000 | 0.000000 | 11.000000 |
| Intent Override | Fixed hybrid | 0.033333 | 0.006667 | 10.766667 |
| Intent Override | Route-aware | 0.033333 | 0.005556 | 10.766667 |
| Boundary | Fixed hybrid | 0.000000 | 0.000000 | 11.000000 |
| Boundary | Route-aware | 0.000000 | 0.000000 | 11.000000 |

Buying route-aware HR@10 changed by `-0.012500` (`-33.33%`) and MRR by
`-0.001250` (`-7.69%`). Intent Override HR@10 was unchanged, while its MRR
changed by `-0.001111` (`-16.66%`). Browsing and Boundary remained at zero.

## Route, failure, fallback, and output diagnostics

The 1,976 route-aware responses selected:

| Route | Response count |
| --- | ---: |
| Buying | 1,156 |
| Uncertain | 518 |
| Boundary | 302 |
| Browsing | 0 |

- Routing failures: 0
- Component failures: 0
- Fallback attempts: 302
- Fallback successes: 302
- Fallback reasons: `boundary_route` = 302
- Average/p95 router latency: 0.118914 / 0.143500 ms
- Average/p95 route-aware retrieval latency: 234.557276 / 129.219750 ms
- Fixed raw outputs audited: 16,650 recommendations, 0 invalid ASINs,
  0 duplicate ASINs
- Route-aware raw outputs audited: 19,740 recommendations, 0 invalid ASINs,
  0 duplicate ASINs

No Browsing route was selected on this evaluator conversation stream. The
structured extractor/router combination classified turns as Buying, uncertain,
or Boundary; this is an observed limitation rather than a rewritten public-set
rule.

## Provided-baseline comparison and default decision

| Metric | Provided baseline | Route-aware | Absolute change | Relative change |
| --- | ---: | ---: | ---: | ---: |
| Boundary HR@10 | 0.000000 | 0.000000 | 0.000000 | n/a |
| Browsing HR@10 | 0.025000 | 0.000000 | -0.025000 | -100.00% |
| TechnicalScore | 0.106710 | 0.012250 | -0.094460 | -88.52% |

Route-aware mode is not the default. It regressed fixed hybrid HR@10, MRR,
MTTC, Efficiency, TechnicalScore, Buying metrics, latency, startup time, and
memory, while producing no Boundary or Browsing gain. The existing `lexical`
default is unchanged.

The main remaining limitations are the finite Issue 1A structured vocabulary,
no Browsing selections on the public evaluator stream, and expensive uncached
Boundary fallback scoring. Candidate reranking and public-target-specific tuning
were intentionally not added.
