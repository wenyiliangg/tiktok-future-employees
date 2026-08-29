# Issue 4B: local semantic reranker experiment

## Configuration and method

The production A/B benchmark completed on 2026-08-29:

- Environment: Apple arm64, macOS 15.6, 10 logical CPUs, Python 3.12.13,
  NumPy 2.5.2, sentence-transformers 5.7.0, Transformers 5.16.1, Torch
  2.13.0, CPU inference.
- Catalog: 50,000 unique valid ASINs; SHA-256
  `da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`.
- Dense retrieval: `sentence-transformers/all-MiniLM-L6-v2` at pinned revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; compatible 78,802,526-byte
  cache; lexical/dense pools 200 each; equal-weight RRF with `k=60`.
- Reranker: `cross-encoder/ms-marco-MiniLM-L6-v2`, resolved revision
  `233902d25c440f23af6f7d6e94d2946bac0bee0a`, CPU, Top-50, batch size 16,
  maximum length 256, deterministic `catalog-semantic-text-v1` product text.
- Evaluation: unchanged 200-session public set, 1,967 responses per variant,
  isolated processes, compatible dense cache for both measured variants.
- The semantic variant invoked the model 1,665 times and scored exactly 83,250
  candidates, never more than 50 in one query. Blank/empty-pool responses did
  not invoke it.

The first execution built the dense cache in the baseline process. It was
excluded from the comparison, and both variants were repeated against the same
compatible cache. The tables and committed JSON below are from that second,
comparable run.

## Official results

| Measure | Hybrid without semantic | Hybrid + semantic | Delta |
| --- | ---: | ---: | ---: |
| HR@10 | 0.020000 | 0.020000 | 0.000000 |
| MRR | 0.007270 | 0.003964 | -0.003306 (-45.5%) |
| MTTC | 10.815 | 10.815 | 0.000 |
| Efficiency | 0.018500 | 0.018500 | 0.000000 |
| TechnicalScore | 0.015881 | 0.014889 | -0.000992 (-6.25%) |
| Agent startup | 9.167 s | 9.923 s | +0.756 s |
| Average response latency | 29.513 ms | 446.216 ms | +416.704 ms (15.1x total) |
| p95 response latency | 56.837 ms | 667.520 ms | +610.683 ms (11.7x total) |
| Average reranking latency | n/a | 478.104 ms | +478.104 ms |
| p95 reranking latency | n/a | 614.569 ms | +614.569 ms |
| Cross-encoder cold start | n/a | 4.089 s | +4.089 s |
| Cross-encoder parameter size | n/a | 90,854,404 bytes | +90.854 MB |
| Peak process RSS | 1,862,041,600 bytes | 2,054,340,608 bytes | +192,299,008 bytes |
| Response exceptions | 0 | 0 | 0 |
| Reranker failures | 0 | 0 | 0 |

### Scenario metrics

| Scenario | Baseline HR@10 | Semantic HR@10 | Baseline MRR | Semantic MRR |
| --- | ---: | ---: | ---: | ---: |
| Buying | 0.025000 | 0.037500 | 0.015000 | 0.006786 |
| Browsing | 0.012500 | 0.000000 | 0.001389 | 0.000000 |
| Intent override | 0.033333 | 0.033333 | 0.004762 | 0.008333 |
| Boundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

Semantic reranking trades the sole baseline browsing hit for one additional
buying hit, leaving overall HR@10 unchanged. It moves the successful buying
targets lower on average, causing the large MRR regression. Intent-override MRR
improves, but not enough to offset the buying and browsing losses.

Full outputs:

- [`results/issue_4b/hybrid_without_semantic.json`](results/issue_4b/hybrid_without_semantic.json)
- [`results/issue_4b/hybrid_with_semantic.json`](results/issue_4b/hybrid_with_semantic.json)
- [`results/issue_4b/comparison.json`](results/issue_4b/comparison.json)

## Dependency caveat

This is an official-set comparison of the currently integrated fixed hybrid
ordering. The intent router remains standalone in this branch, so this run must
not be mislabeled as the dependency-complete route-aware comparison. The
reranker is a route-agnostic shared-pool post-processor and is ready to rerun
unchanged after route-aware ordering lands. Issue 4A's deterministic feature
reranker is not present here, so the optional three-way comparison remains
deferred.

## Recommendation

Do not retain this Top-50 cross-encoder in the active ranking pipeline and do
not enable it by default. It provides no HR@10 gain, materially worsens MRR and
TechnicalScore, exceeds the 250 ms average reranking budget, adds about 192 MB
peak RSS, and increases end-to-end average response latency by more than 15x.
The zero failure count confirms that this is a quality/cost rejection rather
than an availability problem.

The component may remain in the repository as disabled experimental plumbing
for a smaller candidate-count/model trial or the later route-aware comparison,
but the present configuration does not justify its runtime or memory cost.

Reproduce with:

```bash
python3 -m pip install -r requirements.txt
python3 -m benchmarks.benchmark_semantic_reranker \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --dense-cache data/.dense-retrieval/catalog-minilm.npz \
  --semantic-model cross-encoder/ms-marco-MiniLM-L6-v2 \
  --semantic-candidates 50 --semantic-batch-size 16
```

The benchmark validates/builds the dense cache before starting either measured
subprocess, preventing cache generation from biasing one variant's startup or
peak-memory result.

Implementation verification also passes all 124 tests (14 semantic-specific),
Ruff, MyPy for changed modules, compilation, and whitespace validation.
