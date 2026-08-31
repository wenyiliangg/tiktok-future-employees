# Issue 2B: fixed hybrid fusion results

## Configuration and method

- Branch: `feature/issue-2b-hybrid-fusion`, based on commit `f1a2fbf` with the
  Issue 2B working-tree changes under test.
- Date/environment: 2026-08-29; Apple arm64; macOS 26.5.2; Python 3.12.2;
  NumPy 2.5.2; sentence-transformers 5.7.0; CPU inference.
- Model: `sentence-transformers/all-MiniLM-L6-v2`, requested and resolved at
  revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Cache: schema 1, `catalog-semantic-text-v1`, 50,000 rows, 384-dimensional
  normalized float32 vectors, catalog SHA-256
  `da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`.
  The generated 78,802,526-byte NPZ remains in ignored
  `data/.dense-retrieval/` and is not version controlled.
- Candidate counts: lexical 200, dense 200, final 10.
- Fixed fusion: lexical weight 1.0, dense weight 1.0, `rrf_k=60`; source ranks
  are one-based. No alternative weights were tested.
- Every mode uses `ConversationStateManager.update(...)` and its active
  `SearchQuery`; dense receives that query's deterministic `text` field.
- Evaluations used the unchanged 200-session public set, frozen catalog,
  evaluator scoring, `top_k=10`, and fresh state per session.

## Results

| Mode | HR@10 | MRR | MTTC | Efficiency | TechnicalScore | Buying HR@10 | Browsing HR@10 | Intent Override HR@10 | Boundary HR@10 | Startup (s) | Avg response (ms) | p95 response (ms) | Peak RSS (GB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Official weak-BM25 baseline | 0.125 | 0.068034 | 9.81 | 0.119 | 0.106710 | 0.2375 | 0.0250 | 0.133333 | 0.0000 | n/a | n/a | n/a | n/a |
| Improved lexical | 0.000 | 0.000000 | 11.000 | 0.0000 | 0.000000 | 0.0000 | 0.0000 | 0.000000 | 0.0000 | 15.021 | 31.316 | 80.343 | 1.249 |
| Dense (compatible cache) | 0.025 | 0.008798 | 10.775 | 0.0225 | 0.019639 | 0.0375 | 0.0000 | 0.066667 | 0.0000 | 1.092 | 17.685 | 17.163 | 0.830 |
| Fixed hybrid | 0.020 | 0.007270 | 10.815 | 0.0185 | 0.015881 | 0.0250 | 0.0125 | 0.033333 | 0.0000 | 16.124 | 51.666 | 96.407 | 1.579 |

All three normal runs completed all 200 sessions with zero caught response
exceptions. Average and p95 latency are wall-clock measurements around every
`Agent.respond` call, including state update, retrieval, fusion, and the first
dense query's lazy model loading. Because one model-loading outlier is above the
95th percentile, dense average latency can exceed dense p95. Peak RSS is the
maximum resident set size reported for the whole evaluator process by macOS
`getrusage`/`/usr/bin/time -l`, converted from bytes to decimal GB.

Startup is timed around `Agent(...)` construction. It includes catalog-ID
validation and mode-specific lexical-index and/or dense-cache loading. The
sentence-transformer itself is lazy and therefore excluded from startup and
included in first-response latency. The normal dense and hybrid rows used the
compatible cache. The initial cache-generation run is preserved separately: it
completed 200 sessions with identical dense quality metrics, 356.918 seconds of
agent startup, 15.509 ms average response latency, 20.824 ms p95, and 1.318 GB
peak RSS. Model files were already locally available; startup included catalog
embedding generation but excluded model download.

The improved lexical result is materially below the official baseline because
the active-state extractor intentionally retains only its finite supported slot
vocabulary; unsupported target-specific feature text is not copied into the
active query. Adding feature extraction or clarification strategy is outside
Issue 2B and was not introduced to improve the public score.

Hybrid recovered one browsing hit (`0.0125` versus `0.0000` for both individual
retrievers), but did not beat dense on overall HR@10, MRR, MTTC, Efficiency, or
TechnicalScore. It therefore remains available behind configuration and is not
the default. Lexical remains the safe default because it preserves Issue 1
behavior and has no model/cache/network dependency; dense was the best measured
experimental mode but requires a compatible generated cache and local model.

## Reproduction

From the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 -m evaluator.local_evaluator --retrieval-mode lexical \
  --output docs/results/issue_2b/lexical.json
python3 -m evaluator.local_evaluator --retrieval-mode dense \
  --output docs/results/issue_2b/dense.json
python3 -m evaluator.local_evaluator --retrieval-mode hybrid \
  --lexical-candidates 200 --dense-candidates 200 --final-candidates 10 \
  --lexical-weight 1.0 --dense-weight 1.0 --rrf-k 60 \
  --output docs/results/issue_2b/hybrid.json
```

Raw evaluator outputs are versioned in `docs/results/issue_2b/`. The cold-build
run is `dense_cold_cache_build.json`; it is performance evidence rather than a
fourth retrieval configuration.
