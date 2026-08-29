# Route-aware hybrid retrieval

Issue 3C adds `route-aware` as a fourth retrieval mode. `lexical`, `dense`, and
fixed `hybrid` retain their Issue 2B behavior. The default remains `lexical`;
route-aware mode is not promoted without public evaluation evidence that it
does not underperform fixed hybrid on the primary metrics.

## Pipeline

On each turn, the evaluator-facing agent:

1. updates Issue 1A's structured active state and `SearchQuery`;
2. asks the deterministic Issue 3A router for Buying, Browsing, Boundary, or
   uncertain intent;
3. runs the configured lexical and dense candidate generators independently;
4. merges only exact, catalog-valid `parent_asin` identities;
5. applies the selected policy's hard-constraint and exclusion filters;
6. performs weighted reciprocal-rank fusion with a stable `parent_asin`
   tie-breaker; and
7. for justified Boundary routes, merges Issue 3B fallback candidates before
   fusion.

This mode performs no feature reranking, cross-encoding, LLM ranking, or
catalog-wide post-generation reranking.

## Default route policies

All values live in `HybridRetrievalConfig.route_policies` as validated
`RouteRetrievalPolicy` objects.

| Route | Lexical | Dense | Fallback | Pools (lex/dense/fallback) | Hard filters | Exclusions |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Buying | 2.0 | 1.0 | 0.0 | 250 / 200 / 0 | yes | yes |
| Browsing | 0.75 | 1.5 | 0.0 | 250 / 400 / 0 | no | yes |
| Boundary | 0.5 | 0.5 | 1.5 | 100 / 200 / 50 | no | yes |
| Uncertain | 1.0 | 1.0 | 0.0 | 200 / 200 / 0 | no | yes |

Every policy uses `rrf_k=60`, returns at most 10 candidates, and may be replaced
through normal dataclass configuration. Buying requires positive evidence for
active hard category, attribute, and price constraints. Browsing and uncertain
intent do not turn missing optional metadata into a hard rejection. Explicit
known exclusions remain active in every route. Boundary fallback uses only the
active query, safe profile fields, and removed-constraint bookkeeping; it does
not create hard constraints.

Malformed or unknown router output uses the uncertain policy. Lexical, dense,
and fallback calls are independent failure boundaries, so another safe source
can still provide candidates. Invalid candidates are skipped, duplicate source
records preserve the best original source rank, and final candidates retain raw
source scores/ranks, per-source RRF contributions, total fusion score, fallback
rank/score, source labels, and filter diagnostics.

## Diagnostics and evaluation

`Agent.diagnostics_snapshot()` reports route counts, routing failures, component
failures, retrievers attempted/successful, fallback attempts/reasons/success,
candidate counts before and after deduplication/filtering, and routing/retrieval
latency. `Agent.last_candidates(session_id)` exposes copied candidate provenance
without changing the official response payload. `reset()` clears the named
session's active state, candidates, and turn diagnostics.

The diagnostic runner calls the unchanged `evaluator.local_evaluator.evaluate`
function and adds agent-side instrumentation to the result file:

```bash
python3 -m benchmarks.evaluate_route_aware --retrieval-mode hybrid \
  --output docs/results/issue_3c/fixed_hybrid.json
python3 -m benchmarks.evaluate_route_aware --retrieval-mode route-aware \
  --output docs/results/issue_3c/route_aware.json
```
