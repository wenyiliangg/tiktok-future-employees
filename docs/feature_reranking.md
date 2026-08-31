# Deterministic feature reranking

`starter.feature_reranker.FeatureReranker` reorders a bounded list of shared
`Candidate` objects. It never retrieves from or scans the catalog during
`rerank`; the `CatalogView` is used only for identifier-keyed metadata lookups.
The evaluator-facing `Agent` generates at most 100 candidates for this stage by
default (`HybridRetrievalConfig.rerank_candidate_count`) and retains its existing
final output limit.

## Features and defaults

All rewards are in `RerankerConfig.feature_weights`. A partial mapping is valid;
an omitted feature then has zero weight.

| Feature | Input and normalization | Kind | Default weight |
| --- | --- | --- | ---: |
| `lexical_score` | Finite scores present on lexical candidates, min-max normalized within the supplied unique pool; a constant observed value becomes 1 | reward | 0.50 |
| `lexical_rank` | `1 / lexical_rank` for a valid one-based rank | reward | 0.75 |
| `dense_score` | Finite scores present on dense candidates, min-max normalized within the pool | reward | 0.50 |
| `dense_rank` | `1 / dense_rank` for a valid one-based rank | reward | 0.75 |
| `fusion_score` | Finite non-zero fusion scores, min-max normalized within the pool | reward | 1.50 |
| `category_match` | 1 for token coverage in normalized category metadata or exact title/feature evidence (which supports broad catalog taxonomies), 0 for a known mismatch | reward | 3.00 |
| `attribute_coverage` | Fraction of active color, style, material, and use-case constraints known to match | reward | 1.50 |
| `price_compatibility` | 1 when any normalized available price is inside the active bounds, 0 for a known miss | reward | 1.00 |
| `color_match` | Active color tokens against normalized color metadata, with catalog text used only when dedicated metadata is absent | reward | 1.00 |
| `style_match` | Active style tokens using the same policy | reward | 1.00 |
| `material_match` | Active material tokens using the same policy | reward | 1.00 |
| `use_case_match` | Active use-case tokens using the same policy | reward | 1.00 |
| `profile_affinity` | Fraction of active profile-sourced constraints known to match | reward | 0.50 |

Retrieval scores and ranks that are absent, non-finite, or invalid receive
`missing_retrieval_value` (default 0). Unknown catalog matches receive
`missing_metadata_value` (default 0). Missing metadata is not a contradiction or
a hard violation. Malformed or absent price data is unknown rather than a miss.

Known active-intent mismatches each subtract `contradiction_penalty` (2 by
default). Known hard mismatches additionally subtract `hard_constraint_penalty`
(1000). The default `hard_constraint_policy="filter"` removes them; setting it
to `"penalize"` retains them with the configured penalty. Explicit exclusions
are detected from their named metadata/evidence field, subtract
`exclusion_penalty` (1000), and are removed by the default
`exclusion_policy="filter"`. Soft mismatches are retained with only the
contradiction penalty.

## Pool, identity, and diagnostics

`parent_asin` is the canonical identity. Duplicate input ASINs become one copied
candidate at the first input position; their best valid source ranks/scores and
source labels are merged deterministically. Caller-owned candidates and the
caller-owned list are not mutated. Empty identifiers are ignored, and output is
always a unique subset of the supplied identifiers truncated to `top_k`.

The returned copies preserve lexical/dense/fusion scores and ranks. Reranking
uses separate `Candidate.rerank_score`, `Candidate.original_position`, and
`Candidate.rerank_diagnostics` fields. Diagnostics include normalized features,
weighted contributions, contradictions, hard and exclusion violations,
duplicate positions, the final score, and any removal reason. Removed-candidate
diagnostics remain inspectable in `FeatureReranker.last_diagnostics` by ASIN.

Ties use final score descending, then original pool position ascending, then
ASIN ascending. The configured tie-breaker name is
`score_original_position_asin`; unsupported alternatives are rejected so a
configuration cannot silently weaken determinism.

If the catalog view is absent or explicitly reports an unavailable backing
store, reranking uses the unique original input order and truncates to `top_k`.
The fallback does not catch ordinary programming/configuration errors. Its
reason is recorded in diagnostics.

Run the focused tests and synthetic benchmark from the participant-kit folder:

```bash
python3 -m unittest tests.test_feature_reranker -v
python3 -m benchmarks.benchmark_feature_reranker --pool-size 100 --runs 1000
```
