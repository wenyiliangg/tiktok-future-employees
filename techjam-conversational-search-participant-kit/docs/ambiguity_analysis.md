# Candidate-pool ambiguity analysis

`starter.ambiguity_analysis` identifies the single missing attribute that would
most usefully narrow a candidate pool. It consumes Issue 2B candidates, catalog
metadata keyed by `parent_asin`, and Issue 1A session state. It does not retrieve,
rerank, route, or produce user-facing questions.

## Selection measure

For every unknown attribute, the analyzer records value counts, metadata
coverage, dominant-value share, normalized entropy, and expected fractional pool
reduction. The reduction measure is:

```text
coverage × (1 − Σ value_share²)
```

This is a coverage-adjusted Gini split. A complete 50/50 division scores `0.5`;
a 90/10 split scores `0.18`; incomplete metadata reduces the score further.
Missing values are conservatively retained because they cannot be filtered with
confidence.

An attribute is eligible only when its candidate count, usable metadata count,
coverage, distinct values, dominant share, and expected reduction satisfy
`AmbiguityConfig`. The highest-reduction attribute wins. Ties use metadata
coverage and then the configured attribute priority, making selection stable.

Price values use configurable fixed buckets. Catalog features are evaluated as
deterministic present/absent binary splits. Attributes already present in
`SessionState` are excluded before scoring.

