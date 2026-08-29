# System architecture

This document describes the code on the current branch, including which modules
are part of evaluator execution and which are foundations for future
integration.

## Runtime path

The evaluator constructs one `starter.agent.Agent`, then calls `reset` once per
session and `respond` once per turn. `Agent.reset` initializes a clean
`ConversationStateManager` session. `Agent.respond` follows this path:

1. `ConversationStateManager.update(session_id, message, turn)` deterministically
   updates the active `SessionState` and returns a `SearchQuery`.
2. `Agent._retrieve` dispatches that active query to lexical, dense, or hybrid
   retrieval according to `HybridRetrievalConfig.mode`.
3. Results are filtered to identifiers found in the loaded catalog, ranked, and
   truncated to `min(top_k, final_candidate_count)`.
4. The agent returns the identifiers with a fixed message, no clarification
   attribute, and zero model-token usage.

Raw turns are not concatenated into retrieval text. This prevents removed or
overridden preferences from remaining active merely because they occurred
earlier in the transcript.

## Conversation state

`starter.conversation_state` owns the structured state and query contract. Its
supported slots are category, price, style, color, material, and use case.
Extraction uses finite aliases and price patterns; it does not infer arbitrary
attributes with a language model.

Precedence is:

```text
explicit current turn > active conversation history > profile evidence
```

Current/customer evidence is hard by default; hedged and profile-derived
evidence is soft. A new incompatible value replaces the earlier slot value.
Explicit negation removes a positive value and adds an exclusion. Category or
intent changes clear intent-bound style/use-case values while retaining portable
color, material, and price values. Query text is rebuilt in a fixed field order
from active values only.

Every evaluator session calls `reset`, so profile and turn state do not leak
between sessions.

## Lexical retrieval

`starter.lexical_retriever.LexicalRetriever` builds an in-memory SQLite FTS5
index from the catalog. Search uses field-aware weights, with title and category
stronger than attributes/description. The retriever applies explicit hard
filters, configurable soft boosts, exclusions, and price policy. It overfetches
a candidate pool before post-filtering and breaks ties deterministically by
`parent_asin`.

The lexical index is rebuilt at `Agent` construction and is not persisted.
Configuration and supported catalog mappings are documented in
[`lexical_retrieval.md`](lexical_retrieval.md).

## Dense retrieval

`dense_retrieval.DenseRetriever` builds deterministic product text in this
order: title, categories, store/brand, features, description, and allowlisted
details. It encodes catalog text and the active query with
`sentence-transformers/all-MiniLM-L6-v2` at a pinned revision, normalizes
vectors, and ranks by dot product.

The NPZ cache stores float32 embeddings and non-pickle metadata. Loading checks
the complete catalog hash and identifier order, model/revision, schema and text
builder versions, row count, dimensions, dtype, and normalization. An
incompatible or corrupt cache is rebuilt through an atomic replacement.

## Hybrid fusion

`starter.hybrid_retrieval.Candidate` preserves lexical/dense raw scores, ranks,
source labels, and the fusion score. Hybrid retrieval merges exact catalog
identifiers and uses weighted reciprocal-rank fusion because lexical and dense
raw scores are not comparable:

```text
score(d) = lexical_weight / (rrf_k + lexical_rank(d))
         + dense_weight   / (rrf_k + dense_rank(d))
```

A missing source contributes zero. The recorded configuration uses 200 lexical
candidates, 200 dense candidates, ten final candidates, weights 1.0/1.0, and
`rrf_k=60`. Equal scores have a deterministic rank/source/identifier tie break.

## Intent routing

`starter.intent_router.IntentRouter` is implemented but not called by `Agent`.
It deterministically classifies `SessionState` plus `SearchQuery` as Buying,
Browsing, Boundary, or Uncertain. `RoutingDecision` includes confidence,
inspectable reasons, and a configurable policy identifier. Profile-only
evidence cannot produce Buying, and current explicit evidence has precedence.

Because it is not orchestrated, route policy identifiers currently do not select
retrieval modes or scoring behavior.

## Fallback behavior

There are two distinct mechanisms:

1. **Integrated retrieval resilience:** hybrid mode catches dense construction
   or query failures and continues with lexical candidates. An empty dense list
   also leaves the lexical side of RRF intact. Dense-only mode deliberately
   reports its failure.
2. **Standalone Boundary fallback:**
   `starter.fallback_candidates.FallbackCandidateGenerator` can produce valid,
   unique, deterministic candidates from weak active/profile evidence, catalog
   quality priors, exclusions, and diversity caps. It is route-agnostic and is
   not called by `Agent`.

The second mechanism therefore does not improve Boundary evaluator behavior in
the current system.

## Reranking

There is no candidate reranking stage on this branch. Lexical and dense modes
return their source order; hybrid returns fixed-RRF order. Any Issue 4C reranker
is future work and must be documented only after its code and evaluation
evidence are merged.

## Clarification strategy

`starter.ambiguity_analysis.AmbiguityAnalyzer` is a completed analysis
foundation, not a user-facing question system. Given candidates, catalog
metadata, and known state, it excludes known attributes and scores each usable
missing attribute with a coverage-adjusted Gini split:

```text
expected_reduction = metadata_coverage * (1 - sum(value_share ** 2))
```

It returns at most one deterministic `ClarificationOpportunity`. Coverage,
minimum pool size, distinct values, dominant share, reduction threshold, price
buckets, and tie priority are configurable. It rejects sparse metadata and
attributes that barely divide the pool.

The analyzer is not called by `Agent`; question wording, asked-question history,
and evaluator integration do not exist. The current response always returns
`ask_attribute: null`.

## Configuration boundaries

- `HybridRetrievalConfig`: mode, source/final pool sizes, weights, and RRF
  constant.
- Lexical configuration: field weights, boosts, filtering and pool behavior.
- Dense configuration: model/revision, cache path, encoding batch/device.
- `RouterConfig`: thresholds, evidence weights, conflict margin, phrase lists,
  and policy identifiers.
- `FallbackConfig`: evidence/source weights, quality priors, exclusion penalty,
  diversity dimensions/caps/penalties, and price buckets.
- `AmbiguityConfig`: eligible attributes, metadata/pool thresholds, reduction
  threshold, dominance limit, price buckets, and deterministic priority.

Changing standalone module configuration has no evaluator effect until that
module is connected to `starter.agent.Agent`.
