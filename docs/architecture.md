# System architecture

This document describes the code on the current branch, including which modules
are part of evaluator execution and which are foundations for future
integration.

## Runtime path

The evaluator constructs one `starter.agent.Agent`, then calls `reset` once per
session and `respond` once per turn. `Agent.reset` initializes a clean
`ConversationStateManager` session. `Agent.respond` follows this path:

1. Explicit negative feedback marks the prior recommendations as known-negative;
   an intent-override cue clears that set.
2. `ConversationStateManager.update(session_id, message, turn)` preserves the
   raw turn, updates active `SessionState`, and returns a `SearchQuery`.
3. `Agent._retrieve` dispatches according to `HybridRetrievalConfig.mode`.
   The contextual default retrieves exact raw-turn BM25 candidates, removes
   known negatives, and protects the first eight remaining BM25 positions.
4. `IntentRouter` uses active state plus preserved raw intent. Only Browsing
   permits dense evidence to compete for the final two positions. Ranking and
   identity tie-breaks are deterministic.
5. The selected declarative policy analyzes at most 50 already-ranked candidates
   and may attach one clarification only on the runtime-observable Browsing
   route. Recommendation identities/order and zero model-token usage are
   preserved on the question turn.

Raw turns are retained separately from structured state. They are never
concatenated blindly; an active raw intent is replaced on intent override.

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

A missing source contributes zero. The recorded pre-reranker configuration uses
200 lexical candidates, 200 dense candidates, ten final candidates, weights
1.0/1.0, and `rrf_k=60`. Equal scores have a deterministic
rank/source/identifier tie break.

## Intent routing

`starter.intent_router.IntentRouter` is called by contextual retrieval to decide
whether dense evidence is allowed, and by the explicit route-aware mode.
It deterministically classifies `SessionState` plus `SearchQuery` as Buying,
Browsing, Boundary, or Uncertain. `RoutingDecision` includes confidence,
inspectable reasons, and a configurable policy identifier. Profile-only
evidence cannot produce Buying, and current explicit evidence has precedence.

The promoted policy uses only the Buying/Browsing decision to activate dense
retrieval. Full route-specific filtering/fusion remains isolated behind the
explicit `route-aware` mode.

## Fallback behavior

There are two distinct mechanisms:

1. **Integrated retrieval resilience:** hybrid mode catches dense construction
   or query failures and continues with lexical candidates. An empty dense list
   also leaves the lexical side of RRF intact. Dense-only mode deliberately
   reports its failure.
2. **Explicit Boundary fallback:**
   `starter.fallback_candidates.FallbackCandidateGenerator` can produce valid,
   unique, deterministic candidates from weak active/profile evidence, catalog
   quality priors, exclusions, and diversity caps. `Agent` calls it only in
   explicit route-aware mode with `enable_boundary_fallback=True`.

The second mechanism is disabled in the promoted contextual configuration.

## Reranking

`starter.feature_reranker.FeatureReranker` is available after lexical, dense,
and hybrid retrieval only when `enable_feature_reranker=True`. It is disabled
by default and never touches the contextual BM25 prefix. Its configurable features include normalized source scores and
ranks, fusion score, category match, active-attribute coverage, price
compatibility, individual color/style/material/use-case matches, and soft
profile affinity.

Known mismatches receive contradiction penalties. Known hard-constraint and
explicit-exclusion violations are filtered by default; missing metadata is
treated as unknown rather than a contradiction. Duplicates are merged by
`parent_asin`, results remain a subset of the retrieved pool, caller objects are
not mutated, and ties use score, original position, then identifier. If the
catalog view is unavailable, the reranker preserves the unique input order and
records the fallback reason in diagnostics.

The implementation has unit and synthetic performance coverage, but no
post-integration public-set result artifact has been recorded yet. The Issue 2B
quality/runtime tables therefore describe the earlier pre-reranker agent.

## Clarification strategy

`starter.ambiguity_analysis.AmbiguityAnalyzer` is integrated behind the selected
clarification policy. Given candidates, catalog metadata, and known state, it
excludes known attributes and scores each usable missing attribute with a
coverage-adjusted Gini split:

```text
expected_reduction = metadata_coverage * (1 - sum(value_share ** 2))
```

It returns at most one deterministic `ClarificationOpportunity`. Coverage,
minimum pool size, distinct values, dominant share, reduction threshold, price
buckets, and tie priority are configurable. It rejects sparse metadata and
attributes that barely divide the pool.

`ClarificationController` enforces at most one question per session, prevents
turn-10 and repeated questions, and tracks explicit answers or declines.
`config/clarification_policies.json` selects
`clarification.browsing-only.v1`; `contextual.browsing-dense.v1` remains the
clarification-disabled rollback. Clarification-only failures fall back to the
unchanged recommendation response.

## Configuration boundaries

- `HybridRetrievalConfig`: explicit mode, opt-in risky components,
  source/reranking/final pool sizes, weights, and RRF constant.
- `ContextualRetrievalPolicy`: protected BM25 prefix, candidate depth,
  state/dense weights, eligible dense routes, and deterministic RRF constant.
- Lexical configuration: field weights, boosts, filtering and pool behavior.
- Dense configuration: model/revision, cache path, encoding batch/device.
- `RerankerConfig`: feature weights, mismatch/filter penalties and policies,
  missing-value handling, and deterministic tie behavior.
- `RouterConfig`: thresholds, evidence weights, conflict margin, phrase lists,
  and policy identifiers.
- `FallbackConfig`: evidence/source weights, quality priors, exclusion penalty,
  diversity dimensions/caps/penalties, and price buckets.
- `AmbiguityConfig`: eligible attributes, metadata/pool thresholds, reduction
  threshold, dominance limit, price buckets, and deterministic priority.
- `config/clarification_policies.json`: named eligibility/controller values,
  fixed evaluator seed, fingerprints, selected default, and rollback policy.

Route-aware fallback and reranking continue to require explicit flags.
