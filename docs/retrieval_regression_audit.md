# Retrieval regression forensic audit

Date: 2026-08-29
Audited branch: `feature/issue-3c-route-aware-retrieval` at `8fb7f27`
Public set: 200 sessions; no labels or official evaluator logic were modified.

## Executive finding

The principal regression occurs before advanced retrieval can help. The common
2B/3C/4B state extractor replaces the raw current message with a finite slot
summary. Unsupported target-specific phrases disappear, and the field-aware
lexical retriever then treats recognized aliases as hard filters against sparse
or differently normalized dedicated metadata. Equal-weight RRF compounds the
loss by rewarding candidates present in both lists enough to displace lexical
Top-10 evidence.

The premise that fixed hybrid retains four of the official BM25 hits is false.
The official BM25 hit set and the 3C fixed-hybrid hit set are disjoint:

- official BM25: 25 hits;
- 3C fixed hybrid: `public_0136`, `public_0165`, `public_0177`, `public_0194`;
- overlap: zero.

Thus fixed hybrid loses all 25 official successes and gains four different
sessions. The “21 of 25” figure is only `25 - 4`; it is not a per-session
retention calculation.

## Benchmark identity and configuration

| Artifact | Exact agent and ranking path |
| --- | --- |
| Official weak BM25 | Starter `Agent` from `166bae7`; raw current message; SQLite FTS5 fields `title/categories/features/details/store/description`; weights `6/4/2.5/2.5/1.5/1`; direct Top-10; no state, dense retrieval, filtering, reranking, fallback, or clarification. |
| Issue 2B lexical | `Agent` at `36a9974`; `ConversationStateManager` -> field-aware `LexicalRetriever`; 200 candidates -> direct Top-10. |
| Issue 2B dense | Same 2B `Agent`; extracted `SearchQuery.text`; MiniLM exact cosine Top-200 -> direct Top-10. |
| Issue 2B fixed hybrid | Same 2B `Agent`; 200 lexical + 200 dense; equal RRF (`k=60`) -> direct Top-10. |
| Issue 3C “fixed hybrid” | `Agent` on 3C after merge of Issue 4A; 200 + 200; equal RRF -> Top-100 -> deterministic feature reranker with hard-filter policy -> Top-10. This is not the common 2B fixed pipeline. |
| Issue 3C route-aware | 3C `Agent`; route-specific pools and weights; route filter; route RRF -> Top-10; no feature reranker. Boundary fallback was attempted unconditionally under the committed v1 policy. |
| Issue 4B without semantic | 4B `Agent`; true 2B fixed hybrid; equal RRF directly to Top-10. It has router/fallback components in the branch but does not invoke route-aware retrieval. |
| Issue 4B with semantic | Same 4B agent; equal RRF Top-50 -> `cross-encoder/ms-marco-MiniLM-L6-v2` -> Top-10. Feature reranking is absent. |

All 2B, 3C, and 4B variants use the same finite state query and field-aware
lexical implementation. The dense cache is the same compatible 50,000-row,
384-dimensional, normalized float32 MiniLM cache at
`data/.dense-retrieval/catalog-minilm.npz`, with model revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

### Detailed behavior differences

- Query construction: official BM25 uses each raw customer message. All later
  variants use `SearchQuery.text`, which contains only active category, style,
  use-case, color, material, and price slots.
- Lexical fields: official uses six broad catalog fields with the weights above.
  Field-aware lexical uses `title=5`, `category=4`, `features=3`,
  `color/material/style/use_case=3`, `attributes=2.5`, `description=1`, and
  `brand=1.5`, plus structured boosts/filters.
- Hard filtering: field-aware lexical filters hard constraints internally in all
  modes. 3C Buying filters the merged union a second time. Issue 4A also filters
  hard violations during feature reranking.
- Dense generation: exact cosine over all 50,000 cached embeddings; 200
  candidates normally, 400 for 3C Browsing. Before the patch it received the
  extracted slot text, not the raw current message.
- Fusion: 2B/4B use weights `1/1`, `k=60`; 3C Buying uses `2/1`, Browsing
  `0.75/1.5`, Boundary `0.5/0.5` plus fallback `1.5`, and uncertain `1/1`.
- Route selection: 3C observed 1,156 Buying, 518 uncertain, 302 Boundary, and
  zero Browsing turns. The router inspected extracted text, so broad raw cues
  could disappear before routing.
- Fallback: 3C invoked fallback on all 302 Boundary turns, with a reset-scoped
  cache keyed by active query/constraints. This introduced full-catalog scoring
  outliers and no Boundary hit.
- Clarification: every benchmarked agent returns `ask_attribute=None`.
  `ambiguity_analysis.py` is not wired into `Agent`, so benchmark conversation
  streams are identical for a given session. No loss is attributable to a
  clarification-policy difference.
- Tie-breaking: field-aware lexical uses score then ASIN; dense uses stable
  catalog order; fixed and route RRF use fusion score, source ranks, then ASIN;
  feature reranking uses score, original position, then ASIN; semantic reranking
  uses score then original position. The official starter relies on FTS row
  order for equal BM25 scores.

## Metric verification

Every stored score was recomputed from its raw `sessions` records using the
official miss-turn value 11 and the published formula.

| Raw artifact | Hits | HR@10 | MRR | MTTC | Efficiency | TechnicalScore |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2B lexical | 0 | 0 | 0 | 11 | 0 | 0 |
| 2B dense | 5 | 0.025 | 0.008798 | 10.775 | 0.0225 | 0.019639 |
| 2B hybrid | 4 | 0.020 | 0.007270 | 10.815 | 0.0185 | 0.015881 |
| 3C fixed hybrid + feature | 4 | 0.020 | 0.007500 | 10.815 | 0.0185 | 0.015950 |
| 3C route-aware | 3 | 0.015 | 0.006833 | 10.865 | 0.0135 | 0.012250 |
| 4B hybrid without semantic | 4 | 0.020 | 0.007270 | 10.815 | 0.0185 | 0.015881 |
| 4B hybrid with semantic | 4 | 0.020 | 0.003964 | 10.815 | 0.0185 | 0.014889 |

The official aggregate also reconstructs exactly from the discovered 25 hits:
the reciprocal-rank sum is `13.606746031746031`, giving MRR
`0.06803373015873015`; hit turns sum to 37, giving MTTC `9.81`; the rounded
TechnicalScore is `0.106710`.

## Evidence from the 25 official hits

At each official hit turn:

- raw tokens were dropped from the structured query in 25/25 sessions;
- 18/25 targets fail at least one current lexical hard constraint;
- 16/25 targets are absent from both current Top-200 lists and therefore from
  the fixed hybrid union;
- 9/25 targets enter the union but rank below fused Top-10;
- 0/25 are in fused Top-10, so none is lost solely after feature or semantic
  reranking at that turn;
- route-aware filtering additionally removes `public_0156` from an existing
  union; feature reranking records hard-filter removals for `public_0156` and
  `public_0160`, but both were already below fused Top-10;
- there are no interaction/clarification differences.

Representative sessions:

- `public_0090` (Buying): official rank 1. State extraction produces an empty
  query, so current lexical and dense retrieve no target. The router selects
  Boundary and invokes fallback despite strong raw Buying evidence; fallback
  does not recover the target.
- `public_0156` (Buying): official rank 1. The raw field-aware FTS candidate is
  rank 438, but category/use-case hard checks reject it. Dense retrieves it at
  rank 25; equal RRF leaves it at 25. The 3C Buying post-filter then removes it
  for category mismatch.
- `public_0046` (Intent Override): official rank 8 on turn 4. Current lexical
  retrieves it at 39, dense misses it, and RRF demotes it to 90.
- `public_0142` (Intent Override): official rank 1. Current lexical/dense ranks
  are 31/88; fusion rank is 31 and route-aware fusion rank is 30.
- `public_0193` (Buying): official rank 2. Equal-weight fusion of the official
  lexical list with raw-text dense moves the target to rank 50; 48 products
  appearing in both lists rank above it.

## Equal-weight RRF audit

For a lexical-only target at rank `r`, equal-weight RRF gives
`1 / (60 + r)`. A product present in both lists at ranks `a` and `b` receives
`1 / (60 + a) + 1 / (60 + b)`. The double contribution is not a small tie
breaker; it can dominate strong single-source evidence. With equal source ranks
`q`, a dual-source product outranks lexical rank 2 whenever `q < 64`, and
outranks lexical rank 10 whenever `q < 80`.

The direct public-session test confirms eight official lexical Top-10 targets
are displaced beyond Top-10 when raw-text dense candidates are added with equal
RRF: `public_0010`, `0046`, `0067`, `0081`, `0129`, `0143`, `0148`, and
`0193`.

## Recovery patch

The patch is intentionally conservative:

- `SessionState.raw_current_turn_text` retains the exact current message without
  changing the shared `SearchQuery` contract.
- `BM25AnchorRetriever` reproduces official fields, weights, tokenization, and
  raw-turn retrieval.
- exact `bm25` was the active default throughout recovery development, and the
  explicit `anchored` mode protects official BM25 order while allowing only
  vacancy backfill.
- any structured constraints used for backfill are copied as soft boosts, so
  uncertain extraction cannot hard-filter the target.
- feature reranking and Boundary fallback now require explicit configuration;
  the Boundary policy is no longer unconditional.
- route-aware remains an explicit mode. The semantic component exists only on
  the 4B branch and remains disabled by default there.
- sorting and identity handling remain deterministic.

The selected-session recovery verifier reproduced all 25 official hits with
the same first-hit turn and rank. It reported zero mismatches. Its HR@10 `1.0`,
MRR `0.54427`, and MTTC `1.48` are subset diagnostics and must not be reported
as official 200-session metrics.

## Runtime complexity

Let `N=50,000`, embedding dimension `d=384`, source pool sizes `L` and `D`,
union size `U`, rerank pool `R`, and active constraint count `c`.

- BM25 index construction is `O(N)` documents plus tokenization/index storage;
  each query visits matching postings and maintains a small Top-K result set.
- exact dense retrieval is `O(Nd)` time and `O(Nd)` matrix memory per process;
  the raw float32 matrix is about 76.8 MB before container/model overhead.
- merge is `O(L+D)`; RRF sorting is `O(U log U)` and `O(U)` memory.
- hard filtering is `O(Uc)` plus metadata lookups.
- deterministic feature reranking is `O(Rc + R log R)`.
- Boundary fallback performs `O(Nc)` catalog scoring on a cache miss.
- semantic reranking is bounded by `R` cross-encoder inferences plus
  `O(R log R)` sorting and model memory.
- anchored mode normally returns a full BM25 Top-10 and therefore performs no
  dense inference, union sort, fallback, or reranking. Backfill work occurs only
  when the anchor returns fewer than the requested count.

The forensic diagnostic run over the 25 selected sessions took about 120
seconds and wrote a 1.0 MB diagnostic artifact. Recovery verification took
about 12 seconds.

## Recommended architecture and issue order

1. Freeze a per-session baseline trace and make anchor-hit retention a merge
   gate.
2. Preserve raw turn text and keep structured state as a sidecar, never as a
   lossy replacement query.
3. Calibrate constraint confidence; only explicit, metadata-verifiable
   exclusions may filter. Everything else begins as a boost.
4. Establish protected lexical anchoring and deterministic vacancy backfill.
5. Reintroduce dense retrieval as residual/backfill evidence; test protected
   prefixes or strongly lexical-biased fusion before any full run.
6. Re-evaluate routing using raw cues, but do not let routing alter the anchor
   until it passes the retention gate.
7. Add fallback only behind an insufficiency/coverage trigger and latency gate.
8. Evaluate deterministic feature reranking on retained pools.
9. Evaluate semantic reranking last, only after retrieval is stable and only if
   it improves TechnicalScore without HR regression.
10. Add clarification policy after retrieval comparisons share the same
    interaction stream; otherwise changes are not attributable.

Dependencies should therefore be revised from parallel `3C || 4B` development
to `anchor/raw-text gate -> soft constraints -> dense backfill -> routing ->
fallback -> feature reranking -> semantic reranking -> clarification`.

## Commands

From the repository root:

```bash
# Focused/unit regression (150 tests in the audited environment)
python3 -m unittest discover -s tests -v

# Rebuild detailed diagnostics only for the 25 official anchor hits
python3 -m benchmarks.diagnose_retrieval_regression \
  --official-bm25-hits \
  --output diagnostics/retrieval_regression/official_bm25_hits.json

# Verify the recovery path on those selected sessions only
python3 -m benchmarks.verify_anchor_recovery \
  --selection diagnostics/retrieval_regression/official_bm25_hits.json \
  --output diagnostics/retrieval_regression/recovery_verification.json
```

Only after the fast gate passes, run the later full evaluation explicitly:

```bash
python3 -m evaluator.local_evaluator \
  --retrieval-mode anchored \
  --output diagnostics/retrieval_regression/anchored_full_200.json
```

After this audit, the separate contextual policy-selection experiment passed
and the full evaluator was run for `contextual.browsing-dense.v1`. See
`docs/contextual_retrieval_recovery.md`; the audit findings above remain the
frozen explanation of the 2B/3C/4B regression.
