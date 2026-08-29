# Optional local semantic reranker

Issue 4B adds a bounded local cross-encoder post-processor. It is disabled by
default and does not change intent routing, candidate generation, clarification,
or the agent response schema.

## Ranking boundary

`Agent` first obtains the normal ordered shared candidate pool. When the
experiment is enabled, it keeps enough base-ranked candidates to cover
`semantic_candidates`, passes only that prefix to `SemanticReranker`, and then
returns the requested final count. Candidates below the scored prefix retain
their exact base order.

The reranker receives pairs of:

```text
(active SearchQuery.text, deterministic catalog product text)
```

It never searches the catalog. It only reorders existing `Candidate` objects.
The agent verifies exact object and ASIN membership after reranking and rejects
any result that drops, duplicates, replaces, or introduces an item. Final ASINs
therefore remain a subset of the frozen catalog and the shared retrieval pool.

Product text reuses `ProductTextBuilder` and its
`catalog-semantic-text-v1` format: fixed labelled sections, sorted mapping keys,
normalized whitespace, catalog-order lists, a fixed allowlist of searchable
attributes, and no mutation of catalog rows.

## Model loading and failure behavior

The production adapter uses `sentence_transformers.CrossEncoder` on CPU. The
model name, optional revision, scored candidate count, batch size, and maximum
token length are configurable. All query/product pairs for one call are passed
to one `predict` call with the configured batch size.

Models are lazy-loaded and cached for the life of the process by model name,
revision, device, and maximum length. Equivalent reranker instances share the
same model and serialize access to that model's inference call. There is no
port, hosted inference service, external LLM call, or fake production fallback.
The model library may download weights on first use when they are not already
cached; pre-cache the selected model and set `HF_HUB_OFFLINE=1` for a strictly
offline run.

Any missing product text, import/load exception, inference exception, wrong
score count or shape, boolean score, NaN, or infinite score returns the original
candidate objects in their original order. Only validated scores are written to
`Candidate.semantic_score`. The first failure is logged and every failed call is
counted.

## Configuration

The programmatic defaults are deliberately non-operative:

```python
SemanticRerankerConfig(
    enabled=False,
    model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
    model_revision=None,
    candidate_count=50,
    batch_size=16,
    max_length=256,
    device="cpu",
)
```

Evaluator flags mirror those fields:

```text
--semantic-reranker
--semantic-model
--semantic-model-revision
--semantic-candidates
--semantic-batch-size
--semantic-max-length
```

Omitting `--semantic-reranker` leaves ranking and model/product-text loading
unchanged.

## Measurements

`result["semantic_reranker"]` reports:

- configured and resolved model identity;
- model parameter bytes;
- first model-access/cold-start time and whether the process cache was hit;
- peak RSS delta observed while loading and peak process RSS observed;
- query count, total and maximum candidates scored;
- average and p95 query reranking latency, excluding separately reported model
  loading;
- failure count and product-text format version.

The evaluator's existing `performance` and `evaluation_diagnostics` blocks
continue to report overall agent startup, average/p95 response latency, response
failures, and whole-process peak RSS.

## Official A/B benchmark

Run each variant in an isolated process so cold start and peak RSS do not leak
between variants:

```bash
python3 -m benchmarks.benchmark_semantic_reranker \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --dense-cache data/.dense-retrieval/catalog-minilm.npz \
  --semantic-model cross-encoder/ms-marco-MiniLM-L6-v2 \
  --semantic-candidates 50 --semantic-batch-size 16
```

The benchmark writes the complete official evaluator results for hybrid without
semantic reranking and hybrid with semantic reranking, plus `comparison.json`.
The comparison contains HR@10, MRR, MTTC, Efficiency, TechnicalScore, scenario
metrics, model size, cold start, reranking average/p95, overall response
average/p95, peak RSS, failures, and all deltas.

Its conservative, configurable default decision rule requires at least `0.005`
absolute TechnicalScore gain, no HR@10 regression, zero reranker failures, at
most 250 ms average reranking latency, and at most 500 MB additional peak RSS.
Even when every threshold passes, the output recommendation is only
`retain_as_optional_experiment`; `default_enabled` is always false.

See [`issue_4b_results.md`](issue_4b_results.md) for the current evidence and
recommendation.
