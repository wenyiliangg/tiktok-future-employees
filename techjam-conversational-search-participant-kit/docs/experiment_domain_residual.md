# Domain-adapted residual retrieval experiment

This branch tests a second retrieval path while keeping the current winning
architecture intact:

1. Preserve exact raw-turn BM25 as the protected lexical anchor.
2. Encode only the residual candidate pool with a dense model.
3. Use a short memory of informative user turns so a rejection such as
   `no, not that` cannot erase the product intent.
4. Ask the existing clarification question only on the Browsing route.

The new retrieval policy is `contextual.domain-residual.v1`. The new evaluation
policy is `clarification.domain-residual.v1`.

## Why this is a plausible improvement

The current audit found that the main losses occur before ranking: structured
query extraction drops useful raw phrases, hard lexical constraints reject
aliases, and equal-weight fusion can displace a strong BM25 result. The new
policy leaves those safeguards unchanged and changes only the semantic residual
signal. It also uses Amazon-review-domain embeddings rather than a general
MiniLM encoder.

The supported domain model is `hyp1231/blair-roberta-base`. Its CLS embedding is
computed lazily through Transformers and catalog embeddings remain cached. Use
an immutable Hugging Face revision for a reproducible comparison.

## Evaluation command

Run this from the repository root after the official catalog and dense-cache
artifacts are available:

```bash
python -m evaluator.local_evaluator \
  --retrieval-mode contextual \
  --contextual-policy contextual.domain-residual.v1 \
  --clarification-policy clarification.domain-residual.v1 \
  --dense-model hyp1231/blair-roberta-base \
  --dense-model-kind transformers-cls \
  --dense-model-revision <PINNED_HF_COMMIT> \
  --dense-cache data/.dense-retrieval/catalog-blair-base.npz \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --output results/domain_residual.json
```

Use a separate cache path for this model. The cache metadata also records the
model and revision, but a separate path makes accidental cross-model reuse
obvious. The full official catalog is intentionally not included in this
repository, so the public participant kit cannot produce the official score by
itself.

## What stays enabled

- Exact raw-turn BM25 and the protected BM25 anchor.
- Negative-feedback rotation and override handling.
- Browsing-only clarification with one-question-per-session limits.
- Dense retrieval only as a residual signal on Browsing turns.
- Explicit evaluator modes for ablation and rollback.

## What stays disabled by default

- Global field-aware lexical filtering: the audit showed sparse metadata can
  turn aliases into false hard exclusions.
- Equal-weight RRF as the production default: it can displace a strong lexical
  hit even when the dense signal is weak.
- Route-aware fallback and the feature reranker: keep them as explicit
  experiments until they beat the protected anchor on the same split.

## Promotion rule

Promote this branch only if it improves the official TechnicalScore and does
not materially worsen hit rate, mean reciprocal rank, time-to-correct, or
latency. Compare against both stored controls:

| Configuration | TechnicalScore |
|---|---:|
| BM25 baseline | 0.106710 |
| Current stable contextual champion | 0.115527 |
| Current Issue 6A clarification selection | 0.117660 |

The new path is a candidate, not a claimed score improvement, until it is run
against the official catalog with the same evaluator and split.
