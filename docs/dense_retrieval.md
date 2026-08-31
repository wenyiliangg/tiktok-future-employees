# Standalone dense retrieval

Issue 2A adds an independently testable dense retriever. It is deliberately not
wired into `starter/agent.py` and has no dependency on conversation state, BM25,
routing, or hybrid fusion.

## Model and embedding text

The default model is `sentence-transformers/all-MiniLM-L6-v2`, pinned to commit
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. It is a compact, CPU-compatible
general-purpose sentence embedding model with 384-dimensional vectors. The model
name and optional immutable revision are configurable. There
is no random, hashed, or fake production fallback: a missing model dependency or
unavailable model raises a clear error.

`ProductTextBuilder` uses a fixed, labelled order:

1. title, for the product's primary semantic identity;
2. category hierarchy, for product type and audience;
3. store as brand, for brand-oriented searches;
4. features, for materials, fit, benefits, and use cases;
5. description, for additional natural-language context;
6. a deterministic allowlist of customer-searchable `details`, including real
   catalog keys such as `Material`, `Style`, `Color`, `Size`, `Department`,
   `Pattern`, `Occasion`, `Sport Type`, `Fit Type`, and `Closure Type`.

Identifiers, model and part numbers, timestamps, ranks, ratings, package/product
dimensions, weights, shipping data, battery metadata, and price are excluded.
The format version is `catalog-semantic-text-v1`. Dictionaries are sorted by key,
lists retain catalog order, whitespace is normalized, and empty sections are
omitted. Catalog rows are never mutated.

## Installation and usage

From the repository root:

```bash
python3 -m pip install -r requirements.txt
```

Minimal usage:

```python
from dense_retrieval import DenseRetriever

retriever = DenseRetriever.from_catalog(
    "data/catalog.jsonl",
    cache_path="data/.dense-retrieval/catalog-minilm.npz",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    batch_size=64,
)
results = retriever.retrieve("waterproof blue trail shoes", top_n=200)
```

Pass `rebuild_cache=True` to force regeneration. `cache_path` and `batch_size`
are ordinary constructor arguments and may be configured by the caller. The
default encoder explicitly uses CPU; inject another encoder implementing the
small `Encoder` protocol for tests or a deliberate deployment configuration.
Nothing expensive runs at module import time.

When selecting a different `model_name`, also pass its immutable
`model_revision`; pass `None` only when intentionally following that model's
mutable default branch.

## Cache safety and compatibility

The cache is one NumPy `.npz` archive loaded with `allow_pickle=False`. It stores
the `float32` matrix, ordered string ASIN mapping, and JSON metadata. Metadata
contains the SHA-256 checksum of the exact catalog bytes, product count, model
name, configured revision, resolved model commit when the library exposes it,
text-builder version, dimension, dtype, L2-normalization flag, and cache-schema
version.

On every startup the catalog is read and validated, so the cache mapping must
exactly match the catalog's unique non-empty ASIN sequence. Matrix shape, dtype,
finite values, and unit norms are also checked. A stale, corrupt, incomplete, or
incompatible archive is rebuilt. Replacement is written and fsynced as a
same-directory temporary file before one atomic `os.replace`; a failed rebuild
does not overwrite the previous artifact.

Generated caches belong under `data/.dense-retrieval/`, which is narrowly
ignored by Git. Do not commit caches, temporary artifacts, or benchmark output.

## Verification and benchmarks

Run all unit tests:

```bash
python3 -m unittest discover -s tests -v
```

Run the reproducible production benchmark:

```bash
python3 -m benchmarks.benchmark_dense_retrieval \
  --catalog data/catalog.jsonl \
  --cache data/.dense-retrieval/catalog-minilm.npz \
  --batch-size 64 --warmup-runs 2 --runs 10
```

It reports cold startup, time inside initial catalog encoding, cached startup,
matrix bytes, process peak RSS, and average/p95 exact-search latency over broad
and specific queries. Cold startup includes a model download if one occurs.
For 50,000 x 384 `float32` values, the expected raw matrix is exactly 76,800,000
bytes (76.8 decimal MB, about 73.24 MiB).

To verify benchmark plumbing offline without claiming model-quality or real-model
performance, explicitly add `--deterministic-fake-dimension 384`. This fake is
available only in the benchmark module and is never a production fallback.

Actual measurements and any environment blockers are recorded in the Issue 2A
implementation report. The completed Issue 2A run produced these measurements:

- Date: 2026-08-29.
- Environment: Apple arm64, macOS 26.5.2, 12 logical CPUs, Python 3.12.2,
  NumPy 2.5.2, CPU inference.
- Catalog/model: 50,000 products, `all-MiniLM-L6-v2`, resolved model commit
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, 384 dimensions, batch size 64.
- Method: the model was already downloaded, two warm-up passes over six queries,
  then five measured passes (30 query retrievals total). Query latency includes
  query encoding and exact NumPy search; model download time is excluded.
- Initial embedding generation: 293.974 seconds; total cold startup: 295.948
  seconds; compatible cached startup: 0.724 seconds with zero encoding time.
- Matrix: 76,800,000 bytes (76.8 MB / 73.24 MiB); process peak RSS was
  1,265,139,712 bytes (about 1.18 GiB).
- Query latency: 17.276 ms average and 19.426 ms p95.

The separate explicit fake-encoder plumbing run over the same 50,000 rows is not
a model performance result. It confirmed the benchmark/cache path and the same
76.8 MB matrix size.
