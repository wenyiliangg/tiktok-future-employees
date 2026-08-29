# Reproduction guide

Run all commands from `techjam-conversational-search-participant-kit` unless a
step says otherwise.

## 1. Environment

Use Python 3.10 or later in a clean virtual environment:

```bash
python -m venv .venv
```

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Dense/hybrid execution needs the configured sentence-transformer model. Its
first use may download model files; subsequent runs can use the local model
cache. The retrieval code does not need API credentials.

## 2. Data

The committed `data/public_set.jsonl` contains 200 development sessions. The
official `data/catalog.jsonl` should contain 50,000 products but is intentionally
ignored and absent from Git.

The intended source is the repository's [GitHub Release](https://github.com/wenyiliangg/tiktok-future-employees/releases): download
`catalog.jsonl.gz` and `SHA256SUMS`, verify the archive, and decompress it to
`data/catalog.jsonl`. As checked on 2026-08-29, no release is currently
published. Until the organizer publishes it or supplies an authorized copy, a
new clone cannot run catalog-backed evaluation. Do not create a substitute and
report its results as official.

Verify an available archive using `sha256sum` on macOS/Linux or `Get-FileHash
catalog.jsonl.gz -Algorithm SHA256` in PowerShell. Then extract and validate:

```bash
python -c "import gzip, pathlib, shutil; src=pathlib.Path('catalog.jsonl.gz'); dst=pathlib.Path('data/catalog.jsonl'); dst.parent.mkdir(exist_ok=True); i=gzip.open(src,'rb'); o=dst.open('wb'); shutil.copyfileobj(i,o); i.close(); o.close()"
python -c "import json, pathlib; rows=[json.loads(x) for x in pathlib.Path('data/catalog.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]; assert len(rows)==50000 and all(x.get('parent_asin') for x in rows); print('catalog rows:', len(rows))"
```

The Issue 2B result record identifies the evaluated catalog SHA-256 as
`da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`.
This is reproducibility evidence, not permission to redistribute the catalog.

## 3. Index and cache generation

Lexical mode requires no cache-generation command. Constructing `Agent` reads
the catalog and builds an in-memory FTS5 index every run.

Dense/hybrid construction calls `DenseRetriever.from_catalog`. It reuses
`data/.dense-retrieval/catalog-minilm.npz` when all metadata checks pass;
otherwise it encodes the catalog and atomically creates a new cache. The cache
is ignored because it is generated, large (78,802,526 bytes in the recorded
run), and tied to the local catalog/model configuration.

To build/validate the default cache and benchmark it explicitly on macOS/Linux:

```bash
python -m benchmarks.benchmark_dense_retrieval --catalog data/catalog.jsonl --cache data/.dense-retrieval/catalog-minilm.npz --batch-size 64 --warmup-runs 2 --runs 10
```

`DenseRetriever.from_catalog(..., rebuild_cache=True)` is the programmatic
force-rebuild option. Do not commit the resulting NPZ or model files.
The benchmark script currently imports the Unix `resource` module and therefore
does not run on Windows; dense retrieval and the local evaluator themselves do.

## 4. Fixed weak-baseline reproduction

The current `starter/agent.py` replaced the original weak BM25 implementation,
so running current lexical mode is **not** a baseline reproduction. The frozen
baseline code is commit `166bae7`, and its recorded metrics are versioned in
`docs/baseline_results.json`.

Use a separate worktree so the current branch is not modified:

```bash
# Run from the parent repository, not this kit directory
git worktree add ../tiktok-baseline 166bae7
```

Copy the same authorized `data/catalog.jsonl` into
`../tiktok-baseline/techjam-conversational-search-participant-kit/data/`, then:

```bash
cd ../tiktok-baseline/techjam-conversational-search-participant-kit
python -m evaluator.local_evaluator --catalog data/catalog.jsonl --dataset data/public_set.jsonl --output results.json
```

Expected aggregate metrics:

| HR@10 | MRR | MTTC | Efficiency | TechnicalScore |
| ---: | ---: | ---: | ---: | ---: |
| 0.125 | 0.068034 | 9.81 | 0.119 | 0.106710 |

Record any deviation with the commit, catalog hash, Python/platform details, and
result JSON instead of silently replacing the fixed reference.

## 5. Current retrieval evaluations

Return to the current kit worktree and run the current integrated agent. All
three modes automatically feature-rerank a bounded pool before returning ten
recommendations:

```bash
python -m evaluator.local_evaluator --catalog data/catalog.jsonl --dataset data/public_set.jsonl --retrieval-mode lexical --output results.json

python -m evaluator.local_evaluator --catalog data/catalog.jsonl --dataset data/public_set.jsonl --retrieval-mode dense --dense-cache data/.dense-retrieval/catalog-minilm.npz --output results.json

python -m evaluator.local_evaluator --catalog data/catalog.jsonl --dataset data/public_set.jsonl --retrieval-mode hybrid --lexical-candidates 200 --dense-candidates 200 --final-candidates 10 --lexical-weight 1.0 --dense-weight 1.0 --rrf-k 60 --dense-cache data/.dense-retrieval/catalog-minilm.npz --output results.json
```

The versioned raw reference outputs under `docs/results/issue_2b/` predate the
feature reranker and are not expected outputs for these current commands.
Current quality metrics should be deterministic for the same code, catalog,
model revision, configuration, and dataset. Startup/latency/RSS are hardware-
and cache-state-dependent. The default `results.json` is ignored; copy only a
reviewed, intentionally versioned result into a documented results directory.

Run the catalog-independent synthetic reranker benchmark with:

```bash
python -m benchmarks.benchmark_feature_reranker --pool-size 100 --runs 1000
```

For an exact source checkout of the recorded pre-reranker Issue 2B comparison,
create a separate worktree at commit `36a9974`, provide the same authorized
catalog, and use the commands preserved in `docs/issue_2b_results.md` there.

## 6. Final and ablation evaluation

No Issues 6A/6B final configuration or complete ablation artifacts are present
on this branch. No end-to-end result has been recorded after feature-reranker
integration. Do not label the Issue 2B rows as current or final results. The
current evaluator only supports `lexical`, `dense`, and `hybrid`; there is no
`final` mode.

After final integration, run the supported mode with every non-default setting
spelled out, for example:

```bash
python -m evaluator.local_evaluator --catalog data/catalog.jsonl --dataset data/public_set.jsonl --retrieval-mode hybrid --lexical-candidates 200 --dense-candidates 200 --final-candidates 10 --lexical-weight 1.0 --dense-weight 1.0 --rrf-k 60 --dense-cache data/.dense-retrieval/catalog-minilm.npz --output docs/results/final.json
```

For each ablation, change one documented component/configuration at a time and
write a different JSON file. A valid result record includes:

- branch and commit SHA;
- evaluation date, platform, Python and dependency versions;
- exact catalog SHA-256, dataset, and sample count;
- model name/revision and cache state;
- complete command and configuration;
- aggregate and per-scenario quality metrics;
- startup, average response, p95 response, and peak RSS;
- caught response-exception count.

Only commit small reviewed result artifacts. Never commit the catalog, dense
cache, private sessions, credentials, logs containing secrets, or unreviewed
evaluator output.

## 7. Recorded evidence

- `docs/baseline_results.json`: fixed weak-BM25 aggregate reference.
- `docs/issue_2b_results.md`: environment, configuration, quality, scenario,
  runtime, memory, and interpretation for the pre-reranker retrieval-mode
  comparison.
- `docs/results/issue_2b/lexical.json`: improved lexical raw output.
- `docs/results/issue_2b/dense.json`: compatible-cache dense raw output.
- `docs/results/issue_2b/hybrid.json`: fixed-hybrid raw output.
- `docs/results/issue_2b/dense_cold_cache_build.json`: dense cache-build timing
  with the same dense quality results.

Results must be read alongside the limitations in the main README: none of the
recorded experimental modes beats the fixed baseline, and the feature reranker
plus later standalone router/fallback/ambiguity modules are not part of these
evaluations.
