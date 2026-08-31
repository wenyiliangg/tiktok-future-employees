# Demo Runbook

This is a six-minute judge-facing walkthrough of the final TechJam shopping
agent. Run every command from the repository root after activating the project
environment.

## Before the presentation

1. Confirm that `data/catalog.jsonl` is present.
2. Install `requirements.txt` in a Python 3.12 virtual environment.
3. Run `python -m unittest discover -s tests -v` once.
4. Run the final evaluator once and keep `results.json` as a fallback if the
   live terminal or screen sharing fails.
5. Increase the terminal font size and open this runbook beside the terminal.

Do not open private data, API keys, raw reviews, or the hidden target during the
demo.

## Presentation flow

| Time | Action | What to say |
| --- | --- | --- |
| 0:00-0:40 | Show the root README and final-results table. | “The task is exact product discovery, not general similarity. Our deterministic agent searches 50,000 catalog products, asks bounded clarification questions, and returns up to ten exact ASINs.” |
| 0:40-1:15 | Show the architecture diagram. | “A turn updates isolated conversation state and an evidence ledger. BM25 creates an anchor pool; a catalog-derived ranker combines category, phrase, rare-token, constraint, popularity, and history evidence. The question policy acts on the ranked pool, and every response passes central validation.” |
| 1:15-2:45 | Run the interactive demo below. | “The agent is local, has no API key, and spends zero model tokens. It can ask at most two non-repeated questions, remember answers, rotate rejected products, and safely replace intent.” |
| 2:45-4:15 | Run or show the scored reproduction. | “On the public 200-session evaluator the final frozen policy reaches 87% HitRate@10 and a 0.733983 TechnicalScore. The output includes scenario metrics, correctness counters, latency, policy fingerprints, and determinism hashes.” |
| 4:15-5:20 | Show the score-progression table. | “The biggest gains came from remembering the useful query after generic feedback, ranking catalog-derived evidence, and asking only answerable questions with positive expected utility.” |
| 5:20-6:00 | Show limitations. | “The public set is development evidence, Intent Override is still the weakest route, and startup builds a memory-heavy index. We kept rejected experiments and paired comparisons for auditability.” |

## Live interactive demo

Start the terminal interface:

```bash
python demo.py
```

The index takes several seconds to build. After the `Customer>` prompt appears,
use this dialogue:

```text
Customer> I need black running shoes under $100

Customer> Breathable mesh and good arch support matter most

Customer> Those options are not quite right

Customer> Actually, ignore that; I need a blue rain jacket instead

Customer> Waterproof material matters most
```

Points to call out while the dialogue runs:

- Every turn still returns ranked recommendations, even when the agent asks a
  question.
- The question is represented in both natural language and the structured
  `ask_attribute` field.
- The second answer becomes current evidence for the next ranking.
- Generic rejection rotates away from products already shown without replacing
  the useful request text.
- `Actually, ignore that` triggers intent replacement and clears the rejected
  product history for the new request.
- The output includes catalog title and price only for presentation; the
  evaluator-facing response remains the required ASIN contract.

Enter `/quit` when finished.

## Scored reproduction

For a live full run:

```bash
python -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --retrieval-mode contextual \
  --contextual-policy contextual.category-evidence.v1 \
  --clarification-policy clarification.category-evidence-utility-buying.v1 \
  --dense-cache data/.dense-retrieval/catalog-minilm.npz \
  --output results.json
```

Highlight these values in the terminal output:

```text
sample_count:                200
hit_rate_at_10:              0.87
mrr:                         0.519609
mttc:                        3.845
efficiency:                  0.7155
recommended_technical_score: 0.733983
reported total tokens:       0
response contract violations: 0
```

Then confirm the two deterministic hashes:

```bash
python -c "import json; r=json.load(open('results.json')); print(json.dumps(r['determinism'], indent=2))"
```

Expected non-timing values:

```text
normalized_response_sha256 = 0b17cb2037fa6a18580e98b097b8268655d2121855ce8c9cf9f158d1b2a1e486
session_outcome_sha256      = 654dbb33898ec07a742b88c04129519faf82b3c032a0fa5b7f7888e5df24989e
```

## Fast fallback if there is not enough time

Show the committed final comparison without rerunning all sessions:

```bash
python -c "import json; r=json.load(open('docs/results/autonomous_optimization/final_comparison.json')); print(json.dumps(r['champion'], indent=2))"
```

Then show the previously recorded correctness and scenario metrics:

```bash
python -c "import json; r=json.load(open('docs/results/autonomous_optimization/p5_official_run2.json')); keys=['hit_rate_at_10','mrr','mttc','efficiency','recommended_technical_score','response_contract_diagnostics','scenario_metrics','determinism']; print(json.dumps({k:r[k] for k in keys}, indent=2))"
```

## Likely judge questions

**Why not use an LLM?**

The target is exact catalog identity and the simulator supplies catalog-derived
evidence. A local deterministic ranker was faster, free of token cost, easier to
reproduce, and stronger than the tested dense/hybrid alternatives on this set.

**How do you prevent overfitting?**

Experiments were predeclared, tested on disjoint-target shadow suites, compared
per session, checked with paired bootstrap intervals, and frozen before official
runs. Public-set selection risk remains an explicit limitation.

**What happens when the user changes their mind?**

Current intent replaces incompatible active state. Old evidence is stored in a
separate lower-weight historical channel, and known-negative recommendations
are cleared so the new request starts cleanly.

**What happens if a component fails?**

Optional ranking and clarification boundaries fail closed, while the final
response boundary attempts protected BM25 and always validates schema,
catalog membership, uniqueness, and list size.

**What would you improve next?**

Focus on Intent Override, broaden paraphrase-safe state extraction on held-out
data, serialize the category index to reduce startup cost, and add explanations
without changing the scored ranking.
