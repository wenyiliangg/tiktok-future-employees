# Bounded autonomous TechnicalScore optimization

## Campaign status

The campaign is running on local branch
`experiment/bounded-technicalscore-optimization`. The hardened campaign base is
`c57dfce8f7aeb44cb366ab3a90c11e9ac613cf07`, which contains Issue 6B commit
`d7ffe14ef5dbe8b6a8ceea23ed18c68d7618a70c` and exactly reproduces the selected
Issue 6A configuration.

The initial campaign champion is the already completed feedback-memory
experiment at `4d7af4eea2cfb8626fc858452ba4d0ab7f2cc509`. It scored `0.305993` in two
identical official runs. This pre-campaign result is being reused as committed
evidence rather than spending campaign evaluation budget to rediscover it.

No campaign changes will be merged into `main`, and the campaign branch will
not be pushed automatically.

## Original technical thesis

The shopping protocol frequently discloses phrases derived from legitimate
target catalog metadata. A strong offline agent should therefore behave as an
active catalog hypothesis search system: preserve current requirements and
historical identification evidence separately, ask questions that reveal
useful evidence, intersect catalog signatures, and rank exact products at the
head while keeping plausible alternatives in the remaining Top 10.

This thesis was derived independently from the official evaluator and local
catalog. No competitor repository, implementation, constants, or prose were
inspected or copied. The external `0.906151` result is retained only as an
explicitly unverified public reference.

## Integrity baseline

- Evaluator Git-tree listing SHA-256:
  `d36fe325b1d1f97c50e336cb9fe593f2015fc3d012ccaf84ed3015763930cef4`.
- Public set SHA-256:
  `857259f7a438e6188ac63e18995b6ff4489bfcfc4a716a798b9a2aa0ee8f7579`.
- Catalog SHA-256:
  `da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`.
- Runtime source contains no public sample identifiers, target identifiers, or
  evaluator imports. Mentions of the evaluator in runtime files are comments
  and docstrings only.

These identities must be checked before every official run.

## Starting metrics

| Configuration | HR@10 | MRR | MTTC | Efficiency | TechnicalScore |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact BM25 | 0.125 | 0.068034 | 9.810 | 0.1190 | 0.106710 |
| Hardened Issue 6A/6B | 0.140 | 0.074867 | 9.740 | 0.1260 | 0.117660 |
| Feedback-memory champion | **0.405** | **0.153310** | **8.125** | **0.2875** | **0.305993** |

## Retained pre-campaign experiment

`H0-feedback-memory-bm25` reuses the last informative raw turn as the protected
BM25 query only after recognized generic negative feedback, while continuing
to exclude rejected recommendations. It retained every exact-BM25 success,
gained 56 sessions against BM25, and gained a net 53 sessions against the
hardened champion. Its complete evidence is in
`docs/technical_score_iterations/iteration_01_feedback_memory.md`.

## H1 experiment card: protocol-aware open evidence

Experiment ID: `H1-open-evidence-questions-v1`

Hypothesis: the current question selector is poorly aligned with the official
simulator because it asks mostly `category`, which cannot reveal a
target-derived constraint. A reasonable open question represented by the legal
`other` attribute can reveal up to two undisclosed constraints. A bounded
second question may remain valuable after a partial answer or the Boundary
scenario's first decline.

Expected metric changes:

- increase HR@10 by exposing target evidence earlier;
- move matches toward rank one through more specific subsequent queries;
- reduce MTTC despite spending a question turn, because recommendations are
  scored before the reply and the evidence affects later turns;
- increase TechnicalScore by at least `0.010` over `0.305993`.

Expected scenarios: largest gains in Browsing, followed by Buying and Boundary;
Intent Override must remain safe when an override interrupts a pending question.

Minimal implementation:

1. Preserve `contextual.feedback-memory.v1` retrieval unchanged.
2. Add one named, declarative clarification policy.
3. Prefer an open `other` question before low-yield `category` or `brand`.
4. Bound total questions and never repeat a declined or resolved attribute.
5. Clear or safely resolve pending questions when an override arrives.
6. Keep all existing response validation and fallback paths.

Risks: unnecessary dialogue, override misparsing, repeated declines, reduced
negative-feedback rotation, and overfitting to literal public templates.

Falsifiable threshold: pass focused, leakage, determinism, and shadow gates;
then improve official TechnicalScore by at least `0.010`, lose no more than one
champion hit, and introduce no correctness violation. Otherwise reject and
revert only the H1 implementation commit.

## H1 preliminary evidence

The deterministic shadow suite uses 64 catalog targets excluded from all 200
public targets, balanced 16 per scenario. It includes four independent dialogue
templates, case and punctuation variants, 22 partial-disclosure sessions,
catalog reorder verification, malformed metadata and price cases, missing-cache
and component-failure fallbacks, and consecutive-session isolation.

The question-yield diagnostic found potential disclosure counts of `128` for
`other` and `125` for `feature`, compared with zero target-derived disclosures
for `category` and `brand`. The first candidate also asked on Buying and exposed
a route-specific regression. H1 was therefore narrowed, without using public
outcomes, to the observable Browsing and Boundary routes.

The refined candidate improved shadow TechnicalScore from `0.386477` to
`0.458534` (`+0.072057`), gained five hits, lost none, and exactly preserved
Buying and Intent Override outcomes. Its repeat transcript was deterministic;
all correctness counters and all four robustness checks were zero/pass. The
complete artifacts are under `docs/results/autonomous_optimization/shadow_results/`.

The complete test suite passes (`238` tests), lint passes, immutable public and
catalog hashes match the campaign baseline, runtime code contains no benchmark
identifiers or evaluator imports, and the evaluator has no campaign diff. H1 is
therefore qualified for one official full run.

## Evaluation budget

- New campaign official runs used: 0.
- New campaign official runs remaining: 12.
- One of the remaining runs is reserved for final reproduction.
- Fixed seed: `20260830`.

## Immediate next action

Create the deterministic benchmark-only shadow suite and an independent
question-answerability diagnostic. Implement H1 only if those preliminary
artifacts support the experiment card. Do not spend an official run before the
candidate passes those gates.
