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

## H1 official result and rejection

The single official candidate run scored `0.328091`, an aggregate improvement
of `+0.022098` over the feedback-memory champion. HR@10 rose from `0.405` to
`0.415`, MRR from `0.153310` to `0.189637`, MTTC fell from `8.125` to `7.815`,
and Efficiency rose from `0.2875` to `0.3185`. Buying, Intent Override, and
Boundary aggregate outcomes were exactly unchanged; the aggregate gain came
from Browsing. The run had zero exceptions, invalid responses, invalid ASINs,
duplicates, repeated questions, or contract violations.

The required paired analysis rejects H1. It recorded nine new hits but seven
lost champion hits, exceeding the predeclared maximum of one. There were 15
improved sessions, nine regressed sessions, and 176 unchanged sessions. The
10,000-resample paired bootstrap TechnicalScore interval was `[-0.009625625,
0.054877411]`, with `0.9072` of resamples positive; the interval crosses zero.
The mechanism is useful but not sufficiently resolved or HR-preserving in this
isolated form. H1 is therefore rejected, and only its implementation is
reverted. The general shadow and paired-analysis infrastructure is retained.

## H2 experiment card: catalog signature index

Experiment ID: `H2-catalog-signature-index-v1`

Hypothesis: a deterministic offline index over participant-visible catalog
phrases and rare tokens can convert a single disclosed target-derived feature
or detail into a small exact-product hypothesis set. Used as a rank-head source
with the feedback-memory champion unchanged underneath, it should materially
improve MRR and preserve Top-10 coverage.

Expected metric changes: largest improvement in Buying rank and turn-one hits;
improved post-override rank from the current explicit phrase; smaller gains in
Browsing only when a legitimate clarification reply is available. Boundary
should remain neutral. MRR is the primary metric, followed by HR@10 and MTTC.

Minimal implementation:

1. Build an in-memory index only from catalog title, feature, detail, category,
   material/color-like text, and valid price metadata.
2. Normalize deterministically and compute phrase/token document frequency.
3. Retrieve by exact normalized phrase containment and rare-token conjunction,
   with configuration-held thresholds and stable ASIN tie-breaking.
4. Put high-confidence signature candidates at the recommendation head while
   retaining the existing contextual candidates for Top-10 coverage.
5. Preserve raw-message BM25 and every protected fallback.

Risks: common or boilerplate features may produce false confidence; truncated
or paraphrased evidence may miss exact phrases; extra memory may be material;
historical Intent Override evidence must not become an active filter; generic
token overlap must not displace champion hits.

Falsifiable threshold: before runtime integration, a non-public diagnostic must
show that at least one catalog-derived phrase puts its owner in the Top 10 for a
clear majority of targets and rank 1 for a material fraction. The integrated
candidate must pass all correctness/shadow gates, improve official
TechnicalScore by at least `0.010` or have a paired interval lower bound above
zero, and lose no more than one champion hit absent a much larger statistically
resolved gain.

Rollback: commit the isolated index/integration, record paired evidence, and
revert those commits if any promotion gate fails. Do not alter the evaluator,
catalog, public sessions, or the retained feedback-memory retrieval policy.

## H2 preliminary evidence

The benchmark-only diagnostic evaluated 1,024 participant-visible catalog
phrases from 256 non-public targets with zero public-target overlap. A target's
first disclosed phrase recovered its owner in `231/256` cases (`0.902344`). The
first phrase was unique for 57 targets, while considering all four phrases
produced 317 unique-owner cases. Because broader buckets were often very large
(median `11.5`, mean `2498.45`, maximum `25451` candidates), the runtime
candidate deliberately uses only exact phrases whose owner is unique; the
planned rare-token and multi-owner paths remain disabled.

On the deterministic 64-session non-public shadow suite, the unique-signature
rank head improved TechnicalScore from `0.386477` to `0.396672` (`+0.010195`).
It gained and lost zero hits, moved one shared hit earlier, and moved none
later. Buying MRR improved from `0.461979` to `0.534896`; Intent Override MRR
improved from `0.361979` to `0.416667` and MTTC from `7.5625` to `7.4375`.
Browsing and Boundary were exactly unchanged. The repeated transcript was
deterministic and all correctness and robustness checks passed.

The complete suite passes (`245` tests), formatting and lint pass, immutable
public and catalog hashes match the campaign baseline, runtime code contains no
benchmark identifiers or evaluator imports, and the evaluator has no campaign
diff. The selected configuration fingerprints are retrieval
`2012434191e1c138d1c8378b4bb00c825e88e59ce6c300bb0e4f6900d5e356dd`,
signature index
`e4858229d00895266d595405f36ffb84f887e42615a3bf40c92007325f54fa5c`,
and clarification
`56550e0f09f152db8be1a3988bacda499955e8b5683e95eedad44b3ce19fb7a5`.
H2 is qualified for one official full run.

## Evaluation budget

- New campaign official runs used: 1.
- New campaign official runs remaining: 11.
- One of the remaining runs is reserved for final reproduction.
- Fixed seed: `20260830`.

## Immediate next action

Commit the H2 preliminary evidence and isolated runtime candidate, recheck the
immutable inputs and clean worktree, then spend one official run on the exact
fingerprinted H2 configuration. Compare every session with the retained
feedback-memory champion using the fixed-seed paired bootstrap before deciding
retention or rejection.
