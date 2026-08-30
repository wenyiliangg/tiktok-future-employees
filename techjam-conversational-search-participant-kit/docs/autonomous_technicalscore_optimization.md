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

The preliminary evidence is checkpointed at `df239c8`, and the isolated H2
runtime candidate is checkpointed at `c6979a1`. The final pre-official check
confirmed a clean tree, an unchanged evaluator, baseline public/catalog hashes,
and no public-set identifiers or evaluator imports in runtime code. The
invocation then reached its mandatory `4h15m` wall-clock limit, so no H2
official run was started. This is a time pause, not an H2 rejection.

## H2 official result and rejection

The single official H2 run scored `0.305243`, a decrease of `-0.000750` from
the `0.305993` feedback-memory champion. HR@10 remained `0.405`, MTTC remained
`8.125`, and Efficiency remained `0.2875`, but MRR fell from `0.153310` to
`0.150810`. Buying, Browsing, and Boundary aggregate outcomes were unchanged;
Intent Override MRR fell from `0.109722` to `0.093056`. The run had zero
exceptions, invalid responses, invalid ASINs, duplicates, repeated questions,
or contract violations.

The paired comparison found 80 unchanged hits, 119 unchanged misses, and one
worse-rank Intent Override session. It gained and lost no hits. The 10,000
resample paired bootstrap mean was `-0.000750`, with a 95% interval of
`[-0.002250, 0.000000]` and zero probability of a positive delta in the
resamples. The exact-signature mechanism therefore fails the predeclared
promotion threshold and is rejected. Its isolated runtime commit is reverted;
the diagnostic, shadow, official, paired, and campaign evidence is retained.

H1 evidence was checkpointed at `169b2c8` and its implementation was reverted
at `9b62271`. H2 rejection evidence was checkpointed at `dd13f5c` and its
implementation was reverted at `d2a671d`. Both rejected mechanisms remain
fully documented without remaining in evaluator-facing runtime code.

## Final verification and campaign stop

After the H2 revert, `starter/`, `config/`, and `evaluator/` exactly match the
selected `4d7af4e` champion. The final retained suite passes (`239` tests), Ruff
passes, and focused MyPy passes on the clarification, response-validation,
conversation-state, and evaluator boundary. Public and catalog hashes remain
unchanged, the evaluator has no campaign modification, and runtime contains no
public-set identifiers, evaluator imports, benchmark coupling, competitor code,
or hosted-service dependency.

The reserved official reproduction run exactly reproduced TechnicalScore
`0.305993`, HR@10 `0.405`, MRR `0.153310`, MTTC `8.125`, and Efficiency
`0.2875`. Response hash
`9e50a9e37aea0e149bfaff346266d24628679558c1999bde36146681d67bca6d`
and session hash
`ee2d7a3a683273d17dcb2e8a9d0dcc1ab7c05816c4c611a43be9070133e45e67`
also matched exactly. There were zero exceptions, invalid responses, invalid
ASINs, duplicate recommendations, repeated questions, or contract violations.
Cold startup was `5.617351s`, full evaluator wall time was `23.602872s`, mean
response latency was `15.595118ms`, p95 was `22.324583ms`, and peak RSS was
`1157.171875 MiB`.

The final reproduction command was:

```bash
python -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --retrieval-mode contextual \
  --contextual-policy contextual.feedback-memory.v1 \
  --clarification-policy clarification.feedback-memory.v1 \
  --dense-cache data/.dense-retrieval/catalog-minilm.npz \
  --output /tmp/technical-score-final-champion-reproduction.json
```

The campaign initially stopped after H2 at the user's request. The user then
clarified that a worse result should stop only that experiment and that the
campaign should continue to the next distinct iteration. The retained champion
remains `contextual.feedback-memory.v1` with
`clarification.feedback-memory.v1` from commit `4d7af4e`. The score is an
official public-set result, not a private-set claim. The large gap to the
external `0.906151` reference remains unresolved, and public-set selection risk
remains despite non-public shadow diagnostics and mechanism-level guardrails.

## H3 experiment card: override-history tail evidence

Experiment ID: `H3-override-history-tail-v1`

Hypothesis: Intent Override currently replaces the active raw request entirely,
even though the earlier raw phrase can contain legitimate target-derived
product-identification evidence. Storing that phrase separately and using it
only as a low-confidence BM25 source in the unprotected recommendation tail can
improve Intent Override HR/MRR without treating obsolete preferences as active
requirements or changing the champion's protected current-request prefix.

Expected metric changes: Intent Override only; preserve ranks 1-8 from the
current request, improve target recovery or rank in positions 9-10, and leave
Buying, Browsing, and Boundary exactly unchanged. HR@10 and MRR are primary;
MTTC may improve if a recovered tail hit appears earlier.

Minimal implementation:

1. Add an opt-in declarative contextual policy with one configurable historical
   evidence weight and an eight-result current-request protected prefix.
2. On an explicit override, archive the prior active raw request separately
   before replacing active intent and clearing known-negative product IDs.
3. Retrieve archived text only after an override and fuse it only in the
   unprotected tail; never place it in active constraints, routing, dense query,
   clarification state, or contradiction filters.
4. Bound history to one phrase per session, reset it between sessions, and
   preserve all champion behavior when the policy is disabled or retrieval
   fails.

Risks: obsolete preferences can identify a plausible but wrong product;
boilerplate history can disturb tail coverage; repeated overrides can retain
the wrong phrase; a low-weight source may be too weak to matter. The mechanism
must remain semantically subordinate to the current request.

Falsifiable threshold: a deterministic non-public Intent Override shadow must
show a positive TechnicalScore delta with no lost champion hits and no change
to non-override scenarios. Full correctness, isolation, failure-fallback,
integrity, and leakage gates must pass before one official run. Official
promotion still requires at least `+0.010`, or a smaller positive delta whose
paired 95% interval lower bound is above zero, with no material shadow
regression. A worse or unresolved result rejects only H3 and advances the
campaign to H4.

Rollback: commit the isolated history-policy runtime change separately; on
rejection, retain its evidence and revert only that implementation commit.

## H3 preliminary evidence

The benchmark-only diagnostic evaluated 64 Intent Override targets selected
from 256 non-public shadow products with zero public-target overlap. The current
override phrase placed 23 targets in the BM25 Top 100, while the archived phrase
placed 59 there. With an eight-result current prefix and `0.5` historical tail
weight, the diagnostic gained eight Top-10 hits, improved one existing rank,
lost none, changed zero protected prefixes, and increased mean reciprocal rank
by `0.013715278`.

The first integrated shadow exposed and removed a non-override state-fusion
leak before qualification. The narrowed runtime activates the supplemental
source only when archived override text exists. On the final deterministic
64-session balanced shadow, Buying, Browsing, and Boundary were then exactly
identical to the champion. Intent Override HR@10 rose from `0.500` to `0.875`,
MRR from `0.361979` to `0.372917`, and MTTC fell from `7.5625` to `6.000`.
Overall shadow TechnicalScore rose from `0.386477` to `0.441985`
(`+0.055508`), with six gained hits, zero lost hits, three earlier shared hits,
and zero later shared hits.

All shadow correctness counters are zero, the repeat transcript is
deterministic, and malformed metadata, missing dense cache, component failure,
catalog reorder, and consecutive-session isolation checks pass. The complete
suite passes (`247` tests), Ruff passes, focused MyPy passes, and compile checks
pass. The selected fingerprints are retrieval
`3e3fbcb9624e7ffd4a76a90914c9166e9e94a4f0510a2f8319da60046ae79c6e`,
override-history
`f1eff95d936386efe560c927ec1ecb3392fa970d5a1ad830759d30dc016fee1c`,
and clarification
`56550e0f09f152db8be1a3988bacda499955e8b5683e95eedad44b3ce19fb7a5`.
H3 is qualified for one official full run after its evidence and isolated
runtime commits are checkpointed.

The preliminary evidence is checkpointed at `1d39702`, and the isolated H3
runtime candidate is checkpointed at `1172b6a`.

## Evaluation budget

- New campaign official runs used: 3.
- New campaign official runs remaining: 9.
- The reserved final reproduction run has been consumed.
- Fixed seed: `20260830`.

## Immediate next action

Commit the H3 preliminary evidence and isolated runtime candidate, verify the
clean-tree and immutable-input gates, then spend one official run on the exact
fingerprinted configuration followed by the fixed-seed paired bootstrap.
