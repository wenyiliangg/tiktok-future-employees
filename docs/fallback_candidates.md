# Deterministic fallback candidates

`starter.fallback_candidates` produces valid catalog candidates for weak,
missing, or Boundary intent. It only generates candidates; routing code decides
whether to call it.

## Behavior

- An empty query and empty profile still return up to the requested number of
  unique products from the frozen catalog.
- Profile tags and summaries are parsed through Issue 1A's finite slot aliases.
  They always remain soft scoring evidence. Purchase frequency, rating style,
  evaluator labels, and unknown profile values are ignored.
- An active conversation value replaces profile history for the same field.
  Explicit exclusions suppress matching profile evidence and penalize matching
  products. Neither positive evidence nor exclusions become fallback filters,
  so sparse metadata cannot accidentally empty the result set.
- Average rating and rating count provide small deterministic catalog-quality
  priors when present. Missing, non-finite, or out-of-range values contribute
  no prior.
- Duplicate `parent_asin` rows are removed on load. Every returned identifier is
  therefore unique and belongs to the loaded catalog.
- Equal scores are ordered by `parent_asin`. No randomness, public target data,
  evaluator labels, or memorized identifier lists are used.

## Usage

```python
from starter.fallback_candidates import FallbackCandidateGenerator

fallback = FallbackCandidateGenerator.from_jsonl("data/catalog.jsonl")
candidates = fallback.generate(
    query=active_search_query,
    user_profile=user_profile,
    top_n=10,
    removed_constraints=session_state.removed_constraints,
)
```

`query` and `user_profile` are optional. Passing `removed_constraints` is
recommended when a raw profile is provided after a customer explicitly removed
a profile-derived preference. This prevents the raw profile adapter from
reintroducing that slot.

## Candidate contract

Each `FallbackCandidate` contains:

- `parent_asin`
- `fallback_score`
- `source="fallback"`
- one-based `rank`

On current main, `FallbackCandidate` subclasses Issue 2B's shared `Candidate`
and adds `"fallback"` to its `sources` set. It therefore retains the shared
lexical/dense/fusion fields while exposing the fallback-specific fields above.
The module uses a minimal local base only when imported from a pre-2B branch,
which preserves the issues' parallel-development contract.

`adapt_fallback_candidates(candidates)` returns dictionaries containing the
four fallback fields. Passing Issue 2B's `Candidate` class as the second
argument returns the already-compatible fallback candidate objects without
discarding their fallback score or rank.

## Configuration

`FallbackConfig` exposes all scoring and diversity behavior:

- `evidence_weights`: relative category, color, style, material, use-case, and
  price boosts;
- `source_weights`: current-turn, conversation-history, and profile strength;
- `rating_weight` and `popularity_weight`: catalog-quality priors;
- `exclusion_penalty`: soft penalty for explicit exclusions;
- `diversity_dimensions`: any ordered subset of `product_family`, `category`,
  `brand`, `style`, and `price_range`;
- `diversity_caps`: maximum uses of the same known value while alternatives
  remain;
- `diversity_penalties`: repeated-value penalties used by greedy selection;
- `price_bucket_width`: deterministic price-range bucket size.

Caps are applied only where metadata exists. If every remaining product would
violate a cap, selection relaxes the caps and fills from the remaining catalog,
so requested counts are enforced up to the valid candidate pool size.

To disable diversification while retaining deterministic score ordering:

```python
from starter.fallback_candidates import FallbackConfig

config = FallbackConfig(diversity_dimensions=())
```
