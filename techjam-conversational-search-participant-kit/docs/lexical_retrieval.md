# Field-aware lexical retrieval

`starter.lexical_retriever.LexicalRetriever` consumes the shared
`starter.search_models.SearchQuery` representation. It does not inspect
conversation history or parse natural language into slots.

## Index and scoring

The document builder reads but never mutates the catalog row. It recursively
flattens list- and mapping-valued fields and builds separate FTS5 columns for:

| Field | Default weight |
| --- | ---: |
| title | 5.0 |
| category hierarchy | 4.0 |
| features | 3.0 |
| color, material, style, use case | 3.0 each |
| remaining structured attributes | 2.5 |
| brand/store | 1.5 |
| long description | 1.0 |

The weights live in `LexicalRetrievalConfig.field_weights` and can be replaced
per retriever. Soft-match boosts and the BM25 candidate-pool size are also
configurable. SQLite FTS5 provides BM25 with its fixed `k1=1.2` and `b=0.75`
constants. Results are finally ordered by descending combined score and then by
ascending `parent_asin`, so ties are stable.

Structured constraint values are added to candidate-generation terms. The
retriever scans BM25-ranked matches until it has collected the configured number
of candidates that pass all hard filters. It then applies soft boosts and returns
at most `top_n` products.

## Constraint and missing-value policies

- Hard category, color, material, style, and use-case constraints require all
  normalized constraint terms to occur in that product's corresponding
  structured metadata. Missing metadata does not prove a positive hard match,
  so that product is filtered.
- Soft preferences use broader evidence from the relevant structured field,
  title, features, attributes, and description. Missing or nonmatching evidence
  receives no boost and does not remove the product.
- Exclusions use the named field's evidence. A known match is filtered. Missing
  evidence is allowed because absence cannot prove the excluded value is present.
  Unknown detail keys can still be used when their normalized key matches the
  exclusion field.
- If strict filtering leaves no candidates, retrieval returns `[]`. Hard
  constraints are never silently relaxed, so a known violation cannot be
  presented as a perfect fallback match.
- Invalid products with no usable `parent_asin` are skipped. If a duplicate ID
  appears, the first catalog row wins. Every result is therefore unique and
  belongs to the indexed catalog.

## Price policy

Price parsing recursively considers numeric values and numbers in strings,
rejecting negative, infinite, and malformed values. When a product has several
valid prices, the lowest available price is selected. Minimum and maximum bounds
are inclusive. A product with a missing or malformed selected price fails a hard
price constraint, while it merely receives no boost for a soft price preference.
An inverted or invalid range returns an empty result safely.

## Usage

```python
from starter.lexical_retriever import LexicalRetriever
from starter.search_models import Constraint, SearchQuery

retriever = LexicalRetriever.from_jsonl("data/catalog.jsonl")
results = retriever.retrieve(
    SearchQuery(
        text="white casual sneakers",
        category=Constraint("sneakers", "hard", "current_turn", 2),
        color=Constraint("white", "soft", "current_turn", 2),
    ),
    top_n=200,
)
```

Call `close()` when the index is no longer needed, or use the retriever as a
context manager.

## Issue 1A handoff test

`tests/test_search_query_integration.py` is the executable boundary between the
parallel issues. It checks the exact shared dataclass fields, passes a fully
structured 1A-style query directly into retrieval, and proves that sending a new
active query after an override changes ranking without the retriever retaining
old conversation state. The fixtures are manual until Issue 1A lands; at that
point, replace their construction with the real state manager's query producer
and keep the retrieval assertions unchanged.

The retriever validates query objects structurally rather than requiring exact
Python class identity. This supports the original Issue 1A commit, which defined
equivalent dataclasses in `starter.conversation_state` and added an optional
`updated_turn` field to `PriceConstraint`. The permanent integration should move
both modules onto one canonical model definition; structural compatibility keeps
the handoff functional before that cleanup.
