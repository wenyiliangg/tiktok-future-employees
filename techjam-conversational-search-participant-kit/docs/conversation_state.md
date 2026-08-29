# Conversation state and active queries

`starter.conversation_state` provides the deterministic state layer used before
catalog retrieval. It has no model, catalog, or scoring dependency.

Create one `ConversationStateManager`, call `reset(session_id, user_profile)` at
the start of every evaluation session, and call `update(session_id, message,
turn)` for each customer turn. `update` returns the current `SearchQuery`;
`query_for` rebuilds that query without changing state.

## Update rules

- Current customer preferences take precedence over earlier conversation values,
  which take precedence over profile values.
- Customer preferences are hard by default. Hedged preferences are soft, and all
  profile-derived preferences are soft.
- Compatible values accumulate. A new value for the same slot replaces the old
  value.
- A category or explicit intent change clears intent-bound style and use-case
  values while keeping portable color, material, and price preferences.
- Negated values are removed from positive slots and stored in `exclusions`.
  Explicitly removed constraints are recorded in `removed_constraints` and do
  not appear in the generated query.
- A later positive request for an excluded value removes that exclusion.
- Query text is rebuilt in a fixed field order from active values only. It never
  concatenates raw turns.

The supported structured slots are category, price, style, color, material, and
use case. Extraction uses a finite alias table and price patterns so unsupported
preferences are not invented.
