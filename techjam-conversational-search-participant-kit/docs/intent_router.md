# Intent router

`starter.intent_router` deterministically classifies active conversation state
and selects a retrieval-policy identifier. It does not execute retrieval or
change ranking scores.

## Routes

- **Buying:** concrete, non-profile constraints meet the configured buying
  threshold and outweigh broad browsing evidence.
- **Browsing:** broad goals or use cases meet the browsing threshold without
  enough narrow constraints for Buying.
- **Boundary:** no usable active intent is present, or an explicit no-preference
  phrase appears without active constraints.
- **Uncertain:** active evidence is weak, explicitly uncertain, or conflicting.
  Its policy identifier is a safe-default policy.

## Evidence and precedence

The router consumes `SessionState` and `SearchQuery`, never concatenated turn
history. Current-turn and active conversation constraints contribute weighted
evidence. Profile-only slots are reported in `RoutingDecision.reasons` but do
not contribute toward Buying. Current explicit constraints therefore outweigh
profile defaults already resolved by the conversation-state layer.

Every decision contains the matched evidence, Buying and Browsing scores, and
the final rule. Repeated inputs produce identical decisions.

## Configuration

`RouterConfig` exposes route thresholds, conflict margin, evidence weights,
source/strength multipliers, phrase lists, and the four policy identifiers. Pass
an instance to `IntentRouter(config)` to change policy without modifying the
router implementation.

