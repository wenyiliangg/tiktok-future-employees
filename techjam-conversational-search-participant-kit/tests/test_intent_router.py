from __future__ import annotations

from types import SimpleNamespace
import unittest

from starter.conversation_state import (
    Constraint,
    ConversationStateManager,
    PriceConstraint,
    SearchQuery,
    SessionState,
)
from starter.intent_router import IntentRouter, RouterConfig, RoutingDecision


def slot(
    value: str,
    *,
    source: str = "current_turn",
    strength: str = "hard",
    turn: int = 1,
) -> Constraint:
    return Constraint(value=value, source=source, strength=strength, updated_turn=turn)  # type: ignore[arg-type]


class IntentRouterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.router = IntentRouter()

    def test_white_running_shoes_under_100_routes_to_buying(self) -> None:
        state = SessionState(
            category=slot("sneakers"),
            color=slot("white"),
            use_case=slot("running"),
            price=PriceConstraint(maximum=100, updated_turn=1),
        )
        query = SearchQuery(
            text="white running shoes under $100",
            category=state.category,
            color=state.color,
            use_case=state.use_case,
            price=state.price,
        )

        decision = self.router.route(state, query)

        self.assertEqual(decision.route, "buying")
        self.assertEqual(decision.policy_id, "retrieval.buying.v1")
        self.assertGreater(decision.confidence, 0.5)
        self.assertTrue(any(reason.startswith("active:price") for reason in decision.reasons))

    def test_comfortable_long_city_trip_routes_to_browsing(self) -> None:
        state = SessionState(use_case=slot("city exploration", strength="soft"))
        query = SearchQuery(text="something comfortable for a long city trip", use_case=state.use_case)

        decision = self.router.route(state, query)

        self.assertEqual(decision.route, "browsing")
        self.assertEqual(decision.policy_id, "retrieval.browsing.v1")
        self.assertIn("broad_cue:comfortable", decision.reasons)
        self.assertIn("broad_cue:trip", decision.reasons)

    def test_not_sure_what_i_want_routes_to_boundary(self) -> None:
        decision = self.router.route(
            SessionState(),
            SearchQuery(text="I'm not sure what I want"),
        )

        self.assertEqual(decision.route, "boundary")
        self.assertEqual(decision.policy_id, "retrieval.boundary.v1")
        self.assertIn("boundary_cue:not sure what i want", decision.reasons)

    def test_profile_category_alone_does_not_route_to_buying(self) -> None:
        profile_category = slot("sneakers", source="profile", strength="soft", turn=0)
        decision = self.router.route(
            SessionState(category=profile_category),
            SearchQuery(text="sneakers", category=profile_category),
        )

        self.assertEqual(decision.route, "boundary")
        self.assertNotEqual(decision.policy_id, "retrieval.buying.v1")
        self.assertIn("profile_only:category:ignored_for_buying", decision.reasons)

    def test_category_plus_explicit_uncertainty_routes_to_uncertain(self) -> None:
        category = slot("shoes")
        decision = self.router.route(
            SessionState(category=category),
            SearchQuery(text="I want shoes, but I don't know what type", category=category),
        )

        self.assertEqual(decision.route, "uncertain")
        self.assertEqual(decision.policy_id, "retrieval.safe-default.v1")
        self.assertIn("uncertainty_cue:don't know what type", decision.reasons)

    def test_repeated_routing_is_deterministic(self) -> None:
        state = SessionState(category=slot("shoes"), color=slot("white"))
        query = SearchQuery(text="white shoes", category=state.category, color=state.color)

        decisions = [self.router.route(state, query) for _ in range(5)]

        self.assertTrue(all(decision == decisions[0] for decision in decisions))

    def test_missing_and_partial_fields_are_safe(self) -> None:
        partial_state = SimpleNamespace(category=None)
        partial_query = SimpleNamespace(text=None)

        decision = self.router.route(partial_state, partial_query)
        missing_decision = self.router.route(None, None)

        self.assertEqual(decision.route, "boundary")
        self.assertEqual(missing_decision.route, "boundary")

    def test_current_constraints_outweigh_unrelated_profile_evidence(self) -> None:
        state = SessionState(
            category=slot("sneakers", source="current_turn"),
            color=slot("white", source="current_turn"),
            material=slot("leather", source="profile", strength="soft", turn=0),
        )
        query = SearchQuery(
            text="white sneakers",
            category=state.category,
            color=state.color,
            material=state.material,
        )

        decision = self.router.route(state, query)

        self.assertEqual(decision.route, "buying")
        self.assertIn("profile_only:material:ignored_for_buying", decision.reasons)
        self.assertIn("active:color:current_turn:hard", decision.reasons)

    def test_current_query_overrides_conflicting_profile_slot(self) -> None:
        profile_category = slot("sneakers", source="profile", strength="soft", turn=0)
        current_category = slot("boots", source="current_turn", turn=2)

        decision = self.router.route(
            SessionState(category=profile_category),
            SearchQuery(text="I need boots", category=current_category),
        )

        self.assertEqual(decision.route, "buying")
        self.assertIn("active:category:current_turn:hard", decision.reasons)
        self.assertNotIn("profile_only:category:ignored_for_buying", decision.reasons)

    def test_all_routes_and_route_changes_are_supported(self) -> None:
        states_and_queries = (
            (SessionState(), SearchQuery(text="show me something"), "boundary"),
            (
                SessionState(use_case=slot("city trip", strength="soft", turn=2)),
                SearchQuery(text="comfortable city trip"),
                "browsing",
            ),
            (
                SessionState(category=slot("shoes", turn=3)),
                SearchQuery(text="maybe shoes", category=slot("shoes", turn=3)),
                "uncertain",
            ),
            (
                SessionState(category=slot("shoes", turn=4), color=slot("white", turn=4)),
                SearchQuery(
                    text="white shoes",
                    category=slot("shoes", turn=4),
                    color=slot("white", turn=4),
                ),
                "buying",
            ),
        )

        routes = tuple(
            self.router.route(state, query).route
            for state, query, _expected in states_and_queries
        )

        self.assertEqual(routes, tuple(expected for _state, _query, expected in states_and_queries))

    def test_reset_between_sessions_does_not_leak_buying_route(self) -> None:
        manager = ConversationStateManager()
        manager.reset("buying", {})
        buying_query = manager.update("buying", "I need white sneakers", 1)
        buying = self.router.route(manager.state_for("buying"), buying_query)

        manager.reset("empty", {})
        empty = self.router.route(manager.state_for("empty"), manager.query_for("empty"))

        self.assertEqual(buying.route, "buying")
        self.assertEqual(empty.route, "boundary")

    def test_thresholds_and_policy_ids_are_configurable(self) -> None:
        category = slot("shoes")
        strict_router = IntentRouter(
            RouterConfig(
                buying_threshold=10.0,
                uncertain_policy_id="custom.safe",
            )
        )

        decision = strict_router.route(
            SessionState(category=category),
            SearchQuery(text="shoes", category=category),
        )

        self.assertEqual(decision.route, "uncertain")
        self.assertEqual(decision.policy_id, "custom.safe")

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "buying_threshold"):
            RouterConfig(buying_threshold=-1)

    def test_decision_contract_is_inspectable(self) -> None:
        decision = self.router.route(SessionState(), SearchQuery(text=""))

        self.assertIsInstance(decision, RoutingDecision)
        self.assertGreaterEqual(decision.confidence, 0.0)
        self.assertLessEqual(decision.confidence, 1.0)
        self.assertTrue(decision.reasons)
        self.assertTrue(decision.reasons[-1].startswith("rule:"))


if __name__ == "__main__":
    unittest.main()
