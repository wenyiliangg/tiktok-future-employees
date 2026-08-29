from __future__ import annotations

import unittest

from starter.conversation_state import ConversationStateManager, SearchQuery


class ConversationStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = ConversationStateManager()
        self.manager.reset("session", {})

    def test_compatible_preferences_accumulate(self) -> None:
        self.manager.update("session", "I need sneakers", 1)
        self.manager.update("session", "Something white", 2)
        query = self.manager.update("session", "Under $100", 3)

        self.assertEqual(query.category.value, "sneakers")
        self.assertEqual(query.category.source, "conversation")
        self.assertEqual(query.color.value, "white")
        self.assertEqual(query.color.source, "conversation")
        self.assertEqual(query.price.maximum, 100.0)
        self.assertEqual(query.price.source, "current_turn")
        self.assertEqual(query.text, "sneakers white under $100")

    def test_all_supported_slots_include_constraint_metadata(self) -> None:
        query = self.manager.update(
            "session",
            "I need casual white leather sneakers for running between $50 and $100",
            4,
        )

        self.assertIsInstance(query, SearchQuery)
        self.assertEqual(query.category.value, "sneakers")
        self.assertEqual(query.style.value, "casual")
        self.assertEqual(query.color.value, "white")
        self.assertEqual(query.material.value, "leather")
        self.assertEqual(query.use_case.value, "running")
        self.assertEqual(query.price.minimum, 50.0)
        self.assertEqual(query.price.maximum, 100.0)
        for constraint in (
            query.category,
            query.style,
            query.color,
            query.material,
            query.use_case,
            query.price,
        ):
            self.assertEqual(constraint.strength, "hard")
            self.assertEqual(constraint.source, "current_turn")
            self.assertEqual(constraint.updated_turn, 4)

        next_query = self.manager.update("session", "Make the color blue", 5)
        self.assertEqual(next_query.category.source, "conversation")
        self.assertEqual(next_query.category.updated_turn, 4)
        self.assertEqual(next_query.color.source, "current_turn")
        self.assertEqual(next_query.color.updated_turn, 5)

    def test_category_and_preference_override_removes_obsolete_intent(self) -> None:
        self.manager.update("session", "Black running shoes", 1)
        query = self.manager.update("session", "Actually, white casual sneakers", 2)

        self.assertEqual(query.category.value, "sneakers")
        self.assertEqual(query.color.value, "white")
        self.assertEqual(query.style.value, "casual")
        self.assertIsNone(query.use_case)
        self.assertNotIn("black", query.text)
        self.assertNotIn("running", query.text)
        removed = self.manager.state_for("session").removed_constraints
        self.assertIn("color:black", removed)
        self.assertIn("use_case:running", removed)

    def test_category_change_clears_intent_bound_slots_only(self) -> None:
        self.manager.update("session", "I need black leather running shoes under $90", 1)
        query = self.manager.update("session", "Actually I need a bag instead", 2)

        self.assertEqual(query.category.value, "bag")
        self.assertEqual(query.color.value, "black")
        self.assertEqual(query.material.value, "leather")
        self.assertEqual(query.price.maximum, 90.0)
        self.assertIsNone(query.use_case)

    def test_price_removal(self) -> None:
        self.manager.update("session", "Shoes under $80", 1)
        query = self.manager.update("session", "Budget is no longer important", 2)

        self.assertIsNone(query.price)
        self.assertNotIn("80", query.text)
        self.assertIn("price", self.manager.state_for("session").removed_constraints)

    def test_unsupported_feature_removal_is_tracked_without_becoming_a_slot(self) -> None:
        query = self.manager.update("session", "I don't need waterproofing anymore", 1)

        self.assertEqual(query.text, "")
        self.assertIn(
            "feature:waterproofing",
            self.manager.state_for("session").removed_constraints,
        )

    def test_material_negation_and_replacement(self) -> None:
        self.manager.update("session", "I like leather bags", 1)
        query = self.manager.update("session", "Not leather—canvas instead", 2)

        self.assertEqual(query.material.value, "canvas")
        self.assertEqual(query.exclusions, {"material": {"leather"}})
        self.assertNotIn("leather", query.text)

    def test_negation_preserves_unrelated_preferences(self) -> None:
        self.manager.update("session", "I want black leather boots", 1)
        query = self.manager.update("session", "Anything except black", 2)

        self.assertIsNone(query.color)
        self.assertEqual(query.material.value, "leather")
        self.assertEqual(query.category.value, "boots")
        self.assertEqual(query.exclusions, {"color": {"black"}})

    def test_later_positive_request_cancels_matching_exclusion(self) -> None:
        self.manager.update("session", "Anything except black", 1)
        query = self.manager.update("session", "Actually black is fine", 2)

        self.assertEqual(query.color.value, "black")
        self.assertIsNone(query.exclusions)

    def test_profile_preference_is_soft_and_current_request_wins(self) -> None:
        manager = ConversationStateManager()
        initial = manager.reset(
            "profile-session",
            {"preference_tags": ["black"], "summary": "Prefers black products."},
        )
        self.assertEqual(initial.color.value, "black")
        self.assertEqual(initial.color.strength, "soft")
        self.assertEqual(initial.color.source, "profile")

        query = manager.update("profile-session", "Show me white sneakers", 1)
        self.assertEqual(query.color.value, "white")
        self.assertEqual(query.color.strength, "hard")
        self.assertEqual(query.color.source, "current_turn")
        self.assertNotIn("black", query.text)

    def test_profile_does_not_invent_a_value_from_generic_tags(self) -> None:
        manager = ConversationStateManager()
        manager.reset(
            "profile-session",
            {
                "preference_tags": ["material", "style", "comfort"],
                "summary": "Prior purchases emphasize material and style.",
            },
        )

        state = manager.state_for("profile-session")
        self.assertIsNone(state.material)
        self.assertIsNone(state.style)

    def test_ambiguous_language_does_not_invent_a_use_case(self) -> None:
        query = self.manager.update("session", "A white one would work", 1)

        self.assertEqual(query.color.value, "white")
        self.assertIsNone(query.use_case)

    def test_unknown_customer_preferences_are_not_invented(self) -> None:
        query = self.manager.update(
            "session",
            "Something comfortable with good support would be nice",
            1,
        )

        self.assertEqual(query.text, "")
        self.assertIsNone(query.category)
        self.assertIsNone(query.price)

    def test_query_generation_is_deterministic(self) -> None:
        managers = (ConversationStateManager(), ConversationStateManager())
        messages = ("I need sneakers", "Something white", "Under $100")
        results = []
        for manager in managers:
            manager.reset("same", {"preference_tags": ["leather"]})
            for turn, message in enumerate(messages, start=1):
                result = manager.update("same", message, turn)
            results.append(result)

        self.assertEqual(results[0], results[1])

    def test_reset_replaces_all_previous_session_state(self) -> None:
        self.manager.update("session", "No black leather shoes under $50", 1)
        self.assertTrue(self.manager.state_for("session").exclusions)

        state = self.manager.reset("session", {})
        query = self.manager.query_for("session")

        self.assertIsNone(state.category)
        self.assertIsNone(state.material)
        self.assertIsNone(state.price)
        self.assertEqual(state.exclusions, {})
        self.assertEqual(state.removed_constraints, set())
        self.assertEqual(query.text, "")
        self.assertIsNone(query.exclusions)

    def test_sessions_are_isolated(self) -> None:
        manager = ConversationStateManager()
        manager.reset("one", {})
        manager.reset("two", {})
        manager.update("one", "white sneakers", 1)

        self.assertEqual(manager.query_for("one").text, "sneakers white")
        self.assertEqual(manager.query_for("two").text, "")

    def test_state_snapshots_cannot_mutate_managed_state(self) -> None:
        self.manager.update("session", "Anything except black", 1)
        snapshot = self.manager.state_for("session")
        snapshot.exclusions["color"].add("white")

        self.assertEqual(
            self.manager.query_for("session").exclusions,
            {"color": {"black"}},
        )

    def test_price_range_is_normalized(self) -> None:
        query = self.manager.update("session", "A coat between $120 and $80", 1)

        self.assertEqual(query.price.minimum, 80.0)
        self.assertEqual(query.price.maximum, 120.0)
        self.assertIn("$80 to $120", query.text)

    def test_update_requires_reset(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "reset must be called"):
            ConversationStateManager().update("missing", "white shoes", 1)


if __name__ == "__main__":
    unittest.main()
