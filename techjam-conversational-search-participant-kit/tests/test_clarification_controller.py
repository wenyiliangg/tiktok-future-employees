from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path

from starter.agent import Agent
from starter.clarification_controller import (
    OFFICIAL_ATTRIBUTES,
    PROMPT_TEMPLATES,
    ClarificationController,
    ClarificationControllerConfig,
    ClarificationPrompt,
    compose_clarification_response,
    is_explicit_no_preference,
    normalize_attribute,
)
from starter.conversation_state import (
    Constraint,
    PriceConstraint,
    SessionState,
)


class EmptyRetriever:
    def retrieve(self, *_args: object, **_kwargs: object) -> list[object]:
        return []


class ClarificationControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = ClarificationController()
        self.controller.reset("session")

    def test_first_valid_question_records_pending_state(self) -> None:
        prompt = self.controller.build_prompt("session", "color", SessionState(), 1)

        self.assertEqual(
            prompt,
            ClarificationPrompt(
                message="Do you have a preferred color?",
                ask_attribute="color",
            ),
        )
        state = self.controller.state_for("session")
        assert state is not None
        self.assertEqual(state.asked_attributes, frozenset({"color"}))
        self.assertEqual(state.pending_attribute, "color")
        self.assertEqual(state.clarification_count, 1)

    def test_every_official_attribute_has_its_exact_template(self) -> None:
        self.assertEqual(set(PROMPT_TEMPLATES), set(OFFICIAL_ATTRIBUTES))
        for index, attribute in enumerate(OFFICIAL_ATTRIBUTES):
            with self.subTest(attribute=attribute):
                session_id = f"template-{index}"
                self.controller.reset(session_id)
                prompt = self.controller.build_prompt(
                    session_id, attribute, SessionState(), 1
                )
                self.assertEqual(
                    prompt,
                    ClarificationPrompt(PROMPT_TEMPLATES[attribute], attribute),
                )

    def test_canonical_attributes_are_normalized_deterministically(self) -> None:
        for attribute in OFFICIAL_ATTRIBUTES:
            with self.subTest(attribute=attribute):
                self.assertEqual(normalize_attribute(attribute), attribute)
                self.assertEqual(
                    normalize_attribute(f"  {attribute.upper()}  "), attribute
                )

    def test_documented_aliases_are_normalized(self) -> None:
        self.assertEqual(normalize_attribute("price"), "budget")
        self.assertEqual(normalize_attribute("price_range"), "budget")
        self.assertEqual(normalize_attribute("usecase"), "use_case")
        self.assertEqual(normalize_attribute("occasion"), "use_case")

    def test_price_aliases_emit_budget(self) -> None:
        for index, attribute in enumerate(("price", "price_range")):
            with self.subTest(attribute=attribute):
                session_id = f"price-{index}"
                self.controller.reset(session_id)
                prompt = self.controller.build_prompt(
                    session_id, attribute, SessionState(), 1
                )
                assert prompt is not None
                self.assertEqual(prompt.ask_attribute, "budget")
                self.assertEqual(prompt.message, "What price range would you prefer?")

    def test_same_normalized_attribute_is_never_asked_twice(self) -> None:
        self.assertIsNotNone(
            self.controller.build_prompt("session", "price", SessionState(), 1)
        )
        self.assertTrue(
            self.controller.record_resolution("session", "budget", "answered")
        )
        self.assertIsNone(
            self.controller.build_prompt("session", "price_range", SessionState(), 2)
        )

    def test_known_active_attribute_is_rejected_without_changing_state(self) -> None:
        active_state = SessionState(
            category=Constraint("shoes", "hard", "current_turn", 1)
        )

        self.assertIsNone(
            self.controller.build_prompt("session", "category", active_state, 2)
        )
        state = self.controller.state_for("session")
        assert state is not None
        self.assertEqual(state.clarification_count, 0)
        self.assertEqual(state.asked_attributes, frozenset())

    def test_existing_price_bounds_make_budget_known(self) -> None:
        active_state = SessionState(price=PriceConstraint(maximum=100.0))

        self.assertIsNone(
            self.controller.build_prompt("session", "price_range", active_state, 1)
        )

    def test_mapping_active_state_is_supported_for_future_official_slots(self) -> None:
        self.assertIsNone(
            self.controller.build_prompt("session", "brand", {"brand": "Example"}, 1)
        )

    def test_explicit_answer_is_recorded_separately(self) -> None:
        self.controller.build_prompt("session", "material", SessionState(), 1)

        self.assertTrue(
            self.controller.record_resolution("session", "material", "confirmed")
        )
        state = self.controller.state_for("session")
        assert state is not None
        self.assertEqual(state.answered_attributes, frozenset({"material"}))
        self.assertEqual(state.declined_attributes, frozenset())
        self.assertIsNone(state.pending_attribute)

    def test_explicit_no_preference_is_recorded_separately(self) -> None:
        self.controller.build_prompt("session", "style", SessionState(), 1)

        self.assertTrue(
            self.controller.record_resolution("session", "style", "no-preference")
        )
        state = self.controller.state_for("session")
        assert state is not None
        self.assertEqual(state.answered_attributes, frozenset())
        self.assertEqual(state.declined_attributes, frozenset({"style"}))
        self.assertEqual(state.no_preference_attributes, frozenset({"style"}))
        self.assertIsNone(state.pending_attribute)

    def test_evaluator_decline_phrase_is_explicitly_recognized(self) -> None:
        self.assertTrue(
            is_explicit_no_preference(
                "I don't have an additional preference for material."
            )
        )

    def test_unrelated_activity_is_not_automatically_an_answer(self) -> None:
        self.controller.build_prompt("session", "color", SessionState(), 1)

        self.assertIsNone(
            self.controller.build_prompt("session", "material", SessionState(), 2)
        )
        state = self.controller.state_for("session")
        assert state is not None
        self.assertEqual(state.pending_attribute, "color")
        self.assertEqual(state.answered_attributes, frozenset())
        self.assertEqual(state.declined_attributes, frozenset())

    def test_configurable_question_limit(self) -> None:
        controller = ClarificationController(
            ClarificationControllerConfig(max_questions_per_session=2)
        )
        controller.reset("two")
        self.assertIsNotNone(controller.build_prompt("two", "color", SessionState(), 1))
        controller.record_resolution("two", "color", "answered")
        self.assertIsNotNone(
            controller.build_prompt("two", "material", SessionState(), 2)
        )
        controller.record_resolution("two", "material", "declined")
        self.assertIsNone(controller.build_prompt("two", "style", SessionState(), 3))

    def test_invalid_config_is_rejected_deterministically(self) -> None:
        for config in (
            {"max_questions_per_session": -1},
            {"max_questions_per_session": True},
            {"max_turns": 0},
            {"max_turns": 11},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                ClarificationControllerConfig(**config)  # type: ignore[arg-type]

    def test_turn_nine_is_allowed_and_turn_ten_is_rejected(self) -> None:
        allowed = self.controller.build_prompt("session", "color", SessionState(), 9)
        self.assertIsNotNone(allowed)

        controller = ClarificationController()
        controller.reset("late")
        self.assertIsNone(controller.build_prompt("late", "color", SessionState(), 10))

    def test_invalid_turns_fail_without_recording_a_question(self) -> None:
        for turn in (0, -1, 10, 11, True):
            with self.subTest(turn=turn):
                self.assertIsNone(
                    self.controller.build_prompt(
                        "session", "color", SessionState(), turn
                    )
                )
        state = self.controller.state_for("session")
        assert state is not None
        self.assertEqual(state.clarification_count, 0)

    def test_reset_clears_state_and_sessions_are_isolated(self) -> None:
        self.controller.reset("other")
        self.controller.build_prompt("session", "color", SessionState(), 1)

        other = self.controller.state_for("other")
        assert other is not None
        self.assertEqual(other.clarification_count, 0)
        reset = self.controller.reset("session")
        self.assertEqual(reset.clarification_count, 0)
        self.assertEqual(reset.asked_attributes, frozenset())
        self.assertIsNone(reset.pending_attribute)

    def test_unsupported_and_invalid_sessions_fail_safely(self) -> None:
        for attribute in ("department", "use case", "", None, 7):
            with self.subTest(attribute=attribute):
                self.assertIsNone(
                    self.controller.build_prompt(
                        "session", attribute, SessionState(), 1
                    )
                )
        self.assertIsNone(
            self.controller.build_prompt("missing", "other", SessionState(), 1)
        )
        self.assertFalse(
            self.controller.record_resolution("missing", "other", "answered")
        )

    def test_other_is_only_emitted_when_explicitly_requested(self) -> None:
        self.assertIsNone(normalize_attribute("unknown"))
        prompt = self.controller.build_prompt("session", "other", SessionState(), 1)
        self.assertEqual(
            prompt,
            ClarificationPrompt(
                "What matters most to you when choosing?",
                "other",
            ),
        )

    def test_output_is_deterministic(self) -> None:
        outputs = []
        for index in range(2):
            controller = ClarificationController()
            controller.reset(f"same-{index}")
            outputs.append(
                controller.build_prompt(f"same-{index}", "occasion", SessionState(), 4)
            )
        self.assertEqual(outputs[0], outputs[1])

    def test_at_most_one_question_can_be_emitted_for_a_turn(self) -> None:
        controller = ClarificationController(
            ClarificationControllerConfig(max_questions_per_session=3)
        )
        controller.reset("one-response")
        self.assertIsNotNone(
            controller.build_prompt("one-response", "color", SessionState(), 3)
        )
        controller.record_resolution("one-response", "color", "answered")
        self.assertIsNone(
            controller.build_prompt("one-response", "style", SessionState(), 3)
        )

    def test_later_active_state_override_does_not_reenable_asked_attribute(
        self,
    ) -> None:
        self.controller.build_prompt("session", "color", SessionState(), 1)
        self.controller.record_resolution("session", "color", "answered")

        self.assertIsNone(
            self.controller.build_prompt("session", "color", SessionState(), 2)
        )

    def test_interrupt_clears_only_pending_question(self) -> None:
        self.controller.build_prompt("session", "other", SessionState(), 1)

        self.assertEqual(self.controller.interrupt_pending("session"), "other")
        state = self.controller.state_for("session")
        assert state is not None
        self.assertIsNone(state.pending_attribute)
        self.assertEqual(state.asked_attributes, frozenset({"other"}))
        self.assertEqual(state.answered_attributes, frozenset())
        self.assertEqual(state.declined_attributes, frozenset())
        self.assertIsNone(self.controller.interrupt_pending("missing"))


class ClarificationResponseCompositionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.response: dict[str, object] = {
            "message": "Here are the closest matches I found.",
            "ask_attribute": None,
            "recommendations": [
                {"parent_asin": "B", "score": 0.9},
                {"parent_asin": "A", "score": 0.8},
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        }

    def test_prompt_composition_preserves_recommendations_order_and_usage(self) -> None:
        recommendations = json.loads(json.dumps(self.response["recommendations"]))
        original_usage = self.response["usage"]
        assert isinstance(original_usage, Mapping)
        usage = dict(original_usage)

        composed = compose_clarification_response(
            self.response,
            ClarificationPrompt("Do you have a preferred color?", "color"),
        )

        self.assertEqual(composed["recommendations"], recommendations)
        self.assertEqual(composed["usage"], usage)
        self.assertEqual(composed["message"], "Do you have a preferred color?")
        self.assertEqual(composed["ask_attribute"], "color")

    def test_composition_does_not_mutate_or_alias_the_caller_response(self) -> None:
        original = json.loads(json.dumps(self.response))
        composed = compose_clarification_response(self.response, None)
        self.assertEqual(self.response, original)
        self.assertEqual(composed["message"], original["message"])
        self.assertIsNone(composed["ask_attribute"])

        recommendations = composed["recommendations"]
        assert isinstance(recommendations, list)
        recommendations.append({"parent_asin": "C"})
        self.assertEqual(self.response, original)

    def test_composed_fragment_matches_official_response_contract(self) -> None:
        contract_path = Path(__file__).parents[1] / "docs" / "agent_api_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        response_contract = contract["turn_response"]
        ask_contract = response_contract["properties"]["ask_attribute"]
        self.assertEqual(
            set(ask_contract["enum"]),
            {*OFFICIAL_ATTRIBUTES, None},
        )

        composed = compose_clarification_response(
            self.response,
            ClarificationPrompt(PROMPT_TEMPLATES["feature"], "feature"),
        )
        self.assertTrue(set(response_contract["required"]).issubset(composed))
        self.assertEqual(composed["ask_attribute"], "feature")
        self.assertIsInstance(composed["message"], str)


class DefaultAgentClarificationTest(unittest.TestCase):
    def test_selected_default_handles_empty_pool_without_question(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            catalog_path.write_text("", encoding="utf-8")
            agent = Agent(
                catalog_path,
                anchor_retriever=EmptyRetriever(),
                dense_retriever=EmptyRetriever(),
            )
            agent.reset("default", {})

            response = agent.respond("default", "I need sneakers", 1, 10)

        self.assertIsNone(response["ask_attribute"])
        self.assertEqual(response["message"], "Here are the closest matches I found.")
        self.assertTrue(agent.clarification_diagnostics_snapshot()["enabled"])


if __name__ == "__main__":
    unittest.main()
