from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from starter.agent import Agent
from starter.ambiguity_analysis import AttributeValueStatistics
from starter.clarification_controller import (
    ClarificationController,
    ClarificationControllerConfig,
    ClarificationSessionState,
)
from starter.clarification_fallback import (
    catalog_utility_fallback_policy,
    choose_fallback_attribute,
    disabled_fallback_policy,
    open_fallback_policy,
)
from starter.contextual_retrieval import policy_by_id
from starter.hybrid_retrieval import HybridRetrievalConfig, RetrievalMode
from starter.selective_clarification import SelectiveClarificationConfig


class _Retriever:
    def __init__(self, identifiers: list[str]) -> None:
        self.identifiers = identifiers

    def retrieve(self, *_args: object, **kwargs: object) -> list[object]:
        limit = int(kwargs.get("top_n", len(self.identifiers)))
        return [
            SimpleNamespace(parent_asin=value, score=1.0 / rank, rank=rank)
            for rank, value in enumerate(self.identifiers[:limit], 1)
        ]


class _Router:
    def __init__(self, route: str = "uncertain") -> None:
        self.route_name = route

    def route(self, *_args: object, **_kwargs: object) -> object:
        return SimpleNamespace(route=self.route_name)


class _FailingAnalyzer:
    def attribute_statistics(self, *_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected fallback analysis failure")


def _stat(attribute: str, reduction: float) -> AttributeValueStatistics:
    return AttributeValueStatistics(
        attribute=attribute,
        candidate_count=10,
        usable_count=10,
        coverage=1.0,
        value_counts=(("a", 5), ("b", 5)),
        dominant_share=0.5,
        normalized_entropy=1.0,
        expected_reduction=reduction,
    )


class ClarificationFallbackPolicyTest(unittest.TestCase):
    def test_disabled_policy_is_inert_and_fingerprints_are_distinct(self) -> None:
        disabled = disabled_fallback_policy()
        opened = open_fallback_policy()
        utility = catalog_utility_fallback_policy()
        state = ClarificationSessionState()

        decision = choose_fallback_attribute(
            disabled, (_stat("feature", 0.7),), state, None
        )

        self.assertIsNone(decision.attribute)
        self.assertEqual(
            len(
                {
                    disabled.fingerprint_sha256,
                    opened.fingerprint_sha256,
                    utility.fingerprint_sha256,
                }
            ),
            3,
        )

    def test_open_requires_strong_unresolved_ambiguity_and_open_channel(self) -> None:
        policy = open_fallback_policy()
        eligible = choose_fallback_attribute(
            policy,
            (_stat("feature", 0.7),),
            ClarificationSessionState(),
            None,
        )
        weak = choose_fallback_attribute(
            policy,
            (_stat("feature", 0.47),),
            ClarificationSessionState(),
            None,
        )
        asked = choose_fallback_attribute(
            policy,
            (_stat("feature", 0.7),),
            ClarificationSessionState(asked_attributes=frozenset({"other"})),
            None,
        )

        self.assertEqual(eligible.attribute, "other")
        self.assertIsNone(weak.attribute)
        self.assertIsNone(asked.attribute)

    def test_catalog_utility_uses_answerability_yield_and_exclusions(self) -> None:
        policy = catalog_utility_fallback_policy()
        decision = choose_fallback_attribute(
            policy,
            (_stat("material", 0.5), _stat("feature", 0.5)),
            ClarificationSessionState(),
            None,
        )
        excluded = choose_fallback_attribute(
            policy,
            (_stat("material", 0.5), _stat("feature", 0.5)),
            ClarificationSessionState(declined_attributes=frozenset({"feature"})),
            None,
        )

        self.assertEqual(decision.attribute, "feature")
        self.assertEqual(excluded.attribute, "material")

    def test_sufficiently_specified_pending_and_missing_metadata_are_ineligible(
        self,
    ) -> None:
        policy = catalog_utility_fallback_policy()
        specified = SimpleNamespace(
            color=object(),
            material=object(),
            style=object(),
            use_case=object(),
            price=object(),
        )
        known = choose_fallback_attribute(
            policy,
            (
                _stat("color", 0.7),
                _stat("material", 0.7),
                _stat("style", 0.7),
                _stat("use_case", 0.7),
                _stat("price", 0.7),
            ),
            ClarificationSessionState(),
            specified,
        )
        pending = choose_fallback_attribute(
            policy,
            (_stat("feature", 0.7),),
            ClarificationSessionState(pending_attribute="feature"),
            None,
        )
        missing = choose_fallback_attribute(
            policy, (), ClarificationSessionState(), None
        )

        self.assertIsNone(known.attribute)
        self.assertIsNone(pending.attribute)
        self.assertIsNone(missing.attribute)


class ClarificationFallbackIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.catalog = Path(self.directory.name) / "catalog.jsonl"
        self.identifiers = [f"P{index:03d}" for index in range(30)]
        rows = [
            {
                "parent_asin": parent_asin,
                "title": f"canvas shoes {index}",
                "categories": ["Clothing", "Shoes"],
                "color": "red" if index % 2 else "blue",
                "material": "canvas" if index % 3 else "leather",
                "features": [f"feature family {index % 5}"],
                "price": 40 + 20 * (index % 3),
            }
            for index, parent_asin in enumerate(self.identifiers)
        ]
        self.catalog.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def _agent(
        self,
        policy: object,
        *,
        router: _Router | None = None,
        analyzer: object | None = None,
        identifiers: list[str] | None = None,
    ) -> Agent:
        values = self.identifiers if identifiers is None else identifiers
        return Agent(
            self.catalog,
            config=HybridRetrievalConfig(mode=RetrievalMode.CONTEXTUAL),
            anchor_retriever=_Retriever(values),
            dense_retriever=_Retriever([]),
            router=router or _Router(),
            contextual_policy=policy_by_id("contextual.category-evidence.v1"),
            clarification_config=SelectiveClarificationConfig(
                enabled=True,
                required_retrieval_policy_id="contextual.category-evidence.v1",
                eligible_routes=("browsing", "boundary", "buying"),
                question_candidates=("other", "feature"),
                answerability_rates=(("other", 1.0), ("feature", 0.99294)),
                utility_min_candidates=4,
            ),
            clarification_controller=ClarificationController(
                ClarificationControllerConfig(max_questions_per_session=2)
            ),
            ambiguity_analyzer=analyzer,
            clarification_fallback_policy=policy,  # type: ignore[arg-type]
        )

    @staticmethod
    def _recommendations(response: dict[str, object]) -> object:
        return response["recommendations"]

    def _advance_to_fallback(self, agent: Agent, session_id: str) -> dict:
        agent.reset(session_id, {})
        agent.respond(session_id, "I'm browsing shoes", 1, 10)
        agent.respond(session_id, "Those options are not quite right yet.", 2, 10)
        return agent.respond(
            session_id, "Those options are not quite right yet.", 3, 10
        )

    def test_open_fallback_preserves_p5_recommendations_and_usage(self) -> None:
        baseline = self._agent(disabled_fallback_policy())
        candidate = self._agent(open_fallback_policy())
        expected = self._advance_to_fallback(baseline, "baseline")
        actual = self._advance_to_fallback(candidate, "candidate")

        self.assertEqual(actual["ask_attribute"], "other")
        self.assertEqual(self._recommendations(actual), self._recommendations(expected))
        self.assertEqual(actual["usage"], expected["usage"])
        self.assertEqual(
            candidate.clarification_fallback_diagnostics_snapshot()[
                "intervention_count"
            ],
            1,
        )

    def test_utility_fallback_is_one_time_and_session_isolated(self) -> None:
        agent = self._agent(catalog_utility_fallback_policy())
        first = self._advance_to_fallback(agent, "one")
        agent.respond("one", "feature family 2", 4, 10)
        repeated = agent.respond("one", "Those options are not quite right yet.", 5, 10)
        isolated = self._advance_to_fallback(agent, "two")

        self.assertIn(first["ask_attribute"], {"feature", "material", "color"})
        self.assertIsNone(repeated["ask_attribute"])
        self.assertIsNotNone(isolated["ask_attribute"])
        self.assertEqual(
            agent.clarification_fallback_diagnostics_snapshot()[
                "affected_session_count"
            ],
            2,
        )

    def test_turn_window_nonnegative_message_and_empty_pool_are_ineligible(
        self,
    ) -> None:
        agent = self._agent(open_fallback_policy())
        agent.reset("bounds", {})
        early = agent.respond("bounds", "Those options are not quite right yet.", 2, 10)
        plain = agent.respond("bounds", "Please show more shoes", 3, 10)
        late = agent.respond("bounds", "Those options are not quite right yet.", 10, 10)
        empty = self._agent(open_fallback_policy(), identifiers=[])
        empty.reset("empty", {})
        empty_response = empty.respond(
            "empty", "Those options are not quite right yet.", 3, 0
        )

        self.assertIsNone(early["ask_attribute"])
        self.assertIsNone(plain["ask_attribute"])
        self.assertIsNone(late["ask_attribute"])
        self.assertIsNone(empty_response["ask_attribute"])

    def test_pending_declined_and_route_changes_do_not_repeat(self) -> None:
        router = _Router("uncertain")
        agent = self._agent(open_fallback_policy(), router=router)
        first = self._advance_to_fallback(agent, "route")
        router.route_name = "buying"
        declined = agent.respond(
            "route", "I don't have a preference; use your judgment.", 4, 10
        )
        later = agent.respond("route", "Those options are not quite right yet.", 5, 10)

        self.assertEqual(first["ask_attribute"], "other")
        self.assertNotEqual(declined["ask_attribute"], "other")
        self.assertNotEqual(later["ask_attribute"], "other")
        self.assertEqual(
            agent.clarification_fallback_diagnostics_snapshot()["intervention_count"],
            1,
        )

    def test_malformed_answer_does_not_fail_or_repeat_fallback(self) -> None:
        agent = self._agent(open_fallback_policy())
        first = self._advance_to_fallback(agent, "malformed")
        malformed = agent.respond("malformed", "{not valid answer}", 4, 10)
        later = agent.respond(
            "malformed", "Those options are not quite right yet.", 5, 10
        )

        self.assertEqual(first["ask_attribute"], "other")
        self.assertIsNone(malformed["ask_attribute"])
        self.assertIsNone(later["ask_attribute"])
        self.assertEqual(
            agent.clarification_fallback_diagnostics_snapshot()["failure_count"],
            0,
        )

    def test_missing_metadata_and_failure_fall_back_to_exact_p5(self) -> None:
        baseline = self._agent(disabled_fallback_policy())
        failing = self._agent(open_fallback_policy(), analyzer=_FailingAnalyzer())
        expected = self._advance_to_fallback(baseline, "base")
        actual = self._advance_to_fallback(failing, "failure")

        self.assertEqual(actual, expected)
        diagnostics = failing.clarification_fallback_diagnostics_snapshot()
        self.assertEqual(diagnostics["failure_count"], 1)
        self.assertEqual(
            failing.diagnostics_snapshot()["component_failure_counts"],
            {"clarification_fallback": 1},
        )

    def test_disabled_outputs_are_deterministic(self) -> None:
        outputs = []
        for index in range(2):
            agent = self._agent(disabled_fallback_policy())
            outputs.append(self._advance_to_fallback(agent, f"repeat-{index}"))
        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
