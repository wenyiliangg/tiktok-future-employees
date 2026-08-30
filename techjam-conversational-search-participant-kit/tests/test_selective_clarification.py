from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from starter.agent import Agent
from starter.ambiguity_analysis import ClarificationOpportunity
from starter.clarification_controller import ClarificationController
from starter.hybrid_retrieval import HybridRetrievalConfig, RetrievalMode
from starter.selective_clarification import SelectiveClarificationConfig


class CountingRetriever:
    def __init__(self, results: list[object]) -> None:
        self.results = results
        self.calls = 0

    def retrieve(self, *_args: object, **kwargs: object) -> list[object]:
        self.calls += 1
        top_n = int(kwargs.get("top_n", len(self.results)))
        return self.results[:top_n]


class FixedRouter:
    def __init__(self, route: str) -> None:
        self.selected_route = route

    def route(self, *_args: object, **_kwargs: object) -> object:
        return SimpleNamespace(route=self.selected_route)


class RecordingAnalyzer:
    def __init__(
        self,
        opportunity: ClarificationOpportunity | None = None,
        error: Exception | None = None,
    ) -> None:
        self.opportunity = opportunity or ClarificationOpportunity(
            False, None, (), 0.0, "none"
        )
        self.error = error
        self.candidate_counts: list[int] = []

    def analyze(self, candidates: object, *_args: object) -> ClarificationOpportunity:
        pool = list(candidates)  # type: ignore[arg-type]
        self.candidate_counts.append(len(pool))
        if self.error is not None:
            raise self.error
        return self.opportunity


class FailingController(ClarificationController):
    def build_prompt(self, *_args: object, **_kwargs: object) -> object:
        raise RuntimeError("controller failed")


def retrieval_results(parent_asins: list[str]) -> list[object]:
    return [
        SimpleNamespace(parent_asin=parent_asin, score=1.0 / rank, rank=rank)
        for rank, parent_asin in enumerate(parent_asins, start=1)
    ]


def response_asins(response: dict[str, object]) -> list[str]:
    recommendations = response["recommendations"]
    assert isinstance(recommendations, list)
    return [str(item["parent_asin"]) for item in recommendations]


class SelectiveClarificationIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.catalog_path = Path(self.temporary_directory.name) / "catalog.jsonl"

    def write_catalog(
        self,
        count: int = 10,
        *,
        red_count: int | None = None,
        dominant: bool = False,
    ) -> list[str]:
        parent_asins = [f"P{index:03d}" for index in range(count)]
        if red_count is None:
            red_count = count // 2
        rows = []
        for index, parent_asin in enumerate(parent_asins):
            color = "black" if dominant else ("red" if index < red_count else "blue")
            rows.append(
                {
                    "parent_asin": parent_asin,
                    "title": f"{color} canvas shoes {index}",
                    "categories": ["Clothing", "Shoes"],
                    "color": color,
                    "material": "canvas",
                    "price": 60,
                }
            )
        self.catalog_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return parent_asins

    def build_agent(
        self,
        parent_asins: list[str],
        *,
        enabled: bool,
        route: str = "browsing",
        analyzer: object | None = None,
        controller: object | None = None,
        analysis_candidate_limit: int = 50,
    ) -> tuple[Agent, CountingRetriever, CountingRetriever]:
        anchor = CountingRetriever(retrieval_results(parent_asins))
        dense = CountingRetriever([])
        agent = Agent(
            self.catalog_path,
            config=HybridRetrievalConfig(mode=RetrievalMode.CONTEXTUAL),
            anchor_retriever=anchor,
            dense_retriever=dense,
            router=FixedRouter(route),
            clarification_config=SelectiveClarificationConfig(
                enabled=enabled,
                analysis_candidate_limit=analysis_candidate_limit,
            ),
            ambiguity_analyzer=analyzer,
            clarification_controller=controller,
        )
        return agent, anchor, dense

    def test_feature_flag_off_has_exact_champion_response_parity(self) -> None:
        parent_asins = self.write_catalog(10)
        exploding = RecordingAnalyzer(error=AssertionError("must not be invoked"))
        baseline, _, _ = self.build_agent(parent_asins, enabled=False)
        flagged_off, _, _ = self.build_agent(
            parent_asins, enabled=False, analyzer=exploding
        )
        baseline.reset("same", {})
        flagged_off.reset("same", {})

        expected = baseline.respond("same", "I'm browsing shoes", 1, 10)
        actual = flagged_off.respond("same", "I'm browsing shoes", 1, 10)

        self.assertEqual(actual, expected)
        self.assertEqual(exploding.candidate_counts, [])

    def test_ambiguous_browsing_pool_attaches_one_question_unchanged_results(
        self,
    ) -> None:
        parent_asins = self.write_catalog(10)
        off, _, _ = self.build_agent(parent_asins, enabled=False)
        enabled, anchor, dense = self.build_agent(parent_asins, enabled=True)
        off.reset("off", {})
        enabled.reset("enabled", {})

        baseline = off.respond("off", "I'm browsing shoes", 1, 10)
        response = enabled.respond("enabled", "I'm browsing shoes", 1, 10)

        self.assertEqual(response_asins(response), response_asins(baseline))
        self.assertEqual(response["usage"], baseline["usage"])
        self.assertEqual(response["ask_attribute"], "color")
        self.assertEqual(anchor.calls, 1)
        self.assertEqual(dense.calls, 1)

    def test_narrow_or_dominant_pool_returns_recommendations_only(self) -> None:
        parent_asins = self.write_catalog(10, dominant=True)
        agent, _, _ = self.build_agent(parent_asins, enabled=True)
        agent.reset("narrow", {})

        response = agent.respond("narrow", "I'm browsing shoes", 1, 10)

        self.assertIsNone(response["ask_attribute"])
        self.assertEqual(len(response_asins(response)), 10)

    def test_buying_uses_stricter_usefulness_threshold(self) -> None:
        parent_asins = self.write_catalog(10, red_count=7)
        browsing, _, _ = self.build_agent(parent_asins, enabled=True, route="browsing")
        buying, _, _ = self.build_agent(parent_asins, enabled=True, route="buying")
        browsing.reset("browse", {})
        buying.reset("buy", {})

        browsing_response = browsing.respond("browse", "I need shoes", 1, 10)
        buying_response = buying.respond("buy", "I need shoes", 1, 10)

        self.assertEqual(browsing_response["ask_attribute"], "color")
        self.assertIsNone(buying_response["ask_attribute"])

    def test_analyzer_sees_at_most_configured_first_fifty_candidates(self) -> None:
        parent_asins = self.write_catalog(70)
        analyzer = RecordingAnalyzer()
        agent, _, _ = self.build_agent(
            parent_asins,
            enabled=True,
            analyzer=analyzer,
            analysis_candidate_limit=50,
        )
        agent.reset("bounded", {})

        agent.respond("bounded", "I'm browsing shoes", 1, 10)

        self.assertEqual(analyzer.candidate_counts, [50])

    def test_known_attribute_is_not_asked(self) -> None:
        parent_asins = self.write_catalog(10)
        agent, _, _ = self.build_agent(parent_asins, enabled=True)
        agent.reset("known", {})

        response = agent.respond("known", "I'm browsing red shoes", 1, 10)

        self.assertIsNone(response["ask_attribute"])

    def test_pending_answered_declined_and_asked_attributes_are_not_repeated(
        self,
    ) -> None:
        parent_asins = self.write_catalog(10)
        controller = ClarificationController()
        agent, _, _ = self.build_agent(
            parent_asins, enabled=True, controller=controller
        )
        agent.reset("answered", {})
        first = agent.respond("answered", "I'm browsing shoes", 1, 10)
        unrelated = agent.respond("answered", "Show me shoes", 2, 10)
        pending = controller.state_for("answered")
        answered_response = agent.respond("answered", "blue", 3, 10)
        answered = controller.state_for("answered")

        self.assertEqual(first["ask_attribute"], "color")
        self.assertIsNone(unrelated["ask_attribute"])
        assert pending is not None
        self.assertEqual(pending.pending_attribute, "color")
        self.assertEqual(pending.answered_attributes, frozenset())
        self.assertIsNone(answered_response["ask_attribute"])
        assert answered is not None
        self.assertEqual(answered.answered_attributes, frozenset({"color"}))
        self.assertIsNone(answered.pending_attribute)

        agent.reset("declined", {})
        agent.respond("declined", "I'm browsing shoes", 1, 10)
        declined_response = agent.respond("declined", "Anything is fine", 2, 10)
        declined = controller.state_for("declined")
        self.assertIsNone(declined_response["ask_attribute"])
        assert declined is not None
        self.assertEqual(declined.declined_attributes, frozenset({"color"}))

    def test_question_limit_and_session_isolation(self) -> None:
        parent_asins = self.write_catalog(10)
        agent, _, _ = self.build_agent(parent_asins, enabled=True)
        agent.reset("one", {})
        agent.reset("two", {})

        first = agent.respond("one", "I'm browsing shoes", 1, 10)
        agent.respond("one", "blue", 2, 10)
        limited = agent.respond("one", "Actually, red instead", 3, 10)
        isolated = agent.respond("two", "I'm browsing shoes", 1, 10)

        self.assertEqual(first["ask_attribute"], "color")
        self.assertIsNone(limited["ask_attribute"])
        self.assertEqual(isolated["ask_attribute"], "color")

    def test_no_question_is_asked_on_turn_ten(self) -> None:
        parent_asins = self.write_catalog(10)
        agent, _, _ = self.build_agent(parent_asins, enabled=True)
        agent.reset("late", {})

        response = agent.respond("late", "I'm browsing shoes", 10, 10)

        self.assertIsNone(response["ask_attribute"])

    def test_zero_top_k_does_not_reuse_an_earlier_analysis_pool(self) -> None:
        parent_asins = self.write_catalog(10)
        agent, _, _ = self.build_agent(parent_asins, enabled=True)
        agent.reset("zero", {})
        agent.respond("zero", "I'm browsing shoes", 1, 10)

        response = agent.respond("zero", "blue", 2, 0)

        self.assertEqual(response_asins(response), [])
        self.assertIsNone(response["ask_attribute"])

    def test_explicit_answer_updates_only_issue_1a_state_and_preserves_raw_text(
        self,
    ) -> None:
        parent_asins = self.write_catalog(10)
        controller = ClarificationController()
        agent, _, _ = self.build_agent(
            parent_asins, enabled=True, controller=controller
        )
        agent.reset("answer", {})
        agent.respond("answer", "I'm browsing shoes", 1, 10)

        agent.respond("answer", "Actually, blue instead", 2, 10)

        active = agent._state.state_for("answer")
        assert active.color is not None
        self.assertEqual(active.color.value, "blue")
        self.assertEqual(active.color.source, "current_turn")
        self.assertEqual(active.color.updated_turn, 2)
        self.assertEqual(active.raw_current_turn_text, "Actually, blue instead")
        controller_state = controller.state_for("answer")
        assert controller_state is not None
        self.assertEqual(controller_state.answered_attributes, frozenset({"color"}))

    def test_explicit_decline_creates_no_preference_constraint(self) -> None:
        parent_asins = self.write_catalog(10)
        controller = ClarificationController()
        agent, _, _ = self.build_agent(
            parent_asins, enabled=True, controller=controller
        )
        agent.reset("decline", {})
        agent.respond("decline", "I'm browsing shoes", 1, 10)

        agent.respond(
            "decline",
            "I don't have a preference for color; use your judgment.",
            2,
            10,
        )

        active = agent._state.state_for("decline")
        self.assertIsNone(active.color)
        controller_state = controller.state_for("decline")
        assert controller_state is not None
        self.assertEqual(controller_state.declined_attributes, frozenset({"color"}))

    def test_analyzer_or_controller_failure_returns_unchanged_response(self) -> None:
        parent_asins = self.write_catalog(10)
        baseline, _, _ = self.build_agent(parent_asins, enabled=False)
        baseline.reset("base", {})
        expected = baseline.respond("base", "I'm browsing shoes", 1, 10)

        failing_analyzer = RecordingAnalyzer(error=RuntimeError("analysis failed"))
        analyzer_agent, _, _ = self.build_agent(
            parent_asins, enabled=True, analyzer=failing_analyzer
        )
        analyzer_agent.reset("analyzer", {})
        analyzer_response = analyzer_agent.respond(
            "analyzer", "I'm browsing shoes", 1, 10
        )

        failing_controller = FailingController()
        controller_agent, _, _ = self.build_agent(
            parent_asins, enabled=True, controller=failing_controller
        )
        controller_agent.reset("controller", {})
        controller_response = controller_agent.respond(
            "controller", "I'm browsing shoes", 1, 10
        )

        self.assertEqual(analyzer_response, expected)
        self.assertEqual(controller_response, expected)

    def test_identical_runs_are_deterministic_and_never_add_invalid_duplicates(
        self,
    ) -> None:
        parent_asins = self.write_catalog(10)
        noisy_results = [*parent_asins, parent_asins[0], "INVALID"]
        outputs = []
        for index in range(2):
            agent, _, _ = self.build_agent(noisy_results, enabled=True)
            agent.reset(f"repeat-{index}", {})
            outputs.append(
                agent.respond(f"repeat-{index}", "I'm browsing shoes", 1, 10)
            )

        self.assertEqual(outputs[0], outputs[1])
        asins = response_asins(outputs[0])
        self.assertEqual(len(asins), len(set(asins)))
        self.assertTrue(set(asins).issubset(parent_asins))


class SelectiveClarificationConfigTest(unittest.TestCase):
    def test_default_is_disabled_and_buying_gate_is_stricter(self) -> None:
        config = SelectiveClarificationConfig()

        self.assertFalse(config.enabled)
        self.assertGreater(
            config.buying_min_expected_reduction,
            config.browsing_min_expected_reduction,
        )
        self.assertGreaterEqual(
            config.buying_min_candidates, config.browsing_min_candidates
        )


if __name__ == "__main__":
    unittest.main()
