from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from starter.agent import Agent
from starter.ambiguity_analysis import ClarificationOpportunity
from starter.clarification_policies import (
    ROLLBACK_POLICY_ID,
    SELECTED_POLICY_ID,
    load_clarification_policy_by_id,
)
from starter.hybrid_retrieval import HybridRetrievalConfig
from starter.lexical_retriever import parse_prices
from starter.selective_clarification import SelectiveClarificationConfig


class Retriever:
    def __init__(self, values: list[str], error: Exception | None = None) -> None:
        self.values = values
        self.error = error
        self.calls = 0

    def retrieve(self, *_args: object, **kwargs: object) -> list[object]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        top_n = int(kwargs.get("top_n", len(self.values)))
        return [
            SimpleNamespace(parent_asin=value, score=1.0 / rank, rank=rank)
            for rank, value in enumerate(self.values[:top_n], start=1)
        ]


class Router:
    def __init__(self, route: str = "browsing", error: Exception | None = None) -> None:
        self.route_name = route
        self.error = error

    def route(self, *_args: object) -> object:
        if self.error is not None:
            raise self.error
        return SimpleNamespace(route=self.route_name)


class Analyzer:
    def __init__(
        self, attribute: str = "color", error: Exception | None = None
    ) -> None:
        self.attribute = attribute
        self.error = error

    def analyze(self, *_args: object) -> ClarificationOpportunity:
        if self.error is not None:
            raise self.error
        return ClarificationOpportunity(
            True, self.attribute, ("red", "blue"), 0.5, "test"
        )


class ExplodingEligibility:
    enabled = True
    required_retrieval_policy_id = "contextual.browsing-dense.v1"
    analysis_candidate_limit = 50

    def is_eligible(self, *_args: object) -> bool:
        raise RuntimeError("eligibility failed")


def asins(response: dict[str, object]) -> list[str]:
    return [str(item["parent_asin"]) for item in response["recommendations"]]  # type: ignore[index]


class AgentReliabilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.catalog = Path(self.temporary.name) / "catalog.jsonl"
        rows = [
            {"parent_asin": "A", "title": "red shoe", "color": "red", "price": None},
            {"parent_asin": "B", "title": "blue shoe", "color": "blue", "price": "NaN"},
            {"parent_asin": "C", "title": "shoe", "price": float("inf")},
            {"parent_asin": "D"},
        ]
        self.catalog.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def agent(
        self,
        *,
        anchor: Retriever | None = None,
        dense: Retriever | None = None,
        dense_factory=None,
        router: Router | None = None,
        enabled: bool = False,
        analyzer: object | None = None,
        composer=None,
    ) -> Agent:
        values = ["A", "B", "C", "D"]
        kwargs: dict[str, object] = {}
        if dense is None and dense_factory is None:
            dense = Retriever([])
        if composer is not None:
            kwargs["clarification_composer"] = composer
        return Agent(
            self.catalog,
            config=HybridRetrievalConfig(mode="contextual"),
            anchor_retriever=anchor or Retriever(values),
            dense_retriever=dense,
            dense_factory=dense_factory,
            router=router or Router(),
            clarification_config=SelectiveClarificationConfig(enabled=enabled),
            ambiguity_analyzer=analyzer,
            **kwargs,
        )

    def test_empty_and_whitespace_query_skip_retrieval(self) -> None:
        anchor = Retriever(["A"])
        agent = self.agent(anchor=anchor)
        agent.reset("s", {})
        self.assertEqual(asins(agent.respond("s", "", 1, 10)), [])
        self.assertEqual(asins(agent.respond("s", "   ", 2, 10)), [])
        self.assertEqual(anchor.calls, 0)

    def test_dense_missing_corrupt_initialization_and_query_failures_use_bm25(
        self,
    ) -> None:
        for error in (FileNotFoundError("missing cache"), ValueError("corrupt cache")):
            with self.subTest(error=error):
                attempts = 0
                injected_error = error

                def factory(injected_error: Exception = injected_error) -> object:
                    nonlocal attempts
                    attempts += 1
                    raise injected_error

                agent = self.agent(dense_factory=factory)
                agent.reset("s", {})
                self.assertEqual(
                    asins(agent.respond("s", "browse shoes", 1, 3)), ["A", "B", "C"]
                )
                self.assertEqual(attempts, 1)
        failing_dense = Retriever([], RuntimeError("query failed"))
        agent = self.agent(dense=failing_dense)
        agent.reset("q", {})
        self.assertEqual(
            asins(agent.respond("q", "browse shoes", 1, 3)), ["A", "B", "C"]
        )
        self.assertEqual(
            agent.diagnostics_snapshot()["component_failure_counts"], {"dense": 1}
        )

    def test_router_failure_uses_raw_bm25_without_second_retrieval(self) -> None:
        anchor = Retriever(["B", "A", "C"])
        agent = self.agent(anchor=anchor, router=Router(error=RuntimeError("route")))
        agent.reset("s", {})
        self.assertEqual(
            asins(agent.respond("s", "browse shoes", 1, 3)), ["B", "A", "C"]
        )
        self.assertEqual(anchor.calls, 1)
        self.assertEqual(
            agent.diagnostics_snapshot()["component_failure_counts"], {"router": 1}
        )

    def test_fusion_failure_returns_existing_bm25_order_once(self) -> None:
        anchor = Retriever(["C", "B", "A"])
        dense = Retriever(["A"])
        agent = self.agent(anchor=anchor, dense=dense)
        agent.reset("s", {})
        with patch(
            "starter.agent.rank_contextual_candidates",
            side_effect=RuntimeError("fusion"),
        ):
            response = agent.respond("s", "browse shoes", 1, 3)
        self.assertEqual(asins(response), ["C", "B", "A"])
        self.assertEqual((anchor.calls, dense.calls), (1, 1))

    def test_clarification_failures_keep_ranked_recommendations(self) -> None:
        failures = [
            {"analyzer": Analyzer(error=RuntimeError("analysis"))},
            {"analyzer": Analyzer(attribute="unsupported")},
            {
                "analyzer": Analyzer(),
                "composer": lambda *_args: (_ for _ in ()).throw(
                    RuntimeError("template")
                ),
            },
        ]
        for index, options in enumerate(failures):
            with self.subTest(index=index):
                agent = self.agent(enabled=True, **options)
                agent.reset("s", {})
                response = agent.respond("s", "browse shoes", 1, 3)
                self.assertEqual(asins(response), ["A", "B", "C"])
                self.assertIsNone(response["ask_attribute"])
        policy_agent = self.agent(enabled=True, analyzer=Analyzer())
        policy_agent._clarification_config = ExplodingEligibility()  # type: ignore[assignment]
        policy_agent.reset("policy", {})
        self.assertEqual(
            asins(policy_agent.respond("policy", "browse", 1, 3)), ["A", "B", "C"]
        )

    def test_missing_metadata_and_nonfinite_prices_are_unknown(self) -> None:
        self.assertEqual(parse_prices(None), ())
        self.assertEqual(parse_prices("NaN"), ())
        self.assertEqual(parse_prices(float("nan")), ())
        self.assertEqual(parse_prices(float("inf")), ())
        agent = self.agent(enabled=True)
        agent.reset("s", {})
        response = agent.respond("s", "browse shoes", 1, 4)
        self.assertEqual(asins(response), ["A", "B", "C", "D"])

    def test_selected_policy_failure_uses_verified_named_rollback(self) -> None:
        rollback = load_clarification_policy_by_id(ROLLBACK_POLICY_ID)

        def loader(policy_id: str):
            if policy_id == SELECTED_POLICY_ID:
                raise ValueError("selected missing")
            return rollback

        agent = Agent(
            self.catalog,
            anchor_retriever=Retriever(["A"]),
            clarification_policy_loader=loader,
        )
        self.assertFalse(agent._clarification_config.enabled)
        self.assertEqual(agent._initialization_fallback_counts, {"selected_policy": 1})

    def test_both_selected_and_rollback_invalid_fail_initialization(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "selected and rollback"):
            Agent(
                self.catalog,
                clarification_policy_loader=lambda _name: (_ for _ in ()).throw(
                    ValueError("bad")
                ),
            )

    def test_total_generation_and_safe_lexical_failure_returns_empty_contract(
        self,
    ) -> None:
        anchor = Retriever([], RuntimeError("all retrieval failed"))
        agent = self.agent(anchor=anchor)
        agent.reset("s", {})
        response = agent.respond("s", "browse shoes", 1, 10)
        self.assertEqual(response["recommendations"], [])
        self.assertEqual(response["message"], "Here are the closest matches I found.")
        self.assertIsNone(response["ask_attribute"])
        self.assertEqual(anchor.calls, 2)

    def test_unexpected_failure_gets_one_nonrecursive_protected_lexical_attempt(
        self,
    ) -> None:
        anchor = Retriever(["B", "A"])
        agent = self.agent(anchor=anchor)
        agent.reset("s", {})
        with patch.object(
            agent, "_respond_once", side_effect=RuntimeError("unexpected")
        ):
            response = agent.respond("s", "browse shoes", 1, 2)
        self.assertEqual(asins(response), ["B", "A"])
        self.assertEqual(anchor.calls, 1)

    def test_reset_is_idempotent_and_matches_fresh_agent(self) -> None:
        reset_agent = self.agent()
        reset_agent.reset("same", {})
        reset_agent.respond("same", "browse shoes", 1, 2)
        reset_agent.respond("same", "not quite right", 2, 2)
        reset_agent.reset("same", {})
        reset_agent.reset("same", {})
        reset_output = reset_agent.respond("same", "browse shoes", 1, 2)
        fresh = self.agent()
        fresh.reset("fresh", {})
        fresh_output = fresh.respond("fresh", "browse shoes", 1, 2)
        self.assertEqual(reset_output, fresh_output)
        self.assertEqual(reset_agent._known_negative_ids["same"], set())
        self.assertEqual(reset_agent._fallback_cache["same"], {})

    def test_pending_clarification_is_cleared_and_turn_ten_never_asks(self) -> None:
        agent = self.agent(enabled=True, analyzer=Analyzer())
        agent.reset("s", {})
        self.assertEqual(
            agent.respond("s", "browse shoes", 9, 3)["ask_attribute"], "color"
        )
        agent.reset("s", {})
        state = agent._clarification_controller.state_for("s")
        self.assertIsNone(state.pending_attribute)
        self.assertIsNone(agent.respond("s", "browse shoes", 10, 3)["ask_attribute"])

    def test_identical_transcripts_are_deterministic_and_failure_does_not_cross_sessions(
        self,
    ) -> None:
        agent = self.agent(enabled=True, analyzer=Analyzer(error=RuntimeError("once")))
        outputs = []
        for session_id in ("one", "two"):
            agent.reset(session_id, {})
            outputs.append(agent.respond(session_id, "browse shoes", 1, 3))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(agent._known_negative_ids["two"], set())


if __name__ == "__main__":
    unittest.main()
