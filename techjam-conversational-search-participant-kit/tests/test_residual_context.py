from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from starter.agent import Agent
from starter.contextual_retrieval import policy_by_id
from starter.hybrid_retrieval import HybridRetrievalConfig
from starter.selective_clarification import SelectiveClarificationConfig


@dataclass(frozen=True)
class FakeResult:
    parent_asin: str
    score: float
    rank: int


class RecordingRetriever:
    def __init__(self, results: list[FakeResult]) -> None:
        self.results = results
        self.queries: list[object] = []

    def retrieve(self, query: object, top_n: int) -> list[FakeResult]:
        self.queries.append(query)
        return self.results[:top_n]


class ResidualContextTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.catalog = Path(self.temporary.name) / "catalog.jsonl"
        self.catalog.write_text(
            "".join(
                json.dumps({"parent_asin": asin, "title": asin}) + "\n"
                for asin in ("A", "B", "C", "D")
            ),
            encoding="utf-8",
        )
        self.anchor = RecordingRetriever(
            [
                FakeResult("A", 4, 1),
                FakeResult("B", 3, 2),
                FakeResult("C", 2, 3),
                FakeResult("D", 1, 4),
            ]
        )
        self.dense = RecordingRetriever([FakeResult("D", 0.8, 1)])
        self.agent = Agent(
            self.catalog,
            config=HybridRetrievalConfig(mode="contextual"),
            contextual_policy=policy_by_id("contextual.domain-residual.v1"),
            clarification_config=SelectiveClarificationConfig(enabled=False),
            anchor_retriever=self.anchor,
            dense_retriever=self.dense,
        )
        self.agent.reset("session", {})

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_negative_feedback_keeps_informative_intent_for_dense_residual(self) -> None:
        initial = "I'm exploring ideas and inspiration for a comfortable city trip."
        rejection = "Those options are not quite right yet."

        self.agent.respond("session", initial, 1, 2)
        self.agent.respond("session", rejection, 2, 2)

        self.assertEqual(len(self.dense.queries), 2)
        second_query = str(self.dense.queries[1])
        self.assertIn(initial, second_query)
        self.assertNotIn(rejection, second_query)

    def test_override_replaces_residual_intent_memory(self) -> None:
        initial = "I'm exploring ideas and inspiration for a comfortable city trip."
        override = (
            "Actually, ignore my earlier preference. I want ideas and inspiration "
            "for a city trip."
        )

        self.agent.respond("session", initial, 1, 2)
        self.agent.respond("session", override, 2, 2)

        second_query = str(self.dense.queries[1])
        self.assertIn(override, second_query)
        self.assertNotIn(initial, second_query)


if __name__ == "__main__":
    unittest.main()
