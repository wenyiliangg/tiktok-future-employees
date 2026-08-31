"""Interactive terminal demo for the promoted TechJam shopping agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from starter.agent import Agent
from starter.clarification_controller import ClarificationController
from starter.clarification_policies import load_runtime_clarification_policy
from starter.contextual_retrieval import policy_by_id
from starter.hybrid_retrieval import HybridRetrievalConfig, RetrievalMode


def positive_int(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 10:
        raise argparse.ArgumentTypeError("value must be between 1 and 10")
    return parsed


def product_labels(catalog_path: Path) -> dict[str, tuple[str, str]]:
    labels: dict[str, tuple[str, str]] = {}
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            parent_asin = product.get("parent_asin")
            if not isinstance(parent_asin, str) or not parent_asin:
                continue
            title = str(product.get("title") or "Untitled product")
            raw_price = product.get("price")
            if raw_price in (None, ""):
                price = "price unavailable"
            else:
                price = str(raw_price)
                if not price.startswith("$"):
                    price = f"${price}"
            labels[parent_asin] = (title, price)
    return labels


def build_agent(catalog_path: Path, dense_cache_path: Path) -> tuple[Agent, str]:
    clarification_policy, warning = load_runtime_clarification_policy()
    if warning is not None:
        print(f"Policy warning: {warning}")
    retrieval_policy = policy_by_id(clarification_policy.retrieval_policy_id)
    agent = Agent(
        catalog_path,
        config=HybridRetrievalConfig(mode=RetrievalMode.CONTEXTUAL),
        dense_cache_path=dense_cache_path,
        contextual_policy=retrieval_policy,
        clarification_config=clarification_policy.clarification,
        clarification_controller=ClarificationController(
            clarification_policy.controller
        ),
    )
    return agent, clarification_policy.policy_id


def show_response(
    response: dict[str, Any],
    labels: dict[str, tuple[str, str]],
) -> None:
    message = response.get("message") or "Here are the closest matches I found."
    print(f"\nAgent: {message}")
    ask_attribute = response.get("ask_attribute")
    if ask_attribute:
        print(f"Structured question: {ask_attribute}")

    recommendations = response.get("recommendations")
    if not isinstance(recommendations, list) or not recommendations:
        print("Recommendations: none")
        return

    print("Recommendations:")
    for rank, item in enumerate(recommendations, start=1):
        if not isinstance(item, dict):
            continue
        parent_asin = str(item.get("parent_asin") or "")
        title, price = labels.get(
            parent_asin, ("Unknown product", "price unavailable")
        )
        compact_title = " ".join(title.split())
        if len(compact_title) > 88:
            compact_title = compact_title[:85].rstrip() + "..."
        print(f"  {rank:>2}. {parent_asin} | {price} | {compact_title}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Talk to the promoted deterministic TechJam shopping agent."
    )
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument(
        "--dense-cache",
        type=Path,
        default=Path("data/.dense-retrieval/catalog-minilm.npz"),
    )
    parser.add_argument("--top-k", type=positive_int, default=10)
    parser.add_argument(
        "--profile-tags",
        nargs="*",
        default=["comfort", "durability"],
        help="Optional aggregate preference tags for the demo profile.",
    )
    args = parser.parse_args()

    if not args.catalog.is_file():
        parser.error(
            f"catalog not found at {args.catalog}; place the authorized catalog there first"
        )

    print("Building the local catalog index. This can take several seconds...")
    labels = product_labels(args.catalog)
    agent, policy_id = build_agent(args.catalog, args.dense_cache)
    session_id = "interactive-demo"
    agent.reset(
        session_id,
        {
            "preference_tags": list(args.profile_tags),
            "summary": "Interactive demonstration profile.",
        },
    )

    print(f"Ready. Active clarification policy: {policy_id}")
    print("Describe a product, answer questions naturally, or enter /quit.\n")

    turn = 1
    while turn <= 10:
        try:
            user_message = input(f"Customer [turn {turn}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDemo ended.")
            return
        if user_message.lower() in {"/quit", "quit", "exit"}:
            print("Demo ended.")
            return
        if not user_message:
            print("Please enter a request or /quit.")
            continue
        response = agent.respond(session_id, user_message, turn, args.top_k)
        show_response(response, labels)
        print()
        turn += 1

    print("The ten-turn session limit has been reached.")


if __name__ == "__main__":
    main()
