"""Exact weak-BM25 anchor used by the deterministic recovery path."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from .search_models import RetrievalResult

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "some",
    "that",
    "the",
    "this",
    "to",
    "want",
    "with",
    "would",
    "you",
    "looking",
}
FIELDS = ("title", "categories", "features", "details", "store", "description")
BM25_WEIGHTS = (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            token.lower()
            for token in TOKEN_RE.findall(text)
            if len(token) > 1 and token.lower() not in STOPWORDS
        )
    )[:40]


class BM25AnchorRetriever:
    """Reproduce the official starter fields, weights, and raw-turn query."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.connection = sqlite3.connect(":memory:")
        self.connection.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, "
            "description, tokenize='unicode61 remove_diacritics 2')"
        )
        placeholders = ", ".join("?" for _ in range(len(FIELDS) + 1))
        insert_sql = f"INSERT INTO products VALUES ({placeholders})"
        batch: list[tuple[str, ...]] = []
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        *(_text(product.get(field)) for field in FIELDS),
                    )
                )
                if len(batch) >= 1_000:
                    self.connection.executemany(insert_sql, batch)
                    batch.clear()
        if batch:
            self.connection.executemany(insert_sql, batch)
        self.connection.commit()

    def retrieve(
        self, raw_current_turn_text: str, top_n: int = 10
    ) -> list[RetrievalResult]:
        if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n <= 0:
            return []
        terms = _terms(raw_current_turn_text)
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in terms)
        weights = ", ".join(str(value) for value in BM25_WEIGHTS)
        rows = self.connection.execute(
            "SELECT parent_asin, bm25(products, " + weights + ") AS score "
            "FROM products WHERE products MATCH ? "
            "ORDER BY score ASC, rowid ASC LIMIT ?",
            (expression, top_n),
        ).fetchall()
        return [
            RetrievalResult(str(parent_asin), -float(score), rank)
            for rank, (parent_asin, score) in enumerate(rows, start=1)
        ]
