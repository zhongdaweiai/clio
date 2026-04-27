"""Date-sharded document corpus with hard knowledge-cutoff enforcement.

Every search must declare an `as_of` date. Documents with `published_at >= as_of`
are unconditionally excluded. This is the single most important guarantee in the
system — without it, every backtest result is fiction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Iterable


class CutoffViolation(RuntimeError):
    """Raised if a search would return a document published on or after as_of."""


@dataclass(frozen=True)
class Document:
    doc_id: str
    published_at: date
    title: str
    content: str
    source: str = "unknown"
    tags: tuple[str, ...] = field(default_factory=tuple)


class Corpus:
    """An append-only, date-aware document store.

    Search is keyword-AND over title+content, restricted to documents strictly
    earlier than `as_of`. The 'strictly earlier' choice is intentional: a
    document published on the morning of question observation may already be
    factored into the market price, so we err conservative.
    """

    def __init__(self) -> None:
        self._docs: list[Document] = []

    def add(self, doc: Document) -> None:
        self._docs.append(doc)

    def add_many(self, docs: Iterable[Document]) -> None:
        for d in docs:
            self.add(d)

    def __len__(self) -> int:
        return len(self._docs)

    def search(
        self,
        query: str,
        as_of: date,
        limit: int = 10,
    ) -> list[Document]:
        terms = [t.lower() for t in query.split() if t.strip()]
        if not terms:
            return []

        results: list[tuple[int, Document]] = []
        for doc in self._docs:
            if doc.published_at >= as_of:
                continue
            haystack = (doc.title + " " + doc.content).lower()
            score = sum(1 for t in terms if t in haystack)
            if score == 0:
                continue
            results.append((score, doc))

        results.sort(key=lambda x: (-x[0], x[1].published_at), reverse=False)
        results.sort(key=lambda x: (-x[0], -x[1].published_at.toordinal()))
        return [d for _, d in results[:limit]]

    def assert_no_leak(self, returned: Iterable[Document], as_of: date) -> None:
        """Defensive double-check used by tests and the harness."""
        for d in returned:
            if d.published_at >= as_of:
                raise CutoffViolation(
                    f"document {d.doc_id} published {d.published_at} "
                    f"leaked through cutoff {as_of}"
                )
