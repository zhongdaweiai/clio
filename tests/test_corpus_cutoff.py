"""Knowledge cutoff enforcement tests.

If any of these fail, backtest results are fiction. These are the most
important tests in the suite.
"""

from datetime import date

import pytest

from clio.frozen.corpus import Corpus, CutoffViolation, Document


def _doc(doc_id: str, d: date, content: str) -> Document:
    return Document(
        doc_id=doc_id, published_at=d, title=content[:30], content=content
    )


def test_search_excludes_documents_published_at_or_after_cutoff():
    corpus = Corpus()
    corpus.add(_doc("D1", date(2024, 1, 1), "earnings beat hugely"))
    corpus.add(_doc("D2", date(2024, 6, 1), "earnings beat continued"))
    corpus.add(_doc("D3", date(2024, 12, 1), "earnings beat ahead"))

    results = corpus.search("earnings", as_of=date(2024, 6, 1))
    ids = [d.doc_id for d in results]
    # D2 is exactly on the cutoff and must be excluded (strict <).
    assert "D1" in ids
    assert "D2" not in ids
    assert "D3" not in ids


def test_search_returns_empty_for_blank_query():
    corpus = Corpus()
    corpus.add(_doc("D1", date(2024, 1, 1), "content here"))
    assert corpus.search("", as_of=date(2025, 1, 1)) == []


def test_search_term_matching_is_case_insensitive():
    corpus = Corpus()
    corpus.add(_doc("D1", date(2024, 1, 1), "FED raised rates"))
    results = corpus.search("fed rates", as_of=date(2025, 1, 1))
    assert results and results[0].doc_id == "D1"


def test_assert_no_leak_raises_on_violation():
    corpus = Corpus()
    bad = _doc("D1", date(2024, 6, 1), "future content")
    with pytest.raises(CutoffViolation):
        corpus.assert_no_leak([bad], as_of=date(2024, 6, 1))


def test_assert_no_leak_passes_on_clean():
    corpus = Corpus()
    good = _doc("D1", date(2024, 5, 31), "past content")
    corpus.assert_no_leak([good], as_of=date(2024, 6, 1))  # should not raise


def test_search_limit_respected():
    corpus = Corpus()
    for i in range(50):
        corpus.add(_doc(f"D{i}", date(2024, 1, 1), "common token here"))
    results = corpus.search("common", as_of=date(2025, 1, 1), limit=5)
    assert len(results) == 5
