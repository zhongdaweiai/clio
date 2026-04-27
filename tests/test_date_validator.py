"""Strict date validator tests. These guard the most important invariant in
the system — that no document leaks across its market's cutoff."""

from datetime import date

from clio.data.date_validator import (
    extract_date_from_url,
    extract_dates_from_html,
    validate_published_date,
    assert_strict,
)


def test_extract_url_slash_date():
    assert extract_date_from_url("https://example.com/2024/03/15/headline") == date(2024, 3, 15)


def test_extract_url_dash_date():
    assert extract_date_from_url("https://example.com/news/2024-03-15-event/article") == date(2024, 3, 15)


def test_extract_url_query_date():
    assert extract_date_from_url("https://example.com/article?id=42&date=2024-03-15") == date(2024, 3, 15)


def test_extract_url_no_date_returns_none():
    assert extract_date_from_url("https://example.com/article/headline") is None


def test_extract_html_jsonld():
    html = '... "datePublished": "2024-03-15T12:00:00Z" ...'
    assert date(2024, 3, 15) in extract_dates_from_html(html)


def test_extract_html_meta_published_time():
    html = '<meta property="article:published_time" content="2024-03-15T08:00:00Z">'
    assert date(2024, 3, 15) in extract_dates_from_html(html)


def test_extract_html_time_element():
    html = '<time datetime="2024-03-15T08:00:00">March 15</time>'
    assert date(2024, 3, 15) in extract_dates_from_html(html)


def test_extract_html_empty():
    assert extract_dates_from_html("") == []


def test_validate_high_confidence_when_claim_matches_url():
    v = validate_published_date(
        claimed_date=date(2024, 3, 15),
        url="https://example.com/2024/03/15/article",
    )
    assert v.confidence == "high"
    assert v.consensus_date == date(2024, 3, 15)


def test_validate_rejects_no_date_anywhere():
    v = validate_published_date(claimed_date=None, url="https://example.com/article")
    assert v.confidence == "rejected"
    assert v.consensus_date is None


def test_validate_rejects_when_sources_disagree_more_than_two_weeks():
    v = validate_published_date(
        claimed_date=date(2024, 3, 1),
        url="https://example.com/2024/02/01/article",
    )
    assert v.confidence == "rejected"


def test_validate_low_confidence_with_only_claim():
    v = validate_published_date(
        claimed_date=date(2024, 3, 15),
        url="https://example.com/article-without-date",
    )
    assert v.confidence == "low"
    assert v.consensus_date == date(2024, 3, 15)


def test_validate_low_when_claim_disagrees_within_tolerance_band():
    # Within 14-day overall tolerance but outside 2-day cross-check tolerance.
    v = validate_published_date(
        claimed_date=date(2024, 3, 15),
        url="https://example.com/2024/03/22/article",
    )
    assert v.confidence == "low"


def test_assert_strict_passes_on_high():
    v = validate_published_date(
        claimed_date=date(2024, 3, 15),
        url="https://example.com/2024/03/15/article",
    )
    assert assert_strict(v) == date(2024, 3, 15)


def test_assert_strict_raises_on_low():
    v = validate_published_date(
        claimed_date=date(2024, 3, 15),
        url="https://example.com/article",
    )
    import pytest

    with pytest.raises(ValueError):
        assert_strict(v)
