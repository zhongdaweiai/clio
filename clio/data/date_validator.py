"""Strict published-date validation.

Why this is paranoid: every backtest result is fiction if a single document
leaks across its market's cutoff. News sources lie or guess about
`published_date`. Defenses:

1. Extract candidate dates from the URL (e.g., /2024/03/15/ patterns)
2. Extract candidate dates from HTML metadata (<time datetime="...">,
   <meta property="article:published_time">, JSON-LD `datePublished`)
3. Compare the source's claimed date against extracted dates
4. Compute a `consensus_date` and a confidence rating

Decision rules:
- "high":      claim agrees with at least one extracted date within 2 days
- "medium":    one of (claim, url, html) is present but no cross-check
- "low":       only one weak source, e.g. URL pattern with no claim
- "rejected":  no parseable date anywhere, OR conflicting extracted dates
               disagree by > 14 days

We choose "rejected" liberally. False rejects are cheap (we lose a doc);
false accepts can poison the entire backtest.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from datetime import date, timedelta


_URL_DATE_PATTERNS = [
    re.compile(r"/(?P<y>20\d{2})/(?P<m>\d{1,2})/(?P<d>\d{1,2})/"),
    re.compile(r"/(?P<y>20\d{2})-(?P<m>\d{1,2})-(?P<d>\d{1,2})[/-]"),
    re.compile(r"[?&](?:date|published|d)=(?P<y>20\d{2})-(?P<m>\d{1,2})-(?P<d>\d{1,2})"),
]


_HTML_DATE_PATTERNS = [
    # JSON-LD datePublished
    re.compile(r'"datePublished"\s*:\s*"([^"]+)"', re.IGNORECASE),
    # Open Graph / article meta
    re.compile(r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\']([^"\']+)["\']', re.IGNORECASE),
    re.compile(r'<meta[^>]+name=["\']pubdate["\'][^>]+content=["\']([^"\']+)["\']', re.IGNORECASE),
    re.compile(r'<meta[^>]+name=["\']publishdate["\'][^>]+content=["\']([^"\']+)["\']', re.IGNORECASE),
    # <time datetime="...">
    re.compile(r'<time[^>]+datetime=["\']([^"\']+)["\']', re.IGNORECASE),
]


@dataclass
class DateValidation:
    claimed_date: date | None
    url_date: date | None
    html_dates: list[date] = field(default_factory=list)
    consensus_date: date | None = None
    confidence: str = "rejected"  # "high" | "medium" | "low" | "rejected"
    reason: str = ""

    @property
    def all_extracted(self) -> list[date]:
        out: list[date] = []
        if self.url_date:
            out.append(self.url_date)
        out.extend(self.html_dates)
        return out


def _try_parse(s: str) -> date | None:
    s = s.strip()
    if not s:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d %B %Y",
        "%B %d, %Y",
    ):
        try:
            return dt.datetime.strptime(s[: len(fmt)], fmt).date()
        except ValueError:
            continue
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def extract_date_from_url(url: str) -> date | None:
    for pat in _URL_DATE_PATTERNS:
        m = pat.search(url)
        if not m:
            continue
        try:
            y, mo, d = int(m["y"]), int(m["m"]), int(m["d"])
            return date(y, mo, d)
        except (ValueError, KeyError):
            continue
    return None


def extract_dates_from_html(html: str) -> list[date]:
    if not html:
        return []
    out: list[date] = []
    seen: set[date] = set()
    for pat in _HTML_DATE_PATTERNS:
        for m in pat.finditer(html):
            d = _try_parse(m.group(1))
            if d is None or d in seen:
                continue
            out.append(d)
            seen.add(d)
    return out


def _max_diff_days(dates: list[date]) -> int:
    if len(dates) < 2:
        return 0
    return (max(dates) - min(dates)).days


def validate_published_date(
    claimed_date: date | None,
    url: str,
    html: str | None = None,
    *,
    max_disagreement_days: int = 14,
    cross_check_tolerance_days: int = 2,
) -> DateValidation:
    url_date = extract_date_from_url(url) if url else None
    html_dates = extract_dates_from_html(html) if html else []

    v = DateValidation(
        claimed_date=claimed_date,
        url_date=url_date,
        html_dates=html_dates,
    )

    if _max_diff_days(v.all_extracted + ([claimed_date] if claimed_date else [])) > max_disagreement_days:
        v.confidence = "rejected"
        v.reason = (
            f"sources disagree by more than {max_disagreement_days} days: "
            f"claimed={claimed_date} url={url_date} html={html_dates}"
        )
        return v

    extracted = v.all_extracted

    if claimed_date and extracted:
        for ed in extracted:
            if abs((ed - claimed_date).days) <= cross_check_tolerance_days:
                v.consensus_date = claimed_date
                v.confidence = "high"
                v.reason = "claim cross-checked against extracted date"
                return v
        v.consensus_date = min(extracted)
        v.confidence = "low"
        v.reason = "claim disagrees with extracted dates within tolerance; using earliest extracted"
        return v

    if extracted:
        v.consensus_date = min(extracted)
        v.confidence = "medium" if len(extracted) > 1 else "low"
        v.reason = "no claim but extracted from page"
        return v

    if claimed_date:
        v.consensus_date = claimed_date
        v.confidence = "low"
        v.reason = "claim only, no independent verification"
        return v

    v.confidence = "rejected"
    v.reason = "no parseable date anywhere"
    return v


def assert_strict(v: DateValidation) -> date:
    """Raise if the validation is not 'high' confidence. Use in production paths."""
    if v.confidence != "high" or v.consensus_date is None:
        raise ValueError(
            f"strict date validation failed: confidence={v.confidence} reason={v.reason}"
        )
    return v.consensus_date
