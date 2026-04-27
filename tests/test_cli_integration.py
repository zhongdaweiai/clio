"""End-to-end CLI integration: fetch from dump → news pipeline → backtest → red-team."""

import json
from datetime import date
from pathlib import Path

import pytest

from clio.cli import main


FIXTURE_PM = Path(__file__).parent / "fixtures" / "polymarket_dump.json"
FIXTURE_NEWS = Path(__file__).parent / "fixtures" / "news_sample.jsonl"


def test_cli_fetch_polymarket_from_dump(tmp_path, capsys):
    out = tmp_path / "markets.json"
    rc = main([
        "fetch", "polymarket",
        "--from-dump", str(FIXTURE_PM),
        "--since", "2023-01-01",
        "--until", "2025-01-01",
        "--out", str(out),
    ])
    assert rc == 0
    assert out.exists()
    data = json.loads(out.read_text())
    assert len(data["markets"]) == 4
    assert data["resolutions"]["100001"] == 1


def test_cli_fetch_news_jsonl(tmp_path):
    markets_out = tmp_path / "markets.json"
    main([
        "fetch", "polymarket",
        "--from-dump", str(FIXTURE_PM),
        "--since", "2023-01-01",
        "--until", "2025-01-01",
        "--out", str(markets_out),
    ])

    news_out = tmp_path / "news.jsonl"
    rc = main([
        "fetch", "news",
        "--markets", str(markets_out),
        "--source", "jsonl",
        "--jsonl-path", str(FIXTURE_NEWS),
        "--out", str(news_out),
    ])
    assert rc == 0
    assert news_out.exists()
    lines = news_out.read_text().strip().splitlines()
    assert len(lines) > 0


def test_cli_full_pipeline_end_to_end(tmp_path):
    markets_out = tmp_path / "markets.json"
    news_out = tmp_path / "news.jsonl"

    rc1 = main([
        "fetch", "polymarket",
        "--from-dump", str(FIXTURE_PM),
        "--since", "2023-01-01",
        "--until", "2025-01-01",
        "--out", str(markets_out),
    ])
    assert rc1 == 0

    rc2 = main([
        "fetch", "news",
        "--markets", str(markets_out),
        "--source", "jsonl",
        "--jsonl-path", str(FIXTURE_NEWS),
        "--out", str(news_out),
    ])
    assert rc2 == 0

    # Backtest with 1 train + 3 holdout (4 markets in fixture).
    rc3 = main([
        "backtest",
        "--markets", str(markets_out),
        "--news", str(news_out),
        "--train-size", "1",
    ])
    assert rc3 == 0
