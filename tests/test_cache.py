"""DiskCache tests."""

import time

from clio.data.cache import DiskCache


def test_put_get_roundtrip(tmp_path):
    c = DiskCache(tmp_path)
    c.put(("k", 1), {"value": 42})
    assert c.get(("k", 1)) == {"value": 42}


def test_miss_returns_none(tmp_path):
    c = DiskCache(tmp_path)
    assert c.get(("missing",)) is None


def test_ttl_expiry(tmp_path):
    c = DiskCache(tmp_path)
    c.put(("k",), "v", ttl_seconds=0.01)
    time.sleep(0.05)
    assert c.get(("k",)) is None


def test_no_ttl_persists(tmp_path):
    c = DiskCache(tmp_path)
    c.put(("k",), "v", ttl_seconds=None)
    time.sleep(0.05)
    assert c.get(("k",)) == "v"


def test_clear_empties(tmp_path):
    c = DiskCache(tmp_path)
    c.put(("a",), 1)
    c.put(("b",), 2)
    assert len(c) == 2
    c.clear()
    assert len(c) == 0


def test_keys_with_different_part_orders_collide(tmp_path):
    """Cache keys are content-based so semantically equal tuples should collide.
    But order DOES matter — these are tuples, not sets. Verify lexical hashing."""
    c = DiskCache(tmp_path)
    c.put(("a", "b"), 1)
    c.put(("b", "a"), 2)
    # Different order = different key
    assert c.get(("a", "b")) == 1
    assert c.get(("b", "a")) == 2
