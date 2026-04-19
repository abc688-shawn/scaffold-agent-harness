"""Tests for cache layer."""
import time
import pytest

from scaffold.cache.cache import ResultCache


class TestResultCache:
    def test_put_and_get(self):
        cache = ResultCache()
        cache.put("k1", "value1")
        assert cache.get("k1") == "value1"

    def test_miss(self):
        cache = ResultCache()
        assert cache.get("nonexistent") is None

    def test_expiration(self):
        cache = ResultCache(default_ttl=0.1)
        cache.put("k1", "value1")
        assert cache.get("k1") == "value1"
        time.sleep(0.15)
        assert cache.get("k1") is None

    def test_make_key(self):
        k1 = ResultCache.make_key("tool", path="/a", query="test")
        k2 = ResultCache.make_key("tool", path="/a", query="test")
        k3 = ResultCache.make_key("tool", path="/b", query="test")
        assert k1 == k2
        assert k1 != k3

    def test_stats(self):
        cache = ResultCache()
        cache.put("k", "v")
        cache.get("k")  # hit
        cache.get("missing")  # miss
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_invalidate(self):
        cache = ResultCache()
        cache.put("k", "v")
        cache.invalidate("k")
        assert cache.get("k") is None

    def test_clear(self):
        cache = ResultCache()
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.clear()
        assert cache.stats["size"] == 0
