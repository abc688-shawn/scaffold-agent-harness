"""Simple in-memory + optional disk cache for tool results and LLM responses."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    ttl: float = 0  # 0 = no expiration

    @property
    def expired(self) -> bool:
        if self.ttl <= 0:
            return False
        return (time.time() - self.created_at) > self.ttl


class ResultCache:
    """In-memory cache keyed by tool name + arguments hash.

    Use for:
    - Tool result caching (same args → skip re-execution)
    - LLM response caching (same prompt → skip API call, great for eval)
    """

    def __init__(self, default_ttl: float = 300.0) -> None:
        self._store: dict[str, CacheEntry] = {}
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    @staticmethod
    def make_key(namespace: str, **kwargs: Any) -> str:
        raw = json.dumps({"ns": namespace, **kwargs}, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        if entry.expired:
            del self._store[key]
            self._misses += 1
            return None
        self._hits += 1
        return entry.value

    def put(self, key: str, value: Any, ttl: float | None = None) -> None:
        self._store[key] = CacheEntry(
            key=key, value=value, ttl=ttl if ttl is not None else self._default_ttl
        )

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._store)}
