"""ChromaDB client factory and async-to-sync embedding bridge.

Chroma's EmbeddingFunction protocol is synchronous, but EmbeddingClient.embed
is async.  We bridge them with a dedicated daemon thread + event loop so that
asyncio.run() is never called from a running loop, and nest_asyncio is avoided.
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fs_agent.tools.search_tools import EmbeddingClient

_client_cache: dict[str, object] = {}
_cache_lock = threading.Lock()


class _AsyncBridge:
    """Singleton background thread + event loop for sync-calling async embed."""

    _instance: "_AsyncBridge | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="chroma-embed-bridge"
        )
        self._thread.start()

    @classmethod
    def get(cls) -> "_AsyncBridge":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def run(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()


class SyncEmbeddingFunction:
    """Chroma EmbeddingFunction protocol wrapping an async EmbeddingClient.

    Chroma 1.5+ distinguishes embed_documents (__call__) and embed_query;
    both delegate to the same async EmbeddingClient.embed().
    """

    def __init__(self, client: "EmbeddingClient") -> None:
        self._client = client

    def __call__(self, input: list[str]) -> list[list[float]]:
        if not input:
            return []
        return _AsyncBridge.get().run(self._client.embed(list(input)))

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self.__call__(input)

    @staticmethod
    def name() -> str:
        return "fs_agent_openai_compat"

    @staticmethod
    def build_from_config(config: dict) -> "SyncEmbeddingFunction":
        raise NotImplementedError("SyncEmbeddingFunction cannot be reconstructed from config")


def get_client(persist_dir: "Path | None"):
    """Return a cached Chroma client for the given persistence directory."""
    import chromadb  # type: ignore[import]

    key = str(persist_dir) if persist_dir else "__ephemeral__"
    with _cache_lock:
        if key not in _client_cache:
            if persist_dir:
                _client_cache[key] = chromadb.PersistentClient(path=str(persist_dir))
            else:
                _client_cache[key] = chromadb.EphemeralClient()
    return _client_cache[key]


def get_or_create_collection(client, name: str, embed_fn: "SyncEmbeddingFunction"):
    """Get or create a collection with cosine distance metric."""
    return client.get_or_create_collection(
        name=name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def reset_clients() -> None:
    """Clear the Chroma client cache. For testing only."""
    with _cache_lock:
        _client_cache.clear()
