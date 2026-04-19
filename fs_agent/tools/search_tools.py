"""Semantic search tools — embedding-based retrieval for the file workspace.

Uses a lightweight in-memory vector store. Embeddings are computed lazily
(on first search) or explicitly via index_workspace.

Supports pluggable embedding backends; default uses the OpenAI-compatible
embedding endpoint (works with DeepSeek, etc.).
"""
from __future__ import annotations

import hashlib
import math
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

from scaffold.tools.errors import ToolError, ToolErrorCode

from fs_agent.tools.file_tools import registry, _check_sandbox


# ---------------------------------------------------------------------------
# Vector store (lightweight, in-memory, pickle-persisted)
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A text chunk with its embedding and source info."""
    text: str
    file_path: str
    start_line: int
    end_line: int
    file_hash: str
    embedding: list[float] = field(default_factory=list)


class VectorStore:
    """Simple in-memory vector store with cosine similarity search."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._file_hashes: dict[str, str] = {}  # path -> hash

    def add(self, chunk: Chunk) -> None:
        self._chunks.append(chunk)

    def remove_file(self, file_path: str) -> int:
        before = len(self._chunks)
        self._chunks = [c for c in self._chunks if c.file_path != file_path]
        self._file_hashes.pop(file_path, None)
        return before - len(self._chunks)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[tuple[Chunk, float]]:
        if not self._chunks or not query_embedding:
            return []
        results = []
        for chunk in self._chunks:
            if chunk.embedding:
                score = _cosine_similarity(query_embedding, chunk.embedding)
                results.append((chunk, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def is_indexed(self, file_path: str, current_hash: str) -> bool:
        return self._file_hashes.get(file_path) == current_hash

    def mark_indexed(self, file_path: str, file_hash: str) -> None:
        self._file_hashes[file_path] = file_hash

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    @property
    def indexed_files(self) -> int:
        return len(self._file_hashes)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"chunks": self._chunks, "hashes": self._file_hashes}, f)

    def load(self, path: str | Path) -> None:
        p = Path(path)
        if p.exists():
            with open(p, "rb") as f:
                data = pickle.load(f)
            self._chunks = data.get("chunks", [])
            self._file_hashes = data.get("hashes", {})


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Embedding client
# ---------------------------------------------------------------------------

class EmbeddingClient:
    """OpenAI-compatible embedding client."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "text-embedding-ada-002",
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = base_url or os.environ.get("OPENAI_API_BASE", "")
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ToolError(ToolErrorCode.INTERNAL, "openai package required for embeddings")

        client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url or None)
        response = await client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]

    async def embed_single(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, file_path: str, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Split text into overlapping chunks by lines."""
    lines = text.splitlines()
    if not lines:
        return []

    file_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
    chunks = []
    i = 0
    while i < len(lines):
        end = min(i + chunk_size, len(lines))
        chunk_lines = lines[i:end]
        chunk_text_str = "\n".join(chunk_lines)
        if chunk_text_str.strip():
            chunks.append(Chunk(
                text=chunk_text_str,
                file_path=file_path,
                start_line=i + 1,
                end_line=end,
                file_hash=file_hash,
            ))
        i += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Module-level state (configured at startup)
# ---------------------------------------------------------------------------

_store = VectorStore()
_embed_client: EmbeddingClient | None = None
_index_path: str | None = None


def configure_search(
    embed_client: EmbeddingClient,
    index_path: str | None = None,
) -> None:
    global _embed_client, _index_path
    _embed_client = embed_client
    _index_path = index_path
    if index_path:
        _store.load(index_path)


# ---------------------------------------------------------------------------
# Tool: index_files
# ---------------------------------------------------------------------------

@registry.tool
async def index_files(path: str = ".", pattern: str = "*.py", recursive: bool = True) -> str:
    """Index files for semantic search. Call this before using semantic_search.

    Scans files, splits into chunks, and computes embeddings.
    Already-indexed files (unchanged) are skipped.

    path: Directory to index.
    pattern: Glob pattern for files to index (e.g. '*.py', '*.md', '*.txt').
    recursive: Whether to search subdirectories.
    """
    if _embed_client is None:
        raise ToolError(ToolErrorCode.INTERNAL, "Embedding client not configured.")

    resolved = _check_sandbox(path)
    if not resolved.is_dir():
        raise ToolError(ToolErrorCode.NOT_FOUND, f"'{path}' is not a directory.")

    files = list(resolved.rglob(pattern) if recursive else resolved.glob(pattern))
    files = [f for f in files if f.is_file()]

    indexed = 0
    skipped = 0
    errors = 0

    for fpath in files:
        try:
            text = fpath.read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                continue

            file_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
            rel_path = str(fpath.relative_to(resolved))

            if _store.is_indexed(rel_path, file_hash):
                skipped += 1
                continue

            # Remove old chunks for this file
            _store.remove_file(rel_path)

            # Chunk and embed
            chunks = chunk_text(text, rel_path)
            if not chunks:
                continue

            batch_texts = [c.text for c in chunks]
            embeddings = await _embed_client.embed(batch_texts)

            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb
                _store.add(chunk)

            _store.mark_indexed(rel_path, file_hash)
            indexed += 1
        except Exception:
            errors += 1

    # Persist index
    if _index_path:
        _store.save(_index_path)

    return (
        f"Indexing complete: {indexed} files indexed, {skipped} skipped (unchanged), "
        f"{errors} errors. Total: {_store.total_chunks} chunks from {_store.indexed_files} files."
    )


# ---------------------------------------------------------------------------
# Tool: semantic_search
# ---------------------------------------------------------------------------

@registry.tool
async def semantic_search(query: str, top_k: int = 5) -> str:
    """Search indexed files by meaning (semantic similarity).

    Use this when keyword search (search_files) misses relevant results,
    or when the query is conceptual rather than a literal string match.

    Requires index_files to be called first.

    query: Natural language query describing what you're looking for.
    top_k: Number of results to return.
    """
    if _embed_client is None:
        raise ToolError(ToolErrorCode.INTERNAL, "Embedding client not configured.")

    if _store.total_chunks == 0:
        raise ToolError(
            ToolErrorCode.PRECONDITION,
            "No files indexed yet. Call index_files first.",
        )

    query_emb = await _embed_client.embed_single(query)
    results = _store.search(query_emb, top_k=top_k)

    if not results:
        return f"No relevant results found for: '{query}'"

    parts = [f"Found {len(results)} result(s) for: '{query}'\n"]
    for i, (chunk, score) in enumerate(results, 1):
        parts.append(
            f"[{i}] {chunk.file_path} (lines {chunk.start_line}-{chunk.end_line}) "
            f"— relevance: {score:.3f}\n{chunk.text[:500]}\n"
        )

    return "\n".join(parts)
