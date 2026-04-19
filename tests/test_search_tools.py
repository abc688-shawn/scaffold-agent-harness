"""Tests for search tools — vector store, chunking, cosine similarity."""
from __future__ import annotations

import pytest
import math

from fs_agent.tools.search_tools import (
    Chunk,
    VectorStore,
    chunk_text,
    _cosine_similarity,
)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine_similarity(a, b) + 1.0) < 1e-6

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert _cosine_similarity(a, b) == 0.0


class TestChunking:
    def test_basic_chunking(self):
        text = "\n".join(f"Line {i}" for i in range(100))
        chunks = chunk_text(text, "test.py", chunk_size=20, overlap=5)
        assert len(chunks) > 1
        assert all(c.file_path == "test.py" for c in chunks)
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 20

    def test_empty_text(self):
        chunks = chunk_text("", "empty.py")
        assert chunks == []

    def test_small_text_single_chunk(self):
        text = "Hello\nWorld"
        chunks = chunk_text(text, "small.py", chunk_size=500)
        assert len(chunks) == 1
        assert "Hello" in chunks[0].text
        assert "World" in chunks[0].text

    def test_file_hash_set(self):
        text = "Line 1\nLine 2"
        chunks = chunk_text(text, "hashed.py")
        assert all(c.file_hash for c in chunks)
        # Same text → same hash
        chunks2 = chunk_text(text, "other.py")
        assert chunks[0].file_hash == chunks2[0].file_hash

    def test_overlap(self):
        text = "\n".join(f"L{i}" for i in range(30))
        chunks = chunk_text(text, "f.py", chunk_size=10, overlap=3)
        # Second chunk should start before first chunk ends
        assert len(chunks) >= 2
        assert chunks[1].start_line < chunks[0].end_line + 1


class TestVectorStore:
    def _make_chunk(self, text: str, path: str, embedding: list[float]) -> Chunk:
        return Chunk(
            text=text, file_path=path, start_line=1, end_line=1,
            file_hash="abc", embedding=embedding,
        )

    def test_add_and_search(self):
        store = VectorStore()
        store.add(self._make_chunk("python code", "a.py", [1.0, 0.0, 0.0]))
        store.add(self._make_chunk("javascript code", "b.js", [0.0, 1.0, 0.0]))
        store.add(self._make_chunk("more python", "c.py", [0.9, 0.1, 0.0]))

        results = store.search([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        # First result should be most similar
        assert results[0][0].text == "python code"
        assert results[0][1] > results[1][1]

    def test_empty_search(self):
        store = VectorStore()
        results = store.search([1.0, 0.0], top_k=5)
        assert results == []

    def test_remove_file(self):
        store = VectorStore()
        store.add(self._make_chunk("text1", "a.py", [1.0, 0.0]))
        store.add(self._make_chunk("text2", "a.py", [0.0, 1.0]))
        store.add(self._make_chunk("text3", "b.py", [0.5, 0.5]))
        store.mark_indexed("a.py", "hash_a")
        store.mark_indexed("b.py", "hash_b")

        assert store.total_chunks == 3
        removed = store.remove_file("a.py")
        assert removed == 2
        assert store.total_chunks == 1

    def test_is_indexed(self):
        store = VectorStore()
        store.mark_indexed("a.py", "hash1")
        assert store.is_indexed("a.py", "hash1")
        assert not store.is_indexed("a.py", "hash2")
        assert not store.is_indexed("b.py", "hash1")

    def test_save_and_load(self, tmp_path):
        store = VectorStore()
        store.add(self._make_chunk("hello", "a.py", [1.0, 0.0]))
        store.mark_indexed("a.py", "abc")

        path = tmp_path / "index.pkl"
        store.save(str(path))

        store2 = VectorStore()
        store2.load(str(path))
        assert store2.total_chunks == 1
        assert store2.is_indexed("a.py", "abc")

    def test_top_k_limit(self):
        store = VectorStore()
        for i in range(10):
            store.add(self._make_chunk(f"text_{i}", f"f{i}.py", [float(i), 0.0]))
        results = store.search([9.0, 0.0], top_k=3)
        assert len(results) == 3
