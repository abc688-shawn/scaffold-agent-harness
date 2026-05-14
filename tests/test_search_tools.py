"""搜索工具测试 —— ChromaDB 集成、分块与工具端到端行为。"""
from __future__ import annotations

import hashlib
import math
import pytest

from fs_agent.tools.search_tools import chunk_text


# ---------------------------------------------------------------------------
# Fake EmbeddingClient（无 API 调用，基于 sha256 确定性向量）
# ---------------------------------------------------------------------------

class FakeEmbeddingClient:
    """Deterministic hash-based embedding for testing (no API calls)."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.call_count = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        result = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            vec = [b / 255.0 for b in h[: self.dim]]
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            result.append([x / norm for x in vec])
        return result

    async def embed_single(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chroma_env(tmp_path):
    from fs_agent.tools import _chroma, search_tools

    _chroma.reset_clients()
    persist = tmp_path / ".chroma"
    fake = FakeEmbeddingClient()
    search_tools.configure_search(fake, persist_dir=persist)
    yield persist, fake

    search_tools._collection = None
    search_tools._doc_collection = None
    search_tools._embed_client = None
    _chroma.reset_clients()


@pytest.fixture
def workspace_env(tmp_path, chroma_env):
    """Chroma + sandbox wired to a fresh workspace directory."""
    from scaffold.safety.sandbox import PathSandbox
    from fs_agent.tools.file_tools import set_sandbox

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    set_sandbox(PathSandbox([str(workspace)]))
    yield workspace, chroma_env[1]  # (workspace_path, fake_client)


# ---------------------------------------------------------------------------
# TestChunking
# ---------------------------------------------------------------------------

class TestChunking:
    def test_basic_chunking(self):
        text = "\n".join(f"Line {i}" for i in range(100))
        chunks = chunk_text(text, chunk_size=20, overlap=5)
        assert len(chunks) > 1
        first_text, start, end, _ = chunks[0]
        assert start == 1
        assert end == 20

    def test_empty_text(self):
        assert chunk_text("") == []

    def test_small_text_single_chunk(self):
        text = "Hello\nWorld"
        chunks = chunk_text(text, chunk_size=500)
        assert len(chunks) == 1
        chunk_str, _, _, _ = chunks[0]
        assert "Hello" in chunk_str
        assert "World" in chunk_str

    def test_file_hash_set(self):
        text = "Line 1\nLine 2"
        chunks = chunk_text(text)
        assert all(h for _, _, _, h in chunks)
        chunks2 = chunk_text(text)
        assert chunks[0][3] == chunks2[0][3]  # same text → same hash

    def test_overlap(self):
        text = "\n".join(f"L{i}" for i in range(30))
        chunks = chunk_text(text, chunk_size=10, overlap=3)
        assert len(chunks) >= 2
        _, _, end0, _ = chunks[0]
        _, start1, _, _ = chunks[1]
        assert start1 < end0 + 1  # overlap → second chunk starts before first ends


# ---------------------------------------------------------------------------
# TestWorkspaceCollection  — 直接操作 Chroma collection
# ---------------------------------------------------------------------------

class TestWorkspaceCollection:
    def test_collections_initialized(self, chroma_env):
        from fs_agent.tools import search_tools

        assert search_tools._collection is not None
        assert search_tools._doc_collection is not None

    def test_add_and_query(self, chroma_env):
        from fs_agent.tools import search_tools

        coll = search_tools._collection
        coll.add(
            ids=["a#1-5"],
            documents=["python async programming tutorial"],
            metadatas=[{"file_path": "a.py", "start_line": 1, "end_line": 5, "file_hash": "h1"}],
        )
        res = coll.query(query_texts=["async python"], n_results=1, include=["documents"])
        assert len(res["documents"][0]) == 1
        assert "python" in res["documents"][0][0]

    def test_top_k_limit(self, chroma_env):
        from fs_agent.tools import search_tools

        coll = search_tools._collection
        for i in range(10):
            coll.add(
                ids=[f"file{i}#1-1"],
                documents=[f"content chunk number {i}"],
                metadatas=[{"file_path": f"f{i}.py", "start_line": 1, "end_line": 1, "file_hash": f"h{i}"}],
            )
        res = coll.query(query_texts=["content"], n_results=3, include=["documents"])
        assert len(res["documents"][0]) == 3

    def test_persistence_across_restart(self, tmp_path):
        from fs_agent.tools import _chroma, search_tools

        persist = tmp_path / ".chroma"
        fake = FakeEmbeddingClient()

        _chroma.reset_clients()
        search_tools.configure_search(fake, persist_dir=persist)
        search_tools._collection.add(
            ids=["persistent#1-1"],
            documents=["data that should survive restart"],
            metadatas=[{"file_path": "p.py", "start_line": 1, "end_line": 1, "file_hash": "abc"}],
        )
        count = search_tools._collection.count()

        # Simulate restart: re-init with same persist_dir
        _chroma.reset_clients()
        search_tools.configure_search(fake, persist_dir=persist)
        assert search_tools._collection.count() == count

        search_tools._collection = None
        search_tools._doc_collection = None
        search_tools._embed_client = None
        _chroma.reset_clients()

    async def test_empty_collection_raises(self, chroma_env):
        from fs_agent.tools import search_tools
        from scaffold.tools.errors import ToolError

        with pytest.raises(ToolError):
            await search_tools.semantic_search("anything")


# ---------------------------------------------------------------------------
# TestIndexFilesTool — 端到端（需要 sandbox）
# ---------------------------------------------------------------------------

class TestIndexFilesTool:
    async def test_index_and_skip(self, workspace_env):
        from fs_agent.tools import search_tools

        workspace, fake = workspace_env
        (workspace / "main.py").write_text("def main():\n    print('hello')\n")
        (workspace / "utils.py").write_text("def add(a, b):\n    return a + b\n")

        result = await search_tools.index_files(str(workspace), "*.py")
        assert "2 files indexed" in result
        assert "0 skipped" in result

        call_count_after_first = fake.call_count

        result2 = await search_tools.index_files(str(workspace), "*.py")
        assert "0 files indexed" in result2
        assert "2 skipped" in result2
        assert fake.call_count == call_count_after_first  # no new embed calls on skip

    async def test_rehash_on_content_change(self, workspace_env):
        from fs_agent.tools import search_tools

        workspace, _ = workspace_env
        f = workspace / "code.py"
        f.write_text("def foo(): pass\n")
        await search_tools.index_files(str(workspace), "*.py")

        f.write_text("def bar(): return 42\n")
        result = await search_tools.index_files(str(workspace), "*.py")
        assert "1 files indexed" in result

        # Old content should be gone; new content should be present
        stale = search_tools._collection.get(
            where={"file_path": "code.py"}, include=["documents"]
        )
        for doc in stale["documents"]:
            assert "foo" not in doc


# ---------------------------------------------------------------------------
# TestSemanticSearchTool
# ---------------------------------------------------------------------------

class TestSemanticSearchTool:
    async def test_search_returns_results(self, workspace_env):
        from fs_agent.tools import search_tools

        workspace, _ = workspace_env
        (workspace / "vector.py").write_text(
            "def cosine_similarity(a, b):\n    return sum(x*y for x,y in zip(a,b))\n"
        )
        await search_tools.index_files(str(workspace), "*.py")

        result = await search_tools.semantic_search("similarity function", top_k=1)
        assert "result" in result.lower()
        assert "vector.py" in result
        assert "relevance" in result

    async def test_search_respects_top_k(self, workspace_env):
        from fs_agent.tools import search_tools

        workspace, _ = workspace_env
        for i in range(5):
            (workspace / f"file{i}.py").write_text(f"def func_{i}(): pass\n")
        await search_tools.index_files(str(workspace), "*.py")

        result = await search_tools.semantic_search("function definition", top_k=2)
        assert result.count("[1]") == 1
        assert result.count("[2]") == 1
        assert "[3]" not in result
