"""语义搜索工具 —— 基于 embedding 的工作区检索。

使用轻量级内存向量库。Embedding 可以懒加载计算
（首次搜索时）或通过 `index_workspace` 显式建立。

支持可插拔的 embedding 后端；默认使用 OpenAI 兼容的
embedding 接口（也适用于 DeepSeek 等服务）。
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
# 向量库（轻量、内存型、可用 pickle 持久化）
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """带有 embedding 和来源信息的文本分块。"""
    text: str
    file_path: str
    start_line: int
    end_line: int
    file_hash: str
    embedding: list[float] = field(default_factory=list)


class VectorStore:
    """支持余弦相似度搜索的简单内存向量库。"""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._file_hashes: dict[str, str] = {}  # 文件路径 -> 内容哈希

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
# Embedding 客户端
# ---------------------------------------------------------------------------

class EmbeddingClient:
    """OpenAI 兼容的 embedding 客户端。"""

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
        """为一批文本计算 embedding。"""
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
# 文本分块
# ---------------------------------------------------------------------------

def chunk_text(text: str, file_path: str, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    """按行将文本切分为带重叠区域的多个块。"""
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
# 模块级状态（启动时配置）
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
# 工具：index_files
# ---------------------------------------------------------------------------

@registry.tool
async def index_files(path: str = ".", pattern: str = "*.py", recursive: bool = True) -> str:
    """为语义搜索建立文件索引。使用 `semantic_search` 前应先调用它。

    它会扫描文件、切分文本块并计算 embedding。
    已经建立索引且内容未变化的文件会被跳过。

    path: 要建立索引的目录。
    pattern: 需要索引的文件 glob 模式（例如 `*.py`、`*.md`、`*.txt`）。
    recursive: 是否递归搜索子目录。
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

            # 移除该文件的旧分块
            _store.remove_file(rel_path)

            # 切分并计算 embedding
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

    # 持久化索引
    if _index_path:
        _store.save(_index_path)

    return (
        f"Indexing complete: {indexed} files indexed, {skipped} skipped (unchanged), "
        f"{errors} errors. Total: {_store.total_chunks} chunks from {_store.indexed_files} files."
    )


# ---------------------------------------------------------------------------
# 工具：semantic_search
# ---------------------------------------------------------------------------

@registry.tool
async def semantic_search(query: str, top_k: int = 5) -> str:
    """按语义相似度搜索已建立索引的文件内容。

    当关键词搜索（`search_files`）找不到相关结果，
    或查询更偏概念性而非字面字符串匹配时，适合使用它。

    使用前需要先调用 `index_files`。

    query: 用自然语言描述你要找的内容。
    top_k: 返回结果数量。
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
