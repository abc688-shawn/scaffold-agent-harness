"""语义搜索工具 —— 基于 ChromaDB (HNSW ANN) 的工作区与文档检索。

Embedding 通过 OpenAI 兼容接口计算（支持 DeepSeek 等服务）。
持久化由 ChromaDB PersistentClient 负责，落到 <workspace>/.chroma/。
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from scaffold.tools.errors import ToolError, ToolErrorCode

from fs_agent.tools.file_tools import registry, _check_sandbox


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
            raise ToolError(ToolErrorCode.INTERNAL_ERROR, "openai package required for embeddings")

        client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url or None)
        response = await client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]

    async def embed_single(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0]


# ---------------------------------------------------------------------------
# 文本分块
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[tuple[str, int, int, str]]:
    """按行将文本切分为带重叠区域的多个块。

    返回：list of (chunk_text, start_line, end_line, file_hash)
    """
    lines = text.splitlines()
    if not lines:
        return []

    file_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
    step = max(1, chunk_size - overlap)
    chunks: list[tuple[str, int, int, str]] = []
    i = 0
    while i < len(lines):
        end = min(i + chunk_size, len(lines))
        chunk_str = "\n".join(lines[i:end])
        if chunk_str.strip():
            chunks.append((chunk_str, i + 1, end, file_hash))
        i += step
    return chunks


# ---------------------------------------------------------------------------
# 模块级状态（启动时由 configure_search 初始化）
# ---------------------------------------------------------------------------

_embed_client: EmbeddingClient | None = None
_collection = None       # chromadb.Collection — workspace 文件索引
_doc_collection = None   # chromadb.Collection — 单文档段落索引


def configure_search(
    embed_client: EmbeddingClient,
    persist_dir: str | Path | None = None,
) -> None:
    """初始化 embedding 客户端和 ChromaDB collections。

    persist_dir: ChromaDB 持久化目录（None 则使用纯内存 EphemeralClient）。
    """
    global _embed_client, _collection, _doc_collection
    _embed_client = embed_client
    try:
        from fs_agent.tools._chroma import (
            get_client,
            get_or_create_collection,
            SyncEmbeddingFunction,
        )
        embed_fn = SyncEmbeddingFunction(embed_client)
        client = get_client(Path(persist_dir) if persist_dir else None)
        _collection = get_or_create_collection(client, "workspace_index", embed_fn)
        _doc_collection = get_or_create_collection(client, "documents_index", embed_fn)
    except ImportError:
        # chromadb 未安装；工具首次调用时会给出明确报错
        pass


def get_embed_client() -> EmbeddingClient | None:
    """返回当前配置的 embedding 客户端（未配置时返回 None）。"""
    return _embed_client


def _get_doc_collection():
    """返回文档索引 collection（未配置时抛出 ToolError）。"""
    if _doc_collection is None:
        if _embed_client is None:
            raise ToolError(
                ToolErrorCode.INTERNAL_ERROR,
                "Embedding client not configured. Set EMBEDDING_API_KEY and EMBEDDING_MODEL_NAME.",
            )
        raise ToolError(
            ToolErrorCode.INTERNAL_ERROR,
            "ChromaDB not available. Install with: uv sync --extra fs",
        )
    return _doc_collection


# ---------------------------------------------------------------------------
# 工具：index_files
# ---------------------------------------------------------------------------

@registry.tool
async def index_files(path: str = ".", pattern: str = "*.py", recursive: bool = True) -> str:
    """为语义搜索建立文件索引。使用 `semantic_search` 前应先调用它。

    它会扫描文件、切分文本块并计算 embedding。
    已建立索引且内容未变化的文件会被跳过。

    path: 要建立索引的目录。
    pattern: 需要索引的文件 glob 模式（例如 `*.py`、`*.md`、`*.txt`）。
    recursive: 是否递归搜索子目录。
    """
    if _embed_client is None or _collection is None:
        raise ToolError(
            ToolErrorCode.INTERNAL_ERROR,
            "Embedding client not configured. Set EMBEDDING_API_KEY and EMBEDDING_MODEL_NAME.",
        )

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

            # 同路径 + 同 hash 已存在 → 跳过
            existing = _collection.get(
                where={"$and": [{"file_path": rel_path}, {"file_hash": file_hash}]},
                limit=1,
                include=["documents"],
            )
            if existing["documents"]:
                skipped += 1
                continue

            # 删掉该文件的旧分块（hash 变了）
            _collection.delete(where={"file_path": rel_path})

            chunks = chunk_text(text)
            if not chunks:
                continue

            ids = [f"{rel_path}#{s}-{e}" for (_, s, e, _) in chunks]
            docs = [t for (t, _, _, _) in chunks]
            metas = [
                {"file_path": rel_path, "start_line": s, "end_line": e, "file_hash": h}
                for (_, s, e, h) in chunks
            ]
            # Chroma 内部调用 SyncEmbeddingFunction 计算 embedding
            _collection.add(ids=ids, documents=docs, metadatas=metas)
            indexed += 1
        except Exception:
            errors += 1

    total_chunks = _collection.count()
    return (
        f"Indexing complete: {indexed} files indexed, {skipped} skipped (unchanged), "
        f"{errors} errors. Collection now has {total_chunks} chunks."
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
    if _embed_client is None or _collection is None:
        raise ToolError(
            ToolErrorCode.INTERNAL_ERROR,
            "Embedding client not configured. Set EMBEDDING_API_KEY and EMBEDDING_MODEL_NAME.",
        )

    total = _collection.count()
    if total == 0:
        raise ToolError(
            ToolErrorCode.INVALID_ARGUMENTS,
            "No files indexed yet. Call index_files first.",
        )

    actual_n = min(top_k, total)
    res = _collection.query(
        query_texts=[query],
        n_results=actual_n,
        include=["documents", "metadatas", "distances"],
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    if not docs:
        return f"No relevant results found for: '{query}'"

    parts = [f"Found {len(docs)} result(s) for: '{query}'\n"]
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
        relevance = 1.0 - dist  # cosine distance: 0=identical → relevance=1
        parts.append(
            f"[{i}] {meta['file_path']} (lines {meta['start_line']}-{meta['end_line']}) "
            f"— relevance: {relevance:.3f}\n{doc[:500]}\n"
        )
    return "\n".join(parts)
