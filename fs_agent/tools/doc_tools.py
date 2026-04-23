"""文档处理工具 —— 统一读取、预览与语义段落检索。

使用 MarkItDown 将 PDF、DOCX、PPTX、XLSX 等格式统一转换为 Markdown。
search_document 工具在首次调用时对文档分块建立向量索引，后续查询复用缓存。

可选依赖：markitdown[pdf,docx]、chardet（编码检测）。
"""
from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path

from scaffold.tools.errors import ToolError, ToolErrorCode

from fs_agent.tools.file_tools import registry, _check_sandbox

# MarkItDown 能处理的富文档格式（除此之外走普通文本路径）
_RICH_FORMATS = {".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".html", ".htm", ".msg", ".eml"}

# 文档向量索引缓存：str(resolved_path) -> (file_hash, VectorStore)
_doc_stores: dict = {}


# ---------------------------------------------------------------------------
# 工具：read_document
# ---------------------------------------------------------------------------

@registry.tool
def read_document(path: str, max_chars: int = 50_000) -> str:
    """读取文档内容并转换为 Markdown，支持 PDF、DOCX、PPTX、XLSX 等格式。

    对 .pdf / .docx / .pptx / .xlsx 等富文本格式，优先使用它而非 read_file
    （read_file 返回的是原始字节）。转换结果保留标题、表格等文档结构。

    若只需查找文档中的特定内容，请改用 search_document——它只返回相关段落，
    不会把整个文档塞入上下文。

    path: 文档路径。
    max_chars: 最多返回的字符数（默认 50000）。
    """
    resolved = _check_sandbox(path)
    if not resolved.is_file():
        raise ToolError(ToolErrorCode.NOT_FOUND, f"'{path}' is not a file.")

    try:
        from markitdown import MarkItDown
    except ImportError:
        raise ToolError(
            ToolErrorCode.INTERNAL_ERROR,
            "Document support requires markitdown. Install with: pip install 'markitdown[pdf,docx]'",
        )

    try:
        md = MarkItDown()
        result = md.convert(str(resolved))
        text = result.text_content
    except Exception as e:
        raise ToolError(ToolErrorCode.INTERNAL_ERROR, f"Failed to read document: {e}")

    fmt = resolved.suffix.upper().lstrip(".")
    header = f"[{fmt}: {resolved.name}]"
    if len(text) > max_chars:
        return f"{header}\n{text[:max_chars]}\n... (truncated, {len(text):,} total chars)"
    return f"{header}\n{text}"


# ---------------------------------------------------------------------------
# 工具：search_document
# ---------------------------------------------------------------------------

@registry.tool
async def search_document(path: str, query: str, top_k: int = 5) -> str:
    """在单个文档内按语义相似度检索相关段落。

    适合大文档的定向查询，无需读取全文。首次调用时自动将文档分块并建立
    向量索引；文件内容未变时直接复用缓存，无需重复嵌入。

    需要配置 Embedding 服务（EMBEDDING_API_KEY + EMBEDDING_MODEL_NAME）。

    path: 文档路径（PDF、DOCX、PPTX、XLSX 等）。
    query: 用自然语言描述要查找的内容。
    top_k: 返回的相关段落数量。
    """
    try:
        from fs_agent.tools.search_tools import get_embed_client, VectorStore, Chunk
    except ImportError:
        raise ToolError(ToolErrorCode.INTERNAL_ERROR, "search_tools module not available.")

    embed_client = get_embed_client()
    if embed_client is None:
        raise ToolError(
            ToolErrorCode.INTERNAL_ERROR,
            "Embedding client not configured. Set EMBEDDING_API_KEY and EMBEDDING_MODEL_NAME in .env.",
        )

    resolved = _check_sandbox(path)
    if not resolved.is_file():
        raise ToolError(ToolErrorCode.NOT_FOUND, f"'{path}' is not a file.")

    file_bytes = resolved.read_bytes()
    file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
    str_path = str(resolved)

    cached = _doc_stores.get(str_path)
    if cached is None or cached[0] != file_hash:
        try:
            from markitdown import MarkItDown
        except ImportError:
            raise ToolError(
                ToolErrorCode.INTERNAL_ERROR,
                "markitdown required. Install with: pip install 'markitdown[pdf,docx]'",
            )
        try:
            md_parser = MarkItDown()
            result = md_parser.convert(str_path)
            text = result.text_content
        except Exception as e:
            raise ToolError(ToolErrorCode.INTERNAL_ERROR, f"Failed to parse document: {e}")

        if not text.strip():
            raise ToolError(ToolErrorCode.INTERNAL_ERROR, "Document appears to be empty.")

        raw_chunks = _chunk_document(text)
        embeddings = await embed_client.embed(raw_chunks)

        store = VectorStore()
        for chunk_text, emb in zip(raw_chunks, embeddings):
            store.add(Chunk(
                text=chunk_text,
                file_path=resolved.name,
                start_line=0,
                end_line=0,
                file_hash=file_hash,
                embedding=emb,
            ))

        _doc_stores[str_path] = (file_hash, store)
    else:
        store = cached[1]

    query_emb = await embed_client.embed_single(query)
    results = store.search(query_emb, top_k=top_k)

    if not results:
        return f"No relevant content found in '{resolved.name}' for: '{query}'"

    parts = [f"Top {len(results)} passage(s) in '{resolved.name}' for: '{query}'\n"]
    for i, (chunk, score) in enumerate(results, 1):
        parts.append(f"[{i}] relevance: {score:.3f}\n{chunk.text}\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 工具：preview_file
# ---------------------------------------------------------------------------

@registry.tool
def preview_file(path: str) -> str:
    """获取任意文件的智能预览，自动检测格式并提取关键信息。

    当你不了解文件格式，只想快速看一眼时可以使用它。
    支持：PDF、DOCX、PPTX、XLSX、文本、代码、CSV、JSON、YAML。

    path: 需要预览的文件路径。
    """
    resolved = _check_sandbox(path)
    if not resolved.is_file():
        raise ToolError(ToolErrorCode.NOT_FOUND, f"'{path}' is not a file.")

    size = resolved.stat().st_size
    suffix = resolved.suffix.lower()

    if suffix in _RICH_FORMATS:
        return read_document(path, max_chars=5_000)
    elif suffix in (".csv", ".tsv"):
        return _preview_csv(resolved, suffix)
    elif suffix in (".json", ".yaml", ".yml"):
        return _preview_structured(resolved)
    else:
        return _preview_text(resolved, size)


# ---------------------------------------------------------------------------
# 工具：summarize_file
# ---------------------------------------------------------------------------

@registry.tool
def summarize_file(path: str) -> str:
    """提取文件的关键结构信息，帮助快速理解内容。

    返回内容包括：文件类型、大小、结构信息（例如文档标题、代码函数名等）。
    它不会读取完整内容；若需全文，请使用 read_file 或 read_document。

    path: 文件路径。
    """
    resolved = _check_sandbox(path)
    if not resolved.is_file():
        raise ToolError(ToolErrorCode.NOT_FOUND, f"'{path}' is not a file.")

    size = resolved.stat().st_size
    suffix = resolved.suffix.lower()
    mime, _ = mimetypes.guess_type(str(resolved))

    info: list[str] = [
        f"File: {resolved.name}",
        f"Size: {_human_size(size)}",
        f"Type: {mime or 'unknown'} ({suffix or 'no extension'})",
    ]

    if suffix in _RICH_FORMATS:
        try:
            from markitdown import MarkItDown
            md = MarkItDown()
            result = md.convert(str(resolved))
            text = result.text_content
            headings = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("#")]
            if headings:
                info.append("Headings:")
                for h in headings[:15]:
                    info.append(f"  {h}")
        except ImportError:
            info.append("(document details require markitdown)")
        except Exception as e:
            info.append(f"(failed to extract structure: {e})")
    elif suffix in (".py", ".js", ".ts", ".java", ".go", ".rs"):
        info.extend(_summarize_code(resolved))
    elif suffix in (".csv", ".tsv"):
        info.extend(_summarize_csv(resolved, suffix))

    return "\n".join(info)


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

def _chunk_document(text: str, min_chars: int = 200, max_chars: int = 800) -> list[str]:
    """Mobius 风格的段落分块：按双换行分段，合并过短的碎片。"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    buffer = ""

    for para in paragraphs:
        candidate = (buffer + "\n\n" + para).strip() if buffer else para
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer)
            # 如果单个段落超出 max_chars，强制截断为多块
            while len(para) > max_chars:
                chunks.append(para[:max_chars])
                para = para[max_chars:]
            buffer = para

    if buffer:
        chunks.append(buffer)

    return chunks


def _preview_csv(path: Path, suffix: str) -> str:
    import csv
    delimiter = "\t" if suffix == ".tsv" else ","
    lines: list[str] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for i, row in enumerate(reader):
            if i >= 20:
                lines.append("... (showing first 20 rows)")
                break
            lines.append(" | ".join(row))
    header = f"[CSV: {path.name} | delimiter='{delimiter}']"
    return header + "\n" + "\n".join(lines)


def _preview_structured(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > 3000:
        return f"[{path.suffix.upper()}: {path.name} | {len(text):,} chars]\n{text[:3000]}\n... (truncated)"
    return f"[{path.suffix.upper()}: {path.name} | {len(text):,} chars]\n{text}"


def _preview_text(path: Path, size: int) -> str:
    try:
        try:
            import chardet
            raw = path.read_bytes()[:10000]
            detected = chardet.detect(raw)
            encoding = detected.get("encoding", "utf-8") or "utf-8"
        except ImportError:
            encoding = "utf-8"

        text = path.read_text(encoding=encoding, errors="replace")
        preview = text[:3000]
        header = f"[Text: {path.name} | {_human_size(size)} | encoding={encoding}]"
        if len(text) > 3000:
            return f"{header}\n{preview}\n... (truncated, {len(text):,} total chars)"
        return f"{header}\n{preview}"
    except Exception:
        return f"[Binary file: {path.name} | {_human_size(size)}] (cannot preview)"


def _summarize_code(path: Path) -> list[str]:
    info: list[str] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        info.append(f"Lines: {len(lines)}")
        functions = []
        classes = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("def "):
                functions.append(stripped.split("(")[0].replace("def ", ""))
            elif stripped.startswith("class "):
                classes.append(stripped.split("(")[0].split(":")[0].replace("class ", ""))
        if classes:
            info.append(f"Classes: {', '.join(classes[:10])}")
        if functions:
            info.append(f"Functions: {', '.join(functions[:15])}")
    except Exception:
        pass
    return info


def _summarize_csv(path: Path, suffix: str) -> list[str]:
    import csv
    info: list[str] = []
    delimiter = "\t" if suffix == ".tsv" else ","
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader, None)
            row_count = sum(1 for _ in reader) + (1 if header else 0)
            if header:
                info.append(f"Columns: {', '.join(header)}")
            info.append(f"Rows: {row_count}")
    except Exception:
        pass
    return info


def _human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024  # type: ignore
    return f"{size:.1f} TB"
