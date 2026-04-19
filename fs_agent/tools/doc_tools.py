"""Document processing tools — PDF, DOCX, and rich file preview.

Extends the base file tools with format-specific readers.
Uses optional deps: pymupdf (PDF), python-docx (DOCX), chardet (encoding detection).
"""
from __future__ import annotations

import mimetypes
from pathlib import Path

from scaffold.tools.errors import ToolError, ToolErrorCode

from fs_agent.tools.file_tools import registry, _check_sandbox


# ---------------------------------------------------------------------------
# Tool: read_pdf
# ---------------------------------------------------------------------------

@registry.tool
def read_pdf(path: str, pages: str = "all", max_pages: int = 50) -> str:
    """Read text content from a PDF file. Use when the file has a .pdf extension.

    Prefer this over read_file for PDF files, as read_file returns raw bytes.

    path: Path to the PDF file.
    pages: Which pages to read. 'all' for all pages, or '1-5' for a range, or '1,3,5' for specific pages.
    max_pages: Maximum number of pages to extract (safety limit).
    """
    resolved = _check_sandbox(path)
    if not resolved.is_file():
        raise ToolError(ToolErrorCode.NOT_FOUND, f"'{path}' is not a file.")
    if resolved.suffix.lower() != ".pdf":
        raise ToolError(ToolErrorCode.UNSUPPORTED_FORMAT, f"'{path}' is not a PDF file.")

    try:
        import fitz  # pymupdf
    except ImportError:
        raise ToolError(
            ToolErrorCode.INTERNAL,
            "PDF support requires pymupdf. Install with: pip install pymupdf",
        )

    try:
        doc = fitz.open(str(resolved))
    except Exception as e:
        raise ToolError(ToolErrorCode.INTERNAL, f"Failed to open PDF: {e}")

    total_pages = len(doc)
    page_indices = _parse_page_range(pages, total_pages)
    if len(page_indices) > max_pages:
        page_indices = page_indices[:max_pages]

    parts: list[str] = []
    parts.append(f"[PDF: {resolved.name} | {total_pages} pages | extracting {len(page_indices)} page(s)]")

    for idx in page_indices:
        page = doc[idx]
        text = page.get_text()
        if text.strip():
            parts.append(f"\n--- Page {idx + 1} ---\n{text.strip()}")
        else:
            parts.append(f"\n--- Page {idx + 1} ---\n(no extractable text)")

    doc.close()
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tool: read_docx
# ---------------------------------------------------------------------------

@registry.tool
def read_docx(path: str, max_paragraphs: int = 500) -> str:
    """Read text content from a Word document (.docx). Use when the file has a .docx extension.

    path: Path to the Word document.
    max_paragraphs: Maximum number of paragraphs to extract.
    """
    resolved = _check_sandbox(path)
    if not resolved.is_file():
        raise ToolError(ToolErrorCode.NOT_FOUND, f"'{path}' is not a file.")
    if resolved.suffix.lower() != ".docx":
        raise ToolError(ToolErrorCode.UNSUPPORTED_FORMAT, f"'{path}' is not a .docx file.")

    try:
        from docx import Document
    except ImportError:
        raise ToolError(
            ToolErrorCode.INTERNAL,
            "DOCX support requires python-docx. Install with: pip install python-docx",
        )

    try:
        doc = Document(str(resolved))
    except Exception as e:
        raise ToolError(ToolErrorCode.INTERNAL, f"Failed to open document: {e}")

    paragraphs = []
    for i, para in enumerate(doc.paragraphs):
        if i >= max_paragraphs:
            break
        text = para.text.strip()
        if text:
            # Preserve heading structure
            if para.style and para.style.name.startswith("Heading"):
                level = para.style.name.replace("Heading ", "")
                try:
                    hashes = "#" * int(level)
                except ValueError:
                    hashes = "#"
                paragraphs.append(f"{hashes} {text}")
            else:
                paragraphs.append(text)

    # Also extract tables
    tables_text = []
    for t_idx, table in enumerate(doc.tables):
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append(" | ".join(cells))
        if rows:
            tables_text.append(f"\n[Table {t_idx + 1}]\n" + "\n".join(rows))

    header = f"[DOCX: {resolved.name} | {len(doc.paragraphs)} paragraphs | {len(doc.tables)} tables]"
    body = "\n\n".join(paragraphs[:max_paragraphs])
    tables = "\n".join(tables_text) if tables_text else ""

    return f"{header}\n\n{body}{tables}"


# ---------------------------------------------------------------------------
# Tool: preview_file
# ---------------------------------------------------------------------------

@registry.tool
def preview_file(path: str) -> str:
    """Get a smart preview of any file. Auto-detects format and extracts key content.

    Use this for a quick look at a file when you don't know its format.
    Supports: text, PDF, DOCX, code, CSV, JSON, YAML.

    path: Path to the file to preview.
    """
    resolved = _check_sandbox(path)
    if not resolved.is_file():
        raise ToolError(ToolErrorCode.NOT_FOUND, f"'{path}' is not a file.")

    size = resolved.stat().st_size
    suffix = resolved.suffix.lower()

    # Route to specialized readers
    if suffix == ".pdf":
        return read_pdf(path, pages="1-3", max_pages=3)
    elif suffix == ".docx":
        return read_docx(path, max_paragraphs=30)
    elif suffix in (".csv", ".tsv"):
        return _preview_csv(resolved, suffix)
    elif suffix in (".json", ".yaml", ".yml"):
        return _preview_structured(resolved)
    else:
        return _preview_text(resolved, size)


# ---------------------------------------------------------------------------
# Tool: summarize_file
# ---------------------------------------------------------------------------

@registry.tool
def summarize_file(path: str) -> str:
    """Extract key structural information about a file for quick understanding.

    Returns: file type, size, structure (headings for docs, function names for code, etc.)
    Does NOT read the full content — use read_file or read_pdf for that.

    path: Path to the file.
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

    if suffix == ".pdf":
        try:
            import fitz
            doc = fitz.open(str(resolved))
            info.append(f"Pages: {len(doc)}")
            toc = doc.get_toc()
            if toc:
                info.append("Table of Contents:")
                for level, title, page in toc[:20]:
                    indent = "  " * (level - 1)
                    info.append(f"  {indent}{title} (p.{page})")
            doc.close()
        except ImportError:
            info.append("(PDF details require pymupdf)")
    elif suffix == ".docx":
        try:
            from docx import Document
            doc = Document(str(resolved))
            info.append(f"Paragraphs: {len(doc.paragraphs)}")
            info.append(f"Tables: {len(doc.tables)}")
            headings = [p.text for p in doc.paragraphs
                        if p.style and p.style.name.startswith("Heading")]
            if headings:
                info.append("Headings:")
                for h in headings[:15]:
                    info.append(f"  - {h}")
        except ImportError:
            info.append("(DOCX details require python-docx)")
    elif suffix in (".py", ".js", ".ts", ".java", ".go", ".rs"):
        info.extend(_summarize_code(resolved))
    elif suffix in (".csv", ".tsv"):
        info.extend(_summarize_csv(resolved, suffix))

    return "\n".join(info)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_page_range(spec: str, total: int) -> list[int]:
    """Parse page specification into 0-based indices."""
    if spec.strip().lower() == "all":
        return list(range(total))
    indices = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            start = max(0, int(a) - 1)
            end = min(total, int(b))
            indices.extend(range(start, end))
        else:
            idx = int(part) - 1
            if 0 <= idx < total:
                indices.append(idx)
    return indices


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
    # Truncate to 3000 chars for preview
    if len(text) > 3000:
        return f"[{path.suffix.upper()}: {path.name} | {len(text):,} chars]\n{text[:3000]}\n... (truncated)"
    return f"[{path.suffix.upper()}: {path.name} | {len(text):,} chars]\n{text}"


def _preview_text(path: Path, size: int) -> str:
    try:
        # Detect encoding
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
    """Extract function/class names from source code."""
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
                name = stripped.split("(")[0].replace("def ", "")
                functions.append(name)
            elif stripped.startswith("class "):
                name = stripped.split("(")[0].split(":")[0].replace("class ", "")
                classes.append(name)

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
