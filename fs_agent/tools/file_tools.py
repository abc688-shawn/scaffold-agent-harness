"""Core file-system tools for the fs-agent.

Each tool is registered via the @tool decorator and produces an OpenAI
function-call schema automatically.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from scaffold.tools.registry import ToolRegistry
from scaffold.tools.errors import ToolError, ToolErrorCode
from scaffold.safety.sandbox import PathSandbox

# The fs-agent has its own dedicated registry
registry = ToolRegistry()

# Sandbox will be configured at startup
_sandbox: PathSandbox | None = None


def set_sandbox(sandbox: PathSandbox) -> None:
    global _sandbox
    _sandbox = sandbox


def _check_sandbox(path: str) -> Path:
    if _sandbox is None:
        return Path(path).resolve()
    return _sandbox.validate(path)


# ---------------------------------------------------------------------------
# Tool: list_files
# ---------------------------------------------------------------------------

@registry.tool
def list_files(path: str = ".", recursive: bool = False, pattern: str = "*") -> str:
    """List files in a directory. Use this to discover what files are available.

    path: Directory path to list. Defaults to current workspace root.
    recursive: If true, list files recursively in subdirectories.
    pattern: Glob pattern to filter files (e.g. '*.py', '*.pdf').
    """
    resolved = _check_sandbox(path)
    if not resolved.is_dir():
        raise ToolError(ToolErrorCode.NOT_FOUND, f"'{path}' is not a directory.")

    if recursive:
        files = list(resolved.rglob(pattern))
    else:
        files = list(resolved.glob(pattern))

    # Limit output to avoid context explosion
    max_items = 200
    entries = []
    for f in sorted(files)[:max_items]:
        try:
            st = f.stat()
            kind = "dir" if f.is_dir() else "file"
            size = st.st_size
            mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
            rel = f.relative_to(resolved)
            entries.append(f"{kind:4s}  {size:>10,}  {mtime}  {rel}")
        except (OSError, ValueError):
            continue

    header = f"Directory: {resolved}  ({len(files)} items"
    if len(files) > max_items:
        header += f", showing first {max_items}"
    header += ")\n"

    return header + "\n".join(entries) if entries else header + "(empty directory)"


# ---------------------------------------------------------------------------
# Tool: read_file
# ---------------------------------------------------------------------------

@registry.tool
def read_file(path: str, offset: int = 0, length: int = 10000) -> str:
    """Read the text content of a file. Use when you need to see file contents.

    Prefer search_files when looking for specific content across many files.

    path: Path to the file to read.
    offset: Character offset to start reading from (for large files).
    length: Maximum number of characters to read. Use -1 for entire file (caution: large files).
    """
    resolved = _check_sandbox(path)
    if not resolved.is_file():
        raise ToolError(ToolErrorCode.NOT_FOUND, f"'{path}' is not a file.")

    size = resolved.stat().st_size
    if size > 10_000_000:  # 10 MB
        raise ToolError(
            ToolErrorCode.FILE_TOO_LARGE,
            f"File is {size:,} bytes. Use offset/length to read in chunks.",
        )

    try:
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except UnicodeDecodeError:
        raise ToolError(ToolErrorCode.UNSUPPORTED_FORMAT, "File is not valid UTF-8 text.")

    if length == -1:
        chunk = text[offset:]
    else:
        chunk = text[offset:offset + length]

    total_chars = len(text)
    end_offset = offset + len(chunk)
    header = f"[File: {resolved.name} | {total_chars:,} chars total | showing {offset}-{end_offset}]\n"
    return header + chunk


# ---------------------------------------------------------------------------
# Tool: search_files
# ---------------------------------------------------------------------------

@registry.tool
def search_files(query: str, path: str = ".", pattern: str = "*",
                 max_results: int = 20) -> str:
    """Search for files containing a text query. Use when looking for specific content.

    Prefer this over read_file when you don't know which file has the information.

    query: Text to search for (case-insensitive substring match).
    path: Directory to search in.
    pattern: Glob pattern for file types (e.g. '*.py').
    max_results: Maximum number of matching files to return.
    """
    resolved = _check_sandbox(path)
    if not resolved.is_dir():
        raise ToolError(ToolErrorCode.NOT_FOUND, f"'{path}' is not a directory.")

    results = []
    query_lower = query.lower()
    for fpath in resolved.rglob(pattern):
        if not fpath.is_file():
            continue
        try:
            text = fpath.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        if query_lower in text.lower():
            # Find first matching line
            for i, line in enumerate(text.splitlines(), 1):
                if query_lower in line.lower():
                    snippet = line.strip()[:120]
                    rel = fpath.relative_to(resolved)
                    results.append(f"{rel}:{i}: {snippet}")
                    break
        if len(results) >= max_results:
            break

    if not results:
        return f"No files matching '{query}' found in {path}"
    return f"Found {len(results)} match(es):\n" + "\n".join(results)


# ---------------------------------------------------------------------------
# Tool: file_info
# ---------------------------------------------------------------------------

@registry.tool
def file_info(path: str) -> str:
    """Get detailed metadata about a file (size, dates, type).

    path: Path to the file.
    """
    resolved = _check_sandbox(path)
    if not resolved.exists():
        raise ToolError(ToolErrorCode.NOT_FOUND, f"'{path}' does not exist.")

    st = resolved.stat()
    info = {
        "name": resolved.name,
        "path": str(resolved),
        "type": "directory" if resolved.is_dir() else "file",
        "size_bytes": st.st_size,
        "size_human": _human_size(st.st_size),
        "created": datetime.fromtimestamp(st.st_ctime).isoformat(),
        "modified": datetime.fromtimestamp(st.st_mtime).isoformat(),
        "extension": resolved.suffix,
    }
    return "\n".join(f"{k}: {v}" for k, v in info.items())


# ---------------------------------------------------------------------------
# Tool: write_file
# ---------------------------------------------------------------------------

@registry.tool
def write_file(path: str, content: str, mode: str = "overwrite") -> str:
    """Write content to a file. Requires confirmation for destructive operations.

    path: Path to the file to write.
    content: The text content to write.
    mode: 'overwrite' to replace, 'append' to add to end.
    """
    resolved = _check_sandbox(path)

    if mode == "append":
        with open(resolved, "a", encoding="utf-8") as f:
            f.write(content)
        return f"Appended {len(content)} characters to {resolved.name}"
    else:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {resolved.name}"


# ---------------------------------------------------------------------------
# Tool: move_file
# ---------------------------------------------------------------------------

@registry.tool
def move_file(source: str, destination: str) -> str:
    """Move or rename a file.

    source: Current path of the file.
    destination: New path for the file.
    """
    src = _check_sandbox(source)
    dst = _check_sandbox(destination)
    if not src.exists():
        raise ToolError(ToolErrorCode.NOT_FOUND, f"Source '{source}' does not exist.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    return f"Moved {src.name} → {dst}"


# ---------------------------------------------------------------------------
# Tool: delete_file
# ---------------------------------------------------------------------------

@registry.tool
def delete_file(path: str) -> str:
    """Delete a file. This is a destructive operation.

    path: Path to the file to delete.
    """
    resolved = _check_sandbox(path)
    if not resolved.is_file():
        raise ToolError(ToolErrorCode.NOT_FOUND, f"'{path}' is not a file.")
    resolved.unlink()
    return f"Deleted {resolved.name}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024  # type: ignore
    return f"{size:.1f} TB"
