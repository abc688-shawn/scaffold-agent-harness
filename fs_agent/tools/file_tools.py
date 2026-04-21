"""fs-agent 的核心文件系统工具。

每个工具都通过 `@tool` 装饰器注册，并自动生成 OpenAI
function-call schema。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from scaffold.tools.registry import ToolRegistry
from scaffold.tools.errors import ToolError, ToolErrorCode
from scaffold.safety.sandbox import PathSandbox

# fs-agent 使用独立的工具注册表
registry = ToolRegistry()

# 沙箱会在启动时完成配置
_sandbox: PathSandbox | None = None


def set_sandbox(sandbox: PathSandbox) -> None:
    global _sandbox
    _sandbox = sandbox


def _check_sandbox(path: str) -> Path:
    if _sandbox is None:
        return Path(path).resolve()
    return _sandbox.validate(path)


# ---------------------------------------------------------------------------
# 工具：list_files
# ---------------------------------------------------------------------------

@registry.tool
def list_files(path: str = ".", recursive: bool = False, pattern: str = "*") -> str:
    """列出目录中的文件，用于发现有哪些可用文件。

    path: 要列出的目录路径。默认是当前工作区根目录。
    recursive: 为 true 时递归列出子目录中的文件。
    pattern: 用于筛选文件的 glob 模式（例如 `*.py`、`*.pdf`）。
    """
    resolved = _check_sandbox(path)
    if not resolved.is_dir():
        raise ToolError(ToolErrorCode.NOT_FOUND, f"'{path}' is not a directory.")

    if recursive:
        files = list(resolved.rglob(pattern))
    else:
        files = list(resolved.glob(pattern))

    # 限制输出规模，避免上下文膨胀
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
# 工具：read_file
# ---------------------------------------------------------------------------

@registry.tool
def read_file(path: str, offset: int = 0, length: int = 10000) -> str:
    """读取文件的文本内容，适合需要查看文件正文时使用。

    如果是跨多个文件查找特定内容，优先使用 `search_files`。

    path: 要读取的文件路径。
    offset: 起始字符偏移量（适合大文件分段读取）。
    length: 最多读取的字符数。设为 -1 表示读取全部内容（大文件需谨慎）。
    """
    resolved = _check_sandbox(path)
    if not resolved.is_file():
        raise ToolError(ToolErrorCode.NOT_FOUND, f"'{path}' is not a file.")

    size = resolved.stat().st_size
    if size > 10_000_000:  # 10 MB 上限
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
# 工具：search_files
# ---------------------------------------------------------------------------

@registry.tool
def search_files(query: str, path: str = ".", pattern: str = "*",
                 max_results: int = 20) -> str:
    """搜索包含指定文本查询的文件，适合定位特定内容。

    当你不知道信息位于哪个文件时，优先使用它而不是 `read_file`。

    query: 要搜索的文本（大小写不敏感的子串匹配）。
    path: 要搜索的目录。
    pattern: 文件类型的 glob 模式（例如 `*.py`）。
    max_results: 最多返回多少个匹配文件。
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
            # 找到第一行匹配内容
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
# 工具：file_info
# ---------------------------------------------------------------------------

@registry.tool
def file_info(path: str) -> str:
    """获取文件的详细元数据（大小、日期、类型等）。

    path: 文件路径。
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
# 工具：write_file
# ---------------------------------------------------------------------------

@registry.tool
def write_file(path: str, content: str, mode: str = "overwrite") -> str:
    """将内容写入文件。破坏性操作应先经过确认。

    path: 要写入的文件路径。
    content: 要写入的文本内容。
    mode: `overwrite` 表示覆盖，`append` 表示追加到末尾。
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
# 工具：move_file
# ---------------------------------------------------------------------------

@registry.tool
def move_file(source: str, destination: str) -> str:
    """移动或重命名文件。

    source: 文件当前路径。
    destination: 文件新的路径。
    """
    src = _check_sandbox(source)
    dst = _check_sandbox(destination)
    if not src.exists():
        raise ToolError(ToolErrorCode.NOT_FOUND, f"Source '{source}' does not exist.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    return f"Moved {src.name} → {dst}"


# ---------------------------------------------------------------------------
# 工具：delete_file
# ---------------------------------------------------------------------------

@registry.tool
def delete_file(path: str) -> str:
    """删除文件。这是破坏性操作。

    path: 要删除的文件路径。
    """
    resolved = _check_sandbox(path)
    if not resolved.is_file():
        raise ToolError(ToolErrorCode.NOT_FOUND, f"'{path}' is not a file.")
    resolved.unlink()
    return f"Deleted {resolved.name}"


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024  # type: ignore  # 这里接受浮点递减
    return f"{size:.1f} TB"
