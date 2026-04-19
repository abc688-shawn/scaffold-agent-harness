"""Advanced agent tools — cross-document Q&A, file organization, tagging.

These tools orchestrate multiple lower-level tools to provide higher-level
capabilities. They work alongside the LLM's own reasoning.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from scaffold.tools.errors import ToolError, ToolErrorCode

from fs_agent.tools.file_tools import registry, _check_sandbox


# ---------------------------------------------------------------------------
# Tool: organize_files
# ---------------------------------------------------------------------------

@registry.tool
def organize_files(
    source: str,
    strategy: str = "extension",
    dry_run: bool = True,
) -> str:
    """Plan or execute file organization in a directory.

    Creates a plan to organize files by extension, date, or size.
    Always returns the plan first. Set dry_run=False to execute.

    source: Directory containing files to organize.
    strategy: Organization strategy — 'extension' (by file type), 'date' (by modification month), 'size' (small/medium/large).
    dry_run: If true, only show the plan without moving files. Set to false to execute.
    """
    resolved = _check_sandbox(source)
    if not resolved.is_dir():
        raise ToolError(ToolErrorCode.NOT_FOUND, f"'{source}' is not a directory.")

    files = [f for f in resolved.iterdir() if f.is_file()]
    if not files:
        return f"No files found in {source}"

    plan: list[dict[str, str]] = []
    if strategy == "extension":
        plan = _plan_by_extension(files, resolved)
    elif strategy == "date":
        plan = _plan_by_date(files, resolved)
    elif strategy == "size":
        plan = _plan_by_size(files, resolved)
    else:
        raise ToolError(ToolErrorCode.INVALID_ARGUMENTS, f"Unknown strategy: {strategy}")

    if not plan:
        return "All files are already organized."

    # Format plan
    lines = [f"Organization plan ({strategy} strategy): {len(plan)} moves"]
    for item in plan:
        lines.append(f"  {item['file']} → {item['destination']}")

    if dry_run:
        lines.append("\n[DRY RUN] No files were moved. Set dry_run=False to execute.")
    else:
        # Execute moves
        moved = 0
        for item in plan:
            src = resolved / item["file"]
            dst = resolved / item["destination"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists() and not dst.exists():
                src.rename(dst)
                moved += 1
        lines.append(f"\nExecuted: {moved}/{len(plan)} files moved.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: tag_files
# ---------------------------------------------------------------------------

@registry.tool
def tag_files(path: str, tags: str, tags_file: str = ".file_tags.json") -> str:
    """Add tags/labels to a file for categorization.

    Tags are stored in a JSON file in the workspace root.

    path: Path to the file to tag.
    tags: Comma-separated list of tags (e.g. 'important,review,rlhf').
    tags_file: Path to the tags database file.
    """
    resolved = _check_sandbox(path)
    if not resolved.exists():
        raise ToolError(ToolErrorCode.NOT_FOUND, f"'{path}' does not exist.")

    tags_path = _check_sandbox(tags_file)
    tag_db: dict[str, list[str]] = {}
    if tags_path.exists():
        tag_db = json.loads(tags_path.read_text(encoding="utf-8"))

    file_key = str(resolved)
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    existing = set(tag_db.get(file_key, []))
    existing.update(tag_list)
    tag_db[file_key] = sorted(existing)

    tags_path.write_text(json.dumps(tag_db, indent=2, ensure_ascii=False), encoding="utf-8")

    return f"Tagged '{resolved.name}' with: {', '.join(tag_list)}. Total tags: {sorted(existing)}"


# ---------------------------------------------------------------------------
# Tool: search_by_tag
# ---------------------------------------------------------------------------

@registry.tool
def search_by_tag(tag: str, tags_file: str = ".file_tags.json") -> str:
    """Find all files with a specific tag.

    tag: The tag to search for.
    tags_file: Path to the tags database file.
    """
    tags_path = _check_sandbox(tags_file)
    if not tags_path.exists():
        return "No tags database found. Use tag_files to create tags first."

    tag_db: dict[str, list[str]] = json.loads(tags_path.read_text(encoding="utf-8"))

    matches = []
    for file_path, file_tags in tag_db.items():
        if tag.lower() in [t.lower() for t in file_tags]:
            matches.append(f"  {file_path} (tags: {', '.join(file_tags)})")

    if not matches:
        return f"No files found with tag '{tag}'"

    return f"Files with tag '{tag}':\n" + "\n".join(matches)


# ---------------------------------------------------------------------------
# Tool: compare_files
# ---------------------------------------------------------------------------

@registry.tool
def compare_files(path1: str, path2: str, context_lines: int = 3) -> str:
    """Compare two text files and show their differences.

    Useful for understanding changes between versions or finding differences
    between similar documents.

    path1: Path to the first file.
    path2: Path to the second file.
    context_lines: Number of context lines around each difference.
    """
    r1 = _check_sandbox(path1)
    r2 = _check_sandbox(path2)

    for r, p in [(r1, path1), (r2, path2)]:
        if not r.is_file():
            raise ToolError(ToolErrorCode.NOT_FOUND, f"'{p}' is not a file.")

    try:
        text1 = r1.read_text(encoding="utf-8", errors="replace")
        text2 = r2.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        raise ToolError(ToolErrorCode.INTERNAL, f"Error reading files: {e}")

    import difflib
    lines1 = text1.splitlines(keepends=True)
    lines2 = text2.splitlines(keepends=True)

    diff = list(difflib.unified_diff(
        lines1, lines2,
        fromfile=str(r1.name), tofile=str(r2.name),
        n=context_lines,
    ))

    if not diff:
        return f"Files are identical: {r1.name} and {r2.name}"

    # Truncate very large diffs
    diff_text = "".join(diff[:200])
    header = f"Differences between {r1.name} and {r2.name}:\n"
    if len(diff) > 200:
        return header + diff_text + f"\n... ({len(diff) - 200} more diff lines)"
    return header + diff_text


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _plan_by_extension(files: list[Path], root: Path) -> list[dict[str, str]]:
    plan = []
    for f in files:
        ext = f.suffix.lower().lstrip(".") or "no_extension"
        dest = f"{ext}/{f.name}"
        if not (root / ext / f.name).exists():
            plan.append({"file": f.name, "destination": dest})
    return plan


def _plan_by_date(files: list[Path], root: Path) -> list[dict[str, str]]:
    plan = []
    for f in files:
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        folder = mtime.strftime("%Y-%m")
        dest = f"{folder}/{f.name}"
        if not (root / folder / f.name).exists():
            plan.append({"file": f.name, "destination": dest})
    return plan


def _plan_by_size(files: list[Path], root: Path) -> list[dict[str, str]]:
    plan = []
    for f in files:
        size = f.stat().st_size
        if size < 100_000:
            folder = "small"
        elif size < 10_000_000:
            folder = "medium"
        else:
            folder = "large"
        dest = f"{folder}/{f.name}"
        if not (root / folder / f.name).exists():
            plan.append({"file": f.name, "destination": dest})
    return plan
