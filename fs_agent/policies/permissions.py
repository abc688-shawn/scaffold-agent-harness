"""Permission policies for fs-agent.

Three levels:
- read_only: only list/read/search/info tools
- confirm_write: write/move/delete require user confirmation
- autonomous: all operations allowed without confirmation
"""
from __future__ import annotations

import json
from enum import Enum
from typing import Any


class PermissionLevel(str, Enum):
    READ_ONLY = "read_only"
    CONFIRM_WRITE = "confirm_write"
    AUTONOMOUS = "autonomous"


# Tools that modify the file system
WRITE_TOOLS = {"write_file", "move_file", "delete_file", "organize_files", "tag_files"}
READ_TOOLS = {"list_files", "read_file", "search_files", "file_info", "read_pdf",
              "read_docx", "preview_file", "summarize_file", "compare_files",
              "search_by_tag", "semantic_search", "index_files"}


def is_allowed(tool_name: str, level: PermissionLevel) -> bool:
    if level == PermissionLevel.AUTONOMOUS:
        return True
    if level == PermissionLevel.READ_ONLY:
        return tool_name in READ_TOOLS
    # confirm_write: allowed, but caller should prompt for confirmation
    return True


def needs_confirmation(tool_name: str, level: PermissionLevel) -> bool:
    if level == PermissionLevel.CONFIRM_WRITE and tool_name in WRITE_TOOLS:
        return True
    return False


class FSPermissionGuard:
    """Concrete permission guard for the file-system agent.

    Implements the PermissionGuard protocol from scaffold.tools.registry.
    """

    def __init__(self, level: PermissionLevel) -> None:
        self._level = level

    @property
    def level(self) -> PermissionLevel:
        return self._level

    def check(self, tool_name: str, arguments: dict[str, Any]) -> bool | str:
        if not is_allowed(tool_name, self._level):
            return False
        if needs_confirmation(tool_name, self._level):
            return "confirm"
        return True

    def confirm(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        """Interactive confirmation prompt."""
        args_str = json.dumps(arguments, ensure_ascii=False)
        print(f"\n⚠️  Agent wants to call: {tool_name}({args_str})")
        try:
            answer = input("  Allow? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in ("y", "yes")
