"""fs-agent 的权限策略。

共三个等级：
- `read_only`：只允许列出 / 读取 / 搜索 / 信息类工具
- `confirm_write`：写入 / 移动 / 删除前需要用户确认
- `autonomous`：所有操作均可直接执行，无需确认
"""
from __future__ import annotations

import json
from enum import Enum
from typing import Any


class PermissionLevel(str, Enum):
    READ_ONLY = "read_only"
    CONFIRM_WRITE = "confirm_write"
    AUTONOMOUS = "autonomous"


# 会修改文件系统的工具
WRITE_TOOLS = {"write_file", "move_file", "delete_file", "organize_files", "tag_files"}
READ_TOOLS = {"list_files", "read_file", "search_files", "file_info", "read_pdf",
              "read_docx", "preview_file", "summarize_file", "compare_files",
              "search_by_tag", "semantic_search", "index_files"}


def is_allowed(tool_name: str, level: PermissionLevel) -> bool:
    if level == PermissionLevel.AUTONOMOUS:
        return True
    if level == PermissionLevel.READ_ONLY:
        return tool_name in READ_TOOLS
    # confirm_write：允许调用，但调用方应先请求用户确认
    return True


def needs_confirmation(tool_name: str, level: PermissionLevel) -> bool:
    if level == PermissionLevel.CONFIRM_WRITE and tool_name in WRITE_TOOLS:
        return True
    return False


class FSPermissionGuard:
    """面向文件系统 agent 的具体权限守卫。

    实现了 `scaffold.tools.registry` 中定义的 PermissionGuard 协议。
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
        """交互式确认提示。"""
        args_str = json.dumps(arguments, ensure_ascii=False)
        print(f"\n⚠️  Agent wants to call: {tool_name}({args_str})")
        try:
            answer = input("  Allow? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in ("y", "yes")
