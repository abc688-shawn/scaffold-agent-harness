"""Tests for permission guard and tool registry permission integration."""
from __future__ import annotations

import asyncio
import pytest
from typing import Any

from scaffold.tools.registry import ToolRegistry, PermissionGuard
from scaffold.models.base import ToolCall
from fs_agent.policies.permissions import (
    FSPermissionGuard,
    PermissionLevel,
    is_allowed,
    needs_confirmation,
    WRITE_TOOLS,
    READ_TOOLS,
)


class TestPermissionLevels:
    def test_read_only_allows_reads(self):
        for tool in READ_TOOLS:
            assert is_allowed(tool, PermissionLevel.READ_ONLY)

    def test_read_only_blocks_writes(self):
        for tool in WRITE_TOOLS:
            assert not is_allowed(tool, PermissionLevel.READ_ONLY)

    def test_autonomous_allows_all(self):
        for tool in READ_TOOLS | WRITE_TOOLS:
            assert is_allowed(tool, PermissionLevel.AUTONOMOUS)

    def test_confirm_write_allows_all(self):
        for tool in READ_TOOLS | WRITE_TOOLS:
            assert is_allowed(tool, PermissionLevel.CONFIRM_WRITE)

    def test_confirm_write_needs_confirmation(self):
        for tool in WRITE_TOOLS:
            assert needs_confirmation(tool, PermissionLevel.CONFIRM_WRITE)

    def test_confirm_read_no_confirmation(self):
        for tool in READ_TOOLS:
            assert not needs_confirmation(tool, PermissionLevel.CONFIRM_WRITE)


class TestFSPermissionGuard:
    def test_read_only_check_blocks_write(self):
        guard = FSPermissionGuard(PermissionLevel.READ_ONLY)
        assert guard.check("write_file", {"path": "x"}) is False
        assert guard.check("delete_file", {"path": "x"}) is False

    def test_read_only_check_allows_read(self):
        guard = FSPermissionGuard(PermissionLevel.READ_ONLY)
        assert guard.check("read_file", {"path": "x"}) is True
        assert guard.check("list_files", {"path": "."}) is True

    def test_confirm_write_returns_confirm(self):
        guard = FSPermissionGuard(PermissionLevel.CONFIRM_WRITE)
        result = guard.check("write_file", {"path": "x"})
        assert result == "confirm"

    def test_autonomous_allows_write(self):
        guard = FSPermissionGuard(PermissionLevel.AUTONOMOUS)
        assert guard.check("write_file", {"path": "x"}) is True
        assert guard.check("delete_file", {"path": "x"}) is True


class TestRegistryWithPermissions:
    def test_guard_blocks_tool(self):
        reg = ToolRegistry()

        @reg.tool
        def write_file(path: str, content: str) -> str:
            """Write content to a file."""
            return "written"

        guard = FSPermissionGuard(PermissionLevel.READ_ONLY)
        reg.set_permission_guard(guard)

        call = ToolCall(id="1", name="write_file", arguments={"path": "x", "content": "y"})
        result = asyncio.get_event_loop().run_until_complete(reg.execute(call))
        assert result.is_error
        assert "permission" in result.content.lower() or "blocked" in result.content.lower()

    def test_guard_allows_tool(self):
        reg = ToolRegistry()

        @reg.tool
        def read_file(path: str) -> str:
            """Read a file."""
            return "content"

        guard = FSPermissionGuard(PermissionLevel.READ_ONLY)
        reg.set_permission_guard(guard)

        call = ToolCall(id="1", name="read_file", arguments={"path": "x"})
        result = asyncio.get_event_loop().run_until_complete(reg.execute(call))
        assert not result.is_error
        assert result.content == "content"

    def test_hooks_fire(self):
        reg = ToolRegistry()
        hook_log: list[str] = []

        @reg.tool
        def echo(text: str) -> str:
            """Echo text."""
            return text

        reg.add_pre_hook(lambda name, args: hook_log.append(f"pre:{name}"))
        reg.add_post_hook(lambda name, args, res: hook_log.append(f"post:{name}"))

        call = ToolCall(id="1", name="echo", arguments={"text": "hi"})
        asyncio.get_event_loop().run_until_complete(reg.execute(call))

        assert "pre:echo" in hook_log
        assert "post:echo" in hook_log
