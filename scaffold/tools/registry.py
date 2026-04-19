"""Tool registry and @tool decorator.

Usage:
    registry = ToolRegistry()

    @registry.tool
    def list_files(path: str) -> str:
        \"\"\"List files in a directory.\"\"\"
        ...

    # Or use the module-level convenience:
    from scaffold.tools import tool

    @tool(name="read_file", description="Read file contents")
    def read_file(path: str, offset: int = 0, length: int = -1) -> str:
        ...
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Protocol, Sequence

from scaffold.models.base import ToolCall, ToolResult
from scaffold.tools.errors import ToolError, ToolErrorCode
from scaffold.tools.schema import ToolSchema, schema_from_function

logger = logging.getLogger(__name__)


class PermissionGuard(Protocol):
    """Protocol for permission checking before tool execution.

    Return True to allow, False to block, 'confirm' to request user confirmation.
    """
    def check(self, tool_name: str, arguments: dict[str, Any]) -> bool | str: ...
    def confirm(self, tool_name: str, arguments: dict[str, Any]) -> bool: ...


class ToolRegistry:
    """Central store for all registered tools."""

    def __init__(self) -> None:
        self._tools: dict[str, _RegisteredTool] = {}
        self._guard: PermissionGuard | None = None
        self._pre_hooks: list[Callable[[str, dict[str, Any]], None]] = []
        self._post_hooks: list[Callable[[str, dict[str, Any], str], None]] = []

    def set_permission_guard(self, guard: PermissionGuard) -> None:
        """Set a permission guard that checks before every tool execution."""
        self._guard = guard

    def add_pre_hook(self, hook: Callable[[str, dict[str, Any]], None]) -> None:
        """Add a pre-execution hook: (tool_name, arguments) -> None."""
        self._pre_hooks.append(hook)

    def add_post_hook(self, hook: Callable[[str, dict[str, Any], str], None]) -> None:
        """Add a post-execution hook: (tool_name, arguments, result) -> None."""
        self._post_hooks.append(hook)

    # ---- registration ----

    def register(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[..., Any]:
        schema = schema_from_function(fn, name=name, description=description)
        self._tools[schema.name] = _RegisteredTool(fn=fn, schema=schema)
        logger.debug("Registered tool: %s", schema.name)
        return fn

    def tool(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Any:
        """Decorator form: ``@registry.tool`` or ``@registry.tool(name=...)``."""
        if fn is not None:
            return self.register(fn, name=name, description=description)

        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            return self.register(f, name=name, description=description)
        return decorator

    # ---- lookup ----

    def get(self, name: str) -> _RegisteredTool | None:
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Return the list of tool schemas in OpenAI format."""
        return [t.schema.to_openai() for t in self._tools.values()]

    # ---- execution ----

    async def execute(self, call: ToolCall) -> ToolResult:
        """Execute a tool call, returning a ToolResult."""
        registered = self._tools.get(call.name)
        if registered is None:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=ToolError(
                    ToolErrorCode.NOT_FOUND,
                    f"Tool '{call.name}' not found. Available: {self.list_names()}",
                    hint="Check the tool name and try again.",
                ).for_model(),
                is_error=True,
            )

        # Permission guard
        if self._guard is not None:
            decision = self._guard.check(call.name, call.arguments)
            if decision is False:
                return ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    content=f"[BLOCKED] Tool '{call.name}' is not allowed under current permission level.",
                    is_error=True,
                )
            if decision == "confirm":
                approved = self._guard.confirm(call.name, call.arguments)
                if not approved:
                    return ToolResult(
                        tool_call_id=call.id,
                        name=call.name,
                        content=f"[CANCELLED] User declined to confirm '{call.name}' operation.",
                        is_error=True,
                    )

        # Pre-hooks
        for hook in self._pre_hooks:
            hook(call.name, call.arguments)

        result = await registered.execute(call)

        # Post-hooks
        for hook in self._post_hooks:
            hook(call.name, call.arguments, result.content)

        return result

    async def execute_many(self, calls: Sequence[ToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls (concurrently)."""
        return list(await asyncio.gather(*(self.execute(c) for c in calls)))


class _RegisteredTool:
    """Internal wrapper around a registered tool function."""

    def __init__(self, fn: Callable[..., Any], schema: ToolSchema) -> None:
        self.fn = fn
        self.schema = schema
        self._is_async = asyncio.iscoroutinefunction(fn)

    async def execute(self, call: ToolCall) -> ToolResult:
        start = time.monotonic()
        try:
            if self._is_async:
                result = await self.fn(**call.arguments)
            else:
                result = await asyncio.to_thread(self.fn, **call.arguments)

            elapsed = time.monotonic() - start
            content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
            logger.info("Tool %s completed in %.2fs", call.name, elapsed)
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=content,
            )
        except ToolError as e:
            logger.warning("Tool %s raised ToolError: %s", call.name, e.message)
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=e.for_model(),
                is_error=True,
            )
        except Exception as e:
            logger.exception("Tool %s raised unexpected error", call.name)
            err = ToolError(ToolErrorCode.INTERNAL_ERROR, str(e))
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=err.for_model(),
                is_error=True,
            )


# ---------------------------------------------------------------------------
# Module-level convenience — a default global registry
# ---------------------------------------------------------------------------
_default_registry = ToolRegistry()


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Module-level ``@tool`` decorator that uses the default global registry."""
    return _default_registry.tool(fn, name=name, description=description)
