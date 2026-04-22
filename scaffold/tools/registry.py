"""工具注册表与 `@tool` 装饰器。

用法：
    registry = ToolRegistry()

    @registry.tool
    def list_files(path: str) -> str:
        \"\"\"列出目录中的文件。\"\"\"
        ...

    # 或者使用模块级便捷接口：
    from scaffold.tools import tool

    @tool(name="read_file", description="读取文件内容")
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
from scaffold.safety.injection import sanitize_tool_result
from scaffold.tools.errors import ToolError, ToolErrorCode
from scaffold.tools.schema import ToolSchema, schema_from_function

logger = logging.getLogger(__name__)


class PermissionGuard(Protocol):
    """工具执行前权限检查的协议。

    返回 True 表示允许，False 表示阻止，`confirm` 表示请求用户确认。
    """
    def check(self, tool_name: str, arguments: dict[str, Any]) -> bool | str: ...
    def confirm(self, tool_name: str, arguments: dict[str, Any]) -> bool: ...


class ToolRegistry:
    """所有已注册工具的中心存储。"""

    def __init__(self) -> None:
        self._tools: dict[str, _RegisteredTool] = {}
        self._guard: PermissionGuard | None = None
        self._pre_hooks: list[Callable[[str, dict[str, Any]], None]] = []
        self._post_hooks: list[Callable[[str, dict[str, Any], str], None]] = []

    def set_permission_guard(self, guard: PermissionGuard) -> None:
        """设置权限守卫，在每次工具执行前进行检查。"""
        self._guard = guard

    def add_pre_hook(self, hook: Callable[[str, dict[str, Any]], None]) -> None:
        """添加执行前钩子：`(tool_name, arguments) -> None`。"""
        self._pre_hooks.append(hook)

    def add_post_hook(self, hook: Callable[[str, dict[str, Any], str], None]) -> None:
        """添加执行后钩子：`(tool_name, arguments, result) -> None`。"""
        self._post_hooks.append(hook)

    # ---- 注册 ----

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
        """装饰器形式：`@registry.tool` 或 `@registry.tool(name=...)`。"""
        if fn is not None:
            return self.register(fn, name=name, description=description)

        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            return self.register(f, name=name, description=description)
        return decorator

    # ---- 查找 ----

    def get(self, name: str) -> _RegisteredTool | None:
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """返回 OpenAI 格式的工具 schema 列表。"""
        return [t.schema.to_openai() for t in self._tools.values()]

    # ---- 执行 ----

    async def execute(self, call: ToolCall) -> ToolResult:
        """执行一次工具调用，并返回 ToolResult。"""
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

        # 权限守卫
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

        # 执行前钩子
        for hook in self._pre_hooks:
            hook(call.name, call.arguments)

        result = await registered.execute(call)

        # 执行后钩子
        for hook in self._post_hooks:
            hook(call.name, call.arguments, result.content)

        return result

    async def execute_many(self, calls: Sequence[ToolCall]) -> list[ToolResult]:
        """并发执行多个工具调用。"""
        return list(await asyncio.gather(*(self.execute(c) for c in calls)))


class _RegisteredTool:
    """对已注册工具函数的内部包装。"""

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
            raw = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
            logger.info("Tool %s completed in %.2fs", call.name, elapsed)
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=sanitize_tool_result(raw),
            )
        except ToolError as e:
            logger.warning("Tool %s raised ToolError: %s", call.name, e.message)
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=sanitize_tool_result(e.for_model()),
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
# 模块级便捷接口 —— 默认全局注册表
# ---------------------------------------------------------------------------
_default_registry = ToolRegistry()


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """使用默认全局注册表的模块级 `@tool` 装饰器。"""
    return _default_registry.tool(fn, name=name, description=description)
