"""ReAct 循环的 Middleware 协议。

每步有四个切点：
    before_step  — 构建提示词并调用 LLM 之前
    after_llm    — LLM 响应之后、工具执行之前
    after_tool   — 每个工具执行后（返回可能被修改的 ToolResult）
    after_step   — 该步骤中所有工具都处理完之后

具体的 Middleware 应继承 StepMiddleware，只需覆盖关心的钩子；
所有默认实现均为空操作（no-op）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from scaffold.context.window import ContextWindow
from scaffold.models.base import ModelResponse, ToolCall, ToolResult, Usage
from scaffold.tools.registry import ToolRegistry


@dataclass
class StepContext:
    """每个 Middleware 钩子都会传入的共享循环状态。"""

    step: int
    context: ContextWindow
    tools: ToolRegistry
    total_usage: Usage
    max_total_tokens: int = 200_000
    extra: dict[str, Any] = field(default_factory=dict)


class StepMiddleware:
    """ReAct 循环 Middleware 的基类。

    子类只需覆盖所需的钩子即可。
    """

    async def before_step(self, ctx: StepContext) -> None:
        pass

    async def after_llm(self, ctx: StepContext, response: ModelResponse) -> None:
        pass

    async def after_tool(
        self, ctx: StepContext, call: ToolCall, result: ToolResult
    ) -> ToolResult:
        return result

    async def after_step(self, ctx: StepContext) -> None:
        pass
