"""Middleware protocol for the ReAct loop.

Each middleware gets four call-points per step:
    before_step  — before building the prompt and calling the LLM
    after_llm    — after the LLM responds, before tool execution
    after_tool   — after each individual tool executes (returns a possibly-modified ToolResult)
    after_step   — after all tools in a step are processed

Concrete middlewares should subclass StepMiddleware and override only the hooks
they care about; all default implementations are no-ops.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from scaffold.context.window import ContextWindow
from scaffold.models.base import ModelResponse, ToolCall, ToolResult, Usage
from scaffold.tools.registry import ToolRegistry


@dataclass
class StepContext:
    """Shared loop state threaded through every middleware hook."""

    step: int
    context: ContextWindow
    tools: ToolRegistry
    total_usage: Usage
    max_total_tokens: int = 200_000
    extra: dict[str, Any] = field(default_factory=dict)


class StepMiddleware:
    """Base class for ReAct loop middleware.

    Subclass and override only the hooks you need.
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
