"""Tool-call limit middleware — replaces the old _detect_loop mechanism.

Improvements over the old exact-signature approach:
- Tracks each (tool_name, args_hash) pair independently across steps.
- Fires a warning the moment a *specific* call is repeated, not only when the
  entire call *sequence* repeats.
- repeat_limit / run_limit are independently configurable.
- Warning is appended to the tool result itself so the model sees it in context.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Literal

from scaffold.loop.middleware import StepContext, StepMiddleware
from scaffold.models.base import ModelResponse, ToolCall, ToolResult

logger = logging.getLogger(__name__)

ExitBehavior = Literal["warn", "block"]


class ToolCallLimitMiddleware(StepMiddleware):
    """Detect and discourage repeated identical tool calls.

    Args:
        repeat_limit: Max times the same (name, args_hash) pair may be called.
                      On the Nth call a warning is appended to the tool result.
        run_limit:    Optional cap on total calls per tool name in a run.
        exit_behavior: "warn" appends a note to the result; "block" replaces
                       the result with an error so the model stops calling.
    """

    def __init__(
        self,
        repeat_limit: int = 3,
        run_limit: int | None = None,
        exit_behavior: ExitBehavior = "warn",
    ) -> None:
        self._repeat_limit = repeat_limit
        self._run_limit = run_limit
        self._exit_behavior: ExitBehavior = exit_behavior
        # (tool_name, args_hash) -> call count
        self._repeat_counts: dict[str, int] = {}
        # tool_name -> total call count this run
        self._run_counts: dict[str, int] = {}

    @staticmethod
    def _args_hash(arguments: dict) -> str:
        payload = json.dumps(arguments, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:8]

    async def after_tool(
        self, ctx: StepContext, call: ToolCall, result: ToolResult
    ) -> ToolResult:
        # Track per-run total
        self._run_counts[call.name] = self._run_counts.get(call.name, 0) + 1

        # Track per-(name, args) repetitions
        repeat_key = f"{call.name}:{self._args_hash(call.arguments)}"
        self._repeat_counts[repeat_key] = self._repeat_counts.get(repeat_key, 0) + 1
        repeat_count = self._repeat_counts[repeat_key]

        warnings: list[str] = []

        if repeat_count >= self._repeat_limit:
            warnings.append(
                f"⚠ '{call.name}' was called {repeat_count}× with identical arguments. "
                "Do not repeat this call — try a different approach."
            )
            logger.warning(
                "Repeat limit hit: '%s' called %d times with same args",
                call.name,
                repeat_count,
            )

        run_count = self._run_counts[call.name]
        if self._run_limit is not None and run_count >= self._run_limit:
            warnings.append(
                f"⚠ '{call.name}' has been called {run_count}× total this run. "
                "Consider whether further calls are necessary."
            )
            logger.warning("Run limit hit: '%s' called %d times total", call.name, run_count)

        if not warnings:
            return result

        warning_text = "\n\n" + "\n".join(warnings)

        if self._exit_behavior == "block":
            return ToolResult(
                tool_call_id=result.tool_call_id,
                name=result.name,
                content=warning_text.strip(),
                is_error=True,
            )

        # "warn": append to existing content
        return ToolResult(
            tool_call_id=result.tool_call_id,
            name=result.name,
            content=(result.content or "") + warning_text,
            is_error=result.is_error,
        )

    async def after_llm(self, ctx: StepContext, response: ModelResponse) -> None:
        pass
