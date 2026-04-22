"""工具调用限制 Middleware —— 替代旧的 _detect_loop 机制。

相比旧的精确签名方案的改进：
- 跨步骤独立追踪每个 (tool_name, args_hash) 对。
- 一旦*特定*调用被重复便立即触发警告，而不是等整个调用*序列*重复才报。
- repeat_limit / run_limit 可独立配置。
- 警告直接追加到工具结果中，使模型能在上下文中看到提示。
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
    """检测并抑制重复的相同工具调用。

    Args:
        repeat_limit: 相同 (name, args_hash) 对的最大允许调用次数。
                      第 N 次调用时，会在工具结果中追加警告信息。
        run_limit:    单次运行中每个工具名的总调用次数上限（可选）。
        exit_behavior: "warn" 将提示追加到结果末尾；"block" 将结果替换为错误，
                       以阻止模型继续调用。
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
        # (tool_name, args_hash) -> 调用次数
        self._repeat_counts: dict[str, int] = {}
        # tool_name -> 本次运行的总调用次数
        self._run_counts: dict[str, int] = {}

    @staticmethod
    def _args_hash(arguments: dict) -> str:
        payload = json.dumps(arguments, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:8]

    async def after_tool(
        self, ctx: StepContext, call: ToolCall, result: ToolResult
    ) -> ToolResult:
        # 追踪本次运行的总调用次数
        self._run_counts[call.name] = self._run_counts.get(call.name, 0) + 1

        # 追踪每个 (name, args) 对的重复次数
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

        # "warn"：追加到现有内容末尾
        return ToolResult(
            tool_call_id=result.tool_call_id,
            name=result.name,
            content=(result.content or "") + warning_text,
            is_error=result.is_error,
        )

    async def after_llm(self, ctx: StepContext, response: ModelResponse) -> None:
        pass
