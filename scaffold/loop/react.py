"""ReAct agent 循环。

实现了"推理 + 行动"的循环：
    1. 向 LLM 发送消息
    2. 如果 LLM 返回 `tool_calls` → 并发执行所有工具 → 追加结果 → 回到第 1 步
    3. 如果 LLM 返回文本 → 结束

特性：
- 双重预算：最大步数 + 最大 token 数
- Middleware 管道：before_step / after_llm / after_tool / after_step 四个切点
- DynamicPrompt 阶段切换（EXECUTION → REFLECTION 在循环检测时）
- 工具并发执行（asyncio.gather）
- Jinja2 模板渲染 reflection 消息
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from scaffold.context.window import AgentPhase, ContextWindow
from scaffold.loop.middleware import StepContext, StepMiddleware
from scaffold.loop.middlewares.tool_call_limit_middleware import ToolCallLimitMiddleware
from scaffold.models.base import ChatModel, Message, Usage
from scaffold.observability.tracer import SpanKind, Tracer
from scaffold.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class LoopConfig:
    max_steps: int = 20
    max_total_tokens: int = 200_000


@dataclass
class LoopResult:
    """一次 agent 运行的最终结果。"""

    final_message: str | None
    steps: int
    total_usage: Usage
    interrupted: bool = False
    history: list[Message] = field(default_factory=list)


class ReActLoop:
    """带预算控制和 middleware 管道的 ReAct agent 循环。

    Args:
        middlewares: 若为 None，使用默认栈（ToolCallLimitMiddleware）。
                     传入空列表 [] 可显式禁用所有 middleware。
    """

    def __init__(
        self,
        model: ChatModel,
        tools: ToolRegistry,
        context: ContextWindow,
        config: LoopConfig | None = None,
        tracer: Tracer | None = None,
        middlewares: list[StepMiddleware] | None = None,
    ) -> None:
        self._model = model
        self._tools = tools
        self._context = context
        self._config = config or LoopConfig()
        self._tracer = tracer
        self._middlewares: list[StepMiddleware] = (
            [ToolCallLimitMiddleware()] if middlewares is None else middlewares
        )
        self._total_usage = Usage()

    # ------------------------------------------------------------------
    # Middleware helpers
    # ------------------------------------------------------------------

    def _make_ctx(self, step: int) -> StepContext:
        return StepContext(
            step=step,
            context=self._context,
            tools=self._tools,
            total_usage=self._total_usage,
            max_total_tokens=self._config.max_total_tokens,
        )

    async def _before_step(self, ctx: StepContext) -> None:
        for mw in self._middlewares:
            await mw.before_step(ctx)

    async def _after_llm(self, ctx: StepContext, response) -> None:
        for mw in self._middlewares:
            await mw.after_llm(ctx, response)

    async def _after_tool(self, ctx: StepContext, call, result):
        for mw in self._middlewares:
            result = await mw.after_tool(ctx, call, result)
        return result

    async def _after_step(self, ctx: StepContext) -> None:
        for mw in self._middlewares:
            await mw.after_step(ctx)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self, user_message: str) -> LoopResult:
        """为一次用户查询执行完整的 agent 循环。"""
        self._context.add(Message.user(user_message))
        self._context.set_phase(AgentPhase.EXECUTION)

        run_span = None
        if self._tracer:
            run_span = self._tracer.start_span("agent_run", kind=SpanKind.AGENT)

        step = 0
        while step < self._config.max_steps:
            step += 1
            ctx = self._make_ctx(step)

            if self._total_usage.total_tokens >= self._config.max_total_tokens:
                logger.warning("Token budget exhausted at step %d", step)
                break

            await self._before_step(ctx)

            # --- 调用 LLM ---
            llm_span = None
            if self._tracer:
                llm_span = self._tracer.start_span(
                    f"llm_call_step_{step}", kind=SpanKind.LLM, parent=run_span
                )

            prompt = await self._context.build_prompt(self._model)
            tools_schema = self._tools.to_openai_tools() or None
            response = await self._model.chat(messages=prompt, tools=tools_schema)

            self._total_usage.prompt_tokens += response.usage.prompt_tokens
            self._total_usage.completion_tokens += response.usage.completion_tokens

            if self._tracer and llm_span:
                self._tracer.end_span(llm_span, metadata={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "finish_reason": response.finish_reason,
                })

            await self._after_llm(ctx, response)

            msg = response.message
            self._context.add(msg)

            # --- 没有工具调用，则结束 ---
            if not msg.tool_calls:
                logger.info("Agent finished at step %d: text response", step)
                await self._after_step(ctx)
                if self._tracer and run_span:
                    self._tracer.end_span(run_span, metadata={"steps": step})
                return LoopResult(
                    final_message=msg.content,
                    steps=step,
                    total_usage=self._total_usage,
                    history=self._context.history,
                )

            # --- 并发执行所有工具 ---
            tool_calls = msg.tool_calls

            # 启动 trace span（在并发执行前）
            tool_spans = []
            for tc in tool_calls:
                span = (
                    self._tracer.start_span(
                        f"tool_{tc.name}", kind=SpanKind.TOOL, parent=run_span
                    )
                    if self._tracer
                    else None
                )
                tool_spans.append(span)

            raw_results = await asyncio.gather(
                *(self._tools.execute(tc) for tc in tool_calls)
            )

            # 逐个过 middleware 链并写入 context
            for tc, raw_result, tool_span in zip(tool_calls, raw_results, tool_spans):
                result = await self._after_tool(ctx, tc, raw_result)
                self._context.add(Message.tool_result(result))

                if self._tracer and tool_span:
                    self._tracer.end_span(tool_span, metadata={
                        "tool_name": tc.name,
                        "is_error": result.is_error,
                        "result_length": len(result.content),
                    })

            await self._after_step(ctx)

        # 达到最大步数，切换到 reflection 阶段
        logger.warning("Agent hit max steps (%d)", self._config.max_steps)
        self._context.set_phase(AgentPhase.REFLECTION)
        if self._tracer and run_span:
            self._tracer.end_span(run_span, metadata={
                "steps": step,
                "max_steps_hit": True,
            })

        return LoopResult(
            final_message="I reached the maximum number of steps without completing the task.",
            steps=step,
            total_usage=self._total_usage,
            history=self._context.history,
        )
