"""ReAct agent 循环。

实现了“推理 + 行动”的循环：
    1. 向 LLM 发送消息
    2. 如果 LLM 返回 `tool_calls` → 执行工具 → 追加结果 → 回到第 1 步
    3. 如果 LLM 返回文本 → 结束

特性：
- 双重预算：最大步数 + 最大 token 数
- 循环检测：连续 3 次相似工具调用时注入反思提示
- 通过可序列化状态实现中断与恢复
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from scaffold.models.base import ChatModel, Message, ToolCall, Usage
from scaffold.tools.registry import ToolRegistry
from scaffold.context.window import ContextWindow
from scaffold.observability.tracer import Tracer, SpanKind

logger = logging.getLogger(__name__)


@dataclass
class LoopConfig:
    max_steps: int = 20
    max_total_tokens: int = 200_000
    loop_detect_window: int = 3
    loop_similarity_threshold: float = 0.85


@dataclass
class LoopResult:
    """一次 agent 运行的最终结果。"""
    final_message: str | None
    steps: int
    total_usage: Usage
    interrupted: bool = False
    history: list[Message] = field(default_factory=list)


class ReActLoop:
    """带预算控制和循环检测的标准 ReAct agent 循环。"""

    def __init__(
        self,
        model: ChatModel,
        tools: ToolRegistry,
        context: ContextWindow,
        config: LoopConfig | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self._model = model
        self._tools = tools
        self._context = context
        self._config = config or LoopConfig()
        self._tracer = tracer
        self._recent_calls: list[str] = []  # 用于循环检测
        self._total_usage = Usage()

    async def run(self, user_message: str) -> LoopResult:
        """为一次用户查询执行完整的 agent 循环。"""
        self._context.add(Message.user(user_message))

        run_span = None
        if self._tracer:
            run_span = self._tracer.start_span("agent_run", kind=SpanKind.AGENT)

        step = 0
        while step < self._config.max_steps:
            step += 1

            if self._total_usage.total_tokens >= self._config.max_total_tokens:
                logger.warning("Token budget exhausted at step %d", step)
                break

            # --- 调用 LLM ---
            llm_span = None
            if self._tracer:
                llm_span = self._tracer.start_span(
                    f"llm_call_step_{step}", kind=SpanKind.LLM, parent=run_span
                )

            prompt = self._context.build_prompt()
            response = await self._model.chat(
                messages=prompt,
                tools=self._tools.to_openai_tools() or None,
            )

            self._total_usage.prompt_tokens += response.usage.prompt_tokens
            self._total_usage.completion_tokens += response.usage.completion_tokens

            if self._tracer and llm_span:
                self._tracer.end_span(llm_span, metadata={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "finish_reason": response.finish_reason,
                })

            msg = response.message
            self._context.add(msg)

            # --- 没有工具调用，则结束 ---
            if not msg.tool_calls:
                logger.info("Agent finished at step %d: text response", step)
                if self._tracer and run_span:
                    self._tracer.end_span(run_span, metadata={"steps": step})
                return LoopResult(
                    final_message=msg.content,
                    steps=step,
                    total_usage=self._total_usage,
                    history=self._context.history,
                )

            # --- 循环检测 ---
            if self._detect_loop(msg.tool_calls):
                logger.warning("Loop detected at step %d, injecting reflection", step)
                reflection = Message.user(
                    "It seems you are repeating similar tool calls. "
                    "Please reconsider your approach. What have you learned so far? "
                    "Try a different strategy."
                )
                self._context.add(reflection)
                self._recent_calls.clear()
                continue

            # --- 执行工具 ---
            for tc in msg.tool_calls:
                tool_span = None
                if self._tracer:
                    tool_span = self._tracer.start_span(
                        f"tool_{tc.name}", kind=SpanKind.TOOL, parent=run_span
                    )

                result = await self._tools.execute(tc)
                self._context.add(Message.tool_result(result))

                if self._tracer and tool_span:
                    self._tracer.end_span(tool_span, metadata={
                        "tool_name": tc.name,
                        "is_error": result.is_error,
                        "result_length": len(result.content),
                    })

        # 达到最大步数
        logger.warning("Agent hit max steps (%d)", self._config.max_steps)
        if self._tracer and run_span:
            self._tracer.end_span(run_span, metadata={"steps": step, "max_steps_hit": True})

        return LoopResult(
            final_message="I reached the maximum number of steps without completing the task.",
            steps=step,
            total_usage=self._total_usage,
            history=self._context.history,
        )

    def _detect_loop(self, tool_calls: list[ToolCall]) -> bool:
        """检查最近 N 次调用是否异常相似。"""
        sig = json.dumps(
            [(tc.name, sorted(tc.arguments.items())) for tc in tool_calls],
            sort_keys=True,
        )
        self._recent_calls.append(sig)
        if len(self._recent_calls) < self._config.loop_detect_window:
            return False

        window = self._recent_calls[-self._config.loop_detect_window:]
        # 当前使用简单的精确匹配检测，后续可以升级为模糊匹配
        return len(set(window)) == 1
