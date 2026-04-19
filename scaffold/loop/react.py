"""ReAct agent loop.

Implements the Reasoning + Acting loop:
    1. Send messages to LLM
    2. If LLM returns tool_calls → execute tools → append results → goto 1
    3. If LLM returns text → done

Features:
- Dual budget: max steps + max tokens
- Loop detection: 3 consecutive similar tool calls → inject reflection
- Interrupt & resume via serializable state
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
    """Final result of an agent run."""
    final_message: str | None
    steps: int
    total_usage: Usage
    interrupted: bool = False
    history: list[Message] = field(default_factory=list)


class ReActLoop:
    """Standard ReAct agent loop with budgets and loop detection."""

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
        self._recent_calls: list[str] = []  # for loop detection
        self._total_usage = Usage()

    async def run(self, user_message: str) -> LoopResult:
        """Execute the full agent loop for a user query."""
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

            # --- LLM call ---
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

            # --- No tool calls → done ---
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

            # --- Loop detection ---
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

            # --- Execute tools ---
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

        # Max steps reached
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
        """Check if last N calls are suspiciously similar."""
        sig = json.dumps(
            [(tc.name, sorted(tc.arguments.items())) for tc in tool_calls],
            sort_keys=True,
        )
        self._recent_calls.append(sig)
        if len(self._recent_calls) < self._config.loop_detect_window:
            return False

        window = self._recent_calls[-self._config.loop_detect_window:]
        # Simple exact-match detection; can upgrade to fuzzy later
        return len(set(window)) == 1
