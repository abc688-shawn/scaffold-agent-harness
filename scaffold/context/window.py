"""ContextWindow —— 为 LLM 组装消息的顶层管理器。

职责：
- 维护有序的消息历史
- 在预算超限时执行压缩
- 支持按阶段切换的动态 system prompt
- 以适合 KV cache 的布局组装最终提示词：
    [system prompt (stable)] → [tool schemas (stable)] → [history (dynamic)]
"""
from __future__ import annotations

from enum import Enum

from scaffold.models.base import Message
from scaffold.context.budget import TokenBudget
from scaffold.context.compression import (
    CompressionStrategy,
    ReferenceStore,
    compress_messages,
)


class AgentPhase(str, Enum):
    """Agent 的执行阶段 —— 不同阶段可对应不同的 system prompt。"""
    PLANNING = "planning"
    EXECUTION = "execution"
    REFLECTION = "reflection"


class DynamicPrompt:
    """具备阶段感知能力的 system prompt 管理器。

    允许为不同阶段注册不同的提示词模板。
    各阶段共享的稳定前缀会固定放在最前面，以提升 KV cache 友好性。
    """

    def __init__(self, base_prompt: str) -> None:
        self._base = base_prompt
        self._phase_sections: dict[AgentPhase, str] = {}
        self._current_phase = AgentPhase.EXECUTION

    def set_phase_prompt(self, phase: AgentPhase, section: str) -> None:
        """为指定阶段注册一段额外提示词。"""
        self._phase_sections[phase] = section

    @property
    def phase(self) -> AgentPhase:
        return self._current_phase

    @phase.setter
    def phase(self, value: AgentPhase) -> None:
        self._current_phase = value

    def render(self) -> str:
        """构造当前阶段的完整 system prompt。

        布局：基础提示词 + 当前阶段专属段落（如果存在）。
        基础提示词始终作为前缀，从而尽量提高前缀缓存命中率。
        """
        parts = [self._base]
        section = self._phase_sections.get(self._current_phase)
        if section:
            parts.append(f"\n\n## Current Phase: {self._current_phase.value}\n{section}")
        return "\n".join(parts)


class ContextWindow:
    """管理单次 agent 运行的完整对话上下文。"""

    def __init__(
        self,
        system_prompt: str | DynamicPrompt,
        budget: TokenBudget | None = None,
        compression_strategy: CompressionStrategy = CompressionStrategy.SLIDING_WINDOW,
        keep_last_n: int = 10,
    ) -> None:
        if isinstance(system_prompt, str):
            self._dynamic_prompt = DynamicPrompt(system_prompt)
        else:
            self._dynamic_prompt = system_prompt
        self._budget = budget or TokenBudget()
        self._strategy = compression_strategy
        self._keep_last_n = keep_last_n
        self._messages: list[Message] = []
        self._ref_store = ReferenceStore()

    @property
    def ref_store(self) -> ReferenceStore:
        return self._ref_store

    @property
    def prompt(self) -> DynamicPrompt:
        """访问动态提示词管理器。"""
        return self._dynamic_prompt

    def add(self, message: Message) -> None:
        self._messages.append(message)

    def add_many(self, messages: list[Message]) -> None:
        self._messages.extend(messages)

    @property
    def history(self) -> list[Message]:
        return list(self._messages)

    def build_prompt(self) -> list[Message]:
        """为 LLM 组装完整提示词。

        布局（对 KV cache 友好）：
        1. System prompt（稳定前缀，提升前缀缓存命中率）
        2. 对话历史（必要时进行压缩）
        """
        # 检查是否需要压缩
        history_text = "".join(
            (m.content or "") for m in self._messages
        )
        history_tokens = self._budget.count_tokens(history_text)

        history = self._messages
        if self._budget.needs_compression(history_tokens):
            history = compress_messages(
                self._messages,
                strategy=self._strategy,
                keep_last_n=self._keep_last_n,
                ref_store=self._ref_store,
            )

        system_text = self._dynamic_prompt.render()
        return [Message.system(system_text)] + history

    def update_system_prompt(self, new_prompt: str) -> None:
        """替换基础 system prompt 文本。"""
        self._dynamic_prompt = DynamicPrompt(new_prompt)

    def set_phase(self, phase: AgentPhase) -> None:
        """切换 agent 阶段（并相应更新 system prompt）。"""
        self._dynamic_prompt.phase = phase

    def clear(self) -> None:
        self._messages.clear()
