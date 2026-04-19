"""ContextWindow — the top-level manager that assembles messages for the LLM.

Responsibilities:
- Maintain ordered message history
- Apply compression when budget exceeded
- Support phase-aware dynamic system prompts
- Assemble final prompt with KV-cache-friendly layout:
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
    """Agent execution phase — different phases get different system prompts."""
    PLANNING = "planning"
    EXECUTION = "execution"
    REFLECTION = "reflection"


class DynamicPrompt:
    """Phase-aware system prompt manager.

    Allows registering different prompt templates per phase.
    The stable prefix (shared across phases) stays at the top for KV-cache friendliness.
    """

    def __init__(self, base_prompt: str) -> None:
        self._base = base_prompt
        self._phase_sections: dict[AgentPhase, str] = {}
        self._current_phase = AgentPhase.EXECUTION

    def set_phase_prompt(self, phase: AgentPhase, section: str) -> None:
        """Register an additional prompt section for a specific phase."""
        self._phase_sections[phase] = section

    @property
    def phase(self) -> AgentPhase:
        return self._current_phase

    @phase.setter
    def phase(self, value: AgentPhase) -> None:
        self._current_phase = value

    def render(self) -> str:
        """Build the full system prompt for the current phase.

        Layout: base_prompt + phase-specific section (if any).
        Base prompt is always the prefix → maximises prefix cache hit.
        """
        parts = [self._base]
        section = self._phase_sections.get(self._current_phase)
        if section:
            parts.append(f"\n\n## Current Phase: {self._current_phase.value}\n{section}")
        return "\n".join(parts)


class ContextWindow:
    """Manages the full conversation context for one agent run."""

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
        """Access the dynamic prompt manager."""
        return self._dynamic_prompt

    def add(self, message: Message) -> None:
        self._messages.append(message)

    def add_many(self, messages: list[Message]) -> None:
        self._messages.extend(messages)

    @property
    def history(self) -> list[Message]:
        return list(self._messages)

    def build_prompt(self) -> list[Message]:
        """Assemble the full prompt for the LLM.

        Layout (KV-cache friendly):
        1. System prompt (stable prefix — maximizes prefix cache hit)
        2. Conversation history (compressed if needed)
        """
        # Check if compression needed
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
        """Replace the base system prompt text."""
        self._dynamic_prompt = DynamicPrompt(new_prompt)

    def set_phase(self, phase: AgentPhase) -> None:
        """Switch the agent phase (updates system prompt accordingly)."""
        self._dynamic_prompt.phase = phase

    def clear(self) -> None:
        self._messages.clear()
