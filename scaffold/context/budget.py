"""Token budget allocation.

Total context budget is split:
    system_prompt + tool_schemas + history + response_reserve

The budget tracks usage and tells the context window when compression is needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import tiktoken


@dataclass
class TokenBudget:
    """Tracks and enforces token budget for a conversation."""

    max_context_tokens: int = 128_000  # model context window
    response_reserve: int = 4_096      # tokens reserved for model output
    system_reserve: int = 2_000        # estimated system prompt tokens
    tool_schema_reserve: int = 1_500   # estimated tool schema tokens

    _encoding_name: str = field(default="cl100k_base", repr=False)

    def __post_init__(self) -> None:
        try:
            self._enc = tiktoken.get_encoding(self._encoding_name)
        except Exception:
            self._enc = None

    @property
    def available_for_history(self) -> int:
        """Tokens available for conversation history."""
        return (
            self.max_context_tokens
            - self.response_reserve
            - self.system_reserve
            - self.tool_schema_reserve
        )

    def count_tokens(self, text: str) -> int:
        if self._enc is None:
            # Rough fallback: 1 token ≈ 4 chars for English, ~2 chars for Chinese
            return len(text) // 3
        return len(self._enc.encode(text))

    def count_messages_tokens(self, messages: list[dict[str, str]]) -> int:
        """Estimate token count for a list of messages."""
        total = 0
        for m in messages:
            total += 4  # message overhead
            for v in m.values():
                if isinstance(v, str):
                    total += self.count_tokens(v)
        return total

    def needs_compression(self, current_history_tokens: int) -> bool:
        return current_history_tokens > self.available_for_history
