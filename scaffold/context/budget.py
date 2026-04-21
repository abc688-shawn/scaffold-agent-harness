"""Token 预算分配。

总上下文预算被拆分为：
    system_prompt + tool_schemas + history + response_reserve

预算对象会跟踪使用情况，并在需要时通知上下文窗口进行压缩。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import tiktoken


@dataclass
class TokenBudget:
    """跟踪并约束一次对话的 token 预算。"""

    max_context_tokens: int = 128_000  # 模型上下文窗口
    response_reserve: int = 4_096      # 为模型输出预留的 token
    system_reserve: int = 2_000        # system prompt 的预估 token 数
    tool_schema_reserve: int = 1_500   # 工具 schema 的预估 token 数

    _encoding_name: str = field(default="cl100k_base", repr=False)

    def __post_init__(self) -> None:
        try:
            self._enc = tiktoken.get_encoding(self._encoding_name)
        except Exception:
            self._enc = None

    @property
    def available_for_history(self) -> int:
        """可分配给对话历史的 token 数。"""
        return (
            self.max_context_tokens
            - self.response_reserve
            - self.system_reserve
            - self.tool_schema_reserve
        )

    def count_tokens(self, text: str) -> int:
        if self._enc is None:
            # 粗略兜底估算：英文约 4 个字符 1 个 token，中文约 2 个字符 1 个 token
            return len(text) // 3
        return len(self._enc.encode(text))

    def count_messages_tokens(self, messages: list[dict[str, str]]) -> int:
        """估算一组消息的大致 token 数。"""
        total = 0
        for m in messages:
            total += 4  # 每条消息的固定开销
            for v in m.values():
                if isinstance(v, str):
                    total += self.count_tokens(v)
        return total

    def needs_compression(self, current_history_tokens: int) -> bool:
        return current_history_tokens > self.available_for_history
