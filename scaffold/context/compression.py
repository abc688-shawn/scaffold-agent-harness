"""消息压缩策略。

当前实现了三种策略：
1. NaiveSummary —— 将旧消息压缩成一条摘要
2. SummaryWithRefs —— 摘要加引用 ID；工具结果单独存储以便检索
3. SlidingWindow —— 原样保留最近 N 轮，其余丢弃
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum

from scaffold.models.base import Message, Role


class CompressionStrategy(str, Enum):
    SLIDING_WINDOW = "sliding_window"
    SUMMARY_WITH_REFS = "summary_with_refs"


@dataclass
class ReferenceStore:
    """按引用 ID 存储工具结果，供后续检索。"""
    _store: dict[str, str] = field(default_factory=dict)

    def store(self, content: str) -> str:
        ref_id = f"ref_{uuid.uuid4().hex[:8]}"
        self._store[ref_id] = content
        return ref_id

    def retrieve(self, ref_id: str) -> str | None:
        return self._store.get(ref_id)

    def __len__(self) -> int:
        return len(self._store)


def compress_messages(
    messages: list[Message],
    *,
    strategy: CompressionStrategy = CompressionStrategy.SLIDING_WINDOW,
    keep_last_n: int = 6,
    ref_store: ReferenceStore | None = None,
) -> list[Message]:
    """按选定策略压缩消息历史。

    参数：
        messages: 完整消息历史（不包含 system prompt）。
        strategy: 使用哪种压缩算法。
        keep_last_n: 需要原样保留的最近消息数。
        ref_store: `summary_with_refs` 策略使用的引用存储。

    返回：
        压缩后的消息列表。
    """
    if len(messages) <= keep_last_n:
        return messages

    if strategy == CompressionStrategy.SLIDING_WINDOW:
        return _sliding_window(messages, keep_last_n)
    elif strategy == CompressionStrategy.SUMMARY_WITH_REFS:
        return _summary_with_refs(messages, keep_last_n, ref_store if ref_store is not None else ReferenceStore())
    else:
        return messages


def _sliding_window(messages: list[Message], keep_last_n: int) -> list[Message]:
    """仅保留最近 N 条消息。"""
    dropped = len(messages) - keep_last_n
    summary = Message.system(
        f"[Context note: {dropped} earlier messages were truncated to save context space. "
        f"The conversation continues from the most recent messages below.]"
    )
    return [summary] + messages[-keep_last_n:]


def _summary_with_refs(
    messages: list[Message],
    keep_last_n: int,
    ref_store: ReferenceStore,
) -> list[Message]:
    """概括旧消息，并将工具结果存为引用。"""
    old = messages[:-keep_last_n]
    recent = messages[-keep_last_n:]

    # 构造旧消息的摘要
    summary_parts: list[str] = []
    for m in old:
        if m.role == Role.TOOL and m.content:
            ref_id = ref_store.store(m.content)
            summary_parts.append(f"- Tool '{m.name}' returned result (ref: {ref_id})")
        elif m.role == Role.ASSISTANT and m.tool_calls:
            names = [tc.name for tc in m.tool_calls]
            summary_parts.append(f"- Assistant called tools: {', '.join(names)}")
        elif m.role == Role.USER and m.content:
            # 截断过长的用户消息
            snippet = m.content[:100] + "..." if len(m.content) > 100 else m.content
            summary_parts.append(f"- User: {snippet}")
        elif m.role == Role.ASSISTANT and m.content:
            snippet = m.content[:100] + "..." if len(m.content) > 100 else m.content
            summary_parts.append(f"- Assistant: {snippet}")

    summary_text = (
        "[Conversation summary of earlier messages]\n"
        + "\n".join(summary_parts)
        + "\n[Use retrieve_reference tool with ref_id to get full tool results if needed]"
    )

    summary_msg = Message.system(summary_text)
    return [summary_msg] + recent
