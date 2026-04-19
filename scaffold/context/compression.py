"""Message compression strategies.

Three strategies implemented:
1. NaiveSummary — old messages → one summary message
2. SummaryWithRefs — summary + reference IDs; tool results stored for retrieval
3. SlidingWindow — keep last N turns verbatim, drop the rest
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
    """Stores tool results by reference ID for later retrieval."""
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
    """Compress message history according to the chosen strategy.

    Args:
        messages: Full message history (excluding system prompt).
        strategy: Which compression algorithm to use.
        keep_last_n: Number of recent messages to keep verbatim.
        ref_store: Reference store for summary_with_refs strategy.

    Returns:
        Compressed message list.
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
    """Keep only the last N messages."""
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
    """Summarize old messages and store tool results as references."""
    old = messages[:-keep_last_n]
    recent = messages[-keep_last_n:]

    # Build summary of old messages
    summary_parts: list[str] = []
    for m in old:
        if m.role == Role.TOOL and m.content:
            ref_id = ref_store.store(m.content)
            summary_parts.append(f"- Tool '{m.name}' returned result (ref: {ref_id})")
        elif m.role == Role.ASSISTANT and m.tool_calls:
            names = [tc.name for tc in m.tool_calls]
            summary_parts.append(f"- Assistant called tools: {', '.join(names)}")
        elif m.role == Role.USER and m.content:
            # Truncate long user messages
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
