"""消息压缩策略。

三种策略：
1. NAIVE_SUMMARY    —— 调用 LLM 将旧消息压缩成一条自然语言摘要（需要 model）
2. SUMMARY_WITH_REFS —— 规则摘要 + 引用 ID；工具结果单独存储以便检索
3. SLIDING_WINDOW   —— 原样保留最近 N 条，其余直接丢弃

对外只暴露一个入口：async compress_messages()
非 LLM 策略（SLIDING_WINDOW / SUMMARY_WITH_REFS）虽然内部是纯同步逻辑，
但统一包在 async 函数里，调用方无需区分两套接口。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from scaffold.models.base import Message, Role

if TYPE_CHECKING:
    from scaffold.models.base import ChatModel


class CompressionStrategy(str, Enum):
    NAIVE_SUMMARY = "naive_summary"
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


async def compress_messages(
    messages: list[Message],
    *,
    strategy: CompressionStrategy = CompressionStrategy.SLIDING_WINDOW,
    keep_last_n: int = 6,
    ref_store: ReferenceStore | None = None,
    model: ChatModel | None = None,
) -> list[Message]:
    """按选定策略压缩消息历史，唯一对外入口。

    参数：
        messages:    完整消息历史（不含 system prompt）。
        strategy:    压缩算法。
        keep_last_n: 原样保留的最近消息数。
        ref_store:   SUMMARY_WITH_REFS 策略使用的引用存储。
        model:       NAIVE_SUMMARY 策略需要；其余策略忽略此参数。

    返回：
        压缩后的消息列表。
    """
    if len(messages) <= keep_last_n:
        return messages

    if strategy == CompressionStrategy.NAIVE_SUMMARY:
        if model is None:
            # 没有模型时降级到滑动窗口
            return _sliding_window(messages, keep_last_n)
        return await _naive_summary(messages, keep_last_n, model)

    if strategy == CompressionStrategy.SUMMARY_WITH_REFS:
        return _summary_with_refs(
            messages, keep_last_n,
            ref_store if ref_store is not None else ReferenceStore(),
        )

    # 默认：SLIDING_WINDOW
    return _sliding_window(messages, keep_last_n)


# ---------------------------------------------------------------------------
# 私有实现
# ---------------------------------------------------------------------------

def _sliding_window(messages: list[Message], keep_last_n: int) -> list[Message]:
    dropped = len(messages) - keep_last_n
    note = Message.system(
        f"[Context note: {dropped} earlier messages were truncated to save context space. "
        f"The conversation continues from the most recent messages below.]"
    )
    return [note] + messages[-keep_last_n:]


def _summary_with_refs(
    messages: list[Message],
    keep_last_n: int,
    ref_store: ReferenceStore,
) -> list[Message]:
    old = messages[:-keep_last_n]
    recent = messages[-keep_last_n:]

    parts: list[str] = []
    for m in old:
        if m.role == Role.TOOL and m.content:
            ref_id = ref_store.store(m.content)
            parts.append(f"- Tool '{m.name}' returned result (ref: {ref_id})")
        elif m.role == Role.ASSISTANT and m.tool_calls:
            names = [tc.name for tc in m.tool_calls]
            parts.append(f"- Assistant called tools: {', '.join(names)}")
        elif m.role == Role.USER and m.content:
            snippet = m.content[:100] + "..." if len(m.content) > 100 else m.content
            parts.append(f"- User: {snippet}")
        elif m.role == Role.ASSISTANT and m.content:
            snippet = m.content[:100] + "..." if len(m.content) > 100 else m.content
            parts.append(f"- Assistant: {snippet}")

    summary_text = (
        "[Conversation summary of earlier messages]\n"
        + "\n".join(parts)
        + "\n[Use retrieve_reference tool with ref_id to get full tool results if needed]"
    )
    return [Message.system(summary_text)] + recent


async def _naive_summary(
    messages: list[Message],
    keep_last_n: int,
    model: Any,
) -> list[Message]:
    old = messages[:-keep_last_n]
    recent = messages[-keep_last_n:]

    lines: list[str] = []
    for m in old:
        if m.role == Role.USER and m.content:
            lines.append(f"用户：{m.content}")
        elif m.role == Role.ASSISTANT and m.content:
            snippet = m.content[:300] + "…" if len(m.content) > 300 else m.content
            lines.append(f"助手：{snippet}")
        elif m.role == Role.ASSISTANT and m.tool_calls:
            names = [tc.name for tc in m.tool_calls]
            lines.append(f"助手调用工具：{', '.join(names)}")
        elif m.role == Role.TOOL and m.content:
            snippet = m.content[:200] + "…" if len(m.content) > 200 else m.content
            lines.append(f"工具返回：{snippet}")

    response = await model.chat(
        [
            Message.system("你是一个专业的对话摘要助手，请用简洁准确的语言总结对话内容。"),
            Message.user(
                "请将以下历史对话压缩成一段简短摘要（3-6 句话），"
                "保留关键事实、用户意图和工具调用的重要结果，"
                "供后续对话参考：\n\n"
                + "\n".join(lines)
            ),
        ],
        temperature=0.0,
    )
    summary_text = response.message.content or "（摘要生成失败）"
    return [Message.system(f"[以下是早期对话的 LLM 摘要]\n{summary_text}")] + recent
