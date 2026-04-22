"""retrieve_reference 工具 — 按 ref_id 取回 SUMMARY_WITH_REFS 压缩策略存储的完整工具结果。

当上下文使用 SUMMARY_WITH_REFS 策略压缩时，旧的工具结果会被替换为
"Tool 'X' returned result (ref: ref_xxxxxxxx)" 的摘要文本。
模型可以调用此工具按 ref_id 取回原始内容。

注意：ref_store 在每次 FSAgent.run() / new_context() 时通过 set_ref_store() 注入。
进程重启后 in-memory 的 ReferenceStore 会清空；ref_id 仅在当次会话内有效。
"""
from __future__ import annotations

from scaffold.context.compression import ReferenceStore
from fs_agent.tools.file_tools import registry

_ref_store: ReferenceStore | None = None


def set_ref_store(store: ReferenceStore) -> None:
    """将当前 ContextWindow 的 ReferenceStore 注入本模块，供 retrieve_reference 使用。"""
    global _ref_store
    _ref_store = store


@registry.tool(
    name="retrieve_reference",
    description=(
        "Retrieve the full content of a tool result that was summarised by the context compressor. "
        "Use this when the conversation summary mentions a ref_id like 'ref_xxxxxxxx' and you need "
        "the original content to answer accurately."
    ),
)
async def retrieve_reference(ref_id: str) -> str:
    """
    Args:
        ref_id: The reference identifier from the conversation summary (e.g. 'ref_1a2b3c4d').
    """
    if _ref_store is None:
        return "Reference store is not available in this session."
    content = _ref_store.retrieve(ref_id)
    if content is None:
        return f"Reference '{ref_id}' not found. It may have expired or the ref_id is incorrect."
    return content
