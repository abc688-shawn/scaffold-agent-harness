"""上下文管理器测试：预算、压缩与窗口。"""
import pytest

from scaffold.models.base import Message, ToolCall, ToolResult, Role
from scaffold.context.budget import TokenBudget
from scaffold.context.compression import (
    compress_messages,
    CompressionStrategy,
    ReferenceStore,
)
from scaffold.context.window import ContextWindow


class TestTokenBudget:
    def test_available_for_history(self):
        b = TokenBudget(max_context_tokens=128_000, response_reserve=4096,
                        system_reserve=2000, tool_schema_reserve=1500)
        assert b.available_for_history == 128_000 - 4096 - 2000 - 1500

    def test_count_tokens(self):
        b = TokenBudget()
        count = b.count_tokens("Hello world")
        assert count > 0

    def test_needs_compression(self):
        b = TokenBudget(max_context_tokens=1000, response_reserve=100,
                        system_reserve=100, tool_schema_reserve=100)
        # 可用 token 为 700
        assert not b.needs_compression(500)
        assert b.needs_compression(800)


class TestReferenceStore:
    def test_store_and_retrieve(self):
        store = ReferenceStore()
        ref_id = store.store("file content here")
        assert ref_id.startswith("ref_")
        assert store.retrieve(ref_id) == "file content here"
        assert store.retrieve("nonexistent") is None
        assert len(store) == 1


class TestCompression:
    def _make_messages(self, n: int) -> list[Message]:
        msgs = []
        for i in range(n):
            msgs.append(Message.user(f"Question {i}"))
            msgs.append(Message.assistant(f"Answer {i}"))
        return msgs

    @pytest.mark.asyncio
    async def test_sliding_window(self):
        msgs = self._make_messages(10)  # 共 20 条消息
        compressed = await compress_messages(
            msgs, strategy=CompressionStrategy.SLIDING_WINDOW, keep_last_n=6
        )
        # 应该包含 1 条摘要 + 6 条最近消息
        assert len(compressed) == 7
        assert compressed[0].role == Role.SYSTEM
        assert "truncated" in compressed[0].content.lower()

    @pytest.mark.asyncio
    async def test_no_compression_needed(self):
        msgs = self._make_messages(2)  # 共 4 条消息
        compressed = await compress_messages(msgs, keep_last_n=6)
        assert len(compressed) == 4  # 保持不变

    @pytest.mark.asyncio
    async def test_summary_with_refs(self):
        store = ReferenceStore()
        msgs = [
            Message.user("Read file X"),
            Message.assistant(tool_calls=[ToolCall(id="1", name="read_file", arguments={"path": "x"})]),
            Message(role=Role.TOOL, content="long file content...", tool_call_id="1", name="read_file"),
            Message.assistant("File X contains..."),
            Message.user("What about Y?"),
            Message.assistant("Let me check Y"),
        ]
        compressed = await compress_messages(
            msgs, strategy=CompressionStrategy.SUMMARY_WITH_REFS,
            keep_last_n=2, ref_store=store,
        )
        assert len(store) >= 1  # 工具结果已被存储
        assert len(compressed) == 3  # 1 条摘要 + 2 条最近消息


class TestContextWindow:
    @pytest.mark.asyncio
    async def test_build_prompt(self):
        ctx = ContextWindow(system_prompt="You are helpful")
        ctx.add(Message.user("Hi"))
        ctx.add(Message.assistant("Hello!"))
        prompt = await ctx.build_prompt()
        assert prompt[0].role == Role.SYSTEM
        assert prompt[0].content == "You are helpful"
        assert len(prompt) == 3

    @pytest.mark.asyncio
    async def test_update_system_prompt(self):
        ctx = ContextWindow(system_prompt="Phase 1")
        ctx.update_system_prompt("Phase 2")
        prompt = await ctx.build_prompt()
        assert prompt[0].content == "Phase 2"
