"""Tests for scaffold/loop/checkpoint.py — CheckpointStore + ReActLoop integration."""
from __future__ import annotations

import pytest

from scaffold.loop.checkpoint import (
    CheckpointStore,
    CheckpointRecord,
    _msgs_to_json,
    _msgs_from_json,
    _usage_to_json,
    _usage_from_json,
)
from scaffold.models.base import Message, Role, ToolCall, Usage


# ---------------------------------------------------------------------------
# Serialisation round-trips
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_plain_messages(self):
        msgs = [
            Message.user("hello"),
            Message.assistant("world"),
        ]
        assert _msgs_from_json(_msgs_to_json(msgs)) == msgs

    def test_tool_call_message(self):
        tc = ToolCall(id="t1", name="list_files", arguments={"path": "."})
        msgs = [Message.assistant(tool_calls=[tc])]
        restored = _msgs_from_json(_msgs_to_json(msgs))
        assert restored[0].tool_calls[0].name == "list_files"
        assert restored[0].tool_calls[0].arguments == {"path": "."}

    def test_tool_result_message(self):
        from scaffold.models.base import ToolResult
        result = ToolResult(tool_call_id="t1", name="list_files", content="a.txt\nb.txt")
        msgs = [Message.tool_result(result)]
        restored = _msgs_from_json(_msgs_to_json(msgs))
        assert restored[0].role == Role.TOOL
        assert "a.txt" in (restored[0].content or "")

    def test_usage_round_trip(self):
        u = Usage(prompt_tokens=100, completion_tokens=50)
        assert _usage_from_json(_usage_to_json(u)) == u

    def test_none_content_preserved(self):
        tc = ToolCall(id="t1", name="foo", arguments={})
        msg = Message.assistant(content=None, tool_calls=[tc])
        restored = _msgs_from_json(_msgs_to_json([msg]))[0]
        assert restored.content is None


# ---------------------------------------------------------------------------
# CheckpointStore CRUD
# ---------------------------------------------------------------------------

class TestCheckpointStore:
    @pytest.fixture()
    def store(self, tmp_path):
        s = CheckpointStore(tmp_path / "test.db")
        yield s
        s.close()

    def test_save_and_load(self, store):
        run_id = CheckpointStore.new_run_id()
        msgs = [Message.user("整理文件"), Message.assistant("好的")]
        usage = Usage(prompt_tokens=10, completion_tokens=5)

        store.save(run_id, "整理文件", step=2, messages=msgs, usage=usage)

        rec = store.load(run_id)
        assert rec is not None
        assert rec.run_id == run_id
        assert rec.user_input == "整理文件"
        assert rec.step == 2
        assert len(rec.messages) == 2
        assert rec.usage.prompt_tokens == 10
        assert not rec.completed

    def test_save_updates_existing(self, store):
        run_id = CheckpointStore.new_run_id()
        store.save(run_id, "q", step=1, messages=[Message.user("q")], usage=Usage())
        store.save(run_id, "q", step=3, messages=[Message.user("q"), Message.assistant("a")], usage=Usage(prompt_tokens=20))

        rec = store.load(run_id)
        assert rec.step == 3
        assert len(rec.messages) == 2
        assert rec.usage.prompt_tokens == 20

    def test_created_at_preserved_on_update(self, store):
        run_id = CheckpointStore.new_run_id()
        store.save(run_id, "q", step=1, messages=[Message.user("q")], usage=Usage())
        rec1 = store.load(run_id)
        store.save(run_id, "q", step=2, messages=[Message.user("q")], usage=Usage())
        rec2 = store.load(run_id)
        assert rec1.created_at == rec2.created_at

    def test_mark_complete(self, store):
        run_id = CheckpointStore.new_run_id()
        store.save(run_id, "q", step=1, messages=[Message.user("q")], usage=Usage())
        assert not store.load(run_id).completed

        store.mark_complete(run_id)
        assert store.load(run_id).completed

    def test_list_incomplete(self, store):
        ids = []
        for i in range(3):
            rid = CheckpointStore.new_run_id()
            ids.append(rid)
            store.save(rid, f"q{i}", step=i, messages=[Message.user(f"q{i}")], usage=Usage())

        # Complete one
        store.mark_complete(ids[0])

        incomplete = store.list_incomplete()
        incomplete_ids = {r.run_id for r in incomplete}
        assert ids[0] not in incomplete_ids
        assert ids[1] in incomplete_ids
        assert ids[2] in incomplete_ids

    def test_load_nonexistent_returns_none(self, store):
        assert store.load("nonexistent") is None

    def test_new_run_id_unique(self):
        ids = {CheckpointStore.new_run_id() for _ in range(100)}
        assert len(ids) == 100


# ---------------------------------------------------------------------------
# ReActLoop checkpoint integration
# ---------------------------------------------------------------------------

class TestLoopCheckpoint:
    @pytest.mark.asyncio
    async def test_checkpoint_saved_after_tool_step(self, tmp_path):
        """Loop saves a checkpoint after each tool step and marks complete on text reply."""
        from scaffold.loop.react import LoopConfig, ReActLoop
        from scaffold.context.window import ContextWindow
        from scaffold.context.budget import TokenBudget
        from scaffold.models.mock import MockModel
        from scaffold.models.base import ModelResponse, ToolCall as TC
        from scaffold.tools.registry import ToolRegistry

        store = CheckpointStore(tmp_path / "test.db")
        run_id = CheckpointStore.new_run_id()

        script = [
            ModelResponse(message=Message.assistant(tool_calls=[TC(id="t1", name="echo", arguments={})])),
            ModelResponse(message=Message.assistant("done")),
        ]
        model = MockModel(script=script)

        reg = ToolRegistry()

        @reg.tool(name="echo", description="echo")
        async def echo() -> str:
            return "echoed"

        ctx = ContextWindow(system_prompt="sys", budget=TokenBudget())
        loop = ReActLoop(
            model=model, tools=reg, context=ctx,
            config=LoopConfig(max_steps=5),
            middlewares=[],
            checkpoint_store=store,
            run_id=run_id,
        )

        result = await loop.run("test echo")
        assert result.final_message == "done"

        # After step 1 (tool step), checkpoint should exist
        rec = store.load(run_id)
        assert rec is not None
        assert rec.step >= 1
        assert rec.completed  # final text reply marks complete
        store.close()

    @pytest.mark.asyncio
    async def test_resume_continues_from_step(self, tmp_path):
        """resume() restores messages and continues without re-adding the user message."""
        from scaffold.loop.react import LoopConfig, ReActLoop
        from scaffold.context.window import ContextWindow
        from scaffold.context.budget import TokenBudget
        from scaffold.models.mock import MockModel
        from scaffold.models.base import ModelResponse
        from scaffold.tools.registry import ToolRegistry

        store = CheckpointStore(tmp_path / "resume.db")
        run_id = CheckpointStore.new_run_id()

        # Simulate a checkpoint saved after step 1 (tool already done)
        messages = [
            Message.user("say hello"),
            Message.assistant("hello"),
        ]
        store.save(run_id, "say hello", step=1, messages=messages, usage=Usage(prompt_tokens=10))

        # Resume: model should produce just the final text
        script = [ModelResponse(message=Message.assistant("resumed!"))]
        model = MockModel(script=script)

        ctx = ContextWindow(system_prompt="sys", budget=TokenBudget())
        for m in messages:
            ctx.add(m)

        loop = ReActLoop(
            model=model, tools=ToolRegistry(), context=ctx,
            config=LoopConfig(max_steps=5),
            middlewares=[],
            checkpoint_store=store,
            run_id=run_id,
        )
        loop._total_usage = Usage(prompt_tokens=10)

        result = await loop.run("say hello", _resume_step=1)
        assert result.final_message == "resumed!"
        # User message should NOT be added again (already in history)
        user_msgs = [m for m in ctx.history if m.role == Role.USER]
        assert len(user_msgs) == 1

        final_rec = store.load(run_id)
        assert final_rec.completed
        store.close()
