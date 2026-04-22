"""断点检查点 — 每完成一个工具步骤即持久化 agent 状态，支持中断后恢复。

设计原则：
- 每次 ReActLoop.run() 生成一个 run_id，每完成一个工具调用步骤后写入 SQLite。
- 最终 LLM 文本回复后标记 completed=1。
- 如果进程崩溃，completed 仍为 0，可从最后一个检查点继续。
- 消息序列化：Message/ToolCall 转为普通字典后 JSON 存储；反序列化时恢复为 dataclass。
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from scaffold.models.base import Message, Role, ToolCall, Usage


_SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    run_id     TEXT    PRIMARY KEY,
    user_input TEXT    NOT NULL,
    step       INTEGER NOT NULL,
    messages   TEXT    NOT NULL,
    usage      TEXT    NOT NULL,
    created_at REAL    NOT NULL,
    updated_at REAL    NOT NULL,
    completed  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_cp_incomplete
    ON checkpoints(completed, updated_at DESC);
"""


@dataclass
class CheckpointRecord:
    run_id: str
    user_input: str
    step: int
    messages: list[Message]
    usage: Usage
    created_at: float
    updated_at: float
    completed: bool


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _msgs_to_json(messages: list[Message]) -> str:
    def _tc(tc: ToolCall) -> dict:
        return {"id": tc.id, "name": tc.name, "arguments": tc.arguments}

    def _msg(m: Message) -> dict:
        return {
            "role": m.role.value,
            "content": m.content,
            "tool_calls": [_tc(tc) for tc in m.tool_calls] if m.tool_calls else None,
            "tool_call_id": m.tool_call_id,
            "name": m.name,
        }

    return json.dumps([_msg(m) for m in messages], ensure_ascii=False)


def _msgs_from_json(text: str) -> list[Message]:
    result: list[Message] = []
    for d in json.loads(text):
        tcs = None
        if d.get("tool_calls"):
            tcs = [
                ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
                for tc in d["tool_calls"]
            ]
        result.append(Message(
            role=Role(d["role"]),
            content=d.get("content"),
            tool_calls=tcs,
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
        ))
    return result


def _usage_to_json(u: Usage) -> str:
    return json.dumps({"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens})


def _usage_from_json(text: str) -> Usage:
    d = json.loads(text)
    return Usage(prompt_tokens=d["prompt_tokens"], completion_tokens=d["completion_tokens"])


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class CheckpointStore:
    """SQLite-backed checkpoint store shared with traces.db."""

    def __init__(self, db_path: str | Path = "traces.db") -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @staticmethod
    def new_run_id() -> str:
        return uuid.uuid4().hex

    def save(
        self,
        run_id: str,
        user_input: str,
        step: int,
        messages: list[Message],
        usage: Usage,
        *,
        completed: bool = False,
    ) -> None:
        now = time.time()
        self._conn.execute(
            """
            INSERT INTO checkpoints
                (run_id, user_input, step, messages, usage, created_at, updated_at, completed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                step       = excluded.step,
                messages   = excluded.messages,
                usage      = excluded.usage,
                updated_at = excluded.updated_at,
                completed  = excluded.completed
            """,
            (
                run_id, user_input, step,
                _msgs_to_json(messages),
                _usage_to_json(usage),
                now, now, int(completed),
            ),
        )
        self._conn.commit()

    def mark_complete(self, run_id: str) -> None:
        self._conn.execute(
            "UPDATE checkpoints SET completed=1, updated_at=? WHERE run_id=?",
            (time.time(), run_id),
        )
        self._conn.commit()

    def load(self, run_id: str) -> CheckpointRecord | None:
        row = self._conn.execute(
            "SELECT run_id, user_input, step, messages, usage, "
            "created_at, updated_at, completed FROM checkpoints WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_record(row)

    def list_incomplete(self, limit: int = 10) -> list[CheckpointRecord]:
        rows = self._conn.execute(
            "SELECT run_id, user_input, step, messages, usage, "
            "created_at, updated_at, completed "
            "FROM checkpoints WHERE completed=0 "
            "ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_record(r) for r in rows]

    def close(self) -> None:
        self._conn.close()


def _row_to_record(row: tuple) -> CheckpointRecord:
    return CheckpointRecord(
        run_id=row[0],
        user_input=row[1],
        step=row[2],
        messages=_msgs_from_json(row[3]),
        usage=_usage_from_json(row[4]),
        created_at=row[5],
        updated_at=row[6],
        completed=bool(row[7]),
    )
