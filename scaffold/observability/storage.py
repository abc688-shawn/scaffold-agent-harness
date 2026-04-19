"""SQLite-backed trace storage."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from scaffold.observability.tracer import Tracer


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at REAL,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS spans (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    parent_id TEXT,
    start_time REAL,
    end_time REAL,
    latency_ms REAL,
    status TEXT,
    metadata TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_spans_run ON spans(run_id);
"""


class TraceStorage:
    """Persist traces to a local SQLite database."""

    def __init__(self, db_path: str | Path = "traces.db") -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def save_trace(self, tracer: Tracer, metadata: dict[str, Any] | None = None) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, created_at, metadata) VALUES (?, ?, ?)",
            (tracer.run_id, time.time(), json.dumps(metadata or {})),
        )
        for s in tracer.spans:
            self._conn.execute(
                "INSERT OR REPLACE INTO spans "
                "(id, run_id, name, kind, parent_id, start_time, end_time, latency_ms, status, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    s.id, tracer.run_id, s.name, s.kind.value, s.parent_id,
                    s.start_time, s.end_time, s.latency_ms, s.status,
                    json.dumps(s.metadata),
                ),
            )
        self._conn.commit()

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT run_id, created_at, metadata FROM runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"run_id": r[0], "created_at": r[1], "metadata": json.loads(r[2])}
            for r in rows
        ]

    def get_spans(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, name, kind, parent_id, start_time, end_time, latency_ms, status, metadata "
            "FROM spans WHERE run_id = ? ORDER BY start_time",
            (run_id,),
        ).fetchall()
        return [
            {
                "id": r[0], "name": r[1], "kind": r[2], "parent_id": r[3],
                "start_time": r[4], "end_time": r[5], "latency_ms": r[6],
                "status": r[7], "metadata": json.loads(r[8]),
            }
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()
