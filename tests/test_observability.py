"""Tests for observability: tracer and storage."""
import time
import pytest
from pathlib import Path

from scaffold.observability.tracer import Tracer, SpanKind
from scaffold.observability.storage import TraceStorage


class TestTracer:
    def test_span_lifecycle(self):
        t = Tracer(run_id="test-run")
        parent = t.start_span("agent_run", kind=SpanKind.AGENT)
        child = t.start_span("llm_call", kind=SpanKind.LLM, parent=parent)

        time.sleep(0.01)
        t.end_span(child, metadata={"tokens": 100})
        t.end_span(parent)

        assert len(t.spans) == 2
        assert t.spans[1].parent_id == parent.id
        assert t.spans[1].metadata["tokens"] == 100
        assert t.spans[1].latency_ms > 0

    def test_to_dict(self):
        t = Tracer()
        s = t.start_span("test", kind=SpanKind.TOOL)
        t.end_span(s)
        d = t.to_dict()
        assert "run_id" in d
        assert len(d["spans"]) == 1


class TestTraceStorage:
    def test_save_and_list(self, tmp_path):
        db = tmp_path / "test.db"
        storage = TraceStorage(db)

        t = Tracer(run_id="run1")
        s = t.start_span("test", kind=SpanKind.AGENT)
        t.end_span(s)
        storage.save_trace(t, metadata={"query": "hello"})

        runs = storage.list_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == "run1"

        spans = storage.get_spans("run1")
        assert len(spans) == 1
        assert spans[0]["name"] == "test"

        storage.close()
