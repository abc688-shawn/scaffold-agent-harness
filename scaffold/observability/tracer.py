"""Lightweight tracing: each run is a tree of spans."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SpanKind(str, Enum):
    AGENT = "agent"
    LLM = "llm"
    TOOL = "tool"
    OTHER = "other"


@dataclass
class Span:
    id: str
    name: str
    kind: SpanKind
    parent_id: str | None = None
    start_time: float = 0.0
    end_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "running"  # running | completed | error

    @property
    def latency_ms(self) -> float:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0


class Tracer:
    """Collects spans for a single agent run."""

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self._spans: list[Span] = []

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.OTHER,
        parent: Span | None = None,
    ) -> Span:
        span = Span(
            id=uuid.uuid4().hex[:12],
            name=name,
            kind=kind,
            parent_id=parent.id if parent else None,
            start_time=time.time(),
        )
        self._spans.append(span)
        return span

    def end_span(
        self,
        span: Span,
        status: str = "completed",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        span.end_time = time.time()
        span.status = status
        if metadata:
            span.metadata.update(metadata)

    @property
    def spans(self) -> list[Span]:
        return list(self._spans)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "spans": [
                {
                    "id": s.id,
                    "name": s.name,
                    "kind": s.kind.value,
                    "parent_id": s.parent_id,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "latency_ms": s.latency_ms,
                    "metadata": s.metadata,
                    "status": s.status,
                }
                for s in self._spans
            ],
        }
