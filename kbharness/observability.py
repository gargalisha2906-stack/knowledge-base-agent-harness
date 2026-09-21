"""Observability: one structured JSON line per query.

Every run through the pipeline emits a trace with a trace id, per-stage
latencies, the route taken, retrieval scores, token usage and a failure
category when something went wrong. Traces append to ``reports/traces.jsonl``
so they can be grepped, loaded into a notebook, or shipped to a real backend
later (this module is the only place that knows the log format).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StageTimer:
    """Small helper: time named stages inside one trace."""

    started_at: float = field(default_factory=time.perf_counter)
    marks: dict[str, float] = field(default_factory=dict)

    def mark(self, stage: str) -> None:
        self.marks[f"{stage}_ms"] = round((time.perf_counter() - self.started_at) * 1000, 2)


@dataclass
class Trace:
    trace_id: str
    query: str
    domain: str = ""
    route_confidence: float = 0.0
    retrieved: list[dict] = field(default_factory=list)
    answer: str = ""
    escalated: bool = False
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    stages: dict[str, float] = field(default_factory=dict)
    failure_category: str | None = None
    ts: float = field(default_factory=time.time)

    @staticmethod
    def new(query: str) -> "Trace":
        return Trace(trace_id=uuid.uuid4().hex[:12], query=query)


class TraceLogger:
    """Appends traces as JSONL; tolerates a missing/unwritable file."""

    def __init__(self, path: str):
        self.path = Path(path)

    def log(self, trace: Trace) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps(trace.__dict__) + "\n")
        except OSError:
            pass  # logging must never break the answer path


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile, e.g. p50 / p95 latency."""
    if not values:
        return 0.0
    values = sorted(values)
    rank = max(0, min(len(values) - 1, round(p / 100 * len(values)) - 1))
    return values[rank]
