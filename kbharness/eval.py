"""The evaluation harness.

A harness *runs many evals* and reports aggregate numbers. The individual
evals here are deterministic checks - no LLM-as-a-judge needed for v1:

* routing accuracy      - did the router pick the expected domain?
* retrieval hit@k       - was at least one expected doc in the top k?
* citation correctness  - every [chunk_id] cited in the answer must exist in
                          what was actually retrieved (no invented citations)
* escalation correctness- queries marked out-of-scope must escalate, and
                          in-scope queries must not
* latency / cost        - p50 / p95 latency and token usage per query

Eval cases live in evals/eval_set.json so support ops can add cases without
touching code. Multi-turn correction cases live in evals/multiturn.json.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .agent import SupportAgent
from .observability import percentile


@dataclass
class EvalCase:
    id: str
    query: str
    expected_domain: str            # router expectation
    expected_docs: list[str]        # retrieval expectation (doc ids)
    should_escalate: bool = False   # out-of-scope queries expect escalation


@dataclass
class CaseResult:
    case: EvalCase
    domain_ok: bool
    hit_at_k: bool
    citations_ok: bool
    escalation_ok: bool
    latency_ms: float
    notes: list[str] = field(default_factory=list)


def load_eval_set(path: str) -> list[EvalCase]:
    raw = json.loads(Path(path).read_text())
    return [EvalCase(**item) for item in raw]


def run_case(agent: SupportAgent, case: EvalCase) -> CaseResult:
    started = time.perf_counter()
    answer = agent.ask(case.query)
    latency = (time.perf_counter() - started) * 1000
    trace = answer.trace

    domain_ok = (trace.domain == case.expected_domain) or (
        case.expected_domain == "other" and trace.domain == "other"
    )
    retrieved_docs = {r["chunk_id"].split("#")[0] for r in trace.retrieved}
    hit_at_k = bool(retrieved_docs & set(case.expected_docs)) if case.expected_docs else True
    citations_ok = all(c in {r["chunk_id"] for r in trace.retrieved} for c in answer.citations)
    escalation_ok = trace.escalated == case.should_escalate

    notes = []
    if not domain_ok:
        notes.append(f"routed {trace.domain!r}, expected {case.expected_domain!r}")
    if not hit_at_k:
        notes.append(f"expected docs {case.expected_docs} missing from top-k")
    if not citations_ok:
        notes.append("answer cites chunks that were never retrieved")
    if not escalation_ok:
        notes.append(f"escalated={trace.escalated}, expected {case.should_escalate}")
    return CaseResult(case, domain_ok, hit_at_k, citations_ok, escalation_ok,
                      latency, notes)


def run_eval(agent: SupportAgent, eval_set_path: str) -> dict:
    """Run every case and return a report dict (also printed by the CLI)."""
    cases = load_eval_set(eval_set_path)
    results = [run_case(agent, c) for c in cases]

    def rate(pred) -> float:
        return round(sum(1 for r in results if pred(r)) / len(results), 3)

    failures = [
        {"id": r.case.id, "query": r.case.query, "notes": r.notes}
        for r in results if r.notes
    ]
    latencies = [r.latency_ms for r in results]
    report = {
        "cases": len(results),
        "routing_accuracy": rate(lambda r: r.domain_ok),
        "retrieval_hit_at_k": rate(lambda r: r.hit_at_k),
        "citation_correctness": rate(lambda r: r.citations_ok),
        "escalation_correctness": rate(lambda r: r.escalation_ok),
        "latency_ms": {"p50": round(percentile(latencies, 50), 1),
                       "p95": round(percentile(latencies, 95), 1)},
        "failures": failures,   # failures are reported, never hidden
    }
    return report


def run_multiturn(agent: SupportAgent, path: str) -> dict:
    """Multi-turn correction cases: follow-ups must stay on the corrected topic.

    Each scenario is a list of turns. A turn passes when its expected doc is
    retrieved - this catches the classic failure where the agent keeps
    answering the *first* topic after the user corrects it.
    """
    scenarios = json.loads(Path(path).read_text())
    out = []
    for scenario in scenarios:
        turns = []
        for turn in scenario["turns"]:
            answer = agent.ask(turn["query"])
            got = {r["chunk_id"].split("#")[0] for r in answer.trace.retrieved}
            turns.append({"query": turn["query"],
                          "ok": bool(got & set(turn["expected_docs"]))})
        out.append({"id": scenario["id"],
                    "passed": all(t["ok"] for t in turns), "turns": turns})
    passed = sum(1 for s in out if s["passed"])
    return {"scenarios": len(out), "passed": passed, "detail": out}
