"""The answering pipeline: route -> retrieve -> generate -> cite/escalate.

This is the file to read first. Each stage is one small object with one
job, and each stage's output is recorded on the trace, so a wrong answer
can always be diagnosed: was the route wrong, the retrieval empty, or the
generation unfaithful?
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .llm import LLMClient
from .observability import StageTimer, Trace, TraceLogger
from .router import Router
from .store import VectorStore


@dataclass
class Answer:
    text: str
    trace: Trace
    citations: list[str] = field(default_factory=list)


class SupportAgent:
    """Coordinates the pipeline stages and owns the escalation policy."""

    def __init__(self, config: Config, router: Router, store: VectorStore,
                 llm: LLMClient, logger: TraceLogger | None = None):
        self.config = config
        self.router = router
        self.store = store
        self.llm = llm
        self.logger = logger or TraceLogger(config.log_path)

    def ask(self, query: str) -> Answer:
        timer = StageTimer()
        trace = Trace.new(query)

        # 1. Route: pick the domain whose docs retrieval may touch.
        decision = self.router.route(query)
        trace.domain = decision.domain
        trace.route_confidence = round(decision.confidence, 3)
        timer.mark("route")

        # 2. Retrieve: vector search inside the routed documents.
        results = self.store.search(
            query, self.config.top_k,
            doc_ids=set(decision.doc_ids) if decision.domain != "other" else None,
        )
        trace.retrieved = [
            {"chunk_id": r.chunk.id, "score": round(r.score, 4)} for r in results
        ]
        timer.mark("retrieve")

        # 3. Decide: weak evidence means escalate, never guess. Two signals:
        #    no domain claimed the query at all (out of scope for this KB),
        #    or retrieval confidence is too low to ground an answer on.
        top_score = results[0].score if results else 0.0
        no_domain_evidence = max(decision.scores.values(), default=0) == 0
        if no_domain_evidence or top_score < self.config.min_retrieval_score:
            trace.escalated = True
            trace.failure_category = "retrieval_miss"
            trace.answer = ("I don't have anything reliable on that in my "
                            "knowledge base - handing you to a human specialist.")
            timer.mark("answer")
            return self._finish(trace, timer)

        # 4. Generate: the LLM may only use the retrieved chunks.
        response = self.llm.complete(query, results)
        timer.mark("generate")
        trace.model = response.model
        trace.input_tokens = response.input_tokens
        trace.output_tokens = response.output_tokens

        if response.error:
            trace.escalated = True
            trace.failure_category = response.error
            trace.answer = ("I'm having trouble drafting an answer right now - "
                            "handing you to a human specialist.")
        elif response.text.strip() == "ESCALATE":
            trace.escalated = True
            trace.failure_category = "insufficient_context"
            trace.answer = ("My docs don't fully cover that - handing you to "
                            "a human specialist.")
        else:
            trace.answer = response.text
        return self._finish(trace, timer)

    def _finish(self, trace: Trace, timer: StageTimer) -> Answer:
        timer.mark("total")
        trace.stages = timer.marks
        trace.latency_ms = timer.marks["total_ms"]
        self.logger.log(trace)
        citations = [r["chunk_id"] for r in trace.retrieved
                     if r["chunk_id"] in trace.answer]
        return Answer(text=trace.answer, trace=trace, citations=citations)
