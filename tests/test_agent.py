"""End-to-end pipeline tests over the real demo KB (fully offline)."""

from kbharness.agent import SupportAgent
from kbharness.config import Config
from kbharness.embeddings import HashingEmbedder
from kbharness.ingest import ingest
from kbharness.llm import ExtractiveLLM
from kbharness.observability import TraceLogger
from kbharness.router import DOMAINS, KeywordRouter
from kbharness.store import VectorStore


def make_agent(tmp_path) -> SupportAgent:
    config = Config(log_path=str(tmp_path / "traces.jsonl"))
    store = VectorStore(HashingEmbedder())
    ingest(config, store)
    router = KeywordRouter(DOMAINS, min_confidence=config.min_route_confidence)
    return SupportAgent(config, router, store, ExtractiveLLM(),
                        TraceLogger(config.log_path))


def test_answers_with_citation(tmp_path):
    answer = make_agent(tmp_path).ask("How do I get a refund for a yearly plan?")
    assert not answer.trace.escalated
    assert answer.trace.domain == "refunds"
    assert answer.citations  # at least one retrieved chunk was cited


def test_out_of_scope_escalates(tmp_path):
    answer = make_agent(tmp_path).ask("Do you sell used cars?")
    assert answer.trace.escalated
    assert answer.trace.failure_category == "retrieval_miss"


def test_trace_has_stages_and_id(tmp_path):
    answer = make_agent(tmp_path).ask("What is the API rate limit?")
    stages = answer.trace.stages
    assert {"route_ms", "retrieve_ms", "generate_ms", "total_ms"} <= set(stages)
    assert answer.trace.trace_id


def test_traces_logged_as_jsonl(tmp_path):
    agent = make_agent(tmp_path)
    agent.ask("How do I invite a teammate?")
    agent.ask("Do you sell used cars?")
    import json
    lines = (tmp_path / "traces.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["trace_id"]
