"""Command-line entry point: repeatable scenarios, no code edits needed.

    python -m kbharness ingest            # chunk + embed the KB into the store
    python -m kbharness ask "question"    # one query through the pipeline
    python -m kbharness eval              # run the eval set, print the report
    python -m kbharness multiturn         # run the multi-turn correction cases
    python -m kbharness demo              # ingest + 3 showcase questions

Every command takes --config to point at a different JSON config.
"""

from __future__ import annotations

import argparse
import json
import sys

from .agent import SupportAgent
from .config import Config
from .embeddings import HashingEmbedder
from .ingest import ingest
from .llm import default_llm
from .observability import TraceLogger
from .router import DOMAINS, KeywordRouter
from .store import VectorStore


def build_agent(config: Config) -> SupportAgent:
    """Wire the pieces together. This is the composition root."""
    store = VectorStore(HashingEmbedder())
    try:
        store.load(config.store_path)
    except (OSError, json.JSONDecodeError):
        ingest(config, store)  # first run: build the store from the KB
    router = KeywordRouter(DOMAINS, min_confidence=config.min_route_confidence)
    return SupportAgent(config, router, store, default_llm(config),
                        TraceLogger(config.log_path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kbharness")
    parser.add_argument("--config", help="path to a JSON config file")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest")
    ask_p = sub.add_parser("ask")
    ask_p.add_argument("query")
    sub.add_parser("eval")
    sub.add_parser("multiturn")
    sub.add_parser("demo")
    args = parser.parse_args(argv)

    config = Config.load(args.config)

    if args.command == "ingest":
        store = VectorStore(HashingEmbedder())
        count = ingest(config, store)
        store.save(config.store_path)
        print(f"ingested {count} chunks -> {config.store_path}")
        return 0

    agent = build_agent(config)

    if args.command == "ask":
        answer = agent.ask(args.query)
        print(f"[trace {answer.trace.trace_id}] domain={answer.trace.domain} "
              f"latency={answer.trace.latency_ms:.0f}ms")
        print(answer.text)
        return 0

    if args.command == "eval":
        from .eval import run_eval
        report = run_eval(agent, config.eval_set_path)
        print(json.dumps(report, indent=2))
        return 0 if not report["failures"] else 1

    if args.command == "multiturn":
        from .eval import run_multiturn
        print(json.dumps(run_multiturn(agent, config.multiturn_path), indent=2))
        return 0

    if args.command == "demo":
        for q in ["How do I get a refund for a yearly plan?",
                  "My webhook keeps timing out, what should I check?",
                  "Do you sell used cars?"]:
            print(f"\nQ: {q}")
            answer = agent.ask(q)
            print(f"A: {answer.text}")
            print(f"   (trace {answer.trace.trace_id}, "
                  f"domain={answer.trace.domain}, escalated={answer.trace.escalated})")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
