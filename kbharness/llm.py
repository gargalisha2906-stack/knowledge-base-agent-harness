"""The generation seam.

``LLMClient`` is the only interface the agent knows. Two implementations:

* ``ClaudeLLM``     - Anthropic API, with timeout + bounded retries.
* ``ExtractiveLLM`` - offline fallback that stitches the top retrieved
  chunks into an answer-shaped response. It exists so the demo, tests and
  CI run with no API key, and so the system degrades honestly instead of
  failing hard when the API is unreachable.

Both return the same ``LLMResponse`` so evals can measure either one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from .config import Config
from .store import ScoredChunk

SYSTEM_PROMPT = """You are a customer-support assistant for Northwind Software.
Answer ONLY from the provided context chunks. Rules:
- If the chunks answer the question, answer in 2-4 sentences.
- Cite every claim with its chunk id in square brackets, e.g. [refund_policy.md#1].
- If the chunks do not answer the question, say exactly: ESCALATE
- Never invent policies, prices, or timelines."""


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    error: str | None = None       # failure category when generation failed


class LLMClient(Protocol):
    def complete(self, query: str, chunks: list[ScoredChunk]) -> LLMResponse: ...


def build_prompt(query: str, chunks: list[ScoredChunk]) -> str:
    """The grounded prompt: retrieved evidence first, then the question."""
    context = "\n\n".join(f"[{c.chunk.id}] {c.chunk.text}" for c in chunks)
    return f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"


class ClaudeLLM:
    """Anthropic-backed generator. Imported lazily so the package works
    without the ``anthropic`` extra installed."""

    def __init__(self, config: Config):
        if not config.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.config = config

    def complete(self, query: str, chunks: list[ScoredChunk]) -> LLMResponse:
        from anthropic import Anthropic  # lazy: optional dependency

        client = Anthropic(timeout=self.config.llm_timeout_s,
                           max_retries=self.config.llm_max_retries)
        started = time.perf_counter()
        try:
            message = client.messages.create(
                model=self.config.llm_model,
                max_tokens=self.config.llm_max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_prompt(query, chunks)}],
            )
        except Exception as exc:  # API, network and timeout errors land here
            return LLMResponse(text="", model=self.config.llm_model,
                               latency_ms=_elapsed(started),
                               error=f"llm_error:{type(exc).__name__}")
        return LLMResponse(
            text=message.content[0].text.strip(),
            model=self.config.llm_model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            latency_ms=_elapsed(started),
        )


class ExtractiveLLM:
    """Offline generator: quotes the strongest chunks with their citations.

    It cannot paraphrase, so its answers are clunky - but every word is
    traceable to a chunk, which makes it safe as a degraded mode.
    """

    model = "extractive-fallback"

    def complete(self, query: str, chunks: list[ScoredChunk]) -> LLMResponse:
        started = time.perf_counter()
        usable = [c for c in chunks if c.score > 0][:2]
        if not usable:
            return LLMResponse(text="ESCALATE", model=self.model,
                               latency_ms=_elapsed(started))
        parts = [f"From our docs [{c.chunk.id}]: {c.chunk.text}" for c in usable]
        return LLMResponse(text="\n\n".join(parts), model=self.model,
                           latency_ms=_elapsed(started))


def _elapsed(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def default_llm(config: Config) -> LLMClient:
    """Claude when a key is present, otherwise the offline fallback."""
    if config.anthropic_api_key:
        try:
            return ClaudeLLM(config)
        except RuntimeError:
            pass
    return ExtractiveLLM()
