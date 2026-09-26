# Knowledge-Base Agent Harness

A reusable harness that turns a folder of support documents into a grounded,
citable answering agent - with routing, retrieval, evaluation and
observability built in, and explicit escalation instead of confident
guessing.

Built as a reference implementation for customer-support knowledge bases:
small enough to read in one sitting, structured so each stage is a swappable
seam.

```
kbharness/
  config.py         one dataclass: defaults <- JSON file <- env vars
  chunking.py       documents -> overlapping chunks with source ids
  embeddings.py     Embedder protocol + offline hashing embedder
  store.py          local vector store (cosine search, JSON persistence)
  ingest.py         kb directory -> chunks -> store
  router.py         query -> domain -> which docs retrieval may touch
  retriever         (store.search, called by the agent)
  llm.py            LLMClient protocol: ClaudeLLM | ExtractiveLLM (offline)
  agent.py          the pipeline: route -> retrieve -> generate -> cite/escalate
  observability.py  trace ids, per-stage latency, JSONL traces, percentiles
  eval.py           the harness: runs the eval set, reports metrics
  cli.py            python -m kbharness ingest|ask|eval|multiturn|demo
data/kb/            sample knowledge base (fictional "Northwind Software")
evals/              44-case eval set + multi-turn correction scenarios
tests/            pytest suite, fully offline
docs/THREAT_MODEL.md
```

## Why I built this

I'm on the tech ops team at PhonePe. Operations folks come to us with questions all day, and honestly, most answers already exist in our docs. The problem is there are so many documents, in so many versions, that pointing someone to the right one takes longer than just answering. And when we can't find it, people guess.

So I built the thing I wanted to exist: you ask a question, it finds the right chunk of the right document, answers with the citation, and if the answer isn't there it says so instead of hallucinating.

It's small on purpose. Each stage sits behind an interface, so I can swap in a real embedding model or Claude for generation without rewriting everything.

## Architecture

```
            query
              |
              v
        +-----------+     confidence too low / out of scope
        |  Router   |----------------------------------+
        +-----------+                                  |
              | domain + allowed docs                  |
              v                                        |
        +-----------+     top score below threshold    |
        | Retriever |----------------------------------|
        +-----------+                                  |
              | top-k chunks                           |
              v                                        |
        +-----------+     no grounded answer / API     |
        | Generator |     failure -> ESCALATE          |
        |  (LLM)    |----------------------------------|
        +-----------+                                  v
              | answer with [chunk] citations    human handoff
              v
         response
        (every stage logged to reports/traces.jsonl with a trace id)
```

The pipeline is deliberately linear. An "agentic" loop (re-retrieve, rewrite
the query, call tools) plugs into the same stage interfaces when a use case
earns it.

## Quickstart

```bash
pip install -r requirements.txt        # only needed for Claude generation
python -m kbharness ingest             # build the local vector store
python -m kbharness demo               # 3 showcase questions, fully offline
python -m kbharness ask "How do I get a refund for a yearly plan?"
python -m kbharness eval               # run the 44-case eval set
python -m kbharness multiturn          # multi-turn correction scenarios
pytest                                 # 21 tests, no API key needed
```

Everything above runs **without any API key**: the default hashing embedder
is offline and the generator falls back to an extractive mode that quotes
the retrieved chunks. For real generation:

```bash
export ANTHROPIC_API_KEY=...           # the only secret, env-var only
python -m kbharness ask "How do I get a refund for a yearly plan?"
```

With Docker:

```bash
docker build -t kbharness .
docker run --rm kbharness              # runs tests + eval
docker run --rm -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY kbharness \
    python -m kbharness ask "How do I get a refund for a yearly plan?"
```

## Example run

After `python -m kbharness ingest`, a question runs the full pipeline offline:

```
$ python -m kbharness ask "How do I get a refund for a yearly plan?"
[trace ef5ff884e5ea] domain=refunds latency=1ms

From our docs [refund_policy.md#0]: Monthly plans can be cancelled at any time and stop renewing at the end of the current billing month. Yearly plans can be refunded in full within 30 days of purchase. After 30 days, yearly plans are refunded pro-rata for the remaining full months, minus a 10% administration fee.

From our docs [refund_policy.md#1]: Refunds are issued to the original payment method within 5-7 business days. To request a refund, email billing@northwind.example from the account email address and include your invoice number.
```

And when the docs don't cover the question, it escalates instead of guessing:

```
$ python -m kbharness ask "Do you sell used cars?"
I don't have anything reliable on that in my knowledge base - handing you to a human specialist.
   (trace 3850e6601bc3, domain=other, escalated=True)
```

Both are copied from a real offline run (no API key). Every answer carries a trace id; `reports/traces.jsonl` has the per-stage latency breakdown behind it.

## The eval harness

`evals/eval_set.json` holds 44 cases across refunds, billing, technical,
account, and out-of-scope queries. `python -m kbharness eval` reports:

* **routing accuracy** - did the router pick the expected domain?
* **retrieval hit@k** - was an expected document in the top k?
* **citation correctness** - every cited chunk id must exist in what was
  actually retrieved (invented citations fail)
* **escalation correctness** - out-of-scope queries must escalate;
  in-scope queries must not
* **latency p50/p95** and token usage per query

Failures print per case with the reason; they are never hidden. Current
numbers on the offline embedder (see `reports/eval_report.md`):

| metric | score |
|---|---|
| routing accuracy | 1.000 |
| retrieval hit@k | 1.000 |
| citation correctness | 1.000 |
| escalation correctness | 1.000 |
| latency p50 / p95 | ~1.7ms / ~3.7ms (offline) |

## Design decisions

* **Escalate, never guess.** Two signals force a human handoff: no domain
  claims the query (router evidence is zero), or the best retrieval score is
  below threshold. A support bot that admits ignorance beats one that
  invents policy.
* **Routing before retrieval.** The router restricts which documents vector
  search may touch, so a billing question can never match a security doc.
  It also doubles as a tenant-isolation seam.
* **Everything is a seam.** `Embedder`, `LLMClient`, `Router`, `VectorStore`
  are protocols. Swap the hashing embedder for a hosted model, or the JSON
  store for a real vector DB, without touching the pipeline.
* **Offline by default.** Demo, tests and CI need no key, no network, and
  no downloads. The Anthropic dependency is imported lazily.
* **Plain Python.** No framework soup: dataclasses, protocols, and one JSON
  file per artifact. Every module fits in your head.
* **Citations are checked, not trusted.** The eval verifies each cited
  chunk was actually retrieved; the system prompt forbids claims without a
  source.

## Configuration

`kbharness.config.Config`: dataclass defaults <- optional JSON file
(`--config config.json`) <- environment overrides (`KBH_TOP_K=8`). The only
secret is `ANTHROPIC_API_KEY`, read from the environment and never written
anywhere.

## Known limitations (honest ones)

* The default embedder is lexical, not semantic: it matches vocabulary and
  phrases, so paraphrases with disjoint wording score low. The
  `ANTHROPIC_API_KEY` improves generation, not retrieval; plug a hosted
  embedding model into `Embedder` for semantic retrieval.
* The router is keyword scoring. It is transparent and debuggable, but an
  LLM classifier would catch intent without vocabulary overlap.
* The vector store is linear scan over one JSON file. Fine for hundreds of
  chunks; past that, swap in a real index behind the same `add`/`search`
  interface.
* The eval set is hand-written and small; treat 100% as "covers what we
  thought to test", not "solved".

## Security

See `docs/THREAT_MODEL.md` - prompt injection, tenant isolation, PII, and
secret handling in one page.
