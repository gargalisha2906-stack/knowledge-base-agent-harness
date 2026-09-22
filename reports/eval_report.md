# Eval report

Generated 2026-09-21 with `python -m kbharness eval` and
`python -m kbharness multiturn` on the offline embedder + extractive fallback
generator (no API key). Re-run with `ANTHROPIC_API_KEY` set for Claude-backed
numbers.

## Metrics (44 cases)

| metric | score |
|---|---|
| routing accuracy | 1.000 |
| retrieval hit@k | 1.000 |
| citation correctness | 1.000 |
| escalation correctness | 1.000 |
| latency p50 / p95 | 1.4ms / 3.7ms |

Failures: 0 (every failure prints with its reason when present)

## Multi-turn correction scenarios

3/3 passed (mt-refund-to-billing, mt-import-to-webhook, mt-roles-to-2fa)

## Reading these numbers

The eval set is hand-written and small: 100% means "covers what we thought to
test", not "solved". The out-of-scope escalation cases are the ones to grow
first - real users ask much stranger questions than `evals/eval_set.json` does.
