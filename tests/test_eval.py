"""The eval harness itself gets tested: a passing case and a rigged failure."""

from kbharness.observability import percentile


def test_percentile():
    values = list(range(1, 101))  # 1..100
    assert percentile(values, 50) == 50
    assert percentile(values, 95) == 95
    assert percentile([], 95) == 0.0


def test_eval_set_loads():
    from kbharness.eval import load_eval_set
    cases = load_eval_set("evals/eval_set.json")
    assert len(cases) >= 40
    assert any(c.should_escalate for c in cases)
