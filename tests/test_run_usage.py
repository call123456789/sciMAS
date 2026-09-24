"""Tests for token/cost usage capture and aggregation.

The CLI counts cache writes and cache reads *separately* from
``input_tokens``, which is uncached input only. sciMAS used to read just
``input_tokens``/``output_tokens``, so the recorded token count silently
excluded the prompt cache — on a resumed multi-agent session, most of the
context — and could not be reconciled with ``total_cost_usd``, which is
priced against all three. Token usage was also captured per call and never
totalled anywhere, so a batch reported what it cost but not what it used.

These tests pin the counters, the derived total, and the batch aggregate.

Run with:
    cd "$(git rev-parse --show-toplevel)"
    python -m pytest tests/test_run_usage.py -v
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from claude_runner import ClaudeRunner  # noqa: E402
from grader import GradingResult  # noqa: E402
from plan import AgentRun, PlannerRun, RUN_USAGE_FIELDS, sum_run_usage, total_tokens  # noqa: E402


# The shape a real `claude -p --output-format json` result takes, trimmed to
# the keys this code reads. Values are from a real planner invocation.
def _payload(**overrides):
    payload = {
        "type": "result",
        "is_error": False,
        "result": "an answer",
        "session_id": "sess-1",
        "total_cost_usd": 0.128774,
        "usage": {
            "input_tokens": 17726,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 1788,
            "output_tokens": 1570,
        },
    }
    payload.update(overrides)
    return payload


def _result(raw_json):
    return ClaudeRunner._build_result(
        raw_stdout=json.dumps(raw_json),
        raw_json=raw_json,
        result="an answer",
        stderr="",
        started_at="2026-01-01T00:00:00Z",
        duration_ms=1,
    )


def test_cache_counters_are_captured_from_the_usage_block():
    result = _result(_payload())

    assert result.input_tokens == 17726
    assert result.output_tokens == 1570
    assert result.cache_creation_input_tokens == 0
    assert result.cache_read_input_tokens == 1788
    assert result.total_cost_usd == pytest.approx(0.128774)


def test_total_input_tokens_folds_in_the_cache_counters():
    """`input_tokens` alone understates a cache-heavy call."""
    result = _result(_payload())

    assert result.total_input_tokens == 17726 + 0 + 1788 == 19514
    # ...and is strictly larger than the uncached counter it replaces.
    assert result.total_input_tokens > result.input_tokens


def test_a_zero_counter_does_not_fall_through_to_the_usage_block():
    """A real 0 (no cache writes) must not be replaced by another counter.

    The old lookup used `payload.get(k) or usage.get(k)`, where a genuine 0
    is falsy and silently picked up the nested value instead.
    """
    result = _result(
        {"input_tokens": 0, "usage": {"input_tokens": 999, "output_tokens": 5}}
    )

    assert result.input_tokens == 0
    assert result.total_input_tokens == 0


def test_unreported_counters_stay_none_rather_than_becoming_zero():
    """Absent usage must stay distinguishable from a free/zero-usage call.

    `_TextRunner` (plain-text output) and unrecognised models report no
    usage at all; recording 0 there would make "not measured" look like
    "cost nothing", which is how batches came to total $0.0000.
    """
    result = _result({"stream": True})

    assert result.total_cost_usd is None
    assert result.input_tokens is None
    assert result.total_input_tokens is None


def test_unparseable_counters_do_not_raise():
    """Bookkeeping must not fail a call that produced a good answer."""
    result = _result(
        {"total_cost_usd": "n/a", "input_tokens": "lots", "cache_read_input_tokens": "7"}
    )

    assert result.total_cost_usd is None
    assert result.input_tokens is None
    assert result.cache_read_input_tokens == 7


def test_stream_json_path_captures_the_same_counters(monkeypatch):
    """The orchestrator always uses stream-json, so pin that path too.

    Both output formats share `_usage_counters`, but only this test would
    catch the terminal `result` event failing to reach it.
    """
    events = [
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}},
        _payload(),
    ]

    class FakePopen:
        def __init__(self, *args, **kwargs):
            self.stdout = io.StringIO(
                "".join(json.dumps(e) + "\n" for e in events)
            )
            self.stderr = io.StringIO("")
            self.returncode = 0
            self.stdin = None

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr("claude_runner.subprocess.Popen", FakePopen)
    runner = ClaudeRunner(mcp_config_path="")
    result = runner._run_stream_json(["claude"], None, "2026-01-01T00:00:00Z", 0.0)

    assert result.result == "an answer"
    assert result.input_tokens == 17726
    assert result.cache_read_input_tokens == 1788
    assert result.total_input_tokens == 19514
    assert result.session_id == "sess-1"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _agent(**overrides):
    defaults = dict(
        role="physicist",
        step_id="s1",
        prompt_path="prompts/physicist.md",
        input_tokens=1000,
        output_tokens=200,
        cache_creation_input_tokens=50,
        cache_read_input_tokens=4000,
        cost_usd=0.25,
        duration_ms=1000,
    )
    defaults.update(overrides)
    return AgentRun(**defaults)


def test_sum_run_usage_returns_the_graders_own_keywords():
    """The mapping splats straight into `grade_problem`."""
    totals = sum_run_usage([_agent(), _agent(input_tokens=500, cost_usd=0.1)])

    assert set(totals) == set(RUN_USAGE_FIELDS)
    assert totals["input_tokens"] == 1500
    assert totals["cache_read_input_tokens"] == 8000
    assert totals["cost_usd"] == pytest.approx(0.35)
    assert totals["duration_ms"] == 2000


def test_sum_run_usage_tolerates_unmeasured_runs():
    """A run from a provider that reported nothing contributes 0, not None."""
    totals = sum_run_usage([_agent(input_tokens=None, cost_usd=None), None])

    assert totals["input_tokens"] == 0
    assert totals["cost_usd"] == 0
    assert totals["output_tokens"] == 200


def test_total_tokens_accepts_a_mapping_or_a_single_record():
    run = _agent()

    assert total_tokens(run) == 1000 + 200 + 50 + 4000 == 5250
    assert total_tokens(sum_run_usage([run])) == 5250
    # Also works on an archived record that predates the cache fields.
    assert total_tokens(PlannerRun(prompt_path="p", assembled_prompt="a", problem_text="t")) == 0


def test_build_summary_totals_tokens_alongside_cost():
    """The batch report used to carry a cost total and no token total."""
    from runner import RunConfig, _build_summary

    def grade(pid, **overrides):
        defaults = dict(
            problem_id=pid,
            dataset_name="SciAgentGYM",
            input_tokens=1000,
            output_tokens=200,
            cache_creation_input_tokens=50,
            cache_read_input_tokens=4000,
            cost_usd=0.25,
            duration_ms=1000,
        )
        defaults.update(overrides)
        return GradingResult(**defaults)

    summary = _build_summary([grade(1), grade(2, cache_read_input_tokens=0)], RunConfig())

    assert summary["total_input_tokens"] == 2000
    assert summary["total_output_tokens"] == 400
    assert summary["total_cache_creation_input_tokens"] == 100
    assert summary["total_cache_read_input_tokens"] == 4000
    assert summary["total_tokens"] == 2000 + 400 + 100 + 4000
    assert summary["total_cost_usd"] == pytest.approx(0.5)

    # An empty batch reports zeroes rather than omitting the keys, so a
    # reader never has to distinguish "absent" from "empty".
    empty = _build_summary([], RunConfig())
    assert empty["total_tokens"] == 0
    assert all(f"total_{name}" in empty for name in RUN_USAGE_FIELDS)
