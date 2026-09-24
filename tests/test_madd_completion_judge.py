"""Tests for the MADD requirement-completion judge.

MADD's answer metric is a *judged* one now, so the judge is the only place a
score can be invented. Two rules keep that from happening, and both are what
this file pins down:

1. The model returns **counts**, never a score. It says which requirements got
   a molecule and how many properties each molecule carried;
   ``grader._madd_completion_from_parts`` does the arithmetic. A reply that
   reports the counts badly is rejected, not re-read charitably.
2. The requirement count comes from the task row, never from the reply. A
   judge that answers a shorter list than it was asked about would otherwise
   raise its own Half A by dropping the requirements it could not find.

No test here may reach the network. ``conftest.py`` points the judge config at
nonexistent paths and every judge call below is stubbed at ``grader``'s
boundary, because the MADD judge path has no cache and a real call bills.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import grader
import llm_judge
from dataset import MADD, MADD_REQUIRED_METRICS, load_dataset
from grader import VERDICT_JUDGE, VERDICT_LOCAL, grade_problem
from llm_judge import (
    MADDCompletionJudgement,
    _parse_madd_completion,
    judge_madd_completion,
)

TABLE = "\n".join(
    [
        "| SMILES | Docking score | QED | SA | PAINS | SureChEMBL | Glaxo | Brenk | BBB | IC50 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        "| CCO | -7.2 | 0.81 | 2.10 | 0 | 0 | 0 | 0 | 1 | 120 |",
    ]
)


@pytest.fixture(scope="module")
def problem():
    """The four-requirement MADD row used throughout."""
    return load_dataset(ROOT / "dataset" / MADD, dataset_name=MADD)[0]


def _stub_judge_config(monkeypatch) -> None:
    """Make credentials resolve without touching the network."""
    monkeypatch.setattr(
        grader,
        "_llm_resolve_judge_config",
        lambda **_: SimpleNamespace(model="stub", api_base="http://stub"),
    )


def _stub_judgement(monkeypatch, judgement, captured: dict | None = None):
    """Replace the judge helper at grader's boundary."""

    def fake(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        return judgement

    monkeypatch.setattr(grader, "_llm_judge_madd_completion", fake)


# ---------------------------------------------------------------------------
# 1. The judge's counts become the score, in Python
# ---------------------------------------------------------------------------


def test_judge_counts_are_scored_by_the_shared_arithmetic(monkeypatch, problem):
    """Half A from the flags, Half B from the coverage, nothing from the model.

    Two of four requirements answered (0.25) and two molecules carrying 9 and
    5 of the nine properties (mean 7/9, halved to 0.389) is 0.639 — a number
    the reply never contained.
    """
    _stub_judge_config(monkeypatch)
    _stub_judgement(
        monkeypatch,
        MADDCompletionJudgement(
            answered=[True, True, False, False],
            molecule_metric_coverage=[1.0, 5 / 9],
            reason="两个子任务没有给出分子",
        ),
    )

    result = grade_problem(problem, predicted_answer=TABLE, tool_calls=[])

    assert result.answer_score == pytest.approx(0.25 + 0.5 * ((1 + 5 / 9) / 2))
    assert result.answer_correct is False
    assert result.answer_verdict_source == VERDICT_JUDGE
    joined = " ".join(result.answer_notes)
    assert "2/4 sub-task(s)" in joined
    assert "两个子任务没有给出分子" in joined


def test_full_completion_scores_one_and_is_correct(monkeypatch, problem):
    _stub_judge_config(monkeypatch)
    _stub_judgement(
        monkeypatch,
        MADDCompletionJudgement(answered=[True] * 4, molecule_metric_coverage=[1.0] * 4),
    )

    result = grade_problem(problem, predicted_answer=TABLE, tool_calls=[])

    assert result.answer_score == 1.0
    assert result.answer_correct is True
    assert result.answer_verdict_source == VERDICT_JUDGE


def test_judge_reaches_the_tool_coverage_metric_too(monkeypatch, problem):
    """The two metrics are independent: a judged answer still has no calls."""
    _stub_judge_config(monkeypatch)
    _stub_judgement(
        monkeypatch,
        MADDCompletionJudgement(answered=[True] * 4, molecule_metric_coverage=[1.0] * 4),
    )

    result = grade_problem(problem, predicted_answer=TABLE, tool_calls=[])

    assert result.answer_score == 1.0
    assert result.tool_coverage == 0.0
    assert result.missing_tools == list(problem.expected_tools)


def test_the_judge_is_asked_about_every_requirement(monkeypatch, problem):
    """The prompt carries N readable labels, in the task's own order.

    A case code like ``Cnsr`` names nothing a judge could look for, so the
    labels have to come from the row's ``case`` text — and they must be in the
    same order as the codes, or a correct answer to requirement 2 is credited
    to requirement 1.
    """
    _stub_judge_config(monkeypatch)
    captured: dict = {}
    _stub_judgement(
        monkeypatch,
        MADDCompletionJudgement(answered=[True] * 4, molecule_metric_coverage=[1.0] * 4),
        captured,
    )

    grade_problem(problem, predicted_answer=TABLE, tool_calls=[])

    labels = captured["subtask_cases"]
    assert len(labels) == 4
    assert labels == [
        "alzheimer (Alzhmr)",
        "lung cancer (Cnsr)",
        "sclerosis (Sklrz)",
        "dyslipidemia (Dslpdm)",
    ]
    assert captured["required_metrics"] == list(MADD_REQUIRED_METRICS)
    assert captured["dataset"] == MADD
    assert captured["predicted"] == TABLE


# ---------------------------------------------------------------------------
# 2. Failure policy: give up, do not fall back
# ---------------------------------------------------------------------------


def test_a_failed_judge_call_does_not_fall_back_to_the_proxy(monkeypatch, problem):
    """A configured-but-broken judge must not be scored as if it had ruled.

    ``TABLE`` would score 0.625 under the deterministic parser (one molecule,
    full property row). Recording that would turn an API outage into a
    measurement of the answer, so the score is abandoned instead — and the
    note says so, because a bare 0.0 is indistinguishable from a real zero.
    """
    _stub_judge_config(monkeypatch)
    _stub_judgement(monkeypatch, None)

    result = grade_problem(problem, predicted_answer=TABLE, tool_calls=[])

    assert result.answer_score == 0.0
    assert result.answer_correct is False
    assert result.answer_verdict_source == VERDICT_LOCAL
    joined = " ".join(result.answer_notes)
    assert "score abandoned" in joined
    assert "judge not configured" not in joined


def test_unconfigured_judge_uses_the_deterministic_parser(monkeypatch, problem):
    """With no judge at all, the offline parser is the intended grader."""
    calls = [
        grader.ToolCall("generate_molecules_by_case", "pharma-drug-discovery", {"case": case})
        for case in ("alzheimer", "lung cancer", "multiple sclerosis", "dyslipidemia")
    ]

    result = grade_problem(problem, predicted_answer=TABLE, tool_calls=calls)

    assert result.answer_score == 0.5 * 0.25 + 0.5
    assert result.answer_verdict_source == VERDICT_LOCAL
    assert "judge not configured" in " ".join(result.answer_notes)


def test_empty_answer_never_reaches_the_judge(monkeypatch, problem):
    """Nothing to read, so nothing to bill for."""
    _stub_judge_config(monkeypatch)

    def explode(**kwargs):
        raise AssertionError("the judge must not be called for an empty answer")

    monkeypatch.setattr(grader, "_llm_judge_madd_completion", explode)

    result = grade_problem(problem, predicted_answer="", tool_calls=[])

    assert result.answer_score == 0.0
    assert result.answer_correct is False
    assert result.answer_verdict_source == VERDICT_LOCAL


# ---------------------------------------------------------------------------
# 3. Reply parsing — the length rule and the count rule
# ---------------------------------------------------------------------------


def _reply(subtasks, molecules, reason="ok"):
    return json.dumps(
        {"subtask_answered": subtasks, "molecules": molecules, "reason": reason}
    )


def test_reply_is_read_when_the_counts_line_up():
    judgement = _parse_madd_completion(
        _reply([True, False], [{"id": "CCO", "metrics_reported": 9}]), sub_tasks=2, metric_count=9
    )

    assert judgement is not None
    assert judgement.answered == [True, False]
    assert judgement.molecule_metric_coverage == [1.0]


def test_a_short_answer_list_is_rejected():
    """The judge may not shrink the requirement list to raise its own Half A.

    Two flags for four requirements parses as valid JSON and would score a
    confident 1.0 if the reply were trusted for the denominator, so the reply
    is discarded and the grader gives up instead.
    """
    assert (
        _parse_madd_completion(_reply([True], [{"metrics_reported": 9}]), sub_tasks=4, metric_count=9)
        is None
    )
    assert (
        _parse_madd_completion(
            _reply([True] * 5, [{"metrics_reported": 9}]), sub_tasks=4, metric_count=9
        )
        is None
    )


def test_out_of_range_counts_are_dropped():
    """A count outside 0..9 is not a reading of the answer."""
    judgement = _parse_madd_completion(
        _reply([True], [{"metrics_reported": 99}, {"metrics_reported": 9}]),
        sub_tasks=1,
        metric_count=9,
    )
    assert judgement is not None
    assert judgement.molecule_metric_coverage == [1.0]
    assert "out-of-range" in judgement.reason

    # Every row unusable means there is no honest reading of Half B — not even
    # zero, which would be an assertion about the answer rather than the reply.
    assert (
        _parse_madd_completion(
            _reply([True], [{"metrics_reported": 99}]), sub_tasks=1, metric_count=9
        )
        is None
    )


def test_an_empty_molecule_list_is_a_reading_not_a_failure():
    """The judge saying "no molecules" scores zero; it does not give up."""
    judgement = _parse_madd_completion(
        _reply([False, False], []), sub_tasks=2, metric_count=9
    )

    assert judgement is not None
    assert judgement.molecule_metric_coverage == []


def test_a_scalar_flag_is_not_read_as_all_true():
    """``true`` for the whole list would hand out Half A for free."""
    assert (
        _parse_madd_completion(
            _reply(True, [{"metrics_reported": 9}]), sub_tasks=1, metric_count=9
        )
        is None
    )


def test_unparseable_reply_is_rejected():
    assert _parse_madd_completion("I could not tell.", sub_tasks=4, metric_count=9) is None
    assert _parse_madd_completion(None, sub_tasks=4, metric_count=9) is None


def test_no_requirements_means_nothing_to_judge():
    assert _parse_madd_completion(_reply([], []), sub_tasks=0, metric_count=9) is None


# ---------------------------------------------------------------------------
# 4. The judge entry point's own preconditions
# ---------------------------------------------------------------------------


def test_judge_declines_without_requirements_or_metrics(monkeypatch):
    """Nothing to measure, so no call is made — not even a billed one."""
    monkeypatch.setattr(
        llm_judge, "_call_judge", lambda *a, **k: pytest.fail("must not call the API")
    )

    assert judge_madd_completion("q", [], ["QED"], "a", dataset=MADD) is None
    assert judge_madd_completion("q", ["alzheimer"], [], "a", dataset=MADD) is None


def test_judge_declines_when_unconfigured():
    """``conftest`` leaves the config paths missing, so this is the real state."""
    assert (
        judge_madd_completion("q", ["alzheimer"], ["QED"], "a", dataset=MADD) is None
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
