"""Tests for the DrugQA LLM-judge path.

DrugQA grades a *list* of acceptable answer items, and the judge reports two
numbers per problem:

- ``correct``  — every gold item was hit (the dataset's pass/fail), which
  ``runner._build_summary`` aggregates as ``answer_accuracy``.
- ``hit_rate`` — ``matched / total``, the partial-credit signal, aggregated
  as ``answer_score_avg``.

The judge is authoritative when it answers: its two numbers are reported
verbatim and the deterministic alias/token-overlap scorer
(``grader.grade_drugqa_answer``) is consulted only when the judge is
unconfigured, disabled, failed, or returned no usable count.

``conftest.py``'s autouse ``_isolate_judge_config`` points the judge config at
nonexistent paths, so the default state here is "a machine with no judge
credentials" — which is exactly the fallback case, and is also what keeps a
developer's real ``config/llm_judge.local.json`` from billing API calls during
this suite.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import grader
import llm_judge
from dataset import DRUGQA, SCIPREDICT, Problem
from grader import grade_drugqa_answer, grade_problem
from llm_judge import DrugQAJudgement, _parse_drugqa_judgement
from runner import RunConfig, _build_summary

GOLD_ITEMS = ["TNF - cytokine", "IL6 - cytokine"]

# A paraphrase the alias/token-overlap scorer cannot see: it never writes
# "TNF" or "IL6" as a token, so the local scorer gives it 0.0 — while it is a
# complete, correct answer. This is the case the judge exists for, and the
# disagreement is what makes "the judge's numbers are the ones reported"
# testable rather than coincidental.
PARAPHRASE = (
    "tumor necrosis factor alpha and interleukin-6 are the two central "
    "inflammatory mediators driving mucosal damage."
)
LOCAL_VISIBLE_PARTIAL = "TNF is the key cytokine."


def _problem(**overrides) -> Problem:
    base = dict(
        id="target_identification_15",
        filename="target_identification_and_moa_openended.jsonl",
        question=(
            "DrugQA open-ended drug-discovery QA task\n"
            "Task ID: target_identification_15\n"
            "Task type: Target Identification\n"
            "Disease/context: Ulcerative Colitis\n"
            "\nQuestion:\n"
            "Select two genes for modulating inflammation in UC.\n"
            "\nDeliverable:\n"
            "Answer the question directly and concisely."
        ),
        answer="\n".join(f"- {item}" for item in GOLD_ITEMS),
        subject="DrugQA",
        topic="Target Identification",
        dataset_name=DRUGQA,
        task_info={
            "drugqa_id": "target_identification_15",
            "drugqa_task_type": "target_identification",
            "drugqa_disease": "Ulcerative Colitis",
            "drugqa_answers": list(GOLD_ITEMS),
        },
    )
    base.update(overrides)
    return Problem(**base)


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

    monkeypatch.setattr(grader, "_llm_judge_drugqa_items", fake)


# ---------------------------------------------------------------------------
# 1. The two metrics reach GradingResult independently
# ---------------------------------------------------------------------------


def test_judge_splits_full_match_from_hit_rate(monkeypatch):
    """The core of the change: a half-hit answer is scored 0.5 but not correct.

    The answer is the paraphrase the local scorer rates 0.0, so a reported 0.5
    can only have come from the judge — the two metrics are split, and the
    fallback did not run.
    """
    _stub_judge_config(monkeypatch)
    _stub_judgement(
        monkeypatch,
        DrugQAJudgement(correct=False, hit_rate=0.5, matched=1, total=2, reason="只命中 TNF"),
    )

    result = grade_problem(_problem(), predicted_answer=PARAPHRASE, tool_calls=[])

    assert result.answer_correct is False
    assert result.answer_score == 0.5
    joined = " ".join(result.answer_notes)
    assert "matched 1/2" in joined
    assert "hit_rate=0.50" in joined
    assert "只命中 TNF" in joined
    assert "fell back to the local scorer" not in joined


def test_judge_full_match_scores_one(monkeypatch):
    _stub_judge_config(monkeypatch)
    _stub_judgement(
        monkeypatch,
        DrugQAJudgement(correct=True, hit_rate=1.0, matched=2, total=2),
    )

    result = grade_problem(_problem(), predicted_answer=PARAPHRASE, tool_calls=[])

    assert result.answer_correct is True
    assert result.answer_score == 1.0


def test_judge_numbers_win_over_a_disagreeing_local_scorer(monkeypatch):
    """A correct paraphrase the local scorer scores 0.0 must still grade 1.0.

    Without this, "the judge is authoritative" would be untestable: the two
    scorers happen to agree on most inputs.
    """
    problem = _problem()
    local_correct, local_score, _ = grade_drugqa_answer(problem, PARAPHRASE)
    assert (local_correct, local_score) == (False, 0.0)

    _stub_judge_config(monkeypatch)
    _stub_judgement(
        monkeypatch,
        DrugQAJudgement(correct=True, hit_rate=1.0, matched=2, total=2),
    )

    result = grade_problem(problem, predicted_answer=PARAPHRASE, tool_calls=[])

    assert (result.answer_correct, result.answer_score) == (True, 1.0)
    # The local figure survives only as a diagnostic, explicitly marked so.
    assert any("not used" in note for note in result.answer_notes)


def test_judge_sees_the_item_list_and_the_question(monkeypatch):
    _stub_judge_config(monkeypatch)
    captured: dict = {}
    _stub_judgement(
        monkeypatch,
        DrugQAJudgement(correct=True, hit_rate=1.0, matched=2, total=2),
        captured,
    )

    grade_problem(_problem(), predicted_answer=PARAPHRASE, tool_calls=[])

    assert captured["expected_items"] == GOLD_ITEMS
    assert captured["dataset"] == DRUGQA
    # The judge gets the wrapped prompt (task type + disease + deliverable),
    # not just the bare question.
    assert "Disease/context: Ulcerative Colitis" in captured["question"]
    assert captured["predicted"] == PARAPHRASE


def test_judge_falls_back_to_the_rendered_answer_when_no_item_list(monkeypatch):
    _stub_judge_config(monkeypatch)
    captured: dict = {}
    _stub_judgement(
        monkeypatch,
        DrugQAJudgement(correct=True, hit_rate=1.0, matched=1, total=1),
        captured,
    )

    problem = _problem(answer="TNF", task_info={})
    grade_problem(problem, predicted_answer="TNF", tool_calls=[])

    assert captured["expected_items"] == ["TNF"]


# ---------------------------------------------------------------------------
# 2. Fallback: only when the judge cannot rule
# ---------------------------------------------------------------------------


def test_judge_failure_falls_back_and_keeps_the_reason(monkeypatch):
    _stub_judge_config(monkeypatch)

    def failing_judge(**_):
        llm_judge._set_last_error("model 'x' exploded")
        return None

    monkeypatch.setattr(grader, "_llm_judge_drugqa_items", failing_judge)

    problem = _problem()
    expected_correct, expected_score, _ = grade_drugqa_answer(problem, LOCAL_VISIBLE_PARTIAL)

    result = grade_problem(problem, predicted_answer=LOCAL_VISIBLE_PARTIAL, tool_calls=[])

    joined = " ".join(result.answer_notes)
    assert "exploded" in joined
    assert "fell back to the local scorer" in joined
    # The deterministic scores are unchanged by this feature.
    assert (result.answer_correct, result.answer_score) == (expected_correct, expected_score)
    assert (expected_correct, expected_score) == (False, 0.5)


def test_unconfigured_judge_is_never_called(monkeypatch):
    """No credentials (conftest's default state) must not reach the transport."""
    monkeypatch.setattr(
        grader,
        "_llm_judge_drugqa_items",
        lambda **_: pytest.fail("judge called without credentials"),
    )

    result = grade_problem(_problem(), predicted_answer=LOCAL_VISIBLE_PARTIAL, tool_calls=[])

    joined = " ".join(result.answer_notes)
    assert "DrugQA LLM judge not configured" in joined
    assert "fell back to the local scorer" in joined


def test_disabled_switch_forces_the_local_scorer(monkeypatch):
    _stub_judge_config(monkeypatch)
    monkeypatch.setattr(grader, "_drugqa_judge_enabled", False)
    monkeypatch.setattr(
        grader,
        "_llm_judge_drugqa_items",
        lambda **_: pytest.fail("judge called while disabled"),
    )

    problem = _problem()
    result = grade_problem(problem, predicted_answer=PARAPHRASE, tool_calls=[])

    expected_correct, expected_score, _ = grade_drugqa_answer(problem, PARAPHRASE)
    assert (result.answer_correct, result.answer_score) == (expected_correct, expected_score)
    assert "--drugqa-judge off" in " ".join(result.answer_notes)


def test_empty_prediction_is_not_sent_to_the_judge(monkeypatch):
    _stub_judge_config(monkeypatch)
    monkeypatch.setattr(
        grader,
        "_llm_judge_drugqa_items",
        lambda **_: pytest.fail("judge called with an empty answer"),
    )

    result = grade_problem(_problem(), predicted_answer="", tool_calls=[])

    assert (result.answer_correct, result.answer_score) == (False, 0.0)
    assert "no predicted answer" in " ".join(result.answer_notes)


# ---------------------------------------------------------------------------
# 3. Both metrics aggregate into the run summary
# ---------------------------------------------------------------------------


def test_summary_reports_accuracy_and_mean_hit_rate(monkeypatch):
    """Two problems — one full match, one half — give 0.5 accuracy, 0.75 score."""
    _stub_judge_config(monkeypatch)

    judgements = {
        "full": DrugQAJudgement(correct=True, hit_rate=1.0, matched=2, total=2),
        "half": DrugQAJudgement(correct=False, hit_rate=0.5, matched=1, total=2),
    }
    monkeypatch.setattr(
        grader, "_llm_judge_drugqa_items", lambda **kw: judgements[kw["predicted"]]
    )

    problems = [_problem(id="p1"), _problem(id="p2")]
    results = [
        grade_problem(problems[0], predicted_answer="full", tool_calls=[]),
        grade_problem(problems[1], predicted_answer="half", tool_calls=[]),
    ]
    # Sanity: the stub really drove both rows.
    assert [r.answer_correct for r in results] == [True, False]
    assert [r.answer_score for r in results] == [1.0, 0.5]

    summary = _build_summary(results, RunConfig(dataset_name=DRUGQA))

    assert summary["answer_accuracy"] == 0.5
    assert summary["answer_score_avg"] == 0.75
    assert summary["n_problems"] == 2


# ---------------------------------------------------------------------------
# 4. The reply parser
# ---------------------------------------------------------------------------


def test_parser_reads_a_plain_json_reply():
    judgement = _parse_drugqa_judgement(
        '{"correct": true, "matched": 2, "total": 2, "hit_rate": 1.0, "reason": "ok"}',
        total=2,
    )

    assert asdict(judgement) == {
        "correct": True,
        "hit_rate": 1.0,
        "matched": 2,
        "total": 2,
        "reason": "ok",
    }


def test_parser_unwraps_a_fenced_reply_with_prose():
    judgement = _parse_drugqa_judgement(
        "好的，判断如下：\n```json\n{\"correct\": false, \"matched\": 1, \"total\": 5}\n```\n以上。",
        total=2,
    )

    assert (judgement.correct, judgement.matched, judgement.total) == (False, 1, 2)


def test_parser_ignores_the_judges_own_total():
    """The gold item count is the caller's, so a judge cannot shrink the denominator."""
    judgement = _parse_drugqa_judgement(
        '{"correct": true, "matched": 1, "total": 1, "hit_rate": 1.0}', total=4
    )

    assert judgement.total == 4
    assert judgement.matched == 1
    assert judgement.hit_rate == 0.25


def test_parser_derives_hit_rate_from_matched_not_the_reported_rate():
    """`hit_rate` is derived, so the two fields cannot contradict each other."""
    judgement = _parse_drugqa_judgement(
        '{"correct": false, "matched": 1, "hit_rate": 1.0}', total=2
    )

    assert judgement.hit_rate == 0.5
    # The judge's own contradiction is flagged rather than swallowed.
    assert "hit_rate=1.00 with matched=1/2" in judgement.reason


def test_parser_accepts_an_agreeing_rate_without_a_note():
    judgement = _parse_drugqa_judgement(
        '{"correct": true, "matched": 2, "hit_rate": 1.0}', total=2
    )

    assert judgement.reason == ""


def test_parser_derives_matched_from_a_percentage_rate():
    judgement = _parse_drugqa_judgement('{"correct": false, "hit_rate": 50}', total=4)

    assert judgement.matched == 2
    assert judgement.hit_rate == 0.5


def test_parser_infers_a_missing_count_from_the_verdict():
    """A judge that rules but omits the count keeps its verdict."""
    judgement = _parse_drugqa_judgement('{"correct": false, "reason": "wrong"}', total=3)

    assert judgement.correct is False
    assert judgement.hit_rate == 0.0
    assert "no item count" in judgement.reason


def test_parser_records_a_disagreement_instead_of_rewriting_it():
    judgement = _parse_drugqa_judgement('{"correct": true, "matched": 0}', total=2)

    assert judgement.correct is True
    assert judgement.hit_rate == 0.0
    assert "correct=True with matched=0/2" in judgement.reason


def test_parser_returns_none_without_a_usable_count():
    assert _parse_drugqa_judgement("no json here", total=2) is None
    assert _parse_drugqa_judgement("", total=2) is None
    assert _parse_drugqa_judgement('{"matched": 1}', total=0) is None


def test_parser_clamps_an_out_of_range_count():
    judgement = _parse_drugqa_judgement('{"matched": 99}', total=2)

    assert judgement.matched == 2
    assert judgement.hit_rate == 1.0


# ---------------------------------------------------------------------------
# 5. The prompt
# ---------------------------------------------------------------------------


def _rendered_prompt() -> str:
    return llm_judge._DRUGQA_PROMPT.format(
        question="Q?",
        expected="\n".join(f"{i}. {item}" for i, item in enumerate(GOLD_ITEMS, start=1)),
        predicted="A.",
        total=len(GOLD_ITEMS),
    )


class _FakeResponse:
    def __init__(self, text: str):
        self._text = text

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._text}}]}


def _stub_transport(monkeypatch, reply: str, captured: dict) -> None:
    """Give the judge credentials and stub the HTTP call, returning ``reply``.

    The tests above stub ``grader._llm_judge_drugqa_items``, which leaves the
    real prompt formatting, transport and parsing untested. This goes through
    ``llm_judge`` proper so the whole chain is exercised.
    """
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_API_KEY", "sk-test")
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_MODEL", "gpt-4.1")

    def fake_post(url, headers=None, json=None):
        captured["url"] = url
        captured["prompt"] = json["messages"][1]["content"]
        return _FakeResponse(reply)

    fake_client = mock.MagicMock()
    fake_client.__enter__ = mock.MagicMock(return_value=fake_client)
    fake_client.__exit__ = mock.MagicMock(return_value=False)
    fake_client.post = fake_post
    monkeypatch.setattr("httpx.Client", lambda **_: fake_client)


def test_end_to_end_judge_round_trip(monkeypatch):
    """Real prompt → real transport → real parser, one full grade."""
    captured: dict = {}
    _stub_transport(
        monkeypatch,
        '{"correct": false, "matched": 1, "total": 2, "hit_rate": 0.5, '
        '"reason": "只给出 TNF"}',
        captured,
    )

    result = grade_problem(_problem(), predicted_answer=PARAPHRASE, tool_calls=[])

    assert (result.answer_correct, result.answer_score) == (False, 0.5)
    assert "只给出 TNF" in " ".join(result.answer_notes)
    # The judge really was handed the item list, numbered.
    assert "1. TNF - cytokine" in captured["prompt"]
    assert "2. IL6 - cytokine" in captured["prompt"]
    assert "标准答案条目（共 2 条" in captured["prompt"]
    assert captured["url"].endswith("/chat/completions")


def test_end_to_end_unparseable_reply_falls_back_with_a_reason(monkeypatch):
    captured: dict = {}
    _stub_transport(monkeypatch, "我无法判断。", captured)

    result = grade_problem(_problem(), predicted_answer=PARAPHRASE, tool_calls=[])

    joined = " ".join(result.answer_notes)
    assert "unparseable" in joined
    assert "fell back to the local scorer" in joined
    # And the fallback is the deterministic score, not a guess from the reply.
    assert (result.answer_correct, result.answer_score) == (False, 0.0)


def test_end_to_end_transport_failure_falls_back(monkeypatch):
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_API_KEY", "sk-test")
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_MODEL", "gpt-4.1")

    def boom(*_, **__):
        raise ConnectionError("judge host unreachable")

    fake_client = mock.MagicMock()
    fake_client.__enter__ = mock.MagicMock(return_value=fake_client)
    fake_client.__exit__ = mock.MagicMock(return_value=False)
    fake_client.post = boom
    monkeypatch.setattr("httpx.Client", lambda **_: fake_client)

    result = grade_problem(_problem(), predicted_answer=PARAPHRASE, tool_calls=[])

    joined = " ".join(result.answer_notes)
    assert "judge host unreachable" in joined
    assert "fell back to the local scorer" in joined


# ---------------------------------------------------------------------------
# 6. A cut-off reply, and who produced the verdict
# ---------------------------------------------------------------------------


def test_end_to_end_truncated_reply_is_salvaged_not_fallen_back(monkeypatch):
    """The judge's verdict survives a reply that was cut off mid-JSON.

    ``preclinical_research_5`` in run 20260922-122803 came back as
    ``{"correct": false,`` — the model ran out of budget before closing the
    brace. Nothing about that reply is a *failure of the judge's judgement*, and
    discarding it handed the problem to a token-overlap scorer that called all
    five items hit. ``correct`` is the prompt's first field precisely so that
    the verdict is written first and survives a truncation.
    """
    captured: dict = {}
    reply = '{"correct": false, "matched": 2, "total": 2, "hit_rate": 0.5,'
    _stub_transport(monkeypatch, reply, captured)

    result = grade_problem(_problem(), predicted_answer=PARAPHRASE, tool_calls=[])

    joined = " ".join(result.answer_notes)
    assert result.answer_correct is False
    assert result.answer_verdict_source == grader.VERDICT_JUDGE
    assert "cut off mid-object" in joined
    assert "fell back to the local scorer" not in joined


def test_a_truncated_reply_without_the_verdict_is_not_salvaged(monkeypatch):
    """Nothing to recover means nothing to salvage — fall back, don't guess."""
    captured: dict = {}
    _stub_transport(monkeypatch, '{"matched": 2, "total": 2, "hit_rate":', captured)

    result = grade_problem(_problem(), predicted_answer=PARAPHRASE, tool_calls=[])

    assert result.answer_verdict_source == grader.VERDICT_LOCAL
    assert "fell back to the local scorer" in " ".join(result.answer_notes)


def test_a_judged_verdict_is_marked_as_the_judges(monkeypatch):
    _stub_judge_config(monkeypatch)
    _stub_judgement(
        monkeypatch, DrugQAJudgement(correct=True, hit_rate=1.0, matched=2, total=2)
    )

    result = grade_problem(_problem(), predicted_answer=PARAPHRASE, tool_calls=[])

    assert result.answer_verdict_source == grader.VERDICT_JUDGE


def test_a_fallback_verdict_says_so(monkeypatch):
    """The whole point: a local score must not read as a judge ruling.

    ``preclinical_research_5`` was reported ``answer_correct=True`` off a
    token-overlap match, and nothing in the run's aggregate distinguished it
    from a judgment the judge had actually made.
    """
    _stub_judge_config(monkeypatch)

    def failing_judge(**_):
        llm_judge._set_last_error("model 'x' returned no text (finish_reason=length)")
        return None

    monkeypatch.setattr(grader, "_llm_judge_drugqa_items", failing_judge)

    result = grade_problem(_problem(), predicted_answer=PARAPHRASE, tool_calls=[])

    assert result.answer_verdict_source == grader.VERDICT_LOCAL
    # The local scorer rates this paraphrase 0.0 — it is the judge that would
    # have called it correct, which is exactly what was lost.
    assert result.answer_correct is False


def test_an_empty_answer_is_never_reported_as_a_judge_verdict(monkeypatch):
    _stub_judge_config(monkeypatch)

    result = grade_problem(_problem(), predicted_answer="", tool_calls=[])

    assert result.answer_verdict_source == grader.VERDICT_LOCAL
    assert "skipped: no predicted answer" in " ".join(result.answer_notes)


def test_a_dataset_without_a_judge_is_not_reported_as_unjudged():
    """SciPredict and friends: the local scorer *is* the grader.

    MADD used to be the example here and no longer is — its answer metric is a
    judged completion score now, so it reports judge/local like any other
    judged dataset. The distinction this test defends is unchanged: a dataset
    with no judge path must not look like one whose judge never ruled.
    """
    result = grade_problem(
        _problem(dataset_name=SCIPREDICT, id="scipredict_1"),
        predicted_answer="x",
        tool_calls=[],
    )

    assert result.answer_verdict_source == grader.VERDICT_DETERMINISTIC


# ---------------------------------------------------------------------------
# 7. The aggregate keeps the two apart
# ---------------------------------------------------------------------------


def test_summary_separates_verified_from_unjudged(monkeypatch):
    """19 of 21 is not 21.

    This is the number the fixing was for: a run whose judge failed on 2
    problems reported a full point more accuracy than it had established, and
    the summary had no way to say so.
    """
    _stub_judge_config(monkeypatch)

    judged = DrugQAJudgement(correct=True, hit_rate=1.0, matched=2, total=2)

    def judge(**kw):
        if kw["predicted"] == PARAPHRASE:
            return judged
        llm_judge._set_last_error("model 'x' returned no text (finish_reason=length)")
        return None

    monkeypatch.setattr(grader, "_llm_judge_drugqa_items", judge)

    # Both problems end up "correct": p1 because the judge ruled, p2 because
    # the local scorer matched every gold item after the judge failed. Only one
    # of those is a measurement.
    results = [
        grade_problem(_problem(id="p1"), predicted_answer=PARAPHRASE, tool_calls=[]),
        grade_problem(
            _problem(id="p2"),
            predicted_answer="TNF - cytokine and IL6 - cytokine",
            tool_calls=[],
        ),
    ]
    summary = _build_summary(results, RunConfig(dataset_name=DRUGQA))

    assert [r.answer_correct for r in results] == [True, True]
    assert summary["answer_accuracy"] == 1.0
    assert summary["judge_verified_accuracy"] == 0.5
    assert summary["judge_verified_count"] == 1
    assert summary["unjudged_problems"] == 1
    assert summary["unjudged_problem_ids"] == ["p2"]


def test_summary_counts_a_fallback_pass_as_unverified(monkeypatch):
    """A token-overlap match is counted in accuracy, and *not* in verified."""
    _stub_judge_config(monkeypatch)

    def failing_judge(**_):
        llm_judge._set_last_error("model 'x' returned no text (finish_reason=length)")
        return None

    monkeypatch.setattr(grader, "_llm_judge_drugqa_items", failing_judge)

    # The local scorer rates this one a full hit.
    result = grade_problem(_problem(), predicted_answer="TNF - cytokine and IL6 - cytokine", tool_calls=[])

    assert (result.answer_correct, result.answer_verdict_source) == (True, "local")

    summary = _build_summary([result], RunConfig(dataset_name=DRUGQA))

    assert summary["answer_accuracy"] == 1.0
    assert summary["judge_verified_accuracy"] == 0.0
    assert summary["unjudged_problems"] == 1


def test_prompt_renders_literal_braces_for_the_json_example():
    """The prompt goes through str.format, so the JSON example must be escaped.

    A single unescaped brace in the template raises inside ``.format`` — the
    whole judge call would never be made.
    """
    prompt = _rendered_prompt()

    assert '{"correct"' in prompt
    assert '"hit_rate"' in prompt
    assert "{{" not in prompt and "}}" not in prompt
    assert "标准答案条目（共 2 条" in prompt
    assert "1. TNF - cytokine" in prompt
    assert "2. IL6 - cytokine" in prompt
