"""Tests for ``report.py --regrade``.

A re-grade exists for one situation: the judge failed on some problems and the
verdicts written for them are the local scorer's, so the fix is to ask the
judge again without re-running the agent. Two things therefore have to hold:

- it changes the answer fields and *only* the answer fields. Tool coverage,
  cost, duration and tokens are records of what the run did, and the
  per-problem payload does not hold what they would need to be recomputed
  (tool names, not arguments; ``available_tools`` was never recorded). A
  re-grade that quietly re-derived them would replace a measurement with a
  worse one and look identical while doing it.
- it can be aimed. The judge paths have no response cache, so re-grading is
  re-billing, and the whole point of ``--only`` is to pay for the two problems
  that need it rather than the nineteen that do not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import grader
import llm_judge
import report
from dataset import DRUGQA, SCIAGENTGYM
from llm_judge import DrugQAJudgement

GOLD_ITEMS = ["TNF - cytokine", "IL6 - cytokine"]
PARAPHRASE = "tumor necrosis factor alpha and interleukin-6 drive the damage."


def _problem_payload(pid: str, dataset: str = DRUGQA) -> dict:
    return {
        "id": pid,
        "filename": "target_identification_and_moa_openended.jsonl",
        "question": "Select two genes for modulating inflammation in UC.",
        "answer": "\n".join(f"- {item}" for item in GOLD_ITEMS),
        "subject": dataset,
        "topic": "Target Identification",
        "dataset_name": dataset,
        "task_info": {"drugqa_answers": list(GOLD_ITEMS)},
    }


def _write_run(tmp_path: Path, *pids: str, dataset: str = DRUGQA) -> Path:
    (tmp_path / "per-problem").mkdir(parents=True, exist_ok=True)
    for pid in pids:
        grade = {
            "problem_id": pid,
            "dataset_name": dataset,
            "predicted_answer": PARAPHRASE,
            "called_tools": ["mcp__chemistry__search", "other_tool"],
            # A verdict from before the field existed, plus the run metrics a
            # re-grade must not touch.
            "answer_correct": True,
            "answer_score": 1.0,
            "answer_notes": ["DrugQA fell back to the local scorer:"],
            "tool_coverage": 0.5,
            "missing_tools": ["expected_tool"],
            "extra_tools": ["other_tool"],
            "cost_usd": 1.25,
            "duration_ms": 4321,
            "input_tokens": 111,
            "output_tokens": 222,
            "cache_creation_input_tokens": 333,
            "cache_read_input_tokens": 444,
            "error": "",
        }
        (tmp_path / "per-problem" / f"{pid}.json").write_text(
            json.dumps({"problem": _problem_payload(pid, dataset), "grade": grade}),
            encoding="utf-8",
        )
    return tmp_path


def _judge_rules_correct(monkeypatch) -> None:
    monkeypatch.setattr(
        grader,
        "_llm_resolve_judge_config",
        lambda **_: __import__("types").SimpleNamespace(model="stub", api_base="http://stub"),
    )
    monkeypatch.setattr(
        grader,
        "_llm_judge_drugqa_items",
        lambda **_: DrugQAJudgement(correct=False, hit_rate=0.5, matched=1, total=2),
    )


def _grade_of(run: Path, pid: str) -> dict:
    payload = json.loads((run / "per-problem" / f"{pid}.json").read_text(encoding="utf-8"))
    return payload["grade"]


def test_regrade_rewrites_the_verdict(monkeypatch, tmp_path):
    _judge_rules_correct(monkeypatch)
    run = _write_run(tmp_path, "p1")

    changes = report.regrade_run(run, only={"p1"})

    grade = _grade_of(run, "p1")
    assert grade["answer_correct"] is False
    assert grade["answer_score"] == 0.5
    assert grade["answer_verdict_source"] == grader.VERDICT_JUDGE
    assert len(changes) == 1


def test_regrade_leaves_the_run_metrics_alone(monkeypatch, tmp_path):
    """The record of what the run did is not recomputed from less evidence."""
    _judge_rules_correct(monkeypatch)
    run = _write_run(tmp_path, "p1")

    report.regrade_run(run, only={"p1"})

    grade = _grade_of(run, "p1")
    assert grade["tool_coverage"] == 0.5
    assert grade["missing_tools"] == ["expected_tool"]
    assert grade["extra_tools"] == ["other_tool"]
    assert grade["cost_usd"] == 1.25
    assert grade["duration_ms"] == 4321
    assert (grade["input_tokens"], grade["output_tokens"]) == (111, 222)
    assert (
        grade["cache_creation_input_tokens"],
        grade["cache_read_input_tokens"],
    ) == (333, 444)
    # And the payload's other top-level keys survive the rewrite.
    payload = json.loads((run / "per-problem" / "p1.json").read_text(encoding="utf-8"))
    assert payload["problem"]["dataset_name"] == DRUGQA


def test_only_re_grades_what_it_names(monkeypatch, tmp_path):
    """The two problems that failed, not the nineteen that did not."""
    _judge_rules_correct(monkeypatch)
    run = _write_run(tmp_path, "p1", "p2")

    report.regrade_run(run, only={"p1"})

    assert _grade_of(run, "p1")["answer_score"] == 0.5
    assert _grade_of(run, "p2")["answer_score"] == 1.0


def test_a_problem_whose_verdict_does_not_move_is_not_rewritten(monkeypatch, tmp_path):
    """Nothing to say means no write — and no entry in the change list."""
    _judge_rules_correct(monkeypatch)
    run = _write_run(tmp_path, "p1")
    report.regrade_run(run, only={"p1"})
    before = (run / "per-problem" / "p1.json").read_text(encoding="utf-8")

    changes = report.regrade_run(run, only={"p1"})

    assert changes == []
    assert (run / "per-problem" / "p1.json").read_text(encoding="utf-8") == before


def test_a_dry_run_writes_nothing(monkeypatch, tmp_path):
    _judge_rules_correct(monkeypatch)
    run = _write_run(tmp_path, "p1")
    before = (run / "per-problem" / "p1.json").read_text(encoding="utf-8")

    changes = report.regrade_run(run, only={"p1"}, dry_run=True)

    assert len(changes) == 1
    assert (run / "per-problem" / "p1.json").read_text(encoding="utf-8") == before


def test_a_dataset_that_is_not_regradable_is_skipped(monkeypatch, tmp_path):
    """MADD has a judge now and is still skipped, for two separate reasons.

    Its answer metric was redefined from a ground-truth hit rate to a
    requirement-completion score, so a stored run's number came from arithmetic
    that no longer exists — re-deriving it would compare two different
    measurements and call the difference a fix. And the MADD judge path has no
    cache, so a re-grade would re-bill one call per problem.
    """
    monkeypatch.setattr(
        grader,
        "_llm_judge_drugqa_items",
        lambda **_: pytest.fail("the judge was called for a non-regradable dataset"),
    )
    run = _write_run(tmp_path, "m1", dataset="MADD")
    before = (run / "per-problem" / "m1.json").read_text(encoding="utf-8")

    changes = report.regrade_run(run, only={"m1"})

    assert changes == []
    assert (run / "per-problem" / "m1.json").read_text(encoding="utf-8") == before


def test_regrade_recovers_a_verdict_the_local_scorer_got_wrong(monkeypatch, tmp_path):
    """The case the tool exists for, against the score that made it necessary.

    ``preclinical_research_5`` was written as ``answer_correct: True`` off a
    token-overlap match after the judge's reply was cut off. Re-asking the judge
    — now that a truncated reply is salvaged and a bigger budget is tried —
    replaces that with the judge's own count.
    """
    _judge_rules_correct(monkeypatch)
    run = _write_run(tmp_path, "preclinical_research_5")
    assert _grade_of(run, "preclinical_research_5")["answer_correct"] is True

    report.regrade_run(run, only={"preclinical_research_5"})

    grade = _grade_of(run, "preclinical_research_5")
    assert grade["answer_correct"] is False
    assert grade["answer_verdict_source"] == grader.VERDICT_JUDGE
    assert "fell back to the local scorer" not in " ".join(grade["answer_notes"])


def test_regrade_covers_sciagentgym_too(monkeypatch, tmp_path):
    monkeypatch.setattr(
        grader,
        "_llm_judge_correct",
        lambda **_: True,
    )
    monkeypatch.setattr(
        grader,
        "_llm_resolve_judge_config",
        lambda **_: __import__("types").SimpleNamespace(model="stub", api_base="http://stub"),
    )
    run = _write_run(tmp_path, "8", dataset=SCIAGENTGYM)

    report.regrade_run(run, only={"8"})

    assert _grade_of(run, "8")["answer_verdict_source"] == grader.VERDICT_JUDGE


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
