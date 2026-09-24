"""A run the contamination audit marked must say so where its numbers are read.

``tests/audit_contamination.py --mark`` leaves an ``INVALID.json`` in each
affected run directory, and until this test existed nothing read it: the audit
listed the runs, and ``report.py`` went on printing their accuracy as if the
agent had reasoned its way there. The marker is deliberately *not* enforced —
re-grading an archived run on purpose is legitimate — so what these tests pin is
that the reason travels with the number, in the summary and in a diff.

Run with:
    cd "$(git rev-parse --show-toplevel)"
    python -m pytest tests/test_report_invalid.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import report  # noqa: E402


REASON = "ground-truth exposure before run isolation"


def _write_run(
    path: Path,
    *,
    label: str,
    invalid: dict | None = None,
    episodes: tuple[str, ...] = (),
    marked_episodes: tuple[str, ...] = (),
) -> Path:
    """A run directory shaped like a real one, plus whatever markers asked for.

    ``episodes`` are ``scimas-runs/<name>/`` directories, one per problem, and
    ``marked_episodes`` are the subset of them the audit marked — the layout
    ``--mark`` actually produces.
    """
    path.mkdir(parents=True, exist_ok=True)
    (path / "run.json").write_text(
        json.dumps(
            {
                "config": {"dataset_name": "SciAgentGYM", "label": label},
                "n_problems": max(len(episodes), 1),
                "answer_accuracy": 1.0,
                "answer_score_avg": 1.0,
                "tool_coverage_avg": 0.5,
                "total_cost_usd": 0.01,
                "total_duration_ms": 1000,
            }
        ),
        encoding="utf-8",
    )
    if invalid is not None:
        (path / "INVALID.json").write_text(json.dumps(invalid), encoding="utf-8")
    for name in episodes:
        episode = path / "scimas-runs" / name
        episode.mkdir(parents=True, exist_ok=True)
        (episode / "report.json").write_text("{}", encoding="utf-8")
        if name in marked_episodes:
            (episode / "INVALID.json").write_text(
                json.dumps({"reason": REASON, "exposures": ["read", "inherited"]}),
                encoding="utf-8",
            )
    return path


def _summary_output(bundle: report.RunBundle, capsys) -> str:
    report._print_summary(bundle)
    return capsys.readouterr().out


def test_a_marked_episode_reaches_the_batch_summary(tmp_path, capsys):
    """The audit marks the episode directory; this report is read one level up.

    Left alone, a contaminated batch printed its accuracy with nothing to say
    the agent could read the answers.
    """
    path = _write_run(
        tmp_path / "run-a",
        label="old",
        episodes=("20260914-101530", "20260914-101602", "20260914-101634"),
        marked_episodes=("20260914-101530", "20260914-101634"),
    )
    out = _summary_output(report.RunBundle.from_dir(path), capsys)

    assert "2 of 3 episode(s) under scimas-runs/ marked INVALID" in out
    assert "20260914-101530" in out
    assert "the totals below include them" in out
    assert "CONTAMINATION.md" in out
    # The numbers are still shown: the marker annotates, it does not hide.
    assert "answer accuracy : 100.0%" in out


def test_a_job_level_marker_is_reported_too(tmp_path, capsys):
    path = _write_run(
        tmp_path / "run-a2",
        label="old",
        invalid={"reason": REASON, "exposures": ["read", "inherited"]},
    )
    out = _summary_output(report.RunBundle.from_dir(path), capsys)

    assert f"marked INVALID (read, inherited): {REASON}" in out


def test_an_unmarked_run_is_printed_without_a_notice(tmp_path, capsys):
    path = _write_run(
        tmp_path / "run-b", label="new", episodes=("20260914-101530",)
    )
    out = _summary_output(report.RunBundle.from_dir(path), capsys)

    assert "INVALID" not in out


def test_a_long_marked_list_is_truncated(tmp_path, capsys):
    episodes = tuple(f"20260914-1015{i:02d}" for i in range(6))
    path = _write_run(
        tmp_path / "run-d",
        label="old",
        episodes=episodes,
        marked_episodes=episodes,
    )
    out = _summary_output(report.RunBundle.from_dir(path), capsys)

    assert "6 of 6 episode(s)" in out
    assert "+3 more" in out


def test_a_diff_flags_a_contaminated_side(tmp_path, capsys):
    good = report.RunBundle.from_dir(
        _write_run(tmp_path / "new", label="new", episodes=("20260914-101530",))
    )
    bad = report.RunBundle.from_dir(
        _write_run(
            tmp_path / "old",
            label="old",
            episodes=("20260914-101530",),
            marked_episodes=("20260914-101530",),
        )
    )
    report._print_diff(bad, good)
    out = capsys.readouterr().out

    # Named, and named as the contaminated side specifically — a diff between
    # these two measures the boundary, not the change under test.
    assert "!! old: 1 of 1 episode(s) under scimas-runs/ marked INVALID" in out
    assert "!! new" not in out


def test_an_unreadable_marker_does_not_break_the_report(tmp_path, capsys):
    """The report is how archived results are read; a corrupt marker is not a
    reason to refuse to print them."""
    path = _write_run(tmp_path / "run-c", label="c")
    (path / "INVALID.json").write_text("{not json", encoding="utf-8")

    bundle = report.RunBundle.from_dir(path)
    out = _summary_output(bundle, capsys)

    assert bundle.invalid is not None
    assert "could not be read" in out
    assert "answer accuracy : 100.0%" in out


def test_a_directory_without_a_run_json_still_raises(tmp_path):
    """Guard the other direction: the marker reader must not have made a
    non-run directory look readable."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        report.RunBundle.from_dir(empty)
