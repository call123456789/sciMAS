"""Tests for delivering SciAgentGYM question figures to the solver and judge.

``metadata.image_path`` names its files relative to the **upstream repo root**
(``gym/test_images/*.png``), not relative to the dump that names them. The
dumps, however, are normally read from the sciMAS-internal ``dataset/`` copy,
which has no ``gym/`` in it — and ``default_dataset_root()`` returns the sciMAS
root for SciAgentGYM precisely because the dumps live there. So the obvious
``root / path`` resolves to nothing, silently: the figures simply never
appeared, in either the solver prompt or the judge request.

These tests pin the resolution order, the staging copy that puts the figures
where an agent's Read tool can reach them, and the two call sites that hand the
solver a problem text carrying them.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _path in (str(ROOT), str(ROOT / "tests")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import dataset  # noqa: E402
import runner  # noqa: E402
from dataset import SCIAGENTGYM, Problem  # noqa: E402

DUMP_NAME = "refine_merged_multi_questions.json"

# 1x1 transparent PNG, ~67 bytes — the same fixture test_rcb_official_judge uses.
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4"
    "2mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


@pytest.fixture(autouse=True)
def _hermetic_roots(monkeypatch, tmp_path):
    """Keep the candidate roots from reaching the real checkouts.

    ``~/Documents/games/SciAgentGYM-main`` is a real directory on the machine
    this was written on and is the last candidate root, so a test asserting
    "this figure was not found" could otherwise resolve through it.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SCIENT_GYM_ROOT", raising=False)
    monkeypatch.delenv("SCIENTAGENTGYM_ROOT", raising=False)


def _write_png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(TINY_PNG_B64))
    return path


def _write_dump(root: Path, *, entry: dict) -> Path:
    """Write a one-entry SciAgentGYM dump at ``<root>/dataset/<DUMP_NAME>``."""
    import json

    path = root / "dataset" / DUMP_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([entry]), encoding="utf-8")
    return path


def _entry(*, image_paths: list[str], question: str = "What is 2+2?") -> dict:
    return {
        "id": 42,
        "filename": "failed_questions_42.json",
        "question": question,
        "answer": "4",
        "metadata": {
            "subject": "Mathematics",
            "topic": "Arithmetic",
            "image_path": image_paths,
            "solution_steps": [],
            "tool_expected": [],
            "golden_answer": [],
        },
    }


def _load(root: Path) -> list[Problem]:
    return dataset._load_sciagentgym_dataset(root=root, files=(DUMP_NAME,))


def _figure_problem(tmp_path: Path, *, count: int = 1) -> Problem:
    """A problem whose figures are real files in the sibling-checkout layout."""
    names = [f"gym/test_images/f{index}.png" for index in range(count)]
    for name in names:
        _write_png(tmp_path / "SciAgentGYM-main" / name)
    _write_dump(tmp_path / "sciMAS", entry=_entry(image_paths=names))
    return _load(tmp_path / "sciMAS")[0]


def _solver_report(problem_text: str):
    from plan import ExecutionPlan, RunReport

    return RunReport(
        problem=problem_text,
        plan=ExecutionPlan(problem_summary="trivial"),
        runs=[],
        final_answer="4",
        output_dir="/tmp/scimas-runs/fake",
    )


# ---------------------------------------------------------------------------
# 1. Resolution: the dump's paths are relative to the upstream repo root
# ---------------------------------------------------------------------------


def test_resolve_images_finds_the_sibling_upstream_checkout(tmp_path: Path):
    """The real layout: dumps in <sciMAS>/dataset, figures in <games>/SciAgentGYM-main."""
    problem = _figure_problem(tmp_path)

    expected = (tmp_path / "SciAgentGYM-main" / "gym/test_images/f0.png").resolve()

    assert problem.image_paths == [str(expected)]
    assert problem.assets_root == str((tmp_path / "SciAgentGYM-main").resolve())
    assert problem.image_paths_missing == []


def test_resolve_images_honours_scient_gym_root(tmp_path: Path, monkeypatch):
    """$SCIENT_GYM_ROOT wins over the sibling checkout."""
    figure = _write_png(tmp_path / "elsewhere" / "gym/test_images/f0.png")
    _write_dump(
        tmp_path / "sciMAS", entry=_entry(image_paths=["gym/test_images/f0.png"])
    )
    monkeypatch.setenv("SCIENT_GYM_ROOT", str(tmp_path / "elsewhere"))

    problem = _load(tmp_path / "sciMAS")[0]

    assert problem.image_paths == [str(figure.resolve())]


def test_resolve_images_keeps_an_existing_absolute_path(tmp_path: Path):
    """A dump that ships pre-resolved paths needs no candidate root at all."""
    figure = _write_png(tmp_path / "absolute.png")
    _write_dump(tmp_path / "sciMAS", entry=_entry(image_paths=[str(figure)]))

    problem = _load(tmp_path / "sciMAS")[0]

    assert problem.image_paths == [str(figure)]


def test_resolve_images_drops_a_missing_file_and_records_it(tmp_path: Path):
    """A broken path must not be advertised: the solver would spend a tool call
    on it, and the judge would be shown an unreadable block."""
    _write_dump(
        tmp_path / "sciMAS", entry=_entry(image_paths=["gym/test_images/gone.png"])
    )

    problem = _load(tmp_path / "sciMAS")[0]

    assert problem.image_paths == []
    assert problem.image_paths_missing == ["gym/test_images/gone.png"]


def test_entries_without_image_path_are_untouched(tmp_path: Path):
    """48 of the SciAgentGYM entries, every write-in, and the CLI path."""
    _write_dump(tmp_path / "sciMAS", entry=_entry(image_paths=[]))

    problem = _load(tmp_path / "sciMAS")[0]

    assert problem.image_paths == []
    assert problem.image_paths_missing == []
    assert problem.assets_root == ""
    assert dataset.format_problem_for_solver(problem) == problem.question


# ---------------------------------------------------------------------------
# 2. Staging: the copies live inside the run output, and inside the repo
# ---------------------------------------------------------------------------


def test_stage_problem_images_copies_and_is_idempotent(tmp_path: Path):
    problem = _figure_problem(tmp_path, count=2)
    dest = tmp_path / "out" / "figure-assets" / "42"

    staged = dataset.stage_problem_images(problem, dest)

    assert [Path(p).name for p in staged] == ["f0.png", "f1.png"]
    assert all(Path(p).is_file() for p in staged)
    assert all(Path(p).parent == dest for p in staged)
    # The dataset's own copy is left alone.
    assert all(Path(p).is_file() for p in problem.image_paths)
    # Re-running a problem rewrites nothing.
    stamps = {p: Path(p).stat().st_mtime_ns for p in staged}
    assert dataset.stage_problem_images(problem, dest) == staged
    assert {p: Path(p).stat().st_mtime_ns for p in staged} == stamps


def test_stage_problem_images_returns_nothing_without_figures(tmp_path: Path):
    _write_dump(tmp_path / "sciMAS", entry=_entry(image_paths=[]))
    problem = _load(tmp_path / "sciMAS")[0]

    assert dataset.stage_problem_images(problem, tmp_path / "out") == []


def test_stage_problem_images_keeps_both_on_a_name_collision(tmp_path: Path):
    """Two figures from different roots can share a basename; both must survive."""
    first = _write_png(tmp_path / "a" / "fig.png")
    second = _write_png(tmp_path / "b" / "fig.png")
    problem = Problem(
        id=1,
        filename="t.json",
        question="q",
        answer="4",
        subject="s",
        topic="t",
        image_paths=[str(first), str(second)],
    )

    staged = dataset.stage_problem_images(problem, tmp_path / "out")

    assert len(staged) == 2
    assert len(set(staged)) == 2
    assert all(Path(p).is_file() for p in staged)


# ---------------------------------------------------------------------------
# 3. The solver sees the figures, the question string does not change
# ---------------------------------------------------------------------------


def test_solver_text_is_byte_identical_without_figures(tmp_path: Path):
    _write_dump(tmp_path / "sciMAS", entry=_entry(image_paths=[]))
    problem = _load(tmp_path / "sciMAS")[0]

    assert dataset.format_problem_for_solver(problem) == problem.question


def test_solver_text_names_the_figures_and_the_read_tool(tmp_path: Path):
    problem = _figure_problem(tmp_path)
    problem.image_paths_local = dataset.stage_problem_images(
        problem, tmp_path / "out" / "42"
    )

    text = dataset.format_problem_for_solver(problem)

    assert text.startswith(problem.question)
    assert "Question figures" in text
    assert f"- {problem.image_paths_local[0]}" in text
    assert "Read tool" in text
    # The copies are what the solver is pointed at, not the dataset's paths.
    assert problem.image_paths[0] not in text


def test_solver_text_prefers_the_staged_copies_and_leaves_the_question_alone(
    tmp_path: Path,
):
    problem = _figure_problem(tmp_path)
    original_question = problem.question
    staged = dataset.stage_problem_images(problem, tmp_path / "out" / "42")
    problem.image_paths_local = staged

    first = dataset.format_problem_for_solver(problem)
    second = dataset.format_problem_for_solver(problem)

    assert first == second
    # The grader feeds `problem.question` to the judge; figure paths there
    # would be redundant (the images travel as vision blocks) and would change
    # the judged question.
    assert problem.question == original_question
    assert staged[0] not in problem.question


# ---------------------------------------------------------------------------
# 4. Call sites: both the batch runner and the dashboard forward the section
# ---------------------------------------------------------------------------


def test_runner_call_site_forwards_the_figure_section(tmp_path: Path):
    problem = _figure_problem(tmp_path)
    seen: dict = {}

    class _Orchestrator:
        def run(self, *, problem, roles=None):
            seen["problem"] = problem
            return _solver_report(problem)

    grade, _out_dir = runner._run_single(
        problem,
        cfg=runner.RunConfig(dataset_name=SCIAGENTGYM),
        orchestrator_factory=lambda _runner: _Orchestrator(),
        asset_root=tmp_path / "figure-assets",
    )

    staged = problem.image_paths_local
    assert staged and Path(staged[0]).is_file()
    assert Path(staged[0]).parent == (
        tmp_path / "figure-assets" / runner._problem_file_stem(problem)
    ).resolve()
    assert staged[0] in seen["problem"]
    assert problem.question == "What is 2+2?"
    assert grade.problem_id == problem.id


def test_runner_call_site_without_asset_root_still_names_the_figures(tmp_path: Path):
    """A direct ``_run_single`` call skips the copies but must not skip the
    figures entirely."""
    problem = _figure_problem(tmp_path)
    seen: dict = {}

    class _Orchestrator:
        def run(self, *, problem, roles=None):
            seen["problem"] = problem
            return _solver_report(problem)

    runner._run_single(
        problem,
        cfg=runner.RunConfig(dataset_name=SCIAGENTGYM),
        orchestrator_factory=lambda _runner: _Orchestrator(),
    )

    assert problem.image_paths_local == []
    assert problem.image_paths[0] in seen["problem"]


def test_dashboard_call_site_stages_and_forwards_the_figure_section(
    tmp_path: Path, monkeypatch
):
    """End-to-end through ``_run_job``: the figure copies land under the job's
    output root, the solver text carries them, and the persisted record keeps
    the bare question plus the resolved paths."""
    import importlib
    import json

    dashboard = importlib.import_module("web_dashboard")
    problem = _figure_problem(tmp_path)
    seen: dict = {}

    class _FakeOrchestrator:
        def __init__(self, *args, **kwargs):
            self.runner = kwargs.get("runner")

        def run(self, *, problem, roles=None, auto_synthesize=True):
            seen["problem"] = problem
            return _solver_report(problem)

    monkeypatch.setattr(dashboard, "SciMASOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(
        dashboard,
        "_load_filtered_problems",
        lambda config: ([problem], tmp_path, SCIAGENTGYM),
    )

    job = dashboard.DashboardJob(
        id="job-figures",
        config={
            "dataset": SCIAGENTGYM,
            "limit": 1,
            "ids": "42",
            "output_root": str(tmp_path / "out"),
        },
    )
    dashboard.JOBS["job-figures"] = job
    try:
        dashboard._run_job(job)
    finally:
        dashboard.JOBS.pop("job-figures", None)

    job_dirs = list((tmp_path / "out").glob("*/figure-assets/42/*.png"))
    assert job_dirs, "figures should be staged under the job output root"
    assert problem.image_paths_local
    assert str(problem.image_paths_local[0]) in seen["problem"]
    assert problem.question == "What is 2+2?"

    payload = json.loads(
        next((tmp_path / "out").glob("*/per-problem/42.json")).read_text(
            encoding="utf-8"
        )
    )
    assert payload["problem"]["question"] == "What is 2+2?"
    assert payload["problem"]["image_paths_local"] == problem.image_paths_local
