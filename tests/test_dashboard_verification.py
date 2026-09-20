"""Tests for the verification flow wired into web_dashboard.py.

These tests focus on the helpers in web_dashboard.py that wrap
``tests.grader.grade_problem`` and persist the per-problem result file.
The full Flask request flow is intentionally out of scope — we exercise
the new helpers directly with a minimal stub runner + report so the
tests do not depend on the Claude CLI or MCP servers.
"""

from __future__ import annotations

import importlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


# Import dashboard lazily inside helpers — module name has no underscore.
def _import_dashboard():
    sys.path.insert(0, str(ROOT))
    if "web_dashboard" in sys.modules:
        return sys.modules["web_dashboard"]
    return importlib.import_module("web_dashboard")


# ---------------------------------------------------------------------------
# Stub fixtures
# ---------------------------------------------------------------------------


class _StubRunner:
    """Mimics tests/runner._StreamJsonRunner just enough to feed the dashboard."""

    def __init__(self, tool_names: list[str] | None = None) -> None:
        self._tool_calls = [
            {"tool": f"mcp__srv__{name}", "arguments": {}}
            for name in (tool_names or [])
        ]
        self.permission_mode = ""
        self.dangerously_skip_permissions = False
        self.binary = "claude"
        self.timeout = 600.0


def _make_problem(tmp_path: Path) -> Any:
    """Build a SciAgentGYM-style Problem for tests that need a real object."""
    from dataset import Problem

    return Problem(
        id="42",
        filename="failed_questions_42.json",
        question="What is 2+2?",
        answer="4",
        subject="Mathematics",
        topic="Arithmetic",
        image_paths=[],
        solution_steps=[],
        expected_tools=[],
        golden_calls=[],
        original_question_id="42",
        usage_tool_protocol=[],
        source_dataset="refine_merged_single_questions.json",
        dataset_name="SciAgentGYM",
        assets_root=str(tmp_path),
    )


def _make_report(final_answer: str = "The answer is 4."):
    """Build a minimal RunReport so the helpers don't touch the orchestrator."""
    from plan import ExecutionPlan, RunReport

    return RunReport(
        problem="What is 2+2?",
        plan=ExecutionPlan(problem_summary="trivial arithmetic"),
        runs=[],
        final_answer=final_answer,
        output_dir=str((Path("/tmp") / "scimas-runs" / "fake").resolve()),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_persist_grade_writes_expected_payload(tmp_path: Path) -> None:
    """``_persist_grade`` writes problem + grade + run_dir to per-problem/<id>.json."""
    dashboard = _import_dashboard()
    problem = _make_problem(tmp_path)
    grade_dict = {"problem_id": "42", "answer_score": 1.0, "answer_correct": True}
    # Build a real GradingResult so we exercise to_dict.
    from grader import GradingResult

    grade = GradingResult(
        problem_id="42",
        dataset_name="SciAgentGYM",
        subject="Mathematics",
        topic="Arithmetic",
        expected_answer="4",
        predicted_answer="4",
        answer_correct=True,
        answer_score=1.0,
    )

    out_file = dashboard._persist_grade(
        problem,
        grade,
        scimas_run_dir="/tmp/scimas-runs/fake",
        output_root=tmp_path,
    )

    assert out_file.exists()
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["problem"]["id"] == "42"
    assert payload["problem"]["answer"] == "4"
    assert payload["grade"]["answer_score"] == 1.0
    assert payload["grade"]["answer_correct"] is True
    assert payload["scimas_run_dir"] == "/tmp/scimas-runs/fake"
    # File stem is sanitized — see _problem_file_stem.
    assert out_file.name == "42.json"
    assert out_file.parent == tmp_path / "per-problem"


def test_grade_one_problem_persists_with_captured_tools(tmp_path: Path, monkeypatch) -> None:
    """``_grade_one_problem`` should record mcp__ tools the runner saw
    and write the per-problem.json under ``<output_root>/per-problem/``."""
    dashboard = _import_dashboard()
    problem = _make_problem(tmp_path)
    report = _make_report(final_answer="The answer is 4.")
    runner = _StubRunner(tool_names=["alpha", "beta"])

    # Build a real DashboardJob — minimal — to satisfy _smdd_run_config_from_job.
    job = dashboard.DashboardJob(
        id="job-abc",
        config={
            "dataset": "SciAgentGYM",
            "smdd_official_eval": False,
        },
    )

    grade = dashboard._grade_one_problem(
        job,
        problem,
        report,
        runner,
        output_root=tmp_path,
    )

    out_file = tmp_path / "per-problem" / "42.json"
    assert out_file.exists()
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    # Two MCP tools captured, surfaced as ``tool=""`` short names via
    # ToolCall.from_full's mcp__<server>__<tool> split.
    assert payload["grade"]["called_tools"] == ["alpha", "beta"]
    assert payload["grade"]["answer_correct"] is True
    # The GradingResult we returned is the same object persisted.
    assert grade.answer_correct is True
    assert grade.answer_score == 1.0


def test_grade_one_problem_runs_smdd_when_flagged(tmp_path: Path, monkeypatch) -> None:
    """When the job has ``smdd_official_eval=True``, the SMDD official-eval
    hook should be invoked for SMDDBench problems."""
    dashboard = _import_dashboard()

    from dataset import SMDD_BENCH

    problem = _make_problem(tmp_path)
    problem.dataset_name = SMDD_BENCH
    problem.task_info = {"smdd_task_dir": str(tmp_path / "fake_task_dir")}

    calls: list[dict[str, Any]] = []

    def fake_maybe_run(problem, *, final_answer, cfg, smdd_eval_root):
        calls.append(
            {
                "final_answer": final_answer,
                "smdd_eval_root": str(smdd_eval_root),
                "smdd_official_eval": cfg.smdd_official_eval,
                "smdd_docker_image": cfg.smdd_docker_image,
            }
        )
        # Mimic the side effect of writing a result.json that
        # grade_smdd_bench_answer later reads.
        result = {"status": "passed", "steps": [{"status": "passed"}]}
        problem.task_info["smdd_official_result"] = result

    monkeypatch.setattr(dashboard, "_maybe_run_smdd_official_eval", fake_maybe_run)

    job = dashboard.DashboardJob(
        id="job-smdd",
        config={
            "dataset": SMDD_BENCH,
            "smdd_official_eval": True,
            "smdd_docker_image": "smdd-evals",
            "smdd_eval_timeout": 60.0,
            "smdd_gpu_id": None,
        },
    )
    runner = _StubRunner()

    grade = dashboard._grade_one_problem(
        job,
        problem,
        _make_report(final_answer="smiles content"),
        runner,
        output_root=tmp_path,
    )

    assert calls, "SMDD official-eval should have been invoked"
    assert calls[0]["smdd_official_eval"] is True
    assert calls[0]["smdd_docker_image"] == "smdd-evals"
    assert calls[0]["smdd_eval_root"].endswith("smdd-official")
    assert grade.answer_correct is True  # status=passed ⇒ score=1.0


def test_build_failure_grade_zero_score(tmp_path: Path) -> None:
    """When the orchestrator itself throws, the failure-grade still has
    a sane shape with answer_score=0.0 and the orchestrator error attached."""
    dashboard = _import_dashboard()
    problem = _make_problem(tmp_path)
    grade = dashboard._build_failure_grade(
        problem, error="RuntimeError: claude CLI crashed"
    )
    assert grade.problem_id == "42"
    assert grade.answer_correct is False
    assert grade.answer_score == 0.0
    assert grade.predicted_answer == ""
    assert grade.error.startswith("RuntimeError")


def test_problem_meta_for_event_includes_dataset_and_expected() -> None:
    """The event payload should expose dataset + expected_answer so the
    frontend can render badges without a second request."""
    dashboard = _import_dashboard()
    problem = _make_problem(Path("/tmp"))
    meta = dashboard._problem_meta_for_event(problem)
    assert meta["id"] == "42"
    assert meta["dataset"] == "SciAgentGYM"
    assert meta["subject"] == "Mathematics"
    assert meta["expected_answer"] == "4"
    assert meta["expected_tools"] == []


def test_event_payload_includes_grade_field(tmp_path: Path, monkeypatch) -> None:
    """End-to-end: run ``_run_job`` against a mocked orchestrator and
    inspect the SSE event payload to confirm ``grade`` is populated."""
    dashboard = _import_dashboard()

    # Patch the orchestrator before _run_job runs.
    import plan

    class _FakeOrchestrator:
        def __init__(self, *args, **kwargs):
            self.runner = kwargs.get("runner") or _StubRunner()
            self.output_dir = "/tmp/scimas-runs/fake"

        def run(self, *, problem, roles=None, auto_synthesize=True):
            return plan.RunReport(
                problem=problem,
                plan=plan.ExecutionPlan(problem_summary="trivial"),
                runs=[],
                final_answer="The answer is 4.",
                output_dir="/tmp/scimas-runs/fake",
            )

    monkeypatch.setattr(dashboard, "SciMASOrchestrator", _FakeOrchestrator)

    job = dashboard.DashboardJob(
        id="job-stream",
        config={
            "dataset": "SciAgentGYM",
            "root": "",
            "limit": 1,
            "ids": "42",
            "output_root": str(tmp_path),
        },
    )
    dashboard.JOBS["job-stream"] = job
    try:
        # Avoid touching the real dataset loader — fake the filter result.
        monkeypatch.setattr(
            dashboard,
            "_load_filtered_problems",
            lambda config: ([_make_problem(tmp_path)], Path("/tmp"), "SciAgentGYM"),
        )
        # Don't actually run the orchestrator's runner — we patched the class.
        # Run the job synchronously.
        dashboard._run_job(job)
        finished_events = [
            evt for evt in job.events if evt.get("event") == "problem_finished"
        ]
        assert finished_events, "expected a problem_finished event"
        result = finished_events[0]["result"]
        assert "grade" in result, "result must include the grader output"
        assert "problem_meta" in result, "result must include problem_meta"
        assert result["problem_meta"]["dataset"] == "SciAgentGYM"
        assert result["grade"]["answer_score"] == 1.0
        assert result["status"] == "completed"
        # _run_job nests one subdir under output_root (timestamp-job).
        per_problem_files = list(tmp_path.glob("*/per-problem/42.json"))
        assert per_problem_files, "expected per-problem/42.json somewhere under tmp_path"
    finally:
        dashboard.JOBS.pop("job-stream", None)


def test_runner_is_swapped_to_stream_json_in_run_job(tmp_path: Path, monkeypatch) -> None:
    """``_run_job`` should replace ``orchestrator.runner`` with a
    ``_StreamJsonRunner`` instance so tool calls get captured."""
    dashboard = _import_dashboard()

    seen_runners: list[Any] = []

    class _FakeStreamRunner:
        def __init__(self, *args, **kwargs):
            seen_runners.append("stream")
            self._tool_calls = []

    class _FakeOrchestrator:
        def __init__(self, *args, **kwargs):
            # Note: this constructor default is what gets replaced.
            self.runner = "sentinel-not-runner"

        def run(self, *, problem, roles=None, auto_synthesize=True):
            seen_runners.append(self.runner)
            from plan import ExecutionPlan, RunReport

            return RunReport(
                problem=problem,
                plan=ExecutionPlan(problem_summary="trivial"),
                runs=[],
                final_answer="42",
                output_dir="/tmp/scimas-runs/fake",
            )

    monkeypatch.setattr(dashboard, "_StreamJsonRunner", _FakeStreamRunner)
    monkeypatch.setattr(dashboard, "SciMASOrchestrator", _FakeOrchestrator)

    job = dashboard.DashboardJob(
        id="job-swap",
        config={
            "dataset": "SciAgentGYM",
            "root": "",
            "limit": 1,
            "ids": "42",
            "output_root": str(tmp_path),
        },
    )
    monkeypatch.setattr(
        dashboard,
        "_load_filtered_problems",
        lambda config: ([_make_problem(tmp_path)], Path("/tmp"), "SciAgentGYM"),
    )
    dashboard._run_job(job)

    # The stream-json runner was constructed AND attached before run().
    assert "stream" in seen_runners
    assert seen_runners[-1] in seen_runners[:-1] or any(
        isinstance(r, _FakeStreamRunner) for r in seen_runners[1:]
    ), "orchestrator.runner must be replaced with the stream-json runner"


# ---------------------------------------------------------------------------
# MCP enablement:  blank field means "default", not "disabled"
# ---------------------------------------------------------------------------


def _mcp_config_for(config: dict) -> Any:
    dashboard = _import_dashboard()
    return dashboard._mcp_config_for_job(dashboard.DashboardJob(id="j", config=config))


def test_blank_mcp_field_falls_through_to_autodetect() -> None:
    """An empty path field must NOT disable MCP.

    Regression: the form posts '' when the field is left blank, and
    ``ClaudeRunner`` reads '' as the explicit ``--no-mcp`` sentinel, so
    every dashboard run silently ran without a single MCP server attached.
    """
    assert _mcp_config_for({"mcp_config": ""}) is None
    assert _mcp_config_for({"mcp_config": "   "}) is None
    assert _mcp_config_for({}) is None


def test_no_mcp_checkbox_still_opts_out() -> None:
    """The checkbox remains the one way to genuinely disable MCP."""
    assert _mcp_config_for({"no_mcp": True, "mcp_config": ""}) == ""
    # Checkbox wins even if a path was typed.
    assert _mcp_config_for({"no_mcp": True, "mcp_config": "/tmp/x.json"}) == ""


def test_explicit_mcp_path_is_passed_through() -> None:
    assert _mcp_config_for({"mcp_config": "/tmp/custom.json"}) == "/tmp/custom.json"
    assert _mcp_config_for({"mcp_config": "  /tmp/custom.json  "}) == "/tmp/custom.json"


def test_run_job_forwards_mcp_config_to_runner(tmp_path: Path, monkeypatch) -> None:
    """``_run_job`` hands the resolved value to the stream-json runner.

    Both the runner and the orchestrator used to receive the raw config
    dict entry, so this asserts the plumbing end-to-end rather than only
    the helper in isolation.
    """
    dashboard = _import_dashboard()
    seen_kwargs: list[dict] = []

    class _FakeStreamRunner:
        def __init__(self, *args, **kwargs):
            seen_kwargs.append(kwargs)
            self._tool_calls = []

    class _FakeOrchestrator:
        def __init__(self, *args, **kwargs):
            seen_kwargs.append(kwargs)
            self.runner = "sentinel-not-runner"

        def run(self, *, problem, roles=None, auto_synthesize=True):
            from plan import ExecutionPlan, RunReport

            return RunReport(
                problem=problem,
                plan=ExecutionPlan(problem_summary="trivial"),
                runs=[],
                final_answer="42",
                output_dir="/tmp/scimas-runs/fake",
            )

    monkeypatch.setattr(dashboard, "_StreamJsonRunner", _FakeStreamRunner)
    monkeypatch.setattr(dashboard, "SciMASOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(
        dashboard,
        "_load_filtered_problems",
        lambda config: ([_make_problem(tmp_path)], Path("/tmp"), "SciAgentGYM"),
    )

    def _run_with(config: dict) -> list[dict]:
        seen_kwargs.clear()
        job = dashboard.DashboardJob(
            id="job-mcp",
            config={
                "dataset": "SciAgentGYM",
                "root": "",
                "limit": 1,
                "ids": "42",
                "output_root": str(tmp_path),
                **config,
            },
        )
        dashboard._run_job(job)
        return [k["mcp_config_path"] for k in seen_kwargs if "mcp_config_path" in k]

    # Blank field → None on both the runner and the orchestrator, which is
    # what makes ClaudeRunner auto-detect config/mcp.json.
    assert _run_with({"mcp_config": ""}) == [None, None]
    # Checkbox → the opt-out sentinel.
    assert _run_with({"no_mcp": True}) == ["", ""]
    # Typed path → that path.
    assert _run_with({"mcp_config": "/tmp/custom.json"}) == [
        "/tmp/custom.json",
        "/tmp/custom.json",
    ]


# ---------------------------------------------------------------------------
# "当场手写" (ad-hoc write-in) pseudo-dataset
# ---------------------------------------------------------------------------


def test_write_in_problem_from_config() -> None:
    """A non-blank write-in question becomes exactly one synthetic Problem."""
    dashboard = _import_dashboard()
    write_in = dashboard.WRITE_IN_DATASET

    problems, root, dataset_name = dashboard._load_filtered_problems(
        {"dataset": write_in, "write_in_question": "  What is 2+2?  "}
    )

    assert len(problems) == 1
    problem = problems[0]
    assert problem.question == "What is 2+2?"  # stripped
    assert problem.id == dashboard.WRITE_IN_PROBLEM_ID
    assert problem.answer == "", "a write-in problem must have no gold answer"
    assert problem.expected_tools == []
    assert problem.dataset_name == write_in
    assert dataset_name == write_in
    # Non-empty so the Problems row reads "write-in · 当场手写 / ad-hoc"
    # instead of leaving a dangling slash.
    assert problem.subject and problem.topic


def test_write_in_empty_question_yields_no_problems() -> None:
    """A blank question is reported as zero problems, never as an exception.

    The frontend calls both loaders the moment the option is selected, before
    anything has been typed, so raising here would surface as an unhandled
    error in the UI.
    """
    dashboard = _import_dashboard()
    write_in = dashboard.WRITE_IN_DATASET

    for config in (
        {"dataset": write_in, "write_in_question": ""},
        {"dataset": write_in, "write_in_question": "   "},
        {"dataset": write_in},  # key absent entirely
    ):
        problems, _, dataset_name = dashboard._load_filtered_problems(config)
        assert problems == [], config
        assert dataset_name == write_in

        subjects, topics, count, _, dataset_name = dashboard._load_filter_options(config)
        assert (subjects, topics, count) == ([], [], 0), config
        assert dataset_name == write_in


def test_write_in_filter_options_count() -> None:
    """With a question typed, /api/filter-options reports count 1 and cannot 400.

    api_filter_options turns any exception into a 400, and the frontend throws
    on a non-OK response.
    """
    dashboard = _import_dashboard()

    subjects, topics, count, _, dataset_name = dashboard._load_filter_options(
        {"dataset": dashboard.WRITE_IN_DATASET, "write_in_question": "hi"}
    )

    assert count == 1
    assert subjects == [] and topics == []
    assert dataset_name == dashboard.WRITE_IN_DATASET


def test_write_in_never_touches_the_dataset_loader(monkeypatch) -> None:
    """Regression guard for the load_dataset() fallthrough.

    dataset.load_dataset() has no branch for an unregistered name and falls
    through to the SciAgentGYM loader, and default_dataset_root() would hand
    back a bogus SciAgentGYM path. The short-circuit must happen before any of
    that, so all three entry points are booby-trapped here.
    """

    def _boom(*args, **kwargs):
        raise AssertionError("the dataset loader was reached for a write-in run")

    dashboard = _import_dashboard()
    monkeypatch.setattr(dashboard, "load_dataset", _boom)
    monkeypatch.setattr(dashboard, "normalize_dataset_name", _boom)
    monkeypatch.setattr(dashboard, "default_dataset_root", _boom)

    config = {"dataset": dashboard.WRITE_IN_DATASET, "write_in_question": "anything"}
    dashboard._load_filtered_problems(config)  # must not raise
    dashboard._load_filter_options(config)  # must not raise


def test_write_in_grade_skips_grader_and_persists(tmp_path: Path, monkeypatch) -> None:
    """The ad-hoc grade never calls the grader and keeps a numeric score."""
    dashboard = _import_dashboard()

    def _boom(*args, **kwargs):
        raise AssertionError("grade_problem must not be called for a write-in run")

    monkeypatch.setattr(dashboard, "grade_problem", _boom)

    problem = dashboard._write_in_problem(
        {"dataset": dashboard.WRITE_IN_DATASET, "write_in_question": "Why is the sky blue?"}
    )
    report = _make_report(final_answer="Rayleigh scattering.")
    grade = dashboard._write_in_grade(problem, report, output_root=tmp_path)

    # Must stay a float: tests/report.py formats this numerically and would
    # raise TypeError on None.
    assert isinstance(grade.answer_score, float)
    assert grade.answer_score == 0.0
    assert grade.predicted_answer == "Rayleigh scattering."
    assert grade.answer_notes == [dashboard.WRITE_IN_GRADE_NOTE]
    assert grade.error == ""
    assert grade.dataset_name == dashboard.WRITE_IN_DATASET

    persisted = tmp_path / "per-problem" / "write-in.json"
    assert persisted.exists(), "the run must still be persisted to history"
    payload = json.loads(persisted.read_text(encoding="utf-8"))
    assert payload["grade"]["dataset_name"] == dashboard.WRITE_IN_DATASET
    assert payload["grade"]["answer_score"] == 0.0


def test_run_job_write_in_end_to_end(tmp_path: Path, monkeypatch) -> None:
    """Full ``_run_job`` pass over the real short-circuit (not monkeypatched)."""
    dashboard = _import_dashboard()

    import plan

    class _FakeOrchestrator:
        def __init__(self, *args, **kwargs):
            self.runner = kwargs.get("runner") or _StubRunner()

        def run(self, *, problem, roles=None, auto_synthesize=True):
            return plan.RunReport(
                problem=problem,
                plan=plan.ExecutionPlan(problem_summary="trivial"),
                runs=[],
                final_answer="The answer is 4.",
                output_dir="/tmp/scimas-runs/fake",
            )

    monkeypatch.setattr(dashboard, "SciMASOrchestrator", _FakeOrchestrator)

    job = dashboard.DashboardJob(
        id="job-write-in",
        config={
            "dataset": dashboard.WRITE_IN_DATASET,
            "write_in_question": "Why is the sky blue?",
            "output_root": str(tmp_path),
        },
    )
    dashboard.JOBS["job-write-in"] = job
    try:
        dashboard._run_job(job)

        finished = [evt for evt in job.events if evt.get("event") == "problem_finished"]
        assert len(finished) == 1, "a write-in run is always exactly one problem"
        result = finished[0]["result"]
        assert result["status"] == "completed"
        assert result["grade"]["answer_score"] == 0.0
        assert result["grade"]["answer_notes"] == [dashboard.WRITE_IN_GRADE_NOTE]
        assert result["problem_meta"]["dataset"] == dashboard.WRITE_IN_DATASET
        assert result["problem_meta"]["expected_answer"] == ""
        assert result["final_answer"] == "The answer is 4."

        persisted = list(tmp_path.glob("*/per-problem/write-in.json"))
        assert persisted, "expected per-problem/write-in.json under the run dir"
        # set_judge_cache_dir only records a path string; it never creates the
        # directory, so its absence is the evidence that no judge ever ran.
        assert not list(tmp_path.glob("*/judge_cache"))
    finally:
        dashboard.JOBS.pop("job-write-in", None)


def test_run_job_write_in_empty_question_fails_with_specific_error(
    tmp_path: Path, monkeypatch
) -> None:
    """A blank write-in fails with its own message and writes nothing to disk."""
    dashboard = _import_dashboard()

    def _boom(*args, **kwargs):
        raise AssertionError("no orchestrator may be built for a blank write-in")

    monkeypatch.setattr(dashboard, "SciMASOrchestrator", _boom)

    job = dashboard.DashboardJob(
        id="job-write-in-blank",
        config={
            "dataset": dashboard.WRITE_IN_DATASET,
            "write_in_question": "   ",
            "output_root": str(tmp_path),
        },
    )
    dashboard.JOBS["job-write-in-blank"] = job
    try:
        dashboard._run_job(job)

        assert job.status == "failed"
        assert dashboard.WRITE_IN_LABEL in job.error
        assert "No problems match" not in job.error
        # The empty-problem guard precedes the output_root mkdir.
        assert list(tmp_path.iterdir()) == [], "a failed write-in must leave no run dir"
    finally:
        dashboard.JOBS.pop("job-write-in-blank", None)


def test_write_in_markup_present_in_index_html() -> None:
    """Cheap guard against string-surgery regressions in the embedded HTML."""
    dashboard = _import_dashboard()

    for marker in ('id="write_in_question"', 'id="writeInSection"', 'id="datasetFilters"'):
        assert marker in dashboard.INDEX_HTML, marker
    # The sentinel is injected by the / route, never hard-coded in the page.
    assert "@@WRITE_IN@@" in dashboard.INDEX_HTML
    assert dashboard.WRITE_IN_DATASET not in dashboard.INDEX_HTML


def test_graph_legend_and_palette_markup() -> None:
    """Cheap guards on the graph markup/CSS that need no JS engine."""
    dashboard = _import_dashboard()
    html = dashboard.INDEX_HTML

    assert 'id="graphLegend"' in html
    assert "function stageGroupTrace(" in html
    assert "function bindRuntimeNode(" in html
    assert "function buildSkeleton(" in html
    # Four states only. The running state is blue now, not the old amber.
    # (#fff7ed survives in .pill-run, a log badge unrelated to graph nodes.)
    assert "--node-run: #eff6ff" in html, "running state should be blue"
    assert "--node-run: #fff7ed" not in html, "the old amber running fill should be gone"
    assert "#f59e0b" not in html, "the old amber running stroke should be gone"
    for var in ("--node-stroke", "--node-run-stroke", "--node-done-stroke", "--node-fail-stroke"):
        assert var in html, var


# ---------------------------------------------------------------------------
# Behavioural tests: run the dashboard's real embedded JS in node.
#
# The skeleton derivation and the runtime binding are JS, so testing them any
# other way would mean testing a reimplementation. Skipped where node is absent
# (the server has no node), which is why the markup guards above are separate.
# ---------------------------------------------------------------------------

_NODE = shutil.which("node")

_JS_HARNESS = r"""
const fs = require("fs");

// Minimal DOM shim: only what the graph code touches. textContent/innerHTML are
// real accessors so assigning them clears children the way the DOM does.
function makeEl(tag) {
  const el = {
    tagName: tag, _attrs: {}, _children: [], _text: "",
    className: "", value: "", checked: false, disabled: false, title: "",
    style: {}, clientWidth: 0, clientHeight: 0,
    get children() { return this._children; },
    get firstChild() { return this._children[0] || null; },
    appendChild(c) { this._children.push(c); c.parent = this; return c; },
    insertBefore(c, ref) {
      const i = this._children.indexOf(ref);
      if (i === -1) throw new Error("insertBefore: ref is not a child");
      this._children.splice(i, 0, c); c.parent = this; return c;
    },
    removeChild(c) { const i = this._children.indexOf(c); if (i !== -1) this._children.splice(i, 1); return c; },
    setAttribute(k, v) { this._attrs[k] = String(v); },
    getAttribute(k) { return this._attrs[k]; },
    addEventListener() {}, prepend() {}, focus() {}, close() {},
  };
  Object.defineProperty(el, "textContent", {
    get() { return this._text; },
    set(v) { this._text = String(v); this._children.length = 0; },
  });
  Object.defineProperty(el, "innerHTML", {
    get() { return this._text; },
    set(v) { this._text = String(v); this._children.length = 0; },
  });
  return el;
}

const elements = {};
const document = {
  createElement: makeEl,
  createElementNS: (ns, tag) => makeEl(tag),
  createTextNode: (t) => { const e = makeEl("#text"); e.textContent = t; return e; },
  getElementById(id) { if (!elements[id]) elements[id] = makeEl("div"); return elements[id]; },
};

const env = {
  document,
  window: { addEventListener() {} },
  // Never resolves, which keeps the bootstrap chain (loadDatasets ->
  // loadFilterOptions -> previewProblems) inert while the graph code boots.
  fetch: () => new Promise(() => {}),
  EventSource: function () {},
  console,
};

const script = fs.readFileSync(process.argv[2], "utf8");
const api = new Function(
  ...Object.keys(env),
  script + "\nreturn { state, resetGraph, ingestMasEvent, handleEvent, stageGroupTrace,"
         + " buildSkeleton, bindRuntimeNode, reconcilePlan, renderGraph, renderLegend,"
         + " fillFor, strokeFor, $ };"
)(...Object.values(env));

const results = [];
function check(name, cond, detail) {
  results.push({ name, ok: !!cond, detail: cond ? "" : String(detail === undefined ? "" : detail) });
}
function snap() {
  return Array.from(api.state.nodes.values()).map(n => ({
    id: n.id, role: n.role, status: n.status, stage: n.stage,
    instruction: n.instruction, synthetic: !!n.synthetic, boundTo: n.boundTo,
  })).sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}
function byId(id) { return api.state.nodes.get(id); }
function rects() {
  const out = [];
  (function walk(node) {
    if (node.tagName === "rect") {
      out.push({ x: +node.getAttribute("x"), w: +node.getAttribute("width"),
                 fill: node.getAttribute("fill"), stroke: node.getAttribute("stroke") });
    }
    for (const c of node.children) walk(c);
  })(api.$("graph"));
  return out;
}
function stageWidths(trace) { return api.stageGroupTrace(trace).map(g => g.length); }
function stageRoles(trace) { return api.stageGroupTrace(trace).map(g => g.map(r => r.role)); }

const A = (role, instruction) => ({ event: "agent_call", line: 1, role, instruction });

// ---- 1. stage grouping, including the order regression ---------------------
{
  const seq = stageRoles([A("a", "i"), A("b", "i"), A("c", "i")]);
  check("grouping: three plain awaits -> three stages",
        JSON.stringify(seq) === JSON.stringify([["a"], ["b"], ["c"]]), JSON.stringify(seq));

  const parFirst = stageRoles([A("a", "i"), A("b", "i"),
                               { event: "parallel_call", line: 1, width: 2 }, A("c", "i")]);
  check("grouping: parallel first, then sequential",
        JSON.stringify(parFirst) === JSON.stringify([["a", "b"], ["c"]]), JSON.stringify(parFirst));

  // The regression: flushing `pending` wholesale at the end emits the parallel
  // group BEFORE the sequential agent that precedes it, i.e. [[b,c],[a]].
  const seqFirst = stageRoles([A("a", "i"), A("b", "i"), A("c", "i"),
                               { event: "parallel_call", line: 1, width: 2 }]);
  check("grouping: sequential THEN parallel keeps source order",
        JSON.stringify(seqFirst) === JSON.stringify([["a"], ["b", "c"]]), JSON.stringify(seqFirst));

  const loopThenPar = stageRoles([{ event: "for", line: 1, range: 2 }, A("a", "i"),
                                  A("b", "i"), A("c", "i"),
                                  { event: "parallel_call", line: 1, width: 2 }]);
  check("grouping: for-body then parallel keeps source order",
        JSON.stringify(loopThenPar) === JSON.stringify([["a"], ["b", "c"]]), JSON.stringify(loopThenPar));

  check("grouping: unknown records are ignored, not fatal",
        stageWidths([{ event: "if", line: 4 }, A("a", "i"), { event: "return", line: 9 }]).length === 1);
}

// A trace shaped like a real saved run: 5 agent_calls (one behind an untaken
// `if`), opening with a 2-wide parallel.
const TRACE = [
  A("literature-searcher", "Research EIF4E inhibitors."),
  A("drug-discovery-scientist", "Inspect the task files."),
  { event: "parallel_call", line: 2, width: 2 },
  A("drug-discovery-scientist", "Design a pharmacophore."),
  A("drug-discovery-scientist", "Evaluate the candidate."),
  { event: "if", line: 44 },
  A("drug-discovery-scientist", "Revise the rules."),
  { event: "return", line: 61 },
];
const DSL_PLAN = { mode: "python_dsl", steps: [], workflow_static_trace: TRACE };
const RAN = [
  ["node-1", "literature-searcher", "Research EIF4E inhibitors.", []],
  ["node-2", "drug-discovery-scientist", "Inspect the task files.", []],
  ["node-3", "drug-discovery-scientist", "Design a pharmacophore.", ["node-1", "node-2"]],
  ["node-4", "drug-discovery-scientist", "Evaluate the candidate.", ["node-3"]],
];

// ---- 2. the whole structure is drawn up front ------------------------------
{
  api.resetGraph();
  api.ingestMasEvent({ event: "planner_finished", plan: DSL_PLAN });
  const nodes = snap();
  check("skeleton: all 5 agents drawn, including the untaken-branch one",
        api.state.nodes.size === 5, api.state.nodes.size);
  check("skeleton: every slot starts pending", nodes.every(n => n.status === "pending"),
        JSON.stringify(nodes.map(n => n.status)));
  check("skeleton: no fabricated edges", api.state.edges.length === 0, api.state.edges.length);
  check("skeleton: parallel children share a stage, the rest step forward",
        JSON.stringify(nodes.map(n => n.stage)) === JSON.stringify([0, 0, 1, 2, 3]),
        JSON.stringify(nodes.map(n => n.stage)));
}

// ---- 3. runtime agents bind onto the predicted slots ----------------------
{
  for (const [id, role, instr, deps] of RAN) {
    api.ingestMasEvent({ event: "agent_started", node_id: id, role, instruction: instr, received_from: deps });
    check("bind: " + id + " maps onto a predicted slot",
          byId("s" + (Number(id.slice(5)))).status === "running",
          JSON.stringify(snap()));
    api.ingestMasEvent({ event: "agent_finished", node_id: id, role, instruction: instr,
                         received_from: deps, public_output: "ok" });
  }
  const nodes = snap();
  check("bind: the 4 agents that ran are done",
        ["s1", "s2", "s3", "s4"].every(k => byId(k).status === "done"),
        JSON.stringify(nodes.map(n => n.id + ":" + n.status)));
  check("bind: the untaken-branch agent stays pending (never got a turn)",
        byId("s5").status === "pending", byId("s5") && byId("s5").status);
  check("bind: still exactly 5 boxes, no duplicates", api.state.nodes.size === 5, api.state.nodes.size);
  // node-3 declares [node-1, node-2] and node-4 declares [node-3].
  check("bind: real dependencies became edges", api.state.edges.length === 3, api.state.edges.length);
  check("bind: no self-edges", api.state.edges.every(e => e.from !== e.to), JSON.stringify(api.state.edges));
  check("bind: edges land on drawn slots, not runtime ids",
        api.state.edges.every(e => byId(e.from) && byId(e.to)), JSON.stringify(api.state.edges));
  check("bind: the parallel pair both feed the next stage",
        api.state.edges.filter(e => e.to === "s3").length === 2, JSON.stringify(api.state.edges));
}

// ---- 4. problem_finished must not paint a second graph --------------------
{
  const before = api.state.nodes.size;
  api.reconcilePlan({ mode: "python_dsl", execution_order: RAN.map(r => r[0]),
                      steps: RAN.map(([id, role, instr]) => ({ id, role, objective: instr, depends_on: [] })),
                      workflow_static_trace: TRACE });
  check("reconcile: python_dsl steps are NOT re-ingested as extra boxes",
        api.state.nodes.size === before, api.state.nodes.size + " vs " + before);
  check("reconcile: no runtime-id boxes appeared",
        !api.state.nodes.has("node-1") && !api.state.nodes.has("node-4"), JSON.stringify(snap()));
  check("reconcile: finished nodes were not downgraded",
        byId("s1").status === "done", byId("s1").status);
}

// ---- 5. a loop body keeps ONE box and cycles its colour -------------------
{
  const loopPlan = { mode: "python_dsl", steps: [],
                     workflow_static_trace: [A("chemist", "Refine the answer.")] };
  api.resetGraph();
  api.ingestMasEvent({ event: "planner_finished", plan: loopPlan });
  check("loop: one box for a range(N) body", api.state.nodes.size === 1, api.state.nodes.size);
  const seen = [];
  for (let i = 1; i <= 3; i++) {
    const id = "node-" + i;
    api.ingestMasEvent({ event: "agent_started", node_id: id, role: "chemist", instruction: "Refine the answer." });
    seen.push(byId("s1").status);
    api.ingestMasEvent({ event: "agent_finished", node_id: id, role: "chemist", instruction: "Refine the answer.", public_output: "x" });
  }
  check("loop: the same box runs, completes, runs again",
        JSON.stringify(seen) === JSON.stringify(["running", "running", "running"]), JSON.stringify(seen));
  check("loop: no clones invented", api.state.nodes.size === 1, api.state.nodes.size);
  check("loop: ends done", byId("s1").status === "done", byId("s1").status);
}

// ---- 6. legacy-json mode is untouched ------------------------------------
{
  const legacy = { mode: "legacy_json", steps: [
    { id: "a", role: "chemist", objective: "First.", depends_on: [] },
    { id: "b", role: "physicist", objective: "Second.", depends_on: ["a"] },
  ] };
  api.resetGraph();
  api.ingestMasEvent({ event: "planner_finished", plan: legacy });
  check("legacy: no skeleton, steps keyed by their own ids",
        !api.state.hasSkeleton && api.state.nodes.has("a") && api.state.nodes.has("b"),
        JSON.stringify(snap()));
  check("legacy: plan.steps edges drawn", api.state.edges.length === 1, JSON.stringify(api.state.edges));
  api.ingestMasEvent({ event: "agent_started", node_id: "node-7", role: "generalist", instruction: "Ad hoc." });
  check("legacy: an unplanned agent still appears", api.state.nodes.has("node-7"), JSON.stringify(snap()));
}

// ---- 7. the red state is reachable ---------------------------------------
{
  api.resetGraph();
  api.ingestMasEvent({ event: "planner_finished", plan: DSL_PLAN });
  api.ingestMasEvent({ event: "agent_started", node_id: "node-1", role: "literature-searcher",
                       instruction: "Research EIF4E inhibitors.", received_from: [] });
  const running = rects().find(r => r.fill === "var(--node-run)");
  check("colour: a running node is drawn blue", !!running, JSON.stringify(rects().map(r => r.fill)));

  api.handleEvent({
    event: "problem_finished", completed: 1, total: 1, problem: { id: "p1", index: 1 },
    result: {
      status: "failed", error: "agent blew up", output_dir: "/tmp/x", final_answer: "",
      plan: { mode: "python_dsl", steps: [] },
      problem_meta: { id: "p1", dataset: "SciAgentGYM", subject: "s", topic: "t",
                      expected_answer: "", expected_tools: [] },
      grade: { answer_correct: false, answer_score: 0.0, tool_coverage: 0.0, checklist_score: 0.0,
               answer_notes: [], missing_tools: [], called_tools: [], error: "" },
    },
  });
  check("failure: a problem_finished with status=failed turns running nodes red",
        byId("s1").status === "failed", byId("s1") && byId("s1").status);
  check("failure: still no duplicate boxes after failure",
        api.state.nodes.size === 5, api.state.nodes.size);
  const red = rects().find(r => r.fill === "var(--node-fail)");
  check("failure: the node is actually painted red", !!red, JSON.stringify(rects().map(r => r.fill)));
}

// ---- 8. legend + no column overlap ---------------------------------------
{
  api.renderLegend();
  const legend = api.$("graphLegend");
  check("legend: four state swatches", legend.children.length === 5, legend.children.length);
  const swatches = legend.children.slice(0, 4).map(s => s.children[0].style.background);
  check("legend: swatches use the same definitions as the graph",
        JSON.stringify(swatches) === JSON.stringify(["var(--node)", "var(--node-run)",
                                                     "var(--node-done)", "var(--node-fail)"]),
        JSON.stringify(swatches));

  // A 6-stage skeleton at a modest window width used to overlap: the old
  // formula floored the box at 145px while the column pitch fell to 138px.
  api.$("graph").clientWidth = 900;
  api.$("graph").clientHeight = 420;
  api.resetGraph();
  api.ingestMasEvent({ event: "planner_finished", plan: {
    mode: "python_dsl", steps: [],
    workflow_static_trace: [1, 2, 3, 4, 5, 6].map(i => A("chemist", "Step " + i + ".")),
  } });
  const boxes = rects();
  check("layout: all 6 boxes drawn", boxes.length === 6, boxes.length);
  const xs = Array.from(new Set(boxes.map(b => b.x))).sort((a, b) => a - b);
  const gaps = xs.slice(1).map((x, i) => x - xs[i]);
  check("layout: consecutive columns do not overlap",
        gaps.every(g => g >= boxes[0].w), JSON.stringify({ xs, gaps, w: boxes[0].w }));
  check("layout: boxes stay on the canvas",
        boxes.every(b => b.x + b.w <= 900), JSON.stringify(boxes.map(b => b.x + b.w)));
}

console.log(JSON.stringify(results));
"""


@pytest.mark.skipif(_NODE is None, reason="node is not installed")
def test_graph_skeleton_behaviour(tmp_path: Path) -> None:
    """Execute the dashboard's real embedded JS and assert the graph behaviour."""
    dashboard = _import_dashboard()

    script = re.search(r"<script>(.*?)</script>", dashboard.INDEX_HTML, re.S)
    assert script, "INDEX_HTML should contain one <script> block"

    js_path = tmp_path / "dashboard.js"
    js_path.write_text(script.group(1), encoding="utf-8")
    harness_path = tmp_path / "harness.js"
    harness_path.write_text(_JS_HARNESS, encoding="utf-8")

    proc = subprocess.run(
        [_NODE, str(harness_path), str(js_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"harness failed:\nSTDOUT{proc.stdout}\nSTDERR{proc.stderr}"

    results = json.loads(proc.stdout.strip().splitlines()[-1])
    failures = [r for r in results if not r["ok"]]
    assert not failures, "\n".join(f'{r["name"]} -> {r["detail"]}' for r in failures)
    # Guard against the harness silently asserting nothing.
    assert len(results) > 25, f"expected a full run of checks, got {len(results)}"


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))