"""Interactive sciMAS evaluation dashboard.

Run:
    python web_dashboard.py --port 7860
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from flask import Flask, Response, jsonify, request

from orchestrator import SciMASOrchestrator

TESTS_DIR = Path(__file__).resolve().parent / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from dataset import (  # noqa: E402
    DATASET_CHOICES,
    SCIAGENTGYM,
    Problem,
    dataset_dir_name,
    default_dataset_root,
    filter_problems,
    format_problem_for_solver,
    load_dataset,
    normalize_dataset_name,
    stage_problem_images,
)
from grader import (  # noqa: E402
    GradingResult,
    ToolCall,
    grade_problem,
    sciagentgym_expected_display,
    set_judge_cache_dir,
)
from runner import (  # noqa: E402
    _StreamJsonRunner,
    _all_registered_tools,
    _maybe_run_smdd_official_eval,
    _problem_file_stem,
)


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = APP_ROOT / "runs-dashboard"

# The ad-hoc "type your own question" pseudo-dataset. Deliberately NOT added to
# dataset.DATASET_CHOICES: if it ever leaks past the short-circuits below,
# normalize_dataset_name() raises a loud ValueError instead of letting
# load_dataset() fall through to _load_sciagentgym_dataset and quietly return
# an empty list. It is a UI concept, not a dataset.
WRITE_IN_DATASET = "__write_in__"  # wire value: plain ASCII, safe as a dict key
WRITE_IN_LABEL = "当场手写"  # display only
WRITE_IN_PROBLEM_ID = "write-in"
WRITE_IN_GRADE_NOTE = (
    "ad-hoc write-in question: no gold answer, grading skipped "
    "(no LLM judge, no score)"
)


@dataclass
class DashboardJob:
    id: str
    config: dict[str, Any]
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    current_problem: Optional[dict[str, Any]] = None
    total_problems: int = 0
    completed_problems: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    _condition: threading.Condition = field(default_factory=threading.Condition, repr=False)

    def emit(self, event: dict[str, Any]) -> None:
        payload = {
            "seq": 0,
            "time": datetime.now().isoformat(timespec="seconds"),
            **event,
        }
        with self._condition:
            payload["seq"] = len(self.events) + 1
            self.events.append(payload)
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "id": self.id,
                "config": self.config,
                "status": self.status,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "current_problem": self.current_problem,
                "total_problems": self.total_problems,
                "completed_problems": self.completed_problems,
                "results": self.results,
                "error": self.error,
                "event_count": len(self.events),
            }


app = Flask(__name__)
JOBS: dict[str, DashboardJob] = {}


def _parse_ids(ids_text: str | None) -> list[str] | None:
    if not ids_text:
        return None
    ids = [item.strip() for item in ids_text.split(",") if item.strip()]
    return ids or None


def _parse_roles(roles_text: str | None) -> list[str] | None:
    if not roles_text:
        return None
    roles = re.split(r"[\s,]+", roles_text.strip())
    roles = [role for role in roles if role]
    return roles or None


def _problem_summary(problem: Problem, index: int | None = None) -> dict[str, Any]:
    question = problem.question or ""
    payload = {
        "id": str(problem.id),
        "subject": problem.subject,
        "topic": problem.topic,
        "question": question,
        "question_preview": question[:260] + ("..." if len(question) > 260 else ""),
    }
    if index is not None:
        payload["index"] = index
    filename = str(getattr(problem, "filename", "") or "")
    if filename:
        payload["filename"] = filename
    return payload


def _is_write_in(config: dict[str, Any]) -> bool:
    return str(config.get("dataset") or "").strip() == WRITE_IN_DATASET


def _write_in_problem(config: dict[str, Any]) -> Optional[Problem]:
    """Synthesize the single Problem for an ad-hoc write-in question.

    Returns ``None`` for a blank question rather than raising: the request that
    triggered this comes from a ``change`` handler firing the moment the option
    is selected, so an exception here would surface as an unhandled error in the
    UI before the user has typed anything.
    """
    question = str(config.get("write_in_question") or "").strip()
    if not question:
        return None
    return Problem(
        id=WRITE_IN_PROBLEM_ID,
        filename="",  # nothing on disk; _problem_summary omits falsy filenames
        question=question,
        answer="",  # no gold answer, so grade_answer stays non-numeric
        subject=WRITE_IN_LABEL,
        topic="ad-hoc",
        dataset_name=WRITE_IN_DATASET,
        source_dataset=WRITE_IN_DATASET,
    )


def _load_filtered_problems(config: dict[str, Any]) -> tuple[list[Problem], Path, str]:
    if _is_write_in(config):
        # Short-circuit before the dataset loader: `limit`/`subject`/`topic`/
        # `ids`/`query` are all ignored because a write-in run is always exactly
        # one problem. The returned root is unused by _run_job, which reads
        # output_root from the config.
        problem = _write_in_problem(config)
        return ([problem] if problem else []), Path(""), WRITE_IN_DATASET

    dataset_name = normalize_dataset_name(config.get("dataset") or "SciAgentGYM")
    root_text = str(config.get("root") or "").strip()
    root = Path(root_text).expanduser() if root_text else default_dataset_root(dataset_name)
    split = str(config.get("split") or "train")
    problems = load_dataset(root, dataset_name=dataset_name, split=split)
    problems = filter_problems(
        problems,
        subject=config.get("subject") or None,
        topic=config.get("topic") or None,
        ids=_parse_ids(config.get("ids")),
        query=config.get("query") or None,
    )
    limit = int(config.get("limit") or 0)
    if limit > 0:
        problems = problems[:limit]
    return problems, root, dataset_name


def _load_filter_options(config: dict[str, Any]) -> tuple[list[str], list[str], int, Path, str]:
    if _is_write_in(config):
        # Empty subject/topic lists, never a ValueError: the frontend's
        # loadFilterOptions throws on a non-OK response, and this is called the
        # instant the option is selected — before any question has been typed.
        problem = _write_in_problem(config)
        return [], [], (1 if problem else 0), Path(""), WRITE_IN_DATASET

    dataset_name = normalize_dataset_name(config.get("dataset") or "SciAgentGYM")
    root_text = str(config.get("root") or "").strip()
    root = Path(root_text).expanduser() if root_text else default_dataset_root(dataset_name)
    split = str(config.get("split") or "train")
    problems = load_dataset(root, dataset_name=dataset_name, split=split)
    subjects = sorted({problem.subject for problem in problems if problem.subject})
    selected_subject = str(config.get("subject") or "").strip()
    topic_source = [
        problem
        for problem in problems
        if not selected_subject or problem.subject == selected_subject
    ]
    topics = sorted({problem.topic for problem in topic_source if problem.topic})
    return subjects, topics, len(problems), root, dataset_name


def _event_callback(job: DashboardJob, problem: Problem, index: int):
    def callback(event: dict[str, Any]) -> None:
        job.emit(
            {
                "event": "mas_event",
                "problem": _problem_summary(problem, index),
                "payload": event,
            }
        )

    return callback


def _grade_one_problem(
    job: DashboardJob,
    problem: Problem,
    report,
    runner: ClaudeRunner,
    output_root: Path,
) -> GradingResult:
    """Run the dataset-appropriate grader on one completed problem.

    Mirrors the tests/runner._run_single path: capture tool calls, optionally
    invoke the SMDDBench official Docker evaluator, then ``grade_problem``.
    Persists the result to ``<output_root>/per-problem/<id>.json``.
    """
    # Collect every mcp__ tool_use the runner saw across planner +
    # worker steps + synth. (StreamJsonRunner accumulates them; a
    # plain ClaudeRunner keeps the list empty, which is fine — it
    # just means tool coverage shows 0%.)
    tool_calls: list[ToolCall] = []
    if hasattr(runner, "_tool_calls") and runner._tool_calls:
        tool_calls = [
            ToolCall.from_full(call["tool"], call.get("arguments", {}))
            for call in runner._tool_calls
        ]

    final_answer = report.final_answer or ""
    cost_usd = sum((r.cost_usd or 0.0) for r in report.runs)
    duration_ms = sum((r.duration_ms or 0) for r in report.runs)

    # SMDDBench: optionally run the official Docker evaluator so the
    # grade picks up the real per-task score instead of the format-hint
    # proxy. Mutates problem.task_info in-place, mirroring tests/runner.
    if bool(job.config.get("smdd_official_eval")):
        _maybe_run_smdd_official_eval(
            problem,
            final_answer=final_answer,
            cfg=_smdd_run_config_from_job(job),
            smdd_eval_root=output_root / "smdd-official",
        )

    grade = grade_problem(
        problem,
        predicted_answer=final_answer,
        tool_calls=tool_calls,
        available_tools=_all_registered_tools(),
        cost_usd=cost_usd,
        duration_ms=duration_ms,
    )
    _persist_grade(
        problem,
        grade,
        scimas_run_dir=report.output_dir,
        output_root=output_root,
    )
    return grade


def _smdd_run_config_from_job(job: DashboardJob):
    """Build a ``runner.RunConfig``-shaped object for the SMDD helper."""
    from runner import RunConfig

    return RunConfig(
        smdd_official_eval=True,
        smdd_docker_image=str(job.config.get("smdd_docker_image") or "smdd-evals"),
        smdd_eval_timeout=float(job.config.get("smdd_eval_timeout") or 3600.0),
        smdd_gpu_id=job.config.get("smdd_gpu_id") or None,
    )


def _expected_display(problem: Problem) -> str:
    """The expected-answer string to show for ``problem``.

    Only SciAgentGYM needs the special case; every other dataset's
    ``answer`` field is shown verbatim.
    """
    if problem.dataset_name == SCIAGENTGYM:
        return sciagentgym_expected_display(problem)
    return problem.answer


def _problem_meta_for_event(problem: Problem) -> dict[str, Any]:
    """Return the per-problem metadata the frontend badges need."""
    return {
        "id": str(problem.id),
        "dataset": problem.dataset_name,
        "subject": problem.subject,
        "topic": problem.topic,
        # SciAgentGYM's `answer` field is sometimes an intermediate step of
        # the reference solution rather than its result, so the grader
        # renders what the dataset actually provides next to it.
        "expected_answer": _expected_display(problem),
        "expected_tools": list(problem.expected_tools),
    }


def _mcp_config_for_job(job: DashboardJob) -> Optional[str]:
    """Map the dashboard's MCP controls onto ``ClaudeRunner``'s contract.

    ``ClaudeRunner`` reads an empty string as an *explicit* opt-out (the
    same sentinel ``main.py --no-mcp`` passes) and ``None`` as "auto-detect
    ``config/mcp.json``". The form posts ``""`` both when the user ticks
    "Disable MCP" and when they simply leave the path field blank — which
    used to silently disable MCP for every dashboard run despite the field
    advertising "default config/mcp.json". Only the checkbox opts out now;
    a blank field falls through to auto-detection.
    """
    if job.config.get("no_mcp"):
        return ""
    return str(job.config.get("mcp_config") or "").strip() or None


def _run_job(job: DashboardJob) -> None:
    job.status = "running"
    job.started_at = datetime.now().isoformat(timespec="seconds")
    job.emit({"event": "job_started", "job": job.snapshot()})
    try:
        problems, root, dataset_name = _load_filtered_problems(job.config)
        job.total_problems = len(problems)
        if not problems:
            job.status = "failed"
            # Backstop behind the frontend's own empty-question check; this
            # runs before the output_root mkdir below, so a blank write-in
            # attempt leaves no runs-dashboard/ directory behind.
            job.error = (
                f'在"{WRITE_IN_LABEL}"里输入问题后再开始运行。'
                if _is_write_in(job.config)
                else "No problems match the selected filters."
            )
            job.emit({"event": "job_failed", "error": job.error})
            return

        output_root = Path(job.config.get("output_root") or DEFAULT_OUTPUT_ROOT)
        output_root = output_root / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{job.id[:8]}"
        output_root.mkdir(parents=True, exist_ok=True)

        # Mirror tests/runner.run_batch — pin the RCB official-judge cache
        # to this run directory so repeat runs (or per-problem re-grades
        # from report.py) reuse the same scores.
        if str(job.config.get("research_claw_judge") or "").strip().lower() == "off":
            os.environ["RESEARCH_CLAW_JUDGE_DISABLED"] = "1"
            set_judge_cache_dir(None)
        else:
            set_judge_cache_dir(str(output_root / "judge_cache"))

        per_problem_dir = output_root / "per-problem"
        per_problem_dir.mkdir(parents=True, exist_ok=True)

        roles = _parse_roles(job.config.get("roles"))
        planner_mode = str(job.config.get("planner_mode") or "python-dsl").replace("-", "_")
        mcp_config = _mcp_config_for_job(job)

        for index, problem in enumerate(problems, 1):
            job.current_problem = _problem_summary(problem, index)
            job.emit(
                {
                    "event": "problem_started",
                    "problem": job.current_problem,
                    "completed": job.completed_problems,
                    "total": job.total_problems,
                }
            )
            # Build a stream-json runner so the orchestrator captures
            # every mcp__ tool_use across planner + worker + synth steps.
            # Same swap that tests/runner._factory_with_runner performs.
            runner = _StreamJsonRunner(
                binary=str(job.config.get("claude_bin") or "claude"),
                model=str(job.config.get("model") or "") or None,
                mcp_config_path=mcp_config,
                allowed_tools=None,  # orchestrator sets per-step
                permission_mode=str(job.config.get("permission_mode") or "bypassPermissions"),
                dangerously_skip_permissions=not bool(job.config.get("no_skip_permissions")),
                timeout=float(job.config.get("timeout") or 600.0),
            )
            orchestrator = SciMASOrchestrator(
                output_dir=output_root / "scimas-runs",
                use_skill_routing=not bool(job.config.get("no_skill_routing")),
                strict_skill_routing=not bool(job.config.get("allow_role_tool_fallback")),
                claude_bin=str(job.config.get("claude_bin") or "claude"),
                model=str(job.config.get("model") or "") or None,
                mcp_config_path=mcp_config,
                permission_mode=str(job.config.get("permission_mode") or "bypassPermissions"),
                dangerously_skip_permissions=not bool(job.config.get("no_skip_permissions")),
                timeout=float(job.config.get("timeout") or 600.0),
                planner_mode=planner_mode,
                event_callback=_event_callback(job, problem, index),
                max_review_attempts=int(job.config.get("max_review_attempts") or 3),
            )
            orchestrator.runner = runner
            # Stage the question's figures inside this job's output directory
            # so the agents can Read them from inside the repo, and so the
            # per-problem payload keeps a copy of what was actually answered.
            problem.image_paths_local = stage_problem_images(
                problem, output_root / "figure-assets" / _problem_file_stem(problem)
            )
            try:
                report = orchestrator.run(
                    problem=format_problem_for_solver(problem),
                    roles=roles,
                    auto_synthesize=not bool(job.config.get("no_auto_synthesize")),
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                # No report came back — grade with empty predicted_answer
                # so the per-problem.json exists and the failure shows up
                # in the score panel with answer_score=0.0.
                grade = _build_failure_grade(problem, error=error)
                _persist_grade(problem, grade, scimas_run_dir=None, output_root=output_root)
                # The orchestrator raises before `_save_report` runs, so a
                # failed problem leaves no artifact behind and job.events is
                # in-memory only. Dump the events that did arrive, otherwise
                # a planning failure is undebuggable after the fact.
                _dump_failure_events(job, problem, output_root, error)
                result = {
                    "problem": job.current_problem,
                    "status": "failed",
                    "error": error,
                    "output_dir": None,
                    "final_answer": "",
                    "plan": {},
                    "run_count": 0,
                    "problem_meta": _problem_meta_for_event(problem),
                    "grade": grade.to_dict(),
                }
                job.results.append(result)
                job.completed_problems += 1
                job.emit(
                    {
                        "event": "problem_finished",
                        "problem": job.current_problem,
                        "result": result,
                        "completed": job.completed_problems,
                        "total": job.total_problems,
                    }
                )
                continue

            try:
                if _is_write_in(job.config):
                    grade = _write_in_grade(problem, report, output_root=output_root)
                else:
                    grade = _grade_one_problem(job, problem, report, runner, output_root)
            except Exception as exc:
                grade = GradingResult(
                    problem_id=problem.id,
                    dataset_name=problem.dataset_name,
                    subject=problem.subject,
                    topic=problem.topic,
                    expected_answer=problem.answer,
                    predicted_answer=report.final_answer or "",
                    expected_tools=list(problem.expected_tools),
                    error=f"grading failed: {type(exc).__name__}: {exc}",
                )

            result = {
                "problem": job.current_problem,
                "status": "completed",
                "output_dir": report.output_dir,
                "final_answer": report.final_answer,
                "plan": report.plan.to_dict(),
                "run_count": len(report.runs),
                "problem_meta": _problem_meta_for_event(problem),
                "grade": grade.to_dict(),
            }
            job.results.append(result)
            job.completed_problems += 1
            job.emit(
                {
                    "event": "problem_finished",
                    "problem": job.current_problem,
                    "result": result,
                    "completed": job.completed_problems,
                    "total": job.total_problems,
                }
            )

        job.status = "completed"
        job.finished_at = datetime.now().isoformat(timespec="seconds")
        job.emit({"event": "job_finished", "job": job.snapshot()})
    except Exception as exc:
        job.status = "failed"
        job.finished_at = datetime.now().isoformat(timespec="seconds")
        job.error = f"{type(exc).__name__}: {exc}"
        job.emit(
            {
                "event": "job_failed",
                "error": job.error,
                "traceback": traceback.format_exc(),
            }
        )


def _build_failure_grade(problem: Problem, *, error: str) -> GradingResult:
    """Construct a zero-score GradingResult when the orchestrator itself throws."""
    return GradingResult(
        problem_id=problem.id,
        dataset_name=problem.dataset_name,
        subject=problem.subject,
        topic=problem.topic,
        expected_answer=problem.answer,
        predicted_answer="",
        expected_tools=list(problem.expected_tools),
        error=error,
    )


def _write_in_grade(problem: Problem, report, *, output_root: Path) -> GradingResult:
    """Ungraded placeholder for an ad-hoc write-in problem.

    ``grade_problem`` is deliberately never called: there is no gold answer, so
    the only thing it could do is spend a real LLM-judge API call on a question
    that has nothing to be judged against.

    ``answer_score`` stays the float ``0.0`` rather than becoming ``None`` — the
    per-problem payload is shared with tests/runner.py and read back by
    tests/report.py, which formats the score numerically and would raise
    TypeError on None. The "ungraded" signal therefore lives in
    ``dataset_name`` and ``answer_notes``, not in a nulled-out score.
    """
    grade = GradingResult(
        problem_id=problem.id,
        dataset_name=problem.dataset_name,
        subject=problem.subject,
        topic=problem.topic,
        predicted_answer=report.final_answer or "",
        answer_score=0.0,
        answer_notes=[WRITE_IN_GRADE_NOTE],
        # Run metrics, not grading metrics — carried through so the persisted
        # JSON keeps the same shape as a normal run's.
        cost_usd=sum((r.cost_usd or 0.0) for r in report.runs),
        duration_ms=sum((r.duration_ms or 0) for r in report.runs),
    )
    _persist_grade(
        problem,
        grade,
        scimas_run_dir=report.output_dir,
        output_root=output_root,
    )
    return grade


def _persist_grade(
    problem: Problem,
    grade: GradingResult,
    *,
    scimas_run_dir: Optional[str],
    output_root: Path,
) -> Path:
    """Write the ``per-problem/<id>.json`` payload and return its path."""
    per_problem_dir = output_root / "per-problem"
    per_problem_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "problem": problem.to_dict(),
        "grade": grade.to_dict(),
        "scimas_run_dir": scimas_run_dir,
    }
    out_file = per_problem_dir / f"{_problem_file_stem(problem)}.json"
    out_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_file


def _dump_failure_events(
    job: "DashboardJob",
    problem: Problem,
    output_root: Path,
    error: str,
) -> Optional[Path]:
    """Persist the events seen for a failed problem.

    A run that dies inside the orchestrator never reaches `_save_report`, so
    without this the planner prompt, the reviewer's raw reply and the
    validation errors are all lost once the process moves on. Best-effort:
    a failure to write must not mask the original error.
    """
    try:
        per_problem_dir = output_root / "per-problem"
        per_problem_dir.mkdir(parents=True, exist_ok=True)
        out_file = per_problem_dir / f"{_problem_file_stem(problem)}.events.jsonl"
        # job.events spans every problem in the job; keep the ones tagged with
        # this problem, falling back to the whole log if nothing matches.
        current = job.current_problem
        relevant = [e for e in list(job.events) if e.get("problem") == current]
        with out_file.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"event": "problem_error", "error": error}, ensure_ascii=False)
                + "\n"
            )
            for entry in relevant or list(job.events):
                handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return out_file
    except Exception:
        return None


@app.get("/")
def index() -> str:
    # The write-in sentinel is injected rather than hard-coded a second time in
    # the embedded JS, so there is exactly one place it is defined. If this
    # substitution ever breaks, the browser-side value is the literal
    # "@@WRITE_IN@@" and every request fails loudly with a 400 instead of
    # silently resolving to some other dataset.
    return INDEX_HTML.replace("@@WRITE_IN@@", WRITE_IN_DATASET)


@app.get("/api/datasets")
def api_datasets():
    return jsonify(
        {
            "datasets": [*DATASET_CHOICES, WRITE_IN_DATASET],
            "labels": {WRITE_IN_DATASET: WRITE_IN_LABEL},
            # No entry for WRITE_IN_DATASET on purpose: default_dataset_root()
            # has no branch for it and would hand back the SciAgentGYM root as a
            # bogus default. loadDatasets() falls back to "" via `|| ""`.
            "defaults": {
                name: str(default_dataset_root(name))
                for name in DATASET_CHOICES
            },
        }
    )


@app.post("/api/problems")
def api_problems():
    config = request.get_json(force=True, silent=True) or {}
    try:
        problems, root, dataset_name = _load_filtered_problems(config)
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
    subjects = sorted({problem.subject for problem in problems if problem.subject})
    topics = sorted({problem.topic for problem in problems if problem.topic})
    return jsonify(
        {
            "dataset": dataset_name,
            # Path("") stringifies to "." — report it as empty for write-in,
            # since there is no dataset root behind that run.
            "root": "" if _is_write_in(config) else str(root),
            "count": len(problems),
            "subjects": subjects,
            "topics": topics,
            "problems": [
                _problem_summary(problem, index)
                for index, problem in enumerate(problems[:200], 1)
            ],
        }
    )


@app.post("/api/filter-options")
def api_filter_options():
    config = request.get_json(force=True, silent=True) or {}
    try:
        subjects, topics, count, root, dataset_name = _load_filter_options(config)
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
    return jsonify(
        {
            "dataset": dataset_name,
            "root": "" if _is_write_in(config) else str(root),
            "count": count,
            "subjects": subjects,
            "topics": topics,
        }
    )


@app.post("/api/jobs")
def api_start_job():
    config = request.get_json(force=True, silent=True) or {}
    job_id = uuid.uuid4().hex
    config.setdefault("limit", 1)
    config.setdefault("planner_mode", "python-dsl")
    config.setdefault("output_root", str(DEFAULT_OUTPUT_ROOT))
    job = DashboardJob(id=job_id, config=config)
    JOBS[job_id] = job
    thread = threading.Thread(target=_run_job, args=(job,), daemon=True)
    thread.start()
    return jsonify(job.snapshot())


@app.get("/api/jobs")
def api_jobs():
    return jsonify({"jobs": [job.snapshot() for job in JOBS.values()]})


@app.get("/api/jobs/<job_id>")
def api_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    return jsonify(job.snapshot())


@app.get("/api/jobs/<job_id>/events")
def api_job_events(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    try:
        last_event_id = int(request.headers.get("Last-Event-ID") or "0")
    except ValueError:
        last_event_id = 0

    def stream():
        index = max(0, last_event_id)
        while True:
            pending: list[dict[str, Any]] = []
            terminal = False
            with job._condition:
                while index >= len(job.events) and job.status in {"queued", "running"}:
                    job._condition.wait(timeout=15)
                    if index >= len(job.events):
                        pending.append({"keepalive": True})
                while index < len(job.events):
                    pending.append(job.events[index])
                    index += 1
                terminal = job.status not in {"queued", "running"} and index >= len(job.events)
            for event in pending:
                if event.get("keepalive"):
                    yield ": keep-alive\n\n"
                else:
                    yield (
                        f"id: {event.get('seq', index)}\n"
                        f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    )
            if terminal:
                break

    return Response(stream(), mimetype="text/event-stream")


INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>sciMAS Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #1d2433;
      --muted: #667085;
      --line: #d8dee8;
      --accent: #2f6fed;
      --ok: #15803d;
      --warn: #b45309;
      --bad: #b42318;
      /* MAS graph node states. Four only: pending / running / done / failed.
         Colour is the sole state carrier on the graph, so these are the single
         definition shared by the nodes and the legend (see fillFor/strokeFor). */
      --node: #f2f4f7;
      --node-run: #eff6ff;
      --node-done: #ecfdf3;
      --node-fail: #fef3f2;
      --node-stroke: #98a2b3;
      --node-run-stroke: #2f6fed;
      --node-done-stroke: #22c55e;
      --node-fail-stroke: #b42318;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--ink); }
    header {
      height: 56px; display: flex; align-items: center; justify-content: space-between;
      padding: 0 18px; background: #111827; color: white; border-bottom: 1px solid #0b1220;
    }
    header h1 { margin: 0; font-size: 17px; font-weight: 650; letter-spacing: 0; }
    header .meta { color: #cbd5e1; font-size: 13px; }
    main { display: grid; grid-template-columns: 360px minmax(0, 1fr); min-height: calc(100vh - 56px); }
    aside {
      border-right: 1px solid var(--line); background: var(--panel); padding: 14px;
      overflow: auto;
    }
    .workspace { display: grid; grid-template-rows: auto minmax(380px, 1fr) 240px; gap: 12px; padding: 14px; min-width: 0; }
    .band { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
    .section { margin-bottom: 16px; }
    .section h2 { margin: 0 0 10px; font-size: 13px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
    label { display: block; font-size: 12px; color: #344054; margin: 9px 0 5px; font-weight: 600; }
    input, select, textarea {
      width: 100%; border: 1px solid #ccd4df; border-radius: 6px; padding: 8px 9px;
      font: inherit; font-size: 13px; background: white; color: var(--ink);
    }
    textarea { min-height: 58px; resize: vertical; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; }
    .checkrow { display: flex; align-items: center; gap: 8px; margin-top: 9px; font-size: 13px; color: #344054; }
    .checkrow input { width: 16px; height: 16px; }
    button {
      border: 1px solid #1d4ed8; background: var(--accent); color: white;
      border-radius: 6px; padding: 9px 11px; font: inherit; font-size: 13px; font-weight: 650;
      cursor: pointer;
    }
    button.secondary { background: white; color: #1f2937; border-color: #cfd6e2; }
    button:disabled { opacity: .55; cursor: not-allowed; }
    .actions { display: flex; gap: 8px; margin-top: 12px; }
    .statusbar { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; padding: 12px; }
    .metric { border-right: 1px solid var(--line); min-height: 48px; }
    .metric:last-child { border-right: none; }
    .metric .k { font-size: 12px; color: var(--muted); }
    .metric .v { font-size: 18px; font-weight: 700; margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .graph-wrap { position: relative; overflow: hidden; }
    #graph { width: 100%; height: 100%; min-height: 380px; display: block; background: linear-gradient(#fff, #fbfcff); border-radius: 8px; }
    /* Absolutely positioned on purpose: #graph is height:100%, so a normal flow
       sibling would shrink the SVG and change svg.clientHeight, re-flowing the
       whole graph every time the legend is painted. */
    .graph-legend {
      position: absolute; top: 8px; right: 10px; display: flex; flex-wrap: wrap; align-items: center;
      gap: 4px 10px; padding: 5px 9px; font-size: 11px; color: var(--muted);
      background: rgba(255,255,255,.92); border: 1px solid var(--line); border-radius: 6px;
      pointer-events: none; max-width: calc(100% - 20px);
    }
    .graph-legend .lg { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }
    .graph-legend .lg i { width: 10px; height: 10px; border-radius: 3px; border: 1.5px solid; display: inline-block; }
    .graph-legend .note { color: #98a2b3; }
    .bottom { display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(0, .9fr); gap: 12px; min-height: 0; }
    .panel-title { padding: 10px 12px; border-bottom: 1px solid var(--line); font-size: 13px; font-weight: 700; color: #344054; }
    .scroll { overflow: auto; height: calc(100% - 39px); padding: 10px 12px; }
    #problemList { max-height: 260px; overflow: auto; border: 1px solid var(--line); border-radius: 6px; background: #fbfcfe; }
    .problem { padding: 9px; border-bottom: 1px solid #e5eaf1; font-size: 12px; }
    .problem:last-child { border-bottom: none; }
    .problem strong { display: block; margin-bottom: 3px; color: #1f2937; }
    .logline { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; color: #334155; border-bottom: 1px solid #edf1f6; padding: 5px 0; white-space: pre-wrap; }
    .answer { white-space: pre-wrap; font-size: 13px; line-height: 1.45; }
    .badge { display: inline-flex; align-items: center; border: 1px solid var(--line); border-radius: 999px; padding: 2px 7px; font-size: 12px; color: #475467; background: #fff; }
    .pill-run { color: var(--warn); border-color: #fed7aa; background: #fff7ed; }
    .pill-ok { color: var(--ok); border-color: #bbf7d0; background: #f0fdf4; }
    .pill-bad { color: var(--bad); border-color: #fecaca; background: #fef2f2; }
    .grade-block { border: 1px solid var(--line); border-radius: 6px; padding: 9px 11px; margin-bottom: 10px; background: #fff; }
    .grade-block .row1 { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 7px; }
    .grade-block .row1 .dataset { font-size: 11px; color: var(--muted); margin-left: auto; }
    .grade-block .expected { font-size: 12px; color: var(--muted); margin-bottom: 7px; }
    .grade-block .expected code { background: #f3f4f6; padding: 1px 4px; border-radius: 3px; font-size: 11px; }
    .grade-block .notes { font-size: 12px; color: #475467; border-top: 1px dashed var(--line); margin-top: 8px; padding-top: 6px; }
    .grade-block .notes summary { cursor: pointer; color: var(--muted); }
    .grade-block pre.answer-body { white-space: pre-wrap; font: inherit; font-size: 13px; line-height: 1.45; margin: 0; }
    .grade-block .err { color: var(--bad); font-size: 12px; margin-top: 5px; }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: none; border-bottom: 1px solid var(--line); }
      .workspace { grid-template-rows: auto 420px 300px; }
      .bottom { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>sciMAS evaluation dashboard</h1>
    <div class="meta" id="headerMeta">idle</div>
  </header>
  <main>
    <aside>
      <div class="section">
        <h2>Dataset</h2>
        <label>Dataset</label>
        <select id="dataset"></select>
        <div id="datasetFilters">
          <label>Root</label>
          <input id="root" placeholder="default dataset root" />
          <div class="row">
            <div><label>Split</label><input id="split" value="train" /></div>
            <div><label>Limit</label><input id="limit" type="number" value="1" min="1" /></div>
          </div>
          <label>Subject</label>
          <select id="subject"><option value="">All subjects</option></select>
          <label>Topic</label>
          <select id="topic"><option value="">All topics</option></select>
          <label>IDs</label>
          <input id="ids" placeholder="comma separated" />
          <label>Query</label>
          <input id="query" placeholder="question substring" />
        </div>
        <div class="actions">
          <button class="secondary" id="previewBtn">Preview</button>
        </div>
      </div>

      <div class="section" id="writeInSection" style="display:none;">
        <h2>当场手写</h2>
        <label>Scientific question</label>
        <textarea id="write_in_question" style="min-height:96px;"
                  placeholder="直接输入要跑的科学问题 / type the question to run"></textarea>
        <div class="badge" id="writeInHint" style="margin-top:8px;">输入后点击 Start run</div>
      </div>

      <div class="section">
        <h2>Run</h2>
        <label>Planner mode</label>
        <select id="planner_mode">
          <option value="python-dsl">Python DSL</option>
          <option value="legacy-json">Legacy JSON</option>
        </select>
        <label>Roles</label>
        <textarea id="roles" placeholder="optional, comma or space separated"></textarea>
        <label>Model</label>
        <input id="model" placeholder="default Claude CLI model" />
        <div class="row">
          <div><label>Timeout</label><input id="timeout" type="number" value="600" min="30" /></div>
          <div><label>Claude bin</label><input id="claude_bin" value="claude" /></div>
        </div>
        <label>Max reviewer attempts</label>
        <input id="max_review_attempts" type="number" value="3" min="1" max="20" />
        <label>MCP config</label>
        <input id="mcp_config" placeholder="default config/mcp.json" />
        <label class="checkrow"><input type="checkbox" id="no_mcp" /> Disable MCP</label>
        <label class="checkrow"><input type="checkbox" id="no_auto_synthesize" /> Disable auto synthesis</label>
        <label class="checkrow"><input type="checkbox" id="no_skill_routing" /> Disable skill routing</label>
        <label class="checkrow"><input type="checkbox" id="allow_role_tool_fallback" /> Allow role tool fallback</label>
        <div class="actions">
          <button id="startBtn">Start run</button>
          <button class="secondary" id="clearBtn">Clear view</button>
        </div>
      </div>

      <div class="section">
        <h2>Problems</h2>
        <div id="problemSummary" class="badge">not loaded</div>
        <div id="problemList" style="margin-top:8px;"></div>
      </div>
    </aside>
    <section class="workspace">
      <div class="band statusbar">
        <div class="metric"><div class="k">Job</div><div class="v" id="jobStatus">idle</div></div>
        <div class="metric"><div class="k">Progress</div><div class="v" id="progress">0 / 0</div></div>
        <div class="metric"><div class="k">Current problem</div><div class="v" id="currentProblem">none</div></div>
        <div class="metric"><div class="k">Output</div><div class="v" id="outputDir">none</div></div>
      </div>
      <div class="band graph-wrap">
        <svg id="graph" role="img" aria-label="MAS workflow graph"></svg>
        <div class="graph-legend" id="graphLegend"></div>
      </div>
      <div class="bottom">
        <div class="band">
          <div class="panel-title">Event log</div>
          <div class="scroll" id="log"></div>
        </div>
        <div class="band">
          <div class="panel-title">Final answers</div>
          <div class="scroll answer" id="answers"></div>
        </div>
      </div>
    </section>
  </main>

  <script>
    // Injected by the / route from the Python constant of the same name.
    const WRITE_IN = "@@WRITE_IN@@";
    const WRITE_IN_HINT = "输入后点击 Start run";
    const WRITE_IN_EMPTY_MSG = "请先输入科学问题 / type a question first";

    const state = {
      defaults: {},
      labels: {},
      jobId: null,
      source: null,
      nodes: new Map(),
      edges: [],
      lastPlan: null,
      currentProblemId: null,
      streamClosed: true,
      // Skeleton bookkeeping. bindings maps a runtime node id (node-1, ...) onto
      // the synthetic slot that predicted it; hasSkeleton is what tells the
      // runtime handlers whether slot-matching applies at all.
      bindings: new Map(),
      stageCursor: 0,
      maxStage: -1,
      slotSeq: 0,
      hasSkeleton: false,
    };

    const $ = (id) => document.getElementById(id);

    function escapeHtml(text) {
      return String(text ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function configFromForm() {
      return {
        dataset: $("dataset").value,
        write_in_question: $("write_in_question").value.trim(),
        root: $("root").value.trim(),
        split: $("split").value.trim() || "train",
        subject: $("subject").value.trim(),
        topic: $("topic").value.trim(),
        ids: $("ids").value.trim(),
        query: $("query").value.trim(),
        limit: Number($("limit").value || 1),
        planner_mode: $("planner_mode").value,
        roles: $("roles").value.trim(),
        model: $("model").value.trim(),
        timeout: Number($("timeout").value || 600),
        max_review_attempts: Number($("max_review_attempts").value || 3),
        claude_bin: $("claude_bin").value.trim() || "claude",
        mcp_config: $("mcp_config").value.trim(),
        no_mcp: $("no_mcp").checked,
        no_auto_synthesize: $("no_auto_synthesize").checked,
        no_skill_routing: $("no_skill_routing").checked,
        allow_role_tool_fallback: $("allow_role_tool_fallback").checked,
      };
    }

    function resetWriteInHint() {
      $("writeInHint").textContent = WRITE_IN_HINT;
      $("writeInHint").className = "badge";
    }

    function logLine(text, cls = "") {
      const div = document.createElement("div");
      div.className = "logline " + cls;
      div.textContent = text;
      $("log").prepend(div);
    }

    function fmtPct(x) {
      if (typeof x !== "number" || Number.isNaN(x)) return "—";
      return `${Math.round(x * 100)}%`;
    }

    function fmtScore(x) {
      if (typeof x !== "number" || Number.isNaN(x)) return "—";
      return x.toFixed(2);
    }

    function renderGradeBlock(result, host) {
      host.innerHTML = "";
      if (!result) {
        host.textContent = "(no result yet)";
        return;
      }

      const meta = result.problem_meta || {};
      const grade = result.grade || {};
      const status = result.status || "unknown";
      // A write-in run is never graded — no gold answer exists. Suppress the
      // correctness/score/coverage pills rather than showing a misleading
      // "✗ incorrect · score 0.00 · tool cov 0%".
      const isWriteIn = meta.dataset === WRITE_IN;

      const card = document.createElement("div");
      card.className = "grade-block";

      const row1 = document.createElement("div");
      row1.className = "row1";

      // Dataset label (right-aligned).
      const datasetTag = document.createElement("span");
      datasetTag.className = "badge";
      datasetTag.textContent = state.labels[meta.dataset] || meta.dataset || "—";
      const datasetCell = document.createElement("span");
      datasetCell.className = "dataset";
      datasetCell.appendChild(datasetTag);
      row1.appendChild(datasetCell);

      // Status pill.
      const statusPill = document.createElement("span");
      statusPill.className =
        status === "completed" ? "badge pill-ok"
        : status === "failed" ? "badge pill-bad"
        : "badge pill-run";
      statusPill.textContent = status;
      row1.insertBefore(statusPill, datasetCell);

      // Correctness pill.
      if (!isWriteIn && typeof grade.answer_correct === "boolean") {
        const ok = document.createElement("span");
        ok.className = "badge " + (grade.answer_correct ? "pill-ok" : "pill-bad");
        ok.textContent = grade.answer_correct ? "✓ correct" : "✗ incorrect";
        row1.insertBefore(ok, statusPill);
      }

      if (isWriteIn) {
        const ungraded = document.createElement("span");
        ungraded.className = "badge";
        ungraded.textContent = "未评分 / ungraded";
        ungraded.title = "ad-hoc question: no gold answer, grader skipped";
        row1.insertBefore(ungraded, statusPill);
      } else {
        // Score pill.
        const scorePill = document.createElement("span");
        scorePill.className = "badge";
        scorePill.textContent = `score ${fmtScore(grade.answer_score)}`;
        row1.insertBefore(scorePill, statusPill);

        // Tool coverage pill.
        const covPill = document.createElement("span");
        covPill.className = "badge";
        covPill.textContent = `tool cov ${fmtPct(grade.tool_coverage)}`;
        row1.insertBefore(covPill, scorePill);

        // Checklist score (RCB proxy / official judge).
        if (typeof grade.checklist_score === "number" && grade.checklist_score > 0) {
          const cs = document.createElement("span");
          cs.className = "badge";
          cs.textContent = `checklist ${fmtScore(grade.checklist_score)}`;
          row1.insertBefore(cs, scorePill);
        }
      }

      card.appendChild(row1);

      // Expected answer preview.
      if (meta.expected_answer) {
        const exp = document.createElement("div");
        exp.className = "expected";
        const expText = meta.expected_answer;
        const preview = expText.length > 220 ? expText.slice(0, 217) + "…" : expText;
        exp.innerHTML = `expected: <code>${escapeHtml(preview)}</code>`;
        card.appendChild(exp);
      }

      // Expected tools.
      if (Array.isArray(meta.expected_tools) && meta.expected_tools.length) {
        const expT = document.createElement("div");
        expT.className = "expected";
        const missed = grade.missing_tools || [];
        const called = new Set(grade.called_tools || []);
        const chips = meta.expected_tools.map(t => {
          const hit = called.has(t);
          const cls = hit ? "badge pill-ok" : "badge pill-bad";
          const mark = hit ? "✓" : "✗";
          return `<span class="${cls}" style="margin-right:4px;">${mark} ${escapeHtml(t)}</span>`;
        }).join("");
        expT.innerHTML = `expected tools: ${chips}` + (missed.length ? ` <span style="color:var(--muted);">missing ${missed.length}</span>` : "");
        card.appendChild(expT);
      }

      // Error / grading-error display.
      if (grade.error) {
        const err = document.createElement("div");
        err.className = "err";
        err.textContent = `⚠ ${grade.error}`;
        card.appendChild(err);
      } else if (status === "failed" && result.error) {
        const err = document.createElement("div");
        err.className = "err";
        err.textContent = `⚠ ${result.error}`;
        card.appendChild(err);
      }

      // Collapsed notes (grader explanations + missing molecules etc.).
      if (Array.isArray(grade.answer_notes) && grade.answer_notes.length) {
        const details = document.createElement("details");
        details.className = "notes";
        const summary = document.createElement("summary");
        // Not "grader notes" for a write-in: no grader ever ran, and the note
        // itself says so.
        summary.textContent = `${isWriteIn ? "notes" : "grader notes"} (${grade.answer_notes.length})`;
        details.appendChild(summary);
        const body = document.createElement("div");
        body.style.marginTop = "4px";
        for (const note of grade.answer_notes) {
          const line = document.createElement("div");
          line.textContent = "• " + note;
          body.appendChild(line);
        }
        details.appendChild(body);
        card.appendChild(details);
      }

      // Final-answer body.
      const ans = document.createElement("pre");
      ans.className = "answer-body";
      ans.textContent = result.final_answer || "(no final answer)";
      card.appendChild(ans);

      host.appendChild(card);
    }

    function resetGraph() {
      state.nodes.clear();
      state.edges = [];
      state.lastPlan = null;
      // Skeleton bookkeeping: slots predicted up front, the runtime-id -> slot
      // map, and how far along the structure the current run has got.
      state.bindings = new Map();
      state.stageCursor = 0;
      state.maxStage = -1;
      state.slotSeq = 0;
      state.hasSkeleton = false;
      renderGraph();
    }

    function short(text, n = 80) {
      text = String(text ?? "");
      return text.length > n ? text.slice(0, n - 3) + "..." : text;
    }

    function nodeLabel(node) {
      const role = node.role || "agent";
      const title = node.id || node.node_id || "";
      return `${title}\\n${role}\\n${short(node.instruction || node.objective || "", 34)}`;
    }

    // "planned" (legacy) and "pending" (skeleton slot) both render as the grey
    // waiting state. Neither may overwrite a state that has already moved on:
    // problem_finished re-ingests the plan, and without this guard that repainted
    // finished green nodes back to grey. done->running and failed->running stay
    // legal on purpose -- an execution-error retry really does re-run agents, and
    // a loop body really does go round again.
    const STATUS_RANK = { planned: 0, pending: 0, running: 1, done: 2, failed: 3 };

    function addNode(id, data = {}, opts = {}) {
      if (!id) return;
      const prior = state.nodes.get(id) || { id, status: "pending" };
      const next = { ...prior, ...data, id };
      if (!opts.force
          && (STATUS_RANK[data.status] ?? 0) === 0
          && (STATUS_RANK[prior.status] ?? 0) > 0) {
        next.status = prior.status;
      }
      state.nodes.set(id, next);
    }

    function addEdge(from, to) {
      if (!from || !to || from === to) return;
      if (!state.edges.some(e => e.from === from && e.to === to)) {
        state.edges.push({ from, to });
      }
    }

    // Runtime ids (node-1, node-2, ...) are minted at agent launch, so they are
    // not the ids the skeleton drew. Translate before drawing an edge.
    function resolveNodeId(runtimeId) {
      return state.bindings.get(runtimeId) || runtimeId;
    }

    function ingestPlan(plan) {
      state.lastPlan = plan;
      if (!plan) return;
      if (Array.isArray(plan.steps)) {
        for (const step of plan.steps) {
          addNode(step.id, { role: step.role, instruction: step.objective, status: "planned" });
          for (const dep of step.depends_on || []) addEdge(resolveNodeId(dep), step.id);
        }
      }
      renderGraph();
    }

    // Split the DSL validator's static trace into ordered stages (columns).
    //
    // Only agent_call / parallel_call carry agents; for / if / break / return are
    // ignored. This leans on one invariant: _validate_parallel_call validates its
    // children FIRST and appends its own record LAST (workflow_dsl.py:651-664), so
    // a parallel_call always immediately follows the agent_calls of its own
    // children -- the W records just before it are exactly its children.
    //
    // The head MUST be flushed before the group. Flushing all of `pending` at the
    // end instead is a real bug, not a style choice: for
    //     await agent(r0); await parallel(agent(r1), agent(r2))
    // the trace is [r0, r1, r2, parallel_call(2)], which comes out as
    // [[r1,r2],[r0]] -- the sequential agent and the parallel group swapped.
    function stageGroupTrace(trace) {
      const stages = [];
      let pending = [];
      for (const rec of trace || []) {
        if (rec.event === "agent_call") {
          pending.push(rec);
        } else if (rec.event === "parallel_call") {
          const w = Number(rec.width) || 0;
          const n = pending.length;
          const head = w <= n ? pending.slice(0, n - w) : [];
          const grp = w <= n ? pending.slice(n - w) : pending.slice();
          for (const h of head) stages.push([h]);
          pending = [];
          if (grp.length) stages.push(grp);
        }
      }
      for (const h of pending) stages.push([h]);
      return stages;
    }

    // Draw the whole MAS up front, the moment the plan exists.
    //
    // In the default python_dsl mode plan.steps is EMPTY here: it is appended to
    // only during execution, and node ids are minted at agent launch. The sole
    // up-front structure is workflow_static_trace. Legacy mode still goes through
    // plan.steps exactly as before.
    //
    // Slots get synthetic ids (s1, s2, ...) and an explicit `stage`, and NO edges:
    // source order is not dependency order, and a runtime received_from is
    // legitimately empty for an agent whose input ignores the previous output.
    // Wiring is drawn only once it is a fact.
    function buildSkeleton(plan) {
      if (!plan || plan.mode !== "python_dsl") return false;
      const stages = stageGroupTrace(plan.workflow_static_trace);
      if (!stages.length) return false;
      state.hasSkeleton = true;
      state.bindings = new Map();
      state.stageCursor = 0;
      state.slotSeq = 0;
      state.maxStage = stages.length - 1;
      stages.forEach((group, stage) => {
        for (const rec of group) {
          addNode(`s${++state.slotSeq}`, {
            role: rec.role,
            instruction: rec.instruction,
            status: "pending",
            stage,
            synthetic: true,
          }, { force: true });
        }
      });
      return true;
    }

    // Match a runtime agent back to the slot that predicted it.
    //
    // `instruction` is the join key. The DSL forces agent(...)'s instruction to
    // be a string literal and records that exact literal in the static trace,
    // and the runtime passes the same literal through to agent_started /
    // agent_finished -- byte-identical on both sides. `role` would NOT be safe:
    // the trace holds the raw literal while the runtime emits normalize_role().
    function bindRuntimeNode(nodeId, instruction) {
      if (!state.hasSkeleton) return null;
      const cands = Array.from(state.nodes.values())
        .filter(n => n.synthetic && n.instruction === instruction);
      if (!cands.length) return null;
      // Prefer slots at or after the furthest stage reached so far: an identical
      // instruction sitting in a branch that never ran must not absorb a runtime
      // agent that actually belongs to a later slot.
      const ahead = cands.filter(n => n.stage >= state.stageCursor);
      const pool = ahead.length ? ahead : cands;
      const free = pool.filter(n =>
        (n.status === "pending" || n.status === "running") &&
        (n.boundTo === undefined || n.boundTo === nodeId));
      // A free slot is the normal case. A loop re-run finds none free and falls
      // through to reusing a slot, which is what keeps a range(N) body on ONE box
      // instead of inventing N of them.
      const slot = free.length
        ? free[0]
        : pool.slice().sort((a, b) => a.stage - b.stage).pop();
      if (!slot) return null;
      state.bindings.set(nodeId, slot.id);
      state.stageCursor = slot.stage;
      addNode(slot.id, { boundTo: nodeId });
      return slot.id;
    }

    function addRuntimeNode(nodeId, data) {
      const slot = bindRuntimeNode(nodeId, data.instruction);
      if (slot) {
        // Paint the predicted slot with the runtime agent's state.
        addNode(slot, data);
        return slot;
      }
      // No skeleton at all (legacy mode / empty trace): today's behaviour exactly.
      if (!state.hasSkeleton) {
        addNode(nodeId, data);
        return nodeId;
      }
      // A skeleton exists but did not predict this agent -- the workflow was
      // rewritten after an execution error. Give it its own column at the end;
      // an undefined stage would collapse it into column 0, on top of the
      // skeleton, which is the worst possible rendering.
      const known = state.nodes.has(nodeId);
      addNode(nodeId, known ? data : { ...data, stage: ++state.maxStage });
      state.bindings.set(nodeId, nodeId);
      return nodeId;
    }

    // Nothing in the frontend ever set status:"failed" before, so the red state
    // was unreachable. The server does NOT emit `problem_failed` at all -- a
    // failed problem arrives as problem_finished with result.status === "failed"
    // (web_dashboard.py:434) -- so that and workflow_execution_failed are the
    // real hooks.
    function failRunningNodes(reason) {
      let n = 0;
      for (const node of Array.from(state.nodes.values())) {
        if (node.status === "running") {
          addNode(node.id, { status: "failed" }, { force: true });
          n++;
        }
      }
      if (n) logLine(`${n} agent(s) left running marked failed${reason ? `: ${reason}` : ""}`, "pill-bad");
      renderGraph();
      return n;
    }

    // The reviewer replaced the workflow source after it crashed, so the skeleton
    // drawn at planner_finished no longer describes what will run. Put every
    // predicted slot back to grey (keep the boxes so the picture stays stable) and
    // let re-binding repaint as the new run proceeds.
    function resetSkeletonToPending() {
      for (const node of Array.from(state.nodes.values())) {
        if (node.synthetic) {
          addNode(node.id, { status: "pending", boundTo: undefined }, { force: true });
        }
      }
      state.bindings = new Map();
      state.stageCursor = 0;
      renderGraph();
    }

    // Reconcile the plan that rides along with problem_finished.
    //
    // In python_dsl mode report.plan.steps is FULLY POPULATED by this point (the
    // runtime appends to the very same plan object as it executes), and its ids
    // are the runtime node-N ids. Ingesting that on top of the skeleton would
    // paint a second, duplicate set of boxes wired to nothing -- which today is
    // masked only because both sides happen to use node-N as the key.
    function reconcilePlan(plan) {
      if (!plan) return;
      if (plan.mode === "python_dsl") return;
      ingestPlan(plan);
    }

    function ingestMasEvent(payload) {
      const event = payload.event;
      if (event === "planner_started") {
        resetGraph();
        logLine("planner started");
      } else if (event === "planner_finished") {
        // python_dsl draws its own skeleton; anything else is the legacy path,
        // which already carries real step ids and depends_on edges.
        if (!buildSkeleton(payload.plan)) ingestPlan(payload.plan);
        renderGraph();
        logLine(`planner finished: ${state.nodes.size} agent(s) in the planned structure`);
      } else if (event === "agent_started") {
        const id = addRuntimeNode(payload.node_id, {
          role: payload.role,
          instruction: payload.instruction,
          status: "running",
        });
        for (const dep of payload.received_from || []) addEdge(resolveNodeId(dep), id);
        renderGraph();
        logLine(`agent started: ${payload.node_id} / ${payload.role}`);
      } else if (event === "agent_finished") {
        const id = addRuntimeNode(payload.node_id, {
          role: payload.role,
          instruction: payload.instruction,
          status: "done",
          public_output: payload.public_output,
        });
        for (const dep of payload.received_from || []) addEdge(resolveNodeId(dep), id);
        renderGraph();
        logLine(`agent finished: ${payload.node_id} / ${payload.role}`);
      } else if (event === "parallel_finished") {
        logLine(`parallel finished: ${payload.parallel_id}`);
      } else if (event === "workflow_execution_failed") {
        failRunningNodes(String(payload.error || "").slice(0, 160));
        // The retry runs the repaired workflow through a fresh runtime whose node
        // counter restarts at 0, so the old bindings are worthless.
        state.bindings = new Map();
        state.stageCursor = 0;
      } else if (event === "workflow_fixed_after_execution_error") {
        resetSkeletonToPending();
        logLine("workflow was rewritten after an execution error; the drawn structure may be stale", "pill-bad");
      } else if (event === "run_finished") {
        logLine("run finished");
      } else {
        // Anything unrecognized still reaches the log. Without this the
        // reviewer/validation events were dropped silently, so a run that
        // died during planning showed nothing about why.
        logLine(`${event}: ${JSON.stringify(payload).slice(0, 400)}`);
      }
    }

    // The single definition of the four states, shared by the drawn nodes and the
    // legend swatches so they cannot drift apart.
    function fillFor(status) {
      if (status === "running") return "var(--node-run)";
      if (status === "done") return "var(--node-done)";
      if (status === "failed") return "var(--node-fail)";
      return "var(--node)";                    // pending / planned / unknown
    }

    function strokeFor(status) {
      if (status === "running") return "var(--node-run-stroke)";
      if (status === "done") return "var(--node-done-stroke)";
      if (status === "failed") return "var(--node-fail-stroke)";
      return "var(--node-stroke)";
    }

    const LEGEND_STATES = [
      ["pending", "等待中 / pending"],
      ["running", "运行中 / running"],
      ["done", "已完成 / done"],
      ["failed", "失败 / failed"],
    ];

    // Colour is the only state carrier on the graph now, so a legend is not
    // decoration. Swatches read fillFor/strokeFor rather than repeating hexes.
    function renderLegend() {
      const host = $("graphLegend");
      if (!host) return;
      host.innerHTML = "";
      for (const [status, label] of LEGEND_STATES) {
        const span = document.createElement("span");
        span.className = "lg";
        const swatch = document.createElement("i");
        swatch.style.background = fillFor(status);
        swatch.style.borderColor = strokeFor(status);
        span.appendChild(swatch);
        span.appendChild(document.createTextNode(label));
        host.appendChild(span);
      }
      const note = document.createElement("span");
      note.className = "lg note";
      // The static trace is flat -- it records no block extents -- so mutually
      // exclusive branches fall into different columns. Say so rather than let
      // the columns imply a single pipeline.
      note.textContent = "列 = 静态书写顺序，不代表分支互斥";
      host.appendChild(note);
    }

    function renderGraph() {
      const svg = $("graph");
      const w = svg.clientWidth || 800;
      const h = svg.clientHeight || 420;
      svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
      svg.innerHTML = "";
      const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
      defs.innerHTML = `<marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
        <path d="M0,0 L10,4 L0,8 Z" fill="#667085"></path></marker>`;
      svg.appendChild(defs);
      const nodes = Array.from(state.nodes.values());
      if (!nodes.length) {
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", w / 2);
        text.setAttribute("y", h / 2);
        text.setAttribute("text-anchor", "middle");
        text.setAttribute("fill", "#667085");
        text.setAttribute("font-size", "14");
        text.textContent = "Workflow graph will appear after the planner runs";
        svg.appendChild(text);
        return;
      }

      const levels = new Map();
      function levelOf(id, seen = new Set()) {
        if (levels.has(id)) return levels.get(id);
        // A skeleton slot carries its column explicitly. Nothing else does, so
        // Number.isInteger(undefined) is false and the legacy path below is
        // untouched. This also makes the layout immune to cycles among synthetic
        // nodes, which have no edges to break a loop with.
        const self = state.nodes.get(id);
        if (self && Number.isInteger(self.stage)) {
          levels.set(id, self.stage);
          return self.stage;
        }
        if (seen.has(id)) return 0;
        seen.add(id);
        const incoming = state.edges.filter(e => e.to === id).map(e => e.from);
        const lvl = incoming.length ? 1 + Math.max(...incoming.map(x => levelOf(x, seen))) : 0;
        levels.set(id, lvl);
        return lvl;
      }
      for (const n of nodes) levelOf(n.id);
      const byLevel = new Map();
      for (const n of nodes) {
        const lvl = levels.get(n.id) || 0;
        if (!byLevel.has(lvl)) byLevel.set(lvl, []);
        byLevel.get(lvl).push(n);
      }
      const maxLevel = Math.max(...Array.from(byLevel.keys()));
      // Derive the box width from the column pitch so columns can never overlap.
      // The old clamp bottomed out at 145px while the pitch shrinks with the
      // column count, so past ~(w-72)/145 columns they collided -- w=900 with 6
      // columns was already 7px overlapping, and a skeleton has many columns.
      const stepX = (w - 72) / Math.max(1, maxLevel + 1);
      const boxW = Math.max(96, Math.min(210, stepX - 16));
      const boxH = 76;
      const pos = new Map();
      for (const [lvl, items] of byLevel.entries()) {
        items.forEach((n, i) => {
          const x = 36 + lvl * stepX;
          const spacing = h / (items.length + 1);
          const y = spacing * (i + 1) - boxH / 2;
          pos.set(n.id, { x, y });
        });
      }

      for (const edge of state.edges) {
        const a = pos.get(edge.from), b = pos.get(edge.to);
        if (!a || !b) continue;
        const line = document.createElementNS("http://www.w3.org/2000/svg", "path");
        const x1 = a.x + boxW, y1 = a.y + boxH / 2, x2 = b.x, y2 = b.y + boxH / 2;
        const mid = Math.max(20, (x2 - x1) / 2);
        line.setAttribute("d", `M${x1},${y1} C${x1 + mid},${y1} ${x2 - mid},${y2} ${x2},${y2}`);
        line.setAttribute("fill", "none");
        line.setAttribute("stroke", "#667085");
        line.setAttribute("stroke-width", "1.6");
        line.setAttribute("marker-end", "url(#arrow)");
        svg.appendChild(line);
      }

      for (const n of nodes) {
        const p = pos.get(n.id);
        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", p.x);
        rect.setAttribute("y", p.y);
        rect.setAttribute("width", boxW);
        rect.setAttribute("height", boxH);
        rect.setAttribute("rx", "8");
        rect.setAttribute("fill", fillFor(n.status));
        rect.setAttribute("stroke", strokeFor(n.status));
        g.appendChild(rect);
        const lines = nodeLabel(n).split("\\n");
        // Drop the instruction line when the column is too narrow to read it.
        (boxW < 130 ? lines.slice(0, 2) : lines).forEach((line, i) => {
          const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
          t.setAttribute("x", p.x + 12);
          t.setAttribute("y", p.y + 19 + i * 17);
          t.setAttribute("font-size", i === 0 ? "12" : "11");
          t.setAttribute("font-weight", i === 0 ? "700" : "500");
          t.setAttribute("fill", i === 2 ? "#667085" : "#1d2433");
          t.textContent = line;
          g.appendChild(t);
        });
        svg.appendChild(g);
      }
    }

    async function loadDatasets() {
      const res = await fetch("/api/datasets");
      const data = await res.json();
      state.defaults = data.defaults || {};
      // Must be set before the options are built below, which read it.
      state.labels = data.labels || {};
      $("dataset").innerHTML = data.datasets
        .map(d => `<option value="${escapeHtml(d)}">${escapeHtml(state.labels[d] || d)}</option>`)
        .join("");
      $("root").value = state.defaults[$("dataset").value] || "";
      $("dataset").addEventListener("change", async () => {
        const isWriteIn = $("dataset").value === WRITE_IN;
        $("writeInSection").style.display = isWriteIn ? "" : "none";
        $("datasetFilters").style.display = isWriteIn ? "none" : "";
        resetWriteInHint();
        if (isWriteIn) $("write_in_question").focus();
        // state.defaults deliberately has no entry for the sentinel, so this
        // already clears the root input when write-in is selected.
        $("root").value = state.defaults[$("dataset").value] || "";
        $("subject").value = "";
        $("topic").value = "";
        await loadFilterOptions({ resetTopic: true });
        await previewProblems();
      });
      $("root").addEventListener("change", async () => {
        $("subject").value = "";
        $("topic").value = "";
        await loadFilterOptions({ resetTopic: true });
      });
      $("split").addEventListener("change", async () => {
        $("subject").value = "";
        $("topic").value = "";
        await loadFilterOptions({ resetTopic: true });
      });
      $("subject").addEventListener("change", async () => {
        await loadFilterOptions({ resetTopic: true });
        await previewProblems();
      });
      $("topic").addEventListener("change", previewProblems);
    }

    async function loadFilterOptions({ resetTopic = false } = {}) {
      const currentSubject = $("subject").value;
      const currentTopic = resetTopic ? "" : $("topic").value;
      const res = await fetch("/api/filter-options", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(configFromForm()),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "failed to load filter options");

      const subjectOptions = [`<option value="">All subjects</option>`].concat(
        (data.subjects || []).map(s => `<option value="${escapeHtml(s)}">${escapeHtml(s)}</option>`)
      );
      $("subject").innerHTML = subjectOptions.join("");
      $("subject").value = (data.subjects || []).includes(currentSubject) ? currentSubject : "";

      const topicOptions = [`<option value="">All topics</option>`].concat(
        (data.topics || []).map(t => `<option value="${escapeHtml(t)}">${escapeHtml(t)}</option>`)
      );
      $("topic").innerHTML = topicOptions.join("");
      $("topic").value = (data.topics || []).includes(currentTopic) ? currentTopic : "";
      logLine(`filters loaded: ${data.subjects.length} subjects, ${data.topics.length} topics`);
    }

    async function previewProblems() {
      $("previewBtn").disabled = true;
      try {
        const res = await fetch("/api/problems", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(configFromForm()),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "preview failed");
        $("problemSummary").textContent = `${data.count} matching problems`;
        $("problemList").innerHTML = data.problems.map(p => `
          <div class="problem">
            <strong>${escapeHtml(p.index)}. ${escapeHtml(p.id)} · ${escapeHtml(p.subject || "")} / ${escapeHtml(p.topic || "")}</strong>
            <span>${escapeHtml(p.question_preview)}</span>
          </div>
        `).join("");
        logLine(`preview loaded: ${data.count} problems`);
      } catch (err) {
        logLine(`preview error: ${err.message}`, "pill-bad");
      } finally {
        $("previewBtn").disabled = false;
      }
    }

    async function startJob() {
      // Checked before the button is disabled so an empty write-in leaves the
      // button usable rather than stuck in its disabled state.
      if ($("dataset").value === WRITE_IN && !$("write_in_question").value.trim()) {
        $("writeInHint").textContent = WRITE_IN_EMPTY_MSG;
        $("writeInHint").className = "badge pill-bad";
        logLine("write-in: empty question, run not started", "pill-bad");
        $("write_in_question").focus();
        return;
      }
      $("startBtn").disabled = true;
      resetGraph();
      $("answers").textContent = "";
      try {
        const res = await fetch("/api/jobs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(configFromForm()),
        });
        const job = await res.json();
        if (!res.ok) throw new Error(job.error || "failed to start job");
        state.jobId = job.id;
        $("jobStatus").textContent = job.status;
        $("headerMeta").textContent = `job ${job.id.slice(0, 8)}`;
        subscribe(job.id);
      } catch (err) {
        logLine(`start error: ${err.message}`, "pill-bad");
        $("startBtn").disabled = false;
      }
    }

    function subscribe(jobId) {
      if (state.source) state.source.close();
      state.streamClosed = false;
      state.source = new EventSource(`/api/jobs/${jobId}/events`);
      state.source.onmessage = (message) => {
        const event = JSON.parse(message.data);
        handleEvent(event);
      };
      state.source.onerror = () => {
        if (!state.streamClosed) {
          logLine("event stream interrupted");
        }
        $("startBtn").disabled = false;
      };
    }

    function closeEventStream() {
      state.streamClosed = true;
      if (state.source) {
        state.source.close();
        state.source = null;
      }
    }

    function handleEvent(event) {
      if (event.event === "job_started") {
        $("jobStatus").textContent = "running";
        logLine("job started");
      } else if (event.event === "problem_started") {
        state.currentProblemId = event.problem.id;
        resetGraph();
        $("currentProblem").textContent = `${event.problem.index}. ${event.problem.id}`;
        $("progress").textContent = `${event.completed} / ${event.total}`;
        logLine(`problem started: ${event.problem.id}`);
      } else if (event.event === "mas_event") {
        ingestMasEvent(event.payload);
      } else if (event.event === "problem_finished") {
        $("progress").textContent = `${event.completed} / ${event.total}`;
        $("outputDir").textContent = event.result.output_dir || "saved";
        renderGradeBlock(event.result, $("answers"));
        // A failed problem is reported HERE, not through problem_failed -- the
        // server never emits that one (grepping the repo finds only this
        // frontend branch). So this is the hook that makes the red state real.
        if (event.result.status === "failed") {
          failRunningNodes(String(event.result.error || "").slice(0, 160));
        }
        reconcilePlan(event.result.plan);
        logLine(`problem finished: ${event.problem.id}`);
      } else if (event.event === "problem_failed") {
        failRunningNodes(String(event.error || "").slice(0, 160));
        logLine(`problem failed: ${event.error}`, "pill-bad");
      } else if (event.event === "job_finished") {
        $("jobStatus").textContent = "completed";
        $("headerMeta").textContent = "completed";
        $("startBtn").disabled = false;
        logLine("job finished");
        closeEventStream();
      } else if (event.event === "job_failed") {
        $("jobStatus").textContent = "failed";
        $("startBtn").disabled = false;
        failRunningNodes(String(event.error || "").slice(0, 160));
        logLine(`job failed: ${event.error}`, "pill-bad");
        closeEventStream();
      }
    }

    $("previewBtn").addEventListener("click", previewProblems);
    $("startBtn").addEventListener("click", startJob);
    $("write_in_question").addEventListener("input", resetWriteInHint);
    $("clearBtn").addEventListener("click", () => {
      $("log").textContent = "";
      $("answers").textContent = "";
      $("jobStatus").textContent = "idle";
      $("progress").textContent = "0 / 0";
      $("currentProblem").textContent = "none";
      $("outputDir").textContent = "none";
      resetGraph();
    });
    window.addEventListener("resize", renderGraph);
    renderLegend();

    loadDatasets()
      .then(() => loadFilterOptions({ resetTopic: true }))
      .then(previewProblems)
      .catch(err => logLine(err.message, "pill-bad"));
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the sciMAS evaluation dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
