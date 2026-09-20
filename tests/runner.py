"""End-to-end runner: invoke sciMAS on benchmark problems and
collect a graded result.

Usage (from inside the sciMAS/ directory):

    python tests/runner.py --subject Chemistry --limit 3
    python tests/runner.py --ids 12,17
    python tests/runner.py --dataset ResearchClawBench --limit 1
    python tests/runner.py --topic "Analytical Chemistry" --limit 5 \\
        --out tests/results/20260909-baseline/

Output:
    tests/results/<dataset>/<run-id>/
        run.json           — config + aggregate stats
        per-problem/
            <problem_id>.json   — full grading result incl. tool calls
        summary.md         — human-readable table

How it works:
1. Load + filter problems via ``dataset.load_dataset``.
2. For each problem, build a fresh ``SciMASOrchestrator`` whose
   ``ClaudeRunner`` defaults to plain text output. Pass
   ``--capture-tool-calls`` to use ``stream-json`` and capture the
   model's actual MCP tool invocations.
3. Run the problem, parse the run report, run the grader, persist.
4. After all problems, aggregate into ``run.json`` and ``summary.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

# This script lives in tests/ but imports from the parent dir
# (claude_runner, orchestrator). When invoked as `python tests/runner.py`,
# sys.path[0] is tests/ — we need to also expose the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claude_runner import ClaudeRunner
from orchestrator import (
    ROLE_ALLOWED_CHEMISTRY_SERVERS,
    ROLE_ALIASES,
    SciMASOrchestrator,
    allowed_tools_for_role,
)
from dataset import (
    DRUG_DISCOVERY_BENCH,
    MADD,
    SCIAGENTGYM,
    SCIPREDICT,
    SMDD_BENCH,
    BIOMNI_EVAL1,
    Problem,
    dataset_dir_name,
    default_dataset_root,
    filter_problems,
    format_problem_for_solver,
    load_dataset,
    normalize_dataset_name,
    stage_problem_images,
)
from grader import GradingResult, ToolCall, grade_problem, set_judge_cache_dir


@dataclass
class RunConfig:
    dataset_name: str = SCIAGENTGYM
    dataset_root: str = ""
    label: str = "baseline"
    subject: Optional[str] = None
    topic: Optional[str] = None
    ids: Optional[list[str]] = None
    query: Optional[str] = None
    limit: Optional[int] = None
    roles: list[str] = field(default_factory=list)
    mcp_config: Optional[str] = None
    claude_bin: str = "claude"
    model: Optional[str] = None
    timeout: float = 600.0
    capture_tool_calls: bool = False
    planner_mode: str = "python_dsl"
    allow_role_tool_fallback: bool = False
    # ResearchClawBench official-judge selection: "auto" (default; reads env),
    # "force" (require env), or "off" (skip judge even when env is set).
    research_claw_judge: str = "auto"
    smdd_official_eval: bool = False
    smdd_docker_image: str = "smdd-evals"
    smdd_eval_timeout: float = 3600.0
    smdd_gpu_id: Optional[str] = None
    notes: str = ""


def _all_registered_tools() -> set[str]:
    """Return every MCP tool short name registered by sciMAS."""
    from orchestrator import ROLE_ALLOWED_BY_DISCIPLINE

    out: set[str] = set()
    for _name, _allow_map, tool_map in ROLE_ALLOWED_BY_DISCIPLINE:
        for tool_list in tool_map.values():
            out.update(tool_list)
    return out


def _parse_ids(ids_text: Optional[str]) -> Optional[list[str]]:
    if not ids_text:
        return None
    return [x.strip() for x in ids_text.split(",") if x.strip()]


def _problem_file_stem(problem: Problem) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(problem.id)).strip("._")
    return stem or "problem"


class _StreamJsonRunner(ClaudeRunner):
    """A thin ClaudeRunner subclass that asks claude -p to emit
    stream-json so we capture each MCP tool call. The default runner in
    the orchestrator still uses ``json`` mode — we swap it out
    post-construction, before the orchestrator fires any ``run()``
    calls, by replacing ``orchestrator.runner``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tool_calls: list[dict[str, Any]] = []

    def run(self, prompt, stdin_text=None, session_id=None, output_format="json",
            allowed_tools=None):
        result = super().run(
            prompt=prompt,
            stdin_text=stdin_text,
            session_id=session_id,
            output_format="stream-json" if output_format == "json" else output_format,
            allowed_tools=allowed_tools,
        )
        # Aggregate tool calls from every claude invocation (planner +
        # worker steps + synth).
        if result.tool_calls:
            self._tool_calls.extend(result.tool_calls)
        return result


class _TextRunner(ClaudeRunner):
    """Use plain text output for models/providers whose stream-json path is
    not reliable. This does not capture tool calls, but it lets the batch
    evaluate answer quality and preserves sciMAS run reports.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tool_calls: list[dict[str, Any]] = []

    def run(self, prompt, stdin_text=None, session_id=None, output_format="json",
            allowed_tools=None):
        return super().run(
            prompt=prompt,
            stdin_text=stdin_text,
            session_id=session_id,
            output_format="",
            allowed_tools=allowed_tools,
        )


def _run_single(
    problem: Problem,
    *,
    cfg: RunConfig,
    orchestrator_factory,
    smdd_eval_root: Optional[Path] = None,
    asset_root: Optional[Path] = None,
) -> tuple[GradingResult, Optional[str]]:
    """Run one problem through sciMAS and grade. Returns (result,
    output_dir or None on failure).

    ``asset_root`` is where the problem's figures are staged before the run.
    Without it the figure copies are skipped and the solver is handed the
    dataset's own (absolute) paths.
    """
    runner_cls = _StreamJsonRunner if cfg.capture_tool_calls else _TextRunner
    runner = runner_cls(
        binary=cfg.claude_bin,
        model=cfg.model,
        mcp_config_path=cfg.mcp_config,
        allowed_tools=None,  # orchestrator sets per-step
        permission_mode="bypassPermissions" if cfg.capture_tool_calls else "",
        dangerously_skip_permissions=cfg.capture_tool_calls,
        timeout=cfg.timeout,
    )
    orch = orchestrator_factory(runner)
    # Stage the question's figures inside the batch output directory, so the
    # agents' Read tool reaches them without any out-of-tree permission and
    # the run output keeps its own copy of what was answered.
    problem.image_paths_local = (
        stage_problem_images(problem, asset_root / _problem_file_stem(problem))
        if asset_root
        else []
    )
    try:
        report = orch.run(
            problem=format_problem_for_solver(problem), roles=cfg.roles or None
        )
    except Exception as exc:
        # Capture and return a graded "error" result so the batch keeps
        # running.
        return grade_problem(
            problem,
            predicted_answer="",
            tool_calls=[],
            available_tools=_all_registered_tools(),
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        ), None

    tool_calls = [
        ToolCall.from_full(c["tool"], c.get("arguments", {}))
        for c in runner._tool_calls
    ]
    final_answer = report.final_answer
    _maybe_run_smdd_official_eval(
        problem,
        final_answer=final_answer,
        cfg=cfg,
        smdd_eval_root=smdd_eval_root,
    )
    cost = sum((r.cost_usd or 0.0) for r in report.runs)
    duration = sum((r.duration_ms or 0) for r in report.runs)
    grade = grade_problem(
        problem,
        predicted_answer=final_answer,
        tool_calls=tool_calls,
        available_tools=_all_registered_tools(),
        cost_usd=cost,
        duration_ms=duration,
    )
    return grade, report.output_dir


def _maybe_run_smdd_official_eval(
    problem: Problem,
    *,
    final_answer: str,
    cfg: RunConfig,
    smdd_eval_root: Optional[Path],
) -> None:
    """Attach official SMDDBench evaluator output to problem.task_info."""
    if not cfg.smdd_official_eval or problem.dataset_name != SMDD_BENCH:
        return

    task_info = problem.task_info
    if not isinstance(task_info, dict):
        task_info = {}
        problem.task_info = task_info

    task_dir_text = str(task_info.get("smdd_task_dir") or "").strip()
    task_id = str(task_info.get("smdd_task_id") or problem.id)
    root = (smdd_eval_root or Path.cwd() / "smdd-official").resolve()
    agent_outputs_root = root / "agent_outputs"
    results_root = root / "results"

    try:
        from scripts.smdd_official_eval import materialize_and_evaluate_answer

        if not task_dir_text:
            raise ValueError("problem.task_info['smdd_task_dir'] is missing")
        result = materialize_and_evaluate_answer(
            Path(task_dir_text),
            final_answer,
            agent_outputs_root,
            results_root,
            docker_image=cfg.smdd_docker_image,
            timeout=cfg.smdd_eval_timeout,
            gpu_id=cfg.smdd_gpu_id,
        )
        task_id = str(result.get("task_id") or task_id)
    except Exception as exc:
        result = {
            "task_id": task_id,
            "status": "errored",
            "steps": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
        result_dir = results_root / task_id
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    result_path = results_root / task_id / "result.json"
    candidates = task_info.get("smdd_official_result_candidates")
    if not isinstance(candidates, list):
        candidates = []
    result_path_text = str(result_path.resolve())
    if result_path_text not in candidates:
        candidates.insert(0, result_path_text)
    task_info["smdd_official_result_candidates"] = candidates
    task_info["smdd_official_result"] = result
    task_info["smdd_official_result_path"] = result_path_text
    task_info["smdd_official_agent_outputs_root"] = str(agent_outputs_root.resolve())


def run_batch(
    problems: list[Problem],
    *,
    cfg: RunConfig,
    output_root: Path,
) -> dict[str, Any]:
    """Run every problem in order. Returns the run-level summary dict
    (also written to ``run.json``).
    """
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    per_problem_dir = output_root / "per-problem"
    per_problem_dir.mkdir(exist_ok=True)

    # Pin the ResearchClawBench official-judge cache to this run. The grader
    # reads this via get_judge_cache_dir(); files land under
    # ``<output_root>/judge_cache/<task>__<hash>__<model>__v1.json``.
    judge_cache_dir = output_root / "judge_cache"
    if cfg.research_claw_judge == "off":
        # Force the proxy path even if the operator has RESEARCH_CLAW_JUDGE_API_KEY set.
        os.environ["RESEARCH_CLAW_JUDGE_DISABLED"] = "1"
        set_judge_cache_dir(None)
    else:
        set_judge_cache_dir(str(judge_cache_dir))

    results: list[GradingResult] = []

    def _factory_with_runner(runner: ClaudeRunner) -> SciMASOrchestrator:
        orch = SciMASOrchestrator(
            claude_bin=cfg.claude_bin,
            model=cfg.model,
            mcp_config_path=cfg.mcp_config,
            output_dir=output_root / "scimas-runs",
            strict_skill_routing=not cfg.allow_role_tool_fallback,
            permission_mode="bypassPermissions",
            dangerously_skip_permissions=True,
            timeout=cfg.timeout,
            planner_mode=cfg.planner_mode,
        )
        # Patch the runner so tests can choose plain text (default) or
        # stream-json tool-call capture without changing orchestrator code.
        orch.runner = runner
        return orch

    for prob in problems:
        print(f"\n[{len(results)+1}/{len(problems)}] id={prob.id}  "
              f"{prob.subject}/{prob.topic}  q='{prob.question[:60]}…'")
        t0 = time.monotonic()
        grade, sci_dir = _run_single(
            prob,
            cfg=cfg,
            orchestrator_factory=_factory_with_runner,
            smdd_eval_root=output_root / "smdd-official",
            asset_root=output_root / "figure-assets",
        )
        elapsed = time.monotonic() - t0
        results.append(grade)
        # Persist per-problem result.
        out_file = per_problem_dir / f"{_problem_file_stem(prob)}.json"
        payload = {
            "problem": prob.to_dict(),
            "grade": grade.to_dict(),
            "scimas_run_dir": sci_dir,
            "elapsed_s": round(elapsed, 2),
        }
        out_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        status = "✓" if grade.answer_correct else "✗"
        print(
            f"   {status} answer_score={grade.answer_score:.2f}  "
            f"tool_cov={grade.tool_coverage:.0%}  "
            f"cost=${grade.cost_usd:.4f}  t={grade.duration_ms/1000:.1f}s"
        )

    summary = _build_summary(results, cfg)
    (output_root / "run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_summary_md(output_root / "summary.md", summary, results)
    return summary


def _build_summary(results: list[GradingResult], cfg: RunConfig) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {
            "config": asdict(cfg),
            "n_problems": 0,
            "answer_accuracy": 0.0,
            "answer_score_avg": 0.0,
            "tool_coverage_avg": 0.0,
            "total_cost_usd": 0.0,
            "total_duration_ms": 0,
            "by_topic": {},
            "by_subject": {},
            "results": [],
        }
    correct = sum(1 for r in results if r.answer_correct)
    avg_score = sum(r.answer_score for r in results) / n
    avg_cov = sum(r.tool_coverage for r in results) / n
    madd_fa_avg = None
    ddb_score_100 = None
    scipredict_score_100 = None
    smdd_score_100 = None
    biomni_eval1_score_100 = None
    if cfg.dataset_name == MADD:
        madd_fa_avg = sum(r.answer_score * r.tool_coverage for r in results) / n
    if cfg.dataset_name == DRUG_DISCOVERY_BENCH:
        ddb_score_100 = sum(r.answer_score for r in results) / n * 100.0
    if cfg.dataset_name == SCIPREDICT:
        scipredict_score_100 = sum(r.answer_score for r in results) / n * 100.0
    if cfg.dataset_name == SMDD_BENCH:
        smdd_score_100 = sum(r.answer_score for r in results) / n * 100.0
    if cfg.dataset_name == BIOMNI_EVAL1:
        biomni_eval1_score_100 = sum(r.answer_score for r in results) / n * 100.0
    total_cost = sum(r.cost_usd for r in results)
    total_ms = sum(r.duration_ms for r in results)
    by_topic: dict[str, dict[str, Any]] = {}
    by_subject: dict[str, dict[str, Any]] = {}
    for r in results:
        by_topic.setdefault(r.topic, {"n": 0, "correct": 0, "avg_score": 0.0, "avg_cov": 0.0})
        by_topic[r.topic]["n"] += 1
        by_topic[r.topic]["correct"] += int(r.answer_correct)
        by_topic[r.topic]["avg_score"] += r.answer_score
        by_topic[r.topic]["avg_cov"] += r.tool_coverage
        by_subject.setdefault(r.subject, {"n": 0, "correct": 0, "avg_score": 0.0, "avg_cov": 0.0})
        by_subject[r.subject]["n"] += 1
        by_subject[r.subject]["correct"] += int(r.answer_correct)
        by_subject[r.subject]["avg_score"] += r.answer_score
        by_subject[r.subject]["avg_cov"] += r.tool_coverage
    for d in (by_topic, by_subject):
        for k, v in d.items():
            v["avg_score"] = round(v["avg_score"] / v["n"], 4)
            v["avg_cov"] = round(v["avg_cov"] / v["n"], 4)
            v["accuracy"] = round(v["correct"] / v["n"], 4)
    summary = {
        "config": asdict(cfg),
        "n_problems": n,
        "answer_accuracy": round(correct / n, 4),
        "answer_score_avg": round(avg_score, 4),
        "tool_coverage_avg": round(avg_cov, 4),
        "total_cost_usd": round(total_cost, 4),
        "total_duration_ms": total_ms,
        "by_topic": by_topic,
        "by_subject": by_subject,
        "results": [r.to_dict() for r in results],
    }
    if madd_fa_avg is not None:
        summary["madd_fa_avg"] = round(madd_fa_avg, 4)
    if ddb_score_100 is not None:
        summary["ddb_score_100"] = round(ddb_score_100, 1)
    if scipredict_score_100 is not None:
        summary["scipredict_score_100"] = round(scipredict_score_100, 1)
    if smdd_score_100 is not None:
        summary["smdd_score_100"] = round(smdd_score_100, 1)
    if biomni_eval1_score_100 is not None:
        summary["biomni_eval1_score_100"] = round(biomni_eval1_score_100, 1)
    return summary


def _write_summary_md(path: Path, summary: dict[str, Any], results: list[GradingResult]) -> None:
    lines: list[str] = []
    cfg = summary["config"]
    lines.append(f"# sciMAS test summary — {cfg.get('dataset_name', SCIAGENTGYM)} / {cfg['label']}")
    lines.append("")
    lines.append(f"- dataset: **{cfg.get('dataset_name', SCIAGENTGYM)}**")
    lines.append(f"- n_problems: **{summary['n_problems']}**")
    lines.append(f"- answer accuracy: **{summary['answer_accuracy']:.1%}**")
    lines.append(f"- answer/checklist score (0-1): **{summary['answer_score_avg']:.2f}**")
    lines.append(f"- avg tool coverage: **{summary['tool_coverage_avg']:.1%}**")
    if "madd_fa_avg" in summary:
        lines.append(f"- MADD FA proxy (SSA*TS): **{summary['madd_fa_avg']:.2f}**")
    if "ddb_score_100" in summary:
        lines.append(f"- DrugDiscoveryBench local proxy score: **{summary['ddb_score_100']:.1f}/100**")
    if "scipredict_score_100" in summary:
        lines.append(f"- SciPredict local score: **{summary['scipredict_score_100']:.1f}/100**")
    if "smdd_score_100" in summary:
        lines.append(f"- SMDDBench score: **{summary['smdd_score_100']:.1f}/100**")
    if "biomni_eval1_score_100" in summary:
        lines.append(f"- Biomni Eval1 local score: **{summary['biomni_eval1_score_100']:.1f}/100**")
    lines.append(f"- total cost: **${summary['total_cost_usd']:.4f}**")
    lines.append(f"- total duration: **{summary['total_duration_ms']/1000:.1f}s**")
    if cfg.get("subject") or cfg.get("topic"):
        lines.append(f"- filters: subject=`{cfg.get('subject')}` topic=`{cfg.get('topic')}`")
    lines.append("")
    lines.append("## By topic")
    lines.append("")
    lines.append("| topic | n | accuracy | avg score | avg tool cov |")
    lines.append("|---|---|---|---|---|")
    for t, v in sorted(summary["by_topic"].items()):
        lines.append(
            f"| {t} | {v['n']} | {v['accuracy']:.1%} | {v['avg_score']:.2f} | {v['avg_cov']:.1%} |"
        )
    lines.append("")
    lines.append("## Per problem")
    lines.append("")
    lines.append("| id | topic | ✓ | score | tool cov | cost $ | tools called |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        called = ", ".join(r.called_tools) or "—"
        if len(called) > 80:
            called = called[:77] + "…"
        lines.append(
            f"| {r.problem_id} | {r.topic} | {'✓' if r.answer_correct else '✗'} | "
            f"{r.answer_score:.2f} | {r.tool_coverage:.0%} | "
            f"{r.cost_usd:.4f} | {called} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run sciMAS over benchmark problems")
    p.add_argument("--dataset", default=SCIAGENTGYM,
                   metavar="DATASET",
                   help="benchmark dataset to evaluate")
    p.add_argument("--root", default=None,
                   help="dataset root override")
    p.add_argument("--split", default="train",
                   help="dataset split/config (ResearchClawBench split; SciPredict: bk or nbk)")
    p.add_argument("--subject", default=None)
    p.add_argument("--topic", default=None)
    p.add_argument("--ids", default=None, help="comma-separated problem ids")
    p.add_argument("--query", default=None, help="substring match on question/filename")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--roles", nargs="*", default=None,
                   help="role whitelist for the planner")
    p.add_argument("--mcp-config", default=None,
                   help="path to .mcp.json (default: sciMAS/config/mcp.json)")
    p.add_argument("--claude-bin", default="claude")
    p.add_argument("--model", default=None)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument(
        "--planner-mode",
        choices=["python-dsl", "legacy-json"],
        default="python-dsl",
        help=(
            "Planner output mode. python-dsl uses restricted async workflow "
            "source; legacy-json preserves the old JSON topology planner."
        ),
    )
    p.add_argument(
        "--allow-role-tool-fallback",
        action="store_true",
        help=(
            "If skill routing selects no skill for a role, expose the role's "
            "full discipline tool list. Default is strict skill-only routing."
        ),
    )
    capture_group = p.add_mutually_exclusive_group()
    capture_group.add_argument(
        "--capture-tool-calls",
        action="store_true",
        dest="capture_tool_calls",
        help=(
            "Use stream-json Claude output and capture MCP tool calls. "
            "Plain text/no-capture mode is the default."
        ),
    )
    capture_group.add_argument(
        "--no-capture-tool-calls",
        action="store_false",
        dest="capture_tool_calls",
        help=(
            "Use plain text Claude output instead of stream-json. This is the "
            "default; tool coverage will be unavailable."
        ),
    )
    p.set_defaults(capture_tool_calls=False)
    p.add_argument("--label", default="run",
                   help="label embedded in the output directory name")
    p.add_argument("--out", default=None,
                   help="output directory (default: tests/results/<dataset>/<timestamp>-<label>/)")
    p.add_argument(
        "--research-claw-judge", dest="research_claw_judge",
        choices=("auto", "force", "off"), default="auto",
        help=(
            "ResearchClawBench official-judge selection: "
            "'auto' (default; uses env RESEARCH_CLAW_JUDGE_*), "
            "'force' (require env; raise on missing), "
            "'off' (skip judge even when env is set)."
        ),
    )
    p.add_argument(
        "--smdd-official-eval",
        action="store_true",
        help=(
            "For SMDDBench, write the final answer as the requested artifact "
            "and run the official Docker evaluator."
        ),
    )
    p.add_argument(
        "--smdd-docker-image",
        default="smdd-evals",
        help="SMDDBench official evaluator Docker image name.",
    )
    p.add_argument(
        "--smdd-eval-timeout",
        type=float,
        default=3600.0,
        help="Per-task SMDDBench official evaluator timeout in seconds.",
    )
    p.add_argument(
        "--smdd-gpu-id",
        default=None,
        help="Optional GPU id passed to Docker for SMDDBench official evaluation.",
    )
    args = p.parse_args(argv)

    dataset_name = normalize_dataset_name(args.dataset)
    root = Path(args.root) if args.root else default_dataset_root(dataset_name)
    ids = _parse_ids(args.ids)
    problems = load_dataset(root, dataset_name=dataset_name, split=args.split)
    problems = filter_problems(
        problems,
        subject=args.subject,
        topic=args.topic,
        ids=ids,
        query=args.query,
    )
    if args.limit is not None:
        problems = problems[: args.limit]

    if not problems:
        print("(no problems match filters)")
        return 1

    if args.out:
        out_dir = Path(args.out).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = (Path(__file__).resolve().parent / "results" /
                   dataset_dir_name(dataset_name) /
                   f"{stamp}-{args.label}").resolve()

    cfg = RunConfig(
        dataset_name=dataset_name,
        dataset_root=str(root),
        label=args.label,
        subject=args.subject,
        topic=args.topic,
        ids=ids,
        query=args.query,
        limit=args.limit,
        roles=list(args.roles) if args.roles else [],
        mcp_config=args.mcp_config,
        claude_bin=args.claude_bin,
        model=args.model,
        timeout=args.timeout,
        capture_tool_calls=args.capture_tool_calls,
        planner_mode=args.planner_mode.replace("-", "_"),
        allow_role_tool_fallback=args.allow_role_tool_fallback,
        research_claw_judge=args.research_claw_judge,
        smdd_official_eval=args.smdd_official_eval,
        smdd_docker_image=args.smdd_docker_image,
        smdd_eval_timeout=args.smdd_eval_timeout,
        smdd_gpu_id=args.smdd_gpu_id,
    )

    if cfg.research_claw_judge == "force" and not (
        os.environ.get("RESEARCH_CLAW_JUDGE_API_KEY")
        or os.environ.get("JUDGE_API_KEY")
    ):
        raise SystemExit(
            "--research-claw-judge=force requires RESEARCH_CLAW_JUDGE_API_KEY "
            "(or upstream JUDGE_API_KEY) to be set in the environment."
        )

    print(f"Running {len(problems)} {dataset_name} problem(s) — output → {out_dir}")
    summary = run_batch(problems, cfg=cfg, output_root=out_dir)
    print()
    print(f"answer accuracy: {summary['answer_accuracy']:.1%}  "
          f"score avg: {summary['answer_score_avg']:.2f}  "
          f"tool cov: {summary['tool_coverage_avg']:.1%}  "
          f"cost: ${summary['total_cost_usd']:.4f}")
    if "madd_fa_avg" in summary:
        print(f"MADD FA proxy: {summary['madd_fa_avg']:.2f}")
    if "ddb_score_100" in summary:
        print(f"DrugDiscoveryBench local proxy score: {summary['ddb_score_100']:.1f}/100")
    if "scipredict_score_100" in summary:
        print(f"SciPredict local score: {summary['scipredict_score_100']:.1f}/100")
    if "smdd_score_100" in summary:
        print(f"SMDDBench score: {summary['smdd_score_100']:.1f}/100")
    if "biomni_eval1_score_100" in summary:
        print(f"Biomni Eval1 local score: {summary['biomni_eval1_score_100']:.1f}/100")
    print(f"results in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
