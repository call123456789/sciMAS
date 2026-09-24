"""Which paths a run hides from its agent, and which ones it keeps readable.

``agent_sandbox`` knows *how* to mask a path; this module knows *what* to mask.
The two are deliberately separate: the mechanism is generic and testable, the
list below is an audit result about this repository's datasets and has to be
maintained as datasets are added.

Why masking is enough
---------------------

The harness reads every dataset *before* the agent starts — ``load_dataset``
runs in ``run_batch``'s caller, in this process — so by the time a sandboxed
CLI exists, questions, rubrics and gold answers are already in memory. The
agent gets ``problem.question`` as text and never needs the corpus on disk.
That is what makes it safe to mask a file that holds both the question and its
answer: the loader already has it, and the sandbox only covers the CLI
subprocess and the MCP servers it spawns.

The same argument covers the two cases that look like conflicts and are not:

  * SMDDBench's official evaluator mounts ``task_dir`` (including the hidden
    ``eval_*.smi``) into Docker. It is invoked by the harness after the agent
    returns, from the host filesystem — outside the sandbox — so masking those
    files in the agent's view does not touch the score.
  * DrugDiscoveryBench's ``rubrics.json`` holds the task prompt as well as the
    rubrics. The prompt is read at load time; the agent sees it inlined.

What stays readable
-------------------

Everything the prompt names by absolute path, because the agent is told to
open it: SciAgentGYM's question images, ResearchClawBench's ``data/`` and
``related_work/`` (and the ``target_study/images/`` the agent is asked to
look at), DrugDiscoveryBench's ``environment/inputs/``, and the SMDDBench task
inputs. Masking any of these would turn a readable input into an empty file —
a silent wrong answer rather than a visible error.

The repository stays read-only apart from a small writable set: the run's own
output directories, the Claude Code state directory, and ``runs/`` (the CLI's
default output and several MCP servers' default ``output_dir``). Making the
whole repo writable would reopen a cross-run channel this audit is meant to
close — one run leaving a note in a source directory for the next one to read
— so a tool that insists on writing elsewhere fails loudly and can be given a
path under the run directory instead.
"""

from __future__ import annotations

import glob
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class _Group:
    """One dataset's answer-bearing paths.

    ``probe`` is a glob that matches exactly when the dataset is deployed: the
    corpora are downloaded, not versioned (see .gitignore), so a group whose
    probe is missing is reported as skipped rather than as a broken pattern. A
    plain path is a valid glob, but naming the data itself (rather than the
    directory it lives in) is what keeps a directory that exists while holding
    nothing but a README from reading as a deployed dataset.
    """

    name: str
    probe: str
    patterns: tuple[str, ...]


#: Answer-bearing paths, grouped by the dataset they belong to. Globs are
#: relative to the repo root and expanded per run; a file glob masks that file,
#: a directory glob masks the whole subtree.
_GROUPS: tuple[_Group, ...] = (
    _Group(
        name="SciAgentGYM",
        probe="dataset/refine_merged_multi_questions.json",
        # Same file as the question text (``question``) — safe only because the
        # loader has already read it. Holds ``golden_answer``,
        # ``tool_expected`` and ``solution_steps``.
        patterns=(
            "dataset/refine_merged_multi_questions.json",
            "dataset/refine_merged_single_questions.json",
        ),
    ),
    _Group(
        name="MADD",
        probe="dataset/MADD/**/dataset_L.xlsx",
        # dataset_L.xlsx is the table MADD is graded from: its ``case`` column
        # is the expected-tool gold for the coverage metric, so it is
        # answer-bearing even though it holds no ground-truth answer. The
        # legacy large_ds_chemical_result.* tables stay masked too — they do
        # carry ``tools answers`` and still load as a fallback.
        patterns=(
            "dataset/MADD/**/dataset_L.xlsx",
            "dataset/MADD/**/dataset_L.jsonl",
            "dataset/MADD/**/large_ds_chemical_result.jsonl",
            "dataset/MADD/**/large_ds_chemical_result.xlsx",
        ),
    ),
    _Group(
        name="SciPredict",
        probe="dataset/SciPredict/**/main_ds.csv",
        # main_ds.csv carries GTA/CLEAN_GTA next to the question columns.
        patterns=(
            "dataset/SciPredict/**/main_ds.csv",
            "dataset/SciPredict/**/rubrics.csv",
            "dataset/SciPredict/**/human_baseline.csv",
        ),
    ),
    _Group(
        name="BiomniEval1",
        probe="dataset/BiomniEval1/**/*.parquet",
        # The whole directory, not just the parquet: the loader takes the first
        # of ``*.parquet``, ``*.csv`` or ``*.jsonl`` it finds anywhere under the
        # root, and falls back to the HuggingFace cache at ``.hf/datasets`` —
        # all of which hold ``answer`` beside ``prompt``. Nothing here is an
        # input: every BiomniEval1 problem is inline text with no ``image_paths``.
        patterns=("dataset/BiomniEval1",),
    ),
    _Group(
        name="DrugQA",
        probe="dataset/target_identification_and_moa_openended.jsonl",
        # ``answers`` (gold), ``kegg_evidence`` and ``rationale`` sit in the
        # same object as ``question``. The loader accepts the file either loose
        # in dataset/ or in a DrugQA/ subdirectory, hence the second group.
        patterns=("dataset/target_identification_and_moa_openended.jsonl",),
    ),
    _Group(
        name="DrugQA (subdirectory layout)",
        probe="dataset/DrugQA",
        patterns=("dataset/DrugQA/**/*.jsonl",),
    ),
    _Group(
        name="ResearchClawBench",
        probe="dataset/ResearchClawBench/hf_dataset",
        # The checklist lives in the arrow file the loader reads, and is
        # mirrored per task under target_study/. The prompt explicitly tells
        # the agent not to use target_study — but target_study/images/ is the
        # target figure it *is* asked to read, so mask the files, not the dir.
        patterns=(
            "dataset/ResearchClawBench/hf_dataset",
            "dataset/ResearchClawBench/**/checklist*.json",
            "dataset/ResearchClawBench/**/target_study/paper.pdf",
        ),
    ),
    _Group(
        name="DrugDiscoveryBench",
        probe="dataset/DrugDiscoveryBench/**/rubrics.json",
        # tests/ holds rubrics.json (ground_truth, outcome_rubrics,
        # solution_steps) along with the judge harness; the task prompt is
        # inlined at load time and environment/inputs/ stays readable.
        patterns=("dataset/DrugDiscoveryBench/**/tests",),
    ),
    _Group(
        name="SMDDBench",
        probe="dataset/SMDDBench/**/task.yaml",
        # task.yaml names the hidden answer files and carries the evaluation
        # spec; eval_actives/eval_inactives are the held-out actives. The task
        # inputs (actives.smi, inactives.smi, protein.fasta) stay readable.
        patterns=(
            "dataset/SMDDBench/**/task.yaml",
            "dataset/SMDDBench/**/eval_actives.smi",
            "dataset/SMDDBench/**/eval_inactives.smi",
            "dataset/SMDDBench/upstream",
        ),
    ),
    _Group(
        name="past runs",
        probe="tests/results",
        # Previous runs' answers, grades and trajectories: the paths the
        # contamination audit found sessions reading. Runs under
        # runs-dashboard/ are the same story from the dashboard.
        patterns=("tests/results", "runs-dashboard"),
    ),
)

#: Repo-relative directories the agent may write to. ``runs/`` is the CLI's own
#: default output directory and the default ``output_dir`` of the environmental
#: MCP server; everything else the run needs lives under its output directory.
_WRITABLE_REPO_DIRS: tuple[str, ...] = ("runs",)


def answer_markers() -> tuple[str, ...]:
    """Path fragments that identify the answers, for contamination auditing.

    The masking list above says what to hide; this says what "this run read the
    answers" looks like when reading a transcript or a recorded tool call
    (``tests/audit_contamination.py``). Derived from the same patterns so the
    two cannot drift apart — a newly masked dataset becomes a new marker — and
    the fragments that would match anything on their own are dropped.
    """
    markers: list[str] = []
    for group in _GROUPS:
        for pattern in (group.probe, *group.patterns):
            rel = pattern.rstrip("/")
            name = rel.rsplit("/", 1)[-1]
            globbed = "*" in name
            if globbed:
                # ``checklist*.json`` is a family: the fixed prefix is what a
                # transcript would contain.
                prefix = name.split("*", 1)[0]
                name = prefix if len(prefix) >= 8 else name.replace("*", "")
            if len(name) < 6 or (name.startswith(".") and len(name) < 8):
                # ``.jsonl`` and friends match every unrelated path.
                continue
            if not globbed and "." not in name and len(name) < 10 and "/" in rel:
                # A short directory name is only meaningful with its parent:
                # ``upstream`` means nothing, ``SMDDBench/upstream`` does.
                name = "/".join(rel.split("/")[-2:])
            if name not in markers:
                markers.append(name)
    return tuple(markers)


@dataclass
class SandboxPlan:
    """The resolved boundary for one run."""

    hide: tuple[str, ...] = ()
    writable: tuple[str, ...] = ()
    #: ``(group, paths masked)`` per dataset, for audit logs and tests.
    matched: tuple[tuple[str, int], ...] = ()
    #: ``(group, pattern)`` for patterns that matched nothing while their group
    #: probe exists — a real hole, as opposed to a dataset simply not deployed.
    unmatched: tuple[tuple[str, str], ...] = ()
    #: ``(group, probe)`` for datasets that are not deployed at all.
    skipped: tuple[tuple[str, str], ...] = ()

    def render(self) -> str:
        """A one-block summary for the run log."""
        lines = [
            f"sandbox: masking {len(self.hide)} path(s), "
            f"{len(self.writable)} writable"
        ]
        for name, probe in self.skipped:
            lines.append(f"sandbox:   {name}: not deployed ({probe}), skipped")
        for name, pattern in self.unmatched:
            lines.append(
                f"sandbox:   {name}: WARNING {pattern!r} matched nothing "
                f"although the dataset is present"
            )
        return "\n".join(lines)


def plan(
    repo_root: os.PathLike | str,
    *,
    output_dirs: Iterable[os.PathLike | str] = (),
    extra_hide: Iterable[os.PathLike | str] = (),
    extra_writable: Iterable[os.PathLike | str] = (),
    mask_shared_state: Optional[bool] = None,
) -> SandboxPlan:
    """Resolve the hide/writable sets for one run of ``repo_root``.

    ``output_dirs`` are the directories this run writes into; they stay
    writable even when they sit inside a masked tree, which is the normal case
    for ``tests/results/<dataset>/<run>``. ``mask_shared_state`` decides
    whether Claude Code's own bucket for this repository is masked; it defaults
    to "yes only while state isolation is on", because with isolation off the
    CLI writes its transcripts there and a mask would silently swallow them —
    along with anything ``--resume`` needs.
    """
    if mask_shared_state is None:
        import claude_runner  # local: keeps this module usable on its own

        mask_shared_state = claude_runner.get_claude_state_isolated()
    root = Path(repo_root).resolve()
    hide: list[str] = []
    matched: list[tuple[str, int]] = []
    unmatched: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []

    for group in _GROUPS:
        # Patterns are evaluated regardless of the probe: several datasets can
        # be deployed at more than one path (DrugQA is loaded either from
        # dataset/ or from dataset/DrugQA/), so a probe that misses must not
        # decide whether the masking happens.
        probe_exists = bool(glob.glob(str(root / group.probe)))
        count = 0
        for pattern in group.patterns:
            matches = sorted(glob.glob(str(root / pattern), recursive=True))
            if matches:
                hide.extend(matches)
                count += len(matches)
            elif probe_exists:
                # The dataset is here but this pattern found nothing: either it
                # moved or it was renamed, and either way the boundary now has
                # a hole. Worth a warning, not silence.
                unmatched.append((group.name, pattern))
        if count:
            matched.append((group.name, count))
        elif not probe_exists:
            skipped.append((group.name, group.probe))

    if mask_shared_state:
        hide.extend(shared_state_paths(root))

    writable = [str(root / name) for name in _WRITABLE_REPO_DIRS]
    writable.extend(str(Path(p).resolve()) for p in output_dirs)
    writable.extend(str(Path(p).resolve()) for p in extra_writable)
    hide.extend(str(Path(p).resolve()) for p in extra_hide)

    return SandboxPlan(
        hide=_dedupe(hide),
        writable=_dedupe(writable),
        matched=tuple(matched),
        unmatched=tuple(unmatched),
        skipped=tuple(skipped),
    )


def shared_state_paths(repo_root: os.PathLike | str) -> list[str]:
    """Claude Code's own bucket for this working directory.

    The contamination audit found the leak on both sides: memory files that
    record where the answers live, and transcripts of earlier sessions that
    read them. State isolation stops the CLI from *loading* that bucket; this
    stops an agent from walking over and reading it, which is the same leak one
    step to the left. Only ever the bucket for this repo — the rest of the
    operator's config directory stays readable, so plugins and settings the
    CLI resolves by absolute path keep working.
    """
    config = os.environ.get("CLAUDE_CONFIG_DIR", "").strip()
    base = Path(config).expanduser() if config else Path.home() / ".claude"
    # The CLI keys its bucket by the directory it was started in, which is the
    # repo root for every run the harness spawns — and the harness's own cwd if
    # a caller ever starts one from elsewhere. Cover both; a bucket that does
    # not exist is simply not masked.
    cwds = [Path(repo_root).resolve(), Path.cwd().resolve()]
    projects = [
        base / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(cwd)) for cwd in cwds
    ]
    candidates = [*projects, base / "history.jsonl"]
    # The two cwds coincide on every normal run; the caller masks what it is
    # given, so hand it each path once.
    return list(dict.fromkeys(str(p) for p in candidates if p.exists()))


def _dedupe(paths: Iterable[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for raw in paths:
        resolved = os.path.abspath(os.path.expanduser(str(raw)))
        if resolved not in seen:
            seen.append(resolved)
    return tuple(seen)
