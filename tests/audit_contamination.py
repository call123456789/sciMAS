"""Which recorded runs were exposed to the ground-truth leak.

Run isolation (``tests/README.md``, "Run isolation") closes the channel going
forward. This is about the results already on disk: before that work, every
agent in a run could read the dataset's gold answers, and Claude Code's own
memory wrote down where they live — so a run could also inherit the location
without being told. Either way the number it produced is not evidence that
sciMAS reasoned its way there.

Three exposures are looked for, each with its own evidence:

``read``       A session of the run read an answer-bearing path: a marker from
               ``sandbox_policy.answer_markers()`` appears in its transcript,
               or in a tool call the run recorded. The second half is usually
               empty rather than clean: ``tests/runner.py`` records tool calls
               only under ``--capture-tool-calls`` (its default ``_TextRunner``
               "does not capture tool calls"), so a run with no recorded call
               says nothing either way. Transcripts are the evidence that
               carries this finding.
``inherited``  The run's transcripts are in Claude Code's *shared* bucket for
               this repository — i.e. it ran with state isolation off — and it
               started after the memory naming the answer files appeared. Its
               agents had that memory in their system prompt whether or not
               they acted on it.
``wrote``      One of the run's sessions is the ``originSessionId`` of a memory
               file. This is how the leak spread: a run that found the answers
               wrote the location down for every run after it.

Nothing here is deleted or rewritten. The output is a manifest, and with
``--mark`` a small ``INVALID.json`` inside each affected run directory so
whoever reads the run sees why it should not be counted.

    python tests/audit_contamination.py                 # list, write manifest
    python tests/audit_contamination.py --mark          # also write markers
    python tests/audit_contamination.py --all           # include clean runs

The manifest lists answer paths, so it defaults to ``tests/results/`` — a
directory the agent sandbox masks. Point ``--out`` somewhere outside the repo
if you would rather keep it out of the workspace entirely.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sandbox_policy  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
#: Claude Code's bucket for a session started in the repo root.
SHARED_BUCKET = (
    Path.home() / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(REPO_ROOT))
)
MEMORY_DIR = SHARED_BUCKET / "memory"
#: Fragments that only appear in a transcript once the memory was in play.
POISON_MARKERS = ("reference-golden-answers", "Golden answers location")


@dataclass
class Run:
    """One graded run: a caller's directory holding one or more agent sessions."""

    path: Path
    job: str
    started_at: str = ""
    sessions: dict[str, str] = field(default_factory=dict)  # node/role -> session id
    question: str = ""
    findings: dict[str, list[str]] = field(default_factory=dict)

    @property
    def exposures(self) -> tuple[str, ...]:
        return tuple(sorted(self.findings))

    @property
    def bundle_dir(self) -> Path:
        """The directory a reader calls "the run": the one holding episodes.

        ``path`` is one problem's episode under ``scimas-runs/``, which is the
        right unit for the analysis — it owns the session ids and the recorded
        tool calls — and the wrong one for a reader, who opens the job directory
        above it (``tests/runner.py``) or browses ``runs-dashboard/``.
        """
        parent = self.path.parent
        return parent.parent if parent.name == "scimas-runs" else self.path


def _iter_reports(root: Path) -> Iterable[Path]:
    """Every ``report.json`` this deployment has written, dashboard included."""
    for base in ("runs-dashboard", "tests/results"):
        directory = root / base
        if not directory.exists():
            continue
        yield from sorted(directory.glob("*/scimas-runs/*/report.json"))
        # Older layouts keep the runs one level deeper (dataset/run/scimas-runs).
        yield from sorted(directory.glob("*/*/scimas-runs/*/report.json"))


def collect_runs(root: Path) -> list[Run]:
    runs: list[Run] = []
    for report in _iter_reports(root):
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        episodes = data.get("runs") or []
        sessions: dict[str, str] = {}
        recorded: list[str] = []
        for index, episode in enumerate(episodes):
            if not isinstance(episode, dict):
                continue
            session = episode.get("session_id")
            if session:
                sessions[f"{episode.get('step_id') or index}:{episode.get('role')}"] = session
            for call in episode.get("tool_calls") or []:
                if isinstance(call, dict):
                    recorded.append(json.dumps(call.get("arguments"), default=str))
        if not sessions and isinstance(data.get("session_ids"), dict):
            sessions = dict(data["session_ids"])
        job = report.parents[2].name if len(report.parents) > 2 else report.parent.name
        question = data.get("problem") or ""
        runs.append(
            Run(
                path=report.parent,
                job=job,
                started_at=str(data.get("started_at") or ""),
                sessions=sessions,
                question=question if isinstance(question, str) else str(question),
                findings={"recorded": recorded} if recorded else {},
            )
        )
    return runs


def scan_bucket(bucket: Path, markers: tuple[str, ...]) -> dict[str, dict]:
    """Per-session evidence from Claude Code's shared bucket.

    One pass over every transcript, matching a single alternation of markers —
    the corpus is a few hundred MB and this is the only expensive step. Each
    hit keeps a few surrounding path fragments: a bare ``runs-dashboard`` tells
    you nothing, while the path around it tells you whether the session read
    *another* run or merely mentioned its own output directory, which is what
    ``_is_own_reference`` sorts out later.
    """
    pattern = re.compile("|".join(re.escape(m) for m in markers))
    token = re.compile(
        r"[A-Za-z0-9_./~-]{0,120}(?:" + "|".join(re.escape(m) for m in markers) + r")"
        r"[A-Za-z0-9_./~-]{0,120}"
    )
    poison = re.compile("|".join(re.escape(m) for m in POISON_MARKERS))
    found: dict[str, dict] = {}
    if not bucket.exists():
        return found
    for transcript in sorted(bucket.glob("*.jsonl")):
        session = transcript.stem
        hits: dict[str, list[str]] = {}
        first = last = ""
        saw_poison = False
        try:
            with transcript.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    stamp = re.search(r'"timestamp":"([^"]+)"', line)
                    if stamp:
                        first = first or stamp.group(1)
                        last = stamp.group(1)
                    if not saw_poison and poison.search(line):
                        saw_poison = True
                    if not pattern.search(line):
                        continue
                    for match in token.finditer(line):
                        fragment = match.group(0)
                        for marker in markers:
                            if marker in fragment:
                                samples = hits.setdefault(marker, [])
                                if len(samples) < 8 and fragment not in samples:
                                    samples.append(fragment)
                                break
        except OSError:
            continue
        found[session] = {
            "transcript": str(transcript),
            "first": first,
            "last": last,
            "hits": hits,
            "poisoned": saw_poison,
        }
    return found


def memory_origins(memory_dir: Path) -> dict[str, str]:
    """``session id -> memory file`` for the memories that name the answers."""
    origins: dict[str, str] = {}
    if not memory_dir.exists():
        return origins
    for path in sorted(memory_dir.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        match = re.search(r"originSessionId:\s*([0-9a-fA-F-]+)", text)
        if match:
            origins[match.group(1)] = path.name
    return origins


def poison_epoch(bucket: dict[str, dict], origins: dict[str, str]) -> str:
    """When exposure first goes on record, from the transcripts themselves.

    Filesystem timestamps do not help: a memory is rewritten in place every
    time an agent updates it, so its mtime is the last write, not the first.
    The transcripts do. Two things date the exposure — the memory appearing in
    a session's system prompt, and a session reading an answer path — and the
    earlier of the two is when the leak was demonstrably live.
    """
    stamps = [
        entry["first"]
        for entry in bucket.values()
        if entry.get("first") and (entry.get("poisoned") or entry.get("hits"))
    ]
    return min(stamps) if stamps else ""


def _is_own_reference(fragment: str, run: "Run") -> bool:
    """Whether a hit is the run talking about its own directory.

    Every run records its figures and reports under ``runs-dashboard/<job>/``,
    so the string ``runs-dashboard`` shows up in its own transcripts for
    entirely innocent reasons. Comparing against the job and run directory
    names keeps that out of the findings.
    """
    own = {run.path.name}
    if len(run.path.parents) > 1:
        own.add(run.path.parents[1].name)
    return any(token and token in fragment for token in own)


def classify(
    runs: list[Run],
    bucket: dict[str, dict],
    origins: dict[str, str],
    epoch: str,
) -> None:
    for run in runs:
        read: list[str] = []
        inherited: list[str] = []
        wrote: list[str] = []
        for label, session in run.sessions.items():
            entry = bucket.get(session)
            if session in origins:
                wrote.append(f"{label} wrote memory/{origins[session]}")
            if entry is None:
                # No transcript in the shared bucket: either it ran isolated
                # (good) or the bucket was pruned (the CLI cleans by age).
                continue
            outside = {
                marker: samples
                for marker, samples in entry["hits"].items()
                if not all(_is_own_reference(s, run) for s in samples)
            }
            for marker, samples in sorted(outside.items()):
                external = [s for s in samples if not _is_own_reference(s, run)]
                read.append(f"{label} read {marker} ({external[0]})")
            if epoch and run.started_at and run.started_at > epoch:
                inherited.append(label)
        for call in run.findings.pop("recorded", []):
            for marker in sandbox_policy.answer_markers():
                if marker in call and not _is_own_reference(call, run):
                    read.append(f"tool call touched {marker}: {call[:160]}")
                    break
        if read:
            run.findings["read"] = sorted(set(read))
        if inherited:
            run.findings["inherited"] = [
                f"{len(inherited)} session(s) ran after {epoch[:19]} with the "
                f"shared state directory, so the memory was in their prompt"
            ]
        if wrote:
            run.findings["wrote"] = sorted(set(wrote))


def render(runs: list[Run], epoch: str, markers: tuple[str, ...], all_runs: bool) -> str:
    dirty = [run for run in runs if run.exposures]
    lines = [
        "# Contaminated runs",
        "",
        "Generated by `python tests/audit_contamination.py`.",
        "Nothing here is deleted: every affected run is listed so it can be",
        "excluded from scoring, and `--mark` writes an `INVALID.json` into the",
        "run directory itself — plus one in the job directory above it, which is",
        "what a reader browsing `runs-dashboard/` or `tests/results/` actually",
        "opens. `report.py` reads that marker and prints the reason above the",
        "run's numbers, so a contaminated score says so where it is read — it is",
        "annotated, not hidden.",
        "",
        f"- runs examined: **{len(runs)}**",
        f"- runs with at least one exposure: **{len(dirty)}**",
        f"- exposure on record from: **{epoch or 'unknown'}** (earliest transcript "
        "showing the memory or an answer read)",
        "- markers matched (from `sandbox_policy.answer_markers()`): "
        + ", ".join(f"`{m}`" for m in markers),
        "",
        "`read` is direct evidence, `inherited` means the run's agents had the",
        "memory in their system prompt, `wrote` means the run is where the",
        "memory came from. A run is listed if any of the three applies.",
        "",
    ]
    by_job: dict[str, list[Run]] = {}
    for run in dirty if not all_runs else runs:
        by_job.setdefault(run.job, []).append(run)
    for job in sorted(by_job):
        lines.append(f"## {job}")
        lines.append("")
        for run in sorted(by_job[job], key=lambda r: str(r.path)):
            exposures = ",".join(run.exposures) or "clean"
            lines.append(f"- `{run.path}` — **{exposures}** (started {run.started_at or '?'})")
            for kind, notes in sorted(run.findings.items()):
                for note in notes:
                    lines.append(f"    - {kind}: {note}")
        lines.append("")
    clean = [run for run in runs if not run.exposures]
    lines += [
        "## Unaffected runs",
        "",
        f"{len(clean)} of {len(runs)} runs left no trace of the leak: no session "
        "transcript in the shared state directory, no answer path in a "
        "transcript or tool call, no memory written from one of its sessions.",
        "",
    ]
    lines += [f"- `{run.path}`" for run in sorted(clean, key=lambda r: str(r.path))]
    lines += [
        "",
        "Treat that list as the weaker claim of the two. A transcript is written",
        "per session under the config directory in use, so isolation shows up as",
        "an *absence* — indistinguishable from a bucket the CLI has not pruned",
        "yet. Two things also make this audit conservative by construction: a",
        "session shared by several runs (``--resume``) puts its read on all of",
        "them, and the exposure epoch is the earliest transcript still on disk,",
        "which is a lower bound on the memory's age: memory files are rewritten",
        "in place, so their timestamps only ever show the last write. Runs before",
        "that epoch are unverified, not proven clean.",
        "",
    ]
    return "\n".join(lines)


def _write_marker(directory: Path, payload: dict) -> None:
    (directory / "INVALID.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def mark(runs: list[Run], epoch: str) -> tuple[int, int]:
    """Write the markers; returns ``(episodes marked, job directories marked)``.

    Two levels on purpose. The episode marker is the precise record — it is
    where the evidence is. The job marker is what a reader actually trips over:
    a job directory holds one episode per problem and its aggregate includes
    them, and on this deployment the contaminated runs are dashboard runs,
    whose job directories are what ``runs-dashboard/`` lists.
    """
    stamp = datetime.now(timezone.utc).isoformat()
    written = 0
    jobs: dict[Path, list[Run]] = {}
    for run in runs:
        if not run.exposures:
            continue
        _write_marker(
            run.path,
            {
                "invalid": True,
                "reason": "ground-truth exposure before run isolation",
                "exposures": list(run.exposures),
                "detail": {k: v for k, v in sorted(run.findings.items())},
                "detected_by": "tests/audit_contamination.py",
                "detected_at": stamp,
                "memory_epoch": epoch,
            },
        )
        written += 1
        job = run.bundle_dir
        if job != run.path:
            jobs.setdefault(job, []).append(run)
    for job, group in jobs.items():
        _write_marker(
            job,
            {
                "invalid": True,
                "reason": "ground-truth exposure before run isolation",
                "exposures": sorted({e for run in group for e in run.exposures}),
                "exposed_episodes": [
                    run.path.name for run in sorted(group, key=lambda r: str(r.path))
                ],
                "detail": {
                    "note": (
                        "one INVALID.json per episode under scimas-runs/ holds "
                        "that episode's evidence"
                    )
                },
                "detected_by": "tests/audit_contamination.py",
                "detected_at": stamp,
                "memory_epoch": epoch,
            },
        )
    return written, len(jobs)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument(
        "--bucket",
        default=str(SHARED_BUCKET),
        help="Claude Code's bucket for this repo (default: the shared one)",
    )
    parser.add_argument("--memory", default=str(MEMORY_DIR))
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "tests" / "results" / "CONTAMINATION.md"),
        help="manifest path (kept inside the sandbox-masked tests/results by default)",
    )
    parser.add_argument("--json", dest="json_out", default="", help="also write JSON here")
    parser.add_argument("--mark", action="store_true", help="write INVALID.json per run")
    parser.add_argument("--all", action="store_true", help="list clean runs too")
    args = parser.parse_args(argv)

    markers = sandbox_policy.answer_markers()
    runs = collect_runs(Path(args.root))
    bucket = scan_bucket(Path(args.bucket), markers)
    origins = memory_origins(Path(args.memory))
    epoch = poison_epoch(bucket, origins)
    classify(runs, bucket, origins, epoch)

    manifest = render(runs, epoch, markers, args.all)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(manifest, encoding="utf-8")

    dirty = [run for run in runs if run.exposures]
    print(f"audited {len(runs)} runs: {len(dirty)} exposed")
    print(f"manifest: {out}")
    if args.mark:
        episodes, jobs = mark(runs, epoch)
        print(f"marked {episodes} run directory/ies and {jobs} job(s) as INVALID")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                [
                    {
                        "path": str(run.path),
                        "job": run.job,
                        "started_at": run.started_at,
                        "exposures": list(run.exposures),
                        "findings": run.findings,
                    }
                    for run in dirty
                ],
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"json: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
