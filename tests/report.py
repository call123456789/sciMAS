"""Aggregate and compare previously-run test results.

After a batch run, you can re-grade without re-running the model:

    python tests/report.py tests/results/SciAgentGYM/20260909-baseline/

To compare two configurations side-by-side:

    python tests/report.py --diff tests/results/SciAgentGYM/A/ tests/results/SciAgentGYM/B/

The diff view shows per-problem accuracy delta, score delta, and
tool-coverage delta.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class RunBundle:
    path: Path
    summary: dict[str, Any]
    results_by_id: dict[str, dict[str, Any]]

    @classmethod
    def from_dir(cls, path: Path) -> "RunBundle":
        run_json = path / "run.json"
        if not run_json.exists():
            raise FileNotFoundError(f"no run.json in {path}")
        summary = json.loads(run_json.read_text(encoding="utf-8"))
        results_by_id: dict[str, dict[str, Any]] = {}
        per_dir = path / "per-problem"
        if per_dir.exists():
            for f in sorted(per_dir.glob("*.json")):
                payload = json.loads(f.read_text(encoding="utf-8"))
                grade = payload.get("grade", {})
                pid = grade.get("problem_id")
                if pid is not None:
                    results_by_id[str(pid)] = grade
        return cls(path=path, summary=summary, results_by_id=results_by_id)


def _print_summary(bundle: RunBundle) -> None:
    s = bundle.summary
    cfg = s["config"]
    print(f"=== {bundle.path.name} ===")
    print(f"  dataset         : {cfg.get('dataset_name', 'SciAgentGYM')}")
    print(f"  label           : {cfg.get('label', '')}")
    if cfg.get("subject") or cfg.get("topic"):
        print(f"  filters         : subject={cfg.get('subject')!r} topic={cfg.get('topic')!r}")
    print(f"  n_problems      : {s['n_problems']}")
    print(f"  answer accuracy : {s['answer_accuracy']:.1%}")
    print(f"  answer score    : {s['answer_score_avg']:.2f}")
    print(f"  tool coverage   : {s['tool_coverage_avg']:.1%}")
    if "madd_fa_avg" in s:
        print(f"  MADD FA proxy   : {s['madd_fa_avg']:.2f}")
    if "ddb_score_100" in s:
        print(f"  DDB local score : {s['ddb_score_100']:.1f}/100")
    print(f"  total cost      : ${s['total_cost_usd']:.4f}")
    print(f"  total duration  : {s['total_duration_ms']/1000:.1f}s")
    if s.get("by_topic"):
        print("  by_topic:")
        for t, v in sorted(s["by_topic"].items()):
            print(f"    - {t:30s} n={v['n']:2d}  acc={v['accuracy']:.1%}  "
                  f"score={v['avg_score']:.2f}  cov={v['avg_cov']:.1%}")
    print()


def _print_diff(a: RunBundle, b: RunBundle) -> None:
    print(f"### diff: {a.path.name}  vs  {b.path.name}\n")
    sa, sb = a.summary, b.summary
    delta = lambda x, y: (y - x) if isinstance(x, (int, float)) else None
    rows = [
        ("answer_accuracy", sa["answer_accuracy"], sb["answer_accuracy"]),
        ("answer_score_avg", sa["answer_score_avg"], sb["answer_score_avg"]),
        ("tool_coverage_avg", sa["tool_coverage_avg"], sb["tool_coverage_avg"]),
        ("total_cost_usd", sa["total_cost_usd"], sb["total_cost_usd"]),
    ]
    if "madd_fa_avg" in sa or "madd_fa_avg" in sb:
        rows.insert(3, ("madd_fa_avg", sa.get("madd_fa_avg", 0.0), sb.get("madd_fa_avg", 0.0)))
    if "ddb_score_100" in sa or "ddb_score_100" in sb:
        rows.insert(3, ("ddb_score_100", sa.get("ddb_score_100", 0.0), sb.get("ddb_score_100", 0.0)))
    print(f"  {'metric':24s}  {'A':>8s}  {'B':>8s}  {'Δ':>8s}")
    for name, va, vb in rows:
        d = (vb - va) if isinstance(va, (int, float)) else 0.0
        print(f"  {name:24s}  {va:>8.4f}  {vb:>8.4f}  {d:>+8.4f}")
    print()

    # Per-problem diff
    common = sorted(set(a.results_by_id) & set(b.results_by_id))
    if not common:
        print("  (no common problem ids)")
        return
    print(f"  per-problem ({len(common)} common):")
    id_width = max(4, *(len(pid) for pid in common))
    print(f"  {'id':>{id_width}}  {'acc(A)':>6}  {'acc(B)':>6}  {'score(A)':>9}  {'score(B)':>9}  {'cov(A)':>6}  {'cov(B)':>6}")
    flipped = 0
    for pid in common:
        ra, rb = a.results_by_id[pid], b.results_by_id[pid]
        aa = int(ra.get("answer_correct", False))
        bb = int(rb.get("answer_correct", False))
        if aa != bb:
            flipped += 1
        print(
            f"  {pid:>{id_width}}  {aa:>6d}  {bb:>6d}  "
            f"{ra.get('answer_score', 0):>9.3f}  {rb.get('answer_score', 0):>9.3f}  "
            f"{ra.get('tool_coverage', 0):>6.0%}  {rb.get('tool_coverage', 0):>6.0%}"
        )
    print(f"\n  {flipped} problem(s) flipped correctness.")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Aggregate or compare sciMAS test results")
    p.add_argument("run_dirs", nargs="+",
                   help="one or more tests/results/<dataset>/<run-id>/ directories")
    p.add_argument("--diff", action="store_true",
                   help="compare first two runs side-by-side")
    args = p.parse_args(argv)

    bundles = [RunBundle.from_dir(Path(d)) for d in args.run_dirs]
    if args.diff and len(bundles) >= 2:
        _print_summary(bundles[0])
        _print_summary(bundles[1])
        _print_diff(bundles[0], bundles[1])
    else:
        for b in bundles:
            _print_summary(b)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
