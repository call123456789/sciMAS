"""Graders for comparing sciMAS output against benchmark expectations.

For SciAgentGYM, three independent signals are produced per problem:

  1. answer_match  — does the predicted final answer contain / equal the
                     expected answer (numerically or as substring)?
  2. tool_coverage — what fraction of the SciAgentGYM-declared
                     ``tool_expected`` set was actually invoked by the
                     run? Tools that sciMAS does not even ship are
                     reported separately as ``missing_tools``.
  3. cost / duration — straight pass-through from the run report.

Each grader is a pure function over (problem, prediction) so the same
logic can be re-run on saved runs without re-invoking the model.

For ResearchClawBench, the official benchmark uses a multimodal LLM
judge over the agent report, target paper, figures, and checklist. The
local sciMAS grader now mirrors that judge (see ``rcb_official_judge``):
when ``RESEARCH_CLAW_JUDGE_API_KEY`` is set it issues one vision-capable
chat-completions call per checklist item and aggregates with the
upstream weights. Without credentials (or with
``RESEARCH_CLAW_JUDGE_DISABLED=1``) it silently falls back to a
lightweight checklist text/keyword overlap proxy so batch runs still
produce a summary.

For MADD, the upstream benchmark reports SSA (whether molecules returned
by tools are preserved in the final answer) and TS (whether the conductor
selected the expected tools). The local sciMAS grader mirrors those
signals with the migrated golden table and a mapping from MADD's original
function names to sciMAS pharma MCP tools.

For DrugDiscoveryBench, the official benchmark uses an LLM judge over
``rubrics.json`` and the agent trajectory. The local sciMAS grader
provides a deterministic outcome-rubric / ground-truth proxy so batches
can be inspected without judge credentials; empty public rubrics are
reported explicitly as ungradable.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

from dataset import (
    DRUG_DISCOVERY_BENCH,
    MADD,
    RESEARCH_CLAW_BENCH,
    SCIAGENTGYM,
    SCIPREDICT,
    SMDD_BENCH,
    BIOMNI_EVAL1,
    GoldenCall,
    Problem,
    extract_numbers,
)

from llm_judge import (
    LLMJudgeConfig,
    extract_boxed as _llm_extract_boxed,
    get_last_image_note as _llm_last_image_note,
    get_last_judge_error as _llm_last_judge_error,
    judge_correct as _llm_judge_correct,
    resolve_judge_config as _llm_resolve_judge_config,
)

try:
    from rcb_official_judge import score_checklist as _rcb_score_checklist
except ImportError:  # pragma: no cover - module ships next to this file
    _rcb_score_checklist = None  # type: ignore[assignment]

# Per-process handle used by ``grade_problem`` so the runner can thread a
# judge-cache directory into the official ResearchClawBench judge without
# having to plumb a new parameter through every call site. The runner
# calls ``set_judge_cache_dir`` once at the top of each batch.
_judge_cache_dir: Optional[str] = None


def set_judge_cache_dir(path: Optional[str]) -> None:
    """Pin the judge-cache directory used by ResearchClawBench official judge."""
    global _judge_cache_dir
    _judge_cache_dir = str(path) if path else None


def get_judge_cache_dir() -> Optional[str]:
    return _judge_cache_dir


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """A single ``mcp__<server>__<tool>`` invocation captured from the
    run trace (or any other source)."""

    tool: str            # short name (no server prefix)
    server: str          # chemistry-analytical etc., or "" if unknown
    arguments: dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        return f"{self.server}__{self.tool}" if self.server else self.tool

    @classmethod
    def from_full(cls, name: str, arguments: dict[str, Any] | None = None) -> "ToolCall":
        arguments = arguments or {}
        if name.startswith("mcp__"):
            stripped = name[len("mcp__"):]
            if "__" in stripped:
                server, tool = stripped.split("__", 1)
                return cls(tool=tool, server=server, arguments=arguments)
        return cls(tool=name, server="", arguments=arguments)


@dataclass
class GradingResult:
    problem_id: int | str
    dataset_name: str = ""
    subject: str = ""
    topic: str = ""
    expected_answer: str = ""
    predicted_answer: str = ""
    expected_tools: list[str] = field(default_factory=list)
    called_tools: list[str] = field(default_factory=list)
    missing_tools: list[str] = field(default_factory=list)
    extra_tools: list[str] = field(default_factory=list)
    answer_correct: bool = False
    answer_score: float = 0.0
    answer_notes: list[str] = field(default_factory=list)
    tool_coverage: float = 0.0
    checklist_score: float = 0.0
    cost_usd: float = 0.0
    duration_ms: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 1. Answer matching
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _numbers_close(a: float, b: float, rel: float = 0.05, abs_: float = 1e-6) -> bool:
    if a == b:
        return True
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) <= max(abs_, rel * scale)


def grade_answer(
    predicted: str,
    expected: str,
    *,
    tolerance_rel: float = 0.05,
    tolerance_abs: float = 1e-6,
) -> tuple[bool, float, list[str]]:
    """Return ``(correct, score 0..1, notes)``.

    Strategy, in order:
      1. Exact (normalized) match → 1.0
      2. Substring match after normalization → 1.0
      3. Numeric: all expected numbers appear (in order) in predicted
         with each within tolerance → 1.0
      4. Numeric: at least one expected number matches → 0.5 + 0.5 *
         (matched / total)
      5. Otherwise 0.0
    """
    notes: list[str] = []
    if expected is None or expected == "":
        notes.append("no expected answer defined")
        return False, 0.0, notes
    if predicted is None:
        notes.append("no predicted answer")
        return False, 0.0, notes

    pn, en = _norm(predicted), _norm(expected)
    if pn == en:
        return True, 1.0, ["exact"]
    if en in pn:
        return True, 1.0, ["expected is substring of predicted"]

    exp_nums = extract_numbers(expected)
    pred_nums = extract_numbers(predicted)
    if exp_nums:
        if not pred_nums:
            notes.append(f"expected {len(exp_nums)} number(s) but predicted has none")
            return False, 0.0, notes
        # Match in order: walk both lists, advance predicted until we
        # find something close to current expected.
        i_pred = 0
        matched = 0
        for e in exp_nums:
            while i_pred < len(pred_nums):
                if _numbers_close(pred_nums[i_pred], e, tolerance_rel, tolerance_abs):
                    matched += 1
                    i_pred += 1
                    break
                i_pred += 1
        if matched == len(exp_nums):
            return True, 1.0, [f"all {len(exp_nums)} expected number(s) matched"]
        if matched > 0:
            score = 0.5 + 0.5 * (matched / len(exp_nums))
            notes.append(f"matched {matched}/{len(exp_nums)} expected numbers")
            return False, score, notes
        notes.append(
            f"no expected numbers matched (expected {exp_nums}, predicted {pred_nums})"
        )
        return False, 0.0, notes

    # Fallback: token overlap.
    exp_tokens = set(re.findall(r"\w+", en))
    pred_tokens = set(re.findall(r"\w+", pn))
    if exp_tokens:
        overlap = len(exp_tokens & pred_tokens) / len(exp_tokens)
        notes.append(f"token overlap {overlap:.2f}")
        return False, overlap, notes
    return False, 0.0, ["no signal"]


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", _norm(text)))


def _text_overlap(predicted_tokens: set[str], expected_text: str) -> float:
    expected_tokens = _token_set(expected_text)
    if not expected_tokens:
        return 0.0
    return len(predicted_tokens & expected_tokens) / len(expected_tokens)


def grade_checklist_answer(
    predicted: str,
    checklist: list[dict[str, Any]],
) -> tuple[bool, float, list[str]]:
    """Lightweight ResearchClawBench checklist proxy score.

    This is not a replacement for the official multimodal judge. It
    gives a deterministic local signal by measuring how much of each
    checklist item's content/keywords appears in the generated answer.
    """
    if not checklist:
        return False, 0.0, ["no checklist rubric defined"]
    if not predicted:
        return False, 0.0, ["no predicted answer"]

    predicted_tokens = _token_set(predicted)
    weighted_score = 0.0
    total_weight = 0.0
    item_notes: list[str] = []
    for idx, item in enumerate(checklist, start=1):
        try:
            weight = float(item.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        if weight <= 0:
            continue
        content_score = _text_overlap(predicted_tokens, str(item.get("content", "")))
        keywords = item.get("keywords", [])
        keyword_scores: list[float] = []
        if isinstance(keywords, list):
            keyword_scores = [
                _text_overlap(predicted_tokens, str(keyword))
                for keyword in keywords
                if str(keyword).strip()
            ]
        keyword_score = (
            sum(keyword_scores) / len(keyword_scores)
            if keyword_scores else 0.0
        )
        item_score = max(content_score, keyword_score)
        weighted_score += weight * item_score
        total_weight += weight
        item_notes.append(
            f"item {idx} score={item_score:.2f} weight={weight:g}"
        )

    if total_weight <= 0:
        return False, 0.0, ["checklist weights sum to zero"]
    score = max(0.0, min(1.0, weighted_score / total_weight))
    notes = [
        "ResearchClawBench local proxy: weighted checklist text/keyword overlap",
        *item_notes,
    ]
    return score >= 0.5, score, notes


# ---------------------------------------------------------------------------
# ResearchClawBench official LLM judge
# ---------------------------------------------------------------------------
#
# The actual env-var / JSON-config resolution lives in ``tests/llm_judge.py``,
# which is the single source of truth shared with the SciAgentGYM LLM-judge
# path. ``_resolve_judge_config`` below is kept as a thin compatibility shim
# so existing call sites (and tests that reach into ``grader._resolve_judge_config``)
# keep working with the new ``LLMJudgeConfig`` return type.


def _resolve_judge_config() -> Optional[LLMJudgeConfig]:
    """Return judge credentials or ``None`` when unavailable / disabled.

    Honours the prefix chain declared in ``config/llm_judge.json`` (default:
    ``RESEARCH_CLAW_JUDGE``, ``JUDGE``, ``SCIMAS_LLM_JUDGE``). Returns
    ``None`` if the explicit-disable flag is set, the per-dataset flag
    is off, or no prefix supplies a full ``(key, base, model)`` triple.
    """
    return _llm_resolve_judge_config(dataset=RESEARCH_CLAW_BENCH)


def _has_judge_creds() -> bool:
    """Quick precheck used by ``grade_problem`` to decide which path to take."""
    return _rcb_score_checklist is not None and _resolve_judge_config() is not None


def grade_research_claw_official(
    problem: Problem,
    predicted_answer: str,
    cache_dir: Optional[str] = None,
) -> tuple[bool, float, list[str]]:
    """Score ``predicted_answer`` against ``problem.checklist`` using the
    official InternScience multimodal LLM judge.

    Returns ``(correct, score 0-1, notes)`` in the same shape as the
    other graders. Raises ``RuntimeError`` if judge credentials are
    missing — the caller in :func:`grade_problem` catches that and
    falls back to the local proxy so a missing API key never breaks a
    batch run.
    """
    if _rcb_score_checklist is None:
        raise RuntimeError(
            "rcb_official_judge module is not importable; cannot run official judge"
        )
    cfg = _resolve_judge_config()
    if cfg is None:
        raise RuntimeError(
            "ResearchClawBench official judge credentials not configured "
            "(set RESEARCH_CLAW_JUDGE_API_KEY/BASE/MODEL)"
        )

    checklist = list(problem.checklist or [])
    target_paper = getattr(problem, "target_paper", None)
    target_images = list(getattr(problem, "target_images", None) or [])
    instructions = ""
    task_info = problem.task_info or {}
    task_text = str(task_info.get("task", "") or "").strip()
    if task_text:
        instructions = task_text

    payload = _rcb_score_checklist(
        task_id=problem.id,
        predicted=predicted_answer,
        checklist=checklist,
        target_paper_path=target_paper,
        target_image_paths=target_images,
        judge_api_base=cfg.api_base,
        judge_api_key=cfg.api_key,
        judge_model=cfg.model,
        cache_dir=cache_dir or get_judge_cache_dir(),
        instructions=instructions,
    )

    score_100 = float(payload.get("total_score", 0.0))
    score_01 = max(0.0, min(1.0, score_100 / 100.0))
    correct = score_01 >= 0.5

    notes: list[str] = [
        f"ResearchClawBench official LLM judge (model={cfg.model}, "
        f"base={cfg.api_base}, prompt_version={payload.get('prompt_version', '?')})",
        f"weighted total_score={score_100:.2f}/100 (judge_calls={payload.get('judge_calls', 0)}, "
        f"cached={payload.get('cached', False)})",
    ]
    for record in payload.get("items", []):
        notes.append(
            f"item {int(record.get('index', 0)) + 1}: score={int(record.get('score', 0))}/100 "
            f"weight={float(record.get('weight', 0)):.3g} "
            f"degraded={bool(record.get('degraded', False))} "
            f"reason={str(record.get('reasoning', ''))[:300]}"
        )
    if payload.get("note"):
        notes.append(str(payload["note"]))

    return correct, score_01, notes


# ---------------------------------------------------------------------------
# SciAgentGYM LLM judge
# ---------------------------------------------------------------------------
#
# Mirrors the official SciAgentGYM ``is_answer_correct`` prompt: ask an
# LLM judge whether the model's answer is equivalent to the standard
# answer. Falls back to the deterministic ``grade_answer`` scorer when
# the judge is unconfigured (see ``config/llm_judge.json``). The judge is
# selected via the same env-var prefix chain used by ResearchClawBench,
# so any deployment that configures ``RESEARCH_CLAW_JUDGE_*`` (or one of
# the alternate prefixes) automatically gets SciAgentGYM LLM judging too.


# Step outputs that are artifact paths (``…/mass_spectrum.png``), not values.
_ARTIFACT_PATH_RE = re.compile(
    r"\.(?:png|jpe?g|gif|svg|pdf|csv|tsv|jsonl?|txt|xlsx?|npy|npz"
    r"|mol|sdf|pdb|cif|xyz|out|log|dat|h5|hdf5|zip)\s*$",
    re.IGNORECASE,
)


def _sciagentgym_reference_workings(problem: Problem) -> tuple[str, str]:
    """Render a SciAgentGYM golden step chain for the judge.

    Returns ``(workings_text, final_value)``. ``workings_text`` is one
    ``N. tool -> value units (note)`` line per usable step, or ``""`` when
    the chain would not help.

    ``metadata.golden_answer`` is the reference *method* — the ordered tool
    calls that solve the problem. It is not the gold value: the official
    evaluator compares against the ``answer`` field. What the chain adds is
    the arithmetic the reference performed, which helps when ``answer`` is a
    symbolic expression the judge cannot evaluate and the model answered with
    its numeric value.

    For a handful of entries the chain's last *valued* step is a helper
    rather than the asked-for quantity (a tension after a speed question, a
    radius of gyration after a composite-metric question), so the chain is
    labelled context-only in the prompt and the judge is not told to prefer
    it. ``metadata.solution_steps`` is the field that does name the final
    quantity — see :func:`_sciagentgym_solution_steps`.

    A named step is rendered only when its output reduces to something
    comparable — a scalar, or a flat bundle of scalars. Steps that *draw*
    something (an artifact path), that return a nested bundle, or that a
    bare-string ``golden_answer`` left with no tool name are skipped.
    Fewer than two usable steps means the chain adds nothing over
    ``answer``, so nothing is rendered and the judge prompt stays
    byte-identical to the official one.
    """
    steps: list[str] = []
    final_value = ""
    for call in problem.golden_calls:
        if not call.tool:
            continue
        text, scalar = _sciagentgym_step_value(call.output)
        if not text:
            continue
        unit = f" {call.units}" if call.units else ""
        note = f"（{call.note}）" if call.note else ""
        steps.append(f"{len(steps) + 1}. {call.tool} → {text}{unit}{note}")
        # Only a step that stands for a single number can be quoted as
        # "the" final result; a bundle of several says which to take.
        final_value = f"{scalar}{unit}" if scalar else ""
    if len(steps) < 2:
        return "", ""
    return "\n".join(steps) + "\n", final_value


def _sciagentgym_solution_steps(problem: Problem) -> str:
    """Render ``metadata.solution_steps`` for the judge.

    The dataset's own prose account of how the entry is meant to be solved,
    one step per line. Its value is that it names the quantity the question
    asks for, which the ``answer`` field does not always hold: on the
    Horwitz-trumpet entry (id 12) ``answer`` is ``2.8%`` — the
    between-laboratory RSD — while the steps read "1. 依据Horwitz喇叭经验关系
    计算…实验室间RSD / 2. 调用 intra_laboratory_rsd 以系数0.5估算班级内最小RSD /
    3. 核对题意并记录最终数值回答". The question asks for step 2's quantity, so
    1.41421 is right and judging against ``answer`` alone marks it 错误.

    Returns ``""`` when the entry carries no steps, which keeps the judge
    prompt byte-identical to the official one for the 30 of 83 entries
    without them.
    """
    steps = [
        str(step).strip()
        for step in (problem.solution_steps or [])
        if str(step or "").strip()
    ]
    if not steps:
        return ""
    # Joined verbatim, not re-numbered: the dataset's own steps already carry
    # their numbering ("1. 依据Horwitz…", "2. 调用 intra_laboratory_rsd…"), so
    # adding our own would render them as "1. 1. 依据Horwitz…".
    return "\n".join(steps) + "\n"


def _sciagentgym_step_value(output: Any) -> tuple[str, str]:
    """Reduce a golden step's output to ``(display_text, sole_number)``.

    Returns ``("", "")`` when the output is not a comparable value — a
    dict of nested structures, an empty container, or a file path.
    ``sole_number`` is non-empty only when the step yields exactly one
    number, which is the one case where the step can be quoted as *the*
    final answer.
    """
    # `bool` is an `int` subclass; a True/False step is not a value.
    if isinstance(output, bool) or output is None:
        return "", ""
    if isinstance(output, (int, float)):
        return f"{output:g}", f"{output:g}"
    if isinstance(output, str):
        text = output.strip()
        if not text or _ARTIFACT_PATH_RE.search(text):
            return "", ""
        return text, ""
    if isinstance(output, dict):
        leaves = [
            (k, v)
            for k, v in output.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        if not leaves or len(leaves) > 6:
            return "", ""
        rendered = ", ".join(f"{k}={v:g}" for k, v in leaves)
        if len(rendered) > 160:
            return "", ""
        sole = f"{leaves[0][1]:g}" if len(leaves) == 1 else ""
        return rendered, sole
    return "", ""


def _numbers_agree(a: str, b: str) -> bool:
    """True when the two texts share a number within 5% (or are both blank)."""
    a_nums, b_nums = extract_numbers(a), extract_numbers(b)
    if not a_nums or not b_nums:
        return False
    return any(
        abs(x - y) <= 0.05 * max(abs(x), abs(y)) for x in a_nums for y in b_nums
    )


def _sciagentgym_resolved_expected(problem: Problem) -> tuple[str, str]:
    """The value to judge against, and why it differs from ``answer``.

    Returns ``(gold, reason)``. ``reason`` is ``""`` when the dataset's
    ``answer`` field stands, and a human-readable explanation when it was
    replaced by a later step of the reference chain.

    The replacement fires only when the ``answer`` field is provably an
    *intermediate*: it agrees with a step of the chain **and the chain
    continues to a later step that carries a value**. Both halves matter.

    - Answer agrees with a non-final step, chain ends on a value: the
      reference computed the ``answer`` field on its way somewhere else, so
      the field is a waypoint. This is the Horwitz-trumpet entry — ``answer``
      is ``2.8%`` (``horwitz_trumpet``, step 1) and the chain ends on
      ``intra_laboratory_rsd`` → 1.41421, which is what the question asks for.
    - Answer agrees with no step: the field is an independent value, not a
      waypoint. The pulley entry is this case — ``answer`` is ``13.1 m/s²``
      and appears nowhere in a chain that runs 22.9 m/s² → -65.5 N. It stays.
    - Answer agrees with the *last* step: they are consistent. It stays.
    - Chain ends on a step with no value (a ``visualize_*`` call): there is
      nothing to move to, so the field stays even if it matched an earlier
      step. Twelve entries in the multi-question dump are this case.

    Across both dumps only the Horwitz-trumpet entry meets every condition,
    so this is deliberately a one-entry rule rather than a general policy.
    """
    answer = (problem.answer or "").strip()
    if (
        not answer
        or not problem.golden_chain_ordered
        or len(problem.golden_calls) < 2
    ):
        return answer, ""
    steps = [_sciagentgym_step_value(call.output) for call in problem.golden_calls]
    last = problem.golden_calls[-1]
    if not steps[-1][0]:
        return answer, ""
    if _numbers_agree(answer, steps[-1][0]):
        return answer, ""
    if not any(text and _numbers_agree(answer, text) for text, _ in steps[:-1]):
        return answer, ""
    unit = f" {last.units}" if isinstance(last.units, str) and last.units else ""
    gold = f"{steps[-1][1] or steps[-1][0]}{unit}"
    reason = (
        f"dataset answer 字段 {answer!r} 是参考解中间步骤的数值"
        f"（{last.tool} 之前的一步），本题所问的最终量为 {gold}"
    )
    return gold, reason


def sciagentgym_expected_display(problem: Problem) -> str:
    """What to show as a SciAgentGYM problem's expected answer.

    The panel exists to explain a verdict, so it leads with the dataset's
    ``answer`` field and says, in the same string, what the judge does when
    that field is not the final quantity. Two disagreements matter and they
    are not the same case:

    - ``metadata.solution_steps`` names the final quantity. The judge is
      told to follow it (see ``llm_judge._ANSWER_SOLUTION_STEPS_BLOCK``), so
      the panel must say so. On the Horwitz-trumpet entry ``answer`` is
      ``2.8%`` while the steps end on the within-laboratory RSD, 1.41421 —
      which the judge accepts, and which is why leaving the panel reading
      "判分依据 = answer 字段" would misdescribe the verdict.
    - The golden chain's last step disagrees. That one is context only: a
      chain frequently ends on a helper rather than the asked-for quantity
      (``calculate_tension`` → -65.5 N after the acceleration question,
      ``calculate_tension_force`` → 5.658 N after the speed question,
      ``radius_of_gyration`` after a composite-metric question), so it is
      shown and labelled, never preferred.
    """
    answer = (problem.answer or "").strip()
    # The judge's gold comes from here, so the panel has to read it from the
    # same place — otherwise the row shows a value the verdict did not come
    # from, which is precisely how a correct 1.4% answer looked misgraded.
    gold, resolution = _sciagentgym_resolved_expected(problem)
    if not problem.golden_chain_ordered or not problem.golden_calls:
        return gold
    last = problem.golden_calls[-1]
    if not last.tool:
        return gold
    value, _ = _sciagentgym_step_value(last.output)
    if not value:
        return gold
    if not answer:
        return value
    if _numbers_agree(answer, value):
        return answer
    # `units` is occasionally a dict (`{'load': 'N', ...}`), which would
    # stringify into noise.
    unit = f" {last.units}" if isinstance(last.units, str) and last.units else ""
    hint = f"（{last.note}）" if isinstance(last.note, str) and last.note else ""
    if resolution:
        # Say which field was set aside and why, so the reader can check the
        # call rather than having to trust it.
        return f"{gold}（判分依据；dataset answer 字段是 {answer}，为中间量）"
    # Here the chain's last step disagrees but the `answer` field still stands
    # (it matches no step of the chain, so it is not an intermediate of it).
    # Say so, and mention the steps when the judge was given them — on those
    # entries the steps, not the field, are what the judge leans on.
    steps_note = "；本题另附解题步骤说明供判断" if problem.solution_steps else ""
    return (
        f"{answer}（判分依据 = dataset answer 字段；"
        f"参考解最后一步为 {value}{unit}{hint}，二者不一致{steps_note}）"
    )


def grade_sciagentgym_llm(
    problem: Problem,
    predicted: str,
) -> tuple[Optional[bool], float, list[str]]:
    """Score SciAgentGYM answers with the official LLM judge.

    Returns ``(verdict, score 0..1, notes)``:

    - ``verdict`` is ``True``/``False`` on a clean judge response and
      ``None`` when the API is unavailable so the caller can fall back to
      the deterministic ``grade_answer`` scorer.
    - ``score`` is ``1.0``/``0.0`` for True/False; ``0.0`` when the
      judge was unavailable.
    """
    cfg = _llm_resolve_judge_config(dataset=SCIAGENTGYM)
    if cfg is None:
        return None, 0.0, ["SciAgentGYM LLM judge not configured"]
    question = (problem.question or "").strip()
    # `metadata.golden_answer` (which the loader puts in `golden_calls`) is
    # the reference *method*, not the gold value — the gold is normally the
    # `answer` field. (An earlier version read the chain from
    # `problem.task_info`, which `Problem.from_raw` never assigns, so that
    # branch was dead code and no chain ever reached the judge.)
    workings, final_value = _sciagentgym_reference_workings(problem)
    expected, resolution = _sciagentgym_resolved_expected(problem)
    if not expected and final_value:
        # No `answer` field at all: the chain's last step is the only
        # reference available, and judging against "" always fails.
        expected = final_value
    # ``metadata.solution_steps`` — the dataset's own prose account of the
    # intended solution. It is what disambiguates entries whose `answer` field
    # holds an intermediate rather than the asked-for quantity; see the note on
    # `llm_judge._ANSWER_SOLUTION_STEPS_BLOCK`. When the gold was already
    # resolved above there is nothing left to disambiguate, and the rule would
    # instead invite the judge to second-guess a gold that is now correct.
    solution_steps = "" if resolution else _sciagentgym_solution_steps(problem)
    # The question's own figures. ``image_paths_local`` holds the copies staged
    # into the run's output directory (what the solver was told to read);
    # falling back to ``image_paths`` covers a direct grader call that skipped
    # staging. Either way these are the QUESTION's charts — unlike
    # ResearchClawBench, where the attached image is the ground-truth target.
    figures = [
        str(path)
        for path in (
            getattr(problem, "image_paths_local", None)
            or getattr(problem, "image_paths", None)
            or []
        )
        if str(path).strip()
    ]
    # Some SciAgentGYM dumps hide the model answer inside a \boxed{...}.
    boxed = _llm_extract_boxed(predicted or "")
    predicted_for_judge = boxed or predicted or ""
    verdict = _llm_judge_correct(
        question=question,
        predicted=predicted_for_judge,
        expected=expected,
        dataset=SCIAGENTGYM,
        workings=workings or None,
        solution_steps=solution_steps or None,
        images=figures or None,
    )
    image_note = _llm_last_image_note() if figures else ""
    if verdict is None:
        # Carry the transport's own reason: an empty completion (a reasoning
        # model spending its whole ``max_tokens`` budget on thinking) and a
        # dead endpoint both land here, and they need different fixes.
        reason = _llm_last_judge_error()
        detail = f" — {reason}" if reason else ""
        failures = [
            f"SciAgentGYM LLM judge call failed "
            f"(model={cfg.model}, base={cfg.api_base}){detail}"
        ]
        if figures and image_note:
            failures.append(f"question figure(s) NOT sent to the judge: {image_note}")
        return None, 0.0, failures
    notes = [
        f"SciAgentGYM official LLM judge (model={cfg.model}, base={cfg.api_base})",
        f"verdict={'正确' if verdict else '错误'}",
    ]
    if resolution:
        # The row must not read "expected: 2.8%" when the verdict came from
        # 1.41421 — that mismatch is what made a correct answer look misgraded.
        notes.append(f"judged against {expected} instead: {resolution}")
    if workings:
        # Echo the chain so a reader can see what the judge was shown. It is
        # context only; the verdict comes from `expected` above.
        notes.append(
            "judge also saw metadata.golden_answer (reference method, "
            "context only): " + workings.replace("\n", " ; ").strip()[:400]
        )
    if solution_steps:
        # This one *does* steer the verdict — it is the evidence for what the
        # question asks for when `answer` disagrees with it. Say so, so a
        # surprising verdict can be traced to the text that caused it.
        notes.append(
            "judge also saw metadata.solution_steps (used to resolve whether "
            "the answer field is the final quantity): "
            + solution_steps.replace("\n", " ; ").strip()[:400]
        )
    if figures and image_note:
        # The verdict exists but was reached without the figures. Say so
        # plainly: an answer that can only be checked against a chart will be
        # marked 错误 by a judge that never saw the chart, and the reader has
        # to be able to see that from the row.
        notes.append(f"question figure(s) NOT sent to the judge: {image_note}")
    elif figures:
        notes.append(
            f"judge also saw {len(figures)} question figure(s) from "
            "metadata.image_path (the question's own charts, not the model's "
            "answer): " + ", ".join(path.rsplit("/", 1)[-1] for path in figures)[:400]
        )
    elif getattr(problem, "image_paths_missing", None):
        # The dataset named figures and none of them were found: the failure is
        # visible instead of looking like a problem that simply has no figures —
        # which is how the dataset-root trap stayed hidden.
        notes.append(
            "metadata.image_path named figure(s) that do not exist on disk and "
            "were not sent: " + ", ".join(problem.image_paths_missing)[:400]
        )
    if boxed:
        notes.append(f"extracted \\boxed{{...}} value: {boxed!r}")
    return verdict, 1.0 if verdict else 0.0, notes


_MADD_MOL_RE = re.compile(r"\| ([A-Za-z0-9@+\-=#\[\]\(\)\\\/\.\*]+) \|")

_MADD_EXPECTED_CASE_CODES: dict[str, str] = {
    "gen_mols_alzheimer": "Alzhmr",
    "gen_mols_multiple_sclerosis": "Sklrz",
    "gen_mols_dyslipidemia": "Dslpdm",
    "gen_mols_acquired_drug_resistance": "TBLET",
    "gen_mols_lung_cancer": "Cnsr",
    "gen_mols_parkinson": "Prkns",
}

_MADD_CASE_ALIASES: dict[str, str] = {
    "alzhmr": "Alzhmr",
    "alzheimer": "Alzhmr",
    "alzheimers": "Alzhmr",
    "alzheimer disease": "Alzhmr",
    "alzheimer s disease": "Alzhmr",
    "sklrz": "Sklrz",
    "sclerosis": "Sklrz",
    "multiple sclerosis": "Sklrz",
    "dslpdm": "Dslpdm",
    "dyslipidemia": "Dslpdm",
    "prkns": "Prkns",
    "parkinson": "Prkns",
    "parkinsons": "Prkns",
    "parkinson disease": "Prkns",
    "parkinson s disease": "Prkns",
    "cnsr": "Cnsr",
    "lung cancer": "Cnsr",
    "kras": "Cnsr",
    "tblet": "TBLET",
    "drug resistance": "TBLET",
    "acquired drug resistance": "TBLET",
    "rndm": "RNDM",
    "random": "RNDM",
}

_MADD_GENERATION_TOOLS = {
    "generate_molecules_by_case",
    "generate_molecules_with_local_madd",
}

_MADD_PHARMA_HELPER_TOOLS = {
    "evaluate_druglikeness",
    "predict_properties_by_smiles",
    "draw_molecules",
    "list_madd_local_checkpoints",
}

_MADD_UNAVAILABLE_EXPECTED_TOOLS = {
    # Present once in the migrated table. This exact MADD nanomaterial
    # shape predictor has not been ported into sciMAS.
    "predict_nanomaterial_shape",
}


def _extract_madd_molecules(texts: Iterable[str]) -> list[str]:
    molecules: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for match in _MADD_MOL_RE.findall(text or ""):
            if match == "Molecules":
                continue
            if not any(ch.isalpha() for ch in match):
                continue
            if match not in seen:
                molecules.append(match)
                seen.add(match)
    return molecules


def grade_madd_answer(problem: Problem, predicted: str) -> tuple[bool, float, list[str]]:
    """Mirror MADD's SSA metric using the migrated golden tool tables.

    Upstream MADD validates that the final answer includes every molecule
    returned by its tool calls. In sciMAS we do not have tool outputs in
    the grader interface, so the migrated ``tools answers`` column serves
    as the deterministic golden table.
    """
    tool_answers = problem.task_info.get("madd_tool_answers", [])
    if not isinstance(tool_answers, list):
        tool_answers = []
    expected_molecules = _extract_madd_molecules(str(item) for item in tool_answers)
    if not expected_molecules:
        correct, score, notes = grade_answer(predicted, problem.answer)
        return correct, score, ["MADD SSA proxy: no golden molecules found", *notes]

    predicted_text = predicted or ""
    matched = [mol for mol in expected_molecules if mol in predicted_text]
    score = len(matched) / len(expected_molecules)
    correct = len(matched) == len(expected_molecules)
    notes = [
        (
            "MADD SSA proxy: "
            f"{len(matched)}/{len(expected_molecules)} expected tool-output "
            "molecule(s) appear in the final answer"
        )
    ]
    missing = [mol for mol in expected_molecules if mol not in predicted_text]
    if missing:
        notes.append("missing molecules: " + ", ".join(missing[:5]))
    return correct, score, notes


def _flatten_argument_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(
            f"{key} {_flatten_argument_text(val)}" for key, val in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten_argument_text(item) for item in value)
    return str(value)


def _madd_case_code_from_text(text: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    if normalized in _MADD_CASE_ALIASES:
        return _MADD_CASE_ALIASES[normalized]
    for alias, code in sorted(_MADD_CASE_ALIASES.items(), key=lambda item: -len(item[0])):
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            return code
    return None


def _madd_tool_matches(expected: str, call: ToolCall, predicted_answer: str) -> bool:
    expected = expected.strip()
    if call.tool == expected:
        return True
    if expected == "make_answer_chat_model":
        return bool(_norm(predicted_answer))
    if expected == "request_mols_generation":
        if call.tool not in _MADD_GENERATION_TOOLS:
            return False
        case_code = _madd_case_code_from_text(_flatten_argument_text(call.arguments))
        return case_code in (None, "RNDM")
    expected_code = _MADD_EXPECTED_CASE_CODES.get(expected)
    if expected_code is None:
        return False
    if call.tool not in _MADD_GENERATION_TOOLS:
        return False
    case_code = _madd_case_code_from_text(_flatten_argument_text(call.arguments))
    return case_code == expected_code


def grade_madd_tool_coverage(
    expected: Iterable[str],
    tool_calls: list[ToolCall],
    predicted_answer: str,
) -> tuple[float, list[str], list[str], list[str]]:
    """Map MADD's expected function names to sciMAS pharma MCP calls."""
    expected_tools = [tool for tool in expected if tool]
    missing: list[str] = []
    unavailable: list[str] = []
    matched = 0
    for expected_tool in expected_tools:
        if expected_tool in _MADD_UNAVAILABLE_EXPECTED_TOOLS:
            unavailable.append(expected_tool)
            continue
        if expected_tool == "make_answer_chat_model":
            if _norm(predicted_answer):
                matched += 1
            else:
                missing.append(expected_tool)
            continue
        if any(_madd_tool_matches(expected_tool, call, predicted_answer) for call in tool_calls):
            matched += 1
        else:
            missing.append(expected_tool)

    denominator = len(expected_tools) - len(unavailable)
    coverage = matched / denominator if denominator > 0 else 0.0

    acceptable_called: set[str] = set()
    if any(tool in _MADD_EXPECTED_CASE_CODES for tool in expected_tools):
        acceptable_called.update(_MADD_GENERATION_TOOLS)
        acceptable_called.update(_MADD_PHARMA_HELPER_TOOLS)
    if "request_mols_generation" in expected_tools:
        acceptable_called.update(_MADD_GENERATION_TOOLS)
        acceptable_called.update(_MADD_PHARMA_HELPER_TOOLS)
    called = {call.tool for call in tool_calls}
    extra = sorted(called - acceptable_called)
    return coverage, missing, extra, unavailable


_RUBRIC_WEIGHT_RE = re.compile(r"([+-]\d+(?:\.\d+)?)")


def _parse_rubric_weight(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    match = _RUBRIC_WEIGHT_RE.search(str(value or ""))
    return float(match.group(1)) if match else 0.0


def grade_drug_discovery_bench_answer(
    problem: Problem,
    predicted: str,
) -> tuple[bool, float, list[str]]:
    """Local deterministic proxy for DrugDiscoveryBench outcome grading.

    DrugDiscoveryBench's real score is produced by an LLM judge against
    expert rubrics, including process rubrics over the trajectory. That
    official path is intentionally outside this pure grader. This proxy
    uses populated ground truth when available, otherwise weighted token
    overlap against positive outcome-rubric text. If the public placeholder
    rubrics are still empty, it returns an explicit zero with a note.
    """
    task_info = problem.task_info or {}
    outcome_rubrics = task_info.get("ddb_outcome_rubrics", [])
    process_rubrics = task_info.get("ddb_process_rubrics", [])
    if not isinstance(outcome_rubrics, list):
        outcome_rubrics = []
    if not isinstance(process_rubrics, list):
        process_rubrics = []

    notes: list[str] = ["DrugDiscoveryBench local proxy; official score requires the DDB LLM judge"]
    if process_rubrics:
        notes.append(
            f"{len(process_rubrics)} process rubric(s) present but not scored by the local final-answer proxy"
        )

    ground_truth = str(task_info.get("ddb_ground_truth") or problem.answer or "").strip()
    gt_correct = False
    gt_score = 0.0
    if ground_truth:
        gt_correct, gt_score, gt_notes = grade_answer(predicted, ground_truth)
        notes.extend(f"ground truth: {note}" for note in gt_notes)

    if not outcome_rubrics:
        if ground_truth:
            return gt_correct, gt_score, notes
        notes.append(
            "no populated ground_truth or outcome_rubrics; run DrugDiscoveryBench's "
            "scripts/populate_rubrics.py before expecting benchmark-quality scores"
        )
        return False, 0.0, notes

    predicted_tokens = _token_set(predicted)
    earned = 0.0
    possible = 0.0
    rubric_notes: list[str] = []
    for idx, item in enumerate(outcome_rubrics, start=1):
        if not isinstance(item, dict):
            continue
        weight = _parse_rubric_weight(item.get("weight"))
        title = str(item.get("title", "")).strip()
        justification = str(item.get("justification", "")).strip()
        criterion_text = " ".join(part for part in (title, justification) if part)
        item_score = _text_overlap(predicted_tokens, criterion_text)
        if weight > 0:
            possible += weight
            earned += weight * item_score
        elif weight < 0 and item_score >= 0.5:
            earned += weight
        rubric_notes.append(
            f"outcome rubric {idx} proxy={item_score:.2f} weight={weight:g}"
        )

    rubric_score = max(0.0, min(1.0, earned / possible if possible > 0 else 0.0))
    score = max(gt_score, rubric_score)
    correct = gt_correct or score >= 0.5
    notes.extend(rubric_notes)
    notes.append(f"outcome rubric proxy score={rubric_score:.2f}")
    return correct, score, notes


_MCQ_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-H])(?![A-Za-z])", re.I)
_SCIPREDICT_RANGE_RE = re.compile(
    r"""
    (?P<lo>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*
    (?:-|–|—|\bto\b|\band\b)\s*
    (?P<hi>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)
    """,
    re.I | re.VERBOSE,
)


def _unique_letters(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for match in _MCQ_LETTER_RE.finditer(text or ""):
        letter = match.group(1).upper()
        if letter not in seen:
            out.append(letter)
            seen.add(letter)
    return out


def _scipredict_mcq_letters(text: str, *, expected: bool = False) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if expected:
        # CLEAN_GTA is usually the option letter(s). Restrict long
        # prose answers to avoid collecting incidental capital letters.
        return _unique_letters(text if len(text) <= 80 else text[:80])

    answer_like = re.search(
        r"(?:answer|option|choice|select(?:ed)?|prediction)\s*(?:is|:|-)?\s*"
        r"([A-H](?:\s*[,/]\s*[A-H])*)",
        text,
        re.I,
    )
    if answer_like:
        return _unique_letters(answer_like.group(1))
    # Fall back to the opening span; many model answers start with
    # "B" or "B) ...". Avoid scanning the full explanation.
    return _unique_letters(text[:120])


def _scipredict_numeric_ranges(text: str) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for match in _SCIPREDICT_RANGE_RE.finditer(text or ""):
        try:
            lo = float(match.group("lo"))
            hi = float(match.group("hi"))
        except ValueError:
            continue
        ranges.append((min(lo, hi), max(lo, hi)))
    return ranges


def grade_scipredict_answer(
    problem: Problem,
    predicted: str,
) -> tuple[bool, float, list[str]]:
    """Local scorer for SciPredict.

    MCQ and numerical questions are scored deterministically. Free-form
    questions use the migrated expert rubrics when present, via the same
    lightweight checklist-overlap proxy used for ResearchClawBench. The
    official SciPredict leaderboard still relies on judge-based grading
    for free-form tasks.
    """
    task_info = problem.task_info or {}
    pq_format = str(task_info.get("scipredict_pq_format") or "").lower()
    expected = str(
        task_info.get("scipredict_clean_ground_truth")
        or problem.answer
        or task_info.get("scipredict_ground_truth")
        or ""
    ).strip()
    notes = ["SciPredict local scorer"]

    if "mcq" in pq_format or "multiple" in pq_format:
        exp_letters = _scipredict_mcq_letters(expected, expected=True)
        got_letters = _scipredict_mcq_letters(predicted)
        if exp_letters:
            exp_set = set(exp_letters)
            got_set = set(got_letters)
            if got_set == exp_set:
                return True, 1.0, [*notes, f"MCQ exact option match: {', '.join(exp_letters)}"]
            overlap = len(exp_set & got_set)
            score = overlap / len(exp_set) if exp_set else 0.0
            return False, score, [
                *notes,
                f"MCQ option mismatch: expected {exp_letters}, predicted {got_letters or 'none'}",
            ]
        correct, score, answer_notes = grade_answer(predicted, expected)
        return correct, score, [*notes, "MCQ fallback text match", *answer_notes]

    if "numerical" in pq_format or "numeric" in pq_format or "value" in pq_format:
        ranges = _scipredict_numeric_ranges(expected)
        pred_nums = extract_numbers(predicted)
        if ranges:
            matched = 0
            for lo, hi in ranges:
                if any(lo <= value <= hi for value in pred_nums):
                    matched += 1
            score = matched / len(ranges)
            return matched == len(ranges), score, [
                *notes,
                f"numeric range match {matched}/{len(ranges)}",
            ]
        correct, score, answer_notes = grade_answer(predicted, expected)
        return correct, score, [*notes, "numeric fallback answer match", *answer_notes]

    if problem.checklist:
        correct, score, checklist_notes = grade_checklist_answer(predicted, problem.checklist)
        return correct, score, [
            *notes,
            "free-form proxy over migrated expert rubrics; official scoring uses a judge",
            *checklist_notes,
        ]

    correct, score, answer_notes = grade_answer(predicted, expected)
    return correct, score, [*notes, "free-form fallback answer match", *answer_notes]


def _json_from_text(text: str) -> dict[str, Any]:
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_object_from_answer(text: str) -> dict[str, Any]:
    parsed = _json_from_text(text)
    if parsed:
        return parsed
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return {}
    return _json_from_text(match.group(0))


def _read_json_file(path_text: str) -> dict[str, Any]:
    if not path_text:
        return {}
    try:
        parsed = json.loads(open(path_text, "r", encoding="utf-8").read())
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_score_from_official_result(result: dict[str, Any]) -> tuple[bool, float, list[str]] | None:
    notes = ["SMDDBench official evaluator result detected"]
    status = str(result.get("status") or "").strip().lower()
    steps = result.get("steps")
    if status in {"passed", "failed", "errored"}:
        if isinstance(steps, list) and steps:
            passed_steps = sum(
                1 for step in steps
                if isinstance(step, dict) and str(step.get("status") or "").lower() == "passed"
            )
            score = passed_steps / len(steps)
            return status == "passed", score, [
                *notes,
                f"status={status}",
                f"passed_steps={passed_steps}/{len(steps)}",
            ]
        return status == "passed", 1.0 if status == "passed" else 0.0, [
            *notes,
            f"status={status}",
        ]
    for key in ("score", "overall_score", "avg_score", "mean_score", "total_score"):
        value = result.get(key)
        if isinstance(value, (int, float)):
            score = float(value)
            if score > 1.0:
                score = score / 100.0
            score = max(0.0, min(1.0, score))
            return score >= 1.0, score, [*notes, f"{key}={value}"]
    for key in ("success", "passed", "pass", "valid"):
        value = result.get(key)
        if isinstance(value, bool):
            return value, 1.0 if value else 0.0, [*notes, f"{key}={value}"]
    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        numeric = [
            float(value)
            for value in metrics.values()
            if isinstance(value, (int, float))
        ]
        if numeric:
            score = sum(max(0.0, min(1.0, value if value <= 1 else value / 100.0)) for value in numeric) / len(numeric)
            return score >= 1.0, score, [*notes, f"mean metric score over {len(numeric)} numeric metric(s)"]
    return None


_SMILES_TOKEN_RE = re.compile(r"\b[A-Za-z0-9@+\-=#\[\]\(\)\\/\.]{6,}\b")


def _smdd_format_hint(predicted: str, output_file: str, task_text: str) -> str:
    text = predicted or ""
    suffix = output_file.lower().rsplit(".", 1)[-1] if "." in output_file else ""
    if suffix == "json" or "json" in output_file.lower():
        return "JSON-like artifact detected" if _json_from_text(text) else "expected JSON-like artifact"
    if suffix in {"smi", "smiles"} or "smiles" in task_text.lower() or "smiles" in output_file.lower():
        return "SMILES-like token detected" if _SMILES_TOKEN_RE.search(text) else "expected SMILES-like molecule text"
    if suffix == "sdf" or "sdf" in output_file.lower():
        return "SDF-like markers detected" if ("$$$$" in text or "V2000" in text or "V3000" in text) else "expected SDF-like artifact"
    if suffix in {"csv", "tsv"}:
        return "tabular-looking artifact detected" if ("," in text or "\t" in text) else "expected tabular artifact"
    if suffix == "py":
        return "Python-like artifact detected" if ("def " in text or "import " in text) else "expected Python artifact"
    return "nonempty artifact text detected" if text.strip() else "empty artifact text"


def grade_smdd_bench_answer(
    problem: Problem,
    predicted: str,
) -> tuple[bool, float, list[str]]:
    """Read official SMDDBench evaluator output when available.

    SMDD-Bench's true score is produced by its Docker evaluator using the
    task files and external molecular tooling. sciMAS's local grader does
    not attempt to reproduce those metrics from final text alone.
    """
    task_info = problem.task_info or {}
    embedded_result = task_info.get("smdd_official_result")
    if isinstance(embedded_result, dict):
        extracted = _extract_score_from_official_result(embedded_result)
        if extracted:
            return extracted

    candidate_paths = task_info.get("smdd_official_result_candidates", [])
    if not isinstance(candidate_paths, list):
        candidate_paths = []
    for path_text in candidate_paths:
        result = _read_json_file(str(path_text))
        if result:
            extracted = _extract_score_from_official_result(result)
            if extracted:
                return extracted

    output_file = str(task_info.get("smdd_output_file") or "")
    task_text = " ".join(
        str(task_info.get(key) or "")
        for key in ("smdd_type_name", "smdd_description", "smdd_output_file")
    )
    return False, 0.0, [
        "SMDDBench official score unavailable in local grader",
        "Run the upstream SMDD-Bench Docker evaluator against the submitted artifact to obtain the benchmark score.",
        "Format proxy: " + _smdd_format_hint(predicted, output_file, task_text),
    ]


def _norm_symbol(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _json_list_field_casefold(data: dict[str, Any], key: str) -> set[str]:
    value = data.get(key)
    if isinstance(value, str):
        return {_norm_symbol(value)} if value.strip() else set()
    if isinstance(value, list):
        return {
            _norm_symbol(str(item))
            for item in value
            if str(item).strip()
        }
    return set()


def _contains_normalized_token(predicted: str, expected: str) -> bool:
    expected_norm = _norm_symbol(expected)
    if not expected_norm:
        return False
    if _norm_symbol(predicted) == expected_norm:
        return True
    for token in re.findall(r"[A-Za-z0-9_.:-]+", predicted or ""):
        if _norm_symbol(token) == expected_norm:
            return True
    return False


def grade_biomni_eval1_answer(
    problem: Problem,
    predicted: str,
) -> tuple[bool, float, list[str]]:
    """Local deterministic scorer for biomni/Eval1 answer formats."""
    task_info = problem.task_info or {}
    task_name = str(task_info.get("biomni_eval1_task_name") or problem.topic or "")
    task_key = re.sub(r"[^a-z0-9]+", "_", task_name.lower()).strip("_")
    expected = str(task_info.get("biomni_eval1_answer") or problem.answer or "").strip()
    notes = ["BiomniEval1 local scorer"]

    if task_key in {"lab_test_analyzing", "crispr_screening", "crispr_delivery", "lab_bench_dbqa", "lab_bench_seqqa"}:
        exp_letters = _scipredict_mcq_letters(expected, expected=True)
        got_letters = _scipredict_mcq_letters(predicted)
        if exp_letters:
            correct = bool(got_letters) and got_letters[0].upper() == exp_letters[0].upper()
            return correct, 1.0 if correct else 0.0, [
                *notes,
                f"choice expected={exp_letters[:1]} predicted={got_letters[:1] or 'none'}",
            ]
        correct = _norm(expected) == _norm(predicted)
        return correct, 1.0 if correct else 0.0, [*notes, "choice fallback text match"]

    if task_key in {
        "gene_name_conversion",
        "patient_gene_detection",
        "screen_gene_retrieval",
        "gwas_causal_gene_pharmaprojects",
        "gwas_causal_gene_opentargets",
        "gwas_causal_gene_gwas_catalog",
        "gwas_variant_prioritization",
    }:
        if task_key == "patient_gene_detection":
            exp_json = _json_object_from_answer(expected)
            exp_genes = _json_list_field_casefold(exp_json, "causal_gene")
            if exp_genes:
                got_json = _json_object_from_answer(predicted)
                got_genes = _json_list_field_casefold(got_json, "causal_gene")
                correct = got_genes == exp_genes
                score = len(exp_genes & got_genes) / len(exp_genes)
                return correct, 1.0 if correct else score, [
                    *notes,
                    f"causal_gene expected={sorted(exp_genes)} predicted={sorted(got_genes)}",
                ]
        correct = _contains_normalized_token(predicted, expected)
        return correct, 1.0 if correct else 0.0, [*notes, "case-insensitive identifier match"]

    if task_key == "variant_pathogenicity":
        correct = _norm(expected) == _norm(predicted)
        return correct, 1.0 if correct else 0.0, [*notes, "variant pathogenicity exact normalized match"]

    if task_key == "rare_disease_diagnosis":
        exp_json = _json_object_from_answer(expected)
        got_json = _json_object_from_answer(predicted)
        exp_omim = _norm_symbol(str(exp_json.get("OMIM_ID") or exp_json.get("omim_id") or ""))
        got_omim = _norm_symbol(str(got_json.get("OMIM_ID") or got_json.get("omim_id") or ""))
        if exp_omim:
            correct = got_omim == exp_omim
            return correct, 1.0 if correct else 0.0, [*notes, f"OMIM_ID expected={exp_omim} predicted={got_omim or 'none'}"]
        correct = _norm(expected) == _norm(predicted)
        return correct, 1.0 if correct else 0.0, [*notes, "rare-disease fallback text match"]

    if task_key == "patient_gene_detection_json":
        exp_json = _json_object_from_answer(expected)
        got_json = _json_object_from_answer(predicted)
        exp_genes = _json_list_field_casefold(exp_json, "causal_gene")
        got_genes = _json_list_field_casefold(got_json, "causal_gene")
        if exp_genes:
            correct = got_genes == exp_genes
            score = len(exp_genes & got_genes) / len(exp_genes)
            return correct, 1.0 if correct else score, [
                *notes,
                f"causal_gene expected={sorted(exp_genes)} predicted={sorted(got_genes)}",
            ]
        correct = _norm(expected) == _norm(predicted)
        return correct, 1.0 if correct else 0.0, [*notes, "patient-gene fallback text match"]

    correct, score, answer_notes = grade_answer(predicted, expected)
    return correct, score, [*notes, "generic answer matcher", *answer_notes]


# ---------------------------------------------------------------------------
# 2. Tool coverage
# ---------------------------------------------------------------------------

def grade_tool_coverage(
    expected: Iterable[str],
    called: Iterable[str],
    *,
    available_tools: Optional[set[str]] = None,
) -> tuple[float, list[str], list[str], list[str]]:
    """Compare expected tools vs. called tools (short names without the
    ``mcp__<server>__`` prefix).

    Returns ``(coverage, missing, extra, unavailable)`` where:
      - coverage  = |called ∩ expected| / |expected|  (0 if expected empty)
      - missing   = expected − called
      - extra     = called − expected  (informational only)
      - unavailable = expected ∩ complement(available_tools), tools the
                      sciMAS instance doesn't ship.
    """
    exp = set(expected)
    got = set(called)
    available = set(available_tools) if available_tools is not None else None

    missing = sorted(exp - got)
    extra = sorted(got - exp)
    unavailable: list[str] = []
    if available is not None:
        unavailable = sorted(t for t in exp if t not in available)
        # Tools unavailable shouldn't count against coverage: drop them
        # from the denominator.
        denom = len(exp) - len(unavailable)
        cov = (len(exp & got) / denom) if denom > 0 else 0.0
    else:
        cov = (len(exp & got) / len(exp)) if exp else 0.0
    return cov, missing, extra, unavailable


# ---------------------------------------------------------------------------
# 3. End-to-end grading
# ---------------------------------------------------------------------------

def grade_problem(
    problem: Problem,
    *,
    predicted_answer: str,
    tool_calls: list[ToolCall],
    available_tools: Optional[set[str]] = None,
    cost_usd: float = 0.0,
    duration_ms: int = 0,
    error: str = "",
) -> GradingResult:
    short_called = [tc.tool for tc in tool_calls]
    checklist_score = 0.0
    # Display only — the graders below compare against `problem.answer`
    # directly. SciAgentGYM overrides it so the report doesn't show a
    # reference the grader never used (see sciagentgym_expected_display).
    expected_display = problem.answer
    if problem.dataset_name == MADD:
        correct, score, notes = grade_madd_answer(problem, predicted_answer)
        cov, missing, extra, unavailable = grade_madd_tool_coverage(
            problem.expected_tools, tool_calls, predicted_answer
        )
    elif problem.dataset_name == SCIPREDICT:
        correct, score, notes = grade_scipredict_answer(problem, predicted_answer)
        cov, missing, extra, unavailable = grade_tool_coverage(
            problem.expected_tools, short_called, available_tools=available_tools
        )
    elif problem.dataset_name == SMDD_BENCH:
        correct, score, notes = grade_smdd_bench_answer(problem, predicted_answer)
        cov, missing, extra, unavailable = grade_tool_coverage(
            problem.expected_tools, short_called, available_tools=available_tools
        )
    elif problem.dataset_name == BIOMNI_EVAL1:
        correct, score, notes = grade_biomni_eval1_answer(problem, predicted_answer)
        cov, missing, extra, unavailable = grade_tool_coverage(
            problem.expected_tools, short_called, available_tools=available_tools
        )
    elif problem.dataset_name == DRUG_DISCOVERY_BENCH:
        correct, score, notes = grade_drug_discovery_bench_answer(problem, predicted_answer)
        cov, missing, extra, unavailable = grade_tool_coverage(
            problem.expected_tools, short_called, available_tools=available_tools
        )
    elif problem.dataset_name == RESEARCH_CLAW_BENCH and problem.checklist:
        if _has_judge_creds():
            try:
                correct, score, notes = grade_research_claw_official(
                    problem, predicted_answer
                )
                checklist_score = score
            except Exception as exc:
                # Never break a batch run because the judge is down.
                correct, score, notes = grade_checklist_answer(
                    predicted_answer, problem.checklist
                )
                checklist_score = score
                notes = [
                    f"Official judge failed ({type(exc).__name__}: {exc}); "
                    "fell back to local proxy.",
                    *notes,
                ]
        else:
            correct, score, notes = grade_checklist_answer(predicted_answer, problem.checklist)
            checklist_score = score
        cov, missing, extra, unavailable = grade_tool_coverage(
            problem.expected_tools, short_called, available_tools=available_tools
        )
    elif problem.dataset_name == SCIAGENTGYM:
        expected_display = sciagentgym_expected_display(problem)
        verdict, llm_score, llm_notes = grade_sciagentgym_llm(problem, predicted_answer)
        if verdict is not None:
            correct, score, notes = bool(verdict), llm_score, llm_notes
        else:
            # Deterministic fallback. ``llm_notes`` says *why* the judge did
            # not rule (not configured vs. call failed); dropping it made the
            # two indistinguishable in every report.
            correct, score, notes = grade_answer(predicted_answer, problem.answer)
            notes = [
                *llm_notes,
                "SciAgentGYM fell back to the deterministic grader:",
                *notes,
            ]
        cov, missing, extra, unavailable = grade_tool_coverage(
            problem.expected_tools, short_called, available_tools=available_tools
        )
    else:
        correct, score, notes = grade_answer(predicted_answer, problem.answer)
        cov, missing, extra, unavailable = grade_tool_coverage(
            problem.expected_tools, short_called, available_tools=available_tools
        )
    return GradingResult(
        problem_id=problem.id,
        dataset_name=problem.dataset_name,
        subject=problem.subject,
        topic=problem.topic,
        expected_answer=expected_display,
        predicted_answer=predicted_answer,
        expected_tools=list(problem.expected_tools),
        called_tools=short_called,
        missing_tools=missing,
        extra_tools=extra + unavailable,  # lump unavailable into extras
        answer_correct=correct,
        answer_score=score,
        answer_notes=notes,
        tool_coverage=cov,
        checklist_score=checklist_score,
        cost_usd=cost_usd,
        duration_ms=duration_ms,
        error=error,
    )


__all__ = [
    "GradingResult",
    "ToolCall",
    "grade_answer",
    "grade_biomni_eval1_answer",
    "grade_checklist_answer",
    "grade_drug_discovery_bench_answer",
    "grade_madd_answer",
    "grade_madd_tool_coverage",
    "grade_problem",
    "grade_research_claw_official",
    "grade_sciagentgym_llm",
    "grade_scipredict_answer",
    "grade_smdd_bench_answer",
    "grade_tool_coverage",
    "get_judge_cache_dir",
    "set_judge_cache_dir",
]
