"""Load benchmark problems for the sciMAS evaluation harness.

Supported datasets:

* SciAgentGYM: two flat JSON dumps

  dataset/refine_merged_multi_questions.json   (83 entries, all subjects)
  dataset/refine_merged_single_questions.json  (48 entries, all subjects)

Each entry has shape:

  {
    "id": <int>,                            # synthetic, 1..N
    "filename": "failed_questions_<id>.json",
    "question": "<chinese / english text>",
    "answer": "<expected final answer string>",
    "metadata": {
        "subject": "Chemistry" | ...,
        "topic":   "Analytical Chemistry" | ...,
        "image_path": [...],                # rendered chart / diagram
        "solution_steps": [...],
        "tool_expected": [...],             # names the model SHOULD call
        # The reference solution chain: either an ordered list of steps
        # (refine_merged_multi_questions) or a dict keyed by tool name
        # (refine_merged_single_questions). Each step is
        # {call, inputs, output, units?, note?}; some are bare strings.
        "golden_answer": [...],
        "original_question_id": "<int or string>",
    },
    "usage_tool_protocol": [...],           # JSON-schema function specs
    ...
  }

* ResearchClawBench: a Hugging Face dataset table loaded with
  ``datasets.load_dataset("InternScience/ResearchClawBench")``. Each row
  points at a ``tasks/<TaskID>/...`` workspace with raw data, related
  papers, a target paper, target figures, and a checklist rubric.

* MADD: the pharmaceutical multi-agent benchmark/result table migrated
  from ``MADD-main/examples/large_ds_chemical_result.xlsx``. Each row
  contains a user-facing task in ``content`` plus hidden evaluation
  fields for expected MADD tool choices and molecule tables.

* DrugDiscoveryBench: Harbor-formatted biomedical tasks migrated from
  ``DrugDiscoveryBench-main/benchmark``. Each task directory contains an
  ``instruction.md`` prompt, optional ``environment/inputs`` files, a
  ``task.toml`` Harbor config, and ``tests/rubrics.json`` for official
  LLM-judge scoring.

* SciPredict: CSV-formatted scientific outcome-prediction tasks from
  ``scaleapi/scipredict`` / ``ScaleAI/SciPredict``. Each row contains an
  experimental setup, measurement description, prediction question,
  optional expert background knowledge, and a hidden answer.

* SMDDBench: artifact-generation drug-discovery tasks from
  ``t7rs/SMDD-Bench`` / ``extremelyconfused/smdd-bench``. Each task
  directory contains a ``task.yaml`` specification plus task assets; the
  official benchmark score is produced by the upstream Docker evaluator.

* BiomniEval1: answer-format-constrained biomedical QA tasks from
  ``biomni/Eval1``. Each row contains a prompt, task type, split, and
  hidden answer; local grading mirrors the public answer-format rules.

Both datasets are flattened into a `Problem` dataclass that the runner /
grader work with directly. Filtering is by subject / topic / id list /
substring match.

A SciAgentGYM entry may also name question figures in `metadata.image_path`.
Those names are relative to the *upstream repo root* (``gym/test_images/…``),
which is not where the JSON dumps are read from, so they are resolved through
their own candidate-root list — see `_sciagentgym_image_roots`. Names that
resolve to nothing are dropped and reported on `Problem.image_paths_missing`
rather than advertised as paths to open.

By default, SciAgentGYM paths are resolved relative to a `SCIENT_GYM_ROOT`
env var or fall back to the local ``dataset/`` directory. ResearchClawBench
paths are resolved relative to `RESEARCH_CLAW_BENCH_ROOT` or
``dataset/ResearchClawBench/repo``. MADD paths are resolved relative to
`MADD_BENCHMARK_ROOT`, `MADD_ROOT`, local ``dataset/MADD``, or a sibling
``MADD-main`` checkout. DrugDiscoveryBench paths are resolved relative to
`DRUG_DISCOVERY_BENCH_ROOT`, local ``dataset/DrugDiscoveryBench``, or a
sibling ``DrugDiscoveryBench-main`` checkout.
SMDDBench paths are resolved relative to `SMDD_BENCH_ROOT`, local
``dataset/SMDDBench``, or a sibling ``SMDD-Bench`` checkout.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


SCIAGENTGYM = "SciAgentGYM"
RESEARCH_CLAW_BENCH = "ResearchClawBench"
MADD = "MADD"
DRUG_DISCOVERY_BENCH = "DrugDiscoveryBench"
SCIPREDICT = "SciPredict"
SMDD_BENCH = "SMDDBench"
BIOMNI_EVAL1 = "BiomniEval1"
DATASET_CHOICES = (
    SCIAGENTGYM,
    RESEARCH_CLAW_BENCH,
    MADD,
    DRUG_DISCOVERY_BENCH,
    SCIPREDICT,
    SMDD_BENCH,
    BIOMNI_EVAL1,
)
RESEARCH_CLAW_BENCH_REPO_ID = "InternScience/ResearchClawBench"
SCIPREDICT_REPO_ID = "ScaleAI/SciPredict"
BIOMNI_EVAL1_REPO_ID = "biomni/Eval1"

_DATASET_ALIASES = {
    "sciagentgym": SCIAGENTGYM,
    "sciagentgymmain": SCIAGENTGYM,
    "scientgym": SCIAGENTGYM,
    "researchclawbench": RESEARCH_CLAW_BENCH,
    "researchclaw": RESEARCH_CLAW_BENCH,
    "rcb": RESEARCH_CLAW_BENCH,
    "madd": MADD,
    "maddbenchmark": MADD,
    "maddmain": MADD,
    "madd-main": MADD,
    "drugdiscoverybench": DRUG_DISCOVERY_BENCH,
    "drugdiscoverybenchmark": DRUG_DISCOVERY_BENCH,
    "ddb": DRUG_DISCOVERY_BENCH,
    "drugdiscoverybenchmain": DRUG_DISCOVERY_BENCH,
    "drugdiscoverybench-main": DRUG_DISCOVERY_BENCH,
    "scipredict": SCIPREDICT,
    "sci-predict": SCIPREDICT,
    "sci_predict": SCIPREDICT,
    "scipred": SCIPREDICT,
    "smddbench": SMDD_BENCH,
    "smdd": SMDD_BENCH,
    "smdd-bench": SMDD_BENCH,
    "smdd_bench": SMDD_BENCH,
    "biomnieval1": BIOMNI_EVAL1,
    "biomni-eval1": BIOMNI_EVAL1,
    "biomni_eval1": BIOMNI_EVAL1,
    "eval1": BIOMNI_EVAL1,
}

DEFAULT_DATASETS = (
    "refine_merged_multi_questions.json",
    "refine_merged_single_questions.json",
)

MADD_BENCHMARK_FILES = (
    "benchmark/large_ds_chemical_result.jsonl",
    "benchmark/large_ds_chemical_result.xlsx",
    "examples/large_ds_chemical_result.xlsx",
)

SCIPREDICT_MAIN_FILES = (
    "data/main_ds.csv",
    "data/dataset.csv",
    "main_ds.csv",
    "dataset.csv",
)

SCIPREDICT_RUBRICS_FILES = (
    "data/rubrics.csv",
    "rubrics.csv",
)

SMDD_TASK_DIR_NAMES = (
    "tasks",
    "tasks_lite",
    "tasks_4_4_final/tasks",
    "tasks_4_4_final",
    "benchmark/tasks",
    "SMDD-Bench/tasks",
)

SMDD_TYPE_NAMES = {
    "001": "2D Pharmacophore Identification",
    "002": "Interaction Point Discovery",
    "003": "Scaffold Hopping",
    "004": "Lead Optimization",
    "005": "Fragment Assembly",
}

BIOMNI_EVAL1_FILES = (
    "biomni_eval1_dataset.parquet",
    "data/biomni_eval1_dataset.parquet",
    "data/test-00000-of-00001.parquet",
    "test-00000-of-00001.parquet",
    "data/test.parquet",
    "test.parquet",
    "data/eval1.parquet",
    "eval1.parquet",
    "data/eval1.csv",
    "eval1.csv",
    "data/eval1.jsonl",
    "eval1.jsonl",
)


def normalize_dataset_name(name: Optional[str]) -> str:
    """Return the canonical dataset display name."""
    if not name:
        return SCIAGENTGYM
    key = re.sub(r"[^a-z0-9]+", "", name.strip().lower())
    if key in _DATASET_ALIASES:
        return _DATASET_ALIASES[key]
    choices = ", ".join(DATASET_CHOICES)
    raise ValueError(f"unknown dataset {name!r}; choose one of: {choices}")


def dataset_dir_name(name: Optional[str]) -> str:
    """Filesystem-safe output directory segment for a dataset name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", normalize_dataset_name(name))


def default_dataset_root(dataset_name: str = SCIAGENTGYM) -> Path:
    """Resolve the default local root for a supported dataset.

    SciAgentGYM order:
      1. ``$SCIENT_GYM_ROOT`` env var if set
      2. ``<sciMAS>/dataset`` — local copy inside the sciMAS package
      3. ``<repo_parent>/SciAgentGYM-main`` — sibling of sciMAS/
      4. ``~/Documents/games/SciAgentGYM-main`` — user home fallback

    ResearchClawBench order:
      1. ``$RESEARCH_CLAW_BENCH_ROOT`` or ``$RESEARCHCLAWBENCH_ROOT``
      2. ``<sciMAS>/dataset/ResearchClawBench/repo``
      3. ``<sciMAS>/dataset/ResearchClawBench``

    MADD order:
      1. ``$MADD_BENCHMARK_ROOT`` or ``$MADD_ROOT``
      2. ``<sciMAS>/dataset/MADD``
      3. ``<repo_parent>/MADD-main``
      4. ``~/Documents/games/MADD-main``

    DrugDiscoveryBench order:
      1. ``$DRUG_DISCOVERY_BENCH_ROOT`` or ``$DRUGDISCOVERYBENCH_ROOT``
      2. ``<sciMAS>/dataset/DrugDiscoveryBench``
      3. ``<repo_parent>/DrugDiscoveryBench-main``
      4. ``~/Documents/games/DrugDiscoveryBench-main``

    SciPredict order:
      1. ``$SCIPREDICT_ROOT`` or ``$SCI_PREDICT_ROOT``
      2. ``<sciMAS>/dataset/SciPredict``
      3. ``<repo_parent>/scipredict``
      4. ``~/Documents/games/scipredict``

    SMDDBench order:
      1. ``$SMDD_BENCH_ROOT`` or ``$SMDDBENCH_ROOT``
      2. ``<sciMAS>/dataset/SMDDBench``
      3. ``<repo_parent>/SMDD-Bench``
      4. ``~/Documents/games/SMDD-Bench``

    BiomniEval1 order:
      1. ``$BIOMNI_EVAL1_ROOT`` or ``$BIOMNIEVAL1_ROOT``
      2. ``<sciMAS>/dataset/BiomniEval1``
      3. ``<repo_parent>/Eval1``
      4. ``~/Documents/games/Eval1``
    """
    dataset_name = normalize_dataset_name(dataset_name)
    repo_root = Path(__file__).resolve().parent.parent
    if dataset_name == RESEARCH_CLAW_BENCH:
        env = os.environ.get("RESEARCH_CLAW_BENCH_ROOT") or os.environ.get(
            "RESEARCHCLAWBENCH_ROOT"
        )
        if env:
            return Path(env).expanduser().resolve()
        local_bundle = repo_root / "dataset" / RESEARCH_CLAW_BENCH
        local_repo = local_bundle / "repo"
        if local_repo.exists():
            return local_repo.resolve()
        return local_bundle.resolve()
    if dataset_name == MADD:
        env = os.environ.get("MADD_BENCHMARK_ROOT") or os.environ.get("MADD_ROOT")
        if env:
            return Path(env).expanduser().resolve()
        local_bundle = repo_root / "dataset" / MADD
        if local_bundle.exists():
            return local_bundle.resolve()
        sibling = repo_root.parent / "MADD-main"
        if sibling.exists():
            return sibling.resolve()
        return Path("~/Documents/games/MADD-main").expanduser().resolve()
    if dataset_name == DRUG_DISCOVERY_BENCH:
        env = (
            os.environ.get("DRUG_DISCOVERY_BENCH_ROOT")
            or os.environ.get("DRUGDISCOVERYBENCH_ROOT")
            or os.environ.get("DDB_ROOT")
        )
        if env:
            return Path(env).expanduser().resolve()
        local_bundle = repo_root / "dataset" / DRUG_DISCOVERY_BENCH
        if local_bundle.exists():
            return local_bundle.resolve()
        sibling = repo_root.parent / "DrugDiscoveryBench-main"
        if sibling.exists():
            return sibling.resolve()
        return Path("~/Documents/games/DrugDiscoveryBench-main").expanduser().resolve()
    if dataset_name == SCIPREDICT:
        env = os.environ.get("SCIPREDICT_ROOT") or os.environ.get("SCI_PREDICT_ROOT")
        if env:
            return Path(env).expanduser().resolve()
        local_bundle = repo_root / "dataset" / SCIPREDICT
        if local_bundle.exists():
            return local_bundle.resolve()
        sibling = repo_root.parent / "scipredict"
        if sibling.exists():
            return sibling.resolve()
        return Path("~/Documents/games/scipredict").expanduser().resolve()
    if dataset_name == SMDD_BENCH:
        env = os.environ.get("SMDD_BENCH_ROOT") or os.environ.get("SMDDBENCH_ROOT")
        if env:
            return Path(env).expanduser().resolve()
        local_bundle = repo_root / "dataset" / SMDD_BENCH
        if local_bundle.exists():
            return local_bundle.resolve()
        sibling = repo_root.parent / "SMDD-Bench"
        if sibling.exists():
            return sibling.resolve()
        return Path("~/Documents/games/SMDD-Bench").expanduser().resolve()
    if dataset_name == BIOMNI_EVAL1:
        env = os.environ.get("BIOMNI_EVAL1_ROOT") or os.environ.get("BIOMNIEVAL1_ROOT")
        if env:
            return Path(env).expanduser().resolve()
        local_bundle = repo_root / "dataset" / BIOMNI_EVAL1
        if local_bundle.exists():
            return local_bundle.resolve()
        sibling = repo_root.parent / "Eval1"
        if sibling.exists():
            return sibling.resolve()
        return Path("~/Documents/games/Eval1").expanduser().resolve()

    env = os.environ.get("SCIENT_GYM_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    local = repo_root / "dataset"
    if (local / "refine_merged_multi_questions.json").exists() or (
        local / "refine_merged_single_questions.json"
    ).exists():
        return local.parent
    sibling = repo_root.parent / "SciAgentGYM-main"
    if sibling.exists():
        return sibling
    return Path("~/Documents/games/SciAgentGYM-main").expanduser().resolve()


@dataclass
class GoldenCall:
    tool: str
    inputs: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    units: str = ""
    note: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoldenCall":
        # `inputs` is normally a mapping but can be null, or a list of the
        # call's variants; `dict(None)` and `dict([...])` both raise.
        inputs = data.get("inputs")
        # `units` is occasionally a dict (`{'load': 'N', ...}`); coercing it
        # with `str()` turns it into unreadable noise in reports.
        units = data.get("units")
        note = data.get("note")
        return cls(
            tool=str(data.get("call") or ""),
            inputs=dict(inputs) if isinstance(inputs, dict) else {},
            output=data.get("output"),
            units=units if isinstance(units, str) else "",
            note=note if isinstance(note, str) else "",
        )


@dataclass
class Problem:
    id: int | str
    filename: str
    question: str
    answer: str
    subject: str
    topic: str
    # ``metadata.image_path``, resolved to absolute paths at load time (the
    # dump names them relative to the *upstream repo root*, which is not the
    # directory the dump itself is read from).
    image_paths: list[str] = field(default_factory=list)
    # Figures the dump named that resolved to no existing file. Kept so a
    # report can say "the dataset named 2 figures, neither was found" instead
    # of looking exactly like a problem that has no figures at all — which is
    # how the sciMAS-root trap below stayed invisible.
    image_paths_missing: list[str] = field(default_factory=list)
    # Copies of ``image_paths`` staged into the run's output directory. This is
    # what the solver and the judge actually read: the copies live inside the
    # repo (under the run's output root), so an agent's Read tool can open them
    # without any extra directory permission, and the run output stays
    # self-contained if the dataset checkout later moves.
    image_paths_local: list[str] = field(default_factory=list)
    solution_steps: list[str] = field(default_factory=list)
    expected_tools: list[str] = field(default_factory=list)
    golden_calls: list[GoldenCall] = field(default_factory=list)
    # True when ``metadata.golden_answer`` was a *list*, i.e. an ordered
    # solution whose last step is the result. A dict-shaped chain is an
    # unordered bag of tool calls, so "the last step" means nothing there.
    golden_chain_ordered: bool = False
    original_question_id: str = ""
    usage_tool_protocol: list[dict[str, Any]] = field(default_factory=list)
    source_dataset: str = ""
    dataset_name: str = SCIAGENTGYM
    assets_root: str = ""
    task_info: dict[str, Any] = field(default_factory=dict)
    checklist: list[dict[str, Any]] = field(default_factory=list)
    data_files: list[str] = field(default_factory=list)
    related_work: list[str] = field(default_factory=list)
    target_images: list[str] = field(default_factory=list)
    target_paper: str = ""

    @classmethod
    def from_raw(cls, raw: dict[str, Any], *, source: str) -> "Problem":
        """Build a SciAgentGYM problem from one JSON dump entry."""
        md = raw.get("metadata", {}) or {}
        raw_gold = md.get("golden_answer", []) or []
        ordered = isinstance(raw_gold, list)
        if isinstance(raw_gold, dict):
            # Two shapes are in the wild: a list of step dicts
            # (refine_merged_multi_questions) and a dict keyed by tool
            # name (refine_merged_single_questions). Iterating the latter
            # yields its *keys*, which turned every step into a bare tool
            # name and threw the outputs away.
            raw_gold = list(raw_gold.values())
        gold: list[GoldenCall] = []
        for g in raw_gold:
            if isinstance(g, dict):
                gold.append(GoldenCall.from_dict(g))
            elif isinstance(g, str):
                # Some entries are bare strings (a tool name, or a
                # free-form "expected value"). Wrap as a GoldenCall with
                # no tool name so the grader still sees the text.
                gold.append(GoldenCall(tool="", output=g))
            # else: silently skip non-dict / non-str entries.
        return cls(
            id=int(raw["id"]),
            filename=str(raw.get("filename", "")),
            question=str(raw.get("question", "")),
            answer=str(raw.get("answer", "")),
            subject=str(md.get("subject", "")),
            topic=str(md.get("topic", "")),
            image_paths=list(md.get("image_path", []) or []),
            solution_steps=list(md.get("solution_steps", []) or []),
            expected_tools=list(md.get("tool_expected", []) or []),
            golden_calls=gold,
            golden_chain_ordered=ordered and bool(gold),
            original_question_id=str(md.get("original_question_id", "")),
            usage_tool_protocol=list(raw.get("usage_tool_protocol", []) or []),
            source_dataset=source,
            dataset_name=SCIAGENTGYM,
        )

    @classmethod
    def from_research_claw_row(cls, raw: dict[str, Any], *, root: Path) -> "Problem":
        """Build a ResearchClawBench problem from one HF dataset row.

        The generated ``question`` intentionally includes only the task
        description and the allowed input materials. The target paper and
        checklist are preserved for grading/logging but are not shown to
        sciMAS.
        """
        task_id = str(raw.get("task_id", "")).strip()
        if not task_id:
            domain = str(raw.get("domain", "Task")).strip() or "Task"
            task_id = f"{domain}_{raw.get('task_number', '')}"
        root = _research_claw_repo_root(root)
        task_info = _json_dict_field(raw.get("task_info_json"))
        checklist = _json_list_of_dicts_field(raw.get("checklist_json"))
        data_files = _resolve_research_paths(
            _json_list_field(raw.get("data_files_json")), root
        )
        related_work = _resolve_research_paths(
            _json_list_field(raw.get("related_work_json")), root
        )
        target_images = _resolve_research_paths(
            _json_list_field(raw.get("target_images_json")), root
        )
        target_paper = _resolve_research_path(
            str(raw.get("target_paper_path", "") or ""), root
        )
        task_text = str(raw.get("task", "") or task_info.get("task", ""))
        return cls(
            id=task_id,
            filename=f"{task_id}.json",
            question=_format_research_claw_prompt(
                raw=raw,
                task_info=task_info,
                data_files=data_files,
                related_work=related_work,
            ),
            answer=_format_research_claw_expected_answer(checklist),
            subject=str(raw.get("domain", "")),
            topic=str(raw.get("task_tag", "")),
            image_paths=target_images,
            solution_steps=[
                str(item.get("content", ""))
                for item in checklist
                if str(item.get("content", "")).strip()
            ],
            expected_tools=[],
            golden_calls=[],
            original_question_id=task_id,
            usage_tool_protocol=[],
            source_dataset=str(raw.get("source", "")),
            dataset_name=RESEARCH_CLAW_BENCH,
            assets_root=str(root),
            task_info=task_info or {"task": task_text},
            checklist=checklist,
            data_files=data_files,
            related_work=related_work,
            target_images=target_images,
            target_paper=target_paper,
        )

    @classmethod
    def from_madd_row(
        cls,
        raw: dict[str, Any],
        *,
        row_index: int,
        source: str,
        root: Path,
    ) -> "Problem":
        """Build a MADD benchmark problem from one table row.

        MADD's original runner passes only ``content`` to the agent and
        keeps ``case`` / ``task N`` / answer columns for validation. This
        loader preserves that split so the runner does not leak hidden
        expected tool names into the prompt.
        """
        case = _clean_table_cell(raw.get("case"))
        content = _clean_table_cell(raw.get("content"))
        final_answer = (
            _clean_table_cell(raw.get("finall answer"))
            or _clean_table_cell(raw.get("final answer"))
        )
        expected_tools = _madd_expected_tools(raw)
        decomposer_tasks = _literal_list_field(raw.get("decomposers_tasks"))
        tool_answers = _literal_list_field(raw.get("tools answers"))
        try:
            is_correct_decomposition = bool(int(float(raw.get("is_correct", 0) or 0)))
        except (TypeError, ValueError):
            is_correct_decomposition = False
        problem_id = f"MADD_{row_index:04d}"
        return cls(
            id=problem_id,
            filename=source,
            question=_format_madd_prompt(problem_id, content),
            answer=final_answer,
            subject="Pharmaceutical Sciences",
            topic=case,
            image_paths=[],
            solution_steps=decomposer_tasks,
            expected_tools=expected_tools,
            golden_calls=[GoldenCall(tool=tool) for tool in expected_tools],
            original_question_id=str(row_index),
            usage_tool_protocol=[],
            source_dataset=source,
            dataset_name=MADD,
            assets_root=str(root),
            task_info={
                "case": case,
                "decomposers_tasks": decomposer_tasks,
                "is_correct_decomposition": is_correct_decomposition,
                "madd_expected_tools_original": expected_tools,
                "madd_tool_answers": tool_answers,
                "madd_expected_final_answer": final_answer,
            },
        )

    @classmethod
    def from_drug_discovery_bench_task(
        cls,
        task_dir: Path,
        *,
        benchmark_root: Path,
    ) -> "Problem":
        """Build a DrugDiscoveryBench problem from one Harbor task dir.

        The original Harbor prompt asks the agent to write only
        ``/workspace/answer.md`` and to call BiOMNI directly from the trial
        container. For sciMAS, we keep the scientific task and input-file
        paths, but adapt the final-answer instruction to the orchestrator's
        normal return channel and let the planner route through sciMAS /
        BiOMNI MCP skills.
        """
        task_id = task_dir.name
        rubrics_path = task_dir / "tests" / "rubrics.json"
        rubrics = _json_dict_file(rubrics_path)
        prompt = str(rubrics.get("prompt") or "").strip()
        if not prompt:
            prompt = _extract_ddb_prompt_from_instruction(
                (task_dir / "instruction.md").read_text(encoding="utf-8")
            )
        config = _toml_dict_file(task_dir / "task.toml")
        task_meta = config.get("task", {}) if isinstance(config.get("task"), dict) else {}
        env_meta = (
            config.get("environment", {})
            if isinstance(config.get("environment"), dict)
            else {}
        )
        input_files = sorted(
            str(path.resolve())
            for path in (task_dir / "environment" / "inputs").glob("*")
            if path.is_file()
        )
        outcome_rubrics = _json_list_of_dicts_field(rubrics.get("outcome_rubrics"))
        process_rubrics = _json_list_of_dicts_field(rubrics.get("process_rubrics"))
        ground_truth = str(rubrics.get("ground_truth") or "").strip()
        topic = str(task_meta.get("description") or "biomedical research").strip()
        keywords = task_meta.get("keywords", [])
        if isinstance(keywords, list) and keywords:
            topic = ", ".join(str(item) for item in keywords if str(item).strip()) or topic
        return cls(
            id=task_id,
            filename=str(task_dir.relative_to(benchmark_root)),
            question=_format_drug_discovery_bench_prompt(
                task_id=task_id,
                prompt=prompt,
                input_files=input_files,
            ),
            answer=ground_truth,
            subject="Biomedical Sciences",
            topic=topic,
            image_paths=[],
            solution_steps=[
                str(item.get("title", ""))
                for item in outcome_rubrics + process_rubrics
                if str(item.get("title", "")).strip()
            ],
            expected_tools=[],
            golden_calls=[],
            original_question_id=task_id,
            usage_tool_protocol=[],
            source_dataset=str(benchmark_root),
            dataset_name=DRUG_DISCOVERY_BENCH,
            assets_root=str(benchmark_root),
            task_info={
                "ddb_task_dir": str(task_dir.resolve()),
                "ddb_rubrics_path": str(rubrics_path.resolve()),
                "ddb_prompt": prompt,
                "ddb_ground_truth": ground_truth,
                "ddb_outcome_rubrics": outcome_rubrics,
                "ddb_process_rubrics": process_rubrics,
                "ddb_input_files": input_files,
                "ddb_docker_image": env_meta.get("docker_image", ""),
                "ddb_task_config": config,
            },
            data_files=input_files,
        )

    @classmethod
    def from_scipredict_row(
        cls,
        raw: dict[str, Any],
        *,
        row_index: int,
        source: str,
        root: Path,
        rubrics_by_task: dict[str, list[dict[str, Any]]],
        include_background: bool = True,
    ) -> "Problem":
        """Build a SciPredict problem from one CSV row.

        SciPredict is a text benchmark over experimental outcome
        prediction. The ground-truth answer and rubrics are preserved for
        grading but are not included in the prompt shown to sciMAS.
        """
        task_id = _csv_cell(raw, "TASK", "task_id", "id")
        if not task_id:
            task_id = f"SciPredict_{row_index:04d}"
        domain = _csv_cell(raw, "DOMAIN", "domain")
        field = _csv_cell(raw, "FIELD", "field")
        pq_format = _csv_cell(raw, "PQ_FORMAT", "format")
        title = _csv_cell(raw, "TITLE", "title")
        url = _csv_cell(raw, "URL", "url")
        publishing_date = _csv_cell(raw, "PUBLISHING_DATE", "publication_date")
        setup = _csv_cell(raw, "EXPERIMENTAL_SETUP", "experimental_setup")
        measurement = _csv_cell(raw, "MEASUREMENT_TAKEN", "measurement_taken")
        prediction_question = _csv_cell(
            raw,
            "OUTCOME_PREDICTION_QUESTION",
            "outcome_prediction_question",
            "question",
        )
        gta = _csv_cell(raw, "GTA", "ground_truth", "answer")
        clean_gta = _csv_cell(raw, "CLEAN_GTA", "clean_gta", "clean_answer")
        background = _csv_cell(
            raw,
            "REQUIRED_BACKGROUND_KNOWLEDGE",
            "required_background_knowledge",
            "background",
        )
        checklist = rubrics_by_task.get(task_id, [])
        return cls(
            id=task_id,
            filename=source,
            question=_format_scipredict_prompt(
                task_id=task_id,
                domain=domain,
                field=field,
                pq_format=pq_format,
                title=title,
                url=url,
                publishing_date=publishing_date,
                setup=setup,
                measurement=measurement,
                prediction_question=prediction_question,
                background=background,
                include_background=include_background,
            ),
            answer=clean_gta or gta,
            subject=domain or "Science",
            topic=field or pq_format or "Outcome prediction",
            image_paths=[],
            solution_steps=[
                str(item.get("content", ""))
                for item in checklist
                if str(item.get("content", "")).strip()
            ],
            expected_tools=[],
            golden_calls=[],
            original_question_id=task_id,
            usage_tool_protocol=[],
            source_dataset=source,
            dataset_name=SCIPREDICT,
            assets_root=str(root),
            task_info={
                "scipredict_task_id": task_id,
                "scipredict_domain": domain,
                "scipredict_field": field,
                "scipredict_pq_format": pq_format,
                "scipredict_title": title,
                "scipredict_url": url,
                "scipredict_publishing_date": publishing_date,
                "scipredict_ground_truth": gta,
                "scipredict_clean_ground_truth": clean_gta,
                "scipredict_background_included": include_background,
                "scipredict_background": background,
                "scipredict_raw_row": dict(raw),
            },
            checklist=checklist,
            target_paper=url,
        )

    @classmethod
    def from_smdd_task(
        cls,
        task_dir: Path,
        *,
        tasks_root: Path,
    ) -> "Problem":
        """Build an SMDDBench problem from one task directory."""
        task_yaml = task_dir / "task.yaml"
        config = _yaml_dict_file(task_yaml)
        task_id = str(
            config.get("id")
            or config.get("task_id")
            or config.get("name")
            or task_dir.name
        ).strip()
        type_code = _infer_smdd_type_code(task_id, config)
        type_name = _smdd_type_name(type_code)
        description = _smdd_task_description(config)
        output_file = _smdd_output_file(config)
        data_files = _smdd_visible_files(task_dir)
        evaluation = config.get("evaluation", {})
        if not isinstance(evaluation, dict):
            evaluation = {}
        return cls(
            id=task_id,
            filename=str(task_yaml.relative_to(tasks_root.parent)),
            question=_format_smdd_prompt(
                task_id=task_id,
                type_code=type_code,
                type_name=type_name,
                description=description,
                output_file=output_file,
                data_files=data_files,
            ),
            answer="",
            subject="Drug Discovery",
            topic=type_name or type_code or "SMDD task",
            image_paths=[],
            solution_steps=[],
            expected_tools=[],
            golden_calls=[],
            original_question_id=task_id,
            usage_tool_protocol=[],
            source_dataset=str(tasks_root),
            dataset_name=SMDD_BENCH,
            assets_root=str(tasks_root),
            task_info={
                "smdd_task_id": task_id,
                "smdd_task_dir": str(task_dir.resolve()),
                "smdd_task_yaml": str(task_yaml.resolve()),
                "smdd_type_code": type_code,
                "smdd_type_name": type_name,
                "smdd_description": description,
                "smdd_output_file": output_file,
                "smdd_evaluation": evaluation,
                "smdd_raw_config": config,
                "smdd_official_result_candidates": [
                    str(path.resolve())
                    for path in _smdd_official_result_candidates(task_dir)
                ],
            },
            data_files=data_files,
        )

    @classmethod
    def from_biomni_eval1_row(
        cls,
        raw: dict[str, Any],
        *,
        row_index: int,
        source: str,
        root: Path,
    ) -> "Problem":
        """Build a Biomni Eval1 problem from one tabular row."""
        instance_id = _csv_cell(raw, "instance_id", "INSTANCE_ID", "id")
        task_instance_id = _csv_cell(
            raw,
            "task_instance_id",
            "TASK_INSTANCE_ID",
            "task_id",
        )
        task_name = _csv_cell(raw, "task_name", "TASK_NAME", "task")
        split = _csv_cell(raw, "split", "SPLIT") or "test"
        prompt = _csv_cell(raw, "prompt", "PROMPT", "question")
        answer = _csv_cell(raw, "answer", "ANSWER", "ground_truth")
        problem_id = instance_id or (
            f"{task_name}_{task_instance_id}" if task_name or task_instance_id else f"BiomniEval1_{row_index:04d}"
        )
        return cls(
            id=problem_id,
            filename=source,
            question=_format_biomni_eval1_prompt(
                task_id=problem_id,
                task_name=task_name,
                prompt=prompt,
            ),
            answer=answer,
            subject="Biomedical Sciences",
            topic=task_name or "Biomni Eval1",
            image_paths=[],
            solution_steps=[],
            expected_tools=[],
            golden_calls=[],
            original_question_id=task_instance_id or problem_id,
            usage_tool_protocol=[],
            source_dataset=source,
            dataset_name=BIOMNI_EVAL1,
            assets_root=str(root),
            task_info={
                "biomni_eval1_instance_id": instance_id,
                "biomni_eval1_task_instance_id": task_instance_id,
                "biomni_eval1_task_name": task_name,
                "biomni_eval1_split": split,
                "biomni_eval1_answer": answer,
                "biomni_eval1_raw_row": dict(raw),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _json_field(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def _json_dict_field(value: Any) -> dict[str, Any]:
    parsed = _json_field(value, {})
    return parsed if isinstance(parsed, dict) else {}


def _json_list_field(value: Any) -> list[str]:
    parsed = _json_field(value, [])
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item).strip()]


def _json_list_of_dicts_field(value: Any) -> list[dict[str, Any]]:
    parsed = _json_field(value, [])
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _json_dict_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _toml_dict_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            return {}
    try:
        with path.open("rb") as handle:
            parsed = tomllib.load(handle)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _yaml_dict_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            f"YAML loading requires PyYAML to read {path}. Install it with: python -m pip install pyyaml"
        ) from exc
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _clean_table_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _csv_cell(row: dict[str, Any], *names: str) -> str:
    """Return a CSV cell by trying exact and case-insensitive keys."""
    for name in names:
        if name in row:
            value = _clean_table_cell(row.get(name))
            if value:
                return value
    lowered = {str(key).strip().lower(): key for key in row.keys()}
    for name in names:
        key = lowered.get(name.strip().lower())
        if key is not None:
            value = _clean_table_cell(row.get(key))
            if value:
                return value
    return ""


def _first_text_field(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _flatten_dict_text(data: Any) -> str:
    if isinstance(data, dict):
        parts: list[str] = []
        for key, value in data.items():
            text = _flatten_dict_text(value)
            if text:
                parts.append(f"{key}: {text}")
        return "\n".join(parts)
    if isinstance(data, list):
        return "\n".join(_flatten_dict_text(item) for item in data if _flatten_dict_text(item))
    if data is None:
        return ""
    return str(data).strip()


def _literal_list_field(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_clean_table_cell(item) for item in value if _clean_table_cell(item)]
    text = _clean_table_cell(value)
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, list):
        return [_clean_table_cell(item) for item in parsed if _clean_table_cell(item)]
    return [text]


def _madd_expected_tools(raw: dict[str, Any]) -> list[str]:
    tools: list[str] = []
    for idx in range(1, 6):
        tool = _clean_table_cell(raw.get(f"task {idx}"))
        if tool:
            tools.append(tool)
    return tools


def _research_claw_repo_root(root: Path) -> Path:
    root = Path(root).expanduser().resolve()
    if (root / "repo" / "tasks").exists():
        return root / "repo"
    if root.name == "tasks":
        return root.parent
    return root


def _resolve_research_path(path_text: str, root: Path) -> str:
    path_text = path_text.strip()
    if not path_text:
        return ""
    path = Path(path_text)
    if path.is_absolute():
        return str(path)
    candidates = [root / path]
    parts = path.parts
    if parts and parts[0] == "tasks" and root.name == "tasks":
        candidates.append(root / Path(*parts[1:]))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str(candidates[0].resolve())


def _resolve_research_paths(paths: Iterable[str], root: Path) -> list[str]:
    return [_resolve_research_path(path, root) for path in paths]


# ``metadata.image_path`` entries are written relative to the *upstream repo
# root* (``gym/test_images/...``), while the JSON dumps are normally loaded
# from the sciMAS-internal copy (``dataset/``), which has no ``gym/`` in it at
# all. ``default_dataset_root()`` returns the sciMAS root for SciAgentGYM for
# exactly that reason, so a naive ``root / path`` finds nothing and the figures
# silently disappear. Every candidate below has been observed in a real
# layout; the first one holding the file wins.
SCIAGENTGYM_IMAGE_ROOT_ENV = ("SCIENT_GYM_ROOT", "SCIENTAGENTGYM_ROOT")


def _sciagentgym_image_roots(dataset_root: Path) -> list[Path]:
    """Candidate repo roots for SciAgentGYM question figures, in order."""
    candidates: list[Path] = []
    for var in SCIAGENTGYM_IMAGE_ROOT_ENV:
        env = os.environ.get(var, "").strip()
        if env:
            candidates.append(Path(env).expanduser())
    candidates.extend(
        [
            dataset_root,  # $SCIENT_GYM_ROOT layout, or a checkout used as root
            dataset_root / "SciAgentGYM-main",  # <sciMAS>/SciAgentGYM-main
            dataset_root.parent / "SciAgentGYM-main",  # games/SciAgentGYM-main
            Path("~/Documents/games/SciAgentGYM-main").expanduser(),
        ]
    )
    roots: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append(resolved)
    return roots


def _resolve_sciagentgym_images(
    paths: Iterable[str], root: Path
) -> tuple[list[str], str, list[str]]:
    """Resolve ``metadata.image_path`` entries to existing absolute files.

    Returns ``(resolved, assets_root, missing)``. An entry that resolves to
    nothing is *dropped* rather than guessed at: a path handed to the solver is
    an instruction to open that file, so advertising a broken one spends a tool
    call and shows the judge an unreadable block. Absolute paths in the dump
    are honoured as-is when they exist, so a dump that ships pre-resolved paths
    needs no change here.
    """
    entries = [str(path).strip() for path in paths if str(path).strip()]
    if not entries:
        return [], "", []
    roots = _sciagentgym_image_roots(root)
    resolved: list[str] = []
    missing: list[str] = []
    assets_root = ""
    for entry in entries:
        hit = ""
        candidate = Path(entry).expanduser()
        if candidate.is_absolute():
            if candidate.is_file():
                hit = str(candidate)
        else:
            for candidate_root in roots:
                attempt = candidate_root / candidate
                if attempt.is_file():
                    hit = str(attempt.resolve())
                    if not assets_root:
                        assets_root = str(candidate_root)
                    break
        if not hit:
            missing.append(entry)
        elif hit not in resolved:
            resolved.append(hit)
    return resolved, assets_root, missing


def stage_problem_images(problem: "Problem", dest_dir: Path) -> list[str]:
    """Copy this problem's figures into ``dest_dir``; return the copies.

    Staging rather than handing out the dataset's own paths keeps the run
    output self-contained (the report's figures survive the dataset checkout
    moving) and keeps the solving agents' Read/Bash inside the repo, where no
    extra directory permission is needed. Returns ``[]`` when the problem has
    no resolved figures — the caller then behaves exactly as before.

    Copying is idempotent: an existing destination of the same size is left
    alone, so re-running a problem does not rewrite megabytes of PNGs.
    """
    sources = [
        Path(str(path)) for path in (problem.image_paths or []) if str(path).strip()
    ]
    if not sources:
        return []
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []
    used: set[str] = set()
    for index, source in enumerate(sources, start=1):
        if not source.is_file():
            continue
        name = source.name
        if name in used:
            # Two figures from different roots sharing a basename; keep both.
            name = f"{index}-{name}"
        used.add(name)
        target = dest_dir / name
        try:
            if not (target.is_file() and target.stat().st_size == source.stat().st_size):
                shutil.copyfile(source, target)
        except OSError:
            # A figure we cannot stage is one the solver cannot read either;
            # dropping it here is the same policy as an unresolved path.
            continue
        staged.append(str(target.resolve()))
    return staged


_QUESTION_FIGURES_FOOTER = (
    "These files are part of the question, not optional context. Read them "
    "with the Read tool before answering — treat the axes, units, labels and "
    "data points they show as part of the problem statement."
)


def _format_image_section(paths: list[str]) -> list[str]:
    lines = ["Question figures (the figures this question refers to):"]
    for path in paths:
        lines.append(f"- {path}")
    lines.append(_QUESTION_FIGURES_FOOTER)
    return lines


def format_problem_for_solver(problem: "Problem") -> str:
    """Return the problem text the solver should see, figures included.

    Byte-identical to ``problem.question`` when the problem has no figures —
    48 of the SciAgentGYM entries, every write-in, and the ``main.py`` CLI.
    ``problem.question`` itself is never modified: the grader feeds it to the
    LLM judge, where a list of local paths would be redundant (the figures
    travel as vision blocks) and would change the judged question.

    Prefers ``image_paths_local`` (the copies staged for this run) and falls
    back to the resolved dataset paths when nothing was staged.
    """
    question = problem.question or ""
    images = [
        str(path)
        for path in (problem.image_paths_local or problem.image_paths or [])
        if str(path).strip()
    ]
    if not images:
        return question
    return "\n".join([question, "", *_format_image_section(images)])


def _format_path_section(title: str, paths: list[str]) -> list[str]:
    lines = [f"{title}:"]
    if not paths:
        lines.append("- (none listed)")
        return lines
    for path in paths:
        lines.append(f"- {path}")
    return lines


def _format_research_claw_prompt(
    *,
    raw: dict[str, Any],
    task_info: dict[str, Any],
    data_files: list[str],
    related_work: list[str],
) -> str:
    task_text = str(raw.get("task", "") or task_info.get("task", "")).strip()
    domain = str(raw.get("domain", "")).strip()
    task_id = str(raw.get("task_id", "")).strip()
    task_tag = str(raw.get("task_tag", "")).strip()
    lines: list[str] = [
        "ResearchClawBench task",
        f"Task ID: {task_id}",
        f"Domain: {domain}",
        f"Task tag: {task_tag}",
        "",
        "Research objective:",
        task_text,
        "",
    ]
    data_manifest = task_info.get("data", [])
    if isinstance(data_manifest, list) and data_manifest:
        lines.append("Data manifest:")
        for entry in data_manifest:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip()
            kind = str(entry.get("type", "")).strip()
            desc = str(entry.get("description", "")).strip()
            header = f"- {name}" if name else "-"
            if kind:
                header += f" ({kind})"
            if desc:
                header += f": {desc}"
            lines.append(header)
        lines.append("")
    lines.extend(_format_path_section("Available data files", data_files))
    lines.append("")
    lines.extend(_format_path_section("Related work papers", related_work))
    lines.extend(
        [
            "",
            "Deliverable:",
            (
                "Analyze the available data and related work, then produce a "
                "concise scientific report-style final answer with methods, "
                "key quantitative results, limitations, and any figure/table "
                "file paths you create or rely on."
            ),
            (
                "Do not use target_study files or checklist rubrics while "
                "answering; those materials are reserved for grading."
            ),
        ]
    )
    return "\n".join(lines)


def _format_research_claw_expected_answer(checklist: list[dict[str, Any]]) -> str:
    if not checklist:
        return ""
    lines = ["ResearchClawBench checklist rubric:"]
    for i, item in enumerate(checklist, start=1):
        content = str(item.get("content", "")).strip()
        kind = str(item.get("type", "")).strip()
        weight = item.get("weight", "")
        lines.append(f"{i}. type={kind} weight={weight}: {content}")
        keywords = item.get("keywords", [])
        if isinstance(keywords, list) and keywords:
            lines.append("   keywords: " + "; ".join(str(k) for k in keywords))
    return "\n".join(lines)


def _format_madd_prompt(problem_id: str, content: str) -> str:
    lines = [
        "MADD pharmaceutical benchmark task",
        f"Task ID: {problem_id}",
        "",
        content,
        "",
        "Deliverable:",
        (
            "Answer the task directly. If molecule generation or screening is "
            "required, include SMILES strings in a concise markdown table with "
            "relevant drug-discovery properties such as docking score, QED, "
            "synthetic accessibility, alerts, BBB, IC50, or Ki when available."
        ),
    ]
    return "\n".join(lines)


def _extract_ddb_prompt_from_instruction(instruction: str) -> str:
    """Return the task prompt without the common BiOMNI appendix."""
    marker = "\n---\n\nYou have access to Biomni's biomedical tool suite."
    if marker in instruction:
        instruction = instruction.split(marker, 1)[0]
    return instruction.strip()


def _format_drug_discovery_bench_prompt(
    *,
    task_id: str,
    prompt: str,
    input_files: list[str],
) -> str:
    lines: list[str] = [
        "DrugDiscoveryBench biomedical task",
        f"Task ID: {task_id}",
        "",
        "Task prompt:",
        prompt.strip(),
        "",
    ]
    lines.extend(_format_path_section("Available input files", input_files))
    lines.extend(
        [
            "",
            "Deliverable:",
            (
                "Answer the task directly. The original Harbor benchmark asks "
                "agents to write /workspace/answer.md; in sciMAS, return the "
                "final answer in this response instead."
            ),
            (
                "Use the migrated sciMAS / BiOMNI tools when useful, and do "
                "not rely on hidden rubrics or reference answers while answering."
            ),
        ]
    )
    return "\n".join(lines)


def _infer_smdd_type_code(task_id: str, config: dict[str, Any]) -> str:
    for key in ("type", "task_type", "category", "benchmark_type"):
        value = config.get(key)
        if isinstance(value, (int, float)):
            return f"{int(value):03d}"
        if isinstance(value, str) and value.strip():
            match = re.search(r"00[1-5]|\b[1-5]\b", value)
            if match:
                return f"{int(match.group(0)):03d}"
            for code, name in SMDD_TYPE_NAMES.items():
                if value.strip().lower() == name.lower():
                    return code
    match = re.search(r"(?:^|[_\-/])0*([1-5])(?:[_\-/]|$)", task_id)
    if match:
        return f"{int(match.group(1)):03d}"
    match = re.search(r"00[1-5]", task_id)
    return match.group(0) if match else ""


def _smdd_type_name(type_code: str) -> str:
    return SMDD_TYPE_NAMES.get(type_code, "")


def _smdd_task_description(config: dict[str, Any]) -> str:
    text = _first_text_field(
        config,
        "description",
        "task",
        "prompt",
        "instruction",
        "instructions",
        "query",
    )
    if text:
        return text
    for key in ("input", "inputs", "specification"):
        value = config.get(key)
        text = _flatten_dict_text(value)
        if text:
            return text
    # Last-resort description for unusual task.yaml variants. Avoid
    # leaking evaluation internals into the prompt.
    public = {
        key: value
        for key, value in config.items()
        if str(key).lower() not in {"evaluation", "metrics", "answer", "solution", "ground_truth"}
    }
    return _flatten_dict_text(public)


def _smdd_output_file(config: dict[str, Any]) -> str:
    candidate_keys = (
        "output_file",
        "output_path",
        "file_path",
        "submission_file",
        "answer_file",
        "target_file",
    )
    for key in candidate_keys:
        text = _clean_table_cell(config.get(key))
        if text:
            return text
    for key in ("output", "output_config", "submission", "deliverable"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            for subkey in candidate_keys + ("file", "path", "filename"):
                text = _clean_table_cell(value.get(subkey))
                if text:
                    return text
    evaluation = config.get("evaluation")
    if isinstance(evaluation, dict):
        for key in ("output", "output_config", "submission"):
            value = evaluation.get(key)
            if isinstance(value, dict):
                for subkey in candidate_keys + ("file", "path", "filename"):
                    text = _clean_table_cell(value.get(subkey))
                    if text:
                        return text
            elif isinstance(value, str) and value.strip():
                return value.strip()
    return "submission file required by task.yaml"


def _smdd_visible_files(task_dir: Path) -> list[str]:
    hidden_names = {
        "task.yaml",
        "result.json",
        "results.json",
        "eval_result.json",
        "evaluation.json",
    }
    paths: list[str] = []
    for path in sorted(task_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name in hidden_names:
            continue
        if "__pycache__" in path.parts:
            continue
        paths.append(str(path.resolve()))
    return paths


def _smdd_official_result_candidates(task_dir: Path) -> list[Path]:
    return [
        task_dir / "result.json",
        task_dir / "results.json",
        task_dir / "eval_result.json",
        task_dir / "evaluation.json",
        task_dir / "official_result.json",
    ]


def _format_smdd_prompt(
    *,
    task_id: str,
    type_code: str,
    type_name: str,
    description: str,
    output_file: str,
    data_files: list[str],
) -> str:
    lines: list[str] = [
        "SMDDBench molecular drug-discovery artifact task",
        f"Task ID: {task_id}",
    ]
    if type_code or type_name:
        label = " / ".join(part for part in (type_code, type_name) if part)
        lines.append(f"Task type: {label}")
    lines.extend(["", "Task description:", description or "(see available task files)"])
    lines.extend(["", *_format_path_section("Available task files", data_files)])
    lines.extend(
        [
            "",
            "Expected artifact:",
            output_file,
            "",
            "Deliverable:",
            (
                "Return the content that should be written to the expected artifact. "
                "If the artifact is molecule text such as SMILES, SDF, CSV, JSON, "
                "or a Python script, include the complete final content in your "
                "answer. The official SMDD-Bench Docker evaluator must be used "
                "afterward to compute the benchmark score."
            ),
        ]
    )
    return "\n".join(lines)


def _biomni_eval1_format_hint(task_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", (task_name or "").strip().lower()).strip("_")
    if normalized in {"lab_test_analyzing", "lab_bench_dbqa", "lab_bench_seqqa", "crispr_delivery"}:
        return "Answer with the correct choice label, case-insensitive."
    if normalized == "rare_disease_diagnosis":
        return 'Answer as JSON with keys such as "disease_name" and "OMIM_ID".'
    if normalized in {
        "gene_name_conversion",
        "patient_gene_detection",
        "screen_gene_retrieval",
        "gwas_causal_gene_pharmaprojects",
        "gwas_causal_gene_opentargets",
        "gwas_causal_gene_gwas_catalog",
    }:
        return "Answer with the single gene identifier/name only."
    if normalized == "gwas_variant_prioritization":
        return "Answer with the single prioritized variant rsID only."
    if normalized == "variant_pathogenicity":
        return "Answer with the exact predicted pathogenicity label."
    return "Answer in the concise format requested by the task."


def _format_biomni_eval1_prompt(
    *,
    task_id: str,
    task_name: str,
    prompt: str,
) -> str:
    lines = [
        "Biomni Eval1 biomedical QA task",
        f"Task ID: {task_id}",
    ]
    if task_name:
        lines.append(f"Task name: {task_name}")
    lines.extend(
        [
            "",
            "Prompt:",
            prompt.strip(),
            "",
            "Deliverable:",
            _biomni_eval1_format_hint(task_name),
        ]
    )
    return "\n".join(lines)


def _format_scipredict_prompt(
    *,
    task_id: str,
    domain: str,
    field: str,
    pq_format: str,
    title: str,
    url: str,
    publishing_date: str,
    setup: str,
    measurement: str,
    prediction_question: str,
    background: str,
    include_background: bool,
) -> str:
    lines: list[str] = [
        "SciPredict scientific outcome-prediction task",
        f"Task ID: {task_id}",
    ]
    if domain:
        lines.append(f"Domain: {domain}")
    if field:
        lines.append(f"Field: {field}")
    if pq_format:
        lines.append(f"Question format: {pq_format}")
    if title:
        lines.extend(["", "Study title:", title])
    if publishing_date or url:
        source_parts = []
        if publishing_date:
            source_parts.append(f"published {publishing_date}")
        if url:
            source_parts.append(url)
        lines.extend(["", "Source:", " | ".join(source_parts)])
    if setup:
        lines.extend(["", "Experimental setup:", setup])
    if measurement:
        lines.extend(["", "Measurement taken:", measurement])
    if include_background and background:
        lines.extend(["", "Background knowledge:", background])
    lines.extend(["", "Outcome prediction question:", prediction_question])
    lines.extend(
        [
            "",
            "Deliverable:",
            (
                "Predict the experimental outcome directly. For MCQ tasks, "
                "begin with the selected option letter or letters. For "
                "numerical tasks, include the numeric value and units when "
                "available. For free-form tasks, give a concise scientific "
                "answer and enough reasoning to justify it."
            ),
        ]
    )
    return "\n".join(lines)


def load_dataset(
    root: Optional[Path] = None,
    *,
    files: Iterable[str] = DEFAULT_DATASETS,
    dataset_name: str = SCIAGENTGYM,
    split: str = "train",
) -> list[Problem]:
    """Load one supported benchmark dataset.

    For SciAgentGYM, every JSON in ``files`` is read from ``root/dataset/``.
    For ResearchClawBench, the Hugging Face dataset table is loaded from
    ``InternScience/ResearchClawBench`` and enriched with local file paths
    under ``root/tasks`` when a snapshot is available. For MADD, the
    migrated local benchmark table is read from ``root/benchmark`` or
    an upstream MADD checkout's ``examples`` directory. For SciPredict,
    ``split="nbk"`` omits the expert background field; all other split
    names include it. For SMDDBench, task directories are read from
    ``tasks/`` or ``tasks_lite/``.
    """
    dataset_name = normalize_dataset_name(dataset_name)
    root = Path(root) if root else default_dataset_root(dataset_name)
    if dataset_name == RESEARCH_CLAW_BENCH:
        return _load_research_claw_bench_dataset(root=root, split=split)
    if dataset_name == MADD:
        return _load_madd_benchmark_dataset(root=root)
    if dataset_name == DRUG_DISCOVERY_BENCH:
        return _load_drug_discovery_bench_dataset(root=root)
    if dataset_name == SCIPREDICT:
        return _load_scipredict_dataset(root=root, split=split)
    if dataset_name == SMDD_BENCH:
        return _load_smdd_bench_dataset(root=root)
    if dataset_name == BIOMNI_EVAL1:
        return _load_biomni_eval1_dataset(root=root, split=split)
    return _load_sciagentgym_dataset(root=root, files=files)


def _load_sciagentgym_dataset(
    *,
    root: Path,
    files: Iterable[str] = DEFAULT_DATASETS,
) -> list[Problem]:
    """Load every SciAgentGYM JSON dump from ``root/dataset/``."""
    problems: list[Problem] = []
    for fname in files:
        path = root / "dataset" / fname
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            continue
        for raw in data:
            problem = Problem.from_raw(raw, source=fname)
            if problem.image_paths:
                resolved, assets_root, missing = _resolve_sciagentgym_images(
                    problem.image_paths, root
                )
                problem.image_paths = resolved
                problem.image_paths_missing = missing
                if assets_root:
                    problem.assets_root = assets_root
            problems.append(problem)
    return problems


def _load_research_claw_bench_dataset(*, root: Path, split: str = "train") -> list[Problem]:
    """Load ResearchClawBench through the Hugging Face datasets API."""
    try:
        from datasets import load_dataset as hf_load_dataset
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError(
            "ResearchClawBench loading requires the optional 'datasets' "
            "package. Install it with: python -m pip install datasets"
        ) from exc

    repo_root = _research_claw_repo_root(root)
    bundle_root = repo_root.parent if repo_root.name == "repo" else repo_root
    saved_dataset = bundle_root / "hf_dataset"
    if saved_dataset.exists():
        ds = load_from_disk(str(saved_dataset))
    else:
        cache_dir = bundle_root / ".hf" / "datasets"
        ds = hf_load_dataset(RESEARCH_CLAW_BENCH_REPO_ID, cache_dir=str(cache_dir))

    if hasattr(ds, "keys"):
        if split not in ds:
            available = ", ".join(ds.keys())
            raise ValueError(f"split {split!r} not found; available: {available}")
        rows = ds[split]
    else:
        rows = ds
    return [
        Problem.from_research_claw_row(dict(row), root=repo_root)
        for row in rows
    ]


def _find_madd_benchmark_file(root: Path) -> Path:
    root = Path(root).expanduser().resolve()
    if root.is_file():
        return root
    candidates = [root / path for path in MADD_BENCHMARK_FILES]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    formatted = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "MADD benchmark table not found. Expected one of:\n" + formatted
    )


def _load_madd_benchmark_dataset(*, root: Path) -> list[Problem]:
    path = _find_madd_benchmark_file(root)
    rows: list[dict[str, Any]]
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif path.suffix.lower() in {".xls", ".xlsx"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "MADD Excel loading requires pandas/openpyxl, or provide the "
                "JSONL mirror at dataset/MADD/benchmark/large_ds_chemical_result.jsonl"
            ) from exc
        frame = pd.read_excel(path)
        rows = frame.where(pd.notna(frame), None).to_dict(orient="records")
    else:
        raise ValueError(f"unsupported MADD benchmark file type: {path}")
    return [
        Problem.from_madd_row(row, row_index=idx, source=str(path), root=root)
        for idx, row in enumerate(rows, start=1)
    ]


def _find_drug_discovery_benchmark_root(root: Path) -> Path:
    root = Path(root).expanduser().resolve()
    candidates = []
    if (root / "tasks").exists():
        candidates.append(root)
    candidates.extend([
        root / "benchmark",
        root / DRUG_DISCOVERY_BENCH / "benchmark",
    ])
    for candidate in candidates:
        if (candidate / "tasks").exists():
            return candidate.resolve()
    formatted = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "DrugDiscoveryBench benchmark tasks not found. Expected tasks/ under one of:\n"
        + formatted
    )


def _load_drug_discovery_bench_dataset(*, root: Path) -> list[Problem]:
    benchmark_root = _find_drug_discovery_benchmark_root(root)
    tasks_root = benchmark_root / "tasks"
    problems = [
        Problem.from_drug_discovery_bench_task(task_dir, benchmark_root=benchmark_root)
        for task_dir in sorted(tasks_root.iterdir())
        if task_dir.is_dir()
    ]
    return problems


def _find_smdd_tasks_root(root: Path) -> Path:
    root = Path(root).expanduser().resolve()
    if (root / "task.yaml").exists():
        return root.parent
    candidates: list[Path] = []
    if root.name in {"tasks", "tasks_lite"}:
        candidates.append(root)
    candidates.extend(root / path for path in SMDD_TASK_DIR_NAMES)
    candidates.append(root)
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        if any((child / "task.yaml").exists() for child in candidate.iterdir() if child.is_dir()):
            return candidate.resolve()
        if (candidate / "task.yaml").exists():
            return candidate.parent.resolve()
    formatted = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "SMDDBench task directories not found. Expected task.yaml files under one of:\n"
        + formatted
    )


def _load_smdd_bench_dataset(*, root: Path) -> list[Problem]:
    tasks_root = _find_smdd_tasks_root(root)
    if (Path(root).expanduser().resolve() / "task.yaml").exists():
        task_dirs = [Path(root).expanduser().resolve()]
    elif (tasks_root / "task.yaml").exists():
        task_dirs = [tasks_root]
    else:
        task_dirs = [
            task_dir for task_dir in sorted(tasks_root.iterdir())
            if task_dir.is_dir() and (task_dir / "task.yaml").exists()
        ]
    return [
        Problem.from_smdd_task(task_dir, tasks_root=tasks_root)
        for task_dir in task_dirs
    ]


def _find_optional_biomni_eval1_file(root: Path) -> Path | None:
    root = Path(root).expanduser().resolve()
    if root.is_file():
        return root
    for rel in BIOMNI_EVAL1_FILES:
        candidate = root / rel
        if candidate.exists():
            return candidate.resolve()
    parquet_files = sorted(root.rglob("*.parquet")) if root.exists() else []
    if parquet_files:
        return parquet_files[0].resolve()
    for pattern in ("*.csv", "*.jsonl"):
        files = sorted(root.rglob(pattern)) if root.exists() else []
        if files:
            return files[0].resolve()
    return None


def _read_table_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                f"Reading {path} requires pandas/pyarrow. Install them or provide CSV/JSONL."
            ) from exc
        frame = pd.read_parquet(path)
        return frame.where(pd.notna(frame), "").to_dict(orient="records")
    if suffix == ".csv":
        return _read_csv_rows(path)
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    rows.append(parsed)
        return rows
    raise ValueError(f"unsupported BiomniEval1 file type: {path}")


def _biomni_eval1_dataset_split(split: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", (split or "").lower())
    return "test" if normalized in {"", "train", "test", "default"} else split


def _load_biomni_eval1_dataset(*, root: Path, split: str = "test") -> list[Problem]:
    local_file = _find_optional_biomni_eval1_file(root)
    if local_file:
        rows = _read_table_rows(local_file)
        dataset_split = _biomni_eval1_dataset_split(split)
        if any("split" in row for row in rows):
            filtered = [
                row for row in rows
                if str(row.get("split") or "").strip().lower() == dataset_split.lower()
            ]
            rows = filtered or rows
        source = str(local_file)
    else:
        try:
            from datasets import load_dataset as hf_load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "BiomniEval1 local file not found, and loading from Hugging Face "
                "requires the optional 'datasets' package."
            ) from exc
        dataset_split = _biomni_eval1_dataset_split(split)
        cache_dir = Path(root).expanduser().resolve() / ".hf" / "datasets"
        ds = hf_load_dataset(BIOMNI_EVAL1_REPO_ID, split=dataset_split, cache_dir=str(cache_dir))
        rows = [dict(row) for row in ds]
        source = f"{BIOMNI_EVAL1_REPO_ID}:{dataset_split}"
    return [
        Problem.from_biomni_eval1_row(
            row,
            row_index=idx,
            source=source,
            root=root,
        )
        for idx, row in enumerate(rows, start=1)
    ]


def _find_scipredict_main_file(root: Path) -> Path:
    root = Path(root).expanduser().resolve()
    if root.is_file():
        return root
    candidates = [root / path for path in SCIPREDICT_MAIN_FILES]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    formatted = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "SciPredict CSV not found. Expected one of:\n" + formatted
    )


def _find_optional_scipredict_rubrics_file(root: Path, main_path: Path) -> Path | None:
    candidates = [root / path for path in SCIPREDICT_RUBRICS_FILES]
    candidates.extend([
        main_path.parent / "rubrics.csv",
        main_path.parent.parent / "rubrics.csv",
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader if any(_clean_table_cell(v) for v in row.values())]


def _scipredict_include_background(split: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", (split or "").lower())
    return normalized not in {"nbk", "nobackground", "withoutbackground", "noexpertbackground"}


def _load_scipredict_rubrics(root: Path, main_path: Path) -> dict[str, list[dict[str, Any]]]:
    rubrics_path = _find_optional_scipredict_rubrics_file(root, main_path)
    if not rubrics_path:
        return {}
    rows = _read_csv_rows(rubrics_path)
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        task_id = _csv_cell(row, "TASK", "task_id", "id")
        if not task_id:
            continue
        rubric_columns = [
            key for key in row.keys()
            if re.search(r"(criterion|rubric|grading|point)", str(key), re.I)
        ]
        if not rubric_columns:
            rubric_columns = [
                key for key in row.keys()
                if str(key).strip().lower() not in {"task", "task_id", "id"}
            ]
        checklist: list[dict[str, Any]] = []
        for key in rubric_columns:
            text = _clean_table_cell(row.get(key))
            if not text:
                continue
            checklist.append(
                {
                    "type": "rubric",
                    "weight": 1.0,
                    "content": text,
                    "keywords": [],
                    "source_column": str(key),
                }
            )
        if checklist:
            out[task_id] = checklist
    return out


def _load_scipredict_dataset(*, root: Path, split: str = "train") -> list[Problem]:
    try:
        main_path = _find_scipredict_main_file(root)
    except FileNotFoundError as exc:
        return _load_scipredict_hf_dataset(root=root, split=split, source_error=exc)
    rubrics_by_task = _load_scipredict_rubrics(root, main_path)
    include_background = _scipredict_include_background(split)
    rows = _read_csv_rows(main_path)
    return [
        Problem.from_scipredict_row(
            row,
            row_index=idx,
            source=str(main_path),
            root=root,
            rubrics_by_task=rubrics_by_task,
            include_background=include_background,
        )
        for idx, row in enumerate(rows, start=1)
    ]


def _load_scipredict_hf_dataset(
    *,
    root: Path,
    split: str,
    source_error: FileNotFoundError,
) -> list[Problem]:
    """Load SciPredict from Hugging Face when no local CSV is present."""
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "SciPredict local CSV not found, and loading from Hugging Face "
            "requires the optional 'datasets' package. Place main_ds.csv under "
            "dataset/SciPredict/data/ or install datasets."
        ) from source_error

    normalized_split = re.sub(r"[^a-z0-9]+", "", (split or "").lower())
    dataset_split = "train" if normalized_split in {
        "",
        "train",
        "bk",
        "background",
        "nbk",
        "nobackground",
        "withoutbackground",
        "noexpertbackground",
    } else split
    cache_dir = Path(root).expanduser().resolve() / ".hf" / "datasets"
    try:
        ds = hf_load_dataset(SCIPREDICT_REPO_ID, cache_dir=str(cache_dir))
    except Exception as exc:
        raise RuntimeError(
            "SciPredict local CSV not found, and Hugging Face loading failed. "
            "Place GitHub data/main_ds.csv or HF dataset.csv under "
            "dataset/SciPredict/."
        ) from exc

    if hasattr(ds, "keys"):
        if dataset_split not in ds:
            available = ", ".join(ds.keys())
            raise ValueError(f"split {dataset_split!r} not found; available: {available}")
        rows = ds[dataset_split]
    else:
        rows = ds
    include_background = _scipredict_include_background(split)
    return [
        Problem.from_scipredict_row(
            dict(row),
            row_index=idx,
            source=f"{SCIPREDICT_REPO_ID}:{dataset_split}",
            root=root,
            rubrics_by_task={},
            include_background=include_background,
        )
        for idx, row in enumerate(rows, start=1)
    ]


def filter_problems(
    problems: list[Problem],
    *,
    subject: Optional[str] = None,
    topic: Optional[str] = None,
    ids: Optional[Iterable[int | str]] = None,
    query: Optional[str] = None,
    deduplicate_ids: bool = True,
) -> list[Problem]:
    """Apply (case-insensitive) filters. Any filter that is None is
    skipped. ``query`` is a substring search across question + filename.

    If ``deduplicate_ids`` is true and the same ``id`` appears in
    multiple source datasets, only the first occurrence is kept (the
    multi-question set is loaded first, so its version wins).
    """
    out: list[Problem] = []
    ids_set = {str(i) for i in ids} if ids is not None else None
    subj_lc = subject.lower() if subject else None
    topic_lc = topic.lower() if topic else None
    q_lc = query.lower() if query else None
    seen_ids: set[str] = set()
    for p in problems:
        pid = str(p.id)
        if ids_set is not None and pid not in ids_set:
            continue
        if subj_lc and subj_lc != p.subject.lower():
            continue
        if topic_lc and topic_lc != p.topic.lower():
            continue
        if q_lc and q_lc not in p.question.lower() and q_lc not in p.filename.lower():
            continue
        dedupe_key = f"{p.dataset_name}:{pid}"
        if deduplicate_ids and dedupe_key in seen_ids:
            continue
        seen_ids.add(dedupe_key)
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# Numeric / answer extraction helpers — used by grader.py too.
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(
    r"""
    (?P<sign>[+-]?)              # sign
    (?P<num>\d+(?:\.\d+)?        # 12 or 12.34
        (?:[eE][+-]?\d+)?)       # optional exponent
    \s*                           # optional space
    (?P<pct>%)?                  # optional %
    """,
    re.VERBOSE,
)


def extract_numbers(text: str) -> list[float]:
    """Pull every plausible number (including percentages) from a string.

    Returns floats. Examples:
        "answer: 2.8%"            → [2.8]
        "T1 = 38.0 N, T12 = 30.0" → [38.0, 30.0]
    """
    out: list[float] = []
    if not text:
        return out
    for m in _NUM_RE.finditer(text):
        try:
            out.append(float(m.group("sign") + m.group("num")))
        except ValueError:
            continue
    return out


def main(argv: Optional[list[str]] = None) -> int:
    """Tiny CLI: ``python tests/dataset.py [--dataset D] [--subject S]``.

    Prints a table of matching problems and exits. Useful for
    sanity-checking the loader without invoking the model.
    """
    import argparse

    p = argparse.ArgumentParser(description="List sciMAS benchmark problems matching filters")
    p.add_argument("--dataset", default=SCIAGENTGYM,
                   metavar="DATASET",
                   help="benchmark dataset to inspect")
    p.add_argument("--root", default=None,
                   help="dataset root override")
    p.add_argument("--split", default="train",
                   help="dataset split/config (ResearchClawBench split; SciPredict: bk or nbk)")
    p.add_argument("--subject", default=None)
    p.add_argument("--topic", default=None)
    p.add_argument("--ids", default=None, help="comma-separated problem ids")
    p.add_argument("--query", default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    dataset_name = normalize_dataset_name(args.dataset)
    root = Path(args.root) if args.root else default_dataset_root(dataset_name)
    probs = load_dataset(root, dataset_name=dataset_name, split=args.split)
    probs = filter_problems(
        probs,
        subject=args.subject,
        topic=args.topic,
        ids=[x.strip() for x in args.ids.split(",") if x.strip()] if args.ids else None,
        query=args.query,
    )
    if args.limit is not None:
        probs = probs[: args.limit]

    print(f"{len(probs)} {dataset_name} problem(s) match from {root}:")
    print(f"  {'id':>14}  {'subject':<14}  {'topic':<28}  tools/files")
    for prob in probs:
        tools_or_files = ", ".join(prob.expected_tools) or f"{len(prob.data_files)} data file(s)"
        if prob.image_paths:
            tools_or_files += f"  +{len(prob.image_paths)} figure(s)"
        if prob.image_paths_missing:
            # Named but not found: the run will go ahead without them.
            tools_or_files += f"  ({len(prob.image_paths_missing)} figure(s) missing)"
        print(
            f"  {str(prob.id):>14}  {prob.subject:<14}  {prob.topic:<28}  "
            f"{tools_or_files}"
        )
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
