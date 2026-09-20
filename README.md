# sciMAS

`sciMAS` is a small scientific multi-agent system that uses markdown files as
role prompts and `claude -p` as the execution backend.

## Workflow

### Legacy JSON Mode

1. The planner reads the problem and outputs a JSON topology.
2. The orchestrator executes the planned specialist roles in order.
3. Each role receives the shared context through stdin and can be resumed by
   session id on later calls.
4. A synthesizer combines the specialist outputs into one final answer.

### Python DSL Mode (with Planner-Reviewer)

1. The **planner** reads the problem and generates Python DSL workflow code.
2. The **reviewer** validates the code format and syntax:
   - If valid: proceed to execution
   - If invalid: automatically fix the code and re-validate (up to 3 attempts)
3. The orchestrator executes the workflow, calling specialist agents as defined.
4. If execution fails due to format errors, the reviewer fixes and retries (up to 2 attempts).

See [PLANNER_REVIEWER.md](PLANNER_REVIEWER.md) for details on the two-agent validation system.

## Layout

```text
sciMAS/
  main.py                # entry point — `python main.py "..."`
  orchestrator.py
  claude_runner.py
  plan.py
  prompts/
    planner.md
    planner-reviewer.md  # validates and fixes workflow code
    biologist.md
    chemist.md
    physicist.md
    mathematician.md
    literature-searcher.md
    generalist.md
    synthesizer.md
  scimas_skills/
    literature-openalex-search/
      SKILL.md
    analytical-wet-quant/
      SKILL.md
    ...
  tools/
    chemistry/
      analytical_server.py
      computational_server.py
      environmental_server.py
      organic_server.py
      physical_server.py
    pharma/
      drug_discovery_server.py
      data_server.py
      automl_server.py
    web/
  config/
    mcp.json
  tests/               # SciAgentGYM-main evaluation harness
    dataset.py
    grader.py
    runner.py
    report.py
```

## Run

All commands run from inside the `sciMAS/` directory (the package is not
installed — every script is invoked directly so the import path is
`./`):

```bash
cd sciMAS
python main.py "Why does reaction rate increase with temperature?"
```

You can also provide a file:

```bash
cd sciMAS
python main.py -f problem.txt
```

## Claude CLI pattern

The Python runner uses `subprocess.run` to call `claude -p` with stdin payloads
for shared context. That is the programmatic equivalent of:

```bash
cat context.json | claude -p "Analyze this scientific subtask"
```

The runner also supports session reuse through `--resume <session_id>`.

## Skill routing and MCP tools

Scientific roles use a two-pass skill router by default. First, the role sees a
compact catalog of project-local skill packages and selects the smallest useful
set for the current step. Then the orchestrator injects only the selected
`SKILL.md` bodies into the role prompt and narrows `--allowedTools` to the MCP
tools listed by those skills.

This keeps large scientific tool servers out of the model context until a
narrower tool family is actually needed. Disable it with:

```bash
python main.py --no-skill-routing "problem text"
```

Skill packages live under `scimas_skills/<skill-id>/SKILL.md`. The orchestrator
parses this project-specific frontmatter:

```yaml
---
name: analytical-wet-quant
description: Use for titration, dilution, calibration, pH, precipitation, and absorbance calculations.
x-scimas-role: analytical-chemist
x-scimas-server: chemistry-analytical
x-scimas-tools:
  - titration_strong_acid_base
  - serial_dilution
---
```

The `description` is used in the first-pass skill catalog. The
`x-scimas-tools` list is used to construct exact Claude CLI tool names such as
`mcp__chemistry-analytical__titration_strong_acid_base`.

### Literature search tool

`tools/web/litsearch_server.py` exposes `search_literature`, backed by the
OpenAlex `/works` endpoint. It accepts the existing `query` and `k` arguments
and optional OpenAlex-style filters such as `from_year`, `to_year`, `is_oa`,
`work_type`, `sort`, `corpus`, and raw `filters`.

The planner can route paper, citation, reference, prior-work, and recent
evidence tasks to the `literature-searcher` agent. Under skill routing, that
agent selects `literature-openalex-search`, which exposes only
`mcp__litsearch__search_literature`.

Authentication is resolved in this order, first hit wins:

1. `OPENALEX_API_KEY` (or `OPENALEX_KEY`) from the environment.
2. The file named by `OPENALEX_API_KEY_FILE`.
3. `api_key` in `config/openalex.local.json` — git-ignored, so a live key is
   safe to keep there. This is the easiest place to put one.

Anonymous access still works without any of the three, but OpenAlex budgets it
per client IP at roughly $0.10/day, and at $0.001 per request that runs out
after about 100 searches — after which every call fails with `HTTP 429
Insufficient budget` until midnight UTC. A free OpenAlex account raises the
budget to $1/day and ties it to the key instead of the calling IP, which also
sidesteps a shared or proxy egress IP that has already spent the anonymous
allowance.

Do not commit API keys into `config/mcp.json` or `config/llm_judge.json` —
both are tracked, so a key written there lands in git history. Use
`config/openalex.local.json` (ignored) or the environment instead.

### MADD pharma tools

MADD tools from `../MADD-main` are exposed under the `pharma` discipline:

- `pharmacist` — broad pharmaceutical role that can route across pharma skills.
- `drug-discovery-scientist` — local RDKit drug-likeness/alert screening,
  molecule rendering, local MADD checkpoint diagnostics/prediction, MADD
  molecule generation, and MADD property prediction.
- `pharma-data-specialist` — BindingDB/ChEMBL/UniProt retrieval and
  CSV/XLS/XLSX dataset inspection or column filtering.
- `pharma-ml-engineer` — MADD predictive/generative service state checks and
  validated training request submission.

Local descriptor and rendering tools use RDKit from the `scimas` environment.
MADD checkpoint tools look under `MADD_ROOT`, defaulting to
`/Users/a123/Documents/games/MADD-main`. The bundled MADD pickle checkpoints
were built for MADD's pinned runtime (`scikit-learn==1.2.2`); if the active
Python cannot load them, the tool reports the version mismatch and can be
rerun with a `python_bin` from a compatible MADD environment. Remote MADD
service calls still require `URL_GEN` for generation/generative training and
`URL_PRED` for prediction/predictive training.

In the checked local `MADD-main` tree, the predictive `.pkl/.joblib`
checkpoints are present, and the random GAN checkpoint is present. The disease
CVAE generator directories expected by MADD's `case_generator`
(`autotrain/many_prop_CVAE/weights_*`) are not present, so local disease-case
generation needs those weights restored or the MADD API service configured.

### BiOMNI tools migration

BiOMNI's `tool/` modules from `../Biomni` are wrapped under MCP servers
prefixed `biomni-<category>` and routed onto existing sciMAS roles — no new
role names are introduced. There are currently 14 wrapped servers and 24
biomni-prefixed skill bundles:

| Phase | Categories | Dep weight |
| ----- | ---------- | ---------- |
| 1     | biochemistry, literature, protocols, database, pharmacology | light — `biopython`, `rdkit`, `requests` |
| 2     | molecular_biology, genetics, synthetic_biology, systems_biology | medium — adds `cobra`, `dnachisel`, `liftover`, `pymzml` |
| 3     | genomics, bioimaging, cell_biology, cancer_biology, immunology | heavy — `scanpy`, `scvi-tools`, `SimpleITK`, `cellpose`, `flowkit`, `MACS2`, `HOMER`, `esm`, `transcriptformer` |

Phase 3 wrappers exist on disk but every call surfaces a JSON `ImportError`
until the matching pip line is added to `environment.yml` — wrappers and skill
bundles are still generated so the routing layer can be tested in isolation.

To regenerate the wrappers after editing the bundle table:

```bash
python scripts/biomni_codegen.py --phase 1   # categories from PHASE_1_BUNDLES
python scripts/biomni_codegen.py --phase 2   # merges with existing manifest
python scripts/biomni_codegen.py --phase 3   # merges again — Phase 1/2 preserved
```

The codegen merges with `tools/biomni/_manifest.json` rather than overwriting
it, so rerunning a single phase preserves the others. The orchestrator loads
`BIOMNI_SERVER_TOOLS` from that manifest at import time, which means the
manifest is the **single source of truth** for the runtime allow-list.
`tests/test_biomni.py::BiomniManifestInvariantTests` pins this invariant: any
hand-written skill on disk that is not declared in the manifest, or any skill
whose `x-scimas-tools` list references a tool that the matching server does
not expose, fails CI.

To add another category to the migration, append it to the appropriate
`PHASE_N_BUNDLES` dict in `scripts/biomni_codegen.py` and rerun the codegen —
the new tools and skills appear automatically; no `orchestrator.py` edit is
needed because the inventory is loaded from the manifest.

## Testing against benchmark datasets

The `tests/` package ships an end-to-end harness that loads the
selected benchmark question set, runs sciMAS on each problem, captures
optional MCP tool calls, and grades the answer against the dataset's
available expectations.

Supported datasets:

- `SciAgentGYM` — the original SciAgentGYM JSON dumps.
- `ResearchClawBench` — the Hugging Face dataset
  `InternScience/ResearchClawBench`, with task files mirrored under
  `dataset/ResearchClawBench/repo/`.
- `MADD` — the pharmaceutical multi-agent benchmark table migrated from
  `MADD-main/examples/large_ds_chemical_result.xlsx`.
- `DrugDiscoveryBench` — 82 Harbor-formatted biomedical tasks migrated from
  `DrugDiscoveryBench-main/benchmark`, evaluated with a local deterministic
  rubric proxy when populated rubrics are available.
- `SciPredict` — the Scale AI scientific outcome-prediction CSV benchmark,
  with MCQ, numerical, and free-form questions over biology, chemistry, and
  physics experiments.
- `SMDDBench` — the SMDD-Bench molecular drug-discovery artifact benchmark,
  with task directories defined by `task.yaml` and official scoring through
  the upstream Docker evaluator.
- `BiomniEval1` — the BiOMNI Eval1 biomedical QA benchmark with multiple
  answer-format-constrained task types and exact local scoring.

### Prerequisites

The loader looks for the SciAgentGYM dataset in this order:

1. `$SCIENT_GYM_ROOT` env var
2. `sciMAS/dataset/` (local copy — preferred, no setup)
3. `../SciAgentGYM-main/` (sibling of sciMAS/) — drop the upstream repo here
4. `~/Documents/games/SciAgentGYM-main` (home-dir fallback)

```text
/Users/a123/Documents/games/
├── sciMAS/             ← you are here
│   └── dataset/        ← preferred: 2 JSONs checked in here
└── SciAgentGYM-main/   ← optional: upstream repo
    ├── dataset/
    │   ├── refine_merged_multi_questions.json
    │   └── refine_merged_single_questions.json
    └── gym/
        └── test_images/  ← question figures, named by metadata.image_path
```

**Question figures.** 67 of the 83 multi-question entries name one or more
charts in `metadata.image_path`, and those names are relative to the *upstream
repo root* (`gym/test_images/<id>.png`) — not to the dump that names them. The
loader resolves them against `$SCIENT_GYM_ROOT` → the dataset root →
`<dataset_root>/SciAgentGYM-main` → `<dataset_root>.parent/SciAgentGYM-main` →
`~/Documents/games/SciAgentGYM-main`. Because the dumps normally live in
`sciMAS/dataset/`, which has no `gym/` in it, a name that resolves to nothing is
dropped and reported (`Problem.image_paths_missing` plus a grade note) rather
than advertised to the solver as a file to open. The surviving figures are
copied into the run's output directory (`figure-assets/<problem>/`) before the
run, so the agents read them from inside the repo and the run output keeps its
own copy of what was answered.

ResearchClawBench uses:

1. `$RESEARCH_CLAW_BENCH_ROOT` or `$RESEARCHCLAWBENCH_ROOT`
2. `sciMAS/dataset/ResearchClawBench/repo/`
3. the Hugging Face `datasets` cache if the local table is already cached

MADD uses:

1. `$MADD_BENCHMARK_ROOT` or `$MADD_ROOT`
2. `sciMAS/dataset/MADD/`
3. `../MADD-main/` (sibling checkout)
4. `~/Documents/games/MADD-main`

DrugDiscoveryBench uses:

1. `$DRUG_DISCOVERY_BENCH_ROOT`, `$DRUGDISCOVERYBENCH_ROOT`, or `$DDB_ROOT`
2. `sciMAS/dataset/DrugDiscoveryBench/`
3. `../DrugDiscoveryBench-main/` (sibling checkout)
4. `~/Documents/games/DrugDiscoveryBench-main`

SciPredict uses:

1. `$SCIPREDICT_ROOT` or `$SCI_PREDICT_ROOT`
2. `sciMAS/dataset/SciPredict/`
3. `../scipredict/` (sibling checkout)
4. `~/Documents/games/scipredict`

The loader accepts either GitHub-style `data/main_ds.csv` plus optional
`data/rubrics.csv`, or Hugging Face-style `dataset.csv`. Use `--split nbk`
to omit SciPredict's expert background-knowledge field; the default includes
it.

SMDDBench uses:

1. `$SMDD_BENCH_ROOT` or `$SMDDBENCH_ROOT`
2. `sciMAS/dataset/SMDDBench/`
3. `../SMDD-Bench/` (sibling checkout)
4. `~/Documents/games/SMDD-Bench`

The loader accepts `tasks/` or `tasks_lite/` directories containing
per-task `task.yaml` files. Official scoring is available through the
vendored upstream evaluator at `dataset/SMDDBench/upstream`: build its Docker
image, then run the sciMAS runner with `--smdd-official-eval`. The runner
writes the final answer to the official `agent_outputs/<task-id>/<file_path>`
layout, launches the evaluator image, and feeds the resulting `result.json`
back into the local grader. Without `--smdd-official-eval` or an existing
official result file, local grading reports the score as unavailable.

BiomniEval1 uses:

1. `$BIOMNI_EVAL1_ROOT` or `$BIOMNIEVAL1_ROOT`
2. `sciMAS/dataset/BiomniEval1/`
3. `../Eval1/` (sibling checkout)
4. `~/Documents/games/Eval1`

The loader accepts the Hugging Face parquet file
`biomni_eval1_dataset.parquet`, or CSV/JSONL mirrors with the same columns.

### Quickstart (from `sciMAS/`)

```bash
cd sciMAS

# 1. Inspect what the loader finds — no model calls.
python tests/dataset.py --dataset SciAgentGYM --subject Chemistry --limit 5

# 1b. Inspect ResearchClawBench tasks.
python tests/dataset.py --dataset ResearchClawBench --limit 5

# 1c. Inspect MADD pharmaceutical benchmark tasks.
python tests/dataset.py --dataset MADD --limit 5

# 1d. Inspect DrugDiscoveryBench biomedical tasks.
python tests/dataset.py --dataset DrugDiscoveryBench --limit 5

# 1e. Inspect SciPredict experimental outcome-prediction tasks.
python tests/dataset.py --dataset SciPredict --limit 5

# 1f. Inspect SMDDBench molecular artifact tasks.
python tests/dataset.py --dataset SMDDBench --limit 5

# 1g. Inspect Biomni Eval1 biomedical QA tasks.
python tests/dataset.py --dataset BiomniEval1 --limit 5

# 2. Run sciMAS on the first 3 analytical-chemistry problems.
python tests/runner.py \
    --dataset SciAgentGYM \
    --subject Chemistry \
    --topic "Analytical Chemistry" \
    --limit 3 \
    --label baseline

# 2b. Run one ResearchClawBench task.
python tests/runner.py \
    --dataset ResearchClawBench \
    --ids Astronomy_000 \
    --label rcb-smoke

# 2c. Run five MADD tasks with pharma roles.
python tests/runner.py \
    --dataset MADD \
    --roles pharmacist drug-discovery-scientist \
    --limit 5 \
    --label madd-smoke

# 2d. Run one DrugDiscoveryBench task with biomedical/pharma roles.
python tests/runner.py \
    --dataset DrugDiscoveryBench \
    --ids 69b025e20c10fe76b7aaf812 \
    --roles literature-searcher biologist molecular-biologist geneticist cell-biologist structural-biologist pharmacist pharma-data-specialist drug-discovery-scientist \
    --label ddb-smoke

# 2e. Run one SciPredict task.
python tests/runner.py \
    --dataset SciPredict \
    --limit 1 \
    --label scipredict-smoke

# 2f. Run one SMDDBench task.
python tests/runner.py \
    --dataset SMDDBench \
    --limit 1 \
    --label smdd-smoke

# Optional: build and use the official SMDDBench Docker evaluator.
docker build -t smdd-evals -f dataset/SMDDBench/upstream/evaluator/Dockerfile dataset/SMDDBench/upstream
python tests/runner.py \
    --dataset SMDDBench \
    --limit 1 \
    --smdd-official-eval \
    --label smdd-official-smoke

# 2g. Run one Biomni Eval1 task.
python tests/runner.py \
    --dataset BiomniEval1 \
    --limit 1 \
    --label biomni-eval1-smoke

# 3. Re-grade / aggregate a saved run without re-invoking the model.
python tests/report.py tests/results/SciAgentGYM/<timestamp>-baseline/

# 4. Diff two runs side-by-side.
python tests/report.py --diff \
    tests/results/SciAgentGYM/A/ \
    tests/results/SciAgentGYM/B/
```

### Filter flags

Both `dataset` and `runner` accept:

| Flag | Effect |
|---|---|
| `--dataset DATASET` | `SciAgentGYM` (default), `ResearchClawBench`, `MADD`, `DrugDiscoveryBench`, `SciPredict`, `SMDDBench`, or `BiomniEval1` |
| `--root PATH` | override the selected dataset root |
| `--split SPLIT` | dataset split/config; for SciPredict use `bk`/default to include background knowledge or `nbk` to omit it |
| `--subject SUBJECT` | case-insensitive equality match (e.g. `Chemistry`, `Physics`) |
| `--topic TOPIC` | case-insensitive equality match (e.g. `Analytical Chemistry`, `Diagnostic Analysis`) |
| `--ids 12,17,30` | comma-separated problem ids; ResearchClawBench ids are strings like `Astronomy_000`; MADD ids are strings like `MADD_0001`; DrugDiscoveryBench, SciPredict, SMDDBench, and BiomniEval1 ids are task-id strings |
| `--query SUBSTRING` | substring match across question + filename |
| `--limit N` | cap number of problems |

### Runner-only flags

| Flag | Effect |
|---|---|
| `--roles biologist chemist ...` | whitelist of roles available to the planner (default: all roles) |
| `--mcp-config PATH` | alternate MCP config (default: `sciMAS/config/mcp.json`) |
| `--claude-bin BIN` | claude CLI binary (default: `claude`) |
| `--model MODEL` | model flag passthrough |
| `--timeout SECONDS` | per-call timeout (default: 600) |
| `--allow-role-tool-fallback` | if skill routing selects no skill, expose the full broad-role tool list; default is strict skill-only routing |
| `--smdd-official-eval` | for SMDDBench, write the final answer as the expected artifact and run the official Docker evaluator |
| `--smdd-docker-image IMAGE` | SMDDBench evaluator image name (default: `smdd-evals`) |
| `--smdd-eval-timeout SECONDS` | per-task official evaluator timeout (default: 3600) |
| `--smdd-gpu-id GPU` | optional GPU device id passed to Docker for SMDDBench evaluation |
| `--label LABEL` | short tag for the output directory |
| `--out PATH` | override output directory |

### Output layout

```text
sciMAS/tests/results/<dataset>/<timestamp>-<label>/
├── run.json              # config + aggregate stats + per-problem results
├── summary.md            # human-readable table
├── per-problem/
│   ├── 0012.json         # full grade + tool calls
│   ├── Astronomy_000.json # ResearchClawBench ids are preserved
│   ├── MADD_0001.json    # MADD ids are preserved
│   ├── 69b025e20c10fe76b7aaf812.json # DrugDiscoveryBench ids are preserved
│   ├── 68af5550cbf934a01b45c68f.json # SciPredict ids are preserved
│   ├── 004_lead_optimization_smoke.json # SMDDBench ids are preserved
│   ├── 0.json # BiomniEval1 ids are preserved
│   └── ...
└── scimas-runs/          # raw sciMAS run reports (one dir per problem)
    └── <timestamp>/
        ├── report.json
        ├── report.md
        └── trace.jsonl
```

### Grading signals (per problem)

- `answer_correct` (bool) + `answer_score` (0..1) — exact / substring /
  in-order numeric match → 1.0; partial number match → 0.5 + 0.5·(matched/total);
  token overlap fallback otherwise. ResearchClawBench uses a lightweight
  local checklist text/keyword-overlap proxy; its official benchmark scoring
  still requires a judge model over the saved report, target paper, images,
  and checklist. SciPredict uses exact option matching for MCQ, numeric
  matching for numerical questions, and a local rubric-overlap proxy for
  free-form questions. MADD uses an SSA-style proxy: the score is the
  fraction of golden tool-output molecules from `tools answers` that appear
  in the final answer. SciAgentGYM uses the official `is_answer_correct` LLM
  judge (falling back to the deterministic match above when no judge is
  configured), and attaches the question's own figures from
  `metadata.image_path` as vision content blocks — a judge model that cannot
  take images is retried text-only and the reason lands in `answer_notes`, so
  a chart-reading answer marked wrong by a judge that never saw the chart is
  visible from the row.
- `tool_coverage` (0..1) — fraction of `tool_expected` actually invoked.
  Tools sciMAS doesn't ship are excluded from the denominator so the
  score reflects the achievable upper bound. MADD maps upstream function
  names such as `gen_mols_lung_cancer` to sciMAS pharma MCP calls such as
  `generate_molecules_by_case(case="lung cancer")`.
- MADD summaries also include `madd_fa_avg`, a local FA-style proxy computed
  as `answer_score * tool_coverage` per problem and averaged across the run.
- DrugDiscoveryBench summaries include `ddb_score_100`, the local final-answer
  proxy scaled to 0-100. The official DrugDiscoveryBench score still requires
  populated `rubrics.json` files and its LLM judge over the final answer and
  process trajectory; the public placeholder rubrics score as ungradable.
- SciPredict summaries include `scipredict_score_100`, the local score scaled
  to 0-100. Free-form scoring is a deterministic proxy over the migrated
  rubrics, not the leaderboard judge.
- SMDDBench summaries include `smdd_score_100`. This is populated from
  official evaluator output files. Use `--smdd-official-eval` to generate
  those files during a run; the per-run artifacts land under
  `smdd-official/agent_outputs/` and `smdd-official/results/`.
- BiomniEval1 summaries include `biomni_eval1_score_100`, using local exact
  answer-format scoring for choice labels, gene names, JSON OMIM IDs, causal
  gene lists, and pathogenicity labels.
- Worker MCP tools are narrowed by skill routing by default. Generic
  roles such as `physicist` or `mathematician` first choose from their
  discipline's skill catalog; if no skill is selected, the worker gets no
  MCP tools instead of the whole discipline inventory. Use
  `--allow-role-tool-fallback` to reproduce the legacy broad-tool mode.
- `missing_tools` — expected tools sciMAS *does* ship but didn't call.
- `extra_tools` — tools sciMAS called that weren't expected, plus
  expected tools sciMAS doesn't ship.
- `cost_usd`, `duration_ms` — pass-through from the run report.

See `sciMAS/tests/README.md` for more detail and caveats.
