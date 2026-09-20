# sciMAS test framework

End-to-end runner for evaluating sciMAS against `SciAgentGYM`,
`ResearchClawBench`, `MADD`, `DrugDiscoveryBench`, `SciPredict`,
`SMDDBench`, and `BiomniEval1` question sets.
Designed for two
workflows:

1. **Re-grade a saved run** — never invoke the model again, just
   inspect the per-problem JSON files in a `tests/results/<dataset>/<run>/`
   directory and print an aggregate table.
2. **Compare two configurations** — e.g. legacy `chemist` vs the five
   new sub-chemists, or different `permission-mode` settings.

## Quickstart

All commands run from inside `sciMAS/` (the package is not installed —
scripts are invoked directly so the import path is `./`):

```bash
# 0. Activate the conda env (created from environment.yml at the repo root).
source activate scimas
cd /Users/a123/Documents/games/sciMAS

# 1. Inspect what the loader finds (no model calls).
python tests/dataset.py --dataset SciAgentGYM --subject Chemistry
python tests/dataset.py --dataset ResearchClawBench --limit 5
python tests/dataset.py --dataset MADD --limit 5
python tests/dataset.py --dataset DrugDiscoveryBench --limit 5
python tests/dataset.py --dataset SciPredict --limit 5
python tests/dataset.py --dataset SMDDBench --limit 5
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

# 2c. Run five MADD pharmaceutical benchmark tasks.
python tests/runner.py \
    --dataset MADD \
    --roles pharmacist drug-discovery-scientist \
    --limit 5 \
    --label madd-smoke

# 2d. Run one DrugDiscoveryBench biomedical task.
python tests/runner.py \
    --dataset DrugDiscoveryBench \
    --ids 69b025e20c10fe76b7aaf812 \
    --roles literature-searcher biologist molecular-biologist geneticist cell-biologist structural-biologist pharmacist pharma-data-specialist drug-discovery-scientist \
    --label ddb-smoke

# 2e. Run one SciPredict scientific outcome-prediction task.
python tests/runner.py \
    --dataset SciPredict \
    --limit 1 \
    --label scipredict-smoke

# 2f. Run one SMDDBench molecular artifact task.
python tests/runner.py \
    --dataset SMDDBench \
    --limit 1 \
    --label smdd-smoke

# Optional: run the official SMDDBench Docker evaluator.
docker build -t smdd-evals -f dataset/SMDDBench/upstream/evaluator/Dockerfile dataset/SMDDBench/upstream
python tests/runner.py \
    --dataset SMDDBench \
    --limit 1 \
    --smdd-official-eval \
    --label smdd-official-smoke

# 2g. Run one Biomni Eval1 biomedical QA task.
python tests/runner.py \
    --dataset BiomniEval1 \
    --limit 1 \
    --label biomni-eval1-smoke

# 3. Re-grade / aggregate a saved run.
python tests/report.py tests/results/SciAgentGYM/<timestamp>-baseline/

# 4. Diff two runs.
python tests/report.py --diff tests/results/SciAgentGYM/A/ tests/results/SciAgentGYM/B/
```

## CLI flags

### `dataset`
- `--dataset DATASET` — `SciAgentGYM` (default),
  `ResearchClawBench`, `MADD`, `DrugDiscoveryBench`, `SciPredict`, or
  `SMDDBench`, or `BiomniEval1`
- `--root PATH` — override the selected dataset root
- `--split SPLIT` — dataset split/config; for SciPredict use `bk`/default
  to include expert background knowledge or `nbk` to omit it
- `--subject SUBJECT` — case-insensitive equality match (e.g.
  `Chemistry`, `Physics`, `Life Science`)
- `--topic TOPIC` — case-insensitive equality match (e.g.
  `Analytical Chemistry`)
- `--ids 12,17,30` — comma-separated list of problem ids; for
  ResearchClawBench use ids like `Astronomy_000`; for MADD use ids
  like `MADD_0001`; for DrugDiscoveryBench, SciPredict, SMDDBench, and
  BiomniEval1 use task ids
- `--query SUBSTRING` — substring match across question + filename
- `--limit N` — cap number of problems printed

### `runner`
Inherits all `dataset` flags (filter applies identically), plus:
- `--roles biologist chemist pharmacist ...` — whitelist of roles available to
  the planner (default: all generic and sub-expert roles). Note that
  role names like `analytical-chemist` still work; sub-experts each see
  only their own server's tools unless a narrower skill is selected.
- `--mcp-config PATH` — alternate MCP config (defaults to
  `sciMAS/config/mcp.json`)
- `--claude-bin BIN` — claude CLI binary (default: `claude`)
- `--model MODEL` — model flag passthrough
- `--timeout SECONDS` — per-call timeout (default: 600)
- `--allow-role-tool-fallback` — if skill routing selects no skill,
  expose the broad role's full discipline tool list. By default, worker
  tools stay strict to selected skills only.
- `--smdd-official-eval` — for SMDDBench, write the final answer as the
  expected artifact and run the official Docker evaluator
- `--smdd-docker-image IMAGE` — SMDDBench evaluator image name (default:
  `smdd-evals`)
- `--smdd-eval-timeout SECONDS` — per-task official evaluator timeout
  (default: 3600)
- `--smdd-gpu-id GPU` — optional GPU device id passed to Docker for
  SMDDBench evaluation
- `--label LABEL` — short tag for the output directory
- `--out PATH` — override output directory

### `report`
- positional `run_dirs` — one or more `tests/results/<dataset>/<run>/` paths
- `--diff` — when ≥ 2 dirs are given, print side-by-side comparison

## Output layout

```
tests/results/<dataset>/<timestamp>-<label>/
├── run.json              # config + aggregate stats + per-problem results
├── summary.md            # human-readable table
├── per-problem/
│   ├── 0012.json         # one per problem, contains full grade + tool calls
│   ├── Astronomy_000.json # ResearchClawBench ids are preserved
│   ├── MADD_0001.json    # MADD ids are preserved
│   ├── 69b025e20c10fe76b7aaf812.json # DrugDiscoveryBench ids are preserved
│   ├── 68af5550cbf934a01b45c68f.json # SciPredict ids are preserved
│   ├── 004_lead_optimization_smoke.json # SMDDBench ids are preserved
│   ├── 0.json # BiomniEval1 ids are preserved
│   └── ...
└── scimas-runs/          # raw sciMAS run reports (one timestamp dir per problem)
    └── 20260909-181012/
        ├── report.json
        ├── report.md
        └── trace.jsonl
```

## Grading signals

Each problem produces a `GradingResult` with:

- `answer_correct` (bool) and `answer_score` (0..1)
  - exact / substring → 1.0
  - all expected numbers matched in order → 1.0
  - partial number match → 0.5 + 0.5·(matched/total)
  - token overlap fallback otherwise
  - ResearchClawBench dispatches to the official multimodal LLM judge
    (`tests/rcb_official_judge.py`, mirroring InternScience's
    `evaluation/score.py`) when `RESEARCH_CLAW_JUDGE_*` env vars are
    set; otherwise it falls back to a lightweight local checklist
    text/keyword-overlap proxy. See the section below.
  - MADD uses an SSA-style proxy: the score is the fraction of golden
    tool-output molecules from the migrated `tools answers` column that
    appear in the final answer.
  - DrugDiscoveryBench uses a local deterministic proxy over populated
    `ground_truth` / `outcome_rubrics`. The public placeholder rubrics are
    empty, so they are reported as ungradable until upstream rubrics are
    populated.
  - SciPredict uses exact MCQ option matching, numeric/range matching, and
    a deterministic rubric-overlap proxy for free-form questions.
  - SMDDBench reads official evaluator result files when present. Use
    `--smdd-official-eval` to generate them through the upstream Docker
    evaluator during the run; without official output, local grading marks
    the score unavailable.
  - BiomniEval1 uses local exact scoring over each public answer format.
  - SciAgentGYM dispatches to the official `is_answer_correct` LLM judge when
    the judge is configured (see the SciAgentGYM section below), falling back
    to the match rules above when it is not. The question's own figures from
    `metadata.image_path` are attached to that judge call as vision content
    blocks; a judge model that cannot take images is retried text-only and the
    reason is recorded in `answer_notes`.
- `tool_coverage` (0..1) — fraction of `tool_expected` actually
  invoked by sciMAS; tools that sciMAS doesn't even ship are excluded
  from the denominator so the score reflects the achievable upper
  bound. MADD maps upstream function names such as
  `gen_mols_lung_cancer` to sciMAS pharma MCP calls such as
  `generate_molecules_by_case(case="lung cancer")`
- MADD summaries include `madd_fa_avg`, a local FA-style proxy computed
  as `answer_score * tool_coverage` per problem and averaged across the
  run
- DrugDiscoveryBench summaries include `ddb_score_100`, the local proxy
  score scaled to 0-100
- SciPredict summaries include `scipredict_score_100`, the local proxy score
  scaled to 0-100
- SMDDBench summaries include `smdd_score_100`, populated from official
  evaluator output files. During `--smdd-official-eval` runs, artifacts are
  saved under `smdd-official/agent_outputs/` and results under
  `smdd-official/results/`.
- BiomniEval1 summaries include `biomni_eval1_score_100`, the local exact
  score scaled to 0-100
- `missing_tools` — expected tools sciMAS *does* ship but didn't call
- `extra_tools` — tools sciMAS called that weren't expected, plus
  expected tools that sciMAS doesn't ship
- `cost_usd`, `duration_ms` — pass-through from the run report

## Notes

- The runner defaults to plain text Claude output, matching the
  `--no-capture-tool-calls` behavior used for models/providers whose
  `stream-json` path is unreliable. In this mode, answer quality is
  evaluated but tool coverage is unavailable. Pass
  `--capture-tool-calls` when you explicitly want `stream-json` MCP
  tool-call capture.
- Generic roles such as `physicist`, `chemist`, `biologist`,
  `mathematician`, and `pharmacist` can select from all skills in their
  discipline. If no skill is selected, the default strict mode gives
  that worker no MCP tools rather than all tools in the discipline. Use
  `--allow-role-tool-fallback` only when deliberately comparing against
  the legacy broad-tool mode.
- The `tool_expected` field comes straight from SciAgentGYM metadata.
  Several chemistry problems reference tools we didn't port (e.g.
  `chem_visualizer`, `optimize_geometry`); the grader reports those as
  `extra_tools` so you can see which ones sciMAS missed *and* which
  ones were outside scope.
- For dataset loading, the runner searches:
  - `SciAgentGYM`: `$SCIENT_GYM_ROOT`, local `dataset/`, sibling
    `../SciAgentGYM-main`, then `~/Documents/games/SciAgentGYM-main`.
    `metadata.image_path` names its figures relative to the *upstream repo
    root* (`gym/test_images/*.png`), so the figure root is searched
    separately: `$SCIENT_GYM_ROOT` / `$SCIENTAGENTGYM_ROOT`, the dataset
    root, `<dataset_root>/SciAgentGYM-main`,
    `<dataset_root>.parent/SciAgentGYM-main`, then
    `~/Documents/games/SciAgentGYM-main`. A name that resolves to nothing is
    dropped and reported rather than passed on as a path to open.
  - `ResearchClawBench`: `$RESEARCH_CLAW_BENCH_ROOT` or
    `$RESEARCHCLAWBENCH_ROOT`, then `dataset/ResearchClawBench/repo`
  - `MADD`: `$MADD_BENCHMARK_ROOT` or `$MADD_ROOT`, then
    `dataset/MADD`, sibling `../MADD-main`, then
    `~/Documents/games/MADD-main`
  - `DrugDiscoveryBench`: `$DRUG_DISCOVERY_BENCH_ROOT`,
    `$DRUGDISCOVERYBENCH_ROOT`, or `$DDB_ROOT`, then
    `dataset/DrugDiscoveryBench`, sibling `../DrugDiscoveryBench-main`,
    then `~/Documents/games/DrugDiscoveryBench-main`
  - `SciPredict`: `$SCIPREDICT_ROOT` or `$SCI_PREDICT_ROOT`, then
    `dataset/SciPredict`, sibling `../scipredict`, then
    `~/Documents/games/scipredict`
  - `SMDDBench`: `$SMDD_BENCH_ROOT` or `$SMDDBENCH_ROOT`, then
    `dataset/SMDDBench`, sibling `../SMDD-Bench`, then
    `~/Documents/games/SMDD-Bench`
  - `BiomniEval1`: `$BIOMNI_EVAL1_ROOT` or `$BIOMNIEVAL1_ROOT`, then
    `dataset/BiomniEval1`, sibling `../Eval1`, then
    `~/Documents/games/Eval1`

## Caveats

- The model can produce subtly different outputs across runs. We
  recommend running each configuration 2-3 times and averaging.
- A run is non-deterministic and costs real money (~$0.05–$0.20 per
  problem at current rates). Use `--limit` for dry runs.
- `report --diff` compares two runs on the *intersection* of their
  problem sets. If A and B filtered to different problems, only the
  overlap is shown.

## ResearchClawBench official LLM judge

sciMAS can score ResearchClawBench with the upstream multimodal LLM
judge (InternScience/ResearchClawBench `evaluation/score.py`) instead
of the local text/keyword-overlap proxy. The judge is reimplemented
in `tests/rcb_official_judge.py` using raw `httpx` against any
OpenAI-compatible `/chat/completions` endpoint — no `structai`,
`openai`, or `researchharness` pip dependency is required.

The official `RUBRIC` (Mode A "Objective" / Mode B "Subjective",
0–100 scale where 50 = "matches paper"), per-item prompt templates,
and weighted aggregation match upstream behaviour exactly.

### Environment variables

`RESEARCH_CLAW_JUDGE_*` are honoured first; the bare `JUDGE_*` names
that upstream's Web UI / CLI use are accepted as a fallback.

| Variable | Default | Notes |
|---|---|---|
| `RESEARCH_CLAW_JUDGE_API_KEY` | — | **Required** to enable the judge |
| `RESEARCH_CLAW_JUDGE_API_BASE` | `https://api.openai.com/v1` | Any OpenAI-compatible endpoint |
| `RESEARCH_CLAW_JUDGE_MODEL` | — | **Required**; must support vision (e.g. `gpt-5.1`, `claude-sonnet-5`) |
| `RESEARCH_CLAW_JUDGE_DISABLED` | `0` | Set to `1` to force the proxy even with a key set |
| `RESEARCH_CLAW_JUDGE_TIMEOUT` | `180` | Per-request timeout in seconds |
| `RESEARCH_CLAW_JUDGE_MAX_WORKERS` | `8` | Per-checklist-item concurrency (matches upstream `multi_thread`) |

### CLI flag

`python tests/runner.py --research-claw-judge {auto,force,off} …`

- `auto` (default) — use the judge when env is set, proxy otherwise.
- `force` — require env; abort with a clear error if missing.
- `off` — skip the judge even when env is set (set internally to
  `RESEARCH_CLAW_JUDGE_DISABLED=1` for the process).

### Behaviour

- One vision-capable chat-completions call per checklist item.
  `type=text` items send text only; `type=image` items send the
  target figure from `target_study/images/` as an OpenAI vision
  content block.
- sciMAS agents do not generate figures, so only the ground-truth
  target image is attached (the upstream prompt assumes
  `target_image + agent_images[]`; we send just the target).
- Missing target image, non-whitelisted extension (`.svg` rejected
  for XSS reasons), or oversized payload (>4 MB after Pillow
  downscale) → graceful degradation to text-only scoring with
  `degraded=True` recorded per item.
- Per-item scores aggregate to `total_score` (0–100); sciMAS
  normalises to `answer_score` (0–1.0) and uses `>= 0.5` for
  `answer_correct` (equivalent to upstream's "matches paper" bar).
- Judge failures (timeout, parse error, 401, ...) fall back to the
  proxy so a missing API key never breaks a batch run; the failure
  reason is recorded in `answer_notes`.
- Per-run cache: `tests/results/<dataset>/<run>/judge_cache/<task>__<hash16>__<safe_model>__v1.json`
  (atomic write). Re-grading an existing run never re-bills the API.

### Running

```bash
export RESEARCH_CLAW_JUDGE_API_KEY=sk-...
export RESEARCH_CLAW_JUDGE_MODEL=gpt-5.1

python tests/runner.py --dataset ResearchClawBench --ids Astronomy_000 \
    --label rcb-judge-smoke

# Inspect the per-item reasoning for one task:
python -c "import json; d=json.load(open('tests/results/ResearchClawBench/<run>/per-problem/Astronomy_000.json')); print(d['grade']['answer_notes'])"
```

To re-grade an existing run with the judge without re-running agents:

```bash
RESEARCH_CLAW_JUDGE_API_KEY=sk-... python tests/report.py \
    tests/results/ResearchClawBench/20260913-144630-rcb-5-strict/
```

Note: `report.py` does not currently call the judge (it reads saved
JSON). For an A/B comparison, run the runner twice with
`--research-claw-judge=off` then `auto`, or use the cache above to
switch the judge on without re-charging.

## SciAgentGYM LLM judge and question figures

SciAgentGYM grading uses the same judge credentials as ResearchClawBench
(`RESEARCH_CLAW_JUDGE_*` / `JUDGE_*` / `SCIMAS_LLM_JUDGE_*`, or
`config/llm_judge.local.json`) and mirrors upstream's
`is_answer_correct` prompt: question, standard answer, model answer, plus
the optional context blocks described in `config/llm_judge.json`. The
gold is the entry's `answer` field; `metadata.golden_answer` is the
reference *method* and is shown to the judge as context only.

**Figures.** 67 of the 83 multi-question entries name one or more charts in
`metadata.image_path`, relative to the upstream repo root
(`gym/test_images/*.png`). The loader resolves each name (see the dataset
search list above) and the runner copies the ones that exist into the run's
output directory under `figure-assets/<problem>/`. From there:

- the **solver** is handed the copies as a "Question figures" section appended
  to the problem text passed to `orchestrator.run()` — `Problem.question`
  itself is never modified, so the judged question stays the bare question;
- the **judge** receives them as OpenAI vision content blocks in the same
  user message as the prompt, and the prompt carries a numbered rule stating
  that these figures belong to the *question*, not to the model's answer.
  (This is the opposite of the ResearchClawBench contract above, where the
  attached image is the ground-truth *target* the report is scored against.)

Behaviour and limits:

- With no figures — the 48 single-question entries, every write-in, and the
  CLI — the request body is byte-identical to before: the user content stays
  a plain string and no rule is added.
- Caps are code constants in `tests/llm_judge.py`: `MAX_IMAGE_BYTES` (4 MB,
  downscaled with Pillow beyond that), `MAX_TOTAL_IMAGE_BYTES` (12 MB) and
  `MAX_IMAGES` (12 — the largest dump entry carries 8). They are not read
  from `config/llm_judge.json`.
- If the configured model cannot accept image content, the vision call fails
  (HTTP 400), the judge is retried text-only, and the verdict still arrives.
  Either way the grade notes say which happened: `judge also saw N question
  figure(s) …`, or `question figure(s) NOT sent to the judge: …` via
  `llm_judge.get_last_image_note()`. A figure named by the dump but missing on
  disk is likewise reported.
- There is **no per-run cache** on this path (unlike the ResearchClawBench
  judge above): every grading run re-encodes and re-bills the figures, roughly
  1–2k input tokens per chart. Keep `--limit` in mind when re-grading.
