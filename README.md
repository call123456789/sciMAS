# sciMAS

A scientific multi-agent system. Markdown files define specialist roles, `claude -p`
is the execution backend, and an orchestrator plans a workflow and dispatches each
step to the role that owns it.

Ships 33 role prompts, 133 skill packages, and 39 MCP tool servers spanning
chemistry, physics, biology, pharmacology, mathematics, and literature search.

## Install

```bash
conda env create -f environment.yml && conda activate scimas
claude --version   # the Claude CLI must be on PATH
```

## Run

```bash
python main.py "Why does reaction rate increase with temperature?"
python main.py -f problem.txt
```

Reports land in `runs/<timestamp>/` as `report.json` and `report.md`. Useful
flags: `--roles` to whitelist planner roles, `--planner-mode`, `--model`,
`--no-mcp`, `--no-skill-routing`. `python main.py --help` lists the rest.

## Planner modes

- **`python-dsl`** (default) — the planner emits restricted async workflow
  source. A **reviewer** agent validates it and repairs it in place: up to 3
  validation rounds before execution and 2 retry rounds during it.
- **`legacy-json`** — the planner emits a JSON topology, and the orchestrator
  runs those roles in order. Select with `--planner-mode legacy-json`.

See [PLANNER_REVIEWER.md](PLANNER_REVIEWER.md).

## Layout

```text
sciMAS/
  main.py                # entry point
  orchestrator.py        # planning, dispatch, synthesis
  claude_runner.py       # `claude -p` wrapper; MCP config resolution
  plan.py, workflow_dsl.py, tool_spec.py
  web_dashboard.py       # browser UI over run reports
  prompts/               # one markdown file per role
  scimas_skills/         # SKILL.md packages, selected per step
  tools/                 # MCP servers: chemistry physics biology
                         # pharma mathematics biomni web
  config/                # mcp.json, llm_judge.json, *.local.json
  tests/                 # benchmark harness (see tests/README.md)
  scripts/               # codegen + dataset sync helpers
```

## Skill routing

Roles use a two-pass router. The role first sees a compact catalog of skill
packages and picks the smallest useful set; the orchestrator then injects only
those `SKILL.md` bodies and narrows `--allowedTools` to the tools they declare.

This keeps large tool servers out of the context until a narrow tool family is
actually needed. `--no-skill-routing` restores full per-role tool lists, and
`--allow-role-tool-fallback` keeps routing but falls back to the broad list when
a role selects no skill.

A skill declares its role and tools in frontmatter:

```yaml
---
name: analytical-wet-quant
description: Use for titration, dilution, calibration, pH, and absorbance calculations.
x-scimas-role: analytical-chemist
x-scimas-server: chemistry-analytical
x-scimas-tools:
  - titration_strong_acid_base
  - serial_dilution
---
```

`x-scimas-tools` becomes exact CLI tool names such as
`mcp__chemistry-analytical__titration_strong_acid_base`.

## Tool servers

| discipline | servers | notes |
|---|---|---|
| chemistry | 5 | analytical, computational, environmental, organic, physical |
| physics | 5 | classical, electromagnetism, waves/fluid, condensed matter, quantum/atomic |
| biology | 5 | molecular, genetics, cell, structural, mass spec |
| pharmacology | 4 | drug discovery, DrugSDA, data, AutoML |
| mathematics | 5 | algebraic, statistical, geometric, optimization, numerical |
| biomni | 14 | BiOMNI `tool/` modules wrapped per category, routed onto existing roles |
| web | 1 | OpenAlex literature search |

Local pharma descriptor and rendering tools use RDKit from the `scimas` env; MADD
checkpoint tools read `MADD_ROOT`. The biomni wrappers are generated — edit the
`PHASE_N_BUNDLES` tables in `scripts/biomni_codegen.py` and rerun it, which merges
into `tools/biomni/_manifest.json`. That manifest is the single source of truth for
the runtime allow-list, and `tests/test_biomni.py` pins the invariant. Phase 3
biomni categories import-error until their pip lines are added to `environment.yml`.

## Configuration and credentials

**No credentials are committed here.** `config/mcp.json` and
`config/llm_judge.json` are tracked and carry no keys; per-machine secrets live in
git-ignored siblings that are merged at launch:

| file | holds |
|---|---|
| `config/mcp.local.json` | MCP server `env` values, e.g. `DRUGSDA_API_KEY` |
| `config/llm_judge.local.json` | judge `api_key` / `api_base` / `model` |
| `config/openalex.local.json` | OpenAlex `api_key` |

Environment variables win over every file. Never write a key into `mcp.json` or
`llm_judge.json` — being tracked, it lands in git history.

Anonymous OpenAlex access works but is budgeted per client IP at ~$0.10/day;
a free key raises that to $1/day and ties it to the key rather than the IP.

## Benchmarks

`tests/` ships an end-to-end harness: it loads a benchmark question set, runs
sciMAS on each problem, captures MCP tool calls, and grades the answer against
the dataset's expectations.

| dataset | what it is |
|---|---|
| `SciAgentGYM` | the original SciAgentGYM JSON dumps (default) |
| `ResearchClawBench` | `InternScience/ResearchClawBench` |
| `MADD` | pharmaceutical multi-agent benchmark |
| `DrugDiscoveryBench` | 82 Harbor-formatted biomedical tasks |
| `SciPredict` | Scale AI outcome-prediction CSV |
| `SMDDBench` | molecular drug-discovery artifacts |
| `BiomniEval1` | BiOMNI Eval1 biomedical QA |

Each dataset is found via its own `*_ROOT` env var, then `dataset/<name>/`, then a
sibling checkout, then a home-dir fallback. Datasets are git-ignored — fetch them
separately.

```bash
python tests/dataset.py --dataset SciAgentGYM --subject Chemistry --limit 5
python tests/runner.py  --dataset SciAgentGYM --limit 3 --label baseline
python tests/report.py  tests/results/SciAgentGYM/<timestamp>-baseline/
```

Full CLI flags, output layout, per-dataset grading signals, and the LLM-judge
caveats are in **[tests/README.md](tests/README.md)**.

## Docs

- [tests/README.md](tests/README.md) — benchmark harness, flags, grading
- [PLANNER_REVIEWER.md](PLANNER_REVIEWER.md) — the two-agent validation system
- [DATASETS_ANALYSIS.md](DATASETS_ANALYSIS.md) — dataset-by-dataset analysis
