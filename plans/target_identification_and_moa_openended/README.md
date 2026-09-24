# MAS workflows for DrugQA

This directory contains one restricted sciMAS Python workflow for each of the
50 objects in `dataset/target_identification_and_moa_openended.jsonl`.

## Contents

- `workflows/<question-id>.py`: one `async def workflow(task):` program.
- `manifest.jsonl`: dataset order, public question metadata, and workflow path.

## Design contract

The hidden `answers`, `kegg_evidence`, and `rationale` fields were used only
for offline design review: choosing suitable specialist roles, evidence types,
independent checks, and synthesis depth. Runtime workflow instructions do not
contain reference-answer entities, reference outcome directions, answer-specific
citations, probes, cutoffs, trial identifiers, or wording that tells an agent
which conclusion to prefer.

Every workflow:

1. Starts with exactly `async def workflow(task):`.
2. Uses only sciMAS's restricted `agent` / `parallel` DSL.
3. Passes the original task and explicit upstream outputs between isolated
   workers.
4. Requires independent candidate discovery or fact retrieval before ranking.
5. Uses a `synthesizer` for the final evidence-backed answer.
6. Preserves source, model/cohort, date, evidence tier, and material limitations
   where applicable.
7. Does not infer missing multiple-choice options from hidden benchmark fields.

The manifest intentionally excludes reference answers and other hidden evidence.

## Safety scope

All 50 questions were retained. The workflows are limited to literature and
database retrieval, statistical interpretation, high-level experimental
reasoning, and non-operational mechanism summaries. No question required a
dangerous-procedure skip.

## Validation

Validated against the repository's `parse_and_validate_workflow` function and
`DEFAULT_PLANNER_ROLES`:

- dataset objects: 50
- workflow files: 50
- missing IDs: 0
- extra IDs: 0
- DSL/role validation failures: 0
- explicit reference-answer prompt markers: 0
