---
name: cell-experimental-design
description: Use for experimental-design sanity checks: presence of controls, ATP and complex-activity coverage, and overall validity flag.
x-scimas-role: cell-biologist
x-scimas-server: biology-cell
x-scimas-tools:
  - experimental_design_validator
---
# Cell Experimental Design

Use this skill to validate a proposed list of cell-biology assays: each item has a name and a type (atp / complex_activity / glucose_uptake / control).

Report which required assays are present and flag missing controls.