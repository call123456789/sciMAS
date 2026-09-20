---
name: biomni-pharmacology-tumor-inhibition
description: Use to fit xenograft tumor-growth curves and compute treatment / control ratios, TGI%, and growth-delay metrics.
x-scimas-role: drug-discovery-scientist
x-scimas-server: biomni-pharmacology
x-scimas-tools:
  - analyze_xenograft_tumor_growth_inhibition
---

# Xenograft tumor-growth inhibition analysis

## When to use
- User has tumor-volume (mm³) time-courses for vehicle / treatment groups and wants TGI%, AUC ratio, or time-to-progression.

## Limitations
- Returns a research-log string; parse to extract numeric metrics.
