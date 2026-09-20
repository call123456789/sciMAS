---
name: molecular-plasmid-design
description: Use for plasmid lookup, origin / marker compatibility, transformation-difficulty scoring, and alternative-plasmid recommendation.
x-scimas-role: molecular-biologist
x-scimas-server: biology-molecular
x-scimas-tools:
  - query_plasmid_properties
  - calculate_plasmid_compatibility
  - calculate_transformation_difficulty
  - recommend_alternative_plasmid
---
# Molecular Plasmid Design

Use this skill for plasmid identification, co-transformation compatibility checks (same origin AND same marker → incompatible), and difficulty scoring for a swap.

State the source and target plasmid names. For compatibility, separate "same origin" from "same selection marker" failures.