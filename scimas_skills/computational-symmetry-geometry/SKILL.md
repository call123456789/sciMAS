---
name: computational-symmetry-geometry
description: Use for point groups, principal moments of inertia, planarity, and simple symmetry element detection.
x-scimas-role: computational-chemist
x-scimas-server: chemistry-computational
x-scimas-tools:
  - point_group_lookup
  - principal_moment_of_inertia
  - compute_planarity
  - detect_sigma_h
  - detect_c3_axis
  - detect_c2_perp_axes
---
# Computational Symmetry And Geometry

Use this skill for molecular symmetry, point-group lookup, inertia tensor analysis, planarity, and simple C2/C3 or mirror-plane checks.

Separate table lookup from geometry-derived inference, and report the tolerance or heuristic when symmetry is detected from coordinates.
