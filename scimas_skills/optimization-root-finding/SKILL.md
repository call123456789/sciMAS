---
name: optimization-root-finding
description: Use for 1-D root-finding: bisection (heuristic, requires caller-supplied f), Newton–Raphson step (caller supplies f and f'), and fixed-point iteration.
x-scimas-role: optimization-mathematician
x-scimas-server: math-optimization
x-scimas-tools:
  - bisection_root
  - newton_raphson_step
  - fixed_point_iteration
---
# Optimization Root Finding

Use this skill for 1-D root-finding. Caller supplies f(x₀), f(x₁),
and any derivative values needed.

Reject derivative = 0 for Newton–Raphson. Use a fixed-point
iteration only when |g'(x*)| < 1.