---
name: optimization-gradient-and-projection
description: Use for gradient-descent steps, box projection, Armijo-style backtracking line search, and quantum Fisher information for pure-state parameter estimation.
x-scimas-role: optimization-mathematician
x-scimas-server: math-optimization
x-scimas-tools:
  - gradient_descent_step
  - project_box
  - armijo_step_size
  - quantum_fisher_information_pure
---
# Optimization Gradient And Projection

Use this skill for single-step gradient descent, projection onto a
box [lo, hi]^n, Armijo backtracking line search (heuristic), and
quantum Fisher information for pure states (returns 4·Var(G)).

State the learning rate for the descent step. Reject |gradient| ≤ 0
for Armijo.