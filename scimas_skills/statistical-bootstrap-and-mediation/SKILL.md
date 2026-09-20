---
name: statistical-bootstrap-and-mediation
description: Use for bootstrap standard error (mean / median / std), EIF-style ψ̂ estimation, and mediation parameter ψ + bootstrap CI.
x-scimas-role: statistical-mathematician
x-scimas-server: math-statistical
x-scimas-tools:
  - bootstrap_standard_error
  - estimate_eif_psi
  - estimate_mediation_parameter_psi
---
# Statistical Bootstrap And Mediation

Use this skill for bootstrap-based SE estimation, efficient
influence function (EIF) ψ̂ estimation, and mediation parameter
ψ = direct + indirect/2 with a 95% CI.

Default 200 bootstrap resamples. Use a fixed seed for reproducibility.