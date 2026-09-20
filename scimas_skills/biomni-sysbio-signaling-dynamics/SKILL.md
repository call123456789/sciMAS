---
name: biomni-sysbio-signaling-dynamics
description: Use to simulate a protein-signaling ODE network, a metabolic perturbation, a protein-dimerization network, or the renin-angiotensin system.
x-scimas-role: structural-biologist
x-scimas-server: biomni-systems_biology
x-scimas-tools:
  - simulate_protein_signaling_network
  - simulate_metabolic_network_perturbation
  - model_protein_dimerization_network
  - simulate_renin_angiotensin_system_dynamics
---

# Signaling & metabolic dynamics

## When to use
- User has a signaling ODE topology and wants a time-course.
- User has a metabolic network stoichiometry and wants a perturbation response.
- User has monomer concentrations + pairwise affinities and wants a dimerization network prediction.
- User wants the RAS / blood-pressure ODE simulated.

## Limitations
- Uses scipy.integrate.solve_ivp; stiff systems may need manual method selection.
- Metabolic perturbation is a stoichiometric + rate-law hybrid; not a full kinetic model.
