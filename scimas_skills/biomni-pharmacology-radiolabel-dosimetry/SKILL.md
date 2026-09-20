---
name: biomni-pharmacology-radiolabel-dosimetry
description: Use to fit radiolabeled-antibody biodistribution time-courses and estimate alpha-particle radiotherapy dosimetry.
x-scimas-role: drug-discovery-scientist
x-scimas-server: biomni-pharmacology
x-scimas-tools:
  - analyze_radiolabeled_antibody_biodistribution
  - estimate_alpha_particle_radiotherapy_dosimetry
---

# Radiolabeled biodistribution / dosimetry

## When to use
- User has time-resolved tissue-uptake data (%ID/g vs hours) for a radiolabeled antibody and wants mono-/bi-exponential fit parameters.
- User has tumor-absorbed-dose / time pairs and wants alpha-emitter dosimetry (e.g. ``At-211``, ``Pb-212``).

## Limitations
- Pure numpy/scipy fits; no compartment-model identifiability check.
