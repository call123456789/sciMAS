---
name: genetics-chromosome-mitosis
description: Use for chromosome kinetochore spindle tension, segregation simulation, mitotic-behaviour classification, and stability-factor scoring.
x-scimas-role: geneticist
x-scimas-server: biology-genetics
x-scimas-tools:
  - calculate_spindle_tension
  - simulate_chromosome_segregation
  - analyze_mitotic_behavior
  - evaluate_chromosome_stability_factors
---
# Genetics Chromosome And Mitosis

Use this skill for kinetochore / centromere mechanics: tension distribution, simulated pole-ward segregation, mitotic behaviour classification (mono / holo / acrocentric), and stability scoring.

State the cell length and number of centromeres. For segregation, flag lagging chromosomes at > 10% offset from the centre.