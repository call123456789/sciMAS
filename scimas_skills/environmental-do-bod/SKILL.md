---
name: environmental-do-bod
description: Use for dissolved oxygen, Winkler titration, BOD5, dilution corrections, batch BOD, and oxygen saturation.
x-scimas-role: environmental-chemist
x-scimas-server: chemistry-environmental
x-scimas-tools:
  - dissolved_oxygen_winkler
  - bod5_from_titration
  - calculate_oxygen_depletion
  - apply_dilution_factor
  - validate_bod5_measurement
  - batch_calculate_bod5
  - estimate_optimal_dilution
  - dissolved_oxygen_saturation_curve
---
# Environmental DO And BOD

Use this skill for water-quality oxygen calculations, including Winkler dissolved oxygen, BOD5 depletion, blanks, dilution factors, batch processing, and saturation estimates.

Track day-0/day-5 values carefully and state whether blank correction and dilution correction were applied.
