---
name: ms-isotope-pattern
description: Use for theoretical isotope patterns (chlorine / carbon), chlorine-count inference from M/M+2 ratios, and isotope-cluster detection in a peak list.
x-scimas-role: mass-spectrometrist
x-scimas-server: biology-mass-spec
x-scimas-tools:
  - calculate_theoretical_isotope_pattern
  - determine_chlorine_number_from_ratio
  - find_isotope_cluster
---
# MS Isotope Pattern

Use this skill for isotope-pattern calculations: theoretical patterns for n chlorine + n carbon atoms, chlorine-number inference from observed M/M+2 ratio, and isotope-cluster detection.

State the spacing tolerance (default 2 ± 0.05 Da). Reject negative counts.