---
name: waves-spl-algebra
description: Use for sound pressure level arithmetic: dB from pressure, pressure from dB, two-source addition (incoherent vs. coherent), and SPL attenuation with distance.
x-scimas-role: waves-fluid-physicist
x-scimas-server: physics-waves-fluid
x-scimas-tools:
  - spl_from_pressure
  - pressure_from_spl
  - spl_add_two
  - spl_total
  - spl_at_distance
---
# Waves Sound Pressure Level Algebra

Use this skill for acoustic intensity / pressure conversions, two-source addition (with and without correlation), and 1/r² distance fall-off.

State whether sources are coherent (sum of pressures) or incoherent (sum of intensities). For multi-source total, iterate with `spl_total`.