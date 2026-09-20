---
name: cell-chipseq-peaks
description: Use for ChIP-seq crosslinking-efficiency estimation, peak signal simulation, peak-location prediction, and disappearing-peak analysis.
x-scimas-role: cell-biologist
x-scimas-server: biology-cell
x-scimas-tools:
  - calculate_crosslinking_efficiency
  - simulate_chipseq_signal
  - predict_peak_locations
  - analyze_disappearing_peaks
---
# Cell ChIP-Seq Peaks

Use this skill for ChIP-seq protocol and signal analysis: fixation-method efficiency, sharp vs. broad binding-mode peak simulation, peak-position prediction, and disappearing-peak identification.

State the fixation method and interaction type. Use the fold-change threshold to define "disappearing" peaks.