---
name: ms-spectrum-to-structure
description: Use for spectrum-level summary (peak count, base peak, dynamic range), spectrum-to-SMILES matching within tolerance, and batch-structure screening.
x-scimas-role: mass-spectrometrist
x-scimas-server: biology-mass-spec
x-scimas-tools:
  - analyze_spectrum_characteristics
  - match_spectrum_to_structure
  - batch_structure_screening
---
# MS Spectrum To Structure

Use this skill for high-level MS analysis: spectrum statistics (base peak, dynamic range in dB), single spectrum-to-candidate matching within tolerance, and batch ranking of many candidates.

State the tolerance in Da (default 0.5). Sort candidates by number of matched peaks.