---
name: ms-chlorine-quantitation
description: Use for chlorine-number quantitation from a full spectrum: peak extraction, isotope-cluster identification, and end-to-end chlorine count.
x-scimas-role: mass-spectrometrist
x-scimas-server: biology-mass-spec
x-scimas-tools:
  - extract_peaks_from_spectrum
  - analyze_spectrum_for_chlorine
---
# MS Chlorine Quantitation

Use this skill to derive a chlorine count from a raw (m/z, intensity) spectrum.

Set min_intensity to suppress baseline noise. Default spacing is 2.0 Da for ³⁵Cl/³⁷Cl pairs.