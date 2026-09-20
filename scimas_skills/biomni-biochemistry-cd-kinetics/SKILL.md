---
name: biomni-biochemistry-cd-kinetics
description: Use for circular-dichroism secondary-structure / thermal-melting analysis and for Michaelis-Menten / dose-response fits of protease and enzyme time-course fluorescence data.
x-scimas-role: molecular-biologist
x-scimas-server: biomni-biochemistry
x-scimas-tools:
  - analyze_circular_dichroism_spectra
  - analyze_protease_kinetics
  - analyze_enzyme_kinetics_assay
---

# CD spectroscopy & enzyme kinetics

## When to use
- User has CD spectra (wavelength vs CD signal) and wants protein / nucleic-acid secondary-structure classification or Tm.
- User has fluorescence vs time data at multiple substrate concentrations and wants Vmax / Km / kcat from Michaelis-Menten.
- User has dose-response data with modulators and wants IC50 / Hill slope.

## Limitations
- Output is a research-log string; numbers are not pre-parsed — downstream steps must grep ``Km:`` / ``Vmax:`` / ``IC50:`` from the log.
- Files are written under ``/tmp/scimas_biomni/<tool>/`` by default; pass ``output_dir=""`` to accept the default or supply a writable path.
- ``analyze_enzyme_kinetics_assay`` synthesizes a time-course internally; do NOT use it to fit real experimental time-courses — use ``analyze_protease_kinetics`` for that.
