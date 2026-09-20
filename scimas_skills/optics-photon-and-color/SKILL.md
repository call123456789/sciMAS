---
name: optics-photon-and-color
description: Use for photon energy from wavelength or frequency, visible-light colour mapping, and Snell's law refraction.
x-scimas-role: em-optics-physicist
x-scimas-server: physics-electromagnetism
x-scimas-tools:
  - photon_energy_from_wavelength
  - photon_energy_from_frequency
  - wavelength_to_color
  - snells_law
---
# Optics Photon And Color

Use this skill for photon energy (E in J and eV) computed from wavelength or frequency, visible-colour identification from wavelength, and Snell's law refraction (with total-internal-reflection flag).

State which series (Lyman / Balmer / Paschen) the wavelength belongs to when relevant. Reject wavelengths outside 380–780 nm for visible-colour calls.