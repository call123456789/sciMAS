---
name: cm-band-structure
description: Use for 1D tight-binding band structure, 2D square-lattice tight-binding band structure, and diamond-crystal bond-angle geometry.
x-scimas-role: condensed-matter-physicist
x-scimas-server: physics-condensed-matter
x-scimas-tools:
  - tight_binding_1d
  - tight_binding_2d_square
  - diamond_bond_angles
---
# CM Band Structure

Use this skill for nearest-neighbour tight-binding dispersion in 1D / 2D and geometric structure of a diamond cubic crystal.

Report the hopping parameter t (sign) and the lattice constant. For 2D, return E(kx, ky) on a sampled grid.