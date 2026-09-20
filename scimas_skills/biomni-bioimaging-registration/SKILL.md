---
name: biomni-bioimaging-registration
description: Use to run rigid / affine / deformable registration on a pair of medical images, batch-register a folder of moving images, or compute a similarity metric between two images.
x-scimas-role: cell-biologist
x-scimas-server: biomni-bioimaging
x-scimas-tools:
  - quick_rigid_registration
  - quick_affine_registration
  - quick_deformable_registration
  - batch_register_images
  - calculate_similarity_metrics
---

# Medical-image registration (rigid / affine / deformable)

## When to use
- User has a moving + fixed image pair and wants rigid / affine / deformable registration.
- User has a folder of moving images and a fixed image and wants batch registration.
- User has two aligned images and wants a similarity metric (NCC / MI / SSIM).

## Limitations
- Requires ``SimpleITK`` + ``nibabel`` (NOT installed by default).
- Deformable registration is slow on CPU; a GPU build of SimpleITK is preferred.
- ``create_registration_visualization`` is described but has no matching implementation, so the codegen skips it.
