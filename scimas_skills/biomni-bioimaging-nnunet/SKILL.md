---
name: biomni-bioimaging-nnunet
description: Use to split a multi-channel microscopy volume into per-channel images, prepare the input folder for nnU-Net, run nnU-Net inference, and render the resulting segmentation overlay.
x-scimas-role: cell-biologist
x-scimas-server: biomni-bioimaging
x-scimas-tools:
  - split_modalities
  - prepare_input_for_nnunet
  - segment_with_nn_unet
  - create_segmentation_visualization
---

# nnU-Net segmentation pipeline

## When to use
- User has a multi-channel microscopy volume and wants per-channel splits.
- User wants a folder in nnU-Net's expected format.
- User has an nnU-Net task set up and wants inference on a new image.
- User has a segmentation mask + image and wants an overlay figure.

## Limitations
- Requires ``nnunet``, ``SimpleITK``, ``nibabel`` (NOT installed by default).
- nnU-Net inference requires a pre-trained model checkpoint and the ``nnUNet_results`` env var pointing at it.
- ``segment_with_nn_unet`` typically needs a GPU; CPU fallback is very slow.
