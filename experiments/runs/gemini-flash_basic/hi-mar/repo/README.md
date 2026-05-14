# Hi-MAR: Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots - Reproduction

This repository aims to reproduce the core contributions of the paper "Hi-MAR: Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots".

## Implemented Components:

### 1. Hi-MAR Architecture
- **Hierarchical Masked Autoregressive Transformer:**
  - Two-phase modeling:
    - Phase 1: Bidirectional autoregressive modeling over low-resolution visual tokens.
    - Phase 2: Autoregressive modeling over typical dense visual tokens, guided by outputs from Phase 1.
- **Scale-aware Transformer Block:** Incorporates learnable scale vectors injected via AdaLN-Zero operations.
- **Diffusion Transformer Head:** Utilized in the second phase to model interdependency among tokens.

## Codebase Structure:

The codebase will be organized into modules reflecting the architectural components described in the paper.

## Reproduction Notes and Assumptions:

- **VAE (Variational Autoencoder):** The paper mentions using a pre-trained VAE (KL-16 version) from MAR (Li et al., 2024) for image tokenization. For this reproduction, I will assume the VAE is available or can be re-implemented based on the MAR paper if necessary. The focus will be on the Hi-MAR specific architecture.
- **Continuous-valued tokens:** The paper emphasizes the use of continuous-valued tokens, moving away from discrete tokenization. The model implementation will reflect this.
- **Training Details:** Specific training parameters like learning rates, optimizers, and masking strategies are noted and will be reflected in a configuration file or training script if applicable.
- **Diffusion Loss Function:** The objective function for the diffusion heads is based on the standard denoising process as described in the paper.

## Further Work / Missing Details:

- Detailed implementation of the pre-trained VAE if not readily available or if deviations from MAR's implementation are required.
- Full training and evaluation scripts to replicate experimental results.
- Integration of class-conditional and text-to-image generation contexts.

This README will be continuously updated as the reproduction progresses.
