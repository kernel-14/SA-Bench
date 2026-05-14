# Pyramidal Flow Matching for Efficient Video Generative Modeling - Reproduction

This repository aims to reproduce the core contributions of the paper "Pyramidal Flow Matching for Efficient Video Generative Modeling" by Jin et al.

## Overview of Reproduced Components

This reproduction focuses on the following key aspects of the paper:

1.  **Pyramidal Flow Matching (Spatial Pyramid):** Implementation of the piecewise flow for different resolutions, allowing efficient video generation by operating at lower resolutions in earlier stages.
2.  **Unified Training Objective:** The joint optimization of generation and decompression using a single Diffusion Transformer (DiT).
3.  **Inference with Renoising:** The mechanism to ensure continuity of the probability path between different pyramid stages during inference.
4.  **Pyramidal Temporal Condition:** Integration of a temporal pyramid for autoregressive video generation, using progressively compressed, lower-resolution history as conditions.
5.  **Practical Implementations:** Incorporating a standard Transformer architecture with full sequence and blockwise causal attention, and compatible position encoding schemes.
6.  **Model Architecture (Conceptual):** Outlining the use of a DiT-like architecture with a 3D VAE for latent compression.

## Unresolved/Missing Details and Assumptions

*   **Specific Hyperparameters:** While the paper mentions some hyperparameters (e.g., number of pyramid stages = 3, noise strength for history pyramid conditions), full details for all training parameters are not explicitly provided. Default or commonly accepted values will be assumed where necessary.
*   **Dataset Preprocessing:** The exact preprocessing steps for the various image and video datasets (LAION-5B, CC-12M, SA-1B, JourneyDB, WebVid-10M, OpenVid-1M) are not fully detailed. It is assumed standard practices for video/image generation datasets are followed.
*   **Detailed VAE Architecture:** The paper mentions a 3D VAE with an 8x8x8 downsampling ratio, similar to MAGVIT-v2. The exact architecture will be a conceptual implementation based on common VAE designs in video generation.
*   **Training Infrastructure:** Details on distributed training, mixed precision, and specific hardware configurations are not explicitly within the scope of this reproduction, which focuses on the algorithmic aspects.
*   **Evaluation Metrics Implementation:** The implementation of VBench and EvalCrafter metrics is outside the scope, as the focus is on generating the model architecture and training/inference logic.

## Code Structure

-   `pyramidal_flow_matching/`
    -   `model/`: Contains definitions for the Diffusion Transformer (DiT) and the 3D VAE.
    -   `utils/`: Utility functions, including upsampling/downsampling, renoising, and position encoding.
    -   `config/`: Configuration files for model parameters and training settings.
    -   `scripts/`: Placeholder for potential training and inference scripts (not implemented as execution is out of scope).

## Getting Started (Conceptual)

To conceptually reproduce the paper, one would typically:

1.  **Prepare Data:** Download and preprocess the datasets mentioned in the paper.
2.  **Configure Model:** Set up parameters in `config/`.
3.  **Train Model:** Execute a training script using the defined model and loss functions.
4.  **Inference:** Run inference to generate videos.

This reproduction provides the foundational code structure and conceptual implementation of the key algorithmic contributions.
