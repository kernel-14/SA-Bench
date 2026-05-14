# Wavelet Diffusion Neural Operator (WDNO) Reproduction

This repository aims to reproduce the core contributions of the paper "Wavelet Diffusion Neural Operator (WDNO)". The paper introduces a novel PDE simulation and control framework that addresses challenges in handling abrupt changes and generalizing to higher resolutions.

## Core Contributions to be Reproduced:

1.  **Generation in the Wavelet Domain**: Implementing diffusion-based generative modeling within the wavelet domain to effectively handle abrupt changes and long-term dependencies in physical systems.
    *   This involves implementing forward and inverse wavelet transforms. The paper mentions `bior2.4` with `periodization` for 1D Burgers' equation and `bior1.3` with `zero` mode for 2D incompressible fluid/ERA5.
    *   The core diffusion model will operate on wavelet coefficients.

2.  **Multi-Resolution Training**: Developing a training strategy that enables the model to generalize to finer resolutions. This includes:
    *   **Base-Resolution Model (BRM)**: The standard WDNO trained on a base resolution.
    *   **Super-Resolution Model (SRM)**: A conditional diffusion model trained on multi-resolution data pairs to learn patterns between different resolutions, allowing for zero-shot super-resolution.

3.  **Application to Simulation and Control**: Demonstrating WDNO's capability in both PDE simulation and control tasks.
    *   **Simulation**: Learning a mapping from equation parameters to solution functions.
    *   **Control**: Learning optimal control functions, incorporating guidance for objective minimization.

## Repository Structure:

*   `configs/`: Configuration files for different experiments and models.
*   `src/`: Python source code for the WDNO model, wavelet transforms, diffusion processes, and training utilities.
    *   `src/wavelet_transform.py`: Module for wavelet transform functionalities.
    *   `src/diffusion_model.py`: Module for the diffusion model architecture and processes.
    *   `src/wdno.py`: Main WDNO model combining wavelet transforms and diffusion.
    *   `src/multiresolution_training.py`: Utilities for multi-resolution data handling and training.
    *   `src/trainer.py`: Script for orchestrating training and evaluation.
    *   `src/utils.py`: General utility functions.

## Current Status and Next Steps:

This is an initial setup. The next steps will involve fleshing out the `src/` directory with detailed implementations of the wavelet transforms, diffusion model components, and the multi-resolution training logic, following the descriptions in the paper.

## Assumptions and Missing Details:

*   **Specific Network Architectures**: The paper does not explicitly detail the exact neural network architecture used for the `epsilon_theta` prediction in the diffusion model (e.g., U-Net variant, transformer-based). I will assume a standard diffusion model backbone (e.g., U-Net or similar) and abstract this for now.
*   **Hyperparameters**: Specific learning rates, batch sizes, number of diffusion steps, and other training hyperparameters are not fully detailed in the main text. I will use reasonable defaults or infer them where possible.
*   **Dataset Details**: While the paper mentions datasets like 1D Burgers' equation, 1D Advection, 1D Navier-Stokes, 2D Incompressible Fluid, and ERA5, the exact data generation/preprocessing steps beyond wavelet transformation and downsampling are not fully elaborated. I will focus on the modeling aspect assuming suitable data can be provided.
*   **Wavelet Transform Implementation**: The paper mentions `pytorch_wavelets` and `ptwt`. I will need to implement a generic wavelet transform or provide an interface that can utilize such libraries. For this static benchmark, I will focus on the conceptual implementation rather than direct library calls if they are not explicitly within the allowed environment.

