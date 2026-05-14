# Replication of Wavelet Diffusion Neural Operator (WDNO)

## Overview
This repository implements key contributions of the paper: *Wavelet Diffusion Neural Operator (WDNO)*.

The WDNO framework addresses challenges in PDE (partial differential equation) simulation and control by:
1. Introducing generative modeling in the wavelet domain.
2. Employing multi-resolution training for zero-shot super-resolution.
3. Validating performance across multiple physical systems, such as 1D Burgers', Navier-Stokes equations, and ERA5.

## Core Components
1. **Wavelet Domain Modeling**:
   - Discrete wavelet transform for signal decomposition.
2. **Multi-resolution Framework**:
   - Data generation and training pipeline for cross-resolution generalization.
3. **Simulation and Control Pipelines**:
   - Implementation of tasks evaluated in the paper (e.g., Burgers' equation, 2D fluid dynamics).

## Structure
- : Core implementation files (models, data preparation, wavelet processing).
- : Experiment scripts and configuration files.
- : Placeholder for datasets.
- : Overview of the project.

## Next Steps
1. Develop wavelet transform methods.
2. Implement base and super-resolution models.
3. Validate using example settings from the paper.
