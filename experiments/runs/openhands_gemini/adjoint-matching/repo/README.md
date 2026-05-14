
# Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control

This repository contains a faithful reproduction of the paper "Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control" by Domingo-Enrich et al. (FAIR, Meta).

## Project Structure:

- `adjoint_matching/`: Main project directory.
    - `config.py`: Configuration and hyperparameters for models and training.
    - `data.py`: Data loading and preprocessing utilities.
    - `losses/`: Implementations of various loss functions, including Adjoint Matching.
    - `models/`: Neural network architectures for Flow Matching and Diffusion models.
    - `scripts/`: Training, evaluation, and sampling scripts.
    - `utils.py`: Utility functions (e.g., SDE solvers, noise schedules).
    - `requirements.txt`: Python dependencies.
    - `README.md`: Project overview.

## Core Contributions Implemented:

1.  **Unified SDE Framework**: Implementation of a common notation for Flow Matching and Diffusion models.
2.  **Memoryless Noise Schedule**: Integration of the theoretically proven memoryless noise schedule for fine-tuning.
3.  **Stochastic Optimal Control (SOC) Formulation**: Framework for reward fine-tuning as an SOC problem.
4.  **Adjoint Matching Algorithm**: Implementation of the novel Adjoint Matching objective for solving SOC problems.
