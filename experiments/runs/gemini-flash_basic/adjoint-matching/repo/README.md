# Adjoint Matching: Fine-tuning Flow and Diffusion Models with Memoryless SOC - Reproduction

This repository contains a reproduction attempt of the paper "Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control". The goal is to replicate the core contributions of the paper, specifically the mathematical formulations and the Adjoint Matching algorithm for fine-tuning.

## Repository Structure

- `model.py`: Contains the mathematical definitions for Flow Matching and DDPM models, including the `alpha_t`, `beta_t`, `kappa_t`, `eta_t` coefficients, and the `memoryless_sigma_t` function as derived from the paper's equations and Table 1.
- `networks.py`: Provides abstract base classes for `VectorField` (representing `v` for Flow Matching or `epsilon` for DDPM) and `RewardModel`. It also includes simple `nn.Module` implementations (`SimpleVectorField`, `SimpleRewardModel`) as placeholders for actual neural network architectures that would be used in a real training scenario.
- `trainer.py`: Implements the `AdjointMatchingTrainer` class, which encapsulates the core Adjoint Matching algorithm. This includes:
    - `SDEIntegrator`: For forward simulation of stochastic differential equations (SDEs) to generate trajectories `X_t` using the Euler-Maruyama method and the memoryless noise schedule.
    - `_solve_lean_adjoint`: For backward simulation of the lean adjoint ordinary differential equation (ODE) to compute `a_t`.
    - `forward`: Calculates the Adjoint Matching loss, combining the forward SDE simulation, adjoint ODE solution, and the objective function as defined in the paper (Eq. 355 and Proposition 2).

## Core Contributions Replicated

1.  **Memoryless Noise Schedule Implementation**: The `model.py` faithfully implements the calculation of `kappa_t`, `eta_t`, and the critical `memoryless_sigma_t = sqrt(2 * eta_t)` for both Flow Matching and DDPM models, as specified in Table 1 and Proposition 1 (Lines 211, 229).
2.  **Unified SDE Framework**: The `model.py` and `trainer.py` leverage the unified SDE formulation (Eq. 10-11) to handle both Flow Matching and DDPM models within a consistent framework.
3.  **Adjoint Matching Algorithm**: The `trainer.py` implements the Adjoint Matching algorithm (Section 5.2), including:
    *   **Forward SDE Integration**: Simulating `X_t` using the Euler-Maruyama method and the finetuned vector field with the memoryless noise schedule (Eq. 341).
    *   **Terminal Gradient Calculation**: Computing `-nabla_X1 r(X_1)` for the adjoint ODE initialization.
    *   **Lean Adjoint ODE Integration**: Solving the backward ODE for `a_t` using the base model's drift (Eq. 347), carefully handling `stopgrad` operations as implied by the paper for numerical stability and correct gradient flow.
    *   **Adjoint Matching Loss**: Calculating the objective function (Eq. 355, Proposition 2) based on the `control_u` and the computed adjoint states.

## Assumptions and Missing Details

Due to the nature of the task (static code reproduction without execution or external resources beyond the paper itself), certain assumptions and simplifications were made:

1.  **Neural Network Architectures**: The `networks.py` provides `SimpleVectorField` and `SimpleRewardModel` using basic MLPs. In a full reproduction, these would be replaced with more sophisticated architectures (e.g., U-Nets for image data, Transformers for text) as typically used in Flow Matching and Diffusion models, and fine-tuned reward models.
2.  **`alpha_bar_t` Schedule for DDPM**: The paper refers to `alpha_bar_t` schedules for DDPM without explicitly defining a concrete function in the main text. I have provided abstract methods in `DDPMModel` for `alpha_bar_t_schedule` and `dot_alpha_bar_t_schedule`, and included two example concrete implementations (`LinearAlphaBarDDPMModel`, `CosineAlphaBarDDPMModel`). The Cosine schedule is a common choice in diffusion models and is a more robust alternative to a simple linear schedule, especially to avoid division by zero at $t=0$ or $t=1$. A full reproduction would use the exact schedule specified in the original DDPM paper or the specific variant being fine-tuned.
3.  **Numerical Stability at $t=0$ and $t=1$**: For Flow Matching models, `kappa_t` and `eta_t` can approach infinity at $t=0$. Similarly, for DDPM, `alpha_bar_t` schedules might approach 0 or 1 at the boundaries, leading to division by zero or large values in `kappa_t`, `eta_t`, or the control `u` coefficient. While some basic `if` conditions (`if t==0`, `if alpha_bar_t == 0 or alpha_bar_t == 1`) are in place to prevent immediate errors, a production-grade implementation would require more sophisticated numerical stability techniques (e.g., small epsilon offsets, re-parameterizations, or careful design of the schedules themselves).
4.  **`stopgrad` Implementation**: The `stopgrad` operations (using `.detach()`) are applied where indicated by the paper to ensure correct gradient flow for the Adjoint Matching objective. Specifically, the forward SDE trajectory `X_t` and the adjoint trajectory `a_t` are computed without gradients, but their values are used to compute gradients for the `finetuned_vector_field`.
5.  **Stochasticity in SDE Integration**: The Euler-Maruyama integration includes a stochastic noise term. The number of SDE steps (`num_sde_steps`) and adjoint steps (`num_adjoint_steps`) are configurable but are set to a default of 100. In practice, more steps might be required for accurate simulation.
6.  **Optimizer and Training Loop**: The provided code focuses on the mathematical and algorithmic core. A complete system would require an outer training loop, an optimizer (e.g., Adam), data loading, and evaluation metrics. These are outside the scope of this static reproduction task.
7.  **Batching and Device Management**: The code is written with `torch.Tensor` and implicitly assumes batching. Device management (CPU/GPU) would be handled in a complete training script.

This reproduction provides a solid foundation for understanding and implementing the Adjoint Matching algorithm based on the paper's theoretical contributions.
