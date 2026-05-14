# Sensitivity-Constrained Fourier Neural Operators (SC-FNO) - Reproduction

This repository aims to reproduce the core contributions of the paper "Sensitivity-Constrained Fourier Neural Operators (SC-FNO) for Forward and Inverse Problems in Parametric Differential Equations".

## Core Contributions and Reproduction Status:

1.  **Sensitivity-Constrained Fourier Neural Operators (SC-FNO)**: The central model architecture, which extends the Fourier Neural Operator by incorporating sensitivity awareness.
    *   **Status**: Conceptualized and outlined. The `src/models/fno.py` and `src/models/sc_fno.py` files will define the FNO and SC-FNO architectures, respectively, highlighting the modifications for sensitivity integration.

2.  **Sensitivity Loss ($L_s$)**: A novel loss function designed to explicitly enforce accurate prediction of output sensitivities with respect to input parameters.
    *   **Status**: Defined. The `src/losses/sensitivity_loss.py` will implement this loss component.

3.  **Physics-Informed Neural Network (PINN) Loss ($L_{Eq}$)**: An optional regularization term based on the governing differential equations.
    *   **Status**: Defined. The `src/losses/pinn_loss.py` will implement this loss component.

4.  **Composite Loss Functions**: Combination of $L_u$, $L_s$, and $L_{Eq}$ for different training configurations (FNO, FNO-PINN, SC-FNO, SC-FNO-PINN).
    *   **Status**: Defined. The training loops will incorporate these combinations, as detailed in the `src/train.py` (conceptual).

5.  **Differentiable Solvers for Data Generation**: The paper highlights the use of differentiable numerical solvers to generate training data, including true solution paths and their sensitivities.
    *   **Status**: Conceptualized. The `src/solvers/differentiable_solver.py` will outline the approach for generating such data. Specific ODE/PDE implementations are outside the scope of this reproduction, focusing on the SC-FNO methodology itself.

## Assumptions and Unresolved Details:

*   **Specific ODE/PDE Implementations**: The paper covers several ODEs and PDEs (Composite Harmonic Oscillator, Duffing Oscillator, Generalized Nonlinear Damped Wave Equation, Forced Burgers’ Equation, Navier-Stokes Equations, Allen-Cahn equation). This reproduction focuses on the general SC-FNO framework rather than implementing each specific differential equation and its corresponding differentiable solver. The `src/solvers` directory will provide a template for how such solvers *would* be integrated.
*   **Hyperparameters**: Hyperparameters provided in the paper (Tables C.7 and C.8) will be referenced for conceptual implementation but not hardcoded for specific experiments.
*   **Training Loop Details**: Pseudocodes are provided in the appendix (Algorithm 1, 2, 3), which will guide the conceptual `src/train.py` file. The exact data loading, batching, and evaluation mechanisms will be high-level representations.
*   **Neural Operator Variations**: The paper mentions generalizability to WNO, MWNO, and DeepONet. This reproduction will primarily focus on FNO as the base operator for SC-FNO, as it is the main focus of the paper.

## Directory Structure:

*   `repo/`: Root directory for the submission.
    *   `README.md`: This file, detailing the reproduction.
    *   `src/`: Contains the source code.
        *   `models/`: Neural network architectures (e.g., FNO, SC-FNO).
        *   `losses/`: Implementation of custom loss functions (e.g., Sensitivity Loss, PINN Loss).
        *   `solvers/`: Conceptual representation of differentiable ODE/PDE solvers for data generation.
        *   `train.py`: Conceptual training script.
    *   `configs/`: Configuration files (e.g., `config.yaml` for model parameters, although not explicitly creating full YAML files for this static reproduction).

## Next Steps:

1.  Define the FNO model architecture.
2.  Define the Sensitivity Loss.
3.  Define the PINN Loss.
4.  Outline the SC-FNO model by integrating FNO and the losses.
5.  Outline the data generation process using differentiable solvers.
6.  Outline the training script.
