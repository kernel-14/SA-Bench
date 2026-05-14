# Sensitivity-Constrained Fourier Neural Operators (SC-FNO)

Reproduction of the paper:
**"Sensitivity-Constrained Fourier Neural Operators for Forward and Inverse Problems in Parametric Differential Equations"**
by Abdolmehdi Behroozi, Chaopeng Shen, and Daniel Kifer.

## Overview

This repository implements the SC-FNO framework, which enhances Fourier Neural Operators (FNOs) with a sensitivity loss term that constrains the model's Jacobians (parameter sensitivities) to match those computed from differentiable numerical solvers.

### Key Contributions Implemented

1. **SC-FNO Architecture** (`models/fno.py`, `experiments/sc_fno_experiment.py`):
   - Standard FNO with 1D, 2D, and 3D Fourier layers
   - SCFNO wrapper that enables parameter-aware forward passes
   - Sensitivity loss computation via automatic differentiation

2. **Four Model Configurations**:
   - `fno`: Standard FNO with data loss L_u only
   - `fno_pinn`: FNO with L_u + L_eq (PINN equation loss)
   - `sc_fno`: SC-FNO with L_u + L_s (sensitivity loss)
   - `sc_fno_pinn`: SC-FNO-PINN with L_u + L_s + L_eq

3. **Differential Equations** (`data/generate_data.py`, `experiments/`):
   - ODE1: Composite Harmonic Oscillator (analytical solution + sensitivities)
   - ODE2: Duffing Oscillator (numerical solution + FD sensitivities)
   - PDE1: Generalized Nonlinear Damped Wave Equation
   - PDE2: Forced Burgers' Equation (including zoned/high-dimensional variant with 82 params)
   - PDE3: Navier-Stokes (Stream Function-Vorticity formulation)
   - PDE4: Allen-Cahn Equation (bifurcation test case)

4. **Experiments**:
   - Surrogate model quality (solution paths + Jacobians)
   - Parameter inversion (single and multi-parameter)
   - Robustness to parameter perturbations
   - Performance vs training data volume
   - High-dimensional parameter space (82 parameters)
   - Comparison of AD vs finite difference gradient computation

## Repository Structure

```
sc-fno/
├── models/
│   └── fno.py              # FNO architecture (1D, 2D, 3D)
├── experiments/
│   ├── sc_fno_experiment.py # Core SC-FNO training and evaluation
│   ├── run_ode1.py          # ODE1 experiments
│   ├── run_pde1.py          # PDE1 experiments
│   ├── run_pde2_pde3.py     # PDE2, PDE3, and zoned PDE2 experiments
│   ├── run_pde4.py          # PDE4 experiments
│   ├── run_inversion.py     # Parameter inversion experiments
│   └── trainer.py           # Training utilities
├── data/
│   └── generate_data.py     # Data generation for all equations
├── utils/
│   └── metrics.py           # Metrics and visualization utilities
├── run_all_experiments.py   # Main script to run all experiments
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run all experiments:
```bash
python run_all_experiments.py
```

### Run specific experiment:
```bash
python run_all_experiments.py --experiment ode1 --mode sc_fno --device cuda
python run_all_experiments.py --experiment pde1 --mode all --device cpu
python run_all_experiments.py --experiment pde4 --mode sc_fno
```

### Run individual experiment scripts:
```bash
python experiments/run_ode1.py
python experiments/run_pde1.py
python experiments/run_pde2_pde3.py
python experiments/run_pde4.py
```

## Core Algorithm

The SC-FNO training adds a sensitivity loss term:

```
L_total = c1 * L_u + c2 * L_s [+ c3 * L_eq]
```

Where:
- `L_u = ||u_pred - u_true||_rel` (solution path loss)
- `L_s = (1/M) * sum ||d(u_pred)/dp - du/dp||^2` (sensitivity loss)
- `L_eq` = PDE residual loss (optional, for SC-FNO-PINN)

The sensitivity loss is computed by:
1. Pre-computing true Jacobians `du/dp` using differentiable solvers or finite differences
2. During training, computing predicted Jacobians `d(u_pred)/dp` via automatic differentiation
3. Randomly sampling a subset of spatial-temporal points for efficiency

## Key Implementation Details

### Parameter Embedding
Parameters `p` are broadcast to match spatial-temporal dimensions and concatenated with initial conditions and coordinates before the FNO lifting layer. This allows gradients to flow from outputs back to parameters.

### Efficient Jacobian Computation
Instead of computing Jacobians at all output points, we randomly sample a subset in each epoch. This varies between epochs to eventually cover the full solution space, providing efficient sensitivity supervision.

### Hyperparameters (from Table C.7)
- Fourier modes: 8 (for each dimension)
- Width: 20
- Number of Fourier layers: 4
- Learning rate: 0.001
- Epochs: 500
- Batch sizes: 16 (ODE), 4 (PDE1/2/3), 1 (PDE4, zoned PDE2)

## Assumptions and Unresolved Details

1. **FNO Input Format**: The paper describes parameters as being processed through the "lifting layer" alongside spatial coordinates and initial conditions. We implement this by broadcasting parameters to match spatial-temporal dimensions and concatenating them with other inputs.

2. **Jacobian Computation**: The paper uses both AD (via torchdiffeq) and finite differences. Our implementation uses finite differences for data generation and AD for the sensitivity loss during training.

3. **PINN Loss**: The equation loss `L_eq` is equation-specific. We provide a placeholder implementation that can be extended for each PDE.

4. **Inversion Procedure**: The paper uses backpropagation through the surrogate model to optimize parameters. Our implementation uses Adam optimizer with multiple random restarts.

5. **ODE2 (Duffing)**: The paper mentions 7 parameters including initial conditions epsilon and zeta. We include these as parameters in the sensitivity computation.

## Results

The paper reports that SC-FNO:
- Achieves R² > 0.92 for Jacobians vs R² < 0.78 for FNO (PDE1)
- Has 1/6 the inversion error of FNO for multi-parameter inversion
- Maintains accuracy under 40% parameter perturbation (R² = 0.912 vs 0.529 for FNO)
- Requires fewer training samples to achieve high accuracy
- Handles 82-dimensional parameter spaces effectively

## Citation

```bibtex
@article{behroozi2024sensitivity,
  title={Sensitivity-Constrained Fourier Neural Operators for Forward and Inverse Problems in Parametric Differential Equations},
  author={Behroozi, Abdolmehdi and Shen, Chaopeng and Kifer, Daniel},
  year={2024}
}
```
