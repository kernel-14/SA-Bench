# Sensitivity-Constrained Fourier Neural Operators (SC-FNO)

Reproduction of: *"Sensitivity-Constrained Fourier Neural Operators for Forward and Inverse Problems in Parametric Differential Equations"* by Behroozi, Shen & Kifer.

## Overview

This repository reproduces the core contributions of the SC-FNO paper, which introduces a sensitivity-based regularization for Fourier Neural Operators (FNOs). The key innovation is adding a Jacobian loss term `L_s = MSE(∂û/∂p, ∂u/∂p)` that constrains the parameter sensitivities of the neural operator, leading to:

1. **Better parameter inversion** - SC-FNO achieves 5-6× lower relative L² error than FNO in multi-parameter inversion
2. **Robustness to input perturbations** - SC-FNO maintains accuracy when test parameters exceed training ranges
3. **Data efficiency** - SC-FNO with 100 samples matches or exceeds FNO with 500 samples
4. **High-dimensional parameter spaces** - Demonstrated with up to 82 parameters (zoned PDE2)

## Repository Structure

```
models/
├── fno.py              # FNO architecture (SpectralConv1d/2d/3d + FNO)
├── sc_fno.py           # SC-FNO wrapper with sensitivity loss computation
├── other_operators.py  # DeepONet, WNO, MWNO implementations
data/
├── dataset_generation.py  # All 6 ODE/PDE cases with analytical/numerical solutions
├── differentiable_solvers.py  # AD-based ODE/PDE solvers (RK4)
├── finite_difference.py   # 4th-order FD gradient computation
training/
├── train_fno.py        # Training loops for all 4 model variants
├── data_utils.py       # Data preparation & dataloaders
inversion/
├── parameter_inversion.py  # Single & multi-parameter inversion via gradient descent
utils/
├── metrics.py          # R², relative L², sensitivity evaluation
configs/
├── default.yaml        # Default hyperparameters
└── cases/              # Per-case configs (ode1, ode2, pde1-4, pde2_zoned)
main.py                 # Main entry point for experiments
```

## Implemented Components

### Model Architectures (Section 2.1)
- **FNO**: Fourier Neural Operator with learnable spectral convolutions
  - Supports 1D (temporal), 2D (temporal+spatial), 3D (temporal+2D spatial)
  - Configurable Fourier modes, width, and number of layers
  - Identical architecture for both FNO and SC-FNO (only loss differs)

### Loss Functions
- **FNO**: `L = L_u` (MSE on solution paths) - Algorithm 1
- **SC-FNO**: `L = c1·L_u + c2·L_s` (adds Jacobian supervision) - Algorithm 2
- **FNO-PINN**: `L = L_u + c3·L_eq` (adds PDE equation loss) 
- **SC-FNO-PINN**: `L = c1·L_u + c2·L_s + c3·L_eq` - Algorithm 3

### Sensitivity Computation (Section 2.3)
- **Automatic Differentiation (AD)**: Uses PyTorch's autograd through differentiable RK4 solvers
- **Finite Differences (FD)**: 4th-order central differences for non-differentiable models

### Differential Equations (Appendix B, Table B.6)
| Case  | Equation | # Params | Domain |
|-------|----------|----------|--------|
| ODE1  | Composite Harmonic Oscillator | 3 (α,β,γ) | t∈[0,1] |
| ODE2  | Duffing Oscillator | 7 (δ,α,β,γ,ω,ε,ζ) | t∈[0,1] |
| PDE1  | Nonlinear Damped Wave | 5 (c,α,β,γ,ω) | x∈[0,1], t∈[0,1] |
| PDE2  | Forced Burgers' | 4 (α,γ,δ,ω) | x∈[0,1], t∈[0,π] |
| PDE3  | Navier-Stokes (Vorticity) | 2 (α,β) | x,y∈[0,1], t=3 |
| PDE4  | Allen-Cahn | 5 (ε,α,β,c,ω) | x∈[0,1], t∈[0,1] |
| PDE2_Zoned | Burgers' with spatial zones | 82 | x∈[0,1], t∈[0,π] |

### Hyperparameters (Appendix C, Table C.7)
- Fourier modes: 8 per dimension
- Width: 20 channels
- Fourier layers: 4
- Learning rate: 0.001
- Epochs: 500
- Activation: GELU

### Key Experiments (Section 3)

1. **Surrogate Model Quality (Section 3.2)**: Evaluate u prediction and Jacobian accuracy for all 4 model variants
2. **Input Perturbation Robustness (Section 3.2)**: Test model generalization when parameters extend beyond training range
3. **Data Efficiency (Section 3.3)**: Compare model performance with varying training set sizes (100-2000 samples)
4. **High-Dimensional Parameters (Section 3.4)**: Test with 82 zoned parameters
5. **Gradient Method Comparison (Section 3.5)**: AD vs FD gradient quality
6. **Parameter Inversion (Section 3.1)**: Single and multi-parameter recovery via gradient-based optimization

## Usage

```bash
# Train SC-FNO on PDE1
python main.py --case PDE1 --model SC-FNO --n_samples 2000

# Compare all 4 model variants
python main.py --case PDE2 --model all --n_samples 2000

# Run inversion experiment
python main.py --case PDE1 --experiment inversion

# High-dimensional parameter space experiment
python main.py --case PDE2_ZONED --n_samples 500

# Train on ODE1 with analytical solution
python main.py --case ODE1 --model SC-FNO
```

## Assumptions & Missing Details

1. **PDE3 (Navier-Stokes)**: A simplified pseudo-spectral solver is implemented rather than a full CFD solver. The paper uses a numerical solver for training data generation.

2. **PINN Loss (L_eq)**: The PDE residual computation requires case-specific implementations. The framework supports it via `pde_residual_fn` callbacks, but specific PINN loss functions for each equation are not fully implemented.

3. **Zoned PDE2**: The zoned parameter implementation is simplified; the paper's full zonal coupling requires more detailed numerical integration.

4. **Other Neural Operators**: WNO and MWNO implementations use simplified convolution-based approximations rather than full wavelet transforms. The paper reports these as comparison baselines.

5. **Training Data**: All data is generated synthetically using the implemented solvers. The paper reports using torchdiffeq for ODE integration; we use explicit RK4 for simplicity.

6. **Training Time Benchmarks**: Exact epoch times depend on hardware. The paper reports values from a V100 GPU; our implementation is hardware-agnostic.

## Key Equations

### Sensitivity Loss (Eq. in Section 2.1)
```
L_s = (1/M) Σ ||∂û(x_j,t_j;p)/∂p - ∂u(x_j,t_j;p)/∂p||²
```

### Total Loss
```
L_total = c1·L_u + c2·L_s [+ c3·L_eq]
```

## References

- Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations", 2021
- Li et al., "Physics-Informed Neural Operator for Learning Partial Differential Equations", 2023
- Raissi et al., "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems", 2019
