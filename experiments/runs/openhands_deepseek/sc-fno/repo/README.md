# SC-FNO: Sensitivity-Constrained Fourier Neural Operators

Reproduction of experiments from:

> Behroozi, A., Shen, C., & Kifer, D. (2024). Sensitivity-Constrained Fourier
> Neural Operators for Forward and Inverse Problems in Parametric Differential
> Equations.

## Codebase Structure

```
repo/
├── sc_fno/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── fno.py              # Fourier Neural Operator
│   │   ├── fourier_layers.py   # Spectral convolution layers
│   │   ├── deeponet.py         # DeepONet
│   │   ├── wno.py              # Wavelet Neural Operator
│   │   └── mwno.py             # Multiwavelet Neural Operator
│   ├── equations/
│   │   ├── __init__.py
│   │   └── solvers.py          # ODE/PDE solvers with Jacobian computation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py           # L_u, L_s, L_eq loss functions
│   │   ├── trainer.py          # Training loops (FNO, SC-FNO, PINN variants)
│   │   └── inversion.py        # Parameter inversion via optimization
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data.py             # Data preprocessing and dataloaders
│   │   └── metrics.py          # R² and relative L² metrics
│   └── configs/
│       ├── __init__.py
│       └── presets.py          # Experiment configurations
├── train.py                    # Main training and evaluation script
└── requirements.txt
```

## Implemented Differential Equations

| Equation | Type | Description |
|----------|------|-------------|
| ODE1 | Composite Harmonic Oscillator | Analytical solution available |
| ODE2 | Duffing Oscillator | Nonlinear oscillator (7 params) |
| PDE1 | Generalized Nonlinear Damped Wave | 1D+time (5 params) |
| PDE2 | Forced Burgers' Equation | 1D+time (4 params) |
| PDE3 | Navier-Stokes (vorticity) | 2D spatial (2 params) |
| PDE4 | Allen-Cahn Equation | 1D+time (5 params, bifurcation) |

## Implemented Models

| Model | Loss components | Paper Algorithm |
|-------|----------------|-----------------|
| FNO | L_u | Algorithm 1 |
| SC-FNO | L_u + L_s | Algorithm 2 |
| SC-FNO-PINN | L_u + L_s + L_eq | Algorithm 3 |
| FNO-PINN | L_u + L_eq | -- |

## Usage

```bash
# Train SC-FNO on PDE1
python train.py --equation pde1 --model sc-fno

# Train FNO baseline
python train.py --equation pde1 --model fno

# Train with PINN regularization
python train.py --equation pde1 --model sc-fno-pinn

# Train and run parameter inversion
python train.py --equation pde1 --model sc-fno --invert

# Evaluate on perturbed parameters
python train.py --equation pde1 --model sc-fno --perturb

# Use fewer training samples (Section 3.3)
python train.py --equation pde1 --model sc-fno --n_samples 100

# Use finite difference solver (Section 3.5)
python train.py --equation ode1 --model sc-fno --solver fd

# Train on zoned Burger's equation (82 parameters)
python train.py --equation pde2_zoned --model sc-fno --n_samples 500
```

## Key Hyperparameters

Per Tables B.6, C.7, and C.8:

- **Fourier modes**: 8 for all dimensions
- **Width**: 20 channels
- **Fourier layers**: 4
- **Learning rate**: 0.001
- **Epochs**: 500
- **Batch size**: 16 (ODEs), 4 (PDEs), 1 (zoned PDE2, PDE4)
- **Loss coefficients**: c1=1.0, c2=1.0, c3=0.1

## Sensitivity Computation

Two methods implemented (Section 2.3):

1. **Automatic Differentiation (AD)**: Uses the differentiable numerical solvers
2. **Finite Difference (FD)**: 4th-order central differences via the analytic solvers
