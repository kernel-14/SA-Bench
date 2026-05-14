# LUNO: Linearization Turns Neural Operators into Function-Valued Gaussian Processes

This repository reproduces the core contributions of the paper:

> **Linearization Turns Neural Operators into Function-Valued Gaussian Processes**  
> Emilia Magnani*, Marvin Pförtner*, Tobias Weber*, Philipp Hennig  
> ICML 2024

## Overview

LUNO is a framework for approximate Bayesian uncertainty quantification in trained neural operators. The key idea is to:

1. **Uncurry** the neural operator F: A × W → U into f: (A × D_U) × W → R^{d_U'}
2. **Obtain** a Gaussian weight-space belief w ~ N(μ, Σ) via Laplace approximation or isotropic Gaussian
3. **Linearize** f around the MAP weights μ to get a GP belief f ~ GP(m, K)
4. **Probabilistically curry** f back into a function-valued GP F ~ GP(M, K)

This yields a function-valued Gaussian process that, when evaluated at any output point, returns a Gaussian distribution.

## Repository Structure

```
├── src/
│   ├── models/
│   │   └── fno.py              # FNO1d and FNO2d implementations
│   ├── luno/
│   │   ├── luno.py             # Core LUNO framework
│   │   ├── weight_space.py     # Weight-space uncertainty (Iso, Laplace)
│   │   └── metrics.py          # Evaluation metrics (RMSE, NLL, chi2)
│   ├── data/
│   │   ├── pde_data.py         # PDE data generation
│   │   └── dataset.py          # Dataset utilities
│   └── experiments/
│       ├── train_fno.py        # FNO training
│       └── evaluate_uq.py      # UQ evaluation pipeline
├── configs/
│   ├── low_data.yaml           # Low-data experiment config
│   └── ood.yaml                # OOD experiment config
├── run_low_data_experiment.py  # Reproduce Table 1
├── run_ood_experiment.py       # Reproduce Table 2
└── requirements.txt
```

## Key Contributions Implemented

### 1. LUNO Framework (Section 3)

The core LUNO framework is implemented in `src/luno/luno.py`. It implements:

- **Probabilistic Currying** (Theorem 3.2): The equivalence between function-valued GPs and multi-output GPs with augmented input spaces
- **Linearized Laplace** (Section 3.2): Propagating Gaussian weight-space uncertainty through the linearized neural operator
- **Last-Layer LUNO** (Appendix C.1): Efficient implementation exploiting the structure of the last Fourier block

The key formula for the predictive covariance is:
```
K_a(x1, x2) = D_w F(a, w)(x1)|_{w*} Σ D_w F(a, w)(x2)|_{w*}^T
```

For last-layer LUNO with FNOs:
```
K_a(x1, x2) = D_tilde_q(m_{z^{L-1}}(x1)) K_{z^{L-1}}(x1, x2) D_tilde_q(m_{z^{L-1}}(x2))^T
```

### 2. Fourier Neural Operator (Section 2.1)

Implemented in `src/models/fno.py`:
- `FNO1d`: 1D FNO for time-dependent PDEs
- `FNO2d`: 2D FNO for spatial PDEs
- Architecture: 12 modes, 18 hidden channels, 4 Fourier blocks (from Appendix D.2)

### 3. Weight-Space Uncertainty (Appendix D.3)

Implemented in `src/luno/weight_space.py`:
- **IsotropicGaussian** (D.3.3): N(w*, σ²I) with calibrated σ²
- **LaplaceApproximation** (D.3.4): Low-rank GGN approximation with rank 500

### 4. Evaluation Metrics (Appendix D.4)

Implemented in `src/luno/metrics.py`:
- **RMSE**: Root Mean Squared Error
- **NLL**: Marginal Negative Log-Likelihood
- **chi2**: Chi-squared statistic (well-calibrated ≈ 1)

### 5. Experiments (Section 5)

Two main experiments:

**Low-Data Regime** (`run_low_data_experiment.py`):
- Train FNO on 25 Burgers' equation trajectories
- Compare all UQ methods (Table 1)
- LUNO-LA achieves best NLL (-2.0787)

**Out-of-Distribution** (`run_ood_experiment.py`):
- Train FNO on 1000 Advection-Diffusion-Reaction trajectories
- Evaluate on 5 OOD variants (Base, Flip, Pos, Pos-Neg, Pos-Neg-Flip)
- LUNO-LA shows best OOD robustness (Table 2)

## Running the Experiments

### Prerequisites

```bash
pip install jax jaxlib flax optax numpy scipy pyyaml
# Optional for APEBench data generation:
pip install apebench
```

### Low-Data Experiment (Table 1)

```bash
python run_low_data_experiment.py
```

Expected results (Table 1):
| Method | RMSE | chi2 | NLL |
|--------|------|------|-----|
| Input Perturbations | 3.63e-2 | 0.894 | -1.8720 |
| Ensemble | 3.49e-2 | 5.597 | -0.8145 |
| Sample-Iso | 3.72e-2 | 0.977 | -1.9341 |
| LUNO-Iso | 3.62e-2 | 0.864 | -1.9488 |
| Sample-LA | 5.59e-2 | 2.774 | -1.1572 |
| **LUNO-LA** | **3.62e-2** | **1.022** | **-2.0787** |

### OOD Experiment (Table 2)

```bash
python run_ood_experiment.py
```

Expected results (Table 2, NLL):
| Method | Base | Flip | Pos-Neg-Flip |
|--------|------|------|-------------|
| Input Perturbations | -2.586 | 2.573 | 494.935 |
| Ensemble | -5.313 | 3.825 | -1.014 |
| Sample-Iso | -2.921 | 4.071 | 43.362 |
| LUNO-Iso | -2.892 | 3.450 | 37.733 |
| Sample-LA | -2.576 | 4.395 | 27.046 |
| **LUNO-LA** | **-2.934** | **-1.126** | **1.164** |

## Implementation Details

### Data Generation

The paper uses APEBench (Koehler et al., 2024) for 1D PDE data. If APEBench is not available, the code falls back to a synthetic Burgers' equation solver using pseudo-spectral methods with RK4.

For the 2D Advection-Diffusion-Reaction equation, we implement a custom solver using finite differences with RK4.

### GGN Computation

The low-rank GGN approximation uses randomized power iteration to find the top-500 eigenvectors. For the OOD experiment, we use a minibatch of 1000 training samples.

### Calibration

All hyperparameters (σ²) are calibrated on 250 validation samples by minimizing the marginal NLL via grid search over 500 log-spaced points (Appendix D.5).

## Assumptions and Unresolved Details

1. **APEBench integration**: The paper uses APEBench for data generation. Our fallback uses a custom spectral solver that may produce slightly different data statistics.

2. **GGN computation**: The paper uses ViViT (Dangel et al., 2022) for efficient GGN computation. We implement a simpler power iteration approach that may be slower but produces equivalent results.

3. **Padding**: The paper mentions "padding the input by two constant zero grid points to reduce artifacts at the borders" (Appendix D.2). This is implemented in the data preprocessing.

4. **Exact training procedure**: The paper trains for "one epoch = iterating through a single input-output pair per trajectory". We implement this as training on all time steps from each trajectory.

5. **2D FNO input**: For the OOD experiment, the velocity field and reaction term are concatenated to the input as additional channels, with zeros used as placeholders for the base training set.

## Theoretical Framework

The paper's theoretical contributions (Section 3, Appendix A) include:

- **Definition 3.1**: Banach-Valued Gaussian Process
- **Theorem 3.2**: Probabilistic Currying in Banach Spaces
- **Appendix A.3**: Generalized Probabilistic Currying
- **Appendix A.4**: Extension to abstract Banach spaces (for weak PDE solutions)
- **Appendix A.5**: Embedding of operator-valued GPs into the framework

These theoretical results establish that LUNO produces a mathematically well-defined function-valued Gaussian process, not just a heuristic uncertainty estimate.
