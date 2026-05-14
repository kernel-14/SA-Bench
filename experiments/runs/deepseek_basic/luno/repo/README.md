# LUNO: Linearization Turns Neural Operators into Function-Valued Gaussian Processes

Reproduction of the paper "Linearization Turns Neural Operators into Function-Valued Gaussian Processes" 
by Emilia Magnani, Marvin Pförtner, Tobias Weber, and Philipp Hennig.

## Overview

LUNO introduces a framework for approximate Bayesian uncertainty quantification in trained neural operators. 
The approach leverages model linearization to push (Gaussian) weight-space uncertainty forward to the neural 
operator's predictions, yielding a function-valued Gaussian process belief.

Key contributions reproduced:
1. **Probabilistic Currying (Theorem 3.2)**: Formal equivalence between function-valued GPs and 
   multi-output GPs with augmented input spaces
2. **LUNO Framework**: Linearized pushforward for neural operators, compatible with various 
   weight-space uncertainty methods (Laplace, isotropic, SWAG, VI)
3. **FNO-Specific Last-Layer LUNO**: Efficient implementation exploiting FNO structure 
   (Section 3.2.1, Appendix C.1)
4. **UQ Methods**: Implementation and comparison of Input Perturbations, Deep Ensembles,
   Sample-*, and LUNO-* approaches
5. **Experiments**: Low-data regime and out-of-distribution evaluation with standard metrics

## Repository Structure

```
luno/
├── __init__.py                 # Package initialization
├── probabilistic_currying.py   # Theorem 3.2: probabilistic currying equivalence
├── linearized_laplace.py       # Appendix B: Linearized Laplace Approximation
├── luno_fno.py                 # Section 3.2.1, App C.1: FNO-specific LUNO
├── weight_space.py             # Weight-space uncertainty models (*-Iso, *-LA, Ensemble)
├── sampling.py                 # Sampling and moment computation utilities
└── evaluation.py               # Section D.4: Metrics (RMSE, NLL, χ²)

data_generation/
├── __init__.py
└── pde_datasets.py             # Section D.1: PDE data generation (Burgers, ADR, etc.)

experiments/
├── __init__.py
├── uncertainty_methods.py      # All UQ method implementations
├── run_low_data.py             # Section 5, Tables 1/4/5: Low-data experiments
└── run_ood.py                  # Section 5, Tables 2/6-11: OOD experiments
```

## Core Components

### 1. Probabilistic Currying (Theorem 3.2)

The foundational theoretical result showing that function-valued Gaussian processes are equivalent to 
multi-output Gaussian processes with augmented input spaces. This is a probabilistic generalization 
of currying from functional programming.

Implemented in `luno/probabilistic_currying.py`:
- `ProbabilisticCurrying` class with `uncurry_neural_operator()` and `curry_to_function_valued_gp()`
- Demonstrates the equivalence F(a)(x) = f(a, x) for F: A → U and f: A × D_U → R^{d'}

### 2. Weight-Space Uncertainty Models

Implemented in `luno/weight_space.py`:

- **IsotropicGaussian** (`*-Iso`): Σ = σ²I — calibrated prior over weight space
- **LowRankLaplace** (`*-LA`): Σ = (n V V^T + σ I)^{-1} — low-rank GGN approximation
- **DeepEnsembleWeightBelief**: Empirical covariance from ensemble members

### 3. Linearized Laplace Approximation (Appendix B)

In `luno/linearized_laplace.py`:
- GGN matrix computation via matrix-free Jacobian-vector products
- Low-rank GGN approximation using randomized SVD
- Full LLA pipeline: prior, linearization, posterior, pushforward

### 4. FNO-Specific LUNO (Section 3.2.1, Appendix C.1)

In `luno/luno_fno.py`:
- `FourierGaussianRandomOperator`: Function-valued GP from FNO last layer
- Feature function decomposition for z^{(L-1)}: φ_{kj}, ψ_{kj}, v_j
- Efficient covariance computation using Fourier feature structure
- Lazy functional sampling

Key formulas implemented:
- z^{(L-1)}_i(x, w_{L-1}) = Σ_j Σ_k Re(R_{k,ij}) φ_{kj}(x) + Im(R_{k,ij}) ψ_{kj}(x) + Σ_j W_{ij} v_j(x)
- F(a)(x) = q̃(m_z(x)) + Dq̃(m_z(x)) (z^{(L-1)}(x) - m_z(x))
- K_a(x_1, x_2) = Dq̃(x_1) K_z(x_1, x_2) Dq̃(x_2)^T

### 5. UQ Methods Comparison (Section 5, Appendix D.3)

All six methods from the paper are implemented in `experiments/uncertainty_methods.py`:
1. **Input Perturbations** — perturbation-based ensemble
2. **Deep Ensemble** — 10 independently trained FNOs
3. **Sample-Iso** — sampling from isotropic Gaussian weight belief
4. **LUNO-Iso** — linearized pushforward with isotropic belief
5. **Sample-LA** — sampling from Laplace-approximated belief
6. **LUNO-LA** — linearized pushforward with Laplace belief

### 6. Evaluation Metrics (Section D.4)

In `luno/evaluation.py`:
- RMSE: sqrt((1/n) Σ (y_i - ŷ_i)²)
- Marginal NLL: -Σ log(N(y_i; ŷ_i, σ²_i))
- χ²-statistic: (1/n) Σ (y_i - ŷ_i)² / σ²_i
- Calibration via grid search over 500 log-spaced points

## Experiments

### Low-Data Regime (Tables 1, 4, 5)
- Train FNO on 25 trajectories
- Architecture: 4 Fourier blocks, 12 modes, 18 hidden channels
- Test on 250 unseen pairs
- LUNO-LA achieves best NLL across all 1D PDE datasets

### Out-of-Distribution (Tables 2, 6-11)
- Train on 1000 Base trajectories (2D Advection-Diffusion)
- OOD variants: Flip, Pos, Pos-Neg, Pos-Neg-Flip
- LUNO-LA outperforms other weight-space methods
- Deep ensembles achieve lowest NLL but have rank-deficient covariance

## Assumptions and Missing Details

1. **Full LUNO**: The implementation focuses on last-layer LUNO (Section 3.2.1). Full LUNO 
   (linearizing all weights) is outlined but not implemented due to the efficiency of the 
   last-layer approach.

2. **GGN Computation**: The low-rank GGN approximation uses randomized SVD with matrix-free 
   matvec products. The paper uses the approach from Dangel et al. (2022) with rank=500.

3. **PDE Solvers**: The data generation uses spectral methods for 1D PDEs and finite 
   differences for 2D. The paper uses APEBench for 1D and a custom solver for 2D.

4. **Training**: The FNO training using AdamW with cosine decay and MSE loss is described 
   but training code is not included as this is a static reproduction.

5. **Non-Gaussian weight beliefs**: The framework supports arbitrary weight beliefs (Appendix A.4) 
   but the implementation focuses on Gaussian beliefs.

## Dependencies

- JAX (for automatic differentiation and JVP/VJP operations)
- NumPy
- Python 3.8+

## References

- Dangel, F., Tatzel, L., and Hennig, P. ViViT: Curvature access through the generalized Gauss-Newton's low-rank structure. TMLR, 2022.
- Immer, A., Korzepa, M., and Bauer, M. Improving predictions of Bayesian neural networks via local linearization. AISTATS, 2021.
- Koehler, F., Niedermayr, S., Westermann, R., and Thuerey, N. APEbench: A benchmark for autoregressive neural emulators of PDEs. NeurIPS, 2024.
- Li, Z., Kovachki, N. B., et al. Fourier neural operator for parametric partial differential equations. ICLR, 2021.
