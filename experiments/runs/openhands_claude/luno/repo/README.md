# LUNO: Linearization Turns Neural Operators into Function-Valued Gaussian Processes

Implementation of the LUNO framework from Magnani, Pförtner, Weber & Hennig (2024).

## Overview

LUNO provides approximate Bayesian uncertainty quantification for trained neural operators by:
1. Uncurrying the neural operator F(a, w)(x) → f((a, x), w)
2. Obtaining a Gaussian weight-space belief w ~ N(μ, Σ) via Laplace approximation or isotropic prior
3. Linearizing f around μ to induce a GP belief over predictions
4. Probabilistically currying back to a function-valued GP F(a) ~ GP(m_a, K_a)

## Structure

```
repo/
├── config.py                  # All hyperparameters and experiment configs
├── models/
│   ├── fno.py                 # Fourier Neural Operator (1D and 2D)
├── luno/
│   ├── weight_uncertainty.py  # Isotropic and Laplace weight-space uncertainty
│   ├── linearization.py       # Linearized GP pushforward (LUNO-*)
│   └── sampling.py            # Sample-based pushforward (Sample-*)
├── baselines/
│   ├── input_perturbations.py # Input perturbation baseline
│   └── ensemble.py            # Deep ensemble baseline
├── data/
│   ├── apebench_data.py       # 1D PDE data (Burgers, HyperDiffusion, KS)
│   └── advection_diffusion.py # 2D advection-diffusion OOD data
├── train.py                   # Training loop (AdamW + cosine decay)
├── evaluate.py                # RMSE, NLL, chi2 metrics
├── calibrate.py               # Grid search calibration of sigma^2
└── experiments/
    ├── run_low_data.py        # Low-data regime experiments (1D PDEs)
    └── run_ood.py             # OOD experiments (2D advection-diffusion)
```

## Experiments

### Low-Data Regime (1D PDEs)
Train FNO on 25 trajectories, evaluate UQ on 250 test pairs:
```bash
python experiments/run_low_data.py --pde burgers
python experiments/run_low_data.py --pde hyper_diffusion
python experiments/run_low_data.py --pde ks_conservative
```

### Out-of-Distribution (2D Advection-Diffusion)
Train FNO on 1000 Base trajectories, evaluate on OOD datasets:
```bash
python experiments/run_ood.py
```

## Methods

| Method | Description |
|--------|-------------|
| Input Perturbations | Perturb inputs with N(0, σ²) noise |
| Ensemble | 10 independently trained FNOs |
| Sample-Iso | Sample from isotropic Gaussian, forward pass |
| LUNO-Iso | Linearized pushforward with isotropic Gaussian |
| Sample-LA | Sample from low-rank Laplace posterior, forward pass |
| LUNO-LA | Linearized pushforward with low-rank Laplace posterior |

## Key Hyperparameters

- FNO: 12 modes, 18 hidden dims, 4 Fourier blocks
- GGN low-rank approximation: rank 500
- Training: AdamW, cosine decay with warmup
- Low-data: 100 epochs; OOD: 1000 epochs
- Calibration: grid search over 500 log-spaced σ² values
