# Conformal Prediction as Bayesian Quadrature

Reproduction of the paper: "Conformal Prediction as Bayesian Quadrature" by Jake C. Snell & Thomas L. Griffiths.

## Overview

This codebase reproduces the three experiments from the paper:

1. **Synthetic Binomial Data** (Section 5.1) — Demonstrates that CRC marginal guarantees allow many individual trials to exceed risk thresholds, while the Bayesian approach controls this.
2. **Synthetic Heteroskedastic Data** (Section 5.2) — Compares prediction interval sizes and risk control.
3. **MS-COCO False Negative Rate** (Section 5.3) — Multilabel classification with false negative rate control.

## Code Structure

```
repo/
├── config.py              # All hyperparameters from the paper
├── bayesian_quadrature.py # Core L⁺ algorithm and HPD decision rule
├── conformal_methods.py   # CRC and SCP baselines
├── rcps.py                # RCPS+Hoeffding baseline
├── data.py                # Data generation (binomial, heteroskedastic, MS-COCO)
├── experiments.py         # Experiment runners for all three experiments
├── utils.py               # Evaluation metrics and statistics
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Key Algorithm

The core contribution is a Bayesian reinterpretation of conformal prediction via Bayesian quadrature:

1. Given n calibration losses ℓ₁, ..., ℓₙ, sort them: ℓ_{(1)} ≤ ... ≤ ℓ_{(n)}
2. Extend with loss bound B: ℓ_{(n+1)} = B
3. L⁺ = Σᵢ Uᵢ · ℓ_{(i)} where U ~ Dirichlet(1, ..., 1)
4. For any confidence β: b_β^* = inf{b : Pr(L⁺ ≤ b) ≥ β}
5. Select λ via: inf{λ : Pr(L⁺(λ) ≤ α | data) ≥ β}

## Baselines

- **CRC** (Conformal Risk Control): λ = inf{λ : (1/(n+1))(Σ ℓᵢ + B) ≤ α}
- **RCPS** (Risk-Controlling Prediction Sets) with Hoeffding UCB
- **SCP** (Split Conformal Prediction) for coverage experiments

## Running Experiments

```bash
python experiments.py
```

## Dependencies

- numpy
- scipy
