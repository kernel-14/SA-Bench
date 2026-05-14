# Conformal Prediction as Bayesian Quadrature - Reproduction

This repository reproduces the experiments from the paper:

> **Conformal Prediction as Bayesian Quadrature**  
> Jake C. Snell, Thomas L. Griffiths  
> ICML 2025

## Overview

The paper proposes a Bayesian framework for distribution-free uncertainty quantification that:
1. Recovers existing methods (Split Conformal Prediction, Conformal Risk Control) as special cases
2. Provides a richer characterization via the full distribution of possible losses
3. Offers "data-conditional" guarantees that are more conservative than marginal guarantees

The key contribution is the random variable **L+**, defined as:

```
U_1, ..., U_{n+1} ~ Dir(1, ..., 1)
L+ = sum_{i=1}^{n+1} U_i * ell_{(i)}
```

where `ell_{(1)} <= ... <= ell_{(n)}` are the order statistics of calibration losses and `ell_{(n+1)} = B` (upper bound). The Dirichlet distribution of quantile spacings (Lemma 4.2) ensures that `L+` stochastically dominates the posterior expected loss.

The decision rule is:
```
lambda_hpd^beta = inf{lambda : Pr(L+ <= alpha | ell_{1:n}) >= beta}
```

## Repository Structure

```
.
├── methods.py                    # Core algorithms (SCP, CRC, RCPS, BQ)
├── experiment_binomial.py        # Section 5.1: Synthetic Binomial Data
├── experiment_heteroskedastic.py # Section 5.2: Synthetic Heteroskedastic Data
├── experiment_coco.py            # Section 5.3: MS-COCO False Negative Rate
├── download_coco_data.py         # Script to download COCO data
├── run_all_experiments.py        # Run all experiments
├── utils.py                      # Utility functions
└── README.md                     # This file
```

## Methods Implemented

### `methods.py`

- **`split_conformal_prediction(scores, alpha)`**: Split Conformal Prediction (SCP)
- **`conformal_risk_control(losses_fn, lambda_grid, alpha, B)`**: Conformal Risk Control (CRC)
- **`rcps_hoeffding(losses_fn, lambda_grid, alpha, delta, B)`**: RCPS with Hoeffding UCB
- **`sample_L_plus(losses, B, n_samples)`**: Sample from L+ distribution (main contribution)
- **`bayesian_quadrature_decision_rule(...)`**: BQ decision rule (lambda_hpd^beta)
- **`compute_expected_L_plus(losses, B)`**: Analytical E[L+] (recovers CRC)

## Experiments

### Experiment 1: Synthetic Binomial Data (Section 5.1)

**Setup:**
- Loss: `ell(z_i, lambda) = (1/K) * sum_{k=1}^K 1{V_ik > lambda}` where `V_ik ~ Uniform(0,1)`
- Parameters: `n=10`, `K=4`, `alpha=0.4`, `beta=0.95`, `M=10,000` trials
- True expected loss: `1 - lambda`
- Risk exceeds alpha when `lambda < 0.6`

**Expected Results (Table 1):**
| Decision Rule | Relative Freq. | 95% CI |
|---|---|---|
| CRC | 21.20% | [20.40%, 22.01%] |
| RCPS | 0.00% | [0.00%, 0.04%] |
| Ours (β=0.95) | 0.03% | [0.01%, 0.09%] |

### Experiment 2: Synthetic Heteroskedastic Data (Section 5.2)

**Setup:**
- `X ~ Uniform[0, 4]`, `Y | X ~ N(0, X^2)`
- Prediction intervals: `[-lambda, lambda]`
- Loss: miscoverage loss
- Parameters: `n=200`, `alpha=0.1`, `beta=0.95`, `M=10,000` trials

**Expected Results (Table 2):**
| Decision Rule | Relative Freq. | 95% CI | Mean PI Length |
|---|---|---|---|
| Split Conformal Prediction / CRC | 46.19% | [45.21%, 47.17%] | 7.99 |
| RCPS | 0.0% | [0.0%, 0.04%] | 14.29 |
| Ours (β=0.95) | 3.42% | [3.07%, 3.80%] | 9.50 |

### Experiment 3: MS-COCO False Negative Rate (Section 5.3)

**Setup:**
- Multilabel classification on MS-COCO
- 1000 calibration examples, 3952 test examples per trial
- Loss: false negative rate (FNR)
- Parameters: `alpha=0.1`, `beta=0.95`, `M=10,000` trials

**Expected Results (Table 3):**
| Method | Relative Freq. | Pred. Set Size |
|---|---|---|
| CRC | 45.05% | 2.92 |
| RCPS | 0.0% | 3.57 |
| Ours (β=0.95) | 5.43% | 3.04 |

## Running the Experiments

### Prerequisites

```bash
pip install numpy scipy matplotlib
```

### Run All Experiments

```bash
python run_all_experiments.py
```

### Run Individual Experiments

```bash
# Experiment 1: Synthetic Binomial
python experiment_binomial.py

# Experiment 2: Synthetic Heteroskedastic
python experiment_heteroskedastic.py

# Experiment 3: MS-COCO (requires data download first)
python download_coco_data.py
python experiment_coco.py
```

### Download COCO Data

The MS-COCO experiment requires pre-computed model predictions. Run:

```bash
python download_coco_data.py
```

This downloads softmax scores and labels from the conformal-risk repository.

## Key Theoretical Results

### Theorem 4.3 (Main Result)

For any `b ∈ (-∞, B]`:
```
inf_π Pr(L ≤ b | ell_{1:n}) ≥ Pr(L+ ≤ b)
```

This means L+ stochastically dominates the posterior expected loss, regardless of the prior.

### Recovering CRC (Section 4.6)

Taking the expected value of L+:
```
E[L+] = (1/(n+1)) * (sum_{i=1}^n ell_i + B)
```

This is exactly the CRC decision criterion.

### Recovering SCP (Section 4.6)

For miscoverage loss with `ell_i = 1 - 1{s_i <= lambda}`:
```
E[L+] = 1 - k/(n+1)
```

Setting `E[L+] <= alpha` gives `k >= (n+1)(1-alpha)`, recovering SCP.

## Assumptions and Limitations

1. **i.i.d. assumption**: Calibration and test data must be i.i.d.
2. **Upper bound B**: Losses must be bounded above by B.
3. **Conservativeness**: The bounds hold for any prior, so they may be conservative.

## Notes on Implementation

- The BQ decision rule uses Monte Carlo simulation with 1000 Dirichlet samples by default.
- The Dirichlet(1,...,1) distribution is sampled via normalized Exponential(1) samples.
- The RCPS Hoeffding bound uses `delta = 1 - beta = 0.05` for the 95% confidence level.
- For the heteroskedastic experiment, true risk is computed via Monte Carlo integration.

## Differences from Paper

- The paper uses 1000 Monte Carlo samples for the BQ decision rule (as stated in Section 5).
- The paper uses the same random seed for reproducibility (we use seed=42).
- The COCO experiment requires downloading pre-computed model predictions.
