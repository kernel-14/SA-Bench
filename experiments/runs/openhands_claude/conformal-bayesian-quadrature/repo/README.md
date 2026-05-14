# Conformal Prediction as Bayesian Quadrature

Reproduction of "Conformal Prediction as Bayesian Quadrature" by Jake C. Snell and Thomas L. Griffiths.

## Overview

This codebase implements the Bayesian Quadrature HPD (BQ-HPD) method for distribution-free uncertainty quantification, along with the CRC and RCPS baselines. The key contribution is treating conformal prediction as a Bayesian quadrature problem, yielding a full posterior distribution over the expected loss rather than a single point estimate.

## Structure

```
repo/
├── methods.py                        # Core algorithms: CRC, RCPS, BQ-HPD, SCP
├── config.py                         # All hyperparameters from the paper
├── utils.py                          # Statistical utilities (Clopper-Pearson CI, etc.)
├── run_experiments.py                # Main entry point
├── data/
│   └── coco_loader.py                # MS-COCO data loading and FNR loss
└── experiments/
    ├── synthetic_binomial.py         # Section 5.1: Synthetic Binomial experiment
    ├── synthetic_heteroskedastic.py  # Section 5.2: Heteroskedastic experiment
    └── coco.py                       # Section 5.3: MS-COCO FNR experiment
```

## Methods

### BQ-HPD (Proposed Method)
The Bayesian Quadrature HPD decision rule (Equation 31):

```
lambda_hpd^beta = inf{lambda : Pr(L+ <= alpha | ell_{1:n}) >= beta}
```

where `L+` is the random variable from Theorem 4.3:

```
U_1, ..., U_{n+1} ~ Dir(1, ..., 1),  L+ = sum_i U_i * ell_(i)
```

with `ell_(1) <= ... <= ell_(n)` the sorted calibration losses and `ell_(n+1) = B`.

`Pr(L+ <= alpha)` is estimated via Monte Carlo with 1000 Dirichlet samples.

### CRC (Baseline)
Conformal Risk Control (Angelopoulos et al., 2024), Equation 15:

```
lambda_crc = inf{lambda : (1/(n+1)) * (sum_i ell(z_i, lambda) + B) <= alpha}
```

Equivalent to BQ-HPD with `beta = 0.5` (taking the expected value of `L+`).

### RCPS (Baseline)
Risk-Controlling Prediction Sets with Hoeffding UCB (Bates et al., 2021):

```
lambda_rcps = inf{lambda : R_hat_n(lambda) + B * sqrt(log(1/delta) / (2n)) <= alpha}
```

where `delta = 1 - beta`.

## Experiments

### Section 5.1: Synthetic Binomial Data
- `n=10`, `K=4`, `alpha=0.4`, `M=10,000` trials
- Loss: `ell(z_i, lambda) = (1/K) * sum_k 1{V_ik > lambda}`, `V_ik ~ Uniform(0,1)`
- True risk = `1 - lambda`; threshold exceeded when `lambda < 0.6`
- Produces Table 1, Figure 3, Figure 4

### Section 5.2: Synthetic Heteroskedastic Data
- `n=200`, `alpha=0.1`, `beta=0.95`, `M=10,000` trials
- `X ~ Uniform(0,4)`, `Y|X ~ N(0, X^2)`, prediction intervals `[-lambda, lambda]`
- Produces Table 2

### Section 5.3: MS-COCO False Negative Rate
- 1000 calibration / 3952 test examples per split, `M=10,000` trials
- Multilabel classification FNR control
- Requires pre-computed scores (auto-downloaded from CRC paper's repository)
- Produces Table 3

## Usage

```bash
pip install -r requirements.txt

# Run all experiments
python run_experiments.py

# Run a specific experiment
python run_experiments.py --exp binomial
python run_experiments.py --exp heteroskedastic
python run_experiments.py --exp coco

# Override number of trials (for quick testing)
python run_experiments.py --exp binomial --M 100

# Custom output directory
python run_experiments.py --output_dir my_results/
```

Results are saved to `results/` (or the specified output directory) as `.npz` files and figures as `.pdf`/`.png`.

## MS-COCO Data

The COCO experiment requires pre-computed softmax scores from a ResNet-101 model. The data is automatically downloaded from the [CRC paper's repository](https://github.com/aangelopoulos/conformal-risk-control) on first run, or can be placed manually at `data/coco_scores.npz`.

## Reference

```bibtex
@inproceedings{snell2024conformal,
  title={Conformal Prediction as Bayesian Quadrature},
  author={Snell, Jake C. and Griffiths, Thomas L.},
  year={2024}
}
```
