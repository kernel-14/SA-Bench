# Conformal Prediction as Bayesian Quadrature

## Overview
This repository replicates the key contributions of the paper "Conformal Prediction as Bayesian Quadrature" which reinterprets distribution-free uncertainty quantification techniques (Split Conformal Prediction and Conformal Risk Control) through a Bayesian perspective.

### Structure
- **src**: Core implementation of the techniques.
  - `split_conformal.py`: Implements Split Conformal Prediction (SCP).
  - `conformal_risk_control.py`: Implements Conformal Risk Control (CRC).
  - `bayesian_quadrature.py`: General Bayesian extension.
- **data**: Contains synthetic and real-world datasets.
- **experiments**: Scripts for reproducing paper results.

## Reproduction Strategy
Sections replicated:
1. Decision-theoretic formulation.
2. Bayesian Quadrature without dependence on priors.
3. Algorithms for bounding posterior risk.
4. Synthetic and real-world experiments.

### How to Use
1. Install required dependencies (see paper).
2. Run `experiments/scripts` after configuring datasets.
3. Refer to `src` for method implementation.
