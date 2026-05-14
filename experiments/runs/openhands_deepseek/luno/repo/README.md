# LUNO: Linearization Turns Neural Operators into Function-Valued Gaussian Processes

This repository contains a faithful reproduction of the paper:

> *Linearization Turns Neural Operators into Function-Valued Gaussian Processes*
> Emilia Magnani, Marvin Pförtner, Tobias Weber, Philipp Hennig

## Codebase Structure

| File | Description |
|---|---|
| `config.py` | Experiment and model configuration dataclasses |
| `layers.py` | FNO primitive layers (SpectralConv, FourierBlock, Lifting, Projection) |
| `model.py` | FNO1d and FNO2d models with hidden state extraction for LUNO |
| `data.py` | PDE data generation (Burgers', Hyper Diffusion, Kuramoto-Sivashinsky, Advection-Diffusion-Reaction) |
| `train.py` | Training loop with AdamW + cosine decay schedule |
| `uq.py` | Uncertainty quantification: LUNO (Iso/LA), Sample-based, Input Perturbations, Ensemble, Laplace approximation |
| `evaluate.py` | Evaluation metrics (RMSE, chi-squared, NLL) and autoregressive rollout |
| `experiments.py` | Main experiment runner for low-data and OOD experiments |

## Installation

```bash
pip install -r requirements.txt
```

## Reproducing Experiments

### Low-Data Regime (Section 5, Tables 1, 4, 5)

Train FNO on 25 trajectories and evaluate UQ:

```bash
python experiments.py --experiment low_data_burgers --output_dir outputs/burgers
python experiments.py --experiment low_data_hyper_diffusion --output_dir outputs/hyper_diffusion
python experiments.py --experiment low_data_kuramoto_sivashinsky --output_dir outputs/ks
```

### Out-of-Distribution (Section 5, Tables 2, 6-11)

Train FNO on 1000 Base trajectories, evaluate on Flip, Pos, Pos-Neg, Pos-Neg-Flip:

```bash
python experiments.py --experiment ood_advection --output_dir outputs/ood
```

## FNO Architecture (paper specification)

- **Input**: 10 time steps + auxiliary fields (velocity, reaction for 2D)
- **Output**: Next time step
- **Lifting**: Linear layer to 18 hidden dimensions
- **Fourier blocks**: 4 blocks, 12 modes per spatial dimension
- **Projection**: Linear layer to output dimension
- **Activation**: GELU

## UQ Methods Implemented

1. **LUNO-Iso**: Linearized pushforward with isotropic weight covariance Σ = σ² I
2. **LUNO-LA**: Last-layer Laplace approximation with low-rank GGN (rank 500)
3. **Sample-Iso**: Sample-based pushforward with isotropic weight covariance, then moment matching
4. **Sample-LA**: Sample-based pushforward with Laplace posterior, then moment matching
5. **Input Perturbations**: Gaussian noise added to input, predictions averaged (Pathak et al., 2022)
6. **Deep Ensemble**: 10 independently trained FNOs with different initializations

## Key Theoretical Framework

The core insight of LUNO is **probabilistic currying** (Theorem 3.2):

A function-valued Gaussian process F: A → U (where U is a Banach space of functions)
is equivalent to a multi-output Gaussian process f: A × D_U → R^{d'}

This allows linearizing the neural operator's finite-dimensional weight-space uncertainty
and pushing it forward to a structured function-valued GP over the operator's predictions.

### Last-Layer LUNO for FNO (Appendix C.1)

For efficiency, uncertainty is restricted to the last Fourier block parameters.
The resulting function-valued GP is:

```
F(a)(x) = q̃(m_z(x)) + Dq̃(m_z(x)) · (z(x) - m_z(x))
```

where z^{(L-1)} ∼ GP is a multi-output parametric GP induced by the last Fourier layer's
linear structure in R and W.
