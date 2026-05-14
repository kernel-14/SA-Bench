# Towards Universal Neural Operators through Multiphysics Pretraining

Reproduction of the paper: "Towards Universal Neural Operators through Multiphysics Pretraining" by Masliaev et al. (ITMO University).

## Overview

This repository implements the key contributions of the paper:

1. **Adapter-based neural operator framework** for transfer learning across PDE problems
2. **MambaFNO**: FNO with Mamba SSM module inserted after the lifting layer
3. **Perceiver IO-based Neural Operator**: FNO with Perceiver IO blocks for cross-attention
4. **CoDA-NO**: Codomain Attention Neural Operator with function space dot products
5. **LocalAttnFNO**: FNO with local attention (post-lifting)
6. **SwinNO**: Swin Transformer V2-based Neural Operator

## Paper Contributions

The paper proposes a framework for transfer learning in neural operators with three key scenarios:

1. **Out-of-sample parameter values**: Pretrain on one parameter range, fine-tune on different values
2. **Input function set extension**: Pretrain on simpler equations, fine-tune on extended equations with additional terms
3. **Multi-physics pretraining**: Pretrain on multiple PDE types, fine-tune on new physics

The key insight is that the **lifting and projection layers act as adapters** (problem-specific), while the **FNO/attention blocks form the shared backbone**. During fine-tuning, only the adapters are trained, reducing computational cost.

## Architecture

```
Input functions a = {a_1, ..., a_n_in}
        |
   Lifting L (adapter, problem-specific)
        |
   [Optional: Mamba SSM / Local Attention]
        |
   FNO/Perceiver/CoDA blocks (shared backbone)
        |
   Projection P (adapter, problem-specific)
        |
Output functions u = {u_1, ..., u_n_out}
```

## Models

### FNO (Baseline)
Standard Fourier Neural Operator with lifting-FNO blocks-projection architecture.

### MambaFNO
FNO with Mamba SSM module after lifting. The Mamba module acts as a latent preconditioner:
- Encodes long-range temporal/spatial dependencies
- Aligns embeddings with dominant dynamical motifs (transport, diffusion, oscillation)
- Improves stability and transfer efficiency

### Perceiver IO-based NO
Uses Perceiver IO blocks with symmetric cross-attention:
1. Cross-attention: latent queries attend to FNO-processed input (K, V)
2. Self-attention: latent representations attend to each other
3. Cross-attention decode: input queries attend to transformed latent (K, V)

### CoDA-NO
Codomain Attention Neural Operator:
- Attention over feature channels (codomains) rather than spatial positions
- Uses function space dot products (integrated over spatial domain)
- Designed for multiphysics PDE transfer learning

### LocalAttnFNO
FNO with local window attention after lifting (post-lifting).

### SwinNO
Swin Transformer V2-based Neural Operator with alternating window/shifted-window attention.

## Datasets

The paper uses the following PDE datasets:

- **Burgers' equation**: 1D, parametric viscosity
- **Gray-Scott model**: 2D reaction-diffusion, parametric feed/kill rates
- **Navier-Stokes**: 2D incompressible flow, parametric Reynolds number
- **Heat equation**: 1D, with optional convection extension
- **Reaction-diffusion**: 1D, with optional advection extension
- **PDEBench**: Advection, Burgers, Reaction-Diffusion (from Takamoto et al., 2022)

## Metrics

From the paper:
```
NMAE(θ) = (1/|D_test|) * Σ ||G_θ(a) - u||_{1,G} / (max_G u - min_G u + ε)
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run all experiments
```bash
python run_experiments.py --experiment all
```

### Run specific experiment
```bash
# Out-of-sample parameter values (Table 1 in paper)
python run_experiments.py --experiment out_of_sample

# Input function set extension (Table 2 in paper)
python run_experiments.py --experiment input_extension

# Multi-physics pretraining (Table 2 in paper)
python run_experiments.py --experiment multiphysics
```

### With PDEBench data
```bash
# Download PDEBench data first, then:
python run_experiments.py --experiment multiphysics \
    --config configs/default.yaml
# Edit configs/default.yaml to set pdebench_path
```

### Custom configuration
```bash
python run_experiments.py --experiment all --config configs/default.yaml
```

## Repository Structure

```
├── models/
│   ├── fno.py              # Baseline FNO
│   ├── mamba_fno.py        # MambaFNO (FNO + Mamba SSM)
│   ├── perceiver_fno.py    # Perceiver IO-based NO
│   ├── coda_no.py          # Codomain Attention NO
│   ├── local_attn_fno.py   # Local Attention FNO
│   └── swin_no.py          # Swin Transformer V2 NO
├── datasets/
│   ├── burgers.py          # Burgers' equation
│   ├── gray_scott.py       # Gray-Scott reaction-diffusion
│   ├── navier_stokes.py    # Navier-Stokes equations
│   ├── heat_equation.py    # Heat equation (with convection)
│   ├── reaction_diffusion.py # Reaction-diffusion (with advection)
│   ├── advection.py        # Advection equation
│   └── pdebench.py         # PDEBench interface
├── utils/
│   ├── metrics.py          # NMAE, MSE metrics
│   ├── training.py         # Trainer classes
│   └── transfer.py         # Transfer learning utilities
├── experiments/
│   ├── experiment_out_of_sample.py    # Experiment 1
│   ├── experiment_input_extension.py  # Experiment 2
│   └── experiment_multiphysics.py     # Experiment 3
├── configs/
│   └── default.yaml        # Default configuration
├── run_experiments.py      # Main entry point
└── requirements.txt
```

## Results

The paper reports results in two tables:

**Table 1: Out-of-sample parameter values**
| Model | MSE | NMAE (%) |
|-------|-----|----------|
| Mamba FNO (pretr.) | 1.009e-7 | 0.0120 |
| Mamba FNO (scratch) | 1.193e-7 | 0.0213 |
| Perc. (pretr.) | 1.425e-7 | 0.0169 |
| Perc. (scratch) | 1.981e-7 | 0.0219 |
| FNO (scratch) | 1.774e-7 | 0.0204 |
| Swin-v2 (p.+s.) | 4.391e-8 | 0.0092 |
| CoDA-NO (pretr.) | 2.881e-7 | 0.0343 |
| CoDA-NO (scratch) | 4.912e-7 | 0.0712 |

**Table 2: Input extension & Multi-physics**
| Model | MSE | NMAE (%) |
|-------|-----|----------|
| Mamba FNO (pretr.) | 3.91e-6 | 0.0041 |
| Mamba FNO (scratch) | 4.291e-6 | 0.0054 |
| Perc. (pretr.) | 4.107e-6 | 0.0051 |
| Perc. (scratch) | 6.315e-6 | 0.0074 |
| FNO (scratch) | 7.286e-6 | 0.0121 |
| Swin-v2 (p.+s.) | 6.276e-6 | 0.009 |
| CoDA-NO (pretr.) | 1.043e-5 | 0.013 |
| CoDA-NO (scratch) | 1.239e-5 | 0.018 |

## Assumptions and Notes

1. **Data generation**: The paper uses PDEBench for the multi-physics experiment. We provide synthetic data generators as fallback when PDEBench is not available.

2. **Model sizes**: The paper reports approximate parameter counts (~10^6 for FNO, ~10^7 for MambaFNO, ~10^8 for Perceiver/CoDA-NO, ~10^9 for Swin-v2). Our implementations target these scales.

3. **Training details**: The paper doesn't specify all hyperparameters. We use standard choices (Adam optimizer, lr=1e-3, MSE loss).

4. **Mamba SSM**: We implement a simplified version of the Mamba SSM. The paper references Gu & Dao (2023) but doesn't specify exact implementation details.

5. **Multi-physics training**: The paper describes simultaneous training on multiple physics with shared backbone. We implement this by sharing backbone parameters between models for different physics.

6. **Swin-v2**: The paper mentions Swin-v2 as a comparison model. We implement a Swin Transformer V2-based neural operator.

## Citation

```bibtex
@article{masliaev2024universal,
  title={Towards Universal Neural Operators through Multiphysics Pretraining},
  author={Masliaev, Mikhail and Gusarov, Dmitry A. and Markov, Ilya and Hvatov, Alexander},
  year={2024},
  institution={ITMO University}
}
```
