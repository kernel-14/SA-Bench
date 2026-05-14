# Wavelet Diffusion Neural Operator (WDNO)

Implementation of **WDNO** from the paper:
> *Wavelet Diffusion Neural Operator* — Hu et al., 2024

## Overview

WDNO is a PDE simulation and control framework with two key innovations:
1. **Diffusion in the wavelet domain** — handles abrupt changes and long-term dependencies
2. **Multi-resolution training** — enables zero-shot super-resolution via approximate scale invariance

## Repository Structure

```
repo/
├── configs/                  # YAML configs for each experiment
│   ├── burgers_1d.yaml
│   ├── advection_1d.yaml
│   ├── navier_stokes_1d.yaml
│   ├── fluid_2d.yaml
│   └── era5.yaml
├── data/                     # Dataset loading and generation
│   ├── __init__.py
│   ├── burgers.py            # 1D Burgers' equation data generation
│   ├── navier_stokes.py      # 1D compressible NS data loading (PDEBench)
│   ├── advection.py          # 1D advection data loading (PDEBench)
│   ├── fluid_2d.py           # 2D incompressible fluid dataset
│   ├── era5.py               # ERA5 weather dataset
│   └── dataset.py            # Base dataset and multi-resolution dataset
├── models/                   # Neural network components
│   ├── __init__.py
│   ├── unet_1d.py            # 2D-conv U-Net for 1D PDE data
│   ├── unet_3d.py            # 3D-conv U-Net for 2D PDE data
│   └── diffusion.py          # DDPM/DDIM diffusion model
├── wavelet/                  # Wavelet transform utilities
│   ├── __init__.py
│   └── transforms.py
├── baselines/                # Baseline implementations
│   ├── __init__.py
│   ├── fno.py                # Fourier Neural Operator
│   ├── wno.py                # Wavelet Neural Operator
│   ├── mwt.py                # Multiwavelet Neural Operator
│   ├── oformer.py            # Operator Transformer
│   ├── cnn.py                # CNN baseline
│   └── control_baselines.py  # PID, SAC, BC, BPPO, SL
├── wdno.py                   # WDNO: BRM + SRM wrapper
├── train.py                  # Training script (BRM)
├── train_srm.py              # Training script (SRM)
├── evaluate.py               # Evaluation script
├── utils.py                  # Shared utilities
└── requirements.txt
```

## Experiments

| Experiment | Task | Config |
|---|---|---|
| 1D Burgers' equation | Simulation + Control | `configs/burgers_1d.yaml` |
| 1D Advection equation | Simulation | `configs/advection_1d.yaml` |
| 1D Compressible Navier-Stokes | Simulation | `configs/navier_stokes_1d.yaml` |
| 2D Incompressible Fluid | Simulation + Control | `configs/fluid_2d.yaml` |
| ERA5 Weather | Simulation | `configs/era5.yaml` |

## Usage

### Training Base-Resolution Model
```bash
python train.py --config configs/burgers_1d.yaml
```

### Training Super-Resolution Model
```bash
python train_srm.py --config configs/burgers_1d.yaml
```

### Evaluation
```bash
python evaluate.py --config configs/burgers_1d.yaml --checkpoint path/to/checkpoint.pt
```

## Key Hyperparameters

- Wavelet basis: `bior2.4` (1D), `bior1.3` (2D)
- Diffusion steps: K=1000
- DDIM sampling steps: 50 (1D), 100 (2D)
- DDIM η: 1
- Optimizer: Adam, lr=1e-4
- Training steps: 190,000
- LR scheduler: cosine annealing
- Guidance weight λ: 120,000 (1D control), ~1.15e4 (2D control)
