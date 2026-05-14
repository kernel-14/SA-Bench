# Wavelet Diffusion Neural Operator (WDNO) - Reproduction

This repository reproduces the core contributions of the paper:
**"Wavelet Diffusion Neural Operator (WDNO)"** by Peiyan Hu et al.

## Overview

WDNO is a novel PDE simulation and control framework with two key innovations:
1. **Generation in the wavelet domain**: Performs diffusion-based generative modeling in the wavelet domain to handle abrupt changes and long-term dependencies.
2. **Multi-resolution training**: Enables zero-shot super-resolution by training on multi-resolution datasets based on approximate scale invariance.

## Repository Structure

```
submission/
├── models/
│   ├── unet_1d.py          # 1D U-Net denoising model
│   ├── unet_2d.py          # 2D/3D U-Net denoising model
│   ├── diffusion.py        # DDPM/DDIM diffusion model
│   └── wdno.py             # Main WDNO model combining wavelet + diffusion
├── data/
│   ├── burgers_data.py     # 1D Burgers' equation data generation
│   ├── navier_stokes.py    # 1D compressible NS data loading
│   ├── fluid_2d.py         # 2D incompressible fluid data loading
│   └── advection.py        # 1D advection data loading
├── utils/
│   ├── wavelet_utils.py    # Wavelet transform utilities
│   ├── metrics.py          # Evaluation metrics (MSE, MAE, L-inf)
│   └── visualization.py   # Visualization utilities
├── configs/
│   ├── burgers_1d.yaml     # Config for 1D Burgers' equation
│   ├── ns_1d.yaml          # Config for 1D Navier-Stokes
│   └── fluid_2d.yaml       # Config for 2D incompressible fluid
├── experiments/
│   ├── train_1d.py         # Training script for 1D experiments
│   ├── train_2d.py         # Training script for 2D experiments
│   ├── eval_simulation.py  # Evaluation for simulation tasks
│   └── eval_control.py     # Evaluation for control tasks
├── train.py                # Main training entry point
└── evaluate.py             # Main evaluation entry point
```

## Key Components

### 1. Wavelet Transform (utils/wavelet_utils.py)
- 1D, 2D, and 3D discrete wavelet transforms using `pytorch_wavelets` and `ptwt`
- Supports bior2.4 (1D) and bior1.3 (2D) wavelet bases
- Periodization mode for 1D, zero-padding mode for 2D

### 2. Diffusion Model (models/diffusion.py)
- DDPM forward/reverse process
- DDIM sampling for accelerated inference
- Classifier-free guidance for conditional generation
- Classifier-based guidance for control tasks

### 3. U-Net Architecture (models/unet_1d.py, models/unet_2d.py)
- 1D U-Net for 1D PDE experiments (Burgers', advection, NS)
- 3D U-Net for 2D fluid experiments
- ResNet blocks with group normalization
- Multi-head self-attention at bottleneck

### 4. WDNO Framework (models/wdno.py)
- Base-Resolution Model (BRM): Generates wavelet coefficients at training resolution
- Super-Resolution Model (SRM): Generates higher-resolution wavelet coefficients
- Multi-resolution training with downsampled data pairs

## Experiments

### 1D Burgers' Equation
- Simulation: Learn mapping from (u0, f) to u[0,T]
- Control: Find optimal f to minimize control objective I
- Data: 40,000 training trajectories, resolution 81×120

### 1D Advection Equation
- Simulation only (from PDEBench)
- 80 timestep prediction from 1 initial condition

### 1D Compressible Navier-Stokes
- Simulation only (from PDEBench)
- Shock-tube initial conditions with η=ζ=1e-8

### 2D Incompressible Fluid
- Simulation and control
- Resolution: 32×64×64
- Indirect control with 3,584 control variables per timestep

### Zero-Shot Super-Resolution
- 1D: 1, 2, 3 levels of super-resolution (80×120 → 640×960)
- 2D: 1 level (32×64×64 → 32×128×128)

## Assumptions and Missing Details

1. The exact architecture of the 3D U-Net for 2D experiments is partially described; we follow the video diffusion model architecture from Ho et al. (2022).
2. The ERA5 dataset preprocessing details are not fully specified; we use similar preprocessing to the 2D fluid case.
3. The exact data generation code for 2D fluid is not provided; we implement based on the description.
4. The guidance scheduler (cosine) details are inferred from the hyperparameter tables.

## Dependencies

```
torch
pytorch_wavelets
ptwt (pytorch-wavelet-toolbox)
numpy
scipy
matplotlib
h5py
pyyaml
einops
```

## Usage

### Training
```bash
# 1D Burgers' equation simulation
python train.py --config configs/burgers_1d.yaml --task simulation

# 1D Burgers' equation control
python train.py --config configs/burgers_1d.yaml --task control

# 2D incompressible fluid
python train.py --config configs/fluid_2d.yaml --task simulation
```

### Evaluation
```bash
# Evaluate simulation
python evaluate.py --config configs/burgers_1d.yaml --task simulation --checkpoint path/to/checkpoint

# Evaluate control
python evaluate.py --config configs/burgers_1d.yaml --task control --checkpoint path/to/checkpoint

# Zero-shot super-resolution
python evaluate.py --config configs/burgers_1d.yaml --task super_resolution --checkpoint path/to/checkpoint
```
