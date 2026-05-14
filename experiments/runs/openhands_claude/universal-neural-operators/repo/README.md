# Universal Neural Operators through Multiphysics Pretraining

Reproduction of "Towards Universal Neural Operators through Multiphysics Pretraining" (Masliaev et al., 2024).

## Overview

This codebase implements the adapter-based pre-training and fine-tuning framework for neural operators across diverse PDE problems. The core idea: lifting and projection layers act as problem-specific adapters, while the backbone (FNO blocks) is shared and frozen during fine-tuning.

## File Structure

```
repo/
├── layers.py       # Primitive layers: SpectralConv1/2d, FNOBlock1/2d, MambaBlock,
│                   #   LocalAttentionBlock, CodomainAttention, CrossAttentionBlock,
│                   #   SelfAttentionBlock, WindowAttention, SwinBlock
├── modules.py      # Higher-level modules: LiftingLayer, ProjectionLayer,
│                   #   FNOBackbone1/2d, MambaFNOBackbone1/2d, LocalAttnFNOBackbone1/2d,
│                   #   PerceiverIOBackbone, CodaNOBackbone, SwinBackbone
├── model.py        # Complete model architectures:
│                   #   FNO1/2d, MambaFNO1/2d, LocalAttnFNO1/2d, PerceiverNO,
│                   #   CodaNO1/2d, SwinNO, MultiPhysicsModel
│                   #   + build_model() and build_multiphysics_model() factories
├── data.py         # PDEBench dataset loaders:
│                   #   BurgersDataset, GrayScottDataset, NavierStokesDataset,
│                   #   HeatDataset, AdvectionDataset, ReactionDiffusionAdvectionDataset,
│                   #   HeatConvectionDataset, MultiPhysicsDataset
├── train.py        # Training loops: train(), finetune(), pretrain_multiphysics(),
│                   #   run_experiment_from_scratch(), run_experiment_pretrain_finetune()
├── evaluate.py     # Metrics (MSE, NMAE, relative L2), evaluate_model(),
│                   #   benchmark_epoch_time(), print_results_table()
├── config.yaml     # All hyperparameters and experiment configurations
└── requirements.txt
```

## Models

| Model | Params | Description |
|---|---|---|
| FNO | ~10^6 | Baseline Fourier Neural Operator |
| MambaFNO (PL) | ~10^7 | Post-lifting Mamba SSM + FNO blocks |
| LocalAttnFNO (PL) | ~10^7 | Post-lifting local attention + FNO blocks |
| PerceiverNO | ~10^8 | Perceiver IO cross-attention + FNO keys/values |
| CoDA-NO | ~10^8 | Codomain attention (feature-wise) + FNO blocks |
| SwinNO | ~10^9 | Swin-v2 shifted-window transformer |

## Architecture

The operator approximation follows G_θ = P ∘ F ∘ L:

- **L (lifting adapter)**: n_in → hidden_dim, problem-specific
- **F (backbone)**: shared FNO/Mamba/Perceiver blocks, frozen during fine-tuning
- **P (projection adapter)**: hidden_dim → n_out, problem-specific

During pre-training, all parameters are optimized. During fine-tuning, only the new adapter parameters (θ_{P_ft}, θ_{L_ft}) are trained.

## Experiments

Three transfer learning scenarios from the paper:

1. **Out-of-sample parameters** (Table 1): Pre-train on one parameter regime, fine-tune on different parameters. Datasets: Burgers, Gray-Scott, Navier-Stokes.

2. **Input function set extension** (Table 2): Add new input variables. Heat + convection, reaction-diffusion + advection.

3. **Multi-physics transfer** (Table 2): Pre-train on advection + Burgers, fine-tune on reaction-diffusion.

## Metric

NMAE (Normalized Mean Absolute Error, equation 3):

```
NMAE(θ) = (1/|D_test|) Σ ||G_θ(a) - u||_1 / (max_G(u) - min_G(u) + ε)
```

## Data

Download PDEBench datasets from https://darus.uni-stuttgart.de/dataverse/pdebench and update `file_path` entries in `config.yaml`.

Required files:
- `1D/Burgers/Train/1D_Burgers_Sols_Nu*.hdf5`
- `2D/Gray_Scott/Train/2D_Gray_Scott_Sols_GS_type*.hdf5`
- `2D/NS_Incom/Train/NS_Re*_N*_T*.hdf5`
- `2D/Heat/Train/2D_Heat_Sols_alpha*.hdf5`
- `1D/Advection/Train/1D_Advection_Sols_beta*.hdf5`

## Usage

Train from scratch:
```bash
python train.py --config config.yaml --mode scratch
```

Pre-train then fine-tune:
```bash
python train.py --config config.yaml --mode pretrain_finetune
```

Evaluate trained models:
```bash
python evaluate.py --config config.yaml
```

## Dependencies

```
torch>=2.0.0, numpy, scipy, h5py, einops, pyyaml, tqdm, matplotlib, pandas, timm
```
