# Generator-Augmented Flows for Consistency Models

Reproduction of **"Improving Consistency Models with Generator-Augmented Flows"** (Issenhuth et al., 2024).

## Overview

This codebase implements consistency models trained with generator-augmented coupling (GC), which reduces the discrepancy between consistency training and consistency distillation by using the model itself as a proxy to construct better data-noise couplings.

## Repository Structure

```
repo/
├── src/
│   ├── layers.py       # UNet building blocks (ResNet, attention, embeddings)
│   ├── network.py      # SongUNet (NCSNpp) architecture
│   ├── model.py        # Consistency model with c_skip/c_out parametrization
│   ├── schedules.py    # Noise schedule, timestep schedule, loss weighting
│   ├── coupling.py     # IC, GC, and batch-OT couplings
│   ├── losses.py       # CT, CD, and GC consistency losses
│   ├── data.py         # Dataset loading and preprocessing
│   └── metrics.py      # FID, KID, IS evaluation
├── configs/
│   ├── cifar10.yaml
│   ├── imagenet32.yaml
│   ├── celeba64.yaml
│   └── lsun_church64.yaml
├── train.py            # Main training script
├── evaluate.py         # Evaluation script
└── requirements.txt
```

## Key Contributions

1. **Discrepancy analysis**: Proves that consistency training and distillation converge to different objectives in the continuous-time limit.
2. **Generator-Augmented Coupling (GC)**: Uses the consistency model to predict endpoints from IC intermediate points, creating a new coupling that reduces the discrepancy.
3. **Joint learning**: Trains a single model on both IC and GC trajectories with mixing parameter μ.

## Training

```bash
# Train iCT-GC on CIFAR-10
python train.py --config configs/cifar10.yaml --coupling gc --mu 0.5

# Train iCT-IC baseline
python train.py --config configs/cifar10.yaml --coupling ic

# Train iCT-OT baseline
python train.py --config configs/cifar10.yaml --coupling ot
```

## Evaluation

```bash
python evaluate.py --config configs/cifar10.yaml --checkpoint path/to/checkpoint.pt
```

## Hyperparameters

Key hyperparameters from the paper:
- `mu`: Joint learning factor (0.5 for iCT, 0.3 for ECT)
- `s0=10`, `s1=1280`: Initial/final number of timesteps
- `rho=7`: Noise schedule exponent
- `sigma_min=0.002`, `sigma_max=80`: Noise range
- Optimizer: Lion with lr ∈ {1e-4, 3e-5}

## Reference

```bibtex
@article{issenhuth2024improving,
  title={Improving Consistency Models with Generator-Augmented Flows},
  author={Issenhuth, Thibaut and Lee, Sangchul and Dos Santos, Ludovic and Franceschi, Jean-Yves and Kim, Chansoo and Rakotomamonjy, Alain},
  year={2024}
}
```
