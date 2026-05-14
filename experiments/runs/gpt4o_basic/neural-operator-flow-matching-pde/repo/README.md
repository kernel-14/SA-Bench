# Bridging Neural Operator and Flow Matching for a Generative PDE Foundation Model

## Overview
This codebase reproduces the core contributions of the paper:

- Implements **Pretrained Physics Variational Autoencoder (P2VAE)** for efficient compression and reconstruction of PDE snapshots.
- Develops **Flow Marching Transformer (FMT)** for unified generative modeling of PDE dynamics.
- Includes dataset handling utilities for benchmark PDE datasets (e.g., FNO-v, PDEBench).

## Components
- **src/autoencoder.py**: P2VAE autoencoder implementation.
- **src/fmt.py**: Flow Marching Transformer with advanced attention and autoregression.
- **src/dataset.py**: Dataset utilities for loading PDE trajectories.
- **scripts/train_p2vae.py**: Script for training the P2VAE autoencoder.
- **scripts/train_fmt.py**: Script for training the FMT model.

## How to Use
1. Prepare dataset directory with compressed PDE datasets.
2. Run training scripts:
   - Train P2VAE: 
   - Train FMT: 

## Notes
- Loss functions and training hyperparameters follow configurations provided in the paper.
- Placeholder loss used for FMT; detailed flow-matching dynamics need further refinement.

## Future Work
- Fine-tune models on additional benchmark datasets.
- Implement downstream evaluation tasks (rollouts, ensemble generation).
