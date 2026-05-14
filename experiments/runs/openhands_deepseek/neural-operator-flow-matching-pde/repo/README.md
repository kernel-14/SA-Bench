# Bridging Neural Operator and Flow Matching for a Generative PDE Foundation Model

Reproduction of the paper: *Bridging Neural Operator and Flow Matching for a Generative PDE Foundation Model* by Zituo Chen and Sili Deng (MIT).

## Architecture Overview

- **P2VAE**: Pretrained Physics Variational Autoencoder that compresses c3p128 physical fields to c16p16 latent grids (12× compression).
- **FMT**: Flow Marching Transformer — a SiT-based autoregressive latent diffusion model with GRU-based diffusion forcing and latent temporal pyramids.

## Key Components

| File | Description |
|------|-------------|
| `config.py` | All hyperparameters and model configurations |
| `modules.py` | P2VAE, SiT backbone, GRU, temporal pyramid, attention layers |
| `data.py` | Dataset loading, flow marching interpolation, preprocessing |
| `train.py` | Two-stage training loop, loss functions, evaluation metrics |

## Reproduction Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Train P2VAE: `python train.py --stage 1 --model_size 16M`
3. Train FMT: `python train.py --stage 2 --model_size B` (uses frozen P2VAE-16M)
4. Evaluate: `python train.py --eval --checkpoint <path>`

## Model Sizes

- P2VAE: 16M (base_dim=64), 87M (base_dim=128)
- FMT: S (embed_dim=256), B (embed_dim=512), L (embed_dim=768)
