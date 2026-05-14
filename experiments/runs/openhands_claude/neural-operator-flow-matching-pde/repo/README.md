# Generative PDE Foundation Model: Flow Marching Transformer

Implementation of "Bridging Neural Operator and Flow Matching for a Generative PDE Foundation Model" (Chen & Deng, MIT).

## Overview

This codebase implements two main components:

- **P2VAE** (Pretrained Physics Variational Autoencoder): Compresses 3×128×128 PDE field snapshots to 16×16×16 latent representations (12× compression).
- **FMT** (Flow Marching Transformer): Autoregressive generative model that bridges deterministic neural operators and stochastic flow matching via a bridge parameter `k`.

## Key Algorithms

**Flow Marching**: Interpolation kernel parameterized by bridge parameter `k ∈ [0,1]`:
- `k=1`: deterministic neural operator (linear interpolation)
- `k=0`: stochastic flow matching (Gaussian transport)

**Diffusion Forcing**: GRU-based causal conditioning on past states with independent per-step noise levels.

**Latent Temporal Pyramids**: Coarse-to-fine token compression across 4 frames — `(2×2, 4×4, 8×8, 16×16)` — yielding ~15× efficiency over full-resolution video diffusion.

## File Structure

```
repo/
├── config.py        # All model and training hyperparameters
├── layers.py        # Primitive building blocks (RMSNorm, SwiGLU, attention, AdaLN-Zero)
├── modules.py       # Composite modules (VAE blocks, SiT blocks, GRU forcing, pyramid)
├── model.py         # Top-level P2VAE and FMT models
├── data.py          # Dataset loading for all 12 PDE families
├── train_vae.py     # P2VAE training (Stage 1)
├── train_fmt.py     # FMT training with flow marching (Stage 2)
├── evaluate.py      # L2RE / VRMSE metrics, autoregressive rollout, ensemble generation
├── finetune.py      # Few-shot finetuning on Kolmogorov turbulence
└── requirements.txt
```

## Training

**Stage 1 — P2VAE:**
```bash
python train_vae.py --model_size 16M --batch_size 256 --max_steps 100000
python train_vae.py --model_size 87M --batch_size 256 --max_steps 100000
```

**Stage 2 — FMT (requires frozen P2VAE checkpoint):**
```bash
python train_fmt.py --model_size S --vae_ckpt checkpoints/p2vae_16M.pt --batch_size 256 --max_steps 100000
python train_fmt.py --model_size B --vae_ckpt checkpoints/p2vae_16M.pt --batch_size 256 --max_steps 100000
python train_fmt.py --model_size L --vae_ckpt checkpoints/p2vae_16M.pt --batch_size 256 --max_steps 100000
```

**Few-shot finetuning:**
```bash
python finetune.py --fmt_ckpt checkpoints/fmt_B_42M.pt --vae_ckpt checkpoints/p2vae_16M.pt \
    --data_path /path/to/kolmogorov --n_train 200 --max_steps 5000
```

## Dataset

Training uses ~2.5M trajectories across 12 PDE families (233 GB), all standardized to `c3p128` (3 channels, 128×128 spatial, float16):

| Source | Datasets |
|--------|----------|
| FNO-v | FNO-v3, FNO-v4, FNO-v5 |
| PDEArena | NS, NSCond, SWE |
| PDEBench | CNS-Low, CNS-High, SWE |
| The Well | ActiveMatter, GrayScott, PlanetSWE, RayleighBenard, ShearFlow, TurbRadLayer, ViscoElastic |

## Model Sizes

| Model | Embed Dim | Params |
|-------|-----------|--------|
| P2VAE-16M | base_dim=64 | ~16M |
| P2VAE-87M | base_dim=128 | ~87M |
| FMT-S | 256 | ~6M |
| FMT-B | 512 | ~42M |
| FMT-L | 768 | ~138M |
