# Bridging Neural Operator and Flow Matching for a Generative PDE Foundation Model

This repository reproduces the paper "Bridging Neural Operator and Flow Matching for a Generative PDE Foundation Model" by Zituo Chen and Sili Deng (MIT).

## Overview

The paper proposes a generative PDE foundation model that bridges neural operator learning with flow matching. The key components are:

1. **P2VAE** (Pretrained Physics Variational Autoencoder): Compresses PDE field snapshots from c3p128 (3 channels, 128×128) to c16p16 (16 channels, 16×16), achieving 12× compression.

2. **FMT** (Flow Marching Transformer): A conditional flow matching model that learns to transport noisy states toward clean successors, conditioned on past states via a diffusion forcing scheme.

## Key Contributions Implemented

### 1. Flow Marching Algorithm (`models/flow_marching.py`)

The core innovation: a location-scale interpolation kernel that bridges deterministic neural operators (k=1) and stochastic flow matching (k=0):

```
x_t^k = mu_t + sigma_t * z
mu_t = t*x1 + k*(1-t)*x0
sigma_t = (1-t)*(1-k)
z ~ N(0, I)
```

Training objective (numerically stable):
```
L_FM = 0.5 * E[||(1-t)*g(x_t^k, t) - (x1 - x_t^k)||^2]
```

### 2. P2VAE (`models/p2vae.py`)

Based on SD-VAE architecture, adapted for PDE fields:
- **P2VAE-16M**: base_dim=64, ~16M parameters
- **P2VAE-87M**: base_dim=128, ~87M parameters

Training: AdamW (β1=0.9, β2=0.995), cosine LR with 10% warmup, weight decay=1e-4, KL weight β=1e-3, 100k steps.

### 3. Flow Marching Transformer (`models/fmt.py`)

SiT-based transformer with:
- **AdaLN-Zero** conditioning (DiT/SiT style)
- **RMSNorm + SwiGLU** (Llama-2 style)
- **Multi-head self-attention** with head_dim=64
- **Latent temporal pyramids**: Down(y0,8), Down(y1,4), Down(y2,2), y3 → 15× efficiency gain
- **Diffusion forcing RNN** (GRU) for conditional generation

Three variants:
- **FMT-S**: embed_dim=256, depth=12, ~6M params
- **FMT-B**: embed_dim=512, depth=12, ~42M params
- **FMT-L**: embed_dim=768, depth=24, ~138M params

Training: AdamW (β1=0.9, β2=0.95), cosine LR with 10% warmup, weight decay=0.01, 100k steps.

### 4. Diffusion Forcing (`models/diffusion_forcing.py`)

GRU-based RNN that maintains a compressed latent state h_s to condition the flow marching model on past states with different noise levels. This reduces exposure bias during long-term rollout.

### 5. Latent Temporal Pyramids

Efficiency gain calculation from the paper:
```
η = (4×16²)² / ((2²)² + (4²)² + (8²)² + (16²)²) = 15
```

### 6. Downstream Evaluation

- **Few-shot Kolmogorov turbulence adaptation** (`training/finetune_kolmogorov.py`): Finetune on 200 trajectories for 5k steps with end-to-end loss L = L_CFM + λ_VAE * L_VAE (λ_VAE=1), following REPA-E stop-gradient approach.
- **Long-term rollout evaluation** (`evaluation/evaluate.py`): Euler ODE sampler with N=100 steps, evaluated at steps 1, 5, 10, last, and average.

## Repository Structure

```
submission/
├── models/
│   ├── __init__.py
│   ├── p2vae.py              # P2VAE encoder/decoder
│   ├── fmt.py                # Flow Marching Transformer
│   ├── diffusion_forcing.py  # GRU-based diffusion forcing
│   └── flow_marching.py      # Flow marching algorithm & losses
├── data/
│   ├── __init__.py
│   └── pde_dataset.py        # Heterogeneous PDE dataset loader
├── training/
│   ├── __init__.py
│   ├── train_p2vae.py        # P2VAE training script
│   ├── train_fmt.py          # FMT training script
│   └── finetune_kolmogorov.py # Few-shot finetuning
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py            # L2RE and VRMSE metrics
│   └── evaluate.py           # Evaluation pipeline
├── configs/
│   ├── p2vae_16m.yaml
│   ├── p2vae_87m.yaml
│   ├── fmt_small.yaml
│   ├── fmt_base.yaml
│   ├── fmt_large.yaml
│   └── finetune_kolmogorov.yaml
├── requirements.txt
└── README.md
```

## Usage

### Training P2VAE

```bash
python training/train_p2vae.py \
    --data_dir /path/to/pde_data \
    --output_dir ./checkpoints \
    --model_size 16M \
    --batch_size 256 \
    --num_steps 100000
```

### Training FMT

```bash
python training/train_fmt.py \
    --data_dir /path/to/pde_data \
    --output_dir ./checkpoints \
    --vae_checkpoint ./checkpoints/p2vae_16M_final.pt \
    --model_size B \
    --batch_size 256 \
    --num_steps 100000
```

### Few-shot Finetuning on Kolmogorov Turbulence

```bash
python training/finetune_kolmogorov.py \
    --data_dir /path/to/kolmogorov_re222 \
    --output_dir ./checkpoints \
    --vae_checkpoint ./checkpoints/p2vae_16M_final.pt \
    --fmt_checkpoint ./checkpoints/fmt_B_final.pt \
    --num_steps 5000 \
    --lambda_vae 1.0
```

### Evaluation

```bash
python evaluation/evaluate.py \
    --data_dir /path/to/pde_data \
    --vae_checkpoint ./checkpoints/p2vae_16M_final.pt \
    --fmt_checkpoint ./checkpoints/fmt_B_final.pt \
    --datasets PA-NS PB-CNSL PB-CNSH \
    --rollout_steps 14 \
    --ode_steps 100
```

## Dataset

The paper uses a heterogeneous dataset of ~2.5M trajectories across 12 PDE families:

| Dataset | System | Trajectories |
|---------|--------|-------------|
| FNO-v5 | Navier-Stokes vorticity | 15.4k |
| FNO-v4 | Navier-Stokes vorticity | 368k |
| FNO-v3 | Navier-Stokes vorticity | 184k |
| PA-NS | PDEArena Navier-Stokes | 48k |
| PA-NSC | PDEArena NS Conditional | 120k |
| PA-SWE | PDEArena Shallow Water | 470k |
| PB-CNSL | PDEBench CNS Low | 598k |
| PB-CNSH | PDEBench CNS High | 598k |
| PB-SWE | PDEBench Shallow Water | 77.6k |
| W-AM | Well Active Matter | 13.4k |
| W-GS | Well Gray-Scott | 92.2k |
| W-SWE | Well Shallow Water | 96.4k |
| W-RB | Well Rayleigh-Bénard | 266.6k |
| W-SF | Well Shear Flow | 175.6k |
| W-TR | Well Turbulent Radiative | 7k |
| W-VE | Well Visco-Elastic | 5.3k |

All data is compressed to c3p128 (3 channels, 128×128) with float16 precision.

## Assumptions and Unresolved Details

1. **Exact P2VAE architecture**: The paper says "reuse the standard SD-VAE architecture" but doesn't specify exact layer counts. We use the standard SD-VAE configuration with 3 downsampling stages (128→64→32→16).

2. **FMT depth for Large variant**: The paper specifies embed_dim=768 for FMT-L but doesn't explicitly state the depth. We use depth=24 to achieve ~138M parameters.

3. **Temporal pyramid implementation**: The paper describes downsampling factors [8, 4, 2, 1] for the 4 input frames. We implement this using average pooling.

4. **Diffusion forcing RNN details**: The paper uses a GRU with "the same internal dimension as the embedding dimension in SiT" and cross-attention to compress the current state to a single token. We implement this as described.

5. **Inference procedure**: The paper uses Euler ODE sampler with N=100 steps (dt=0.01). For deterministic prediction, k=(1,1,1,1); for generation, k=(1,1,1,k3) with k3<1.

6. **Dataset preprocessing**: The paper describes specific compression ratios for each dataset. We implement flexible loading that handles various formats and resizes to 128×128.

## References

- Chen & Deng (2025): "Bridging Neural Operator and Flow Matching for a Generative PDE Foundation Model"
- Rombach et al. (2022): "High-Resolution Image Synthesis with Latent Diffusion Models" (SD-VAE)
- Ma et al. (2024): "SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers"
- Chen et al. (2024): "Diffusion Forcing: Next-token Prediction Meets Full-sequence Diffusion"
- Jin et al. (2025): "Pyramidal Flow Matching for Efficient Video Generative Modeling"
- Leng et al. (2025): "REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers"
