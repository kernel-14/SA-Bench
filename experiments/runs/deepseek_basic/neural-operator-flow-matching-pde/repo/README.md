# Bridging Neural Operator and Flow Matching for a Generative PDE Foundation Model

Reproduction of the paper "Bridging Neural Operator and Flow Matching for a Generative PDE Foundation Model" by Zituo Chen and Sili Deng (MIT).

## Overview

This repository implements:

1. **P2VAE (Pretrained Physics Variational Autoencoder)** - Compresses PDE field snapshots from c3p128 to c16p16 latent grids (12× compression). Based on SD-VAE architecture with configurable base dimensions (64 for 16M, 128 for 87M).

2. **FMT (Flow Marching Transformer)** - A conditional flow matching model that:
   - Bridges deterministic neural operators (bridge parameter k=1) and stochastic flow matching (k=0)
   - Uses diffusion forcing with GRU to maintain temporal latent state
   - Implements latent temporal pyramids for computational efficiency
   - Uses AdaLN-Zero conditioning (from SiT) with RMSNorm and SwiGLU (from Llama-2)
   - Available in Small (6M), Base (42M), and Large (138M) configurations

3. **Training Pipeline**:
   - Stage 1: Train P2VAE on heterogeneous PDE datasets
   - Stage 2: Freeze P2VAE, train FMT on latent codes

4. **Downstream Tasks**:
   - Few-shot adaptation (REPA-E style)
   - Long-term autoregressive rollout
   - Ensemble generation via bridge parameter k

## Architecture Details

### P2VAE
- Input: c3p128 (3 channels, 128×128 spatial)
- Latent: c16p16 (16 channels, 16×16 spatial, 12× compression)
- Architecture: Standard SD-VAE with configurable base dims
- P2VAE-16M: base_dim=64
- P2VAE-87M: base_dim=128

### FMT
- Architecture: SiT-style Transformer with AdaLN-Zero
- Conditioning: Diffusion forcing with GRU RNN
- Temporal pyramids: [8×, 4×, 2×, 1×] downsampling on 4 consecutive frames
- FMT-S (6M): embed_dim=256, head_dim=64
- FMT-B (42M): embed_dim=512, head_dim=64
- FMT-L (138M): embed_dim=768, head_dim=64
- FlashAttention v2 for multi-head self-attention
- RMSNorm, SwiGLU activation

## Training Details

- P2VAE: AdamW (β1=0.9, β2=0.995), cosine lr with 10% warmup, weight_decay=1e-4
- FMT: AdamW (β1=0.9, β2=0.95), cosine lr with 10% warmup, weight_decay=0.01
- Base lr: 1e-4 for batch_size=256, adjusted for batch/model size
- P2VAE trained 100k steps with β_KL=1e-3
- FMT trained 100k steps with frozen P2VAE

## Dataset

Trained on a heterogeneous collection from FNO-v, PDEArena, PDEBench, and The Well
- 12 distinct PDE families
- ~2.5M trajectories
- Unified format: c3p128, float16, 233GB total

## Key Equations

### Flow Marching Kernel
x_t^k = μ_t + σ_t z
μ_t = t·x_1 + k(1-t)·x_0
σ_t = (1-t)(1-k)
where t,k ~ Uniform(0,1)

### Flow Marching Objective
L_FM = 1/2 E[||(1-t)·g_θ(x_t^k, t) - (x_1 - x_t^k)||²]

### Conditional Flow Marching
L_CFM = 1/2 E_{h_s ~ p_φ} Σ_i [||(1-t_s)·g_θ(x_{s,t_s}^{k_s}, t_s, h_{s-1}) - (x_{s+1} - x_{s,t_s}^{k_s})||²]

## Efficiency

The latent temporal pyramid achieves ~15× efficiency gain over vanilla video diffusion:
η = (4×16²)² / ((2²)² + (4²)² + (8²)² + (16²)²) = 15

## Assumptions and Missing Details

- Some architectural hyperparameters (e.g., exact number of transformer layers) are inferred from model sizes
- Dataset preprocessing scripts are provided but the actual data download must be done separately
- The Kolmogorov turbulence dataset requires separate download
- Training stability details around batch size scaling are implemented as described
