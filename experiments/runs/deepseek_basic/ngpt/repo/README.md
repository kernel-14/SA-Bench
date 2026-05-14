# nGPT: Normalized Transformer with Representation Learning on the Hypersphere

Reproduction of the paper "nGPT: Normalized Transformer with Representation Learning on the Hypersphere" by Ilya Loshchilov, Cheng-Ping Hsieh, Simeng Sun & Boris Ginsburg (NVIDIA).

## Overview

This repository implements the normalized Transformer (nGPT) architecture as described in the paper. The key innovation is that all vectors forming embeddings, MLP, attention matrices, and hidden states are unit norm normalized, so the entire computation happens on the surface of a hypersphere.

## Core Contributions Reproduced

### 1. Optimization on the Hypersphere (`model.py`)
- All matrices (E_input, E_output, W_q, W_k, W_v, W_o, W_u, W_v, W_o_mlp) are normalized along their embedding dimension
- Matrix-vector multiplications become dot products representing cosine similarities bounded in [-1, 1]
- Weight decay is unnecessary (and removed)

### 2. Normalized Transformer Architecture (`model.py`)
- **NormalizedLinear**: Linear layers that normalize weight vectors along embedding dim during forward pass
- **NormalizedEmbedding**: Embedding layers with normalized vectors
- **AttentionBlock**: 
  - Normalized W_q, W_k, W_v, W_o matrices
  - QK normalization with learnable per-dimension scaling factors s_qk (equations 15-16)
  - Softmax scaling factor changed from 1/√d_k to √d_k
  - RoPE applied to normalized q and k
- **MLPBlock**:
  - Normalized W_u, W_v, W_o_mlp matrices
  - SwiGLU activation with learnable scaling factors s_u, s_v (equations 20-21)
  - v scaled by √d_model to benefit from SiLU non-linearity
- **nGPTBlock**: Combines attention and MLP with eigen learning rates
  - h ← Norm(h + α_A * (h_A - h))
  - h ← Norm(h + α_M * (h_M - h))

### 3. Eigen Learning Rates (`model.py`, Section 2.5)
- α_A and α_M: per-dimension learnable parameters controlling contribution of each block
- Implemented via `ScaledParameter` with init/scale decomposition
- α_init = 0.05 (~1/n_layers), α_scale = 1/√d_model
- Enables interpretation as diagonal variable-metric matrix

### 4. Training Procedure (`train.py`)
- Adam without weight decay (AdamW with weight_decay=0.0)
- Cosine annealing learning rate schedule
- No warmup for nGPT (vs 2000 steps for baseline GPT)
- Weight normalization after each optimizer step (`model.normalize_weights()`)
- Baseline GPT implementation for comparison

### 5. Riemannian Optimization View (`slerp.py`)
- SLERP (Spherical Linear Interpolation) implementation
- LERP approximation (default, used in paper)
- Riemannian update with tangent space projection (Appendix A.4)
- Normalization as retraction step

### 6. Analysis Tools (`analysis.py`)
- Embedding properties: norm distributions, eigenvalue distributions, pairwise dot products
- Condition numbers of attention and MLP matrices per layer
- Eigen learning rates and scaling factors analysis across layers
- Singular value distributions per head
- GPT vs nGPT comparison utilities

### 7. Ablation Study Support (`config.py`)
- Configurations for various ablation experiments described in Appendix A.9
- Different s_init/s_scale combinations
- Scalar vs per-element scaling factors
- QK normalization removal
- SLERP vs LERP comparison

## Model Configurations

Matching the paper's Table 2:

| Parameter | 0.5B Model | 1B Model |
|-----------|-----------|---------|
| Layers | 24 | 36 |
| d_model | 1024 | 1280 |
| Heads | 16 | 20 |
| d_k | 64 | 64 |
| d_mlp | 4096 | 5120 |
| Parameters (nGPT) | ~468.4M | ~1026.1M |

## Key Hyperparameters

From Section 2.6 and Appendix A.6:

| Parameter | GPT | nGPT |
|-----------|-----|------|
| Optimizer | AdamW | Adam (wd=0) |
| Weight Decay | 0.1 | 0.0 |
| Warmup Steps | 2000 | 0 |
| LR Schedule | Cosine | Cosine |
| Init std (matrices) | 0.02 | 1/√d_model |
| α_A_init, α_M_init | N/A | 0.05 |
| α_scale | N/A | 1/√d_model |
| s_qk_init, s_u_init, s_v_init, s_z_init | N/A | 1.0 |
| s_qk_scale, s_z_scale | N/A | 1/√d_model |
| s_u_scale, s_v_scale | N/A | 1.0 |
| Softmax scale | 1/√d_k | √d_k |
| RoPE base | 10000 | 10000 |

## Files

- `model.py` - Core nGPT and baseline GPT model implementations
- `train.py` - Training loop with Adam/AdamW, cosine schedule, weight normalization
- `slerp.py` - SLERP, LERP, Riemannian update implementations
- `analysis.py` - Model analysis and inspection tools (Section 3.2)
- `config.py` - Configuration, hyperparameters, ablation setups
- `data_utils.py` - Data loading for OpenWebText with LLaMA-2 tokenizer
- `test_model.py` - Unit tests for model components

## Usage Example

```python
from model import create_ngpt_model
import torch

# Create 0.5B nGPT model
model = create_ngpt_model('0.5B', vocab_size=32000, max_seq_len=4096)

# Forward pass
batch = torch.randint(0, 32000, (2, 1024))
targets = torch.randint(0, 32000, (2, 1024))
logits, loss = model(batch, targets)

# Training step
optimizer = model.configure_optimizers(learning_rate=1e-3)
optimizer.zero_grad()
loss.backward()
optimizer.step()
model.normalize_weights()  # Critical: normalize after each step
```

## Assumptions and Missing Details

1. **Data**: The paper uses OpenWebText with LLaMA-2 tokenizer (32k tokens). Our data utilities support this but require downloading the dataset separately.

2. **Distributed Training**: The paper uses 64 A100 GPUs across 8 nodes. Our training code is single-GPU but can be extended with distributed data parallelism.

3. **Mixed Precision**: The paper uses bfloat16. Our code supports this but defaults to float32 for simplicity.

4. **Learning Rate Selection**: The paper tunes initial LR per configuration. Our defaults are reasonable starting points.

5. **Position IDs**: The paper likely uses position IDs properly. We use simple sequential positions.

6. **QK Normalization**: The paper's equations 15-16 normalize q and k then scale by s_qk. Our implementation follows this exactly.

7. **ScaledParameter**: The init/scale decomposition for effective Adam learning rate control is implemented as described in Section 2.5.

8. **Normalization timing**: The paper mentions normalizing "after each training step (and, optionally, during the forward pass)". We normalize during forward pass (in NormalizedLinear/NormalizedEmbedding) AND explicitly after each optimizer step via `normalize_weights()`.

## References

- Paper: "nGPT: Normalized Transformer with Representation Learning on the Hypersphere"
- Original implementation: https://github.com/NVIDIA/ngpt (not consulted per reproduction rules)
