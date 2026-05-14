# MoE-POT: Mixture-of-Experts Operator Transformer for Large-Scale PDE Pre-Training

This repository contains a reproduction of the MoE-POT paper:

> **MoE-POT: Mixture-of-Experts Operator Transformer for Large-Scale PDE Pre-Training**
> Hong Wang, Haiyang Xim, Jie Wang, Xuanze Yang, Fei Zha, Huanshuo Dong, Yan Jiang
> University of Science and Technology of China

## Overview

MoE-POT is a sparse-activated neural operator architecture for pre-training on heterogeneous PDE datasets. It addresses two key challenges:

1. **Dataset heterogeneity**: Different PDE types have conflicting knowledge patterns that hurt mixed training
2. **Scaling bottlenecks**: Dense models incur high inference costs when scaling parameters

The key innovation is a **Mixture-of-Experts (MoE) layer** that:
- Uses 16 routed experts, activating only Top-4 per input (sparse activation)
- Uses 2 shared experts that always activate (for common PDE properties)
- Employs a CNN-based router-gating network that learns to distinguish PDE types

## Architecture

```
Input: u^{<T} ∈ R^{H×W×T×C}
  ↓
Patchification (Conv2D, patch_size=8) + Positional Encoding
  ↓
Temporal Aggregation (MLP + Fourier feature constants)
  ↓
N × MoE-POT Block:
  ├── Multi-head Fourier Layer (frequency-domain MLP per head)
  └── MoE Layer:
      ├── 2 Shared Experts (CNN, always active)
      ├── 16 Routed Experts (CNN, Top-4 selected)
      └── Router-Gating Network (CNN → global avg pool → linear)
  ↓
Output Projection (ConvTranspose2D)
  ↓
Predicted next frame: u^T ∈ R^{H×W×C}
```

## Model Variants

| Model | Attention Dim | MLP Dim | Layers | Heads | Total Params | Activated Params |
|-------|--------------|---------|--------|-------|-------------|-----------------|
| Tiny  | 512          | 512     | 4      | 4     | 30M         | 17M             |
| Small | 1024         | 1024    | 6      | 8     | 166M        | 90M             |
| Medium| 1024         | 2048    | 8      | 8     | 489M        | 288M            |

## Training Details

- **Optimizer**: Adam with lr=1e-3, weight_decay=1e-6, betas=(0.9, 0.9)
- **Schedule**: One-cycle LR over 1000 epochs, 200 warmup epochs
- **Batch size**: 20 (total across 8 RTX 4090 GPUs)
- **Patch size**: P=8
- **Input timesteps**: T=10
- **Spatial resolution**: H=128
- **Noise injection**: ε ~ N(0, ε||u^{<t}||I) during pre-training only

## Pre-training Datasets

| Dataset | PDE Type | Train | Test |
|---------|----------|-------|------|
| FNO-NS (1e-5) | Navier-Stokes | 1000 | 200 |
| FNO-NS (1e-3) | Navier-Stokes | 1000 | 200 |
| PDEBench-CNS (0.1, 0.01) | Compressible NS | 9000 | 200 |
| PDEBench-SWE | Shallow Water | 900 | 60 |
| PDEBench-DR | Diffusion-Reaction | 900 | 60 |
| CFDBench | Incompressible NS | 9000 | 1000 |

## Loss Function

```
L = Σ_{t≤T} ||G_w(u^{<t} + ε) - u^t||_2^2 + Σ_l L_balance^l

L_balance^l = w_bal · CV({Importance_i^l})^2
Importance_i^l = Σ_b w_{i,b}^l(x)
```

where `w_bal = 0.1` and CV is the coefficient of variation.

## Fine-tuning

During fine-tuning:
- The **router-gating network is frozen** to preserve learned expert assignments
- Only expert networks are updated
- 200 epochs with 40 warmup epochs (or 500/100 for downstream tasks)

## Interpretability Analysis

The router-gating network achieves ~98% accuracy in classifying input data by PDE type:

1. Compute average routing distribution Y_i for each dataset
2. For new input X, compute routing distribution I_0
3. Classify as dataset i_0 = argmin_i f(I_0, Y_i) where f is cross-entropy

## Usage

### Pre-training

```bash
# Pre-train MoE-POT-Tiny
python pretrain.py --config configs/pretrain_tiny.yaml --data_dir /path/to/data

# Pre-train MoE-POT-Small
python pretrain.py --config configs/pretrain_small.yaml --data_dir /path/to/data
```

### Fine-tuning

```bash
# Fine-tune on NS (1e-5)
python finetune.py \
    --checkpoint checkpoints/pretrain/pretrain_final.pt \
    --model_size tiny \
    --dataset NS_1e-5 \
    --data_path /path/to/fno_ns_1e-5.npy

# Fine-tune on downstream task
python finetune.py \
    --checkpoint checkpoints/pretrain/pretrain_final.pt \
    --model_size small \
    --dataset NS_1e-4 \
    --data_path /path/to/ns_1e-4.npy \
    --downstream
```

### Evaluation

```bash
# Zero-shot evaluation
python evaluate.py \
    --checkpoint checkpoints/pretrain/pretrain_final.pt \
    --model_size tiny \
    --data_dir /path/to/data \
    --zero_shot

# With interpretability analysis
python evaluate.py \
    --checkpoint checkpoints/pretrain/pretrain_final.pt \
    --model_size tiny \
    --data_dir /path/to/data \
    --interpretability
```

## Data Preparation

The datasets used in the paper come from:
- **FNO datasets**: [FNO GitHub](https://github.com/neuraloperator/neuraloperator)
- **PDEBench**: [PDEBench GitHub](https://github.com/pdebench/PDEBench)
- **CFDBench**: [CFDBench GitHub](https://github.com/luo-yining/CFDBench)

Data should be preprocessed to shape `(N, T, C, H, W)` and saved as `.npy` files.

## Repository Structure

```
├── moe_pot/
│   ├── __init__.py
│   ├── model.py          # Main MoE-POT model architecture
│   ├── fourier_layer.py  # Multi-head Fourier layer
│   ├── moe_layer.py      # MoE layer with shared/routed experts
│   ├── datasets.py       # Dataset classes and data loading
│   ├── trainer.py        # Training and evaluation utilities
│   └── interpretability.py  # Router interpretability analysis
├── configs/
│   ├── pretrain_tiny.yaml
│   ├── pretrain_small.yaml
│   ├── pretrain_medium.yaml
│   ├── finetune.yaml
│   └── downstream.yaml
├── pretrain.py           # Pre-training script
├── finetune.py           # Fine-tuning script
├── evaluate.py           # Evaluation script
├── requirements.txt
└── README.md
```

## Implementation Notes and Assumptions

1. **Fourier Layer**: The paper describes a multi-head Fourier layer where each head applies a 2-layer MLP in the frequency domain. We implement this with separate real/imaginary weight matrices for complex arithmetic.

2. **Expert Architecture**: The paper states experts are CNN-based. We implement each expert as a depthwise-separable CNN: `Conv1x1 → GELU → DepthwiseConv3x3 → GELU → Conv1x1` with a residual connection.

3. **Router Architecture**: The router uses a CNN with global average pooling followed by a linear projection to produce routing logits.

4. **Temporal Aggregation**: The paper describes `z_agg = Σ_t W_t · z_p^t · exp(-i·γ·t)`. We implement this using the real part (cosine modulation) and a learned linear projection over concatenated timesteps.

5. **Noise Injection**: During pre-training, noise ε ~ N(0, ε||u^{<t}||I) is added to inputs. We use ε=0.01 as a default scale factor.

6. **Load Balancing**: The balance loss uses CV² of expert importance scores, weighted by w_bal=0.1.

7. **Fine-tuning**: The router-gating network is frozen during fine-tuning to preserve learned routing behavior.

8. **Data Preprocessing**: Spatial resolution is standardized to 128×128. Channels are padded to max_channels=4 with constant value 1. Irregular geometries use an additional mask channel.

## What Was Reproduced

- ✅ Full MoE-POT architecture (patchification, temporal aggregation, Fourier layers, MoE layers)
- ✅ Three model size variants (Tiny, Small, Medium) with correct parameter configurations
- ✅ Auto-regressive denoising pre-training objective
- ✅ Load balancing loss (CV-based)
- ✅ Balanced dataset sampling strategy
- ✅ Fine-tuning with frozen router
- ✅ L2 relative error evaluation metric
- ✅ Interpretability analysis (dataset classification via routing weights)
- ✅ One-cycle learning rate schedule
- ✅ Data preprocessing pipeline

## What Could Not Be Verified

- Exact parameter counts (30M/166M/489M) depend on implementation details not fully specified
- Exact noise injection scale ε (we use 0.01 as a reasonable default)
- Exact expert architecture details (paper says "CNN" but doesn't specify depth/width)
- Training was not run due to compute constraints
