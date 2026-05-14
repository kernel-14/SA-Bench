# Pyramidal Flow Matching for Efficient Video Generative Modeling

This repository contains a reproduction of the paper:

> **Pyramidal Flow Matching for Efficient Video Generative Modeling**  
> Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, Zhouchen Lin  
> ICLR 2025

## Overview

This paper introduces **Pyramidal Flow Matching**, a novel video generation algorithm that:

1. **Spatial Pyramid**: Divides the denoising trajectory into K pyramid stages, where only the final stage operates at full resolution. This reduces computational cost by ~1/K.

2. **Temporal Pyramid**: Uses progressively compressed, lower-resolution history as conditions for autoregressive video generation, reducing token count from 119,040 to ≤15,360 for a 10-second video.

3. **Unified Training**: A single Diffusion Transformer (DiT) jointly optimizes all pyramid stages with a unified flow matching objective.

4. **Inference with Renoising**: Corrective Gaussian noise at jump points between pyramid stages ensures continuity of the probability path.

## Repository Structure

```
submission/
├── models/
│   ├── __init__.py
│   ├── pyramid_dit.py          # MM-DiT model (24 layers, 2B params)
│   ├── pyramidal_flow.py       # Core pyramidal flow matching algorithm
│   └── vae_3d.py               # 3D VAE (8x8x8 compression)
├── training/
│   ├── __init__.py
│   └── trainer.py              # Training framework (3-stage procedure)
├── inference/
│   ├── __init__.py
│   └── pipeline.py             # Inference pipeline (T2V, I2V, T2I)
├── utils/
│   ├── __init__.py
│   ├── patch_n_pack.py         # Patch n' Pack for variable-length batches
│   └── position_encoding.py   # Sinusoidal + RoPE position encodings
├── configs/
│   ├── model_config.yaml       # Model architecture config
│   ├── train_stage1_image.yaml # Stage 1: Image training (50k steps)
│   ├── train_stage2_video_low_res.yaml  # Stage 2: Low-res video (200k steps)
│   └── train_stage3_video_high_res.yaml # Stage 3: High-res video (50k steps)
├── scripts/
│   ├── train.py                # Main training script
│   └── generate.py             # Video/image generation script
├── requirements.txt
└── README.md
```

## Core Algorithm

### Pyramidal Flow Matching (Section 3.2)

The key insight is that early denoising steps are very noisy and don't need full resolution. The algorithm divides the trajectory into K stages:

**Training** (Eqs. 9-11):
- For stage k with time window [s_k, e_k]:
  - End: `x_ek = e_k * Down(x1, 2^k) + (1 - e_k) * n`
  - Start: `x_sk = s_k * Up(Down(x1, 2^(k+1))) + (1 - s_k) * n`
  - Shared noise `n` couples endpoints for straighter trajectories
- Loss: `E_{k,t} || v_t(x_t) - (x_ek - x_sk) ||^2`

**Inference** (Algorithm 1):
1. Initialize `x_0 ~ N(0, I)` at lowest resolution
2. For each stage k (lowest to highest resolution):
   - Run ODE integration within stage
   - Apply renoising at jump point (Eq. 15):
     `x_sk = (1 + s_k)/2 * Up(x_{e_{k+1}}) + sqrt(3)*(1 - s_k)/2 * n'`
   - This ensures continuity of the probability path

### Temporal Pyramid (Section 3.3)

For autoregressive video generation, history is compressed:
```
... -> Down(x^{i-2}, 2^{k+1}) -> Down(x^{i-1}, 2^k) -> x_t^i
```

- Older frames are more compressed
- Corruption noise (strength ∈ [0, 1/3]) added during training to mitigate error accumulation
- Reduces tokens from 119,040 to ≤15,360 for 10-second videos

### Model Architecture (Section 3.4, Appendix B)

- **Base**: MM-DiT from SD3 Medium (Esser et al., 2024)
- **Size**: 24 transformer layers, 2B parameters
- **Text conditioning**: T5 (4096-dim) + CLIP (768-dim) encoders (following FLUX.1)
- **Spatial position encoding**: 2D sinusoidal with extrapolation for spatial pyramid
- **Temporal position encoding**: 1D RoPE with interpolation for temporal pyramid
- **Attention**: Blockwise causal attention for autoregressive generation
- **VAE**: 3D causal convolution VAE (MAGVIT-v2 style), 8×8×8 compression

## Training Procedure (Appendix B)

Three-stage training on 128 NVIDIA A100 GPUs:

| Stage | Steps | GPU Hours | Data | Resolution |
|-------|-------|-----------|------|------------|
| 1. Image | 50k | ~1,536 | 180M+ images | Various |
| 2. Low-res Video | 200k | ~11,520 | 12M videos | 384p |
| 3. High-res Video | 50k | ~7,680 | 12M videos | 768p |

**Total**: ~20.7k A100 GPU hours

### Hyperparameters (Table 4)

| Config | Stage 1 | Stage 2 | Stage 3 |
|--------|---------|---------|---------|
| Optimizer | AdamW | AdamW | AdamW |
| β1, β2 | 0.9, 0.999 | 0.9, 0.95 | 0.9, 0.95 |
| LR | 1e-4 | 1e-4 | 5e-5 |
| Batch size | 1536 | 768 | 384 |
| Warmup | 1k steps | 1k steps | 1k steps |
| Weight decay | 1e-4 | 1e-4 | 1e-4 |
| Grad clip | 1.0 | 1.0 | 1.0 |
| Precision | bfloat16 | bfloat16 | bfloat16 |

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
# Stage 1: Image training
python scripts/train.py --config configs/train_stage1_image.yaml

# Stage 2: Low-resolution video training
python scripts/train.py --config configs/train_stage2_video_low_res.yaml \
    --resume outputs/checkpoint_00050000.pt

# Stage 3: High-resolution video training
python scripts/train.py --config configs/train_stage3_video_high_res.yaml \
    --resume outputs/checkpoint_00250000.pt

# Distributed training (128 GPUs as in paper)
torchrun --nproc_per_node=8 --nnodes=16 scripts/train.py \
    --config configs/train_stage1_image.yaml
```

### Generation

```bash
# Text-to-video (5 seconds, 768p, 24fps)
python scripts/generate.py \
    --prompt "A beautiful sunset over the ocean" \
    --output output.mp4 \
    --num_frames 121 \
    --height 768 \
    --width 768 \
    --fps 24 \
    --checkpoint path/to/checkpoint.pt

# Image-to-video
python scripts/generate.py \
    --prompt "The waves crash against the shore" \
    --image input.jpg \
    --output output.mp4 \
    --checkpoint path/to/checkpoint.pt

# Text-to-image
python scripts/generate.py \
    --prompt "A beautiful landscape" \
    --output output.png \
    --mode image \
    --checkpoint path/to/checkpoint.pt
```

## Key Results

### VBench (Table 1)
Our method achieves:
- **Total Score**: 81.72 (best among public-data models)
- **Quality Score**: 84.74 (best overall, including commercial models)
- **Motion Smoothness**: 99.12
- **Dynamic Degree**: 64.63

### EvalCrafter (Table 2)
Competitive performance across visual quality, motion quality, and semantic alignment metrics.

### Efficiency
- **Training**: 20.7k A100 GPU hours for 10-second video generation
- **Inference**: 56 seconds for a 5-second, 384p video clip
- **Token reduction**: ≤15,360 vs 119,040 tokens for 10-second videos

## Implementation Notes and Assumptions

### What's Implemented
1. **Core pyramidal flow matching algorithm** - Full implementation of Eqs. 1-15
2. **Temporal pyramid condition** - History compression with corruption noise
3. **MM-DiT architecture** - Based on SD3 Medium with blockwise causal attention
4. **3D VAE** - MAGVIT-v2 style with 3D causal convolutions
5. **Training framework** - Three-stage procedure with correct hyperparameters
6. **Inference pipeline** - T2V, I2V, T2I with CFG and pyramidal renoising
7. **Position encodings** - Sinusoidal (spatial) + RoPE (temporal) with extrapolation/interpolation
8. **Patch n' Pack** - Variable-length batch packing

### Assumptions and Unresolved Details
1. **VAE architecture**: The paper says "similar to MAGVIT-v2" but doesn't give exact architecture details. We implemented a reasonable approximation with 3D causal convolutions.

2. **Pyramid stage time windows**: The paper uses uniform partitioning [0, 1/3], [1/3, 2/3], [2/3, 1] for K=3 stages. The exact values are not explicitly stated but can be inferred.

3. **Downsampling factors**: The paper uses 2^k downsampling per stage. With K=3 stages, this gives factors of 4x, 2x, 1x for stages 0, 1, 2 respectively.

4. **History compression**: The exact number of history frames and their compression factors are not fully specified. We use a reasonable approximation based on the paper's description.

5. **SD3 Medium initialization**: The paper initializes from SD3 Medium weights. Our implementation builds the architecture from scratch without pre-trained weights.

6. **Video captioning**: The paper uses Video-LLaMA2 for recaptioning. This preprocessing step is not included in our implementation.

7. **Patch n' Pack details**: The exact packing strategy for variable-resolution training is not fully specified. We implement a greedy bin-packing approach.

## Citation

```bibtex
@article{jin2024pyramidal,
  title={Pyramidal Flow Matching for Efficient Video Generative Modeling},
  author={Jin, Yang and Sun, Zhicheng and Li, Ningyuan and Xu, Kun and Jiang, Hao and Zhuang, Nan and Huang, Quzhe and Song, Yang and Mu, Yadong and Lin, Zhouchen},
  journal={arXiv preprint},
  year={2024}
}
```
