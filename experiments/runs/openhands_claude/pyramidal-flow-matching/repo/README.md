# Pyramidal Flow Matching for Efficient Video Generative Modeling

Implementation of [Pyramidal Flow Matching for Efficient Video Generative Modeling](https://pyramid-flow.github.io) (Jin et al., 2024).

## Overview

This codebase reproduces the core contributions of the paper:

1. **Spatial Pyramid Flow Matching**: Divides the generation trajectory into K stages operating at progressively higher resolutions. Only the final stage runs at full resolution, reducing computation by ~1/K.

2. **Temporal Pyramid Conditioning**: Compresses history frames at progressively lower resolutions for autoregressive video generation, reducing history tokens by up to 1/4^K.

3. **Unified MM-DiT**: A single Diffusion Transformer (based on SD3 Medium) jointly trained across all pyramid stages with a unified flow matching objective.

4. **Corrective Renoising**: At jump points between pyramid stages, applies a corrective Gaussian noise with blockwise decorrelation structure (γ = -1/3) to maintain probability path continuity.

## Repository Structure

```
repo/
├── config.py                    # All hyperparameters and configurations
├── train.py                     # Main training script (3-stage procedure)
├── train_vae.py                 # 3D VAE training script
├── generate.py                  # Generation script (t2i, t2v, i2v)
├── ablation.py                  # Ablation study experiments
├── requirements.txt
│
├── model/
│   ├── layers.py                # Basic building blocks (norms, embeddings, conv)
│   ├── attention.py             # Joint attention, causal attention, RoPE
│   ├── dit.py                   # MM-DiT architecture (24 layers, 2B params)
│   └── vae.py                   # 3D VAE with 8x8x8 compression (MAGVIT-v2 style)
│
├── pyramid_flow/
│   ├── spatial_pyramid.py       # Spatial pyramid flow matching algorithm
│   ├── temporal_pyramid.py      # Temporal pyramid history conditioning
│   └── scheduler.py             # Inference scheduler with renoising
│
├── data/
│   ├── dataset.py               # Image/video datasets with aspect ratio bucketing
│   └── text_encoder.py          # T5 + CLIP dual text encoder
│
├── training/
│   └── trainer.py               # Training loop with mixed precision
│
└── inference/
    ├── pipeline.py              # End-to-end generation pipeline
    └── evaluation.py            # FID, FVD, VBench, EvalCrafter metrics
```

## Key Algorithms

### Spatial Pyramid Flow (Section 3.2)

The generation trajectory [0,1] is divided into K=3 stages. For stage k with time window [s_k, e_k]:

**Training endpoints** (coupled noise sampling, Eqs. 9-10):
```
x_hat_{e_k} = e_k * Down(x_1, 2^k) + (1-e_k) * n
x_hat_{s_k} = s_k * Up(Down(x_1, 2^{k+1})) + (1-s_k) * n
```

**Training objective** (Eq. 11):
```
L = E[||v_theta(x_t) - (x_hat_{e_k} - x_hat_{s_k})||^2]
```

**Renoising at jump points** (Eq. 15):
```
x_hat_{s_k} = (1+s_k)/2 * Up(x_hat_{e_{k+1}}) + sqrt(3)*(1-s_k)/2 * n'
with e_{k+1} = 2*s_k/(1+s_k)
```

### Temporal Pyramid (Section 3.3)

History frames are compressed at progressively lower resolutions:
```
... -> Down(x^{i-2}, 2^{k+1}) -> Down(x^{i-1}, 2^k) -> x_hat_t^i
```
During training, history is corrupted with noise strength ∈ [0, 1/3].

## Training

Three-stage training procedure on 128 NVIDIA A100 GPUs:

| Stage | Data | Steps | LR | Batch | GPU Hours |
|-------|------|-------|-----|-------|-----------|
| 1 | Images (180M) | 50k | 1e-4 | 1536 | 1,536 |
| 2 | Videos (10M) + Images | 200k | 1e-4 | 768 | 11,520 |
| 3 | High-res videos | 50k | 5e-5 | 384 | 7,680 |

```bash
# Stage 1: Image training
torchrun --nproc_per_node=8 train.py --stage 1 --output_dir outputs/

# Stage 2: Low-resolution video training
torchrun --nproc_per_node=8 train.py --stage 2 --output_dir outputs/ \
    --resume_from outputs/checkpoint-50000/model.pt

# Stage 3: High-resolution video training
torchrun --nproc_per_node=8 train.py --stage 3 --output_dir outputs/ \
    --resume_from outputs/checkpoint-250000/model.pt
```

## Generation

```bash
# Text-to-image
python generate.py --checkpoint_dir outputs/ --mode image \
    --prompt "A beautiful mountain landscape at sunset"

# Text-to-video (5s, 768p, 24fps)
python generate.py --checkpoint_dir outputs/ --mode video \
    --prompt "A steam train crossing the Glenfinnan Viaduct" \
    --height 768 --width 768 --num_frames 121

# Text-to-video (10s)
python generate.py --checkpoint_dir outputs/ --mode video \
    --prompt "..." --num_frames 241

# Image-to-video
python generate.py --checkpoint_dir outputs/ --mode i2v \
    --prompt "The moon rises over the city" \
    --image_path input.jpg
```

## Ablation Studies

```bash
# All ablations
python ablation.py --ablation all

# Specific ablations
python ablation.py --ablation spatial    # Spatial pyramid vs standard flow (Fig. 7)
python ablation.py --ablation temporal   # Temporal pyramid vs full-sequence (Fig. 8)
python ablation.py --ablation renoising  # Corrective noise at jump points (Fig. 10)
python ablation.py --ablation causal     # Causal vs bidirectional attention (Fig. 11)
python ablation.py --ablation coupled    # Coupled vs random noise (Fig. 13)
```

## Model Architecture

- **DiT**: MM-DiT based on SD3 Medium, 24 layers, 2B parameters
  - Sinusoidal position encoding (spatial)
  - 1D RoPE (temporal)
  - Blockwise causal attention for autoregressive generation
  - Joint text-visual attention (T5 + CLIP conditioning)

- **VAE**: 3D causal VAE (MAGVIT-v2 style)
  - 8×8×8 compression (spatial 8×, temporal 8×)
  - 3D causal convolutions
  - 16 latent channels

## Efficiency

For a 10s, 241-frame video at 768p:
- Full-sequence diffusion: 119,040 tokens
- Pyramidal flow matching: ≤15,360 tokens (~7.8× reduction)
- Training time: 20.7k A100 GPU hours total

## Results

On VBench (5s, 768p, 24fps videos):
| Metric | Score |
|--------|-------|
| Total Score | 81.72 |
| Quality Score | **84.74** (best among public-data models) |
| Motion Smoothness | 99.12 |
| Dynamic Degree | 64.63 |

## Citation

```bibtex
@article{jin2024pyramidal,
  title={Pyramidal Flow Matching for Efficient Video Generative Modeling},
  author={Jin, Yang and Sun, Zhicheng and Li, Ningyuan and Xu, Kun and Jiang, Hao and Zhuang, Nan and Huang, Quzhe and Song, Yang and Mu, Yadong and Lin, Zhouchen},
  journal={arXiv preprint},
  year={2024}
}
```
