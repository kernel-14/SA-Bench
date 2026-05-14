# Pyramidal Flow Matching for Efficient Video Generative Modeling

This repository contains a reproduction of the paper **"Pyramidal Flow Matching for Efficient Video Generative Modeling"** (Jin et al., 2024).

> **Paper**: https://pyramid-flow.github.io

## Overview

Pyramidal Flow Matching introduces a unified framework for efficient video generation that combines:

1. **Spatial Pyramid Flow Matching**: Multi-resolution piecewise flow where only the final stage operates at full resolution, reducing computation by ~1/K factor.
2. **Temporal Pyramid History**: Compressed lower-resolution history conditioning for autoregressive video generation, reducing history tokens by up to 1/4^K.
3. **Unified Training**: Single DiT model jointly optimized across all pyramid stages, unlike cascaded approaches requiring separate models.

The method achieves 20.7k A100 GPU hours for training 10-second 768p 24fps videos, significantly more efficient than prior approaches.

## Repository Structure

```
pyramidal_flow/
├── __init__.py                  # Package exports
├── pyramidal_flow.py            # Core PyramidalFlowMatching class
├── spatial_pyramid.py           # Spatial pyramid implementation
├── temporal_pyramid.py          # Temporal pyramid history conditioning
├── unified_training.py          # Three-stage training pipeline
├── models/
│   ├── __init__.py
│   ├── dit.py                   # MM-DiT architecture with causal attention
│   └── velocity_model.py        # Velocity prediction wrapper
├── training/
│   ├── __init__.py
│   ├── config.py                # Training hyperparameters (Table 4)
│   ├── trainer.py               # Training loop implementation
│   └── data_pipeline.py         # Data loading with Patch n' Pack
├── inference/
│   ├── __init__.py
│   ├── renoising.py             # Inference with corrective renoising
│   └── sampler.py               # T2V and I2V generation samplers
└── utils/
    ├── __init__.py
    ├── efficiency.py            # Computational efficiency metrics
    ├── evaluation.py            # VBench and EvalCrafter evaluators
    └── visualization.py         # Visualization tools
```

## Core Contributions Reproduced

### 1. Spatial Pyramid (Section 3.2)

The spatial pyramid divides the denoising trajectory into K stages, each operating at a different resolution:
- Stage k interpolates between `Up(Down(x_1, 2^{k+1}))` and `Down(x_1, 2^k)`
- Only the final stage (k=0) operates at full resolution
- Computational cost reduced by factor ~1/K

**Key implementation**: `spatial_pyramid.py`
- `SpatialPyramid` class with piecewise flow, coupled noise sampling
- Eqs. (7), (9), (10), (11) implemented
- Corrective noise generation with blockwise covariance Σ'_block (Eq. 14)

### 2. Temporal Pyramid (Section 3.3)

Compresses autoregressive history using progressively lower resolutions:
- Newer frames at higher resolution, older frames at lower resolution
- Reduces history tokens by up to 1/4^K
- Corruptive noise added during training (strength ∈ [0, 1/3])

**Key implementation**: `temporal_pyramid.py`
- `TemporalPyramidHistory` for building compressed history
- `TemporalPyramidConditioning` for combining history with current latent
- Eqs. (16) and (17) implemented

### 3. Unified Flow Matching Objective (Section 3.2.1)

Joint training of all pyramid stages in a single DiT:
- Random stage sampling per update iteration
- Coupled noise endpoints for straight trajectories
- Single model handles generation + decompression

**Key implementation**: `pyramidal_flow.py`
- `PyramidalFlowMatching.compute_loss()` implements Eq. (11)
- Supports both image and video training

### 4. Inference with Renoising (Section 3.2.2)

Algorithm 1: Corrective renoising at jump points between pyramid stages:
- Upsampling + rescaling to match distribution means
- Corrective Gaussian noise with blockwise covariance to decorrelate
- Eq. (15): `x_{s_k} = (1+s_k)/2 * Up(x_{e_{k+1}}) + sqrt(3)*(1-s_k)/2 * n'`

**Key implementation**: `inference/renoising.py`
- `RenoisingInference` class with ODE solvers (Euler, Midpoint, RK4)
- Classifier-free guidance support
- Text-to-video and image-to-video generation

### 5. Model Architecture (Section 3.4)

MM-DiT based on SD3 Medium with 2B parameters:
- 24 transformer layers, 24 heads, 3072 hidden dim
- Blockwise causal attention for autoregressive generation
- Sinusoidal position encoding (spatial) + RoPE (temporal)
- Adaptive layer norm (adaLN) modulation

**Key implementation**: `models/dit.py`
- `PyramidalDiT` with full MM-DiT architecture
- `BlockwiseCausalAttention` for temporal causality
- `MMDiTBlock` with adaLN-Zero modulation

### 6. Three-Stage Training (Appendix B)

| Stage | Data | Steps | GPU Hours | Config |
|-------|------|-------|-----------|--------|
| 1 | Images only | 50k | 1,536 | LR=1e-4, β2=0.999, BS=1536 |
| 2 | Low-res video + 12.5% images | 200k | 11,520 | LR=1e-4, β2=0.95, BS=768 |
| 3 | High-res video (5-10s) | 50k | 7,680 | LR=5e-5, β2=0.95, BS=384 |

**Key implementation**: `unified_training.py`, `training/trainer.py`, `training/config.py`

## Usage

### Basic Setup

```python
import torch
from pyramidal_flow import PyramidalFlowMatching
from pyramidal_flow.models import PyramidalDiT, VelocityModel
from pyramidal_flow import SpatialPyramid, TemporalPyramid

# Build the model
dit = PyramidalDiT(
    input_dim=16,
    hidden_dim=3072,
    num_heads=24,
    num_layers=24,
    text_embed_dim=4096,
    use_causal_attention=True,
    num_spatial_stages=3,
)
velocity_model = VelocityModel(dit)

# Create pyramidal flow matching model
model = PyramidalFlowMatching(
    velocity_model=velocity_model,
    num_spatial_stages=3,
    num_temporal_levels=3,
    max_history_frames=12,
)

# Training step
x1 = torch.randn(4, 16, 96, 96)  # (B, C, H, W) clean latent
loss_dict = model.compute_loss(x1)
print(f"Loss: {loss_dict['loss']:.4f}")

# Sampling
sample = model.sample(
    image_shape=(1, 16, 96, 96),
    num_sampling_steps=50,
    guidance_scale=7.0,
)
```

### Full Training

```python
from pyramidal_flow.unified_training import UnifiedTrainingPipeline
from pyramidal_flow.training.config import TrainingConfig

config = TrainingConfig()
pipeline = UnifiedTrainingPipeline(config)
pipeline.train()  # Runs all 3 stages
```

### Text-to-Video Generation

```python
sampler = model.get_sampler(
    num_sampling_steps=50,
    guidance_scale=7.0,
    fps=24,
)

# With text embeddings
video_latent = sampler.text_to_video(
    prompt_embeddings=text_embeds,
    num_frames=121,  # 5s at 24fps
    latent_shape=(96, 96),
)
```

### Efficiency Analysis

```python
stats = model.get_efficiency_stats(video_frames=241, frame_resolution=(96, 96))
print(f"Token reduction: {stats['spatial_reduction_factor']}x (spatial)")
print(f"Compute reduction: {stats['compute_reduction_factor']:,}x")
print(f"GPU hours: {stats['estimated_gpu_hours_10s_video']:,}")
```

## Key Mathematical Details

### Corrective Renoising Derivation (Appendix A)

The renoising scheme at jump points ensures continuity of the probability path:

1. Upsampled endpoint distribution: `Up(x_{e_{k+1}}) ~ N(e_{k+1}*Up(Down(x_1, 2^{k+1})), (1-e_{k+1})^2*Σ)`
2. Target start distribution: `x_{s_k} ~ N(s_k*Up(Down(x_1, 2^{k+1})), (1-s_k)^2*I)`
3. Covariance matching yields: `e_{k+1} = 2s_k/(1+s_k)`, `α = sqrt(3)*(1-s_k)/2`
4. With γ = -1/3 for maximal signal preservation

### Coupled Noise Sampling (Appendix C.4)

Endpoints share the same noise direction for straighter trajectories:
- `x_{e_k} = e_k*Down(x_1, 2^k) + (1-e_k)*n`
- `x_{s_k} = s_k*Up(Down(x_1, 2^{k+1})) + (1-s_k)*n`

## Assumptions and Missing Details

1. **3D VAE**: The paper uses a custom 3D VAE (similar to MAGVIT-v2) with 8×8×8 compression. We provide the interface but actual VAE weights are not included (trained from scratch on WebVid-10M).

2. **SD3 Medium weights**: The MM-DiT is initialized from SD3 Medium. We provide the architecture and weight loading interface but not the actual weights.

3. **Text encoders**: T5 and CLIP encoders are used. Integration points are provided but actual encoder implementations depend on external libraries.

4. **Dataset**: Training requires the specific datasets (LAION-5B subset, CC-12M, SA-1B, JourneyDB, WebVid-10M, OpenVid-1M, Open-Sora Plan). Data loading interfaces are provided.

5. **Evaluation metrics**: VBench and EvalCrafter evaluators contain the paper's reported scores. Full metric computation requires external evaluation pipelines.

6. **Training scale**: The paper uses 128 A100 GPUs. Our implementation supports distributed training but actual multi-GPU orchestration depends on the training framework used.

## References

- Flow Matching: Lipman et al. (2023), Liu et al. (2023)
- DiT: Peebles & Xie (2023)
- MM-DiT: Esser et al. (2024) (SD3)
- VBench: Huang et al. (2024)
- EvalCrafter: Liu et al. (2024)
