# Pyramidal Flow Matching for Efficient Video Generative Modeling

Reproduction of the paper "Pyramidal Flow Matching for Efficient Video Generative Modeling" (Jin et al., 2024).

## Overview

This codebase implements a unified pyramidal flow matching algorithm for video generation that:
- Uses spatial pyramid flow across multiple resolutions (K=3 stages) to reduce computation
- Employs temporal pyramid conditions for efficient autoregressive video generation
- Trains a single unified Diffusion Transformer (DiT) end-to-end
- Generates high-quality 5-10 second videos at 768p resolution, 24 FPS
- Achieves training efficiency with only 20.7k A100 GPU hours

## Codebase Structure

```
repo/
├── config.py          # Dataclass configuration
├── config.yaml        # YAML configuration with all hyperparameters
├── layers.py          # Building blocks: RoPE, attention, MLP, transformer layers
├── model.py           # MM-DiT model (based on SD3 Medium, 2B params)
├── vae.py             # 3D VAE with 8x8x8 compression (MAGVIT-v2 architecture)
├── pyramidal_flow.py  # Core algorithm: spatial pyramid, temporal pyramid, flow matching
├── data.py            # Dataset loading and preprocessing
├── train.py           # Three-stage training pipeline
├── inference.py       # Text-to-video and image-to-video generation
├── evaluate.py        # Evaluation metrics: VBench, EvalCrafter, FID, FVD
├── utils.py           # Utility functions
└── requirements.txt   # Python dependencies
```

## Key Algorithms

### Spatial Pyramid (Section 3.2)
- Divides [0,1] into K=3 time windows, each interpolating between successive resolutions
- Only the final stage operates at full resolution
- Unified training via coupled noise sampling (Eqs. 9-10)
- Flow matching objective (Eq. 11): `|| v_t(ˆx_t) - (ˆx_e_k - ˆx_s_k) ||²`
- Inference with corrective renoising at jump points (Eq. 15)

### Temporal Pyramid (Section 3.3)
- Compressed lower-resolution history for autoregressive generation
- Gradually increasing resolutions for earlier frames
- Corruptive noise added to history during training to mitigate error accumulation

### Model Architecture
- MM-DiT: 24 transformer layers, 2048 hidden dim, 32 heads, ~2B parameters
- Blockwise causal attention for autoregressive generation
- Sinusoidal spatial position encoding + RoPE temporal encoding
- T5 + CLIP text conditioning

## Training

Three-stage training procedure on 128 A100 GPUs:
1. **Stage 1 (50k steps)**: Image training on LAION-5B, CC-12M, SA-1B, JourneyDB, synthetic data
2. **Stage 2 (200k steps)**: Low-resolution video training (80k at 2s, 120k at 5s) with WebVid-10M, OpenVid-1M, Open-Sora Plan
3. **Stage 3 (50k steps)**: High-resolution video training on 5-10s videos

Total: ~20.7k A100 GPU hours

## Usage

### Training
```bash
python train.py --stage 1  # Image training
python train.py --stage 2  # Video training
python train.py --stage 3  # High-res video fine-tuning
python train.py            # Full pipeline
```

### Inference
```bash
# Text-to-video
python inference.py --prompt "A beautiful sunset over the ocean" \
    --checkpoint checkpoints/stage3/checkpoint_50000.pt \
    --num_frames 121 --fps 24 --output sunset.mp4

# Image-to-video
python inference.py --prompt "A moon rises from the sky" \
    --checkpoint checkpoints/stage3/checkpoint_50000.pt \
    --first_frame image.png --num_frames 121 --output moon.mp4
```

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Pyramid stages (K) | 3 |
| Spatial downsampling | 2x per stage |
| Corrective gamma | -1/3 |
| CFG scale | 7.0 |
| CFG training probability | 0.1 |
| History noise max | 1/3 |
| VAE compression | 8×8×8 |
| Patch size | 2 |

## References

- Pyramidal Flow Matching (Jin et al., 2024) - https://pyramid-flow.github.io
- SD3 Medium (Esser et al., 2024) - MM-DiT architecture
- MAGVIT-v2 (Yu et al., 2024) - 3D VAE architecture
- Flow Matching (Lipman et al., 2023) - Simulation-free training objective
