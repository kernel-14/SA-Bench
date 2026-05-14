# Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing

This repository contains a reproduction of the paper:

> **Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing**
> Kaifeng Gao, Jiaxin Shi, Hanwang Zhang, Chunping Wang, Jun Xiao, Long Chen

## Overview

Ca2-VDM is an efficient autoregressive video diffusion model that introduces two key innovations:

1. **Causal Generation**: Replaces bidirectional temporal attention with causal (lower triangular) attention, enabling unidirectional feature computation. This ensures that conditional frames' KV-cache can be precomputed in previous autoregression steps and reused.

2. **Cache Sharing**: Shares the KV-cache across all denoising timesteps by using a distinct timestep embedding (t=0) for clean prefix frames. This avoids storing separate caches for each denoising step.

The result is a significantly more efficient autoregressive VDM that maintains comparable quality to state-of-the-art models while eliminating redundant computations.

## Repository Structure

```
ca2-vdm/
├── ca2_vdm/                    # Core model package
│   ├── __init__.py
│   ├── model.py               # Ca2VDM model + TransformerBlock
│   ├── attention.py           # CausalTemporalAttention + PrefixEnhancedSpatialAttention
│   ├── cache.py               # KVCacheManager + TemporalKVCacheQueue
│   ├── tpe.py                 # CyclicTPE + PositionalEmbeddings
│   └── diffusion.py           # DiffusionProcess (training + sampling)
├── training/
│   └── train.py               # Training script (two-stage procedure)
├── inference/
│   ├── generate.py            # Autoregressive inference with KV-cache
│   └── evaluate.py            # FVD computation + FLOPs analysis
├── utils/
│   └── flops_analysis.py      # FLOPs and memory analysis
├── configs/
│   ├── t2v_config.yaml        # Text-to-Video configuration
│   ├── vp_config.yaml         # Video Prediction configuration
│   └── baseline_os_fix.yaml   # OS-Fix baseline configuration
├── tests/
│   └── test_model.py          # Unit tests for all components
└── README.md
```

## Core Contributions Reproduced

### 1. Causal Temporal Attention (Section 3.2, Eq. 3)

Implemented in `ca2_vdm/attention.py` as `CausalTemporalAttention`:
- Lower triangular attention mask ensures each frame only attends to preceding frames
- KV-cache support: prepends cached prefix K/V to current K/V during inference
- Spatial dimension treated as batch for computational efficiency

### 2. Prefix-Enhanced Spatial Attention (Section 3.2, Eq. 4)

Implemented in `ca2_vdm/attention.py` as `PrefixEnhancedSpatialAttention`:
- For denoising target frames: concatenates P' prefix frames' K/V spatially
- For clean prefix frames: self-repeats K/V P' times
- Spatial KV-cache stores only the most recent P' frames

### 3. KV-Cache Queue with Cache Sharing (Section 3.3)

Implemented in `ca2_vdm/cache.py`:
- `TemporalKVCacheQueue`: FIFO queue with max length P_max, auto-dequeue
- `KVCacheManager`: Manages both temporal (per-layer queue) and spatial (per-layer overwrite) caches
- Cache sharing: All denoising timesteps share the same cache since clean prefix always uses t=0 embedding

### 4. Cyclic Temporal Positional Embeddings (Section 3.3)

Implemented in `ca2_vdm/tpe.py`:
- `CyclicTPE`: Sinusoidal TPEs with cyclic shift mechanism
- During training: random cyclic offset
- During inference: chunk-by-chunk assignment with cyclic wrapping
- Ensures training/inference alignment when KV-cache causes TPEs to be bound to stored K/V

### 5. Distinct Timestep Embeddings (Section 3.2)

Implemented in `ca2_vdm/model.py`:
- Clean prefix frames: always assigned `tEmb(0)` 
- Denoising target frames: assigned `tEmb(t)` based on diffusion timestep
- Enables cache sharing across denoising steps

### 6. Partial Noising for Training (Section 3.2, Eq. 5)

Implemented in `ca2_vdm/diffusion.py`:
- Randomly sampled prefix length P ∈ {1, 1+l, ..., 1+nl}
- Loss mask excludes clean prefix from L_simple computation
- Combined loss: L_simple + λ * L_vlb (following Nichol & Dhariwal, 2021)

### 7. Two-Stage Training Procedure (Section 4.1, Appendix C)

Implemented in `training/train.py`:
- Stage 1: Causal pretraining without prefix on shorter videos
- Stage 2: Training with extendable clean prefix + cyclic TPEs

### 8. Autoregressive Inference Pipeline (Section 3.3)

Implemented in `inference/generate.py`:
- Denoising stage: generates l frames using KV-cache
- Cache writing stage: computes clean K/V of generated frames
- Classifier-free guidance support
- Improved DDPM schedule with 100 steps

### 9. FLOPs and Memory Analysis (Section 4.3, Figures 6, 8, Table 6)

Implemented in `utils/flops_analysis.py`:
- Per-component FLOPs counting (temporal, spatial, cross attention)
- Comparison between Ca2-VDM and bidirectional baselines
- Memory usage estimation for KV-cache

## Model Configurations

### Text-to-Video (T2V)
| Parameter | Value |
|-----------|-------|
| Architecture | Spatial-temporal Transformer |
| Hidden dim | 1152 |
| Heads | 16 |
| Layers | 28 |
| Chunk length (l) | 16 |
| Max prefix (P_max) | 49 (1 + 3×16) |
| Training length (L_train) | 65 |
| Prefix enhancement (P') | 3 |
| Resolution | 256×256 |
| VAE latent | 32×32×4 |
| Text encoder | T5-XXL (4096-dim) |
| Dataset | InternVid (4.9M pairs) |
| Stage 1 batch/step | 288 / 32k |
| Stage 2 batch/step | 144 / 21k |

### Video Prediction (SkyTimelapse)
| Parameter | Value |
|-----------|-------|
| Chunk length (l) | 8 |
| Max prefix (P_max) | 25 (1 + 3×8) |
| Training length (L_train) | 33 |
| Dataset | SkyTimelapse (997 train, 111 test) |
| Batch size / steps | 8 / 11k |

## Key Design Decisions

1. **Why causal attention enables cache sharing**: In bidirectional attention, K/V of prefix frames depend on future noisy frames, so they change at every denoising step. Causal attention makes prefix K/V independent of future frames.

2. **Why distinct timestep embeddings matter**: If all frames shared the same t-embedding, the prefix's K/V would differ across denoising timesteps, requiring T separate caches. Using tEmb(0) for the prefix makes cache sharing possible.

3. **Why Cyclic-TPEs are necessary**: With KV-cache, TPEs are bound to stored K/V. When old frames are dequeued, we can't reassign TPEs from scratch. Cyclic shift ensures the new frames' TPEs match the training distribution.

## Assumptions and Unresolved Details

1. **Initialization**: The paper initializes from Open-Sora v1.0 weights. Our implementation supports loading pretrained weights but doesn't include the specific weight conversion from Open-Sora's bidirectional attention to causal attention.

2. **VAE**: We assume the Stable Diffusion VAE (8x downsampling, 4 channels). The actual pretrained VAE weights need to be downloaded separately.

3. **Text encoder**: T5-XXL text embeddings (4096-dim). Our implementation includes the projection layer but doesn't include the T5 model itself.

4. **Exact architecture details**: The paper doesn't specify the exact number of layers, heads, etc. for Open-Sora v1.0. We use reasonable values (28 layers, 16 heads, dim=1152) consistent with the PixArt-α architecture they reference.

5. **AdaLN implementation**: We use AdaLN-Zero with 12-dim modulation (shift, scale, gate for 4 sub-modules). The paper mentions using AdaLN but exact details may differ slightly.

6. **VLB loss computation**: The exact VLB loss formulation follows Nichol & Dhariwal (2021) with learnable covariance. Implementation details may need refinement.

## Usage

### Training

```bash
# Stage 1: Causal pretraining (no prefix)
python training/train.py --mode t2v --stage 1 --data_dir /path/to/data

# Stage 2: Training with prefix
python training/train.py --mode t2v --stage 2 --data_dir /path/to/data \
    --pretrained checkpoints/stage1.pt

# Video prediction
python training/train.py --mode vp --stage 2 --data_dir /path/to/skytimelapse
```

### Inference

```bash
python inference/generate.py \
    --checkpoint checkpoints/model.pt \
    --first_frame path/to/frame.png \
    --text_prompt "A cat walking on a beach" \
    --num_ar_steps 5 \
    --cfg_scale 7.5
```

### Testing

```bash
python tests/test_model.py
```

### FLOPs Analysis

```bash
python utils/flops_analysis.py
```

## Key Results (from paper)

- **Generation quality**: Ca2-VDM achieves FVD of 181 on MSR-VTT (zero-shot) and 184.5 on UCF-101 (finetuned), competitive with state-of-the-art.
- **Speed**: 52.1s for 80 frames at 256×256 vs 130.1s for OS-Ext (2.5x speedup) and 77.5s for OS-Fix.
- **Memory**: KV-cache uses ~0.86 GB (with PE) vs ~17.70 GB for Live2diff (20x savings).
- **FLOPs**: Ca2-VDM FLOPs grow linearly with AR steps; OS-Ext grows quadratically.

## License

This reproduction is for research purposes only. Please refer to the original paper and its codebase for license information.

## Citation

```bibtex
@article{gao2025ca2vdm,
  title={Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing},
  author={Gao, Kaifeng and Shi, Jiaxin and Zhang, Hanwang and Wang, Chunping and Xiao, Jun and Chen, Long},
  year={2025}
}
```
