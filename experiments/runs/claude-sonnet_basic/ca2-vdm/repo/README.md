# Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing

This repository contains a reproduction of the paper:

> **Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing**  
> Kaifeng Gao*, Jiaxin Shi*, Hanwang Zhang, Chunping Wang, Jun Xiao, Long Chen  
> ICML 2024

## Overview

Ca2-VDM addresses the inefficiency of existing autoregressive video diffusion models (VDMs) by introducing two key mechanisms:

1. **Causal Generation**: Replaces bidirectional temporal attention with causal (lower-triangular masked) temporal attention, enabling KV-cache reuse across autoregression steps.

2. **Cache Sharing**: Shares the KV-cache across all denoising timesteps by using a distinct timestep embedding (t=0) for clean prefix frames, eliminating the need to store separate caches per denoising step.

Additional contributions:
- **KV-Cache Queue**: A queue structure for temporal KV-cache that enables extendable long-term context while maintaining bounded memory.
- **Cyclic-TPEs**: Cyclic temporal positional embeddings that handle video generation beyond the training length.
- **Prefix-Enhanced Spatial Attention**: Concatenates a sub-prefix of clean frames to spatial attention for stronger conditional guidance.

## Repository Structure

```
ca2_vdm/
├── models/
│   ├── attention.py          # CausalTemporalAttention, PrefixEnhancedSpatialAttention
│   ├── transformer.py        # Ca2VDMBlock, Ca2VDMTransformer
│   ├── diffusion.py          # Ca2VDM (full model with training/inference)
│   ├── kv_cache.py           # KVCacheQueue, SpatialKVCache
│   └── positional_embedding.py  # SPE, TPE, Cyclic-TPE
├── data/
│   ├── dataset.py            # SkyTimelapseDataset, MSRVTTDataset, UCF101Dataset
│   └── transforms.py         # VideoTransform
├── utils/
│   ├── flops.py              # FLOPs counting (Figure 8)
│   └── timing.py             # Generation timing (Table 5, Figure 6)
└── configs/
    ├── sky_timelapse.yaml    # Video prediction config
    └── t2v.yaml              # Text-to-video config

scripts/
├── train.py                  # Training script
├── evaluate.py               # FVD evaluation script
└── generate.py               # Video generation script
```

## Key Implementation Details

### Causal Temporal Attention (Section 3.2)

The causal temporal attention uses a lower-triangular mask so each frame only attends to preceding frames:

```
CausalAttn(Q, K, V) = Softmax(QK^T / sqrt(C') + M) V
```

where M[i,j] = -inf if i < j else 0.

During inference, the KV-cache from clean prefix frames is concatenated with the current noisy frames' KVs, and the mask only applies within the query frames (not to the cached prefix).

### Cache Sharing (Section 3.2, 3.3)

The key insight is that with causal generation, the KV-cache of clean prefix frames is independent of the denoising timestep t. This is achieved by:
- Using `tEmb(0)` for clean prefix frames (both in training and inference)
- Using `tEmb(t)` for denoising target frames

This allows the same cache to be reused across all 100 denoising steps, reducing memory from O(T × cache_size) to O(cache_size).

### KV-Cache Queue (Section 3.3)

The temporal KV-cache is stored as a queue:
- New chunks are enqueued after each AR step (cache writing stage)
- Oldest chunks are dequeued when P_k reaches P_max
- This maintains bounded memory while providing long-term context

### Cyclic-TPEs (Section 3.3)

When the cumulative video length exceeds L_train = P_max + l:
- The denoising target frames are assigned TPEs cyclically from the beginning
- During training, each sample uses a randomly shifted TPE sequence to support this

### Prefix-Enhanced Spatial Attention (Section 3.2, Eq. 4)

For each denoising target frame i, the spatial attention key/value is enhanced by concatenating P' clean prefix frames:
```
K(i) = W_K [h_0^{P-P'}, ..., h_0^{P-1}, h_t^i]  for i >= P
```

This provides stronger guidance from the conditional frames at the spatial level.

## Training

### Video Prediction (SkyTimelapse)

```bash
python scripts/train.py \
    --task video_prediction \
    --data_dir /path/to/sky_timelapse \
    --output_dir ./outputs/sky_timelapse \
    --chunk_size 8 \
    --max_prefix_len 25 \
    --batch_size 8 \
    --max_steps 11000 \
    --lr 2e-5
```

### Text-to-Video (Stage 1: Causal pretraining)

```bash
python scripts/train.py \
    --task t2v_stage1 \
    --data_dir /path/to/internvid \
    --output_dir ./outputs/t2v_stage1 \
    --chunk_size 16 \
    --max_prefix_len 1 \
    --batch_size 288 \
    --max_steps 32000 \
    --lr 2e-5
```

### Text-to-Video (Stage 2: With clean prefix)

```bash
python scripts/train.py \
    --task t2v_stage2 \
    --data_dir /path/to/internvid \
    --output_dir ./outputs/t2v_stage2 \
    --pretrained ./outputs/t2v_stage1/model_final.pt \
    --chunk_size 16 \
    --max_prefix_len 49 \
    --batch_size 144 \
    --max_steps 21000 \
    --lr 2e-5
```

## Evaluation

### FVD on SkyTimelapse (Table 4)

```bash
python scripts/evaluate.py \
    --checkpoint ./outputs/sky_timelapse/model_final.pt \
    --dataset sky_timelapse \
    --data_dir /path/to/sky_timelapse \
    --chunk_size 8 \
    --max_prefix_len 25 \
    --num_denoising_steps 100
```

### Zero-shot FVD on MSR-VTT (Table 1)

```bash
python scripts/evaluate.py \
    --checkpoint ./outputs/t2v_stage2/model_final.pt \
    --dataset msrvtt \
    --data_dir /path/to/msrvtt \
    --chunk_size 16 \
    --max_prefix_len 49 \
    --guidance_scale 7.5
```

## Generation

```bash
python scripts/generate.py \
    --checkpoint ./outputs/sky_timelapse/model_final.pt \
    --first_frame /path/to/first_frame.png \
    --num_frames 80 \
    --output_path generated_video.mp4 \
    --use_kv_cache
```

## Hyperparameters

From the paper (Section 4.1 and Appendix C):

| Setting | Value |
|---------|-------|
| Diffusion timesteps T | 1000 |
| β₁ | 1e-4 |
| β_T | 0.02 |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Inference steps | 100 (improved DDPM) |
| CFG scale (T2V) | 7.5 |
| Resolution | 256×256 |
| Chunk size l (T2V) | 16 |
| Chunk size l (video pred.) | 8 |
| P_max (T2V) | 49 = 1 + 3×16 |
| P_max (video pred.) | 25 = 1 + 3×8 |
| Spatial prefix P' | 3 |

## Experimental Results

The paper reports the following key results:

**Table 1: Zero-shot FVD on MSR-VTT and UCF-101**
- Ca2-VDM: MSR-VTT FVD = 181, UCF-101 FVD = 277.7

**Table 2: Finetuned FVD on UCF-101**
- Ca2-VDM: FVD = 184.5

**Table 5: Generation time for 80 frames at 256×256**
- Ca2-VDM: 52.1s (vs OS-Ext: 130.1s, OS-Fix: 77.5s, StreamT2V: 150s)

**Table 4: Ablation on SkyTimelapse (P_max=25, with PE)**
- Chunk 1 FVD: 257.4, Chunk 2: 216.5, Chunk 3: 238.5

## Assumptions and Unresolved Details

1. **VAE**: The paper uses the VAE from Stable Diffusion (8× spatial downsampling, 4 latent channels). This implementation assumes the input is already in latent space.

2. **Text Encoder**: T5-XXL (4096-dim output) is used for text conditioning. The text encoding step is not implemented in the training/generation scripts (placeholder).

3. **Open-Sora Initialization**: The paper initializes from Open-Sora v1.0 weights. This reproduction trains from scratch.

4. **Exact Architecture**: The paper builds on Open-Sora/PixArt-alpha style DiT. Some architectural details (exact normalization, attention variants) may differ from the original.

5. **InternVid Dataset**: The T2V training uses a filtered subset of 4.9M video-text pairs from InternVid. Dataset filtering details are not specified.

6. **FVD Computation**: Uses I3D features following StyleGAN-V codebase. The I3D model weights need to be downloaded separately.
