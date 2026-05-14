# Hi-MAR: Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots

This repository contains a reproduction of the Hi-MAR paper:

> **Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots**
> Guangting Zheng, Yehao Li, Yingwei Pan, Jiajun Deng, Ting Yao, Yanyong Zhang, Tao Mei

## Overview

Hi-MAR introduces a hierarchical masked autoregressive framework for image generation that:

1. **Phase 1**: Generates low-resolution (128×128) image tokens using masked autoregressive modeling, capturing global structure
2. **Phase 2**: Uses the phase 1 conditional tokens as "pivots" to guide high-resolution (256×256) token generation

Key innovations:
- **Scale-aware Transformer blocks**: AdaLN-Zero conditioning with learnable scale vectors to distinguish between low-res and high-res phases
- **Diffusion Transformer head**: Self-attention-based diffusion head for phase 2 that models inter-token dependencies (vs. MLP-based head in MAR)
- **Training-inference consistency**: Uses conditional tokens (not visual tokens) from phase 1 as pivots, avoiding the discrepancy between training and inference

## Architecture

```
Hi-MAR
├── Shared Scale-aware Transformer Backbone
│   ├── Phase 1: Low-res tokens (128×128 → 8×8 latent = 64 tokens)
│   └── Phase 2: High-res tokens (256×256 → 16×16 latent = 256 tokens)
│       └── Concatenated with phase 1 conditional tokens (pivots)
├── Phase 1 Diffusion Head: MLP-based (per-token)
└── Phase 2 Diffusion Head: Transformer-based (global context)
```

### Model Variants (Table 1)

| Model | Transformer Layers | Hidden Size | Diff. Head1 | Diff. Head2 | Params |
|-------|-------------------|-------------|-------------|-------------|--------|
| Hi-MAR-B | 24 | 768 | 6L, 1024 | 6L, 512 | 244M |
| Hi-MAR-L | 32 | 1024 | 8L, 1280 | 8L, 512 | 529M |
| Hi-MAR-H | 40 | 1280 | 12L, 1536 | 12L, 768 | 1090M |

## Results

### Class-Conditional ImageNet 256×256 (Table 2)

| Model | FID↓ (w/o CFG) | IS↑ (w/o CFG) | FID↓ (w/ CFG) | IS↑ (w/ CFG) |
|-------|----------------|----------------|----------------|---------------|
| Hi-MAR-B | 2.11 | 251.46 | 1.93 | 293.0 |
| Hi-MAR-L | 1.72 | 278.63 | 1.66 | 322.3 |
| Hi-MAR-H | 1.55 | 300.72 | 1.52 | 322.78 |

### Text-to-Image MS-COCO 256×256 (Table 3)

| Model | FID↓ |
|-------|------|
| Hi-MAR-S | 4.77 |

## Repository Structure

```
hi-mar/
├── models/
│   ├── __init__.py
│   ├── hi_mar.py          # Main Hi-MAR model (class-conditional)
│   ├── hi_mar_t2i.py      # Hi-MAR for text-to-image generation
│   ├── transformer.py     # Scale-aware Transformer backbone
│   └── diffusion_loss.py  # MLP and Transformer diffusion heads
├── utils/
│   ├── __init__.py
│   ├── ema.py             # Exponential Moving Average
│   ├── logger.py          # Training logger
│   ├── vae.py             # VAE tokenizer utilities
│   └── coco_dataset.py    # MS-COCO dataset
├── configs/
│   ├── hi_mar_b_imagenet.yaml
│   ├── hi_mar_l_imagenet.yaml
│   ├── hi_mar_h_imagenet.yaml
│   └── hi_mar_s_coco.yaml
├── train.py               # Training script
├── generate.py            # Generation and evaluation script
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

You also need the KL-16 VAE from MAR. Download it from the MAR repository and place it at `pretrained/kl16.ckpt`.

## Training

### Class-Conditional ImageNet

```bash
# Single GPU
python train.py \
    --task imagenet \
    --model hi_mar_b \
    --data_path /path/to/imagenet \
    --vae_path pretrained/kl16.ckpt \
    --output_dir output/hi_mar_b_imagenet

# Multi-GPU (8 GPUs)
torchrun --nproc_per_node=8 train.py \
    --task imagenet \
    --model hi_mar_b \
    --data_path /path/to/imagenet \
    --vae_path pretrained/kl16.ckpt \
    --output_dir output/hi_mar_b_imagenet
```

### Text-to-Image MS-COCO

```bash
python train.py \
    --task coco \
    --model hi_mar_s \
    --data_path /path/to/coco \
    --vae_path pretrained/kl16.ckpt \
    --lr 8e-4 \
    --weight_decay 0.03 \
    --output_dir output/hi_mar_s_coco
```

## Generation and Evaluation

```bash
# Generate 50K images for ImageNet FID evaluation
python generate.py \
    --task imagenet \
    --model hi_mar_b \
    --checkpoint output/hi_mar_b_imagenet/checkpoint_epoch0799.pth \
    --num_samples 50000 \
    --cfg_scale 1.5 \
    --num_steps_phase1 32 \
    --num_steps_phase2 4 \
    --output_dir generated/hi_mar_b \
    --save_images \
    --ref_path /path/to/imagenet/val

# Generate 30K images for COCO FID evaluation
python generate.py \
    --task coco \
    --model hi_mar_s \
    --checkpoint output/hi_mar_s_coco/checkpoint_best.pth \
    --num_samples 30000 \
    --cfg_scale 1.5 \
    --coco_path /path/to/coco \
    --output_dir generated/hi_mar_s
```

## Implementation Details

### Scale-Aware Transformer Block

Following Eq. 2 in the paper, each Transformer block uses AdaLN-Zero conditioning:

```
v_tilde = a * v + b
alpha1, beta1, gamma1, alpha2, beta2, gamma2 = split(v_tilde)
z_a = z^i + gamma1 * Attention(alpha1 * LN(z^i) + beta1)
z^{i+1} = z_a + gamma2 * FFN(alpha2 * LN(z_a) + beta2)
```

where `v` is the scale vector from sinusoidal embedding + MLP.

### Diffusion Transformer Head

Following Eq. 3 in the paper, the Diffusion Transformer head uses:

```
alpha1, beta1, gamma1, alpha2, beta2, gamma2 = split(c)
y_a = y^i + gamma1 * Attention(alpha1 * LN(y^i) + beta1)
y^{i+1} = y_a + gamma2 * FFN(alpha2 * LN(y_a) + beta2)
```

where `c` is the context vector (sum of timestep embedding and conditional tokens).

### Training-Inference Consistency

A key design choice: instead of using ground-truth low-resolution visual tokens as pivots (which causes training-inference discrepancy), Hi-MAR uses the **conditional tokens** output from the Transformer after processing low-resolution tokens. This ensures the same type of information is used during both training and inference.

### Masking Strategies

- **Phase 1**: Uniform sampling in [0.7, 1.0] (same as MAR)
- **Phase 2**: Cosine masking schedule from MaskGIT
- **Text-to-image**: Beta distribution (α=4, β=1) for masking ratio

### Inference

- Phase 1: 32 autoregressive steps with cosine schedule
- Phase 2: 4 autoregressive steps (much fewer due to global structure from phase 1)
- CFG: Applied in phase 2 (for w/o CFG setting, CFG is only turned off for phase 2)

## Assumptions and Unresolved Details

1. **VAE**: The paper uses the KL-16 VAE from MAR. The exact architecture details are not specified in this paper but follow MAR's implementation.

2. **Patch size**: The paper uses 16×16 patches (KL-16 VAE stride), giving 16×16=256 tokens for 256×256 images and 8×8=64 tokens for 128×128 images.

3. **Hi-MAR-S architecture**: The paper mentions a "light-weight version" for COCO but doesn't specify exact dimensions. We use a smaller variant with hidden_size=512, depth=16.

4. **Scale vector injection**: The paper says scale information is injected via AdaLN-Zero. We implement this by adding scale embeddings to class/text embeddings as the conditioning signal.

5. **Diffusion head for phase 2**: The paper uses 4 diffusion steps at inference for phase 2. The exact number of training diffusion steps is not specified; we use 100.

6. **CFG for w/o CFG setting**: The paper notes "the CFG is only turned off during the prediction of dense tokens" for the w/o CFG setting, meaning phase 1 still uses CFG.
