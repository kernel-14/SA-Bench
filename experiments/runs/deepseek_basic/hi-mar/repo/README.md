# Hi-MAR: Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots

This repository contains an implementation of the Hi-MAR paper:

> **Hi-MAR: Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots**
> Guangting Zheng, Yehao Li, Yingwei Pan, Jiajun Deng, Ting Yao, Yanyong Zhang, Tao Mei

## Overview

Hi-MAR introduces a hierarchical masked autoregressive framework for image generation that:

1. **First phase**: Predicts low-resolution image tokens (128×128) to capture global structure
2. **Second phase**: Uses those predicted tokens as "pivots" to guide high-resolution (256×256) token generation
3. **Scale-aware Transformer**: Injects scale information via adaLN-Zero operations
4. **Diffusion Transformer head**: Models inter-token dependencies with self-attention in the second phase

The model achieves state-of-the-art FID on ImageNet 256×256 class-conditional generation (1.93 FID for Hi-MAR-B) with only 54% of the computational cost of the baseline MAR.

## Repository Structure

```
.
├── himar/                      # Core library
│   ├── __init__.py             # Package exports
│   ├── model.py                # Hi-MAR main model (two-phase architecture)
│   ├── transformer.py          # Scale-aware Transformer blocks
│   ├── diffusion_head.py       # MLP and Transformer diffusion heads
│   ├── masking.py              # Masking strategies (random, cosine, Beta)
│   ├── training.py             # Training loop, EMA, LR schedules
│   ├── data.py                 # Data loading for ImageNet and MS-COCO
│   └── evaluation.py           # FID, IS, Precision/Recall evaluation
├── configs/                    # Configuration files
│   ├── imagenet_himar_b.yaml   # Hi-MAR-Base for ImageNet
│   ├── imagenet_himar_l.yaml   # Hi-MAR-Large for ImageNet
│   ├── imagenet_himar_h.yaml   # Hi-MAR-Huge for ImageNet
│   └── coco_himar_s.yaml       # Hi-MAR-Small for MS-COCO
├── scripts/                    # Executable scripts
│   ├── train_imagenet.py       # Training on ImageNet
│   ├── generate.py             # Image generation
│   └── ablation.py             # Ablation study variants
└── README.md                   # This file
```

## Key Components

### 1. Hi-MAR Transformer (Section 3.2)

The `HiMARTransformer` in `himar/transformer.py` implements a scale-aware Transformer backbone. Each block uses adaLN-Zero operations conditioned on a scale vector derived from sinusoidal embeddings:

```python
v_tilde = a * v + b
alpha1, beta1, gamma1, alpha2, beta2, gamma2 = split(v_tilde)
z_a = z^i + gamma1 * Attention(alpha1 * LN(z^i) + beta1)
z^{i+1} = z_a + gamma2 * FFN(alpha2 * LN(z_a) + beta2)
```

The scale vector `v` is computed by encoding the scale index (0 for low-res, 1 for high-res) through sinusoidal embeddings and an MLP.

### 2. Diffusion Heads (Section 3.3)

- **MLP Diffusion Head** (`himar/diffusion_head.py:MLPDiffusionHead`): Used in Phase 1. Treats each token independently with adaLN conditioning.
- **Diffusion Transformer Head** (`himar/diffusion_head.py:DiffusionTransformerHead`): Used in Phase 2. Uses self-attention across all tokens to model inter-token dependencies. Conditioned on the sum of timestep embedding and mean conditional tokens.

### 3. Two-Phase Architecture (Figure 2b)

The `HiMAR` class in `himar/model.py` implements the full two-phase pipeline:

- **Phase 1**: Hi-MAR Transformer (scale=0) → MLP Diffusion Head → low-res tokens + conditional tokens
- **Phase 2**: [conditional tokens from Phase 1, masked high-res tokens] → Hi-MAR Transformer (scale=1) → Diffusion Transformer Head → high-res tokens

### 4. Masking Strategies (Section 4.2)

- **Phase 1**: Random masking with ratio r ~ Uniform(0.7, 1.0) (same as MAR)
- **Phase 2 (ImageNet)**: Cosine masking schedule (following MaskGIT)
- **Phase 2 (MS-COCO)**: Beta distribution masking with α=4, β=1

### 5. Model Configurations (Table 1)

| Model | Transformer Layers | Hidden Size | Diff Head1 Layers/Size | Diff Head2 Layers/Size | #Params |
|-------|-------------------|-------------|----------------------|----------------------|--------|
| Hi-MAR-B | 24 | 768 | 6/1024 | 6/512 | 244M |
| Hi-MAR-L | 32 | 1024 | 8/1280 | 8/512 | 529M |
| Hi-MAR-H | 40 | 1280 | 12/1536 | 12/768 | 1090M |

## Usage

### Training

```bash
# Class-conditional on ImageNet
python scripts/train_imagenet.py --config configs/imagenet_himar_b.yaml

# Text-to-image on MS-COCO
python scripts/train_coco.py --config configs/coco_himar_s.yaml
```

### Generation

```bash
# Class-conditional generation
python scripts/generate.py --config configs/imagenet_himar_b.yaml \
    --checkpoint checkpoints/final_model.pt \
    --class_idx 0 1 2 3 4 5 6 7 --num_images 16

# Text-to-image generation
python scripts/generate.py --config configs/coco_himar_s.yaml \
    --checkpoint checkpoints/final_model.pt \
    --prompt "a red car on a sunny street" --num_images 4
```

### Ablation Study

```bash
# Evaluate different ablation variants
python scripts/ablation.py --config configs/imagenet_himar_b.yaml --variant full
python scripts/ablation.py --config configs/imagenet_himar_b.yaml --variant mar
python scripts/ablation.py --config configs/imagenet_himar_b.yaml --variant diff_head
```

## Training Details

### ImageNet (Class-Conditional)
- **Optimizer**: AdamW (β₁=0.9, β₂=0.95)
- **Weight decay**: 0.02
- **LR schedule**: Constant 1e-4 with 100-epoch linear warmup
- **Epochs**: 800
- **EMA**: 0.9999 momentum
- **Inference**: 32 steps Phase 1, 4 steps Phase 2

### MS-COCO (Text-to-Image)
- **Optimizer**: AdamW with 8e-4 LR
- **Weight decay**: 0.03
- **Warmup**: 8K-step linear warmup
- **Masking**: Beta(4, 1) distribution
- **Inference**: 32 steps Phase 1, 4 steps Phase 2

## Implementation Notes

### What We Reproduce

This implementation covers all core contributions of the paper:

1. ✅ **Hierarchical two-phase architecture**: Low-res → high-res token prediction with conditional token pivots
2. ✅ **Scale-aware Transformer blocks**: adaLN-Zero with sinusoidal scale embeddings
3. ✅ **Diffusion Transformer head**: Self-attention across tokens in Phase 2
4. ✅ **All three model scales**: Base (244M), Large (529M), Huge (1090M)
5. ✅ **Masking strategies**: Random, cosine schedule, and Beta distribution
6. ✅ **Training setup**: AdamW, EMA, linear warmup, CFG dropout
7. ✅ **Evaluation**: FID, IS, Precision/Recall computation
8. ✅ **Ablation study framework**: All variants from Table 5
9. ✅ **Text-to-image support**: MS-COCO with CLIP text encoder

### Assumptions and Missing Details

- **VAE**: Uses the KL-16 VAE from Stable Diffusion (same as MAR). The exact checkpoint used in MAR may produce slightly different latent spaces but this is the standard choice.
- **ImageNet statistics**: FID computation requires pre-computed ImageNet validation set statistics. The paper uses standard 50K generated samples.
- **T2I-CompBench**: Full T2I-CompBench evaluation requires external VQA models (BLIP, etc.). We provide the evaluation structure but full evaluation needs the official benchmark code.
- **Speed measurements**: The paper reports speed on H100 GPUs with batch size 128. Our code is designed to be GPU-agnostic.
- **Training-inference discrepancy**: The paper addresses this by using conditional tokens (not visual tokens) as pivots. Our implementation follows this approach by default.

### Dependencies

- PyTorch >= 2.0
- torchvision
- diffusers (for VAE)
- transformers (for CLIP text encoder)
- scipy, numpy, scikit-learn
- PyYAML

## References

- MAR: [Autoregressive Image Generation without Vector Quantization](https://arxiv.org/abs/2406.11838)
- DiT: [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- MaskGIT: [Masked Generative Image Transformer](https://arxiv.org/abs/2202.04200)
