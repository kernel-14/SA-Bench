# NFIG: Multi-Scale Autoregressive Image Generation via Frequency Ordering

This repository contains a reproduction of the paper:

> **NFIG: Multi-Scale Autoregressive Image Generation via Frequency Ordering**  
> Zhihao Huang, Xi Qiu, Yukuo Ma, Yifu Zhou, Junjie Chen, Hongyuan Zhang, Chi Zhang, Xuelong Li  
> NeurIPS 2025

## Overview

NFIG (Next-Frequency Image Generation) is a novel autoregressive image generation framework that decomposes the generation process into multiple frequency-guided stages. Instead of generating pixels in raster scan order or patches at increasing resolutions (like VAR), NFIG generates image tokens from low to high frequency bands, aligning with the natural hierarchical structure of image information in the spectral domain.

### Key Contributions

1. **Next-Frequency Image Generation (NFIG) framework**: Generates images by progressively predicting frequency components from low to high, resulting in a coarse-to-fine spatial reconstruction.

2. **Frequency-guided Residual-quantized VAE (FR-VAE)**: A novel image tokenizer that:
   - Decomposes latent features into frequency bands using FFT-based masks
   - Uses residual quantization to efficiently represent each frequency band
   - Achieves rFID of 0.85 on ImageNet-256

3. **State-of-the-art performance**: NFIG-310M achieves FID=2.81 and IS=332.42 on ImageNet-256, outperforming VAR-d20 (600M parameters) while being 1.25x faster.

## Architecture

### FR-VAE (Frequency-guided Residual-quantized VAE)

The FR-VAE tokenizer consists of:
- **Encoder**: CNN that downsamples 256×256 images to 16×16 latent features
- **Frequency Decomposer**: Decomposes latent features into 10 frequency bands using FFT
- **Frequency Residual Quantizer**: Quantizes each band at its natural resolution
- **Frequency Composer**: Reconstructs the full feature map from quantized bands
- **Decoder**: CNN that upsamples latent features back to 256×256 images

**Scale factors**: `[1, 2, 3, 4, 5, 6, 8, 10, 13, 16]`  
These represent the spatial resolution (h_i × w_i) of each frequency band's token map.  
Token counts per band: `[1, 4, 9, 16, 25, 36, 64, 100, 169, 256]` = **680 total tokens**

**Frequency band boundaries** are computed as:
```
σ_i = σ_{i-1} + (h_i × w_i) / Σ(h_j × w_j) × σ_max
```

**Loss function** (from paper Appendix B.1):
```
L = ||I - Î||₂² + ||f - f̂||₂² + L_p(I) + 0.5 × L_g(I)
```
where L_p is LPIPS perceptual loss and L_g is GAN loss.

### NFIG Transformer

The autoregressive generator uses a decoder-only transformer with:
- **Block-wise causal attention**: Tokens in band i can attend to all tokens in bands 0..i
- **AdaLN conditioning**: Class-conditional adaptive layer normalization
- **Shared token embedding**: Single codebook shared across all frequency bands
- **Learned positional embeddings**: Per-band positional embeddings

**Model sizes**:
| Model | Depth | Embed Dim | Heads | Parameters |
|-------|-------|-----------|-------|------------|
| NFIG-310M | 16 | 1024 | 16 | ~310M |
| NFIG-600M | 20 | 1152 | 16 | ~600M |

**Training details**:
- Optimizer: Adam, lr=8×10⁻⁵, batch_size=768
- Epochs: 350 (for 310M model)
- CFG dropout probability: 0.1
- Inference: CFG=4.5, top_k=990

## Results

### Main Results on ImageNet-256

| Model | rFID↓ | gFID↓ | IS↑ | Pre↑ | Rec↑ | #Params | #Steps | Time |
|-------|-------|-------|-----|------|------|---------|--------|------|
| VAR-d16 | 0.9 | 3.55 | 274.4 | 0.84 | 0.51 | 310M | 10 | 1× |
| VAR-d20 | 0.9 | 2.95 | 302.6 | 0.83 | 0.56 | 600M | 10 | 1.25× |
| **NFIG (Ours)** | **0.85** | **2.81** | **332.42** | 0.77 | 0.59 | 310M | 10 | **1×** |

### Scaling Results (55 epochs)

| Model | FID↓ | IS↑ | Precision↑ | Recall↑ |
|-------|------|-----|-----------|---------|
| NFIG-310M | 5.47 | 224.20 | 0.7569 | 0.4914 |
| NFIG-600M | 5.07 | 225.16 | 0.7184 | 0.5546 |

### Frequency Keep Score (FKS)

| Model | PSD↓ | FKS↑ | Low↑ | Middle↑ | High↑ |
|-------|------|------|------|---------|-------|
| VAR-16 | 0.87 | 79.5% | 98.3% | 57.6% | 48.2% |
| NFIG | 0.47 | 87.6% | 98.9% | 75.3% | 66.7% |

## Repository Structure

```
.
├── tokenizer/
│   ├── __init__.py
│   └── fr_vae.py          # FR-VAE: Frequency-guided Residual-quantized VAE
├── models/
│   ├── __init__.py
│   └── nfig_transformer.py # NFIG Transformer with block-wise causal attention
├── utils/
│   ├── __init__.py
│   └── frequency_analysis.py # Frequency analysis utilities (PSD, FKS)
├── scripts/
│   ├── train_fr_vae.py    # Training script for FR-VAE tokenizer
│   ├── train_nfig.py      # Training script for NFIG Transformer
│   └── evaluate.py        # Evaluation and image generation script
├── configs/
│   ├── fr_vae.json        # FR-VAE configuration
│   ├── nfig_310m.json     # NFIG-310M configuration
│   └── nfig_600m.json     # NFIG-600M configuration
└── README.md
```

## Usage

### Step 1: Train FR-VAE Tokenizer

```bash
python scripts/train_fr_vae.py \
    --data-path /path/to/imagenet \
    --output-dir output/fr_vae \
    --image-size 256 \
    --batch-size 8 \
    --epochs 100 \
    --lr 1e-4 \
    --codebook-size 4096
```

### Step 2: Train NFIG Transformer

```bash
python scripts/train_nfig.py \
    --data-path /path/to/imagenet \
    --tokenizer-path output/fr_vae/fr_vae_final.pt \
    --output-dir output/nfig \
    --model-size 310m \
    --batch-size 768 \
    --epochs 350 \
    --lr 8e-5
```

### Step 3: Generate Images

```bash
python scripts/evaluate.py \
    --tokenizer-path output/fr_vae/fr_vae_final.pt \
    --model-path output/nfig/nfig_final.pt \
    --output-dir output/generated \
    --n-samples 50000 \
    --cfg-scale 4.5 \
    --top-k 990 \
    --save-images
```

## Implementation Notes

### Token Count Calculation

The paper uses `scale_factors = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]` which represent the spatial resolution (h_i, w_i) of each frequency band's token map. Band i has `scale_factors[i]²` tokens:
- Band 0: 1×1 = 1 token (lowest frequency, global structure)
- Band 9: 16×16 = 256 tokens (highest frequency, fine details)
- Total: 680 tokens

### Frequency Band Division

The frequency band boundaries are proportional to the token counts:
```python
sigma_i = sigma_{i-1} + (h_i * w_i) / sum(h_j * w_j) * sigma_max
```

This ensures that bands with more tokens (higher frequency) cover a larger portion of the frequency spectrum.

### Residual Quantization

The residual quantization scheme captures information progressively:
1. Band 0: Quantize the lowest-frequency component directly
2. Band i (i>0): Quantize the residual between the accumulated signal and the current band

This ensures that each band captures the information not already represented by lower-frequency bands.

### Classifier-Free Guidance (CFG)

During training, class labels are randomly dropped with probability 0.1 (replaced with an unconditional token). During inference, CFG is applied:
```
logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
```
with `cfg_scale=4.5` and `top_k=990`.

## Assumptions and Unresolved Details

1. **Encoder architecture**: The paper mentions using DINOv2-base pretrained weights for the encoder. Our implementation uses a standard CNN encoder. For best results, initialize the encoder with DINOv2-base weights.

2. **DINO discriminator**: The paper uses a DINO-based discriminator (from VAR's tokenizer). Our implementation uses a standard PatchGAN discriminator. The DINO discriminator would likely improve rFID.

3. **Exact encoder downsampling**: The paper doesn't specify the exact encoder architecture. We use 4 stride-2 convolutions for 16× downsampling (256→16).

4. **Training infrastructure**: The paper uses NVIDIA H100 GPUs. Training details like gradient clipping, weight decay, and exact LR schedule are not fully specified.

5. **Token count interpretation**: The paper states "vocabulary size of 680 tokens" with scale_factors [1,2,3,4,5,6,8,10,13,16]. We interpret these as spatial resolutions (h_i=w_i=scale_factors[i]), giving token counts [1,4,9,16,25,36,64,100,169,256]=680.

## Dependencies

```
torch>=2.0
torchvision
numpy
```

Optional for evaluation:
```
torch-fidelity  # for FID/IS computation
lpips           # for LPIPS perceptual loss
```
