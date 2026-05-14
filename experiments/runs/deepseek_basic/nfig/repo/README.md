# NFIG: Multi-Scale Autoregressive Image Generation via Frequency Ordering

This repository contains a reproduction of the paper **"NFIG: Multi-Scale Autoregressive Image Generation via Frequency Ordering"** by Zhihao Huang et al.

## Overview

NFIG (Next-Frequency Image Generation) is a novel autoregressive image generation framework that decomposes the image generation process into multiple frequency-guided stages. It first generates low-frequency components to capture global structure with fewer tokens, then progressively adds higher-frequency details.

Key components:
- **FR-VAE**: Frequency-guided Residual-quantized VAE for image tokenization
- **NFIG Transformer**: Autoregressive transformer with block-wise causal attention for next-frequency prediction

## Repository Structure

```
nfig/
├── __init__.py              # Package init
├── fr_vae.py                # FR-VAE tokenizer implementation
├── nfig_transformer.py      # NFIG autoregressive transformer
├── frequency_utils.py       # Frequency decomposition/composition utilities
├── evaluation.py            # Evaluation metrics (FID, IS, Precision, Recall)
└── trainer.py               # Training loops for FR-VAE and NFIG Transformer

scripts/
├── train_frvae.py           # Train FR-VAE tokenizer
├── train_nfig.py            # Train NFIG transformer
├── extract_tokens.py        # Extract tokens from images using FR-VAE
├── generate.py              # Generate images from trained model
└── evaluate.py              # Evaluate generation quality

configs/
├── frvae_imagenet.yaml      # FR-VAE configuration
└── nfig_transformer.yaml    # NFIG Transformer configuration
```

## Installation

```bash
git clone <repository>
cd nfig
pip install -r requirements.txt
```

## Architecture Details

### FR-VAE (Frequency-guided Residual-quantized VAE)

The image tokenizer (Section 3.1) consists of:

1. **Encoder**: Maps image x ∈ R^{H×W×3} → feature f ∈ R^{H'×W'×C}
   - CNN-based with DINOv2-base pretrained initialization
   - 4x downsampling (256→16)

2. **Frequency-guided Decomposer**: f → {f̂ᵢ} via FFT-based frequency band masking
   - f̂ᵢ = F⁻¹(F(f) ⊙ Mᵢ)

3. **Residual Quantizer**: Progressive quantization with residual learning
   - Low frequency bands use fewer tokens, high frequency bands use more
   - Shared codebook (K=4096, C=256)
   - 10 frequency bands with scales [1,2,3,4,5,6,8,10,13,16]

4. **Frequency-guided Composer**: Merges quantized components
   - f̃ = Σᵢ T(v_qᵢ, H', W')

5. **Decoder**: Reconstructs image from combined features
   - CNN-based with 4x upsampling

**Training Loss** (Appendix B.1):
```
L = ||I - Î||₂² + ||f - f̃||₂² + L_p(I) + 0.5·L_g(I)
```
where L_p is LPIPS perceptual loss and L_g is GAN loss with DINO discriminator.

### NFIG Transformer (Section 3.2)

The autoregressive generator:

1. **Architecture**: Decoder-only transformer with:
   - Depth: 16 blocks
   - Hidden dim: 768
   - 12 attention heads
   - AdaLN (Adaptive Layer Normalization) for class conditioning

2. **Block-wise Causal Attention**:
   - Tokens in band i attend to all tokens in bands < i (full attention)
   - Tokens in band i attend causally within band i

3. **Next-Frequency Prediction**:
   - p(T₁, T₂, ..., Tₙ) = ∏ᵢ p(Tᵢ | T₁, ..., Tᵢ₋₁)

4. **Inference**:
   - Top-k sampling (k=990)
   - Classifier-Free Guidance (CFG scale=4.5)
   - 10 generation steps (one per frequency band)

## Usage

### 1. Train FR-VAE Tokenizer

```bash
python scripts/train_frvae.py \
    --data_path /path/to/imagenet \
    --output_dir ./checkpoints \
    --batch_size 256 \
    --epochs 100 \
    --device cuda
```

### 2. Extract Tokens

```bash
python scripts/extract_tokens.py \
    --frvae_ckpt ./checkpoints/frvae_final.pt \
    --data_path /path/to/imagenet/train \
    --output_dir ./tokens \
    --batch_size 128
```

### 3. Train NFIG Transformer

```bash
python scripts/train_nfig.py \
    --token_data ./tokens \
    --output_dir ./checkpoints \
    --batch_size 768 \
    --epochs 350 \
    --lr 8e-5 \
    --device cuda
```

### 4. Generate Images

```bash
python scripts/generate.py \
    --frvae_ckpt ./checkpoints/frvae_final.pt \
    --nfig_ckpt ./checkpoints/nfig_final.pt \
    --class_id 0 \
    --num_images 16 \
    --output_dir ./generated \
    --top_k 990 \
    --cfg_scale 4.5
```

### 5. Evaluate

```bash
python scripts/evaluate.py \
    --frvae_ckpt ./checkpoints/frvae_final.pt \
    --nfig_ckpt ./checkpoints/nfig_final.pt \
    --real_data /path/to/imagenet/val \
    --num_samples 50000 \
    --output_dir ./eval_results
```

## Key Hyperparameters (from paper)

### FR-VAE
| Parameter | Value |
|-----------|-------|
| Codebook size (K) | 4096 |
| Codebook dimension (C) | 256 |
| Latent dimension | 256 |
| Scaling factors | [1, 2, 3, 4, 5, 6, 8, 10, 13, 16] |
| Total tokens per image | 680 |
| Reconstruction FID (rFID) | 0.85 |

### NFIG Transformer
| Parameter | Value |
|-----------|-------|
| Depth | 16 |
| Hidden dimension | 768 |
| Attention heads | 12 |
| Batch size | 768 |
| Learning rate | 8×10⁻⁵ |
| Training epochs | 350 |
| Top-k | 990 |
| CFG scale | 4.5 |

## Expected Results (from paper)

| Metric | Value |
|--------|-------|
| gFID | 2.81 |
| IS | 332.42 |
| Precision | 0.77 |
| Recall | 0.59 |
| Parameters | 310M |
| Generation steps | 10 |
| Relative inference time | 1.0× |

Compared to VAR-d20: 1.25× speedup while achieving better FID (2.81 vs 2.95).

## Implementation Notes

### What is Reproduced
- Full FR-VAE architecture with frequency-guided decomposition, residual quantization, and VQ-GAN framework
- NFIG Transformer with block-wise causal attention, AdaLN, CFG, and top-k sampling
- Training infrastructure for both components
- Evaluation pipeline (FID, IS, Precision, Recall)
- Frequency analysis tools (FKS, PSD)
- Configuration files matching paper specifications

### Assumptions and Simplifications
1. **Encoder initialization**: The paper mentions using DINOv2-base pretrained weights. Our implementation supports this but falls back to random initialization if DINOv2 is not available.
2. **DINO Discriminator**: We implement a CNN-based discriminator; the exact DINO discriminator architecture from VAR may differ.
3. **LPIPS loss**: Falls back to MSE if the `lpips` package is not installed.
4. **The training scripts are designed for single-GPU training**; the paper used multi-GPU H100 training.
5. **Block-wise causal attention**: For first-scale generation, we use causal token-by-token prediction. For subsequent scales, all tokens in the scale are predicted in parallel given previous scales.
6. **Frequency mask creation**: We implement radial frequency masks in the FFT domain as described, though exact mask design details were inferred.

### Unresolved Details
- Exact DINOv2 encoder weight loading and freezing strategy
- Specific DINO discriminator architecture from VAR
- Training schedule details (learning rate warmup, decay, etc.)
- Data augmentation specifics
- The paper's FR-VAE encodes images to feature maps; the exact spatial dimensions and relationship between FFT mask and spatial scales involved interpretation

### Differences from VAR
1. NFIG uses frequency-domain decomposition instead of spatial pyramid
2. Residual quantization across frequency bands instead of separate scales
3. Frequency band division based on token budget (Eq. in Section 3.2)
4. More balanced loss across scales compared to VAR

## License

This reproduction is provided for research purposes. Please cite the original paper:

```
@article{huang2024nfig,
  title={NFIG: Multi-Scale Autoregressive Image Generation via Frequency Ordering},
  author={Huang, Zhihao and Qiu, Xi and Ma, Yukuo and Zhou, Yifu and Chen, Junjie and Zhang, Hongyuan and Zhang, Chi and Li, Xuelong},
  year={2024}
}
```
