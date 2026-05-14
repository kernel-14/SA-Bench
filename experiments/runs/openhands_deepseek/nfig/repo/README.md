# NFIG: Next-Frequency Image Generation

Reproduction of "Multi-Scale Autoregressive Image Generation via Frequency Ordering"
by Huang et al.

## Overview

NFIG is a novel autoregressive framework that decomposes image generation into
frequency-guided stages. The key insight is that low-frequency components encode
global structure with fewer tokens, while high-frequency components capture
local details requiring more tokens. By progressively generating from low to high
frequencies, NFIG achieves SOTA performance (FID: 2.81 on ImageNet-256) with
1.25× speedup vs VAR-d20.

## Code Structure

```
nfig/
├── config.py              # All hyperparameters and configuration
├── data.py                # ImageNet data loading and preprocessing
├── models/
│   ├── __init__.py
│   ├── fr_vae.py          # FR-VAE tokenizer (encoder, decoder, frequency ops)
│   ├── frequency_ops.py   # Frequency decomposer, composer, residual quantizer
│   ├── transformer.py     # NFIG autoregressive transformer with AdaLN
│   └── discriminator.py   # DINO discriminator for VQGAN training
├── utils/
│   ├── __init__.py
│   ├── metrics.py         # FID, IS, Precision, Recall evaluation
│   └── setup.py           # Training utilities, checkpointing, EMA
├── train_vae.py           # Train FR-VAE tokenizer
├── train_transformer.py   # Train NFIG transformer
├── inference.py           # Generate images from trained model
├── evaluate.py            # Comprehensive evaluation suite
├── ablation.py            # Ablation study (Table 5)
├── requirements.txt
└── README.md
```

## Key Components

### FR-VAE (Frequency-guided Residual-quantized VAE)
- Encodes images into 16×16 feature maps
- FrequencyDecomposer splits features into 10 frequency bands via FFT masking
- FrequencyResidualQuantizer: residual quantization across scales [1,2,3,4,5,6,8,10,13,16]
- Total tokens: 680 (sum of tokens across all bands)
- Codebook size: 4096

### NFIG Transformer
- Predicts tokens for each frequency band sequentially (low→high)
- Block-wise causal attention: band i can attend to bands 1..i
- AdaLN: class-conditional generation via adaptive layer normalization
- 16 layers, 1024 hidden dim, 16 attention heads
- 310M parameters for the base model

### Loss Functions
- FR-VAE: L = ||I - Î||² + ||f - f̂||² + Lp(I) + 0.5·Lg(I)
- Transformer: Cross-entropy loss across all frequency tokens

## Usage

### 1. Train FR-VAE Tokenizer
```bash
python train_vae.py --data_path /path/to/ImageNet --output_dir ./checkpoints
```

### 2. Train NFIG Transformer
```bash
python train_transformer.py \
    --data_path /path/to/ImageNet \
    --vae_checkpoint ./checkpoints/fr_vae_best.pt \
    --output_dir ./checkpoints
```

### 3. Generate Images
```bash
python inference.py \
    --vae_checkpoint ./checkpoints/fr_vae_best.pt \
    --transformer_checkpoint ./checkpoints/nfig_transformer_best.pt \
    --class_id 0 1 2 3 4 \
    --num_samples 16 \
    --cfg_scale 4.5 \
    --top_k 990 \
    --output generated.png
```

### 4. Evaluate
```bash
python evaluate.py \
    --vae_checkpoint ./checkpoints/fr_vae_best.pt \
    --transformer_checkpoint ./checkpoints/nfig_transformer_best.pt \
    --data_path /path/to/ImageNet \
    --num_generated 50000
```

### 5. Ablation Study
```bash
python ablation.py --data_path /path/to/ImageNet
```

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image size | 256×256 |
| Feature map size | 16×16 |
| Frequency bands | 10 |
| Scale factors | [1,2,3,4,5,6,8,10,13,16] |
| Total tokens | 680 |
| Codebook size | 4096 |
| Transformer layers | 16 |
| Hidden dim | 1024 |
| Learning rate | 8e-5 |
| Batch size | 768 |
| Epochs | 350 |
| CFG scale (inference) | 4.5 |
| Top-k (inference) | 990 |

## Paper Results (ImageNet-256)

| Metric | Value |
|--------|-------|
| rFID | 0.85 |
| gFID | 2.81 |
| IS | 332.42 |
| Precision | 0.77 |
| Recall | 0.59 |
| #Params | 310M |
| #Steps | 10 |
| Speedup vs VAR-d20 | 1.25× |

## Requirements

- PyTorch >= 2.0
- torchvision
- numpy, scipy
- einops
- tqdm
- 8× NVIDIA H100 GPUs (for full training)
