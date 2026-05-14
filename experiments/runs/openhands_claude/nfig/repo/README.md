# NFIG: Next-Frequency Image Generation

Implementation of **NFIG: Multi-Scale Autoregressive Image Generation via Frequency Ordering** (NeurIPS 2025).

NFIG decomposes image generation into frequency-guided stages: first generating low-frequency components (global structure) with fewer tokens, then progressively adding higher-frequency details. This achieves FID 2.81 on ImageNet 256×256 with a 1.25× speedup over VAR-d20.

## Repository Structure

```
repo/
├── config.py               # All hyperparameters and model configurations
├── data.py                 # ImageNet dataset loading and preprocessing
├── losses.py               # FR-VAE loss (reconstruction + perceptual + GAN) and transformer CE loss
├── train_tokenizer.py      # FR-VAE training script
├── train_generator.py      # NFIG transformer training script
├── generate.py             # Image generation with CFG and top-k sampling
├── evaluate.py             # FID, IS, Precision, Recall evaluation
├── requirements.txt
├── models/
│   ├── fr_vae.py           # Frequency-guided Residual-quantized VAE
│   ├── quantizer.py        # VectorQuantizer and FrequencyResidualQuantizer
│   ├── transformer.py      # NFIGTransformer with block-wise causal attention
│   └── discriminator.py    # DINO + PatchGAN discriminator
└── modules/
    ├── frequency.py        # FFT-based FrequencyDecomposer and FrequencyComposer
    └── layers.py           # Encoder, Decoder, AdaLN, ResidualBlock, FeedForward
```

## Key Components

### FR-VAE (Frequency-guided Residual-quantized VAE)

The tokenizer encodes images into multi-scale frequency token sequences:

1. **Encoder** — DINOv2-base initialized VQGAN encoder, downsamples 256×256 → 16×16 feature map
2. **FrequencyDecomposer** — Decomposes feature map into n frequency bands via FFT masking (Eq. 1)
3. **FrequencyResidualQuantizer** — Residual quantization across frequency levels with shared codebook (Eq. 3)
4. **FrequencyComposer** — Reconstructs feature map by summing upsampled quantized components (Eq. 2)
5. **Decoder** — VQGAN decoder reconstructs image from feature map

Scaling factors `[1, 2, 3, 4, 5, 6, 8, 10, 13, 16]` produce token grids of sizes 1×1 through 16×16, totaling **680 tokens** with codebook size **4096**.

### NFIG Transformer

Decoder-only transformer with block-wise causal attention:
- Tokens in frequency band i can attend to all tokens in bands 1..i (full attention within band)
- AdaLN class conditioning with classifier-free guidance (CFG)
- Autoregressive generation from low to high frequency bands

Model sizes:
- **310M**: depth=16, embed_dim=1024, num_heads=16
- **600M**: depth=20, embed_dim=1152, num_heads=16

### Frequency Band Division

Frequency bands are divided proportionally to token count (Eq. 6):

```
σ_i = σ_{i-1} + (h_i * w_i / Σ h_j * w_j) * σ_max
```

## Training

### Step 1: Train FR-VAE Tokenizer

```bash
python train_tokenizer.py \
    --data-root /data/imagenet \
    --checkpoint-dir ./checkpoints/tokenizer \
    --batch-size 8 \
    --num-epochs 20 \
    --lr 4.5e-6 \
    --disc-start 50001 \
    --use-dino-disc \
    --use-amp
```

### Step 2: Train NFIG Transformer

```bash
python train_generator.py \
    --data-root /data/imagenet \
    --tokenizer-ckpt ./checkpoints/tokenizer/frvae_final.pt \
    --checkpoint-dir ./checkpoints/transformer \
    --model-size 310M \
    --batch-size 768 \
    --num-epochs 350 \
    --lr 8e-5 \
    --use-amp
```

## Generation

```bash
python generate.py \
    --tokenizer-ckpt ./checkpoints/tokenizer/frvae_final.pt \
    --transformer-ckpt ./checkpoints/transformer/nfig_310M_final.pt \
    --output-dir ./generated \
    --cfg-scale 4.5 \
    --top-k 990 \
    --num-samples-per-class 50 \
    --save-grid
```

## Evaluation

```bash
# Reconstruction FID (tokenizer quality)
python evaluate.py \
    --mode rfid \
    --tokenizer-ckpt ./checkpoints/tokenizer/frvae_final.pt \
    --data-root /data/imagenet

# Generation FID + IS + Precision + Recall
python evaluate.py \
    --mode gfid \
    --tokenizer-ckpt ./checkpoints/tokenizer/frvae_final.pt \
    --transformer-ckpt ./checkpoints/transformer/nfig_310M_final.pt \
    --data-root /data/imagenet \
    --num-samples 50000 \
    --cfg-scale 4.5 \
    --top-k 990
```

## Results (ImageNet 256×256)

| Model | rFID↓ | gFID↓ | IS↑ | Pre↑ | Rec↑ | Params | Steps | Time |
|-------|-------|-------|-----|------|------|--------|-------|------|
| VAR-d16 | 0.9 | 3.55 | 274.4 | 0.84 | 0.51 | 310M | 10 | 1× |
| VAR-d20 | 0.9 | 2.95 | 302.6 | 0.83 | 0.56 | 600M | 10 | 1.25× |
| **NFIG (ours)** | **0.85** | **2.81** | **332.42** | 0.77 | 0.59 | 310M | 10 | **1×** |

## Hyperparameters

All hyperparameters are in `config.py`. Key settings from the paper:

| Parameter | Value |
|-----------|-------|
| Scale factors | [1, 2, 3, 4, 5, 6, 8, 10, 13, 16] |
| Total tokens | 680 |
| Codebook size | 4096 |
| Transformer depth (310M) | 16 |
| Transformer depth (600M) | 20 |
| Training epochs | 350 |
| Batch size | 768 |
| Learning rate | 8×10⁻⁵ |
| CFG scale (inference) | 4.5 |
| Top-k (inference) | 990 |
