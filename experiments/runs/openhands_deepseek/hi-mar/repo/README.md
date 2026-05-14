# Hi-MAR: Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots

Reproduction of the paper "Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots" by Guangting Zheng et al.

## Codebase Structure

```
repo/
├── config.py              # Model architectures (B/L/H) and training hyperparameters
├── requirements.txt       # Python dependencies
├── models/
│   ├── layers.py          # AdaLN, AdaLN-Zero, sinusoidal embeddings, self-attention, MLP
│   ├── transformer.py     # Scale-Aware Transformer Block, HiMARTransformer, MLP/DiffTransformer heads
│   └── hi_mar.py          # Main Hi-MAR model (two-phase training, inference/decoding, CFG)
├── data/
│   └── dataset.py         # ImageNet/MS-COCO datasets, VAE encoder wrapper, text embedder
├── training/
│   ├── trainer.py         # Trainer class, mask ratio samplers, EMA, LR scheduler
│   └── train.py           # Main training/generation script with argparse
├── evaluation/
│   └── metrics.py         # FID, IS, Precision/Recall computation
└── utils/
    └── diffusion.py       # Noise scheduler, mask schedules, iterative decoding helpers
```

## Key Design Components

1. **Scale-Aware Transformer Block** (Figure 2c): AdaLN-Zero modulation with scale sinusoidal embedding — same weights used for both phases with scale-index differentiation.

2. **Two-Phase Architecture** (Figure 2b):
   - Phase 1: Low-res (16×16) masked tokens → Transformer → MLP Diffusion Head → Z^s pivots
   - Phase 2: [Z^s, high-res (32×32) masked tokens] → Transformer → Diffusion Transformer Head (with self-attention across all tokens) → final prediction

3. **Diffusion Transformer Head** (Figure 2e): Self-attention among all tokens during denoising, replacing per-token MLP head for global context mining.

4. **Classifier-Free Guidance**: Applied in phase 2 only during inference.

## Model Variants (Table 1)

| Model    | Transformer Layers | Hidden Dim | Head1 Layers | Head1 Hidden | Head2 Layers | Head2 Hidden | Params |
|----------|-------------------|------------|-------------|-------------|-------------|-------------|--------|
| Hi-MAR-B | 24                | 768        | 6           | 1024        | 6           | 512         | 244M   |
| Hi-MAR-L | 32                | 1024       | 8           | 1280        | 8           | 512         | 529M   |
| Hi-MAR-H | 40                | 1280       | 12          | 1536        | 12          | 768         | 1090M  |

## Usage

```bash
# Training on ImageNet
python training/train.py --mode train --dataset imagenet \
    --data_root /path/to/imagenet --vae_path /path/to/kl16_vae \
    --model_size Hi-MAR-B --batch_size 256 --epochs 800

# Training on MS-COCO
python training/train.py --mode train --dataset coco \
    --data_root /path/to/coco/images --ann_file /path/to/coco/annotations.json \
    --vae_path /path/to/kl16_vae --model_size Hi-MAR-B --lr 8e-4 --weight_decay 0.03

# Generate samples
python training/train.py --mode generate --dataset imagenet \
    --checkpoint ./checkpoints/checkpoint_epoch_800.pt --cfg_scale 3.0
```

## Key Hyperparameters

- **ImageNet**: lr=1e-4, weight_decay=0.02, 800 epochs, 100-epoch warmup, phase2=cosine masking
- **MS-COCO**: lr=8e-4, weight_decay=0.03, Beta(4, 1) masking for phase2
- **Inference**: 32 steps phase 1, 4 steps phase 2, cosine unmasking schedule
- **VAE**: KL-16 from MAR (16-channel latent, 8x downsampling)
