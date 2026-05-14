# Hi-MAR: Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots

Implementation of [Hi-MAR](https://arxiv.org/abs/...) — a hierarchical masked autoregressive model for image generation that pivots on low-resolution tokens to provide global structural context for high-resolution generation.

## Structure

```
repo/
├── layers.py       # Basic building blocks: attention, FFN, sinusoidal embeddings, AdaLN-Zero
├── modules.py      # Hi-MAR modules: ScaleAwareTransformerBlock, MLPDiffusionHead,
│                   #   DiffusionTransformerHead, GaussianDiffusion
├── model.py        # Full Hi-MAR model (HiMAR, HiMARTransformer) and model factory
├── config.py       # All hyperparameters (ModelConfig, TrainConfig, InferenceConfig)
├── data.py         # ImageNet and MS-COCO datasets, VAETokenizer, masking utilities
├── train.py        # Training loop with DDP, AMP, EMA support
├── evaluate.py     # FID, IS, Precision/Recall evaluation
└── requirements.txt
```

## Method

Hi-MAR extends MAR with two key contributions:

**1. Hierarchical Masked Autoregressive Transformer (§3.2)**

Two-phase generation using a shared scale-aware Transformer backbone:
- Phase 1: bidirectional masked autoregressive modeling over low-resolution tokens (128×128 → 8×8 latent = 64 tokens). The output conditional tokens Z^s capture global structure.
- Phase 2: masked autoregressive modeling over high-resolution tokens (256×256 → 16×16 latent = 256 tokens), conditioned on Z^s as intermediary pivots. Using conditional tokens (not ground-truth visual tokens) eliminates the training-inference discrepancy.

Scale awareness is injected via sinusoidal scale embeddings through AdaLN-Zero operations in every Transformer block.

**2. Diffusion Transformer Head (§3.3)**

Phase 2 uses a Diffusion Transformer head instead of the per-token MLP head. It processes all tokens jointly via self-attention, capturing inter-token dependencies during denoising. Context per token: `c = time_embedding + conditional_token`.

## Architecture (Table 1)

| Model     | Transformer Layers | Hidden | DiffHead1 (layers/hidden) | DiffHead2 (layers/hidden) | Params |
|-----------|--------------------|--------|---------------------------|---------------------------|--------|
| Hi-MAR-B  | 24                 | 768    | 6 / 1024                  | 6 / 512                   | 244M   |
| Hi-MAR-L  | 32                 | 1024   | 8 / 1280                  | 8 / 512                   | 529M   |
| Hi-MAR-H  | 40                 | 1280   | 12 / 1536                 | 12 / 768                  | 1090M  |
| Hi-MAR-S  | 16                 | 512    | 4 / 768                   | 4 / 384                   | ~100M  |

## Training

**ImageNet 256×256 (class-conditional)**
```bash
# Single GPU
python train.py --dataset imagenet --model Hi-MAR-B --data_path /data/imagenet

# 8 GPUs
torchrun --nproc_per_node=8 train.py --dataset imagenet --model Hi-MAR-B \
    --data_path /data/imagenet --batch_size 32
```

Key hyperparameters (from paper §4.2):
- Optimizer: AdamW, β1=0.9, β2=0.95, weight_decay=0.02
- LR: 1e-4 with 100-epoch linear warmup, constant thereafter
- Epochs: 800
- Phase 1 masking: uniform in [0.7, 1.0]
- Phase 2 masking: cosine schedule (MaskGIT)

**MS-COCO 256×256 (text-to-image)**
```bash
python train.py --dataset coco --model Hi-MAR-S \
    --data_path /data/coco --coco_ann_path /data/coco/annotations
```

Key hyperparameters:
- Optimizer: AdamW, lr=8e-4, weight_decay=0.03, 8K-step warmup
- Phase 2 masking: Beta(α=4, β=1)
- EMA momentum: 0.9999

## Inference

```bash
python evaluate.py --checkpoint outputs/checkpoint_epoch0799.pt \
    --dataset imagenet --model Hi-MAR-B \
    --data_path /data/imagenet \
    --steps_phase1 32 --steps_phase2 4 \
    --cfg_scale 1.5 --num_samples 50000
```

Default inference settings (from paper §4.5):
- Phase 1: 32 autoregressive steps
- Phase 2: 4 autoregressive steps
- CFG scale: 1.5 (for w/o CFG: phase 2 CFG disabled)

## Results (Table 2, ImageNet 256×256)

| Model    | FID (w/o CFG) | FID (w/ CFG) | IS (w/ CFG) |
|----------|---------------|--------------|-------------|
| Hi-MAR-B | 2.11          | 1.93         | 293.0       |
| Hi-MAR-L | 1.72          | 1.66         | 322.3       |
| Hi-MAR-H | 1.55          | 1.52         | 322.78      |

## Dependencies

```bash
pip install -r requirements.txt
```

The VAE tokenizer uses the KL-16 VAE from MAR. You can use `stabilityai/sd-vae-ft-ema` as a compatible alternative, or download the MAR VAE weights directly.
