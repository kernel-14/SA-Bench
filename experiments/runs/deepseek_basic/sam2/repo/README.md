# SAM 2: Segment Anything in Images and Videos

A reproduction of the SAM 2 model from the paper ["SAM 2: Segment Anything in Images and Videos"](https://arxiv.org/abs/2408.00714) by Meta FAIR.

## Repository Structure

```
sam2/
├── model/                      # Model architecture
│   ├── image_encoder.py        # Hiera-based image encoder with FPN
│   ├── memory_attention.py     # Memory attention module (L transformer blocks)
│   ├── memory_encoder.py       # Memory encoder (mask + embedding fusion)
│   ├── memory_bank.py          # FIFO memory bank (N recent + M prompted frames)
│   ├── mask_decoder.py         # Mask decoder with skip connections + occlusion head
│   ├── prompt_encoder.py       # Prompt encoder (clicks, boxes, masks)
│   └── sam2.py                 # Full SAM 2 model
├── training/                   # Training utilities
│   ├── trainer.py              # SAM2Trainer with AdamW + reciprocal sqrt LR
│   ├── losses.py               # Focal + Dice + L1 + CE losses (20:1:1:1 ratio)
│   └── interactive_sampler.py  # Interactive prompt simulation for training
├── eval/                       # Evaluation utilities
│   ├── metrics.py              # J&F, mIoU metrics
│   └── evaluator.py            # VOS, interactive, and image evaluators
├── data/                       # Data handling
│   ├── dataset.py              # Image and video dataset classes
│   ├── transforms.py           # Augmentations (hflip, affine, jitter, mosaic)
│   └── data_engine.py          # Data engine (3 phases) + SA-V dataset spec
configs/
└── sam2_config.yaml            # Full configuration with hyperparameters
scripts/
├── train.py                    # Training script (pretrain/full/finetune)
└── eval.py                     # Evaluation script
```

## Architecture Overview

SAM 2 extends SAM to video by introducing a **streaming memory architecture**:

1. **Image Encoder** (Hiera): MAE-pretrained hierarchical vision transformer that produces multi-scale features. FPN fuses stride 16/32 features from stages 3/4. Stride 4/8 features bypass memory attention as skip connections to the mask decoder.

2. **Memory Attention**: Stacks L=4 transformer blocks, each performing self-attention (with 2D-RoPE) followed by cross-attention to memory features and object pointers, then an MLP. Conditions current frame features on past observations.

3. **Memory Bank**: Maintains two FIFO queues:
   - Up to N=6 recent frame memories (with temporal position embeddings)
   - Up to M prompted frame memories (without temporal position embeddings)
   - Object pointers (256-dim tokens from mask decoder, split into 4×64-dim)

4. **Memory Encoder**: Downsamples the output mask via convolution, sums element-wise with the unconditioned image embedding, then fuses via lightweight convolutions. Output projected to 64-dim.

5. **Mask Decoder**: Follows SAM's two-way transformer design with additions:
   - Skip connections from stride 4/8 features (bypass memory attention)
   - Occlusion prediction head (predicts if object is visible)
   - Object pointer token stored in memory bank

6. **Prompt Encoder**: Identical to SAM. Encodes clicks (positive/negative), boxes, and mask prompts.

## Training

### Pre-training (on SA-1B)
- AdamW optimizer, β=(0.9, 0.999), weight decay 0.1
- Reciprocal square-root LR schedule (timescale=1000)
- Linear warmup (1k steps) and cooldown (5k steps)  
- Layer-wise decay: 0.8 (T,S), 0.9 (B+), 0.925 (L)
- Batch size 256, ~90k steps, 1024² resolution
- bfloat16 precision, gradient clipping (l2, max 0.1)

### Full Training (mixed data)
- 8-frame sequences, up to 2 prompted frames
- Interactive simulation: mask (50%), click (25%), box (25%) initial prompts
- 7 correction clicks, mosaic transform (10%), reverse (50%)
- Alternating image/video batches proportional to data source sizes
- 200k steps, batch size 128

### Fine-tuning
- 16-frame sequences on most-edited 50% of masklets
- 50k steps, half learning rate, frozen image encoder

## Losses
- Mask: focal loss (weight 20) + dice loss (weight 1)
- IoU: L1/MAE loss (weight 1)  
- Object presence: cross-entropy (weight 1)
- Multi-mask: supervise IoU of ALL masks; only supervise mask with lowest segmentation loss

## Evaluation Protocols

### Semi-supervised VOS
Prompts only on first frame. Evaluated on 17 zero-shot datasets.

### Interactive (Offline)
Multiple passes: select frame with lowest IoU for correction. 3 clicks per frame.

### Interactive (Online)
Single forward pass: pause at IoU < 0.75 for corrections. 3 clicks per frame.

### Image Segmentation
1-click and 5-click mIoU on 37 zero-shot datasets.

## Model Sizes

| Encoder | Global Attn Blocks | Drop Path | Speed (FPS) |
|---------|-------------------|-----------|-------------|
| Hiera-T | 5, 7, 9           | 0.1       | -           |
| Hiera-S | 7, 10, 13         | 0.1       | -           |
| Hiera-B+| 12, 16, 20        | 0.2       | 43.8 (VOS)  |
| Hiera-L | 23, 33, 43        | 0.3       | 30.2 (VOS)  |

Image FPS (single A100): SAM 2 (Hiera-B+): 130.1, SAM 2 (Hiera-L): 61.4

## Key Design Choices (from Ablations)

- 2D-RoPE in memory attention (but not in image encoder)
- No relative positional bias (RPB) in image encoder → enables FlashAttention-2
- No GRU for memory → direct FIFO storage is simpler and effective
- Object pointers beneficial for SA-V and long-term tracking (LVOS)
- Memory features stored at 64-dim (4× smaller than 256-dim)

## Data Engine (SA-V Dataset)

- Phase 1: SAM per frame (37.8 s/frame, 16K masklets)
- Phase 2: SAM + SAM 2 Mask (7.4 s/frame, 63.5K masklets)
- Phase 3: Full SAM 2 (4.5 s/frame, 197.0K masklets)
- Auto masklets: 451.7K via grid prompting + verification
- Total: 50.9K videos, 642.6K masklets, 35.5M masks

## What's Implemented

✅ Complete model architecture (all components)
✅ Full training pipeline with correct hyperparameters
✅ Interactive prompt sampling for training
✅ Loss functions matching paper specifications
✅ Evaluation protocols (VOS, interactive offline/online, image)
✅ Data engine specification and SA-V dataset stats
✅ Data augmentations including mosaic transform
✅ Configurations for all model sizes (T, S, B+, L)

## What's Partially Covered / Placeholder

⚠️ Actual data loading (requires dataset files)
⚠️ Hiera pretrained weight initialization (requires MAE pretrained weights)
⚠️ Full training/evaluation execution (static-only benchmark)

## Unresolved Details from Paper

- Exact Hiera architecture details (kernel sizes, normalization types at each layer)
- Precise memory attention implementation details for FlashAttention-2 compatibility
- SA-V dataset is not publicly available for this reproduction
- Internal dataset details are not specified
- DAVIS interactive benchmark server access for evaluation

## Usage

```python
from sam2.model import SAM2

# Build model
model = SAM2(encoder_size="base_plus", img_size=1024)

# Single frame inference
output = model(
    frame=image_tensor,
    points=(point_coords, point_labels),
    multimask_output=True,
    is_first_frame=True,
)

# Video inference
results = model.process_video(frames, prompts)
```

## References

Kirillov et al. "Segment Anything." ICCV 2023.
Ravi et al. "SAM 2: Segment Anything in Images and Videos." arXiv 2024.
