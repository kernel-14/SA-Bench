# SAM 2: Segment Anything in Images and Videos

Reproduction of "SAM 2: Segment Anything in Images and Videos" (Ravi et al., 2024).

## Structure

```
repo/
├── config.py               # All hyperparameters and model configurations
├── train.py                # Training loop (pre-training + full training + fine-tuning)
├── evaluate.py             # Evaluation: J&F for video, mIoU for images
├── requirements.txt
├── model/
│   ├── sam2.py             # Main SAM 2 model with memory bank
│   ├── image_encoder.py    # Hiera hierarchical ViT + FPN
│   ├── memory_attention.py # Memory attention (L=4 transformer blocks)
│   ├── prompt_encoder.py   # Prompt encoder (clicks, boxes, masks)
│   ├── mask_decoder.py     # Two-way transformer decoder + occlusion head
│   ├── memory_encoder.py   # Convolutional memory encoder
│   └── layers.py           # Shared layers: attention, MLP, 2d-RoPE, positional encodings
├── data/
│   ├── dataset.py          # SA-V, VOS, SA-1B dataset classes
│   ├── transforms.py       # Augmentations: hflip, affine, color jitter, mosaic
│   └── utils.py            # Prompt simulation, click sampling
└── utils/
    ├── metrics.py          # J&F (Jaccard & F-measure), mIoU computation
    └── misc.py             # Misc utilities: checkpointing, logging, layer decay
```

## Key Components

**Image Encoder**: MAE pre-trained Hiera (T/S/B+/L variants) with windowed absolute positional
embeddings. FPN fuses stride-16 and stride-32 features for the frame embedding; stride-4 and
stride-8 features are passed as skip connections to the mask decoder.

**Memory Attention**: L=4 transformer blocks. Each block performs self-attention on the current
frame features, then cross-attention to the memory bank (spatial features + object pointer
vectors), followed by an MLP. Uses 2d-RoPE in self- and cross-attention.

**Prompt Encoder**: Identical to SAM. Sparse prompts (clicks, boxes) → positional encodings +
learned type embeddings. Dense prompts (masks) → convolutional embedding summed with frame
embedding.

**Mask Decoder**: Two-way transformer blocks updating prompt and frame tokens. Predicts multiple
masks for ambiguous prompts; selects highest-IoU mask for propagation. Includes an occlusion
prediction head (object presence score).

**Memory Encoder**: Downsamples the predicted mask with a conv module, sums element-wise with
the unconditioned image embedding, then applies light-weight conv layers to produce a 64-dim
spatial memory feature.

**Memory Bank**: FIFO queue of N=6 recent frame memories + FIFO queue of M prompted frame
memories. Temporal position embeddings applied only to recent frames. Object pointer vectors
(256-dim split into 4×64-dim tokens) stored alongside spatial memories.

## Training

- **Pre-training**: SA-1B, ~90k steps, batch 256, resolution 1024, AdamW + reciprocal-sqrt LR.
- **Full training**: SA-V + Internal + 10% SA-1B + VOS datasets, 200k steps, 8-frame sequences.
- **Fine-tuning**: 16-frame sequences on challenging videos, 50k steps at half LR.
- **Losses**: focal (×20) + dice (×1) for masks, L1 (×1) for IoU, cross-entropy (×1) for occlusion.
- **Prompt simulation**: GT mask (50%), positive click (25%), bounding box (25%) as initial prompt.

## Evaluation

- **Video**: J&F metric; interactive offline/online settings; semi-supervised VOS (mask on frame 1).
- **Image**: mIoU with 1-click and 5-click prompts on 23/37 zero-shot datasets.
