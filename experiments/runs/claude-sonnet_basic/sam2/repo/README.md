# SAM 2: Segment Anything in Images and Videos - Reproduction

This repository contains a reproduction of the SAM 2 paper:
> **SAM 2: Segment Anything in Images and Videos**  
> Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, et al.  
> Meta FAIR, 2024

## Overview

SAM 2 is a foundation model for promptable visual segmentation in both images and videos. It extends SAM (Segment Anything Model) to the video domain through a streaming memory architecture.

## Key Contributions Reproduced

### 1. Model Architecture (Section 4)

The SAM 2 model consists of:

- **Image Encoder** (`sam2/modeling/hiera_image_encoder.py`): MAE pre-trained Hiera hierarchical ViT
  - 4 stages with stride 4, 8, 16, 32
  - Stages 3 & 4 fused via FPN for main image embedding
  - Stages 1 & 2 used as skip connections in mask decoder
  - Window attention with global attention in subset of layers
  - No relative positional biases (RPB) for speed

- **Memory Attention** (`sam2/modeling/memory_attention.py`): L=4 transformer blocks
  - Self-attention on current frame features
  - Cross-attention to memory bank (spatial features + object pointers)
  - 2D Rotary Positional Embedding (RoPE) in attention layers
  - Temporal position encoding for recent frame memories

- **Prompt Encoder** (`sam2/modeling/prompt_encoder.py`): Identical to SAM
  - Supports points (positive/negative), boxes, and masks
  - Random Fourier feature positional encoding for points

- **Mask Decoder** (`sam2/modeling/mask_decoder.py`): Extended from SAM
  - Two-way transformer blocks
  - Multi-mask prediction for ambiguous prompts
  - **New**: Occlusion prediction head (predicts if object is visible)
  - **New**: Object pointer token (mask token used for memory bank)
  - **New**: Skip connections from stride 4 and 8 features

- **Memory Encoder** (`sam2/modeling/memory_encoder.py`):
  - Downsamples predicted mask to match image embedding size
  - Element-wise sum with image encoder features
  - Lightweight convolutional fusion layers
  - Projects to 64-dim memory features
  - Learned occlusion embedding for occluded frames

- **Memory Bank**: FIFO queue
  - N=6 recent frames (with temporal position encoding)
  - M=8 prompted frames (without temporal position encoding)
  - Object pointer vectors (256-dim split into 4×64-dim tokens)

### 2. Training Procedure (Section D.2)

**Pre-training** (`sam2/configs/pretrain_config.yaml`):
- SA-1B dataset, ~90k steps, batch size 256
- AdamW with layer-wise decay (0.9 for B+)
- Reciprocal square-root LR schedule
- L1 loss for IoU (vs BCE in SAM)
- 7 correction clicks (vs 8 in SAM)

**Full Training** (`sam2/configs/full_training_config.yaml`):
- SA-V + Internal + SA-1B + VOS datasets
- Alternating video/image batches
- 8-frame sequences, up to 2 prompted frames
- Interactive prompt simulation:
  - GT mask (50%), positive click (25%), bounding box (25%)
  - Corrective clicks from error region center
  - 10% random clicks from GT mask
- Mosaic transform (10% probability)
- Temporal reversal (50% probability)

**Fine-tuning** (`sam2/configs/finetune_config.yaml`):
- 16-frame sequences on challenging videos
- 50k iterations (1/3 of original)
- Half learning rate, frozen image encoder
- Top 50% most edited masklets

### 3. Loss Functions (`sam2/modeling/losses.py`)

Combined loss with ratio 20:1:1:1:
- **Focal loss** (weight 20): for mask prediction
- **Dice loss** (weight 1): for mask prediction
- **L1 loss** (weight 1): for IoU prediction (with sigmoid activation)
- **Cross-entropy** (weight 1): for occlusion prediction

Multi-mask handling:
- Only supervise mask with lowest segmentation loss
- Supervise IoU predictions of all masks
- Skip mask supervision for occluded frames (always supervise occlusion)

### 4. Evaluation Protocols (Section 6)

**Promptable Video Segmentation** (`evaluate.py`):
- Offline: multiple passes, select frame with lowest IoU
- Online: single pass, pause at IoU < 0.75
- 3 clicks per frame, up to 8 interacted frames
- J&F metric

**Semi-supervised VOS**:
- First-frame prompts: 1/3/5 clicks, box, or GT mask
- 17 zero-shot video datasets
- J&F metric

**Image Segmentation**:
- 37 zero-shot datasets (23 from SAM + 14 new video)
- 1-click and 5-click mIoU

## Model Sizes

| Size | Encoder | Global Attn Blocks | Drop Path |
|------|---------|-------------------|-----------|
| Tiny (T) | embed_dim=96, stages=(1,2,7,2) | 5,7,9 | 0.1 |
| Small (S) | embed_dim=96, stages=(1,2,11,2) | 7,10,13 | 0.1 |
| Base+ (B+) | embed_dim=112, stages=(2,3,16,3) | 12,16,20 | 0.2 |
| Large (L) | embed_dim=144, stages=(2,6,36,4) | 23,33,43 | 0.3 |

## Usage

### Installation

```bash
pip install -e .
```

### Build Model

```python
from sam2.modeling import build_sam2

# Build SAM 2 (Base+)
model = build_sam2(model_size='base_plus', image_size=1024)

# Build SAM 2 (Large)
model = build_sam2(model_size='large', image_size=1024)
```

### Image Segmentation

```python
import torch
from sam2.modeling import build_sam2

model = build_sam2('base_plus')
model.eval()

# Point prompt
img = torch.randn(1, 3, 1024, 1024)
coords = torch.tensor([[[512, 512]]], dtype=torch.float32)  # [B, N, 2]
labels = torch.tensor([[1]], dtype=torch.long)  # [B, N]

with torch.no_grad():
    outputs = model.forward_image(
        img=img,
        points=(coords, labels),
        multimask_output=True,
    )

masks = outputs['masks']  # [B, num_masks, H, W]
iou_pred = outputs['iou_predictions']  # [B, num_masks]
```

### Video Segmentation

```python
import torch
from sam2.modeling import build_sam2

model = build_sam2('base_plus')
model.eval()

# Initialize memory bank
memory_bank = {
    'recent_feats': [],
    'recent_pos': [],
    'prompted_feats': [],
    'prompted_pos': [],
    'object_ptrs': [],
}

# Process video frame by frame
for t, frame in enumerate(video_frames):
    frame_tensor = preprocess(frame)  # [1, 3, H, W]
    
    # Add prompts on first frame
    points = None
    if t == 0:
        coords = torch.tensor([[[cx, cy]]], dtype=torch.float32)
        labels = torch.tensor([[1]], dtype=torch.long)
        points = (coords, labels)
    
    with torch.no_grad():
        outputs, memory = model.forward_video_frame(
            img=frame_tensor,
            memory_bank=memory_bank if t > 0 else None,
            points=points,
        )
    
    # Update memory bank
    memory_bank['recent_feats'].append(memory)
    # ... (see evaluate.py for full memory bank update)
    
    pred_mask = outputs['masks']  # [1, 1, H, W]
```

### Training

```bash
# Pre-training on SA-1B
python train.py \
    --mode pretrain \
    --image_dir /path/to/sa_1b/images \
    --image_ann_dir /path/to/sa_1b/annotations \
    --model_size base_plus \
    --batch_size 256 \
    --lr 4e-4

# Full training
python train.py \
    --mode full \
    --video_dir /path/to/sa_v/videos \
    --video_ann_dir /path/to/sa_v/annotations \
    --image_dir /path/to/sa_1b/images \
    --image_ann_dir /path/to/sa_1b/annotations \
    --model_size base_plus \
    --batch_size 128

# Fine-tuning on 16-frame sequences
python train.py \
    --mode finetune \
    --video_dir /path/to/sa_v/videos \
    --video_ann_dir /path/to/sa_v/annotations \
    --model_size base_plus \
    --resume checkpoints/sam2_full.pth
```

### Evaluation

```bash
# Semi-supervised VOS evaluation
python evaluate.py \
    --checkpoint checkpoints/sam2_final.pth \
    --model_size base_plus \
    --eval_type semi_supervised \
    --data_dir /path/to/davis \
    --dataset davis \
    --prompt_type mask
```

## Repository Structure

```
sam2/
├── modeling/
│   ├── __init__.py
│   ├── sam2_model.py          # Main SAM 2 model
│   ├── hiera_image_encoder.py # Hiera hierarchical ViT encoder
│   ├── memory_attention.py    # Memory attention module
│   ├── memory_encoder.py      # Memory encoder
│   ├── mask_decoder.py        # Mask decoder with two-way transformer
│   ├── prompt_encoder.py      # Prompt encoder (points, boxes, masks)
│   └── losses.py              # Training losses
├── datasets/
│   ├── __init__.py
│   └── video_dataset.py       # Video and image datasets
└── configs/
    ├── pretrain_config.yaml   # Pre-training hyperparameters
    ├── full_training_config.yaml  # Full training config
    └── finetune_config.yaml   # Fine-tuning config
train.py                       # Training script
evaluate.py                    # Evaluation script
setup.py
requirements.txt
```

## Assumptions and Unresolved Details

1. **FPN Architecture**: The paper mentions using a FPN to fuse stride 16 and 32 features. The exact FPN design (number of layers, normalization) is not fully specified; we use a standard top-down FPN.

2. **Memory Bank Positional Encoding**: The paper uses sinusoidal absolute positional embeddings plus 2D RoPE in memory attention. The exact implementation details of how these are combined are inferred from the paper description.

3. **Object Pointer Splitting**: The paper mentions splitting the 256-dim object pointer into 4 tokens of 64-dim for cross-attention. We implement this as described.

4. **Mosaic Transform**: The paper describes a 2×2 mosaic transform with 10% probability. The exact implementation details (overlap, padding) are inferred.

5. **Hiera Downsampling**: The exact downsampling mechanism between Hiera stages is inferred as average pooling followed by linear projection, consistent with the Hiera paper.

6. **Training Scale**: The full training used 256 A100 GPUs for 108 hours. This reproduction is designed to be functionally equivalent but may require adaptation for different hardware configurations.

7. **SA-V Dataset**: The SA-V dataset is not included in this repository. Users need to download it from https://ai.meta.com/datasets/segment-anything-video/ to train the model.

## Key Results from Paper

| Method | DAVIS 2017 val J&F | MOSE val J&F | SA-V val J&F |
|--------|-------------------|--------------|--------------|
| Cutie-base+ | 88.1 | 71.7 | 61.3 |
| SAM 2 (Hiera-B+) | 90.2 | 76.6 | 76.8 |
| SAM 2 (Hiera-L) | 90.7 | 77.9 | 77.9 |

Image segmentation (1-click mIoU on SA-23):
| Model | SA-23 All | FPS |
|-------|-----------|-----|
| SAM (ViT-H) | 58.1 | 21.7 |
| SAM 2 (Hiera-B+) | 58.9 | 130.1 |
| SAM 2 (Hiera-B+, our mix) | 61.9 | 130.1 |
