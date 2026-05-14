# SAM 2: Segment Anything Model 2

Reproduction of the SAM 2 model from the paper:
> "SAM 2: Segment Anything in Images and Videos" (Ravi et al., 2024, Meta FAIR)

## Architecture

SAM 2 extends SAM to video segmentation with a streaming memory architecture:

1. **Image Encoder** (`image_encoder.py`): MAE pre-trained Hiera (hierarchical vision transformer) with FPN
2. **Memory Attention** (`memory_attention.py`): L transformer blocks conditioning current frame on past memories
3. **Prompt Encoder** (`prompt_encoder.py`): SAM-style encoder for clicks, boxes, and masks
4. **Mask Decoder** (`mask_decoder.py`): Two-way transformer with multi-mask output and occlusion prediction head
5. **Memory Encoder** (`memory_encoder.py`): Fuses predicted mask with image features for future frames
6. **Memory Bank** (`memory_bank.py`): FIFO queues for recent frames, prompted frames, and object pointers

## Usage

### Training
```bash
# Pre-train on SA-1B only
python train.py --mode pretrain --encoder-size b_plus

# Full training on mixed image+video data
python train.py --mode full --encoder-size b_plus --sa1b-root data/sa1b --sav-root data/sav

# All stages
python train.py --mode all --encoder-size b_plus
```

### Inference
```python
from sam2_model import build_sam2

model = build_sam2("b_plus")
model.eval()

# Single image (behaves like SAM)
output = model.forward_image(image_tensor, coords=click_coords, labels=click_labels)

# Video streaming
outputs = model.forward_video(frames_tensor, first_frame_prompts={"coords": coords, "labels": labels})
```

## Files

- `config.py` - All hyperparameters and model configurations
- `hiera.py` - Hiera hierarchical vision transformer backbone
- `image_encoder.py` - Image encoder wrapping Hiera with FPN
- `transformer.py` - Attention, RoPE, and transformer blocks
- `memory_attention.py` - Memory conditioning via cross-attention
- `prompt_encoder.py` - SAM-compatible prompt encoding
- `mask_decoder.py` - Two-way transformer mask decoder with occlusion head
- `memory_encoder.py` - Memory feature generation
- `memory_bank.py` - FIFO memory bank with object pointers
- `sam2_model.py` - Full SAM 2 model
- `losses.py` - Focal, dice, IoU, and occlusion losses
- `data.py` - Dataset loading and augmentations
- `train.py` - Training loops with interactive prompting simulation

## Key Design Decisions (per paper ablations)

- **No RPB** in image encoder (removed for speed, with neglible accuracy loss)
- **2D RoPE** in memory attention self-attention and cross-attention
- **No GRU** in memory (simpler FIFO bank is sufficient)
- **Object pointers** enabled (boosts SA-V val and LVOS)
- **Temporal positional encoding** only on recent frames (not prompted frames)
- **6 recent frames, 4 prompted frames** in memory bank
- **8-frame training sequences** (16 for fine-tuning)
