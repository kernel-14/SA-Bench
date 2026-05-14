# Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing

Implementation of the paper "Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing" (Gao et al., 2025).

## Overview

Ca2-VDM introduces two key mechanisms for efficient autoregressive video generation:

1. **Causal Generation**: Replaces bidirectional temporal attention with causal (unidirectional) temporal attention, enabling KV-cache precomputation and reuse across autoregression steps.

2. **Cache Sharing**: Shares the KV-cache across all denoising timesteps by assigning `tEmb(0)` to conditional frames, eliminating per-timestep cache storage.

Additional components:
- **KV-Cache Queue**: Bounded-length queue for temporal KV-cache enabling long-term context.
- **Cyclic-TPEs**: Cyclic temporal positional embeddings for generation beyond training length.
- **Prefix-Enhanced Spatial Attention**: Spatially concatenates prefix frames to enhance conditioning.

## Repository Structure

```
repo/
├── layers.py       # Fundamental layers: attention, positional embeddings, MLP
├── modules.py      # Transformer blocks: causal generation block, cross-attention block
├── model.py        # Full model: Ca2VDM, OSFix (baseline), OSExt (baseline)
├── diffusion.py    # Diffusion process: DDPM, improved DDPM, noise schedules
├── data.py         # Dataset classes: InternVid, SkyTimelapse, UCF-101, MSR-VTT
├── config.py       # All hyperparameters and configurations
├── train.py        # Training loop with combined L_simple + L_vlb loss
├── inference.py    # Autoregressive inference with KV-cache queue and Cyclic-TPEs
├── evaluate.py     # FVD evaluation using I3D features
└── requirements.txt
```

## Key Design Choices (from paper)

### Training
- Clean prefix frames use `tEmb(0)`; denoising target frames use `tEmb(t)`
- Loss mask `m` excludes clean prefix from gradient computation
- Combined loss: `L_simple + L_vlb` with learnable covariance
- Cyclic TPE shift with random offset per sample

### Inference
- Each AR step: denoising stage (use cached KV) + cache writing stage (compute new KV)
- Temporal KV-cache queue: dequeue oldest chunk when `P_k >= P_max`
- Spatial KV-cache: only most recent chunk (overwritten each AR step)
- Cache shared across all denoising timesteps `t`

### Hyperparameters
| Setting | Value |
|---------|-------|
| T2V chunk length `l` | 16 |
| T2V `P_max` | 49 (= 1 + 3×16) |
| Video pred chunk length `l` | 8 |
| Video pred `P_max` | 25 (= 1 + 3×8) |
| Prefix enhancement `P'` | 3 |
| DDPM steps `T` | 1000 |
| Inference steps | 100 (improved DDPM) |
| Learning rate | 2e-5 |
| CFG scale | 7.5 |
| Resolution | 256×256 |

## Baselines

- **OS-Fix**: Bidirectional attention, fixed-length conditional frames (`P = L_train/2`)
- **OS-Ext**: Bidirectional attention, extendable conditional frames (same training config as Ca2-VDM)

## Usage

### Training (T2V)
```bash
python train.py --config configs/t2v_ca2vdm.yaml
```

### Training (Video Prediction)
```bash
python train.py --config configs/vidpred_ca2vdm.yaml
```

### Autoregressive Inference
```bash
python inference.py --config configs/t2v_ca2vdm.yaml --checkpoint path/to/ckpt.pt \
    --prompt "A beautiful sunset over the ocean" --num_ar_steps 6
```

### Evaluation (FVD)
```bash
python evaluate.py --config configs/t2v_ca2vdm.yaml --checkpoint path/to/ckpt.pt \
    --dataset msrvtt --split test
```
