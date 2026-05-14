"""
SAM 2 configuration.

All hyperparameters from the paper (Table 12, Appendix D.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SAM2Config:
    # -----------------------------------------------------------------------
    # Model architecture
    # -----------------------------------------------------------------------
    encoder_variant: str = "B+"          # T, S, B+, L
    image_size: int = 1024               # Input resolution (square)
    embed_dim: int = 256                 # Frame embedding dimension
    memory_dim: int = 64                 # Memory feature dimension (projected)
    num_memory_attention_layers: int = 4 # L=4 memory attention layers
    max_recent_frames: int = 6           # N: FIFO queue size for recent frames
    max_prompted_frames: int = 16        # M: FIFO queue size for prompted frames
    num_multimask_outputs: int = 3       # Number of multi-mask outputs
    pointer_dim: int = 256               # Object pointer dimension
    num_pointer_tokens: int = 4          # Split pointer into 4×64 tokens

    # -----------------------------------------------------------------------
    # Pre-training (SA-1B, ~90k steps)
    # -----------------------------------------------------------------------
    pretrain_max_steps: int = 90_000
    pretrain_batch_size: int = 256
    pretrain_lr: float = 4e-4
    pretrain_warmup_steps: int = 1_000
    pretrain_cooldown_steps: int = 5_000
    max_masks_per_image: int = 64        # Max masks per image in pre-training
    num_correction_clicks_pretrain: int = 7  # 7 correction clicks (vs 8 in SAM)

    # -----------------------------------------------------------------------
    # Full training (SA-V + Internal + SA-1B + VOS, 200k steps)
    # -----------------------------------------------------------------------
    train_max_steps: int = 200_000
    train_batch_size: int = 128          # Per-GPU batch size
    train_lr: float = 4e-4
    train_warmup_steps: int = 1_000
    train_cooldown_steps: int = 5_000
    num_frames: int = 8                  # Frames per training sequence
    max_masklets_per_sequence: int = 3   # Max masklets per 8-frame sequence
    max_prompted_frames_per_seq: int = 2 # Up to 2 prompted frames per sequence
    reverse_temporal_prob: float = 0.5   # Reverse temporal order probability
    mosaic_prob: float = 0.1             # Mosaic transform probability
    random_gt_click_prob: float = 0.1    # Prob of random GT click (not error-based)

    # Data mixture weights (Appendix D.2.2)
    # ~15.2% SA-1B, ~70% SA-V, ~14.8% Internal
    image_data_weight: float = 0.152

    # -----------------------------------------------------------------------
    # Fine-tuning (16-frame sequences, 50k steps)
    # -----------------------------------------------------------------------
    finetune_max_steps: int = 50_000
    finetune_lr_scale: float = 0.5       # Half of training LR

    # -----------------------------------------------------------------------
    # Optimizer (AdamW)
    # -----------------------------------------------------------------------
    weight_decay: float = 0.1
    optimizer_betas: tuple = (0.9, 0.999)
    grad_clip_max_norm: float = 0.1      # L2 gradient clipping
    lr_timescale: int = 1_000            # Reciprocal sqrt schedule timescale

    # Layer-wise learning rate decay per encoder variant (Table 12)
    layer_decay: Dict[str, float] = field(default_factory=lambda: {
        "T": 0.8,
        "S": 0.8,
        "B+": 0.9,
        "L": 0.925,
    })

    # -----------------------------------------------------------------------
    # Drop path rates per encoder variant (Table 12)
    # -----------------------------------------------------------------------
    drop_path_rate: Dict[str, float] = field(default_factory=lambda: {
        "T": 0.1,
        "S": 0.1,
        "B+": 0.2,
        "L": 0.3,
    })

    # -----------------------------------------------------------------------
    # Loss weights (ratio 20:1:1:1 for focal:dice:iou:occlusion)
    # -----------------------------------------------------------------------
    loss_focal_weight: float = 20.0
    loss_dice_weight: float = 1.0
    loss_iou_weight: float = 1.0
    loss_occ_weight: float = 1.0

    # -----------------------------------------------------------------------
    # Prompt simulation
    # -----------------------------------------------------------------------
    # Initial prompt probabilities
    prompt_mask_prob: float = 0.50
    prompt_click_prob: float = 0.25
    prompt_box_prob: float = 0.25
    num_correction_clicks: int = 7

    # -----------------------------------------------------------------------
    # Data paths (set via CLI or environment)
    # -----------------------------------------------------------------------
    sa1b_root: str = ""
    sav_root: str = ""
    vos_roots: List[str] = field(default_factory=list)  # DAVIS, MOSE, YouTubeVOS
    internal_root: str = ""

    # -----------------------------------------------------------------------
    # Training infrastructure
    # -----------------------------------------------------------------------
    device: str = "cuda"
    num_workers: int = 8
    use_amp: bool = True                 # bfloat16 mixed precision
    output_dir: str = "./output"
    log_interval: int = 50
    save_interval: int = 5_000
    max_steps: int = 200_000             # Updated per stage

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------
    eval_num_clicks: int = 3             # Clicks per frame in interactive eval
    eval_max_frames: int = 8             # Max interacted frames in offline eval
    eval_iou_threshold: float = 0.75     # IoU threshold for online eval pausing
    eval_batch_size: int = 1             # Batch size for video evaluation


def get_config(variant: str = "B+") -> SAM2Config:
    """Get configuration for a specific encoder variant."""
    cfg = SAM2Config(encoder_variant=variant)
    return cfg


# ---------------------------------------------------------------------------
# Hiera encoder configurations (from Table 12 and paper text)
# ---------------------------------------------------------------------------

HIERA_ENCODER_CONFIGS = {
    "T": {
        "embed_dim": 96,
        "depths": [2, 3, 16, 3],
        "num_heads": [1, 2, 4, 8],
        "window_size": 8,
        "global_attn_blocks": [5, 7, 9],
        "drop_path_rate": 0.1,
        "layer_decay": 0.8,
    },
    "S": {
        "embed_dim": 96,
        "depths": [2, 3, 16, 3],
        "num_heads": [1, 2, 4, 8],
        "window_size": 8,
        "global_attn_blocks": [7, 10, 13],
        "drop_path_rate": 0.1,
        "layer_decay": 0.8,
    },
    "B+": {
        "embed_dim": 112,
        "depths": [2, 3, 16, 3],
        "num_heads": [2, 4, 8, 16],
        "window_size": 8,
        "global_attn_blocks": [12, 16, 20],
        "drop_path_rate": 0.2,
        "layer_decay": 0.9,
    },
    "L": {
        "embed_dim": 144,
        "depths": [2, 6, 36, 4],
        "num_heads": [2, 4, 8, 16],
        "window_size": 8,
        "global_attn_blocks": [23, 33, 43],
        "drop_path_rate": 0.3,
        "layer_decay": 0.925,
    },
}

# ---------------------------------------------------------------------------
# Training data mixture (Appendix D.2.2)
# ---------------------------------------------------------------------------

# Full training data mixture (with open-source VOS datasets):
# ~1.3% DAVIS, ~9.4% MOSE, ~9.2% YouTubeVOS, ~15.5% SA-1B, ~49.5% SA-V, ~15.1% Internal
FULL_DATA_MIXTURE = {
    "DAVIS": 0.013,
    "MOSE": 0.094,
    "YouTubeVOS": 0.092,
    "SA-1B": 0.155,
    "SA-V": 0.495,
    "Internal": 0.151,
}

# Ablation default (SA-V manual + 10% SA-1B):
ABLATION_DATA_MIXTURE = {
    "SA-1B": 0.10,
    "SA-V": 0.90,
}
