"""Configuration for SAM 2 training and model architecture."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class HieraConfig:
    """Hiera image encoder configuration (following Ryali et al., 2023 / Bolya et al., 2023)."""
    # Encoder size variants: T, S, B+, L
    model_name: str = "hiera_b_plus"

    # Input
    image_size: int = 1024

    # Architecture dimensions
    embed_dim: int = 112           # T:96, S:96, B+:112, L:144
    depths: Tuple[int, ...] = (2, 3, 16, 22)   # T:(1,2,7,5), S:(1,2,11,7), B+:(2,3,16,22), L:(2,6,36,4)
    num_heads: Tuple[int, ...] = (1, 2, 4, 8)

    # Window attention
    window_size: int = 14
    window_spec: Tuple[int, ...] = (8, 4, 14, 7)  # for stages 0-3, -1 means global

    # Global attention blocks (used only in specific layers)
    # B+: [12, 16, 20]; L: [23, 33, 43]; T: [5, 7, 9]; S: [7, 10, 13]
    global_att_blocks: List[int] = field(default_factory=lambda: [12, 16, 20])

    # Patch embedding
    patch_stride: Tuple[int, int] = (4, 4)
    patch_kernel: Tuple[int, int] = (7, 7)
    patch_padding: int = 3

    # FPN
    fpn_output_dim: int = 256

    # Remove relative positional bias from image encoder (Bolya et al. 2023 improvement)
    use_rpb: bool = False

    # Absolute positional encoding (windowed absolute, interpolated globally)
    use_abs_pos: bool = True

    # MLP ratio
    mlp_ratio: float = 4.0

    # Dropout
    drop_path_rate: float = 0.2   # T:0.1, S:0.1, B+:0.2, L:0.3
    dropout: float = 0.0

    # Stage strides
    strides: Tuple[int, ...] = (4, 8, 16, 32)


@dataclass
class MemoryAttentionConfig:
    """Memory attention configuration."""
    num_layers: int = 4           # L = 4 blocks
    d_model: int = 256
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    # 2D RoPE in self-attention and cross-attention
    use_rope: bool = True

    # Sinusoidal absolute positional embeddings
    use_abs_pos: bool = True

    # Cross-attention settings
    memory_cross_attn: bool = True
    pointer_cross_attn: bool = True


@dataclass
class MemoryBankConfig:
    """Memory bank configuration."""
    num_recent_frames: int = 6    # N = 6 recent (unprompted) frames
    num_prompted_frames: int = 4  # M = 4 prompted frames
    memory_channel_dim: int = 64  # projected to 64 channels
    object_pointer_dim: int = 256
    object_pointer_num_tokens: int = 4  # split 256-dim into 4 x 64-dim tokens

    # Occlusion embedding
    use_occlusion_embedding: bool = True

    # Temporal positional encoding on recent frames only (not prompted frames)
    use_temporal_pos: bool = True


@dataclass
class MaskDecoderConfig:
    """Mask decoder configuration (following SAM design)."""
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 2           # Two-way transformer blocks
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    # Multi-mask prediction
    num_multimask_outputs: int = 3  # predict 3 masks for ambiguous prompts
    multimask_on: bool = True

    # IoU token
    iou_head_hidden_dim: int = 256

    # Occlusion head (additional token for presence prediction)
    occlusion_head_enabled: bool = True
    occlusion_head_hidden_dim: int = 256

    # High-resolution skip connections from image encoder stages 1,2
    use_high_res_features: bool = True
    high_res_feature_dims: Tuple[int, ...] = (112, 224)  # stride 4, 8 features


@dataclass
class MemoryEncoderConfig:
    """Memory encoder configuration."""
    encoder_dim: int = 256
    memory_dim: int = 64          # Output dimension stored in memory bank

    # Fusion convolutions
    conv_layers: int = 2
    kernel_size: int = 3

    # Input: mask is downsampled using conv and summed with image embeddings
    mask_downsample_kernel: int = 7
    mask_downsample_stride: int = 4


@dataclass
class PromptEncoderConfig:
    """Prompt encoder configuration (identical to SAM)."""
    embed_dim: int = 256
    image_embedding_size: Tuple[int, int] = (64, 64)  # at stride 16

    # Point and box prompts
    mask_input_channels: int = 16
    mask_output_channels: int = 256

    # Number of masks to predict for ambiguous prompts
    num_multimask_outputs: int = 3


@dataclass
class TrainingConfig:
    """Training hyperparameters for SAM 2."""
    # Pre-training (SA-1B)
    pretrain_steps: int = 90000
    pretrain_batch_size: int = 256
    pretrain_image_size: int = 1024
    pretrain_learning_rate: float = 4e-4

    # Full training
    train_steps: int = 200000
    train_batch_size_images: int = 256
    train_batch_size_video: int = 128
    train_image_size: int = 1024
    train_num_frames: int = 8  # sample 8-frame sequences
    train_num_frames_finetune: int = 16  # for long-video fine-tuning

    # Learning rate
    learning_rate: float = 4e-4
    lr_schedule: str = "reciprocal_sqrt"  # reciprocal square-root schedule
    lr_timescale: int = 1000
    warmup_steps: int = 1000
    cooldown_steps: int = 5000

    # Layer-wise decay
    layer_decay: float = 0.9   # T/S:0.8, B+:0.9, L:0.925

    # Optimizer
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.1
    gradient_clip: float = 0.1
    precision: str = "bfloat16"

    # Drop path rates by encoder size
    drop_path_rates: dict = field(default_factory=lambda: {
        "hiera_t": 0.1, "hiera_s": 0.1, "hiera_b_plus": 0.2, "hiera_l": 0.3
    })

    # Loss weights
    focal_loss_weight: float = 20.0
    dice_loss_weight: float = 1.0
    iou_loss_weight: float = 1.0
    occlusion_loss_weight: float = 1.0

    # Interactive training simulation
    max_correction_clicks: int = 7  # during pre-training
    max_correction_clicks_video: int = 3  # during video training

    # Prompt sampling probabilities
    mask_prompt_prob: float = 0.5
    click_prompt_prob: float = 0.25
    box_prompt_prob: float = 0.25

    # Maximum prompted frames per sequence
    max_prompted_frames: int = 2

    # Maximum masklets per 8-frame sequence
    max_masklets_per_sequence: int = 3

    # Reverse temporal order probability
    reverse_time_prob: float = 0.5

    # Random click from ground truth probability (for flexibility)
    random_gt_click_prob: float = 0.1

    # Mosaic augmentation
    mosaic_prob: float = 0.1

    # Augmentations
    use_horizontal_flip: bool = True
    use_affine_transform: bool = True
    use_color_jitter: bool = True
    use_grayscale: bool = True

    # Data filtering
    max_mask_area_ratio: float = 0.9  # filter masks covering >90% of image
    max_masks_per_image: int = 64

    # Fine-tuning
    finetune_steps: int = 50000
    finetune_learning_rate_multiplier: float = 0.5
    finetune_freeze_image_encoder: bool = True

    # Data mixture percentages
    sa1b_ratio: float = 0.152
    sav_ratio: float = 0.70
    internal_ratio: float = 0.148


@dataclass
class SAM2Config:
    """Full SAM 2 model configuration."""
    # Image input
    image_size: int = 1024
    patch_size: int = 16

    # Sub-module configs
    hiera: HieraConfig = field(default_factory=HieraConfig)
    memory_attention: MemoryAttentionConfig = field(default_factory=MemoryAttentionConfig)
    memory_bank: MemoryBankConfig = field(default_factory=MemoryBankConfig)
    memory_encoder: MemoryEncoderConfig = field(default_factory=MemoryEncoderConfig)
    mask_decoder: MaskDecoderConfig = field(default_factory=MaskDecoderConfig)
    prompt_encoder: PromptEncoderConfig = field(default_factory=PromptEncoderConfig)

    # Image encoder output
    image_encoder_output_dim: int = 256

    # Training config
    training_config: TrainingConfig = field(default_factory=TrainingConfig)


def get_config(encoder_size: str = "b_plus") -> SAM2Config:
    """Get SAM 2 configuration for a given encoder size."""
    assert encoder_size in ("t", "s", "b_plus", "l")

    hiera_configs = {
        "t": HieraConfig(
            model_name="hiera_t", embed_dim=96, depths=(1, 2, 7, 5),
            global_att_blocks=[5, 7, 9], drop_path_rate=0.1,
        ),
        "s": HieraConfig(
            model_name="hiera_s", embed_dim=96, depths=(1, 2, 11, 7),
            global_att_blocks=[7, 10, 13], drop_path_rate=0.1,
        ),
        "b_plus": HieraConfig(
            model_name="hiera_b_plus", embed_dim=112, depths=(2, 3, 16, 22),
            global_att_blocks=[12, 16, 20], drop_path_rate=0.2,
        ),
        "l": HieraConfig(
            model_name="hiera_l", embed_dim=144, depths=(2, 6, 36, 4),
            global_att_blocks=[23, 33, 43], drop_path_rate=0.3,
        ),
    }

    layer_decays = {"t": 0.8, "s": 0.8, "b_plus": 0.9, "l": 0.925}

    hiera = hiera_configs[encoder_size]
    training = TrainingConfig(layer_decay=layer_decays[encoder_size])

    return SAM2Config(hiera=hiera, training_config=training)
