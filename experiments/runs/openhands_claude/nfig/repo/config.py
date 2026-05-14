from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TokenizerConfig:
    # Image dimensions
    image_size: int = 256
    in_channels: int = 3
    # Encoder/decoder
    z_channels: int = 256
    ch: int = 128
    ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    num_res_blocks: int = 2
    attn_resolutions: List[int] = field(default_factory=lambda: [16])
    dropout: float = 0.0
    # Codebook
    codebook_size: int = 4096
    codebook_dim: int = 256
    # Frequency residual quantization
    # Scaling factors determine token grid sizes: 1x1, 2x2, ..., 16x16
    scale_factors: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 8, 10, 13, 16])
    # Total tokens = sum of (s*s) for s in scale_factors = 680
    # Feature map size after encoder (256 / 16 = 16)
    feature_map_size: int = 16
    # Training
    lr: float = 4.5e-6
    disc_lr: float = 4.5e-6
    batch_size: int = 8
    num_epochs: int = 20
    warmup_steps: int = 1000
    # Loss weights
    rec_loss_weight: float = 1.0
    freq_loss_weight: float = 1.0
    perceptual_loss_weight: float = 1.0
    gan_loss_weight: float = 0.5
    codebook_loss_weight: float = 1.0
    commitment_loss_weight: float = 0.25
    # Discriminator
    disc_start: int = 50001
    disc_num_layers: int = 3
    disc_in_channels: int = 3
    # DINOv2 discriminator
    dino_model: str = "dinov2_vitb14"
    # Pretrained encoder init
    encoder_pretrained: str = "dinov2_vitb14"


@dataclass
class TransformerConfig:
    # Model size variants
    # 310M: depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4
    # 600M: depth=20, embed_dim=1152, num_heads=16, mlp_ratio=4
    depth: int = 16
    embed_dim: int = 1024
    num_heads: int = 16
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0
    # Vocabulary
    vocab_size: int = 4096
    num_classes: int = 1000
    # Frequency token sequence
    scale_factors: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 8, 10, 13, 16])
    # Total sequence length = 680
    # Class conditioning
    class_embed_dim: int = 1024
    # Training
    lr: float = 8e-5
    batch_size: int = 768
    num_epochs: int = 350
    warmup_epochs: int = 5
    weight_decay: float = 0.05
    grad_clip: float = 2.0
    # Inference
    cfg_scale: float = 4.5
    top_k: int = 990
    temperature: float = 1.0


@dataclass
class NFIGConfig:
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    # Data
    data_root: str = "/data/imagenet"
    num_workers: int = 8
    # Logging
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    log_every: int = 100
    save_every: int = 5
    # Evaluation
    eval_batch_size: int = 50
    num_eval_samples: int = 50000
    fid_stats_path: str = "./fid_stats/imagenet256_fid_stats.npz"


def get_token_counts(scale_factors: List[int]) -> List[int]:
    return [s * s for s in scale_factors]


def get_total_tokens(scale_factors: List[int]) -> int:
    return sum(s * s for s in scale_factors)


def get_token_grid_sizes(scale_factors: List[int]) -> List[Tuple[int, int]]:
    return [(s, s) for s in scale_factors]


# Default config
default_config = NFIGConfig()

# 600M config
config_600m = NFIGConfig(
    transformer=TransformerConfig(
        depth=20,
        embed_dim=1152,
        num_heads=16,
    )
)
