"""Configuration for Pyramidal Flow Matching."""
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class VAEConfig:
    """3D VAE configuration (similar to MAGVIT-v2)."""
    in_channels: int = 3
    latent_channels: int = 16
    base_channels: int = 128
    channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    temporal_downsample: Tuple[bool, ...] = (True, True, True)  # 8x temporal
    spatial_downsample: Tuple[bool, ...] = (True, True, True)   # 8x spatial
    kl_weight: float = 1e-6
    dropout: float = 0.0


@dataclass
class DiTConfig:
    """MM-DiT configuration (based on SD3 Medium)."""
    num_layers: int = 24
    hidden_size: int = 2048
    num_heads: int = 32
    head_dim: int = 64
    ff_mult: float = 4.0
    patch_size: int = 2  # spatial patchification
    in_channels: int = 16  # VAE latent channels
    out_channels: int = 16
    pooled_text_dim: int = 2048
    context_dim: int = 4096  # T5 context
    clip_dim: int = 768       # CLIP context
    rope_theta: float = 10000.0
    max_seq_len: int = 4096
    dropout: float = 0.0
    qk_norm: bool = True
    use_swiglu: bool = True


@dataclass
class PyramidConfig:
    """Pyramidal flow matching configuration."""
    num_stages: int = 3  # K
    spatial_downsample_factor: int = 2  # halving each stage
    # Start and end timesteps for each stage (uniform partitioning)
    stage_boundaries: Optional[List[Tuple[float, float]]] = None
    # Corrective noise gamma
    corrective_gamma: float = -1.0 / 3.0
    # Whether to use coupled noise sampling
    coupled_sampling: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Stage 1: Image training
    stage1_steps: int = 50000
    stage1_batch_size: int = 1536
    stage1_lr: float = 1e-4
    stage1_betas: Tuple[float, float] = (0.9, 0.999)

    # Stage 2: Low-resolution video training
    stage2_steps: int = 200000  # 80k at 2s + 120k at 5s
    stage2_batch_size: int = 768
    stage2_lr: float = 1e-4
    stage2_betas: Tuple[float, float] = (0.9, 0.95)

    # Stage 3: High-resolution video training
    stage3_steps: int = 50000
    stage3_batch_size: int = 384
    stage3_lr: float = 5e-5
    stage3_betas: Tuple[float, float] = (0.9, 0.95)

    # Optimizer
    weight_decay: float = 1e-4
    eps: float = 1e-6
    gradient_clip: float = 1.0
    warmup_steps: int = 1000

    # Mixed precision
    mixed_precision: str = "bf16"

    # Classifier-free guidance
    cfg_prob: float = 0.1
    cfg_scale: float = 7.0

    # History corruption (for temporal pyramid)
    history_noise_max: float = 1.0 / 3.0

    # Sequence lengths
    max_frames: int = 241  # 10s at 24fps
    latent_temporal_compression: int = 8

    # Image proportion in video training
    image_proportion: float = 0.125


@dataclass
class DataConfig:
    """Data configuration."""
    image_size: int = 768
    latent_size: int = 96  # 768 / 8

    # Image datasets
    laion_path: str = ""
    cc12m_path: str = ""
    sa1b_path: str = ""
    journeydb_path: str = ""
    synthetic_data_path: str = ""

    # Video datasets
    webvid_path: str = ""
    openvid_path: str = ""
    opensora_plan_path: str = ""

    # Video settings
    fps: int = 24
    min_duration: float = 2.0
    max_duration: float = 10.0

    # T5 and CLIP
    t5_model: str = "google/t5-v1_1-xxl"
    clip_model: str = "openai/clip-vit-large-patch14"


@dataclass
class Config:
    """Main configuration."""
    vae: VAEConfig = field(default_factory=VAEConfig)
    dit: DiTConfig = field(default_factory=DiTConfig)
    pyramid: PyramidConfig = field(default_factory=PyramidConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: str = "./outputs"
    seed: int = 42
    num_gpus: int = 128
