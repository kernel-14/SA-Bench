from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class VAEConfig:
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 16
    base_channels: int = 128
    channel_multipliers: List[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    spatial_downsample: int = 8
    temporal_downsample: int = 8
    kl_weight: float = 1e-6
    use_causal_conv: bool = True


@dataclass
class DiTConfig:
    hidden_size: int = 1536
    num_layers: int = 24
    num_heads: int = 24
    mlp_ratio: float = 4.0
    in_channels: int = 16
    patch_size: int = 2
    temporal_patch_size: int = 1
    context_dim: int = 4096
    clip_dim: int = 768
    pooled_dim: int = 2048
    num_registers: int = 0
    use_causal_attention: bool = True
    dropout: float = 0.0
    qk_norm: bool = True


@dataclass
class PyramidConfig:
    num_stages: int = 3
    # Stage time windows [s_k, e_k] for k=0..K-1 (k=0 is lowest resolution, k=K-1 is full res)
    #
    # The boundaries are derived from the renoising constraint (Appendix A):
    #   e_k = 2 * s_{k+1} / (1 + s_{k+1})
    # where s_{k+1} is the start of the next (higher-res) stage.
    #
    # For K=3 with s_1=1/3, s_2=2/3:
    #   e_0 = 2*(1/3)/(1+1/3) = 0.5
    #   e_1 = 2*(2/3)/(1+2/3) = 0.8
    #
    # The trajectory visits each stage with a rollback at jump points:
    #   Stage 0 (lowest res): t in [0, 0.5]
    #   Jump back to t=1/3, Stage 1 (mid res): t in [1/3, 0.8]
    #   Jump back to t=2/3, Stage 2 (full res): t in [2/3, 1.0]
    stage_range: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.5), (1/3, 0.8), (2/3, 1.0)]
    )
    # Renoising gamma for corrective noise (paper uses -1/3 for max decorrelation)
    renoising_gamma: float = -1 / 3
    # History corruption noise strength range [0, max_corrupt]
    history_noise_max: float = 1 / 3
    # Upsampling mode for spatial pyramid
    upsample_mode: str = "nearest"
    # Downsampling mode for spatial pyramid
    downsample_mode: str = "bilinear"


@dataclass
class TrainingConfig:
    # Stage 1: Image training
    stage1_steps: int = 50_000
    stage1_lr: float = 1e-4
    stage1_batch_size: int = 1536
    stage1_warmup_steps: int = 1_000
    stage1_beta1: float = 0.9
    stage1_beta2: float = 0.999
    stage1_eps: float = 1e-6

    # Stage 2: Low-resolution video training
    stage2_steps: int = 200_000
    stage2_lr: float = 1e-4
    stage2_batch_size: int = 768
    stage2_warmup_steps: int = 1_000
    stage2_beta1: float = 0.9
    stage2_beta2: float = 0.95
    stage2_eps: float = 1e-6

    # Stage 3: High-resolution video training
    stage3_steps: int = 50_000
    stage3_lr: float = 5e-5
    stage3_batch_size: int = 384
    stage3_warmup_steps: int = 1_000
    stage3_beta1: float = 0.9
    stage3_beta2: float = 0.95
    stage3_eps: float = 1e-6

    # Common
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    precision: str = "bfloat16"
    num_gpus: int = 128
    image_data_ratio: float = 0.125  # 12.5% image data in video batches

    # CFG
    cfg_dropout_prob: float = 0.1
    cfg_scale: float = 7.5


@dataclass
class InferenceConfig:
    num_inference_steps: int = 20
    cfg_scale: float = 7.5
    # Video generation settings
    fps: int = 24
    video_duration_5s_frames: int = 121
    video_duration_10s_frames: int = 241
    # Resolution settings
    low_res_height: int = 384
    low_res_width: int = 384
    high_res_height: int = 768
    high_res_width: int = 768


@dataclass
class DataConfig:
    # Image datasets
    laion_path: str = "data/laion5b"
    cc12m_path: str = "data/cc12m"
    sa1b_path: str = "data/sa1b"
    journeydb_path: str = "data/journeydb"
    synthetic_path: str = "data/synthetic"

    # Video datasets
    webvid_path: str = "data/webvid10m"
    openvid_path: str = "data/openvid1m"
    opensora_path: str = "data/opensora"

    # Preprocessing
    max_image_size: int = 1024
    min_image_size: int = 256
    video_fps: int = 24
    max_video_frames: int = 241

    # Bucket settings for aspect ratio preservation
    bucket_sizes: List[Tuple[int, int]] = field(
        default_factory=lambda: [
            (256, 256), (256, 384), (384, 256),
            (384, 384), (384, 512), (512, 384),
            (512, 512), (512, 768), (768, 512),
            (768, 768),
        ]
    )


@dataclass
class ModelConfig:
    vae: VAEConfig = field(default_factory=VAEConfig)
    dit: DiTConfig = field(default_factory=DiTConfig)
    pyramid: PyramidConfig = field(default_factory=PyramidConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Text encoder settings
    t5_model: str = "google/t5-v1_1-xxl"
    clip_model: str = "openai/clip-vit-large-patch14"
    max_text_length: int = 256

    # Checkpoint
    pretrained_dit: str = "stabilityai/stable-diffusion-3-medium"
    output_dir: str = "outputs"
    save_every: int = 5_000
    eval_every: int = 5_000


def get_default_config() -> ModelConfig:
    return ModelConfig()
