from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class ModelConfig:
    # Latent space
    latent_channels: int = 4
    spatial_size: int = 32  # after 8x VAE downsampling from 256

    # Transformer
    hidden_size: int = 1152
    num_heads: int = 16
    num_layers: int = 28
    patch_size: int = 2  # for spatial patching

    # Attention
    cross_attn_head_dim: int = 72
    spatial_attn_head_dim: int = 72
    temporal_attn_head_dim: int = 72

    # Prefix-enhanced spatial attention
    prefix_len_enhance: int = 3  # P' in paper

    # Text conditioning
    text_encoder_dim: int = 4096  # T5-XXL
    max_text_len: int = 120

    # Classifier-free guidance
    cfg_dropout_prob: float = 0.1


@dataclass
class DiffusionConfig:
    num_diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    # Inference
    num_inference_steps: int = 100
    prediction_type: str = "epsilon"  # or "v_prediction"

    # Loss
    loss_type: str = "l2_and_vlb"  # combined simplified + vlb


@dataclass
class TrainingConfig:
    # Video settings
    video_resolution: int = 256  # pixel
    video_frames_chunk: int = 16  # l
    max_prefix_len: int = 49  # P_max = 1 + 3*l for T2V (l=16)
    max_train_len: int = 65  # L_train = P_max + l

    # Training hyperparameters
    batch_size: int = 144
    learning_rate: float = 2e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    max_grad_norm: float = 1.0

    # Training stages
    stage1_steps: int = 32000
    stage2_steps: int = 21000
    # Stage 1: causal pretrain without clean prefix on 32-frame videos
    stage1_frames: int = 32
    # Stage 2: train with clean prefix on 65-frame videos

    # Mixed precision
    mixed_precision: str = "fp16"

    # Classifier-free guidance scale
    guidance_scale: float = 7.5


@dataclass
class InferenceConfig:
    chunk_length: int = 16  # l
    max_prefix_length: int = 49  # P_max
    num_inference_steps: int = 100
    guidance_scale: float = 7.5
    enable_kv_cache: bool = True
    use_cyclic_tpes: bool = True


@dataclass
class DataConfig:
    # T2V training
    internvid_filtered_path: str = "data/internvid_filtered_4.9M"
    # Video prediction
    skytimelapse_path: str = "data/skytimelapse"
    # Evaluation
    msrvtt_path: str = "data/msrvtt"
    ucf101_path: str = "data/ucf101"

    # Preprocessing
    frame_sample_rate: int = 1
    min_video_frames: int = 65
    resolution: int = 256


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Task-specific overrides
    # For video prediction on SkyTimelapse:
    #   video_frames_chunk = 8
    #   max_prefix_len = 25 (1 + 3*8)
    #   max_train_len = 33
    #   batch_size = 8
    #   training_steps = 11000


# Task-specific configs
def get_t2v_config() -> Config:
    c = Config()
    c.training.video_frames_chunk = 16
    c.training.max_prefix_len = 49
    c.training.max_train_len = 65
    c.training.batch_size = 144
    c.training.stage1_frames = 32
    c.training.stage1_steps = 32000
    c.training.stage2_steps = 21000
    c.inference.chunk_length = 16
    c.inference.max_prefix_length = 49
    return c


def get_video_prediction_config() -> Config:
    c = Config()
    c.training.video_frames_chunk = 8
    c.training.max_prefix_len = 25
    c.training.max_train_len = 33
    c.training.batch_size = 8
    c.training.stage1_steps = 0
    c.training.stage2_steps = 11000
    c.training.stage1_frames = 0
    c.inference.chunk_length = 8
    c.inference.max_prefix_length = 25
    return c
