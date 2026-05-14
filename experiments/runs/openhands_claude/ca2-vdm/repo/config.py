from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Model Architecture Config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Architecture hyperparameters for the spatial-temporal Transformer."""

    # Patch embedding
    in_channels: int = 4            # VAE latent channels (SD VAE: 4)
    patch_size: int = 2             # Spatial patch size

    # Transformer
    hidden_dim: int = 1152          # Model dimension (Open-Sora v1.0)
    num_layers: int = 28            # Number of transformer blocks
    num_heads: int = 16             # Attention heads
    ff_mult: int = 4                # Feed-forward expansion factor
    dropout: float = 0.0

    # Text conditioning (T5 encoder)
    use_text: bool = True
    context_dim: int = 4096         # T5-XXL embedding dimension

    # Positional embeddings
    max_spatial_h: int = 64         # Max spatial height in patches (256/4=64 with patch_size=2 and VAE 8x)
    max_spatial_w: int = 64
    max_temporal_len: int = 512     # Max temporal sequence length for TPE table

    # Prefix enhancement (P')
    prefix_len: int = 3             # P' = 3 (paper Section 3.2)


# ---------------------------------------------------------------------------
# Diffusion Config
# ---------------------------------------------------------------------------

@dataclass
class DiffusionConfig:
    """Diffusion process hyperparameters."""

    # Training schedule (DDPM, Ho et al. 2020)
    num_train_timesteps: int = 1000
    beta_start: float = 1e-4        # beta_1
    beta_end: float = 0.02          # beta_T
    schedule: str = "linear"        # Linear schedule for training

    # Inference schedule (Improved DDPM, Nichol & Dhariwal 2021)
    num_inference_steps: int = 100  # 100 denoising steps at inference
    inference_schedule: str = "cosine"

    # Model output
    learn_variance: bool = True     # Learnable covariance Sigma_theta
    predict_xstart: bool = False    # Predict noise (not x_0)

    # Classifier-free guidance (T2V only)
    cfg_scale: float = 7.5          # Guidance scale for T2V
    cfg_dropout: float = 0.1        # Probability of dropping text condition during training


# ---------------------------------------------------------------------------
# T2V Training Config (InternVid)
# ---------------------------------------------------------------------------

@dataclass
class T2VTrainingConfig:
    """
    Text-to-video training configuration.
    Two-stage training on InternVid (4.9M filtered pairs).
    """

    # Dataset
    dataset: str = "internvid"
    internvid_root: str = "data/internvid"
    resolution: int = 256           # 256x256

    # Stage 1: Causal modeling without clean prefix
    stage1_num_frames: int = 32     # 32-frame videos
    stage1_batch_size: int = 288
    stage1_num_steps: int = 32000   # 32k steps
    stage1_prefix_frames: int = 0   # No conditional frames

    # Stage 2: With clean prefix
    chunk_len: int = 16             # l = 16
    p_max: int = 49                 # P_max = 1 + 3*l = 49
    max_train_frames: int = 65      # L_train = P_max + l = 65
    stage2_batch_size: int = 144
    stage2_num_steps: int = 21000   # 21k steps

    # Optimizer (AdamW, Loshchilov & Hutter 2019)
    learning_rate: float = 2e-5
    weight_decay: float = 1e-2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    grad_clip: float = 1.0

    # Training
    num_workers: int = 8
    save_every: int = 1000
    log_every: int = 100
    eval_every: int = 5000

    # Prefix length sampling: P in {1, 1+l, 1+2l, ..., 1+nl=P_max}
    # Multiples of chunk length l
    prefix_multiples: List[int] = field(default_factory=lambda: [0, 1, 2, 3])

    # Checkpoint
    output_dir: str = "checkpoints/t2v_ca2vdm"
    resume_from: Optional[str] = None
    pretrained_opensora: Optional[str] = None  # Path to Open-Sora v1.0 weights

    # Model
    model_type: str = "ca2vdm"

    # Fixed prefix for OS-Fix baseline
    fixed_prefix: int = 16          # P = L_train / 2 = 16


# ---------------------------------------------------------------------------
# Video Prediction Training Config (SkyTimelapse)
# ---------------------------------------------------------------------------

@dataclass
class VideoPredTrainingConfig:
    """
    Video prediction training configuration.
    Training on SkyTimelapse without text input.
    """

    # Dataset
    dataset: str = "skytimelapse"
    skytimelapse_root: str = "data/skytimelapse"
    resolution: int = 256

    # Training config
    chunk_len: int = 8              # l = 8
    p_max: int = 25                 # P_max = 1 + 3*l = 25
    max_train_frames: int = 33      # L_train = P_max + l = 33
    batch_size: int = 8
    num_steps: int = 11000          # 11k steps

    # OS-Fix baseline: fixed P = 8, L_train = 16
    fixed_prefix: int = 8
    osfix_max_train_frames: int = 16

    # Optimizer
    learning_rate: float = 2e-5
    weight_decay: float = 1e-2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    grad_clip: float = 1.0

    # Training
    num_workers: int = 4
    save_every: int = 500
    log_every: int = 50
    eval_every: int = 2000

    # Prefix length sampling
    prefix_multiples: List[int] = field(default_factory=lambda: [0, 1, 2, 3])

    # Checkpoint
    output_dir: str = "checkpoints/vidpred_ca2vdm"
    resume_from: Optional[str] = None
    pretrained_opensora: Optional[str] = None

    # Model (no text conditioning for video prediction)
    model_type: str = "ca2vdm"
    use_text: bool = False


# ---------------------------------------------------------------------------
# Inference Config
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    """Autoregressive inference configuration."""

    # Generation
    num_ar_steps: int = 6           # Number of autoregression steps
    chunk_len: int = 16             # l frames per AR step
    p_max: int = 49                 # Maximum condition length

    # Sampling
    num_inference_steps: int = 100  # Improved DDPM steps
    cfg_scale: float = 7.5          # Classifier-free guidance scale

    # Resolution
    resolution: int = 256
    latent_h: int = 32              # H after 8x VAE downsampling: 256/8 = 32
    latent_w: int = 32

    # KV-cache
    use_kv_cache: bool = True
    use_spatial_cache: bool = True  # Prefix-enhanced spatial attention cache

    # Cyclic-TPE
    max_train_len: int = 65         # L_train = P_max + l

    # Output
    output_dir: str = "outputs"
    save_video: bool = True
    fps: int = 8


# ---------------------------------------------------------------------------
# Evaluation Config
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Evaluation configuration for FVD computation."""

    # Datasets
    msrvtt_root: str = "data/msrvtt"
    ucf101_root: str = "data/ucf101"
    skytimelapse_root: str = "data/skytimelapse"

    # FVD evaluation
    num_eval_videos: int = 2048     # UCF-101: 2048 samples
    msrvtt_num_videos: int = 2990   # MSR-VTT: 2990 test videos
    fvd_num_frames: int = 16        # I3D accepts 16 frames minimum

    # I3D model for FVD
    i3d_checkpoint: str = "pretrained/i3d_kinetics.pt"

    # Chunk-wise FVD (Table 3, 4 in paper)
    eval_chunk_fvd: bool = True
    num_ar_steps_eval: int = 6      # 6 AR steps -> 48 or 96 frames
    fvd_chunk_size: int = 16        # 16-frame chunks for FVD

    # VBench metrics
    eval_vbench: bool = False
    vbench_metrics: List[str] = field(default_factory=lambda: [
        "aesthetic_quality",
        "imaging_quality",
        "motion_smoothness",
        "temporal_flickering",
    ])

    # Batch size for evaluation
    eval_batch_size: int = 4
    num_workers: int = 4


# ---------------------------------------------------------------------------
# Full Config
# ---------------------------------------------------------------------------

@dataclass
class Ca2VDMConfig:
    """Complete configuration for Ca2-VDM."""

    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    t2v_train: T2VTrainingConfig = field(default_factory=T2VTrainingConfig)
    vidpred_train: VideoPredTrainingConfig = field(default_factory=VideoPredTrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Experiment
    seed: int = 42
    device: str = "cuda"
    mixed_precision: str = "bf16"   # bf16 or fp16 or no


# ---------------------------------------------------------------------------
# Preset Configs
# ---------------------------------------------------------------------------

def get_t2v_ca2vdm_config() -> Ca2VDMConfig:
    """Ca2-VDM config for T2V generation on InternVid."""
    cfg = Ca2VDMConfig()
    cfg.model.use_text = True
    cfg.t2v_train.model_type = "ca2vdm"
    return cfg


def get_t2v_osfix_config() -> Ca2VDMConfig:
    """OS-Fix baseline config for T2V generation."""
    cfg = Ca2VDMConfig()
    cfg.model.use_text = True
    cfg.t2v_train.model_type = "osfix"
    cfg.t2v_train.max_train_frames = 32
    cfg.t2v_train.stage1_num_frames = 32
    cfg.t2v_train.stage2_batch_size = 288
    cfg.t2v_train.stage2_num_steps = 20000
    return cfg


def get_vidpred_ca2vdm_config() -> Ca2VDMConfig:
    """Ca2-VDM config for video prediction on SkyTimelapse."""
    cfg = Ca2VDMConfig()
    cfg.model.use_text = False
    cfg.model.context_dim = None
    cfg.vidpred_train.model_type = "ca2vdm"
    cfg.inference.chunk_len = 8
    cfg.inference.p_max = 25
    cfg.inference.max_train_len = 33
    return cfg


def get_vidpred_osfix_config() -> Ca2VDMConfig:
    """OS-Fix baseline config for video prediction."""
    cfg = Ca2VDMConfig()
    cfg.model.use_text = False
    cfg.model.context_dim = None
    cfg.vidpred_train.model_type = "osfix"
    cfg.vidpred_train.max_train_frames = 16
    cfg.vidpred_train.fixed_prefix = 8
    return cfg


def get_vidpred_osext_config() -> Ca2VDMConfig:
    """OS-Ext baseline config for video prediction."""
    cfg = Ca2VDMConfig()
    cfg.model.use_text = False
    cfg.model.context_dim = None
    cfg.vidpred_train.model_type = "osext"
    return cfg
