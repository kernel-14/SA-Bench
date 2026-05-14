"""
All hyperparameters and configuration for Hi-MAR.

Paper sources:
  - Architecture: Table 1
  - Training: §4.2 Experimental Settings
  - Inference: §4.5 Experimental Analysis
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    # Model variant: "Hi-MAR-B", "Hi-MAR-L", "Hi-MAR-H", "Hi-MAR-S"
    model_name: str = "Hi-MAR-B"

    # VAE / tokenizer
    # KL-16 VAE: downsampling factor 16
    # 128x128 image → 8x8 latent → 64 tokens (small scale)
    # 256x256 image → 16x16 latent → 256 tokens (large scale)
    vae_path: str = "stabilityai/sd-vae-ft-ema"
    token_dim: int = 16          # VAE latent channel dimension (KL-16)
    num_tokens_small: int = 64   # 8x8 = 64 low-res tokens
    num_tokens_large: int = 256  # 16x16 = 256 high-res tokens
    image_size_small: int = 128  # low-res image size
    image_size_large: int = 256  # high-res image size

    # Conditioning
    # Class-conditional (ImageNet)
    num_classes: int = 1000
    # Text-to-image (MS-COCO): set text_embed_dim > 0 to enable
    text_embed_dim: int = 0      # 0 = class-conditional; 768 = CLIP ViT-B/32
    clip_model: str = "openai/clip-vit-base-patch32"

    # Diffusion
    num_diffusion_timesteps: int = 1000
    beta_schedule: str = "cosine"

    # Scale embedding
    scale_emb_dim: int = 256

    # Transformer
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # CFG
    use_cfg: bool = True
    cfg_dropout: float = 0.1     # probability of dropping class label during training


@dataclass
class TrainConfig:
    # Dataset
    dataset: str = "imagenet"    # "imagenet" or "coco"
    data_path: str = "/data/imagenet"
    coco_ann_path: str = "/data/coco/annotations"

    # Training
    epochs: int = 800
    batch_size: int = 256        # per GPU
    num_workers: int = 8
    pin_memory: bool = True

    # Optimizer (AdamW)
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.02
    grad_clip: float = 1.0

    # LR schedule: constant with linear warmup
    warmup_epochs: int = 100     # 100-epoch linear warmup (ImageNet)
    warmup_steps: int = 8000     # 8K-step warmup (MS-COCO)
    lr_schedule: str = "constant_with_warmup"

    # EMA (used for MS-COCO)
    use_ema: bool = False
    ema_momentum: float = 0.9999

    # Masking
    # Phase 1: uniform in [0.7, 1.0]
    mask_ratio_min_phase1: float = 0.7
    mask_ratio_max_phase1: float = 1.0
    # Phase 2: cosine masking (MaskGIT) for ImageNet;
    #          Beta(4, 1) for MS-COCO
    mask_strategy_phase2: str = "cosine"  # "cosine" or "beta"
    beta_alpha: float = 4.0      # Beta distribution α (MS-COCO)
    beta_beta: float = 1.0       # Beta distribution β (MS-COCO)

    # Logging / checkpointing
    log_every: int = 100
    save_every: int = 10         # epochs
    output_dir: str = "outputs"
    resume: Optional[str] = None

    # Distributed training
    world_size: int = 1
    local_rank: int = 0
    dist_backend: str = "nccl"

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"


@dataclass
class InferenceConfig:
    # Autoregressive steps
    steps_phase1: int = 32       # low-res generation steps
    steps_phase2: int = 4        # high-res generation steps

    # Diffusion steps within each head
    diff_steps_phase1: int = 100
    diff_steps_phase2: int = 100

    # CFG
    cfg_scale: float = 1.5       # classifier-free guidance scale
    # Note: for w/o CFG setting, CFG is only turned off for phase 2
    cfg_scale_phase1: float = 1.5
    cfg_scale_phase2: float = 1.0

    # Sampling
    temperature: float = 1.0
    batch_size: int = 128

    # Evaluation
    num_samples: int = 50000     # for FID/IS on ImageNet
    num_samples_coco: int = 30000  # for FID on MS-COCO

    # Output
    output_dir: str = "generated"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


# ---------------------------------------------------------------------------
# Pre-defined experiment configurations
# ---------------------------------------------------------------------------

def imagenet_base_config() -> Config:
    """Hi-MAR-B on ImageNet 256x256 class-conditional."""
    cfg = Config()
    cfg.model.model_name = "Hi-MAR-B"
    cfg.model.dataset = "imagenet"
    cfg.train.dataset = "imagenet"
    cfg.train.epochs = 800
    cfg.train.lr = 1e-4
    cfg.train.beta1 = 0.9
    cfg.train.beta2 = 0.95
    cfg.train.weight_decay = 0.02
    cfg.train.warmup_epochs = 100
    cfg.train.lr_schedule = "constant_with_warmup"
    cfg.train.mask_strategy_phase2 = "cosine"
    cfg.train.use_ema = False
    return cfg


def imagenet_large_config() -> Config:
    cfg = imagenet_base_config()
    cfg.model.model_name = "Hi-MAR-L"
    return cfg


def imagenet_huge_config() -> Config:
    cfg = imagenet_base_config()
    cfg.model.model_name = "Hi-MAR-H"
    return cfg


def coco_small_config() -> Config:
    """Hi-MAR-S on MS-COCO 256x256 text-to-image."""
    cfg = Config()
    cfg.model.model_name = "Hi-MAR-S"
    cfg.model.text_embed_dim = 768   # CLIP ViT-B/32
    cfg.model.num_classes = 0
    cfg.train.dataset = "coco"
    cfg.train.lr = 8e-4
    cfg.train.beta1 = 0.9
    cfg.train.beta2 = 0.95
    cfg.train.weight_decay = 0.03
    cfg.train.warmup_steps = 8000
    cfg.train.lr_schedule = "constant_with_warmup"
    cfg.train.mask_strategy_phase2 = "beta"
    cfg.train.use_ema = True
    cfg.train.ema_momentum = 0.9999
    return cfg
