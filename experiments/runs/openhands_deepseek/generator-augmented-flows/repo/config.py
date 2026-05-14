from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ModelConfig:
    """Configuration for the SongUNet architecture and consistency model."""
    model_channels: int = 128
    num_blocks: Tuple[int, ...] = (3,)
    channel_mult: Tuple[int, ...] = (1, 2, 2)
    attn_resolutions: Tuple[int, ...] = ()
    dropout: float = 0.0
    use_ema: bool = True
    ema_rate: float = 0.9999
    embedding_type: str = "positional"  # positional or fourier
    sigma_data: float = 0.5  # sigma_d in the paper


@dataclass
class ScheduleConfig:
    """Noise schedule and timestep discretization configuration."""
    sigma_min: float = 0.002  # sigma_0
    sigma_max: float = 80.0  # sigma_T
    rho: float = 7.0  # for noise schedule from Karras et al.
    s0: int = 10  # initial number of timesteps
    s1: int = 1280  # final number of timesteps
    p_mean: float = -1.1  # for timestep sampling distribution
    p_std: float = 2.0  # for timestep sampling distribution


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    dataset: str = "cifar10"  # cifar10, imagenet, celeba, lsun_church
    image_size: int = 32
    batch_size: int = 512
    total_steps: int = 100_000
    learning_rate: float = 0.0001
    optimizer: str = "lion"  # lion, adam
    weight_decay: float = 0.0
    mu: float = 0.5  # joint learning parameter for GC
    loss_type: str = "pseudo_huber"  # pseudo_huber, l2
    use_dropout: bool = False
    use_ema_target: bool = True  # whether to use EMA for GC endpoint predictions
    coupling: str = "gc"  # ic, ot, gc
    num_fid_samples: int = 50_000
    seed: int = 42
    mixed_precision: bool = True
    gradient_clip: float = 0.0


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# Dataset-specific configurations
DATASET_CONFIGS = {
    "cifar10": Config(
        model=ModelConfig(
            model_channels=128,
            num_blocks=(3,),
            channel_mult=(1, 2, 2),
            attn_resolutions=(),
            dropout=0.0,
        ),
        schedule=ScheduleConfig(),
        training=TrainingConfig(
            dataset="cifar10",
            image_size=32,
            batch_size=512,
            total_steps=100_000,
            learning_rate=0.0001,
            mu=0.5,
        ),
    ),
    "imagenet": Config(
        model=ModelConfig(
            model_channels=128,
            num_blocks=(3, 5, 7),
            channel_mult=(1, 1, 2),
            attn_resolutions=(16,),
            dropout=0.0,
        ),
        schedule=ScheduleConfig(),
        training=TrainingConfig(
            dataset="imagenet",
            image_size=32,
            batch_size=512,
            total_steps=150_000,
            learning_rate=0.00008,
            mu=0.5,
        ),
    ),
    "celeba": Config(
        model=ModelConfig(
            model_channels=128,
            num_blocks=(3, 3, 4, 5),
            channel_mult=(1, 2, 2, 2),
            attn_resolutions=(),
            dropout=0.0,
        ),
        schedule=ScheduleConfig(),
        training=TrainingConfig(
            dataset="celeba",
            image_size=64,
            batch_size=128,
            total_steps=150_000,
            learning_rate=0.00008,
            mu=0.5,
        ),
    ),
    "lsun_church": Config(
        model=ModelConfig(
            model_channels=128,
            num_blocks=(3, 3, 4, 5),
            channel_mult=(1, 2, 2, 2),
            attn_resolutions=(),
            dropout=0.0,
        ),
        schedule=ScheduleConfig(),
        training=TrainingConfig(
            dataset="lsun_church",
            image_size=64,
            batch_size=128,
            total_steps=150_000,
            learning_rate=0.00008,
            mu=0.5,
        ),
    ),
}
