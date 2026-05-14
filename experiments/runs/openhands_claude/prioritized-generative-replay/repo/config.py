from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DiffusionConfig:
    n_diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    hidden_dim: int = 256
    n_hidden_layers: int = 4
    time_embed_dim: int = 128
    cond_embed_dim: int = 128
    p_uncond: float = 0.25
    guidance_scale: float = 1.5
    clip_denoised: bool = True
    ddim_sampling_steps: int = 50
    use_ddim: bool = False


@dataclass
class REDQConfig:
    hidden_dim: int = 256
    n_hidden: int = 2
    n_critics: int = 10
    n_target_critics: int = 2
    utd_ratio: int = 20
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    tau: float = 0.005
    gamma: float = 0.99
    init_temperature: float = 0.1
    target_entropy: Optional[float] = None
    batch_size: int = 256


@dataclass
class SACConfig:
    hidden_dim: int = 256
    n_hidden: int = 2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    tau: float = 0.005
    gamma: float = 0.99
    init_temperature: float = 0.1
    target_entropy: Optional[float] = None
    batch_size: int = 256
    utd_ratio: int = 1


@dataclass
class DRQv2Config:
    hidden_dim: int = 1024
    n_hidden: int = 2
    feature_dim: int = 50
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    tau: float = 0.01
    gamma: float = 0.99
    batch_size: int = 256
    utd_ratio: int = 1
    n_aug: int = 2
    image_pad: int = 4
    encoder_lr: float = 1e-4


@dataclass
class ICMConfig:
    hidden_dim: int = 256
    feature_dim: int = 64
    forward_lr: float = 1e-3
    inverse_lr: float = 1e-3
    update_freq: float = 0.05


@dataclass
class RNDConfig:
    hidden_dim: int = 512
    latent_dim: int = 64
    output_dim: int = 512
    lr: float = 1e-3
    use_cnn: bool = False


@dataclass
class CTSConfig:
    n_context_bins: int = 8
    image_size: int = 42
    epsilon: float = 0.01


@dataclass
class ECOConfig:
    embed_dim: int = 512
    memory_size: int = 200
    alpha: float = 0.03
    beta: float = 0.5
    percentile: float = 90.0
    comparator_lr: float = 1e-3
    embedder_lr: float = 1e-3


@dataclass
class ReplayBufferConfig:
    real_buffer_size: int = 1_000_000
    syn_buffer_size: int = 1_000_000
    synthetic_ratio: float = 0.5
    top_k_ratio: float = 0.1


@dataclass
class TrainingConfig:
    total_env_steps: int = 100_000
    seed_steps: int = 5_000
    inner_loop_freq: int = 10_000
    inner_loop_steps: int = 1
    diffusion_train_steps: int = 50_000
    diffusion_batch_size: int = 256
    eval_freq: int = 5_000
    eval_episodes: int = 10
    log_freq: int = 1_000
    checkpoint_freq: int = 25_000
    seed: int = 0
    device: str = "cuda"
    use_wandb: bool = False
    wandb_project: str = "pgr"
    exp_name: str = "pgr"


@dataclass
class PixelConfig:
    pixel_obs: bool = False
    image_size: int = 84
    frame_stack: int = 3
    action_repeat: int = 2
    latent_dim: int = 50


@dataclass
class ScalingConfig:
    hidden_dim: int = 512
    n_hidden: int = 3
    batch_size: int = 1024
    synthetic_ratio: float = 0.75
    utd_ratio: int = 40
    syn_buffer_size: int = 2_000_000


@dataclass
class PGRConfig:
    env: str = "quadruped-walk"
    relevance: str = "curiosity"
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    redq: REDQConfig = field(default_factory=REDQConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    drqv2: DRQv2Config = field(default_factory=DRQv2Config)
    icm: ICMConfig = field(default_factory=ICMConfig)
    rnd: RNDConfig = field(default_factory=RNDConfig)
    cts: CTSConfig = field(default_factory=CTSConfig)
    eco: ECOConfig = field(default_factory=ECOConfig)
    buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    pixel: PixelConfig = field(default_factory=PixelConfig)


DMC_STATE_ENVS = [
    "quadruped-walk",
    "cheetah-run",
    "reacher-hard",
    "finger-turn-hard",
    "walker-walk",
    "hopper-hop",
]

DMC_PIXEL_ENVS = [
    "walker-walk",
    "cheetah-run",
]

GYM_ENVS = [
    "Walker2d-v2",
    "HalfCheetah-v2",
    "Hopper-v2",
]

DMLAB_ENVS = [
    "dmlab-sparse",
    "dmlab-very-sparse",
]

RELEVANCE_FUNCTIONS = [
    "curiosity",
    "td_error",
    "return",
    "reward",
    "rnd",
    "cts",
    "eco",
]

SPARSE_REWARD_ENVS = {"finger-turn-hard"}
FINGER_TURN_HARD_STEPS = 300_000

ENV_ACTION_REPEAT = {
    "quadruped-walk": 2,
    "cheetah-run": 4,
    "reacher-hard": 4,
    "finger-turn-hard": 2,
    "walker-walk": 2,
    "hopper-hop": 2,
}

ENV_FRAME_STACK = {
    "walker-walk": 3,
    "cheetah-run": 3,
}
