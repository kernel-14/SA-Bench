"""Configuration for Prioritized Generative Replay (PGR)."""

import dataclasses
from dataclasses import dataclass, field
from typing import Optional, List, Literal


@dataclass
class DiffusionConfig:
    # Diffusion architecture (residual MLP, same as SynthER)
    n_timesteps: int = 1000  # N, diffusion timesteps
    beta_start: float = 1e-4
    beta_end: float = 0.02
    model_dims: int = 512  # hidden dimension
    n_residual_blocks: int = 6  # residual blocks
    block_dims: int = 512  # dimension per residual block
    time_emb_dims: int = 256  # time embedding dimension
    cond_emb_dims: int = 128  # condition embedding dimension

    # Training
    lr: float = 3e-4
    batch_size: int = 256
    n_grad_steps_per_loop: int = 5000  # gradient steps per inner loop call

    # Classifier-free guidance
    p_uncond: float = 0.25  # probability of dropping condition
    guidance_scale: float = 1.5  # omega, CFG scale at sampling time

    # Conditioning prompt strategy (Section 4.3)
    top_k_ratio: float = 0.1  # k: fraction of highest-F transitions to sample from

    # Data
    context_len: int = 1  # single transition generation


@dataclass
class PolicyConfig:
    # Policy architecture
    hidden_dims: int = 256
    n_hidden_layers: int = 2

    # SAC/REDQ
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    target_entropy: Optional[float] = None
    utd: int = 20  # update-to-data ratio

    # REDQ specific
    n_critics: int = 10  # ensemble size
    n_target_critics: int = 2  # random subset for target

    # DRQ-v2 specific (pixel-based)
    cnn_features: int = 64
    cnn_layers: int = 4
    latent_dim: int = 50  # visual encoder output dim
    image_size: int = 84
    image_channels: int = 3
    image_augmentation: bool = True

    # Batch
    batch_size: int = 256
    synthetic_ratio: float = 0.5  # r in Algorithm 1


@dataclass
class CuriosityConfig:
    # ICM (Intrinsic Curiosity Module, Pathak et al. 2017)
    feature_dim: int = 512
    hidden_dim: int = 256
    lr: float = 1e-3
    forward_loss_weight: float = 1.0
    inverse_loss_weight: float = 0.2
    intrinsic_reward_weight: float = 0.1  # when used as exploration bonus

    # RND (Random Network Distillation, Burda et al. 2018)
    rnd_feature_dim: int = 512
    rnd_bottleneck: int = 64
    rnd_lr: float = 1e-3

    # CTS (Context Tree Switching density model, Bellemare et al. 2016)
    cts_context_bins: int = 8
    cts_image_size: int = 42
    cts_beta: float = 0.01  # for pseudo-count formula

    # ECO (Episodic Curiosity, Savinov et al. 2018)
    eco_memory_size: int = 200
    eco_alpha: float = 0.03
    eco_beta: float = 0.5
    eco_percentile: int = 90


@dataclass
class ReplayConfig:
    real_buffer_capacity: int = 1_000_000  # D_real
    syn_buffer_capacity: int = 1_000_000   # D_syn
    inner_loop_frequency: int = 10_000     # regenerates diffusion buffer every 10K env steps
    n_seed_steps: int = 5000  # random warmup steps


@dataclass
class EnvConfig:
    # DeepMind Control Suite
    dmc_domain: str = "quadruped"
    dmc_task: str = "walk"
    dmc_action_repeat: int = 1
    dmc_image_size: int = 84

    # OpenAI Gym
    gym_env_name: str = "HalfCheetah-v2"
    gym_max_episode_steps: int = 1000

    # DMLab
    dmlab_level: str = "sparse"
    dmlab_repeat: int = 4
    dmlab_episode_steps: int = 1800


@dataclass
class RunConfig:
    # General
    total_env_steps: int = 100_000
    eval_frequency: int = 5000
    n_eval_episodes: int = 10
    seed: int = 42

    # Experiment name
    experiment: str = "pgr_curiosity"
    relevance_fn: Literal["curiosity", "td_error", "return", "reward", "rnd", "cts", "eco"] = "curiosity"

    # Training device
    device: str = "cuda"

    # Sub-configs
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    curiosity: CuriosityConfig = field(default_factory=CuriosityConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    env: EnvConfig = field(default_factory=EnvConfig)

    # Scaling experiments (Section 5.3)
    scaling_larger_network: bool = False
    scaling_larger_batch: bool = False
    scaling_higher_ratio: bool = False
    scaling_utd: int = -1  # overrides policy.utd if > 0

    # Baselines
    use_per: bool = False  # prioritized experience replay
    use_exploration_bonus: bool = False
    noisy_nets: bool = False
    bootstrapped_q: bool = False


# Pre-configured experiment setups
def get_dmc_state_config() -> RunConfig:
    """State-based DMC-100K benchmark."""
    return RunConfig(
        total_env_steps=100_000,
        experiment="dmc_state",
    )


def get_dmc_pixel_config() -> RunConfig:
    """Pixel-based DMC-100K benchmark."""
    cfg = RunConfig(
        total_env_steps=100_000,
        experiment="dmc_pixel",
    )
    cfg.policy.image_channels = 3
    cfg.policy.image_size = 84
    return cfg


def get_gym_config() -> RunConfig:
    """OpenAI Gym state-based benchmark."""
    return RunConfig(
        total_env_steps=100_000,
        experiment="gym_state",
    )


def get_dmlab_config() -> RunConfig:
    """DMLab randomized environment."""
    return RunConfig(
        total_env_steps=10_000_000,
        experiment="dmlab",
    )


def get_scaling_config() -> RunConfig:
    """Scaling experiments (Section 5.3)."""
    cfg = RunConfig(
        total_env_steps=100_000,
        experiment="scaling",
        scaling_larger_network=True,
    )
    cfg.policy.hidden_dims = 512
    cfg.policy.n_hidden_layers = 3
    cfg.policy.batch_size = 1024
    return cfg
