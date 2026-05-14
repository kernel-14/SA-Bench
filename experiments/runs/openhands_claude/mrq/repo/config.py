from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MRQConfig:
    # ── Encoder ──────────────────────────────────────────────────────────────
    lambda_dynamics: float = 1.0       # λ_Dynamics (Table 3 – unlabelled, default 1)
    lambda_reward: float = 0.1         # λ_Reward
    lambda_terminal: float = 0.1       # λ_Terminal
    lambda_pre_activ: float = 1e-5     # λ_pre-activ (policy pre-activation regulariser)
    enc_horizon: int = 5               # H_Enc – encoder unroll horizon

    # ── TD3 value learning ───────────────────────────────────────────────────
    q_horizon: int = 3                 # H_Q – multi-step return horizon
    target_noise_std: float = 0.2      # σ for target policy smoothing
    target_noise_clip: float = 0.3     # c for target policy noise clipping

    # ── LAP (prioritised replay) ─────────────────────────────────────────────
    lap_alpha: float = 0.4             # probability smoothing exponent
    lap_min_priority: float = 1.0      # minimum priority

    # ── Exploration ──────────────────────────────────────────────────────────
    init_random_steps: int = 10_000    # warm-up steps with random actions
    expl_noise_std: float = 0.2        # exploration noise std

    # ── Common ───────────────────────────────────────────────────────────────
    discount: float = 0.99
    buffer_size: int = 1_000_000
    batch_size: int = 256
    target_update_freq: int = 250      # T_target – hard target-network copy period
    replay_ratio: int = 1              # gradient updates per environment step

    # ── Optimisers ───────────────────────────────────────────────────────────
    enc_lr: float = 1e-4
    enc_weight_decay: float = 1e-4
    value_lr: float = 3e-4
    value_weight_decay: float = 1e-4
    value_grad_clip: float = 20.0
    policy_lr: float = 3e-4
    policy_weight_decay: float = 1e-4

    # ── Architecture ─────────────────────────────────────────────────────────
    zs_dim: int = 512                  # state embedding dimension
    zsa_dim: int = 512                 # state-action embedding dimension
    za_dim: int = 256                  # action embedding dimension (internal)
    hidden_dim: int = 512

    # ── Reward categorical representation ────────────────────────────────────
    reward_bins: int = 65
    reward_range: float = 10.0         # bins span symexp([-10, 10])

    # ── Gumbel-Softmax (discrete actions) ────────────────────────────────────
    gumbel_tau: float = 10.0

    # ── Environment ──────────────────────────────────────────────────────────
    env_name: str = "HalfCheetah-v4"
    benchmark: str = "gym"             # "gym" | "dmc_proprio" | "dmc_visual" | "atari"
    seed: int = 0
    total_steps: int = 1_000_000
    eval_freq: int = 5_000
    eval_episodes: int = 10

    # ── Observation / action space ───────────────────────────────────────────
    obs_type: str = "vector"           # "vector" | "image"
    action_type: str = "continuous"    # "continuous" | "discrete"

    # ── Image-specific ───────────────────────────────────────────────────────
    image_size: int = 84
    frame_stack: int = 3               # 3 for DMC-visual, 4 for Atari
    action_repeat: int = 1             # 2 for DMC, 4 for Atari

    # ── Logging ──────────────────────────────────────────────────────────────
    log_freq: int = 1_000
    save_model: bool = False
    save_dir: str = "results"


# ── Benchmark-specific preset overrides ──────────────────────────────────────

GYM_CONFIG = MRQConfig(
    benchmark="gym",
    obs_type="vector",
    action_type="continuous",
    total_steps=1_000_000,
    eval_freq=5_000,
    action_repeat=1,
    frame_stack=1,
)

DMC_PROPRIO_CONFIG = MRQConfig(
    benchmark="dmc_proprio",
    obs_type="vector",
    action_type="continuous",
    total_steps=500_000,
    eval_freq=5_000,
    action_repeat=2,
    frame_stack=1,
)

DMC_VISUAL_CONFIG = MRQConfig(
    benchmark="dmc_visual",
    obs_type="image",
    action_type="continuous",
    total_steps=500_000,
    eval_freq=5_000,
    action_repeat=2,
    frame_stack=3,   # 3 RGB frames stacked → 9 channels
    image_size=84,
)

ATARI_CONFIG = MRQConfig(
    benchmark="atari",
    obs_type="image",
    action_type="discrete",
    total_steps=2_500_000,
    eval_freq=100_000,
    action_repeat=4,
    frame_stack=4,   # 4 grayscale frames stacked → 4 channels
    image_size=84,
)


BENCHMARK_CONFIGS = {
    "gym": GYM_CONFIG,
    "dmc_proprio": DMC_PROPRIO_CONFIG,
    "dmc_visual": DMC_VISUAL_CONFIG,
    "atari": ATARI_CONFIG,
}
