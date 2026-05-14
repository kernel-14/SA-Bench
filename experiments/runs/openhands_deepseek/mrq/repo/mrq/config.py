"""MR.Q hyperparameters — all values from Table 3 of the paper."""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class MRQConfig:
    # ---- Environment ----
    env_name: str = "HalfCheetah-v4"
    observation_type: str = "vector"  # "vector" or "image"
    discrete_actions: bool = False

    # ---- Training ----
    total_timesteps: int = 1_000_000  # 1M for Gym, 500k for DMC, 2.5M for Atari
    start_timesteps: int = 10_000  # initial random exploration
    replay_buffer_capacity: int = 1_000_000
    batch_size: int = 256
    replay_ratio: int = 1
    eval_freq: int = 5_000
    eval_episodes: int = 10
    seed: int = 0

    # ---- Encoder ----
    zs_dim: int = 512
    zsa_dim: int = 512
    za_dim: int = 256
    hidden_dim: int = 512
    encoder_lr: float = 1e-4
    encoder_weight_decay: float = 1e-4

    # Encoder horizons
    h_enc: int = 5  # H_Enc
    lambda_dynamics: float = 0.1
    lambda_reward: float = 0.1
    lambda_terminal: float = 0.1

    # Reward prediction
    reward_bins: int = 65
    reward_range: Tuple[float, float] = (-10.0, 10.0)

    # ---- Value ----
    value_lr: float = 3e-4
    value_weight_decay: float = 0.0
    h_q: int = 3  # multi-step returns horizon
    target_noise_std: float = 0.2
    target_noise_clip: float = 0.3
    discount: float = 0.99
    huber_delta: float = 1.0

    # ---- Policy ----
    policy_lr: float = 3e-4
    policy_weight_decay: float = 0.0
    lambda_pre_activ: float = 1e-5
    gumbel_softmax_tau: float = 10.0

    # ---- Exploration ----
    exploration_noise: float = 0.2

    # ---- LAP (Prioritized Experience Replay) ----
    lap_alpha: float = 0.4
    lap_min_priority: float = 1.0

    # ---- Target networks ----
    target_update_freq: int = 250

    # ---- Optimizer ----
    optimizer: str = "AdamW"
    gradient_clip_norm: float = 20.0

    # ---- Image preprocessing (Atari / DMC visual) ----
    image_size: int = 84
    image_channels: int = 3  # RGB for DMC visual, grayscale stack for Atari
    frame_stack: int = 1  # 3 for DMC visual (stacked), 4 for Atari
    action_repeat: int = 1  # 2 for DMC, 4 for Atari

    # ---- Activation functions ----
    encoder_activation: str = "ELU"
    value_activation: str = "ELU"
    policy_activation: str = "ReLU"


# Predefined configs for each benchmark
def gym_locomotion_config() -> MRQConfig:
    return MRQConfig(
        env_name="HalfCheetah-v4",
        observation_type="vector",
        discrete_actions=False,
        total_timesteps=1_000_000,
        action_repeat=1,
        frame_stack=1,
    )


def dmc_proprio_config() -> MRQConfig:
    return MRQConfig(
        env_name="cheetah_run",
        observation_type="vector",
        discrete_actions=False,
        total_timesteps=500_000,
        action_repeat=2,
        frame_stack=1,
    )


def dmc_visual_config() -> MRQConfig:
    return MRQConfig(
        env_name="cheetah_run",
        observation_type="image",
        discrete_actions=False,
        total_timesteps=500_000,
        action_repeat=2,
        frame_stack=3,
        image_channels=3,
    )


def atari_config() -> MRQConfig:
    return MRQConfig(
        env_name="Alien",
        observation_type="image",
        discrete_actions=True,
        total_timesteps=2_500_000,
        action_repeat=4,
        frame_stack=4,
        image_channels=1,
    )
