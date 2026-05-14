from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Config:
    """Configuration for MR.Q experiments and training.

    This dataclass holds all hyperparameters and experimental settings,
    serving as a single source of truth across the project.
    Default values are loaded from the provided config.yaml.
    """
    # Training loop parameters
    total_timesteps: int = 1000000
    initial_random_steps: int = 10000
    replay_buffer_capacity: int = 1000000
    minibatch_size: int = 256
    discount_factor: float = 0.99
    target_update_frequency: int = 250

    # Loss function and model-based representation parameters
    encoder_horizon: int = 5
    value_horizon: int = 3
    lambda_reward: float = 0.1
    lambda_dynamics: float = 1.0  # Assumed default from config.yaml as it was blank in paper's table
    lambda_terminal: float = 0.1
    lambda_pre_activ: float = 0.00001

    # Optimizer settings
    learning_rate_encoders: float = 0.0001
    learning_rate_rl: float = 0.0003
    weight_decay: float = 0.0001
    grad_clip_norm: float = 20.0

    # Policy and exploration
    policy_noise_std: float = 0.2
    policy_noise_clip: float = 0.3

    # Prioritized Experience Replay (PER) settings
    prioritized_replay_alpha: float = 0.4

    # Reward processing (Categorical)
    reward_bins: int = 65
    reward_range: Tuple[float, float] = (-10.0, 10.0)

    # Environment settings
    env_name: str = "default_env"
    seed: int = 0
    image_obs: bool = False
    action_repeat: int = 1
    frame_stack: int = 1

    # Logging and evaluation
    log_interval: int = 1000
    eval_interval: int = 5000
    checkpoint_interval: int = 100000
