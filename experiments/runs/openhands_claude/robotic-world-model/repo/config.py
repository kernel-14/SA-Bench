from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class RobotConfig:
    name: str
    obs_dim: int
    action_dim: int
    privileged_dim: int
    policy_obs_dim: int
    default_joint_positions: Optional[List[float]] = None


ANYMAL_D_CONFIG = RobotConfig(
    name="anymal_d",
    obs_dim=45,       # v(3) + omega(3) + g(3) + q(12) + q_dot(12) + tau(12)
    action_dim=12,    # joint position targets
    privileged_dim=8, # knee_contact(4) + foot_contact(4)
    policy_obs_dim=48, # v(3) + omega(3) + g(3) + cmd(3) + q(12) + q_dot(12) + a_prev(12)
)

UNITREE_G1_CONFIG = RobotConfig(
    name="unitree_g1",
    obs_dim=96,        # v(3) + omega(3) + g(3) + q(29) + q_dot(29) + tau(29)
    action_dim=29,     # joint position targets
    privileged_dim=30, # body_contact(26) + foot_height(2) + foot_velocity(2)
    policy_obs_dim=99, # v(3) + omega(3) + g(3) + cmd(3) + q(29) + q_dot(29) + a_prev(29)
)


@dataclass
class RWMArchConfig:
    gru_hidden_size: int = 256
    gru_num_layers: int = 2
    head_hidden_size: int = 128
    head_activation: str = "relu"


@dataclass
class MLPBaselineConfig:
    hidden_sizes: Tuple[int, ...] = (256, 256)
    activation: str = "relu"


@dataclass
class RSSMConfig:
    rnn_type: str = "gru"
    hidden_size: int = 256
    num_layers: int = 2
    latent_dim: int = 64       # total stochastic state dim = num_categories * category_size
    prior_type: str = "categorical"
    num_categories: int = 32   # number of categorical variables
    category_size: int = 32    # number of classes per categorical variable (stoch_dim = 32*32=1024)


@dataclass
class TransformerConfig:
    model_type: str = "decoder"
    d_model: int = 64
    num_heads: int = 8
    num_layers: int = 2
    context_length: int = 32
    positional_encoding: str = "sinusoidal"
    dropout: float = 0.0


@dataclass
class RWMTrainingConfig:
    step_time: float = 0.02
    max_iterations: int = 2500
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 1024
    history_horizon: int = 32   # M
    forecast_horizon: int = 8   # N
    forecast_decay: float = 1.0  # alpha
    num_seeds: int = 5
    device: str = "cuda"
    log_interval: int = 50
    save_interval: int = 500
    checkpoint_dir: str = "checkpoints"


@dataclass
class MBPOPPOConfig:
    imagination_envs: int = 4096
    imagination_steps: int = 100   # T
    step_time: float = 0.02
    buffer_size: int = 1000
    max_iterations: int = 2500
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    learning_epochs: int = 5
    num_mini_batches: int = 4
    kl_target: float = 0.01
    discount_factor: float = 0.99  # gamma
    clip_range: float = 0.2        # epsilon
    entropy_coef: float = 0.005
    gae_lambda: float = 0.95
    value_loss_coef: float = 0.5
    max_grad_norm: float = 1.0
    num_seeds: int = 5
    device: str = "cuda"
    log_interval: int = 10
    save_interval: int = 100
    checkpoint_dir: str = "checkpoints_policy"


@dataclass
class PolicyArchConfig:
    hidden_sizes: Tuple[int, ...] = (128, 128, 128)
    activation: str = "elu"
    log_std_init: float = 0.0
    log_std_min: float = -5.0
    log_std_max: float = 2.0


@dataclass
class ValueArchConfig:
    hidden_sizes: Tuple[int, ...] = (128, 128, 128)
    activation: str = "elu"


@dataclass
class RewardConfig:
    # ANYmal D reward weights
    anymal_d: dict = field(default_factory=lambda: {
        "w_vxy": 1.0,
        "w_wz": 0.5,
        "w_vz": -2.0,
        "w_wxy": -0.05,
        "w_qtau": -2.5e-5,
        "w_qddot": -2.5e-7,
        "w_adot": -0.01,
        "w_fa": 0.5,
        "w_c": -1.0,
        "w_g": -5.0,
        "w_fc": 0.0,
        "w_qd": 0.0,
        "sigma_vxy": 0.25,
        "sigma_wz": 0.25,
    })
    # Unitree G1 reward weights
    unitree_g1: dict = field(default_factory=lambda: {
        "w_vxy": 1.0,
        "w_wz": 0.5,
        "w_vz": -2.0,
        "w_wxy": -0.05,
        "w_qtau": -2.5e-5,
        "w_qddot": -2.5e-7,
        "w_adot": -0.05,
        "w_fa": 0.0,
        "w_c": -1.0,
        "w_g": -5.0,
        "w_fc": 1.0,
        "w_qd": -1.0,
        "sigma_vxy": 0.25,
        "sigma_wz": 0.25,
    })


@dataclass
class ExperimentConfig:
    robot: str = "anymal_d"  # "anymal_d" or "unitree_g1"
    rwm_arch: RWMArchConfig = field(default_factory=RWMArchConfig)
    mlp_baseline: MLPBaselineConfig = field(default_factory=MLPBaselineConfig)
    rssm: RSSMConfig = field(default_factory=RSSMConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    rwm_training: RWMTrainingConfig = field(default_factory=RWMTrainingConfig)
    mbpo_ppo: MBPOPPOConfig = field(default_factory=MBPOPPOConfig)
    policy_arch: PolicyArchConfig = field(default_factory=PolicyArchConfig)
    value_arch: ValueArchConfig = field(default_factory=ValueArchConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    seed: int = 42
    data_dir: str = "data"
    output_dir: str = "outputs"

    def get_robot_config(self) -> RobotConfig:
        if self.robot == "anymal_d":
            return ANYMAL_D_CONFIG
        elif self.robot == "unitree_g1":
            return UNITREE_G1_CONFIG
        else:
            raise ValueError(f"Unknown robot: {self.robot}")

    def get_reward_weights(self) -> dict:
        if self.robot == "anymal_d":
            return self.reward.anymal_d
        elif self.robot == "unitree_g1":
            return self.reward.unitree_g1
        else:
            raise ValueError(f"Unknown robot: {self.robot}")
