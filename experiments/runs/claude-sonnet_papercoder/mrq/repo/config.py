## Code: config.py

```python
## config.py
"""Configuration dataclass for MR.Q (Model-based Representations for Q-learning).

This module is the single source of truth for all hyperparameters and experiment
settings. All other modules receive a Config instance and read from it directly.
No magic numbers should appear outside this file.

All default values match Table 3 (Appendix B.1) of the MR.Q paper.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import ClassVar, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Normalization reference scores (Appendix B.3)
# ---------------------------------------------------------------------------

GYM_RANDOM_SCORES: Dict[str, float] = {
    "Ant-v4": -70.288,
    "HalfCheetah-v4": -289.415,
    "Hopper-v4": 18.791,
    "Humanoid-v4": 120.423,
    "Walker2d-v4": 2.791,
}

GYM_TD3_SCORES: Dict[str, float] = {
    "Ant-v4": 3942.0,
    "HalfCheetah-v4": 10574.0,
    "Hopper-v4": 3226.0,
    "Humanoid-v4": 5165.0,
    "Walker2d-v4": 3946.0,
}

ATARI_RANDOM_SCORES: Dict[str, float] = {
    "Alien": 227.8,
    "Amidar": 5.8,
    "Assault": 222.4,
    "Asterix": 210.0,
    "Asteroids": 719.1,
    "Atlantis": 12850.0,
    "BankHeist": 14.2,
    "BattleZone": 2360.0,
    "BeamRider": 363.9,
    "Berzerk": 123.7,
    "Bowling": 23.1,
    "Boxing": 0.1,
    "Breakout": 1.7,
    "Centipede": 2090.9,
    "ChopperCommand": 811.0,
    "CrazyClimber": 10780.5,
    "Defender": 2874.5,
    "DemonAttack": 152.1,
    "DoubleDunk": -18.6,
    "Enduro": 0.0,
    "FishingDerby": -91.7,
    "Freeway": 0.0,
    "Frostbite": 65.2,
    "Gopher": 257.6,
    "Gravitar": 173.0,
    "Hero": 1027.0,
    "IceHockey": -11.2,
    "Jamesbond": 29.0,
    "Kangaroo": 52.0,
    "Krull": 1598.0,
    "KungFuMaster": 258.5,
    "MontezumaRevenge": 0.0,
    "MsPacman": 307.3,
    "NameThisGame": 2292.3,
    "Phoenix": 761.4,
    "Pitfall": -229.4,
    "Pong": -20.7,
    "PrivateEye": 24.9,
    "Qbert": 163.9,
    "Riverraid": 1338.5,
    "RoadRunner": 11.5,
    "Robotank": 2.2,
    "Seaquest": 68.4,
    "Skiing": -17098.1,
    "Solaris": 1236.3,
    "SpaceInvaders": 148.0,
    "StarGunner": 664.0,
    "Surround": -10.0,
    "Tennis": -23.8,
    "TimePilot": 3568.0,
    "Tutankham": 11.4,
    "UpNDown": 533.4,
    "Venture": 0.0,
    "VideoPinball": 16256.9,
    "WizardOfWor": 563.5,
    "YarsRevenge": 3092.9,
    "Zaxxon": 32.5,
}

ATARI_HUMAN_SCORES: Dict[str, float] = {
    "Alien": 7127.7,
    "Amidar": 1719.5,
    "Assault": 742.0,
    "Asterix": 8503.3,
    "Asteroids": 47388.7,
    "Atlantis": 29028.1,
    "BankHeist": 753.1,
    "BattleZone": 37187.5,
    "BeamRider": 16926.5,
    "Berzerk": 2630.4,
    "Bowling": 160.7,
    "Boxing": 12.1,
    "Breakout": 30.5,
    "Centipede": 12017.0,
    "ChopperCommand": 7387.8,
    "CrazyClimber": 35829.4,
    "Defender": 18688.9,
    "DemonAttack": 1971.0,
    "DoubleDunk": -16.4,
    "Enduro": 860.5,
    "FishingDerby": -38.7,
    "Freeway": 29.6,
    "Frostbite": 4334.7,
    "Gopher": 2412.5,
    "Gravitar": 3351.4,
    "Hero": 30826.4,
    "IceHockey": 0.9,
    "Jamesbond": 302.8,
    "Kangaroo": 3035.0,
    "Krull": 2665.5,
    "KungFuMaster": 22736.3,
    "MontezumaRevenge": 4753.3,
    "MsPacman": 6951.6,
    "NameThisGame": 8049.0,
    "Phoenix": 7242.6,
    "Pitfall": 6463.7,
    "Pong": 14.6,
    "PrivateEye": 69571.3,
    "Qbert": 13455.0,
    "Riverraid": 17118.0,
    "RoadRunner": 7845.0,
    "Robotank": 11.9,
    "Seaquest": 42054.7,
    "Skiing": -4336.9,
    "Solaris": 12326.7,
    "SpaceInvaders": 1668.7,
    "StarGunner": 10250.0,
    "Surround": 6.5,
    "Tennis": -8.3,
    "TimePilot": 5229.2,
    "Tutankham": 167.6,
    "UpNDown": 11693.2,
    "Venture": 1187.5,
    "VideoPinball": 17667.9,
    "WizardOfWor": 4756.5,
    "YarsRevenge": 54576.9,
    "Zaxxon": 9173.3,
}

# Valid benchmark identifiers
VALID_BENCHMARKS: Tuple[str, ...] = (
    "gym",
    "dmc_proprio",
    "dmc_visual",
    "atari",
)

# Valid ablation variant names (matching config.yaml ablations section)
VALID_ABLATIONS: Tuple[str, ...] = (
    "none",
    "linear_value",
    "dynamics_target",
    "no_target_encoder",
    "revert",
    "nonlinear_model",
    "mse_reward",
    "no_reward_scaling",
    "no_min",
    "no_lap",
    "no_mr",
    "one_step_return",
    "no_unroll",
)


@dataclass
class Config:
    """Flat configuration dataclass for MR.Q.

    All fields have defaults matching the paper's Table 3 hyperparameters.
    Benchmark-specific overrides are applied via get_env_config().
    Ablation variants are applied via from_dict() or _apply_ablation().

    Attributes:
        env_name: Specific environment identifier (e.g., 'HalfCheetah-v4').
        benchmark: Benchmark category, one of 'gym', 'dmc_proprio',
            'dmc_visual', 'atari'.
        seed: Random seed for reproducibility.
        total_steps: Total environment interaction steps for training.
        eval_freq: Frequency (in steps) at which evaluation is performed.
        eval_episodes: Number of episodes per evaluation.
        replay_capacity: Maximum number of transitions in replay buffer.
        batch_size: Minibatch size for gradient updates.
        discount: Discount factor gamma.
        target_update_freq: Steps between synchronized target network updates
            (T_target). This controls when target networks, reward scaling,
            and encoder target are all synced simultaneously.
        replay_ratio: Gradient updates per environment step (assumed 1).
        enc_horizon: Encoder unroll horizon H_Enc.
        hq_horizon: Multi-step return horizon H_Q.
        lambda_reward: Weight for categorical reward loss in encoder.
        lambda_dynamics: Weight for dynamics MSE loss in encoder.
        lambda_terminal: Weight for terminal MSE loss in encoder. Note:
            agent.py multiplies by 0 until first terminal is seen.
        enc_lr: AdamW learning rate for encoder networks.
        enc_weight_decay: AdamW weight decay for encoder networks.
        value_lr: AdamW learning rate for value networks.
        policy_lr: AdamW learning rate for policy network.
        grad_clip_norm: Gradient clipping norm applied to value network.
        lambda_pre_activ: Pre-activation regularization weight for policy.
        gumbel_tau: Gumbel-Softmax temperature for discrete action spaces.
        zs_dim: State embedding dimension.
        za_dim: Action embedding dimension (internal to StateActionEncoder).
        zsa_dim: State-action embedding dimension.
        hidden_dim: Hidden layer width for all MLPs.
        reward_bins: Number of bins for categorical reward representation.
        reward_range: Symlog-space bounds for reward bins as (low, high).
            Effective raw range is symexp(+/-10) ~= +/-22026.
        target_noise_std: Std for target policy smoothing noise (sigma).
        target_noise_clip: Clipping bound c for target policy noise.
        explore_noise_std: Std for exploration noise.
        initial_random_steps: Random exploration steps before learning.
        lap_alpha: LAP probability smoothing exponent.
        lap_min_priority: LAP minimum priority floor.
        action_repeat: Number of times each action is repeated in env.
        frame_stack: Number of frames stacked as observation.
        image_obs: Whether observations are images (True) or vectors (False).
        discrete: Whether action space is discrete (True) or continuous.
        ablation: Active ablation variant name for logging.
        value_linear: If True, use linear value function (ablation).
        use_sa_dynamics_target: If True, use z_{s'a'} as dynamics target.
        use_target_encoder: If False, use current encoder for dynamics target.
        nonlinear_model: If True, replace linear MDP predictor with MLPs.
        use_mse_reward: If True, use MSE reward loss instead of cross-entropy.
        use_reward_scaling: If False, r_bar = r_bar_prime = 1.0 always.
        use_min_q: If False, use mean of Q-networks instead of minimum.
        use_lap: If False, use uniform sampling with MSE value loss.
        use_encoder_loss: If False, train encoder end-to-end with value.
    """

    # -----------------------------------------------------------------------
    # Environment identification
    # -----------------------------------------------------------------------
    env_name: str = "HalfCheetah-v4"
    benchmark: str = "gym"
    seed: int = 0

    # -----------------------------------------------------------------------
    # Training duration and evaluation
    # -----------------------------------------------------------------------
    total_steps: int = 1_000_000
    eval_freq: int = 5_000
    eval_episodes: int = 10

    # -----------------------------------------------------------------------
    # Replay buffer
    # -----------------------------------------------------------------------
    replay_capacity: int = 1_000_000
    batch_size: int = 256
    replay_ratio: int = 1

    # -----------------------------------------------------------------------
    # Core RL
    # -----------------------------------------------------------------------
    discount: float = 0.99
    target_update_freq: int = 250

    # -----------------------------------------------------------------------
    # Encoder hyperparameters (Table 3)
    # -----------------------------------------------------------------------
    enc_horizon: int = 5
    lambda_reward: float = 0.1
    lambda_dynamics: float = 1.0
    lambda_terminal: float = 0.1
    enc_lr: float = 1e-4
    enc_weight_decay: float = 1e-4

    # -----------------------------------------------------------------------
    # Value network hyperparameters (Table 3)
    # -----------------------------------------------------------------------
    hq_horizon: int = 3
    value_lr: float = 3e-4
    grad_clip_norm: float = 20.0

    # -----------------------------------------------------------------------
    # Policy network hyperparameters (Table 3)
    # -----------------------------------------------------------------------
    lambda_pre_activ: float = 1e-5
    policy_lr: float = 3e-4
    gumbel_tau: float = 10.0

    # -----------------------------------------------------------------------
    # TD3 noise parameters (Table 3)
    # -----------------------------------------------------------------------
    target_noise_std: float = 0.2
    target_noise_clip: float = 0.3
    explore_noise_std: float = 0.2

    # -----------------------------------------------------------------------
    # LAP prioritized replay (Table 3)
    # -----------------------------------------------------------------------
    lap_alpha: float = 0.4
    lap_min_priority: float = 1.0

    # -----------------------------------------------------------------------
    # Exploration (Table 3)
    # -----------------------------------------------------------------------
    initial_random_steps: int = 10_000

    # -----------------------------------------------------------------------
    # Network architecture (Table 3)
    # -----------------------------------------------------------------------
    zs_dim: int = 512
    za_dim: int = 256
    zsa_dim: int = 512
    hidden_dim: int = 512
    reward_bins: int = 65
    reward_range: Tuple[float, float] = (-10.0, 10.0)

    # -----------------------------------------------------------------------
    # Environment-specific derived fields (set by get_env_config)
    # -----------------------------------------------------------------------
    action_repeat: int = 1
    frame_stack: int = 1
    image_obs: bool = False
    discrete: bool = False

    # -----------------------------------------------------------------------
    # Ablation control flags
    # -----------------------------------------------------------------------
    ablation: str = "none"
    value_linear: bool = False
    use_sa_dynamics_target: bool = False
    use_target_encoder: bool = True
    nonlinear_model: bool = False
    use_mse_reward: bool = False
    use_reward_scaling: bool = True
    use_min_q: bool = True
    use_lap: bool = True
    use_encoder_loss: bool = True

    def __post_init__(self) -> None:
        """Validate configuration fields after initialization.

        Raises:
            ValueError: If any field has an invalid value.
        """
        if self.benchmark not in VALID_BENCHMARKS:
            raise ValueError(
                f"Invalid benchmark '{self.benchmark}'. "
                f"Must be one of {VALID_BENCHMARKS}."
            )

        if self.ablation not in VALID_ABLATIONS:
            raise ValueError(
                f"Invalid ablation '{self.ablation}'. "
                f"Must be one of {VALID_ABLATIONS}."
            )

        if len(self.reward_range) != 2:
            raise ValueError(
                f"reward_range must be a tuple of length 2, "
                f"got length {len(self.reward_range)}."
            )

        if self.reward_range[0] >= self.reward_range[1]:
            raise ValueError(
                f"reward_range[0] must be less than reward_range[1], "
                f"got {self.reward_range}."
            )

        # Validate positive-valued hyperparameters
        _positive_fields = {
            "total_steps": self.total_steps,
            "eval_freq": self.eval_freq,
            "eval_episodes": self.eval_episodes,
            "replay_capacity": self.replay_capacity,
            "batch_size": self.batch_size,
            "target_update_freq": self.target_update_freq,
            "enc_horizon": self.enc_horizon,
            "hq_horizon": self.hq_horizon,
            "enc_lr": self.enc_lr,
            "enc_weight_decay": self.enc_weight_decay,
            "value_lr": self.value_lr,
            "policy_lr": self.policy_lr,
            "grad_clip_norm": self.grad_clip_norm,
            "gumbel_tau": self.gumbel_tau,
            "target_noise_std": self.target_noise_std,
            "target_noise_clip": self.target_noise_clip,
            "explore_noise_std": self.explore_noise_std,
            "lap_alpha": self.lap_alpha,
            "lap_min_priority": self.lap_min_priority,
            "initial_random_steps": self.initial_random_steps,
            "zs_dim": self.zs_dim,
            "za_dim": self.za_dim,
            "zsa_dim": self.zsa_dim,
            "hidden_dim": self.hidden_dim,
            "reward_bins": self.reward_bins,
            "discount": self.discount,
        }
        for name, value in _positive_fields.items():
            if value <= 0:
                raise ValueError(
                    f"Field '{name}' must be positive, got {value}."
                )

        # Validate non-negative loss weights
        _nonneg_fields = {
            "lambda_reward": self.lambda_reward,
            "lambda_dynamics": self.lambda_dynamics,
            "lambda_terminal": self.lambda_terminal,
            "lambda_pre_activ": self.lambda_pre_activ,
        }
        for name, value in _nonneg_fields.items():
            if value < 0:
                raise ValueError(
                    f"Field '{name}' must be non-negative, got {value}."
                )

        if not (0.0 < self.discount < 1.0):
            raise ValueError(
                f"discount must be in (0, 1), got {self.discount}."
            )

        if not (0.0 < self.lap_alpha <= 1.0):
            raise ValueError(
                f"lap_alpha must be in (0, 1], got {self.lap_alpha}."
            )

    def _apply_ablation(self, ablation_name: str) -> None:
        """Apply ablation variant overrides to this Config instance.

        Maps ablation variant names to their corresponding field overrides
        as specified in config.yaml's ablations section.

        Args:
            ablation_name: Name of the ablation variant to apply.

        Raises:
            ValueError: If ablation_name is not a valid variant.
        """
        if ablation_name not in VALID_ABLATIONS:
            raise ValueError(
                f"Invalid ablation '{ablation_name}'. "
                f"Must be one of {VALID_ABLATIONS}."
            )

        self.ablation = ablation_name

        if ablation_name == "none":
            pass  # No overrides for the default configuration

        elif ablation_name == "linear_value":
            self.value_linear = True

        elif ablation_name == "dynamics_target":
            self.use_sa_dynamics_target = True

        elif ablation_name == "no_target_encoder":
            self.use_target_encoder = False

        elif ablation_name == "revert":
            # All three relaxations simultaneously
            self.value_linear = True
            self.use_sa_dynamics_target = True
            self.use_target_encoder = False

        elif ablation_name == "nonlinear_model":
            self.nonlinear_model = True

        elif ablation_name == "mse_reward":
            self.use_mse_reward = True

        elif ablation_name == "no_reward_scaling":
            self.use_reward_scaling = False

        elif ablation_name == "no_min":
            self.use_min_q = False

        elif ablation_name == "no_lap":
            self.use_lap = False

        elif ablation_name == "no_mr":
            self.use_encoder_loss = False

        elif ablation_name == "one_step_return":
            self.hq_horizon = 1

        elif ablation_name == "no_unroll":
            self.enc_horizon = 1

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Construct a Config from a flat or partially-nested dictionary.

        Handles input from argparse (flat dict via vars(args)) or from a
        pre-processed YAML dict. Applies ablation overrides if the 'ablation'
        key is present and not 'none'.

        Type coercion is applied for fields that may arrive as strings from
        argparse. The 'reward_range' field is handled specially to convert
        from list to tuple.

        Args:
            d: Dictionary of configuration values. Keys should match Config
                field names. Unknown keys are silently ignored.

        Returns:
            A fully initialized Config instance with all specified overrides
            applied.
        """
        # Start with default values
        cfg = cls()

        # Field type map for explicit coercion
        field_types: Dict[str, type] = {
            f.name: f.type if isinstance(f.type, type) else type(getattr(cfg, f.name))
            for f in dataclasses.fields(cfg)
        }

        # Apply all matching keys from the input dict
        for key, value in d.items():
            if not hasattr(cfg, key):
                continue  # Silently ignore unknown keys

            if key == "reward_range":
                # Convert list/tuple to tuple[float, float]
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    object.__setattr__(cfg, key, (float(value[0]), float(value[1])))
                continue

            if key == "ablation":
                # Defer ablation application until after all other fields
                continue

            current_value = getattr(cfg, key)
            # Coerce type based on the current default value's type
            try:
                if isinstance(current_value, bool):
                    # bool must be checked before int (bool is subclass of int)
                    if isinstance(value, str):
                        coerced = value.lower() in ("true", "1", "yes")
                    else:
                        coerced = bool(value)
                elif isinstance(current_value, int):
                    coerced = int(value)
                elif isinstance(current_value, float):
                    coerced = float(value)
                elif isinstance(current_value, str):
                    coerced = str(value)
                else:
                    coerced = value
                object.__setattr__(cfg, key, coerced)
            except (ValueError, TypeError):
                # Keep default if coercion fails
                pass

        # Apply ablation overrides last (may override other fields)
        ablation_name: str = str(d.get("ablation", "none"))
        if ablation_name and ablation_name != "none":
            cfg._apply_ablation(ablation_name)
        else:
            object.__setattr__(cfg, "ablation", ablation_name)

        # Re-run validation after all overrides
        cfg.__post_init__()

        return cfg

    def to_dict(self) -> dict:
        """Serialize this Config to a plain dictionary.

        Returns a flat dictionary of all field names and their current values.
        The 'reward_range' tuple is converted to a list for JSON serialization
        compatibility.

        Returns:
            Dictionary mapping field names to their values.
        """
        result: dict = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, tuple):
                value = list(value)
            result[f.name] = value
        return result

    @staticmethod
    def get_env_config(benchmark: str, env_name: str) -> dict:
        """Return benchmark-specific configuration overrides.

        These overrides are applied on top of the default Config values to
        set environment-specific parameters like action_repeat, frame_stack,
        image_obs, discrete, total_steps, and eval_freq.

        Args:
            benchmark: Benchmark category, one of 'gym', 'dmc_proprio',
                'dmc_visual', 'atari'.
            env_name: Specific environment name (currently unused but
                included for future per-environment overrides).

        Returns:
            Dictionary of field overrides to apply to a Config instance.

        Raises:
            ValueError: If benchmark is not a valid benchmark identifier.
        """
        if benchmark not in VALID_BENCHMARKS:
            raise ValueError(
                f"Invalid benchmark '{benchmark}'. "
                f"Must be one of {VALID_BENCHMARKS}."
            )

        if benchmark == "gym":
            return {
                "total_steps": 1_000_000,
                "eval_freq": 5_000,
                "action_repeat": 1,
                "frame_stack": 1,
                "image_obs": False,
                "discrete": False,
            }

        elif benchmark == "dmc_proprio":
            return {
                "total_steps": 500_000,
                "eval_freq": 5_000,
                "action_repeat": 2,
                "frame_stack": 1,
                "image_obs": False,
                "discrete": False,
            }

        elif benchmark == "dmc_visual":
            return {
                "total_steps": 500_000,
                "eval_freq": 5_000,
                "action_repeat": 2,
                "frame_stack": 3,
                "image_obs": True,
                "discrete": False,
            }

        elif benchmark == "atari":
            return {
                "total_steps": 2_500_000,
                "eval_freq": 100_000,
                "action_repeat": 4,
                "frame_stack": 4,
                "image_obs": True,
                "discrete": True,
            }

        # Unreachable due to validation above, but satisfies type checkers
        return {}

    def apply_env_config(self) -> None:
        """Apply benchmark-specific overrides in-place.

        Calls get_env_config with the current benchmark and env_name fields
        and updates this Config instance accordingly. Should be called after
        setting benchmark and env_name but before constructing any components.
        """
        overrides = self.get_env_config(self.benchmark, self.env_name)
        for key, value in overrides.items():
            if hasattr(self, key):
                object.__setattr__(self, key, value)

    def __repr__(self) -> str:
        """Return a human-readable string representation of the Config.

        Groups fields by category for readability.

        Returns:
            Multi-line string with all configuration fields.
        """
        lines = ["Config("]
        lines.append(f"  # Environment")
        lines.append(f"  env_name={self.env_name!r},")
        lines.append(f"  benchmark={self.benchmark!r},")
        lines.append(f"  seed={self.seed},")
        lines.append(f"  # Training")
        lines.append(f"  total_steps={self.total_steps},")
        lines.append(f"  eval_freq={self.eval_freq},")
        lines.append(f"  eval_episodes={self.eval_episodes},")
        lines.append(f"  replay_capacity={self.replay_capacity},")
        lines.append(f"  batch_size={self.batch_size},")
        lines.append(f"  discount={self.discount},")
        lines.append(f"  target_update_freq={self.target_update_freq},")
        lines.append(f"  replay_ratio={self.replay_ratio},")
        lines.append(f"  # Encoder")
        lines.append(f"  enc_horizon={self.enc_horizon},")
        lines.append(f"  lambda_reward={self.lambda_reward},")
        lines.append(f"  lambda_dynamics={self.lambda_dynamics},")
        lines.append(f"  lambda_terminal={self.lambda_terminal},")
        lines.append(f"  enc_lr={self.enc_lr},")
        lines.append(f"  enc_weight_decay={self.enc_weight_decay},")
        lines.append(f"  # Value")
        lines.append(f"  hq_horizon={self.hq_horizon},")
        lines.append(f"  value_lr={self.value_lr},")
        lines.append(f"  grad_clip_norm={self.grad_clip_norm},")
        lines.append(f"  # Policy")
        lines.append(f"  lambda_pre_activ={self.lambda_pre_activ},")
        lines.append(f"  policy_lr={self.policy_lr},")
        lines.append(f"  gumbel_tau={self.gumbel_tau},")
        lines.append(f"  # Architecture")
        lines.append(f"  zs_dim={self.zs_dim},")
        lines.append(f"  za_dim={self.za_dim},")
        lines.append(f"  zsa_dim={self.zsa_dim},")
        lines.append(f"  hidden_dim={self.hidden_dim},")
        lines.append(f"  reward_bins={self.reward_bins},")
        lines.append(f"  reward_range={self.reward_range},")
        lines.append(f"  # TD3 noise")
        lines.append(f"  target_noise_std={self.target_noise_std},")
        lines.append(f"  target_noise_clip={self.target_noise_clip},")
        lines.append(f"  explore_noise_std={self.explore_noise_std},")
        lines.append(f"  # LAP")
        lines.append(f"  lap_alpha={self.lap_alpha},")
        lines.append(f"  lap_min_priority={self.lap_min_priority},")
        lines.append(f"  # Exploration")
        lines.append(f"  initial_random_steps={self.initial_random_steps},")
        lines.append(f"  # Environment-specific")
        lines.append(f"  action_repeat={self.action_repeat},")
        lines.append(f"  frame_stack={self.frame_stack},")
        lines.append(f"  image_obs={self.image_obs},")
        lines.append(f"  discrete={self.discrete},")
        lines.append(f"  # Ablation")
        lines.append(f"  ablation={self.ablation!r},")
        lines.append(f"  value_linear={self.value_linear},")
        lines.append(f"  use_sa_dynamics_target={self.use_sa_dynamics_target},")
        lines.append(f"  use_target_encoder={self.use_target_encoder},")
        lines.append(f"  nonlinear_model={self.nonlinear_model},")
        lines.append(f"  use_mse_reward={self.use_mse_reward},")
        lines.append(f"  use_reward_scaling={self.use_reward_scaling},")
        lines.append(f"  use