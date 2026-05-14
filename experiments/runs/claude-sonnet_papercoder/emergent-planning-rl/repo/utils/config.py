## utils/config.py
"""Configuration module for reproducing 'Interpreting Emergent Planning in Model-Free RL'.

This module defines all hyperparameters as typed dataclasses that mirror the YAML
configuration file structure. It is the single source of truth for all experiment
settings and has zero project-level dependencies.
"""

from __future__ import annotations

import pathlib
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import yaml


@dataclass
class AgentConfig:
    """DRC agent architecture hyperparameters (paper Sections 2.3, E.3)."""

    type: str = "drc"
    obs_channels: int = 7          # Symbolic observation channels: 8x8x7 one-hot
    hidden_dim: int = 32           # G_d = 32 channels for encoder and all ConvLSTM units
    num_layers: int = 3            # D = 3 ConvLSTM layers
    num_ticks: int = 3             # N = 3 internal ticks per environment step
    grid_h: int = 8                # H_d = 8, spatial height matches Sokoban grid
    grid_w: int = 8                # W_d = 8, spatial width matches Sokoban grid
    kernel_size: int = 3           # Convolutional kernel size
    padding: int = 1               # Single layer of input zero padding


@dataclass
class TrainingConfig:
    """IMPALA training hyperparameters (paper Section E.4)."""

    total_steps: int = 250_000_000       # 250 million transitions
    unroll_length: int = 20              # Propagation through time unroll length
    batch_size: int = 16
    learning_rate: float = 4.0e-4        # Initial LR, decays linearly to 0
    lr_schedule: str = "linear_decay"
    optimizer: str = "adam"              # Adam optimizer

    # V-trace off-policy correction
    gamma: float = 0.97                  # Discount rate
    lambda_vtrace: float = 0.97          # V-trace lambda
    clip_rho: float = 1.0                # V-trace rho clipping threshold
    clip_c: float = 1.0                  # V-trace c clipping threshold

    # Regularization
    l2_logits: float = 1.0e-3            # L2 penalty on action logits
    l2_heads: float = 1.0e-5             # L2 regularization on policy and value heads
    entropy_coef: float = 1.0e-2         # Entropy penalty strength on policy

    # Infrastructure
    n_envs: int = 16                     # Number of parallel environments
    checkpoint_every: int = 1_000_000    # Save checkpoint every 1M transitions
    checkpoint_dir: str = "checkpoints/drc33"

    # Action selection
    train_action: str = "sample"         # Sample from categorical during training
    eval_action: str = "greedy"          # Greedy (argmax) at test time


@dataclass
class EnvConfig:
    """Sokoban environment configuration (paper Sections 2.2, E.2)."""

    name: str = "sokoban"
    grid_size: int = 8
    n_boxes: int = 4
    n_targets: int = 4
    obs_type: str = "symbolic"
    max_steps_min: int = 115             # Episode ends at random step in [115, 120]
    max_steps_max: int = 120

    # Reward structure
    reward_step: float = -0.01
    reward_box_on_target: float = 1.0
    reward_box_off_target: float = -1.0
    reward_solved: float = 10.0

    n_actions: int = 5                   # {UP, DOWN, LEFT, RIGHT, NOOP}


@dataclass
class DataConfig:
    """Boxoban dataset configuration (paper Sections 2.3, E.4)."""

    data_dir: str = "data/boxoban"
    train_split: str = "unfiltered_train"
    val_split: str = "unfiltered_val"
    test_split: str = "unfiltered_test"
    medium_split: str = "medium"
    hard_split: str = "hard"
    n_medium_eval: int = 1000
    n_hard_eval: int = 1000


@dataclass
class ProbingConfig:
    """Linear probe training configuration (paper Sections 4.1, D.1)."""

    optimizer: str = "adamw"             # AdamW optimizer
    n_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-3
    n_seeds: int = 5                     # 5 unique initialization seeds

    probe_sizes: List[str] = field(
        default_factory=lambda: ["1x1", "3x3", "5x5", "7x7"]
    )

    tick: str = "final"                  # Use cell state after final tick N=3
    state_type: str = "cell"             # Cell state g_t^d, not output h_t^d

    # Dataset sizes for fully-trained agent
    n_train_episodes: int = 3000         # ~106.6k transitions
    n_val_episodes: int = 1000           # ~25.7k transitions

    # Dataset sizes for checkpoint probing
    n_train_episodes_ckpt: int = 1000
    n_val_episodes_ckpt: int = 500

    metric: str = "macro_f1"             # Macro F1 due to class imbalance
    n_classes: int = 5                   # {NEVER, UP, DOWN, LEFT, RIGHT}

    concepts: List[str] = field(
        default_factory=lambda: [
            "ca",           # Agent Approach Direction C_A
            "cb",           # Box Push Direction C_B
            "ca_binary",    # Simplified binary: {NEVER, AGAIN}
            "cb_binary",    # Simplified binary: {NEVER, AGAIN}
            "ca_exit",      # Agent Exit Direction - reversed asymmetry
            "cb_approach",  # Box Approach Direction - reversed asymmetry
        ]
    )

    save_dir: str = "probes/drc33"


@dataclass
class InterventionConfig:
    """Causal intervention experiment configuration (paper Sections 6.1, B.2)."""

    alpha: float = 1.0                   # Default intervention strength
    n_base_levels: int = 25              # Handcrafted base levels per type
    n_transformations: int = 8           # Vertical reflection + 0/90/180/270 rotations
    n_total_levels: int = 200            # 25 * 8 = 200 levels per intervention type
    n_seeds: int = 5

    default_p: int = 1                   # Default directional intervention squares
    max_p: int = 3                       # Maximum p for ablation study

    alpha_values: List[float] = field(
        default_factory=lambda: [0.25, 0.5, 1.0, 2.0, 4.0]
    )

    levels_dir: str = "data/intervention_levels"


@dataclass
class AnalysisConfig:
    """Analysis and evaluation configuration (paper Sections 5, 6.2, Appendix A, C)."""

    max_thinking_steps: int = 5          # Force agent stationary for 5 steps
    n_thinking_episodes: int = 1000

    corridor_lengths: List[int] = field(
        default_factory=lambda: [2, 6, 10, 14]
    )
    n_corridor_base: int = 8
    n_corridor_total: int = 80

    # Training emergence analysis
    emergence_start_step: int = 1_000_000
    emergence_end_step: int = 50_000_000
    emergence_checkpoint_interval: int = 1_000_000
    n_emergence_checkpoints: int = 50

    figures_dir: str = "figures/drc33"


@dataclass
class AgentVariantConfig:
    """Configuration for alternative agent variants (paper Appendices F, G)."""

    type: str = "drc"
    obs_channels: int = 7
    hidden_dim: int = 32
    num_layers: int = 3
    num_ticks: int = 3
    grid_h: int = 8
    grid_w: int = 8
    kernel_size: int = 3
    padding: int = 1
    total_steps: int = 100_000_000       # Variant-specific training budget
    intervention_alpha: float = 1.0      # Some variants need alpha=4


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration."""

    tensorboard_dir: str = "runs/drc33"
    log_every: int = 10_000
    eval_every: int = 1_000_000
    print_every: int = 100_000


@dataclass
class Config:
    """Top-level configuration container.

    Mirrors the structure of configs/drc33.yaml. All nested sections are
    typed dataclasses with sensible defaults so partial YAML files work.

    Usage:
        config = Config.load("configs/drc33.yaml")
        hidden_dim = config.agent.hidden_dim  # 32
        config.save("experiments/run_001/config.yaml")
    """

    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    data: DataConfig = field(default_factory=DataConfig)
    probing: ProbingConfig = field(default_factory=ProbingConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    agent_variants: Dict[str, AgentVariantConfig] = field(default_factory=dict)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, path: str) -> Config:
        """Load configuration from a YAML file.

        Missing sections or keys fall back to dataclass defaults, so partial
        YAML files are fully supported.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A fully populated Config instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            yaml.YAMLError: If the YAML file is malformed.
        """
        config_path = pathlib.Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(config_path, "r", encoding="utf-8") as f:
            raw: dict = yaml.safe_load(f) or {}

        # Build each nested section, falling back to defaults for missing keys.
        agent_cfg = _build_dataclass(AgentConfig, raw.get("agent", {}))
        training_cfg = _build_dataclass(TrainingConfig, raw.get("training", {}))
        env_cfg = _build_dataclass(EnvConfig, raw.get("env", {}))
        data_cfg = _build_dataclass(DataConfig, raw.get("data", {}))
        probing_cfg = _build_dataclass(ProbingConfig, raw.get("probing", {}))
        intervention_cfg = _build_dataclass(
            InterventionConfig, raw.get("intervention", {})
        )
        analysis_cfg = _build_dataclass(AnalysisConfig, raw.get("analysis", {}))
        logging_cfg = _build_dataclass(LoggingConfig, raw.get("logging", {}))

        # Agent variants is a dict of dicts; construct each as AgentVariantConfig.
        raw_variants: dict = raw.get("agent_variants", {})
        agent_variants: Dict[str, AgentVariantConfig] = {
            name: _build_dataclass(AgentVariantConfig, variant_dict)
            for name, variant_dict in raw_variants.items()
        }

        return cls(
            agent=agent_cfg,
            training=training_cfg,
            env=env_cfg,
            data=data_cfg,
            probing=probing_cfg,
            intervention=intervention_cfg,
            analysis=analysis_cfg,
            agent_variants=agent_variants,
            logging=logging_cfg,
        )

    def save(self, path: str) -> None:
        """Serialize the configuration to a YAML file.

        Creates parent directories as needed. Uses dataclasses.asdict() for
        recursive serialization of all nested dataclasses.

        Args:
            path: Destination path for the YAML file.
        """
        output_path = pathlib.Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = asdict(self)

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def get_agent_variant(self, variant_name: str) -> Optional[AgentVariantConfig]:
        """Retrieve a named agent variant configuration.

        Args:
            variant_name: Key in the agent_variants dict (e.g., 'drc_1_9').

        Returns:
            The AgentVariantConfig for the given variant, or None if not found.
        """
        return self.agent_variants.get(variant_name, None)

    def __repr__(self) -> str:
        """Human-readable summary of key configuration values."""
        return (
            f"Config("
            f"agent={self.agent.type}(D={self.agent.num_layers},"
            f"N={self.agent.num_ticks},G={self.agent.hidden_dim}), "
            f"training=steps={self.training.total_steps:,},"
            f"lr={self.training.learning_rate}, "
            f"probing=seeds={self.probing.n_seeds},"
            f"epochs={self.probing.n_epochs}"
            f")"
        )


def _build_dataclass(cls, raw_dict: dict):
    """Construct a dataclass instance from a raw dict, ignoring unknown keys.

    This allows YAML files to contain extra keys (e.g., comments-as-keys or
    future additions) without raising TypeError. Only keys that match the
    dataclass fields are passed to the constructor; all others are silently
    ignored. Missing keys use the dataclass field defaults.

    Args:
        cls: The dataclass class to instantiate.
        raw_dict: A plain dict (typically from yaml.safe_load) with config values.

    Returns:
        An instance of cls with values from raw_dict and defaults for missing keys.
    """
    import dataclasses

    valid_fields = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in raw_dict.items() if k in valid_fields}
    return cls(**filtered)
