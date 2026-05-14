## config.py

import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional
import json
import os
import argparse

import yaml


# ----------------------------------------------------------------------
# Helper: convert a dictionary to a dataclass instance (recursive merging)
# ----------------------------------------------------------------------
def _dict_to_dataclass(cls, data: Optional[dict]):
    """Recursively construct a dataclass instance from a dictionary.
    If data is None or missing keys, use default values.
    """
    if data is None:
        return cls()  # use defaults
    field_map = {f.name: f for f in dataclasses.fields(cls)}
    kwargs = {}
    for name, f in field_map.items():
        if name in data:
            value = data[name]
            # If the field type is itself a dataclass and value is a dict, recurse
            if dataclasses.is_dataclass(f.type) and isinstance(value, dict):
                kwargs[name] = _dict_to_dataclass(f.type, value)
            else:
                kwargs[name] = value
        else:
            # Use default or default_factory
            if f.default is not dataclasses.MISSING:
                kwargs[name] = f.default
            else:
                kwargs[name] = f.default_factory()
    return cls(**kwargs)


# ----------------------------------------------------------------------
# Helper: set an attribute in a nested dataclass using dot‑notation
# ----------------------------------------------------------------------
def _set_nested_attr(obj, key_path: str, value_str: str):
    """Traverse obj via dot‑separated path and set the leaf attribute.
    Converts the string value to the type currently held by the leaf.
    """
    *parts, last = key_path.split('.')
    current = obj
    for part in parts:
        current = getattr(current, part)

    cur_val = getattr(current, last)
    # Determine conversion based on type of current value
    if isinstance(cur_val, bool):
        # Accept common boolean strings
        if value_str.lower() in ('true', '1', 'yes'):
            new_val = True
        elif value_str.lower() in ('false', '0', 'no'):
            new_val = False
        else:
            raise ValueError(f'Cannot convert "{value_str}" to bool')
    elif isinstance(cur_val, int):
        new_val = int(value_str)
    elif isinstance(cur_val, float):
        new_val = float(value_str)
    elif isinstance(cur_val, str):
        new_val = value_str
    elif isinstance(cur_val, list):
        # Try JSON list first, else comma‑separated
        value_str = value_str.strip()
        if value_str.startswith('['):
            new_val = json.loads(value_str)
        else:
            items = [item.strip() for item in value_str.split(',') if item.strip()]
            # Infer element type from current list (if not empty)
            if cur_val:
                elem_type = type(cur_val[0])
                new_val = [elem_type(item) for item in items]
            else:
                new_val = items  # keep as strings
    else:
        # Fallback: attempt conversion to the current type
        try:
            new_val = type(cur_val)(value_str)
        except Exception:
            raise ValueError(f'Cannot convert "{value_str}" to {type(cur_val)}')
    setattr(current, last, new_val)


# ----------------------------------------------------------------------
# Nested configuration dataclasses
# ----------------------------------------------------------------------
@dataclass
class CuriosityConfig:
    """Configuration for the ICM‑based curiosity relevance function."""
    encoder_hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    forward_dynamics_hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    learning_rate: float = 3.0e-4
    update_fraction: float = 0.05               # fraction of policy updates where ICM is trained


@dataclass
class EnvironmentConfig:
    """Environment‑related settings."""
    task_name: str = "cheetah-run"
    state_based: bool = True
    total_env_steps: int = 100_000
    random_seed: int = 0


@dataclass
class ReplayBufferConfig:
    """Replay buffer capacities and sampling parameters."""
    real_capacity: int = 1_000_000
    synthetic_capacity: int = 1_000_000
    synthetic_ratio: float = 0.5
    batch_size: int = 256
    real_per_batch: int = 128                  # kept constant when scaling batch size


@dataclass
class PolicyConfig:
    """Configuration for the policy learning algorithm (REDQ / DRQ‑v2)."""
    algorithm: str = "redq"                    # "redq" or "drqv2"
    utd_ratio: int = 20
    learning_rate: float = 3.0e-4
    discount_gamma: float = 0.99
    polyak_tau: float = 0.005
    critic_ensemble_size: int = 5
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"


@dataclass
class RelevanceConfig:
    """Selects and configures the relevance function."""
    type: str = "curiosity"                    # one of: "reward","return","td_error","curiosity"
    curiosity: CuriosityConfig = field(default_factory=CuriosityConfig)


@dataclass
class DiffusionConfig:
    """Diffusion model (conditional) configuration."""
    denoising_steps: int = 1000
    noise_schedule: str = "linear"
    batch_size: int = 256
    learning_rate: float = 1.0e-4
    inner_train_steps: int = 5000             # gradient steps per inner‑loop retrain
    uncond_prob: float = 0.25                 # CFG dropout probability
    guidance_scale: float = 1.0               # classifier‑free guidance strength ω
    condition_dim: int = 1                    # scalar relevance value
    prompt_ratio: float = 0.5                 # fraction of top‑k real transitions used for conditioning during generation


@dataclass
class PGRAlgorithmConfig:
    """Outer / inner loop schedule."""
    inner_loop_frequency: int = 10_000        # retrain diffusion every I env steps
    synthetic_samples_per_loop: int = 1_000_000
    eval_interval: int = 5_000                # evaluate policy every N environment steps
    eval_episodes: int = 10


@dataclass
class LoggingConfig:
    """Logging and checkpointing settings."""
    use_wandb: bool = False
    project_name: str = "PGR"
    run_name: str = "default"
    checkpoint_dir: str = "checkpoints"


# ----------------------------------------------------------------------
# Top‑level configuration aggregating all sections
# ----------------------------------------------------------------------
@dataclass
class Config:
    """Master configuration for a PGR experiment."""
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    relevance: RelevanceConfig = field(default_factory=RelevanceConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    pgr_algorithm: PGRAlgorithmConfig = field(default_factory=PGRAlgorithmConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file, merging with defaults."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        # data should be a dict whose keys match the top‑level sections.
        return _dict_to_dataclass(Config, data)


# ----------------------------------------------------------------------
# Command‑line interface
# ----------------------------------------------------------------------
def parse_args() -> Config:
    """Parse command‑line arguments and return a Config instance.
    Usage:
        python main.py --config config.yaml --set environment.total_env_steps=200000
    """
    parser = argparse.ArgumentParser(description="Prioritized Generative Replay (PGR) experiment")
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (defaults used if not provided)')
    parser.add_argument('--set', type=str, action='append', default=[],
                        help='Override config values using dot‑notation, e.g. "environment.total_env_steps=200000". '
                             'Can be repeated.')
    args = parser.parse_args()

    # Base configuration
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    # Apply overrides
    for override in args.set:
        if '=' not in override:
            raise ValueError(f"Invalid --set argument: {override}. Expected key=value.")
        key, value = override.split('=', 1)
        _set_nested_attr(config, key.strip(), value.strip())

    return config


# Simple test when run directly
if __name__ == '__main__':
    cfg = parse_args()
    print(cfg)
