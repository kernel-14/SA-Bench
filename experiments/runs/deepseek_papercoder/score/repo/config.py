"""
Configuration module for SCoRe reproduction.
Defines a hierarchical dataclass structure to mirror config.yaml
and provides a loader with optional command-line override support.
"""

import dataclasses
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Any
import yaml
import sys
import os


# --------------------------------------------------------------------------- #
# Sub‑configuration dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class ModelConfig:
    name: str
    max_seq_length: int = 2048

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(
            name=d["name"],
            max_seq_length=d.get("max_seq_length", 2048),
        )


@dataclass
class StageConfig:
    steps: int
    batch_size: int
    learning_rate: float
    temperature: float

    @classmethod
    def from_dict(cls, d: dict) -> "StageConfig":
        return cls(
            steps=d["steps"],
            batch_size=d["batch_size"],
            learning_rate=d["learning_rate"],
            temperature=d["temperature"],
        )


@dataclass
class OfflineConfig:
    mixing_probability: float = 0.5
    num_samples_per_prompt: int = 4

    @classmethod
    def from_dict(cls, d: dict) -> "OfflineConfig":
        if not d:
            return cls()
        return cls(
            mixing_probability=d.get("mixing_probability", 0.5),
            num_samples_per_prompt=d.get("num_samples_per_prompt", 4),
        )


@dataclass
class RewardShapingConfig:
    alpha: float

    @classmethod
    def from_dict(cls, d: dict) -> "RewardShapingConfig":
        return cls(alpha=d["alpha"])


@dataclass
class KLPenaltyConfig:
    beta1_default: float
    beta2_stage1_first_turn: float

    @classmethod
    def from_dict(cls, d: dict) -> "KLPenaltyConfig":
        return cls(
            beta1_default=d["beta1_default"],
            beta2_stage1_first_turn=d["beta2_stage1_first_turn"],
        )


@dataclass
class DatasetConfig:
    subset: str
    extra_from_test: Optional[int] = None
    num_test: Optional[int] = None
    prompt_style: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetConfig":
        return cls(
            subset=d["subset"],
            extra_from_test=d.get("extra_from_test"),
            num_test=d.get("num_test"),
            prompt_style=d.get("prompt_style"),
        )


@dataclass
class InferenceScalingConfig:
    temperature: float = 0.7
    num_parallel_samples_K: int = 16

    @classmethod
    def from_dict(cls, d: dict) -> "InferenceScalingConfig":
        return cls(
            temperature=d.get("temperature", 0.7),
            num_parallel_samples_K=d.get("num_parallel_samples_K", 16),
        )


@dataclass
class EvaluationConfig:
    temperature: float
    inference_scaling: InferenceScalingConfig

    @classmethod
    def from_dict(cls, d: dict) -> "EvaluationConfig":
        return cls(
            temperature=d["temperature"],
            inference_scaling=InferenceScalingConfig.from_dict(d["inference_scaling"]),
        )


@dataclass
class PromptsConfig:
    math_zero_shot: str
    math_self_correction: str
    mbpp_3shot: str
    humaneval_zero_shot: str
    code_self_correction: str

    @classmethod
    def from_dict(cls, d: dict) -> "PromptsConfig":
        return cls(
            math_zero_shot=d["math_zero_shot"],
            math_self_correction=d["math_self_correction"],
            mbpp_3shot=d["mbpp_3shot"],
            humaneval_zero_shot=d["humaneval_zero_shot"],
            code_self_correction=d["code_self_correction"],
        )


@dataclass
class TrainingMetaConfig:
    optimizer: str = "adam"
    gradient_accumulation_steps: int = 1
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 10
    eval_interval: int = 500
    seed: int = 42

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingMetaConfig":
        return cls(
            optimizer=d.get("optimizer", "adam"),
            gradient_accumulation_steps=d.get("gradient_accumulation_steps", 1),
            checkpoint_dir=d.get("checkpoint_dir", "./checkpoints"),
            log_interval=d.get("log_interval", 10),
            eval_interval=d.get("eval_interval", 500),
            seed=d.get("seed", 42),
        )


@dataclass
class ExperimentConfig:
    training: Dict[str, StageConfig]
    reward_shaping: RewardShapingConfig
    kl_penalty: KLPenaltyConfig
    dataset: DatasetConfig
    offline: OfflineConfig = field(default_factory=OfflineConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        # Process training: extract stage configurations and offline settings
        training_raw = d.get("training", {})
        stage_dict = {}
        offline_dict = {}

        for key, value in training_raw.items():
            stage_values = {}
            offline_values = {}
            for k, v in value.items():
                if k in ("offline_mixing_probability", "num_offline_samples_per_prompt"):
                    if k == "offline_mixing_probability":
                        offline_values["mixing_probability"] = v
                    elif k == "num_offline_samples_per_prompt":
                        offline_values["num_samples_per_prompt"] = v
                else:
                    stage_values[k] = v
            stage_dict[key] = StageConfig.from_dict(stage_values)
            # Only stage2 can carry offline values in the current YAML layout
            if key == "stage2" and offline_values:
                offline_dict = offline_values

        offline_config = OfflineConfig.from_dict(offline_dict) if offline_dict else OfflineConfig()

        return cls(
            training=stage_dict,
            reward_shaping=RewardShapingConfig.from_dict(d["reward_shaping"]),
            kl_penalty=KLPenaltyConfig.from_dict(d["kl_penalty"]),
            dataset=DatasetConfig.from_dict(d["dataset"]),
            offline=offline_config,
        )


# --------------------------------------------------------------------------- #
# Top‑level Config
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    model: ModelConfig
    math: ExperimentConfig
    code: ExperimentConfig
    evaluation: EvaluationConfig
    prompts: PromptsConfig
    training: TrainingMetaConfig

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(
            model=ModelConfig.from_dict(d["model"]),
            math=ExperimentConfig.from_dict(d["math"]),
            code=ExperimentConfig.from_dict(d["code"]),
            evaluation=EvaluationConfig.from_dict(d["evaluation"]),
            prompts=PromptsConfig.from_dict(d["prompts"]),
            training=TrainingMetaConfig.from_dict(d["training"]),
        )


# --------------------------------------------------------------------------- #
# Override helpers
# --------------------------------------------------------------------------- #

def _convert_value(value: str, target_type: Any) -> Any:
    """Attempt to convert a string value to the target type."""
    if target_type == str:
        return value
    if target_type == int:
        return int(value)
    if target_type == float:
        return float(value)
    if target_type == bool:
        return value.lower() in ("true", "1", "yes")
    # For any other type (e.g., nested dataclasses) we leave as is
    return value


def apply_overrides(config: Config, overrides: Dict[str, str]) -> None:
    """Apply dotted-key overrides to a Config instance (in‑place)."""
    for key_path, value in overrides.items():
        parts = key_path.split(".")
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)

        target_field_name = parts[-1]
        # Find the target field type to convert the value
        target_type = type(value)  # fallback
        if hasattr(obj, "__dataclass_fields__"):
            for f in fields(obj):
                if f.name == target_field_name:
                    target_type = f.type
                    break

        converted_value = _convert_value(value, target_type)
        setattr(obj, target_field_name, converted_value)


def parse_cmd_overrides(args: List[str]) -> Dict[str, str]:
    """
    Parse command-line arguments of the form --key=value
    and return a dictionary of overrides (keys as dotted paths).
    """
    overrides = {}
    for arg in args:
        if arg.startswith("--"):
            arg = arg[2:]
            if "=" in arg:
                key, val = arg.split("=", 1)
                overrides[key] = val
    return overrides


# --------------------------------------------------------------------------- #
# Main loading function
# --------------------------------------------------------------------------- #

def load_config_from_yaml(path: str, overrides: Optional[Dict[str, str]] = None) -> Config:
    """
    Load configuration from a YAML file and optionally apply command‑line overrides.

    Args:
        path: Path to the YAML configuration file.
        overrides: Optional dictionary of dotted‑key overrides (e.g., {"math.training.stage1.steps": "2000"}).

    Returns:
        Fully populated Config object.
    """
    with open(path, "r") as f:
        raw_dict = yaml.safe_load(f)
    config = Config.from_dict(raw_dict)
    if overrides:
        apply_overrides(config, overrides)
    return config


if __name__ == "__main__":
    # Demo: load config.yaml from command-line argument or default location
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cmd_overrides = parse_cmd_overrides(sys.argv[2:])
    cfg = load_config_from_yaml(cfg_path, overrides=cmd_overrides)
    print(f"Loaded config:\n{dataclasses.asdict(cfg)}")
