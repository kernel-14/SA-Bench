## config.py
"""Configuration dataclasses for MA-RLHF.

This module defines the complete configuration hierarchy for the MA-RLHF
training pipeline. All hyperparameters are sourced from Table 5 and
Appendix B of the paper "MA-RLHF: Reinforcement Learning from Human
Feedback with Macro Actions".

This file has zero internal project dependencies. Only standard library
modules and PyYAML are imported.
"""

import copy
import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Sub-configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DataSplitConfig:
    """Configuration for dataset splitting across training stages.

    The three ratios must sum to 1.0. For APPS (code generation), rm_ratio
    is set to 0.0 and ppo_ratio to 0.8 since there are no preference pairs.
    """

    sft_ratio: float = 0.20
    rm_ratio: float = 0.40
    ppo_ratio: float = 0.40
    seed: int = 42


@dataclass
class SFTConfig:
    """Configuration for Supervised Fine-Tuning (Stage 1).

    Default values correspond to Gemma-2B on TL;DR / HH-RLHF.
    Task-specific overrides are applied via YAML files in configs/.
    """

    batch_size: int = 512
    epochs: int = 3
    learning_rate: float = 5e-5
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.1
    max_prompt_length: int = 512
    max_response_length: int = 512
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8


@dataclass
class RMConfig:
    """Configuration for Reward Modeling (Stage 2).

    Set skip=True for APPS (code generation) since no preference pairs
    are available. The compiler signal replaces the reward model.
    """

    batch_size: int = 64
    epochs: int = 1
    learning_rate: float = 1e-5
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    skip: bool = False


@dataclass
class PPOConfig:
    """Configuration for PPO / MA-PPO training (Stage 3).

    Default values correspond to Gemma-2B. Per-model overrides are in
    configs/*.yaml. Key paper-specific notes:
    - gae_gamma=1.0: explicitly stated in Table 5 for all model sizes.
    - kl_coeff=0.05: default; 0.01 for Gemma-7B on TL;DR (instability).
    - rho=1.0: intra-macro discount factor, always 1.0 per paper.
    - total_steps=4600: TL;DR default matching Figure 2's x-axis.
    """

    batch_size: int = 256
    policy_lr: float = 1.5e-5
    critic_lr: float = 1.5e-5
    epochs: int = 1
    ppo_epochs: int = 1
    rollout: int = 1
    clip_ratio: float = 0.2
    gae_lambda: float = 0.95
    gae_gamma: float = 1.0
    kl_coeff: float = 0.05
    max_prompt_length: int = 512
    max_response_length: int = 512
    warmup_steps: int = 200
    temperature: float = 0.8
    top_p: float = 1.0
    top_k: int = 50
    total_steps: int = 4600
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    rho: float = 1.0


@dataclass
class MacroActionConfig:
    """Configuration for macro action termination and value estimation.

    This is the core MA-RLHF-specific configuration. Key paper alignments:
    - termination="fixed_ngram": default per Section 3.2.1.
    - n_gram=5: default per ablation results (Section 4.3.2).
    - sigma_type="equal": default per Appendix D.1.
    - randomized_lengths=[2,3,5,10]: matches Section 3.2.1 and Appendix E.
    - randomized_repeat=3: matches Appendix E code.
    - parse_cutoff=5: matches Appendix B.4 (C=5).
    - n_gram=1 reduces MA-PPO to vanilla PPO (token-level RLHF baseline).
    - n_gram=None represents n=infinity (REINFORCE/RLOO equivalent).
    """

    termination: str = "fixed_ngram"
    n_gram: Optional[int] = 5
    sigma_type: str = "equal"
    rho: float = 1.0
    randomized_lengths: List[int] = field(
        default_factory=lambda: [2, 3, 5, 10]
    )
    randomized_repeat: int = 3
    parse_cutoff: int = 5


@dataclass
class EvalConfig:
    """Configuration for evaluation metrics and protocols.

    Paper alignments:
    - rm_eval_samples=2000: Section 4.1 "randomly sample 2k validation instances".
    - gpt4_eval_samples=50: Section 4.1 "win-rate on 50 instances".
    - gpt4_model="gpt-4o-2024-05-13": Appendix F.1.
    - best_of_n_sizes and temperatures: Section 4.4 / Figure 8.
    - temperature_sweep: Appendix D.4 / Figure 21.
    """

    rm_eval_samples: int = 2000
    gpt4_eval_samples: int = 50
    gpt4_model: str = "gpt-4o-2024-05-13"
    openai_api_key: str = ""
    best_of_n_sizes: List[int] = field(
        default_factory=lambda: [4, 8, 16, 32]
    )
    best_of_n_temperatures: List[float] = field(
        default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    )
    temperature_sweep: List[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    pass_at_k_values: List[int] = field(default_factory=lambda: [1, 5])
    code_exec_timeout: float = 10.0
    human_eval_samples: int = 50
    num_annotators: int = 1


@dataclass
class AblationConfig:
    """Configuration for ablation studies (Section 4.3).

    ngram_values: None represents n=infinity (REINFORCE/RLOO).
    Corresponds to YAML null values.
    """

    ngram_values: List[Optional[int]] = field(
        default_factory=lambda: [1, 3, 5, 10, None]
    )
    termination_strategies: List[str] = field(
        default_factory=lambda: [
            "fixed_ngram",
            "randomized_ngram",
            "parsing",
            "ppl",
        ]
    )
    sigma_types: List[str] = field(
        default_factory=lambda: ["equal", "unit", "position_decayed"]
    )


@dataclass
class DistributedConfig:
    """Configuration for distributed training via DeepSpeed.

    ZeRO stage 2 for 2B/7B models, stage 3 for 27B (inferred from
    infrastructure requirements). dtype="bf16" is standard for Gemma.
    """

    zero_stage: int = 2
    cpu_offload: bool = False
    gradient_checkpointing: bool = False
    dtype: str = "bf16"


@dataclass
class LoggingConfig:
    """Configuration for experiment tracking and logging."""

    use_wandb: bool = True
    wandb_project: str = "ma-rlhf"
    wandb_entity: str = ""
    use_tensorboard: bool = True
    tensorboard_dir: str = "./runs"
    log_l2_norms: bool = True
    log_score_distribution: bool = True


@dataclass
class ModelConfig:
    """Configuration for the base language model."""

    base_model: str = "google/gemma-2b"
    tokenizer: str = "google/gemma-2b"


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and splitting."""

    name: str = "openai/summarize_from_feedback"
    task: str = "tldr"
    split: DataSplitConfig = field(default_factory=DataSplitConfig)


# ---------------------------------------------------------------------------
# Top-level Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Top-level configuration for the MA-RLHF pipeline.

    This class aggregates all sub-configurations and provides factory
    methods for loading from YAML files and serializing to dicts.

    Usage:
        config = Config.from_yaml("configs/tldr_2b.yaml")
        config_dict = config.to_dict()
    """

    task: str = "tldr"
    stage: str = "all"
    seed: int = 42
    output_dir: str = "./outputs"
    log_interval: int = 10
    eval_interval: int = 200
    save_interval: int = 500
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    rm: RMConfig = field(default_factory=RMConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    macro_action: MacroActionConfig = field(default_factory=MacroActionConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file.

        The YAML file may specify any subset of fields. Missing fields
        fall back to dataclass defaults. Nested sub-configs are
        instantiated recursively.

        The OpenAI API key is resolved with the following priority:
        1. OPENAI_API_KEY environment variable
        2. eval.openai_api_key in the YAML file
        3. Empty string (no key configured)

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A fully populated Config instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ValueError: If any configuration value fails validation.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Configuration file not found: {path}"
            )

        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}

        config = cls._build_from_dict(raw)

        # Resolve OpenAI API key from environment variable (takes priority).
        env_api_key = os.environ.get("OPENAI_API_KEY", "")
        if env_api_key:
            config.eval.openai_api_key = env_api_key

        config._validate()
        return config

    @classmethod
    def _build_from_dict(cls, raw: Dict[str, Any]) -> "Config":
        """Recursively build Config from a raw dictionary.

        Args:
            raw: Dictionary loaded from YAML.

        Returns:
            A Config instance with all sub-configs populated.
        """
        # Extract top-level scalar fields.
        top_level_fields = {
            "task",
            "stage",
            "seed",
            "output_dir",
            "log_interval",
            "eval_interval",
            "save_interval",
        }
        kwargs: Dict[str, Any] = {}
        for f_name in top_level_fields:
            if f_name in raw:
                kwargs[f_name] = raw[f_name]

        # Build sub-configs from their respective YAML sections.
        # The YAML uses "model", "dataset", "sft", "rm", "ppo",
        # "macro_action", "eval", "ablation", "distributed", "logging".
        kwargs["model"] = _build_dataclass(
            ModelConfig, raw.get("model", {})
        )
        kwargs["dataset"] = _build_dataset_config(raw.get("dataset", {}))
        kwargs["sft"] = _build_dataclass(SFTConfig, raw.get("sft", {}))
        kwargs["rm"] = _build_dataclass(RMConfig, raw.get("rm", {}))
        kwargs["ppo"] = _build_dataclass(PPOConfig, raw.get("ppo", {}))
        kwargs["macro_action"] = _build_dataclass(
            MacroActionConfig, raw.get("macro_action", {})
        )
        kwargs["eval"] = _build_dataclass(EvalConfig, raw.get("eval", {}))
        kwargs["ablation"] = _build_dataclass(
            AblationConfig, raw.get("ablation", {})
        )
        kwargs["distributed"] = _build_dataclass(
            DistributedConfig, raw.get("distributed", {})
        )
        kwargs["logging"] = _build_dataclass(
            LoggingConfig, raw.get("logging", {})
        )

        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration to a nested dictionary.

        None values in ngram_values (representing n=infinity) are
        converted to the string "inf" for human-readable output.

        Returns:
            A nested dictionary representation of the configuration.
        """
        result = dataclasses.asdict(self)

        # Convert None in ablation.ngram_values to "inf" for readability.
        if "ablation" in result and "ngram_values" in result["ablation"]:
            result["ablation"]["ngram_values"] = [
                "inf" if v is None else v
                for v in result["ablation"]["ngram_values"]
            ]

        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Validate configuration values for consistency and correctness.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        self._validate_data_split()
        self._validate_macro_action()
        self._validate_ppo()
        self._validate_distributed()

    def _validate_data_split(self) -> None:
        """Validate that data split ratios are consistent.

        Raises:
            ValueError: If ratios do not sum to 1.0 or are out of range.
        """
        split = self.dataset.split
        total = split.sft_ratio + split.rm_ratio + split.ppo_ratio

        # Allow rm_ratio=0.0 when RM stage is skipped (e.g., APPS).
        if not self.rm.skip and split.rm_ratio <= 0.0:
            raise ValueError(
                f"rm_ratio must be > 0 when rm.skip=False, "
                f"got rm_ratio={split.rm_ratio}."
            )

        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Data split ratios must sum to 1.0, "
                f"got sft={split.sft_ratio} + rm={split.rm_ratio} + "
                f"ppo={split.ppo_ratio} = {total:.6f}."
            )

        for name, ratio in [
            ("sft_ratio", split.sft_ratio),
            ("rm_ratio", split.rm_ratio),
            ("ppo_ratio", split.ppo_ratio),
        ]:
            if ratio < 0.0 or ratio > 1.0:
                raise ValueError(
                    f"{name} must be in [0.0, 1.0], got {ratio}."
                )

    def _validate_macro_action(self) -> None:
        """Validate macro action configuration.

        Raises:
            ValueError: If termination or sigma_type is invalid.
        """
        valid_terminations = {
            "fixed_ngram",
            "randomized_ngram",
            "parsing",
            "ppl",
        }
        if self.macro_action.termination not in valid_terminations:
            raise ValueError(
                f"macro_action.termination must be one of "
                f"{valid_terminations}, "
                f"got '{self.macro_action.termination}'."
            )

        valid_sigma_types = {"equal", "unit", "position_decayed"}
        if self.macro_action.sigma_type not in valid_sigma_types:
            raise ValueError(
                f"macro_action.sigma_type must be one of "
                f"{valid_sigma_types}, "
                f"got '{self.macro_action.sigma_type}'."
            )

        # n_gram=None is valid (represents infinity / REINFORCE).
        # n_gram must be a positive integer if not None.
        if self.macro_action.n_gram is not None:
            if (
                not isinstance(self.macro_action.n_gram, int)
                or self.macro_action.n_gram < 1
            ):
                raise ValueError(
                    f"macro_action.n_gram must be a positive integer or "
                    f"None (infinity), got {self.macro_action.n_gram}."
                )

        if self.macro_action.parse_cutoff < 1:
            raise ValueError(
                f"macro_action.parse_cutoff must be >= 1, "
                f"got {self.macro_action.parse_cutoff}."
            )

        if not self.macro_action.randomized_lengths:
            raise ValueError(
                "macro_action.randomized_lengths must be a non-empty list."
            )

        if self.macro_action.randomized_repeat < 1:
            raise ValueError(
                f"macro_action.randomized_repeat must be >= 1, "
                f"got {self.macro_action.randomized_repeat}."
            )

    def _validate_ppo(self) -> None:
        """Validate PPO configuration values.

        Raises:
            ValueError: If any PPO hyperparameter is out of valid range.
        """
        if self.ppo.clip_ratio <= 0.0:
            raise ValueError(
                f"ppo.clip_ratio must be > 0, got {self.ppo.clip_ratio}."
            )

        if not (0.0 < self.ppo.gae_lambda <= 1.0):
            raise ValueError(
                f"ppo.gae_lambda must be in (0, 1], "
                f"got {self.ppo.gae_lambda}."
            )

        if self.ppo.kl_coeff < 0.0:
            raise ValueError(
                f"ppo.kl_coeff must be >= 0, got {self.ppo.kl_coeff}."
            )

        if self.ppo.temperature < 0.0:
            raise ValueError(
                f"ppo.temperature must be >= 0, "
                f"got {self.ppo.temperature}."
            )

        if not (0.0 < self.ppo.top_p <= 1.0):
            raise ValueError(
                f"ppo.top_p must be in (0, 1], got {self.ppo.top_p}."
            )

        if self.ppo.top_k < 0:
            raise ValueError(
                f"ppo.top_k must be >= 0, got {self.ppo.top_k}."
            )

        if self.ppo.total_steps < 1:
            raise ValueError(
                f"ppo.total_steps must be >= 1, "
                f"got {self.ppo.total_steps}."
            )

        if self.ppo.rho < 0.0 or self.ppo.rho > 1.0:
            raise ValueError(
                f"ppo.rho must be in [0, 1], got {self.ppo.rho}."
            )

    def _validate_distributed(self) -> None:
        """Validate distributed training configuration.

        Raises:
            ValueError: If dtype or zero_stage is invalid.
        """
        valid_dtypes = {"fp16", "bf16", "fp32"}
        if self.distributed.dtype not in valid_dtypes:
            raise ValueError(
                f"distributed.dtype must be one of {valid_dtypes}, "
                f"got '{self.distributed.dtype}'."
            )

        valid_zero_stages = {0, 1, 2, 3}
        if self.distributed.zero_stage not in valid_zero_stages:
            raise ValueError(
                f"distributed.zero_stage must be one of "
                f"{valid_zero_stages}, "
                f"got {self.distributed.zero_stage}."
            )


# ---------------------------------------------------------------------------
# Helper functions for building dataclasses from raw dicts
# ---------------------------------------------------------------------------


def _build_dataclass(cls: type, raw: Dict[str, Any]) -> Any:
    """Instantiate a dataclass from a raw dictionary.

    Only keys that correspond to fields of the dataclass are used.
    Extra keys in the raw dict are silently ignored. Missing keys
    fall back to the dataclass field defaults.

    Args:
        cls: The dataclass type to instantiate.
        raw: Raw dictionary (typically from YAML parsing).

    Returns:
        An instance of cls with fields populated from raw.
    """
    if not raw:
        return cls()

    valid_fields = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in raw.items() if k in valid_fields}

    # Deep-copy mutable defaults to avoid shared state between instances.
    instance = cls(**filtered)
    return instance


def _build_dataset_config(raw: Dict[str, Any]) -> DatasetConfig:
    """Build DatasetConfig, handling the nested DataSplitConfig.

    Args:
        raw: Raw dictionary for the dataset section.

    Returns:
        A DatasetConfig instance with split sub-config populated.
    """
    if not raw:
        return DatasetConfig()

    split_raw = raw.get("split", {})
    split_config = _build_dataclass(DataSplitConfig, split_raw)

    # Build the top-level DatasetConfig fields (excluding 'split').
    dataset_fields = {
        f.name for f in dataclasses.fields(DatasetConfig)
        if f.name != "split"
    }
    filtered = {k: v for k, v in raw.items() if k in dataset_fields}
    filtered["split"] = split_config

    return DatasetConfig(**filtered)


# ---------------------------------------------------------------------------
# Convenience factory functions for common configurations
# ---------------------------------------------------------------------------


def make_vanilla_ppo_config(base_config: Config) -> Config:
    """Create a vanilla PPO configuration from a MA-PPO config.

    Sets n_gram=1 which reduces MA-PPO to standard token-level RLHF.
    This is the primary baseline used in the paper.

    Args:
        base_config: The MA-PPO configuration to derive from.

    Returns:
        A new Config with macro_action.n_gram=1 (vanilla PPO).
    """
    new_config = copy.deepcopy(base_config)
    new_config.macro_action.n_gram = 1
    new_config.macro_action.termination = "fixed_ngram"
    return new_config


def make_reinforce_config(base_config: Config) -> Config:
    """Create a REINFORCE/RLOO configuration from a MA-PPO config.

    Sets n_gram=None (infinity) which treats the entire sequence as
    one macro action, equivalent to a contextual bandit problem.

    Args:
        base_config: The MA-PPO configuration to derive from.

    Returns:
        A new Config with macro_action.n_gram=None (REINFORCE).
    """
    new_config = copy.deepcopy(base_config)
    new_config.macro_action.n_gram = None
    new_config.macro_action.termination = "fixed_ngram"
    return new_config


def get_default_config_for_task(task: str) -> Config:
    """Get a default configuration for a specific task.

    This provides sensible defaults without requiring a YAML file,
    useful for testing and quick experiments.

    Args:
        task: One of "tldr", "hh_rlhf", "webgpt", "apps".

    Returns:
        A Config instance with task-appropriate defaults.

    Raises:
        ValueError: If task is not recognized.
    """
    valid_tasks = {"tldr", "hh_rlhf", "webgpt", "apps"}
    if task not in valid_tasks:
        raise ValueError(
            f"task must be one of {valid_tasks}, got '{task}'."
        )

    config = Config()
    config.task = task
    config.dataset.task = task

    if task == "tldr":
        config.dataset.name = "openai/summarize_from_feedback"
        config.model.base_model = "google/gemma-2b"
        config.model.tokenizer = "google/gemma-2b"
        config.sft.batch_size = 512
        config.sft.epochs = 3
        config.sft.learning_rate = 5e-5
        config.rm.batch_size = 64
        config.rm.learning_rate = 1e-5
        config.ppo.total_steps = 4600
        config.ppo.kl_coeff = 0.05

    elif task == "hh_rlhf":
        config.dataset.name = "Anthropic/hh-rlhf"
        config.model.base_model = "google/gemma-2b"
        config.model.tokenizer = "google/gemma-2b"
        config.sft.batch_size = 512
        config.sft.epochs = 3
        config.sft.learning_rate = 5e-5
        config.rm.batch_size = 64
        config.rm.learning_rate = 1e-5
        config.ppo.total_steps = 5600
        config.ppo.kl_coeff = 0.05

    elif task == "webgpt":
        config.dataset.name = "openai/webgpt_comparisons"
        config.model.base_model = "google/gemma-2b"
        config.model.tokenizer = "google/gemma-2b"
        config.sft.batch_size = 64
        config.sft.epochs = 3
        config.sft.learning_rate = 1e-4
        config.rm.batch_size = 32
        config.rm.learning_rate = 2e-5
        config.ppo.epochs = 4
        config.ppo.total_steps = 3200
        config.ppo.kl_coeff = 0.05

    elif task == "apps":
        config.dataset.name = "codeparrot/apps"
        config.model.base_model = "google/codegemma-2b"
        config.model.tokenizer = "google/codegemma-2b"
        config.sft.batch_size = 16
        config.sft.epochs = 1
        config.sft.learning_rate = 5e-6
        config.sft.warmup_ratio = 0.0
        config.rm.skip = True
        config.dataset.split.sft_ratio = 0.20
        config.dataset.split.rm_ratio = 0.00
        config.dataset.split.ppo_ratio = 0.80
        config.ppo.batch_size = 16
        config.ppo.policy_lr = 5e-7
        config.ppo.critic_lr = 5e-5
        config.ppo.temperature = 1.0
        config.ppo.top_k = 5
        config.ppo.warmup_steps = 20
        config.ppo.max_prompt_length = 600
        config.ppo.max_response_length = 512

    return config
