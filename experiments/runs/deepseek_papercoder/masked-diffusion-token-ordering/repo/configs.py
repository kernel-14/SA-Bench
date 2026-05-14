## configs.py

"""
Configuration dataclass for the masked diffusion reproduction project.

Provides a single, immutable `ExperimentConfig` object that holds all
hyperparameters.  The configuration can be loaded from a YAML file
(e.g., `config.yaml`) with optional task‑specific overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


# ---------------------------------------------------------------------------
# Helper: recursive merge that respects existing keys only
# ---------------------------------------------------------------------------

def _merge_base_with_overrides(
    base: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Recursively merge overrides into base, skipping keys not in base."""
    merged = dict(base)
    for key, value in overrides.items():
        if key not in base:
            continue
        if isinstance(base[key], dict) and isinstance(value, dict):
            merged[key] = _merge_base_with_overrides(base[key], value)
        else:
            merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# Nested configuration dataclasses (frozen for immutability)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Architecture‑related hyperparameters."""
    vocab_size: int = 50257
    max_seq_length: int = 2048
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    positional_embedding_type: str = "learned"
    use_pretrained: bool = False
    pretrained_model_name: Optional[str] = None


@dataclass(frozen=True)
class DiffusionConfig:
    """Diffusion process settings."""
    schedule: str = "cosine"
    mask_token_id: int = 0
    inference_steps: int = 50
    adaptive_sampler: str = "top_margin"   # "vanilla", "top_prob", "top_margin"
    gumbel_noise_coeff: float = 0.5
    sampling_temperature: float = 1.0


@dataclass(frozen=True)
class TrainingConfig:
    """Optimizer, learning rate schedule, and iteration settings."""
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.1
    learning_rate: float = 4.0e-4
    min_learning_rate: float = 4.0e-5
    lr_schedule: str = "cosine"
    warmup_steps: int = 2000
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    # One of num_iterations or epochs should be set; the trainer decides
    num_iterations: Optional[int] = 200000
    epochs: Optional[int] = None
    log_interval: int = 100
    eval_interval: int = 2000
    save_interval: int = 2000
    checkpoint_dir: str = "./checkpoints"


@dataclass(frozen=True)
class NAESATConfig:
    """Synthetic NAE‑SAT distribution parameters."""
    N: int = 20
    P: int = 280
    vocab_size: int = 3


@dataclass(frozen=True)
class SudokuConfig:
    """Sudoku puzzle dataset parameters."""
    vocab_size: int = 10
    train_file: str = "./data/sudoku_train.txt"
    test_file: str = "./data/sudoku_test.txt"
    hard_test_file: str = "./data/sudoku_hard_test.txt"


@dataclass(frozen=True)
class ZebraConfig:
    """Zebra / Einstein puzzle dataset parameters."""
    vocab_size: Optional[int] = None   # inferred from data
    train_file: str = "./data/zebra_train.txt"
    test_file: str = "./data/zebra_test.txt"


@dataclass(frozen=True)
class TextSamplingConfig:
    """Parameters for generative text evaluation."""
    noise_std: float = 0.1
    num_samples: int = 1000


@dataclass(frozen=True)
class DataConfig:
    """Dataset paths and generation parameters."""
    tokenizer_name: str = "gpt2"
    text_data_path: str = "./data/slimpajama"
    perm_type: Optional[str] = None   # "identity", "uniform", "closer", "much_closer"
    nae_sat: NAESATConfig = field(default_factory=NAESATConfig)
    sudoku: SudokuConfig = field(default_factory=SudokuConfig)
    zebra: ZebraConfig = field(default_factory=ZebraConfig)


@dataclass(frozen=True)
class EvaluationConfig:
    """External LLM and sampling evaluation settings."""
    llm_model_name: str = "meta-llama/Llama-2-7b-hf"
    llm_device: str = "cuda"
    text_sampling: TextSamplingConfig = field(default_factory=TextSamplingConfig)


# ---------------------------------------------------------------------------
# Top‑level configuration dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentConfig:
    """Master configuration for an experiment run."""
    task: str = "nae_sat"               # e.g., "nae_sat", "sudoku", "zebra", "llada", "scaling"
    seed: int = 42
    device: str = "cuda"
    dtype: str = "float32"
    use_wandb: bool = False
    wandb_project: str = "mdm_reproduction"
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # ------------------------------------------------------------------
    # Factory method
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls, path: str, override_task: Optional[str] = None
    ) -> "ExperimentConfig":
        """Load configuration from a YAML file, optionally overriding with a
        task‑specific section.

        The YAML file is expected to have a top‑level ``defaults`` mapping and
        optionally one or more task‑specific sections (e.g., ``nae_sat_experiment``).
        The task section is merged recursively into the defaults.

        Args:
            path: Path to the YAML file.
            override_task: Name of the task section to override; if ``None``,
                the ``task`` field inside defaults is used.

        Returns:
            An :class:`ExperimentConfig` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If mandatory fields are missing or invalid.
        """
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        defaults = raw.get("defaults", {})
        if not defaults:
            raise ValueError("YAML file must contain a 'defaults' section.")

        # Build base config from defaults (flatten nested dicts)
        base = cls._from_dict(defaults)

        # Determine which task section to apply
        task = override_task or base.task
        # Map short task names to the actual YAML section names (they may differ)
        task_map = {
            "scaling": "π_learner_scaling",          # the YAML key uses π
            "nae_sat": "nae_sat_experiment",
            "sudoku": "logic_puzzle_sudoku",
            "zebra": "logic_puzzle_zebra",
            "llada": "llada_8b_inference",
        }
        section_name = task_map.get(task, task)

        if section_name in raw:
            overrides = raw[section_name]
            base = cls._apply_overrides(base, overrides)

        # The task name may have been overridden; keep it as the target task
        base.task = task
        return base

    # ------------------------------------------------------------------
    # Internal helpers for construction from nested dicts
    # ------------------------------------------------------------------

    @staticmethod
    def _from_dict(d: Dict[str, Any]) -> "ExperimentConfig":
        """Instantiate an ExperimentConfig from a flat‑keyed dictionary
        (as in the YAML defaults block)."""

        def _nested_get(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
            """Extract sub‑dict with keys starting with `prefix.`"""
            extracted = {}
            strip_len = len(prefix) + 1
            for k, v in d.items():
                if k.startswith(prefix + ".") or k == prefix:
                    if k == prefix:
                        if isinstance(v, dict):
                            extracted.update(v)
                    else:
                        extracted[k[strip_len:]] = v
            # If no sub‑keys, return as is (scalar case)
            return extracted if extracted else d.get(prefix, {})

        def _build_dataclass(cls_type: type, d: Dict[str, Any]) -> Any:
            """Recursively build a dataclass instance from a dict."""
            # Filter fields present in the dataclass
            field_types = {f.name: f.type for f in cls_type.__dataclass_fields__.values()}
            init_kwargs = {}
            for key, value in d.items():
                if key in field_types:
                    if isinstance(field_types[key], type) and hasattr(field_types[key], "__dataclass_fields__"):
                        init_kwargs[key] = _build_dataclass(field_types[key], value if isinstance(value, dict) else {})
                    else:
                        init_kwargs[key] = value
            return cls_type(**init_kwargs)

        model_kwargs = _nested_get(d, "model")
        if isinstance(model_kwargs, dict):
            model = _build_dataclass(ModelConfig, model_kwargs)
        else:
            model = ModelConfig(**model_kwargs) if isinstance(model_kwargs, dict) else ModelConfig()

        diffusion_kwargs = _nested_get(d, "diffusion")
        if isinstance(diffusion_kwargs, dict):
            diffusion = _build_dataclass(DiffusionConfig, diffusion_kwargs)
        else:
            diffusion = DiffusionConfig(**diffusion_kwargs) if isinstance(diffusion_kwargs, dict) else DiffusionConfig()

        training_kwargs = _nested_get(d, "training")
        if isinstance(training_kwargs, dict):
            training = _build_dataclass(TrainingConfig, training_kwargs)
        else:
            training = TrainingConfig(**training_kwargs) if isinstance(training_kwargs, dict) else TrainingConfig()

        # Data: three sub‑configs
        data_kwargs = _nested_get(d, "data")
        if not isinstance(data_kwargs, dict):
            data_kwargs = {}
        nae_sat_raw = data_kwargs.get("nae_sat", {})
        nae_sat = _build_dataclass(NAESATConfig, nae_sat_raw) if isinstance(nae_sat_raw, dict) else NAESATConfig(**nae_sat_raw)
        sudoku_raw = data_kwargs.get("sudoku", {})
        sudoku = _build_dataclass(SudokuConfig, sudoku_raw) if isinstance(sudoku_raw, dict) else SudokuConfig(**sudoku_raw)
        zebra_raw = data_kwargs.get("zebra", {})
        zebra = _build_dataclass(ZebraConfig, zebra_raw) if isinstance(zebra_raw, dict) else ZebraConfig(**zebra_raw)

        # For data sub‑configs, pick per‑config fields
        data_config = DataConfig(
            tokenizer_name=data_kwargs.get("tokenizer_name", "gpt2"),
            text_data_path=data_kwargs.get("text_data_path", "./data/slimpajama"),
            perm_type=data_kwargs.get("perm_type", None),
            nae_sat=nae_sat,
            sudoku=sudoku,
            zebra=zebra,
        )

        eval_kwargs = _nested_get(d, "evaluation")
        if not isinstance(eval_kwargs, dict):
            eval_kwargs = {}
        text_samp_raw = eval_kwargs.get("text_sampling", {})
        text_samp = _build_dataclass(TextSamplingConfig, text_samp_raw) if isinstance(text_samp_raw, dict) else TextSamplingConfig(**text_samp_raw)
        evaluation = EvaluationConfig(
            llm_model_name=eval_kwargs.get("llm_model_name", "meta-llama/Llama-2-7b-hf"),
            llm_device=eval_kwargs.get("llm_device", "cuda"),
            text_sampling=text_samp,
        )

        return cls(
            task=d.get("task", "nae_sat"),
            seed=d.get("seed", 42),
            device=d.get("device", "cuda"),
            dtype=d.get("dtype", "float32"),
            use_wandb=d.get("use_wandb", False),
            wandb_project=d.get("wandb_project", "mdm_reproduction"),
            model=model,
            diffusion=diffusion,
            training=training,
            data=data_config,
            evaluation=evaluation,
        )

    @staticmethod
    def _apply_overrides(
        base: "ExperimentConfig", overrides: Dict[str, Any]
    ) -> "ExperimentConfig":
        """Merge overrides into a base config object by converting both
        to dictionaries, merging, and re‑instantiating."""
        # Convert base to dict
        base_dict = {
            "task": base.task,
            "seed": base.seed,
            "device": base.device,
            "dtype": base.dtype,
            "use_wandb": base.use_wandb,
            "wandb_project": base.wandb_project,
            "model": {
                "vocab_size": base.model.vocab_size,
                "max_seq_length": base.model.max_seq_length,
                "hidden_size": base.model.hidden_size,
                "num_layers": base.model.num_layers,
                "num_attention_heads": base.model.num_attention_heads,
                "intermediate_size": base.model.intermediate_size,
                "hidden_dropout_prob": base.model.hidden_dropout_prob,
                "attention_probs_dropout_prob": base.model.attention_probs_dropout_prob,
                "positional_embedding_type": base.model.positional_embedding_type,
                "use_pretrained": base.model.use_pretrained,
                "pretrained_model_name": base.model.pretrained_model_name,
            },
            "diffusion": {
                "schedule": base.diffusion.schedule,
                "mask_token_id": base.diffusion.mask_token_id,
                "inference_steps": base.diffusion.inference_steps,
                "adaptive_sampler": base.diffusion.adaptive_sampler,
                "gumbel_noise_coeff": base.diffusion.gumbel_noise_coeff,
                "sampling_temperature": base.diffusion.sampling_temperature,
            },
            "training": {
                "optimizer": base.training.optimizer,
                "adam_beta1": base.training.adam_beta1,
                "adam_beta2": base.training.adam_beta2,
                "weight_decay": base.training.weight_decay,
                "learning_rate": base.training.learning_rate,
                "min_learning_rate": base.training.min_learning_rate,
                "lr_schedule": base.training.lr_schedule,
                "warmup_steps": base.training.warmup_steps,
                "batch_size": base.training.batch_size,
                "gradient_accumulation_steps": base.training.gradient_accumulation_steps,
                "num_iterations": base.training.num_iterations,
                "epochs": base.training.epochs,
                "log_interval": base.training.log_interval,
                "eval_interval": base.training.eval_interval,
                "save_interval": base.training.save_interval,
                "checkpoint_dir": base.training.checkpoint_dir,
            },
            "data": {
                "tokenizer_name": base.data.tokenizer_name,
                "text_data_path": base.data.text_data_path,
                "perm_type": base.data.perm_type,
                "nae_sat": {
                    "N": base.data.nae_sat.N,
                    "P": base.data.nae_sat.P,
                    "vocab_size": base.data.nae_sat.vocab_size,
                },
                "sudoku": {
                    "vocab_size": base.data.sudoku.vocab_size,
                    "train_file": base.data.sudoku.train_file,
                    "test_file": base.data.sudoku.test_file,
                    "hard_test_file": base.data.sudoku.hard_test_file,
                },
                "zebra": {
                    "vocab_size": base.data.zebra.vocab_size,
                    "train_file": base.data.zebra.train_file,
                    "test_file": base.data.zebra.test_file,
                },
            },
            "evaluation": {
                "llm_model_name": base.evaluation.llm_model_name,
                "llm_device": base.evaluation.llm_device,
                "text_sampling": {
                    "noise_std": base.evaluation.text_sampling.noise_std,
                    "num_samples": base.evaluation.text_sampling.num_samples,
                },
            },
        }

        # Recursively merge overrides into base_dict
        merged = _merge_base_with_overrides(base_dict, overrides)

        # Re‑build ExperimentConfig from merged dict
        return ExperimentConfig._from_dict(merged)
