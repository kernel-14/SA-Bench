## config.py
"""Configuration dataclass for LoRA-SB reproduction experiments.

This module defines the Config dataclass that serves as the single source of
truth for all hyperparameters, model settings, dataset configurations, and
experiment metadata. All other modules import and use Config instances rather
than reading YAML or hardcoding values.

Typical usage:
    config = Config.from_yaml("config.yaml", experiment_key="mistral_math")
    config = Config.from_yaml(
        "config.yaml",
        experiment_key="roberta_glue",
        overrides={"method": "lora_sb", "rank": 16}
    )
"""

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Valid literal sets for validation
# ---------------------------------------------------------------------------
_VALID_METHODS = frozenset({
    "lora_sb", "lora_xs", "lora", "pissa", "rslora", "dora", "lora_pro", "full_ft"
})
_VALID_TASKS = frozenset({"math", "commonsense", "glue"})
_VALID_SCHEDULERS = frozenset({"cosine", "linear"})
_VALID_PRECISIONS = frozenset({"bfloat16", "float16", "float32"})
_VALID_EXPERIMENT_KEYS = frozenset({
    "mistral_math", "gemma_math", "llama_commonsense", "roberta_glue"
})


@dataclass
class Config:
    """Flat configuration dataclass for LoRA-SB experiments.

    All fields are sourced from config.yaml and exposed as flat attributes
    for ergonomic access by consumer modules (Trainer, Evaluator, etc.).

    Attributes:
        model_name: HuggingFace model identifier (e.g., 'mistralai/Mistral-7B-v0.1').
        task: Task type, one of 'math', 'commonsense', 'glue'.
        method: Fine-tuning method, one of 'lora_sb', 'lora_xs', 'lora',
            'pissa', 'rslora', 'dora', 'lora_pro', 'full_ft'.
        dataset_name: HuggingFace dataset identifier or local name.
        experiment_key: Key identifying the experiment block in config.yaml.
        output_dir: Directory for saving checkpoints and results.
        rank: LoRA rank r. {32,64,96} for LLMs; {8,16,24} for RoBERTa.
        scaling: Scaling factor s. Always 1.0 for LoRA-SB (Theorem 5).
        target_modules: List of layer name patterns to replace with LoRA modules.
        alpha: Alpha hyperparameter for LoRA-XS and LoRA baselines.
            For LoRA-SB this is unused (scaling=1.0 is sufficient).
        learning_rate: AdamW learning rate.
        batch_size: Per-device training batch size.
        grad_accum_steps: Gradient accumulation steps before optimizer update.
        max_seq_len: Maximum tokenized sequence length.
        epochs: Number of training epochs.
        dropout: LoRA dropout probability.
        lr_scheduler: Learning rate scheduler type, 'cosine' or 'linear'.
        warmup_ratio: Fraction of total steps used for linear warmup.
        weight_decay: AdamW weight decay coefficient.
        num_init_samples: Number of samples for LoRA-SB gradient estimation.
            Corresponds to 0.1% of training set (e.g., 50 for 50K MetaMathQA).
        init_fraction: Fraction of training set for initialization (0.001 = 0.1%).
        min_init_samples: Minimum floor for num_init_samples on small datasets.
        svd_niter: Number of power iterations for torch.svd_lowrank.
        eval_datasets: List of evaluation dataset names.
        max_new_tokens: Maximum tokens to generate for math evaluation.
        do_sample: Whether to use sampling during generation (False = greedy).
        seeds: List of random seeds for reproducibility (paper uses [42, 43, 44]).
        precision: Model precision, one of 'bfloat16', 'float16', 'float32'.
        gradient_checkpointing: Whether to enable gradient checkpointing.
        log_every_n_steps: Logging frequency during training.
        use_wandb: Whether to log to Weights & Biases.
        use_tensorboard: Whether to log to TensorBoard.
        log_trainable_params: Whether to log trainable parameter count at start.
        num_train_samples: Number of training samples to use (dataset-specific).
        lora_baseline_rank: Rank used for LoRA baselines (e.g., 8 for RoBERTa).
        sequence_lengths: Per-task sequence length overrides (GLUE only).
    """

    # -----------------------------------------------------------------------
    # Identity fields
    # -----------------------------------------------------------------------
    model_name: str = "mistralai/Mistral-7B-v0.1"
    task: str = "math"
    method: str = "lora_sb"
    dataset_name: str = "meta-math/MetaMathQA"
    experiment_key: str = "mistral_math"
    output_dir: str = "outputs/"

    # -----------------------------------------------------------------------
    # LoRA architecture fields
    # -----------------------------------------------------------------------
    rank: int = 32
    scaling: float = 1.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    alpha: float = 32.0

    # -----------------------------------------------------------------------
    # Training hyperparameter fields
    # -----------------------------------------------------------------------
    learning_rate: float = 1e-4
    batch_size: int = 1
    grad_accum_steps: int = 32
    max_seq_len: int = 512
    epochs: int = 1
    dropout: float = 0.0
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.02
    weight_decay: float = 0.01

    # -----------------------------------------------------------------------
    # Initialization fields (LoRA-SB specific)
    # -----------------------------------------------------------------------
    num_init_samples: int = 50
    init_fraction: float = 0.001
    min_init_samples: int = 10
    svd_niter: int = 4

    # -----------------------------------------------------------------------
    # Evaluation fields
    # -----------------------------------------------------------------------
    eval_datasets: List[str] = field(default_factory=lambda: ["gsm8k", "math"])
    max_new_tokens: int = 512
    do_sample: bool = False

    # -----------------------------------------------------------------------
    # Reproducibility and hardware fields
    # -----------------------------------------------------------------------
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44])
    precision: str = "bfloat16"
    gradient_checkpointing: bool = True

    # -----------------------------------------------------------------------
    # Logging fields
    # -----------------------------------------------------------------------
    log_every_n_steps: int = 100
    use_wandb: bool = False
    use_tensorboard: bool = False
    log_trainable_params: bool = True

    # -----------------------------------------------------------------------
    # Dataset-specific fields
    # -----------------------------------------------------------------------
    num_train_samples: int = 50000
    lora_baseline_rank: int = 32
    sequence_lengths: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate all fields after construction."""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration field values.

        Raises:
            ValueError: If any field has an invalid value.
        """
        if self.method not in _VALID_METHODS:
            raise ValueError(
                f"Invalid method '{self.method}'. "
                f"Must be one of: {sorted(_VALID_METHODS)}"
            )
        if self.task not in _VALID_TASKS:
            raise ValueError(
                f"Invalid task '{self.task}'. "
                f"Must be one of: {sorted(_VALID_TASKS)}"
            )
        if self.lr_scheduler not in _VALID_SCHEDULERS:
            raise ValueError(
                f"Invalid lr_scheduler '{self.lr_scheduler}'. "
                f"Must be one of: {sorted(_VALID_SCHEDULERS)}"
            )
        if self.precision not in _VALID_PRECISIONS:
            raise ValueError(
                f"Invalid precision '{self.precision}'. "
                f"Must be one of: {sorted(_VALID_PRECISIONS)}"
            )
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
        if self.scaling <= 0:
            raise ValueError(f"scaling must be positive, got {self.scaling}")
        if self.method == "lora_sb" and self.num_init_samples <= 0:
            raise ValueError(
                f"num_init_samples must be positive for lora_sb, "
                f"got {self.num_init_samples}"
            )
        if len(self.seeds) == 0:
            raise ValueError("seeds list must not be empty")
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if not (0.0 <= self.warmup_ratio <= 1.0):
            raise ValueError(
                f"warmup_ratio must be in [0, 1], got {self.warmup_ratio}"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(
                f"dropout must be in [0, 1), got {self.dropout}"
            )

    @classmethod
    def from_yaml(
        cls,
        path: str,
        experiment_key: str = "mistral_math",
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "Config":
        """Construct a Config from a YAML file.

        Loads the YAML, merges defaults with the experiment-specific block,
        flattens nested keys to top-level fields, applies CLI overrides, and
        derives computed fields (alpha, num_init_samples, output_dir).

        Args:
            path: Path to the config.yaml file.
            experiment_key: Key of the experiment block to load, e.g.
                'mistral_math', 'gemma_math', 'llama_commonsense', 'roberta_glue'.
            overrides: Optional dict of field overrides from CLI arguments.
                Keys must match Config field names. Applied last, after all
                YAML-based merging.

        Returns:
            A fully validated Config instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            KeyError: If experiment_key is not found in the YAML.
            ValueError: If any field has an invalid value after construction.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f)

        if experiment_key not in raw:
            raise KeyError(
                f"Experiment key '{experiment_key}' not found in {path}. "
                f"Available keys: {list(raw.keys())}"
            )

        flat = _flatten_config(raw, experiment_key)
        flat["experiment_key"] = experiment_key

        # Apply CLI overrides last (highest priority)
        if overrides:
            for key, value in overrides.items():
                if value is not None:
                    flat[key] = value

        # Derive computed fields
        flat = _derive_computed_fields(flat)

        # Build output_dir with experiment-specific subdirectory
        base_output = flat.get("output_dir", "outputs/")
        method = flat.get("method", "lora_sb")
        rank = flat.get("rank", 32)
        flat["output_dir"] = os.path.join(
            base_output, experiment_key, method, f"rank{rank}"
        )

        # Filter to only known Config fields to avoid unexpected keyword args
        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in flat.items() if k in known_fields}

        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the Config to a plain dictionary.

        Returns:
            A dict representation of all Config fields, suitable for JSON
            serialization and saving alongside experiment results.
        """
        return dataclasses.asdict(self)

    def __repr__(self) -> str:
        """Return a human-readable string representation."""
        fields_str = "\n  ".join(
            f"{f.name}={getattr(self, f.name)!r}"
            for f in dataclasses.fields(self)
        )
        return f"Config(\n  {fields_str}\n)"


# ---------------------------------------------------------------------------
# Private helper functions
# ---------------------------------------------------------------------------

def _flatten_config(raw: Dict[str, Any], experiment_key: str) -> Dict[str, Any]:
    """Flatten the hierarchical YAML structure into a single-level dict.

    Merges in order: defaults -> experiment block (top-level keys) ->
    experiment.training -> experiment.initialization -> lora_sb settings ->
    evaluation settings -> hardware/reproducibility settings.

    Args:
        raw: The full parsed YAML as a nested dict.
        experiment_key: The experiment block key to extract.

    Returns:
        A flat dict mapping Config field names to their values.
    """
    flat: Dict[str, Any] = {}

    # 1. Load global defaults
    defaults = raw.get("defaults", {})
    flat["method"] = defaults.get("method", "lora_sb")
    flat["rank"] = defaults.get("rank", 32)
    flat["scaling"] = defaults.get("scaling", 1.0)
    flat["seeds"] = defaults.get("seeds", [42, 43, 44])
    flat["precision"] = defaults.get("precision", "bfloat16")
    flat["output_dir"] = defaults.get("output_dir", "outputs/")

    # 2. Load experiment-specific top-level fields
    exp = raw.get(experiment_key, {})
    if "model_name" in exp:
        flat["model_name"] = exp["model_name"]
    if "task" in exp:
        flat["task"] = exp["task"]
    if "dataset_name" in exp:
        flat["dataset_name"] = exp["dataset_name"]
    if "target_modules" in exp:
        flat["target_modules"] = exp["target_modules"]
    if "eval_datasets" in exp:
        flat["eval_datasets"] = exp["eval_datasets"]
    if "num_train_samples" in exp:
        flat["num_train_samples"] = exp["num_train_samples"]
    if "lora_baseline_rank" in exp:
        flat["lora_baseline_rank"] = exp["lora_baseline_rank"]
    if "ranks" in exp:
        # Store the list of ranks; the active rank comes from defaults or override
        flat["_available_ranks"] = exp["ranks"]

    # 3. Flatten training sub-block
    training = exp.get("training", {})
    if training:
        flat["learning_rate"] = training.get("learning_rate", flat.get("learning_rate", 1e-4))
        flat["batch_size"] = training.get("batch_size", flat.get("batch_size", 1))
        flat["grad_accum_steps"] = training.get("grad_accum_steps", flat.get("grad_accum_steps", 32))
        flat["max_seq_len"] = training.get("max_seq_len", flat.get("max_seq_len", 512))
        flat["epochs"] = training.get("epochs", flat.get("epochs", 1))
        flat["dropout"] = training.get("dropout", flat.get("dropout", 0.0))
        flat["learning_rate"] = training.get("learning_rate", flat.get("learning_rate", 1e-4))
        flat["lr_scheduler"] = training.get("lr_scheduler", flat.get("lr_scheduler", "cosine"))
        flat["warmup_ratio"] = training.get("warmup_ratio", flat.get("warmup_ratio", 0.02))
        flat["weight_decay"] = training.get("weight_decay", flat.get("weight_decay", 0.01))

    # 4. Flatten initialization sub-block
    init_block = exp.get("initialization", {})
    if init_block:
        if "num_init_samples" in init_block:
            flat["num_init_samples"] = init_block["num_init_samples"]
        if "init_fraction" in init_block:
            flat["init_fraction"] = init_block["init_fraction"]
        if "min_init_samples" in init_block:
            flat["min_init_samples"] = init_block["min_init_samples"]

    # 5. Flatten LoRA-SB global settings
    lora_sb_block = raw.get("lora_sb", {})
    if lora_sb_block:
        flat["scaling"] = lora_sb_block.get("scaling", flat.get("scaling", 1.0))
        flat["svd_niter"] = lora_sb_block.get("svd_niter", 4)

    # 6. Flatten evaluation settings
    eval_block = raw.get("evaluation", {})
    if eval_block:
        math_eval = eval_block.get("math", {})
        if math_eval:
            flat["max_new_tokens"] = math_eval.get("max_new_tokens", 512)
            flat["do_sample"] = math_eval.get("do_sample", False)

    # 7. Flatten hardware settings
    hardware_block = raw.get("hardware", {})
    if hardware_block:
        flat["gradient_checkpointing"] = hardware_block.get(
            "gradient_checkpointing", True
        )
        flat["precision"] = hardware_block.get("precision", flat.get("precision", "bfloat16"))

    # 8. Flatten reproducibility settings
    repro_block = raw.get("reproducibility", {})
    if repro_block:
        flat["seeds"] = repro_block.get("seeds", flat.get("seeds", [42, 43, 44]))

    # 9. Flatten logging settings
    logging_block = raw.get("logging", {})
    if logging_block:
        flat["log_every_n_steps"] = logging_block.get("log_every_n_steps", 100)
        flat["use_wandb"] = logging_block.get("use_wandb", False)
        flat["use_tensorboard"] = logging_block.get("use_tensorboard", False)
        flat["log_trainable_params"] = logging_block.get("log_trainable_params", True)

    # 10. Flatten per-task sequence lengths (GLUE only)
    if "sequence_lengths" in exp:
        flat["sequence_lengths"] = exp["sequence_lengths"]

    # 11. Extract alpha from experiment-specific lora_xs block
    exp_lora_xs = exp.get("lora_xs", {})
    global_lora_xs = raw.get("baselines", {}).get("lora_xs", {})

    # Determine alpha: experiment-level overrides global
    if "alpha" in exp_lora_xs:
        flat["_lora_xs_alpha"] = exp_lora_xs["alpha"]
        flat["_lora_xs_alpha_equals_rank"] = exp_lora_xs.get("alpha_equals_rank", False)
    elif "alpha_equals_rank" in exp_lora_xs:
        flat["_lora_xs_alpha_equals_rank"] = exp_lora_xs["alpha_equals_rank"]
    elif "alpha_equals_rank" in global_lora_xs:
        flat["_lora_xs_alpha_equals_rank"] = global_lora_xs["alpha_equals_rank"]
    else:
        flat["_lora_xs_alpha_equals_rank"] = True

    return flat


def _derive_computed_fields(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Derive computed fields from the flattened config dict.

    Handles:
    - alpha: derived from rank and lora_xs settings
    - num_init_samples: derived from init_fraction and num_train_samples
    - scaling: enforced to 1.0 for lora_sb

    Args:
        flat: The flattened config dict from _flatten_config.

    Returns:
        The updated flat dict with computed fields set.
    """
    method = flat.get("method", "lora_sb")
    rank = flat.get("rank", 32)

    # -----------------------------------------------------------------------
    # Derive alpha
    # -----------------------------------------------------------------------
    if method == "lora_sb":
        # LoRA-SB: scaling=1.0 is enforced; alpha is unused but set for consistency
        flat["scaling"] = flat.get("scaling", 1.0)
        flat["alpha"] = float(rank)  # unused, set for consistency
    elif method == "lora_xs":
        # LoRA-XS: alpha from explicit setting or alpha=rank
        if "_lora_xs_alpha" in flat:
            flat["alpha"] = float(flat["_lora_xs_alpha"])
        elif flat.get("_lora_xs_alpha_equals_rank", True):
            flat["alpha"] = float(rank)
        else:
            flat["alpha"] = float(rank)  # fallback
    else:
        # All other LoRA variants: alpha = rank (standard setting from paper)
        flat["alpha"] = float(rank)

    # -----------------------------------------------------------------------
    # Derive num_init_samples (only relevant for lora_sb)
    # -----------------------------------------------------------------------
    if "num_init_samples" not in flat:
        init_fraction = flat.get("init_fraction", 0.001)
        num_train = flat.get("num_train_samples", 50000)
        min_samples = flat.get("min_init_samples", 10)
        flat["num_init_samples"] = max(min_samples, int(init_fraction * num_train))

    # -----------------------------------------------------------------------
    # Ensure min_init_samples has a default
    # -----------------------------------------------------------------------
    if "min_init_samples" not in flat:
        flat["min_init_samples"] = 10

    # -----------------------------------------------------------------------
    # Ensure init_fraction has a default
    # -----------------------------------------------------------------------
    if "init_fraction" not in flat:
        flat["init_fraction"] = 0.001

    # -----------------------------------------------------------------------
    # Ensure svd_niter has a default
    # -----------------------------------------------------------------------
    if "svd_niter" not in flat:
        flat["svd_niter"] = 4

    # -----------------------------------------------------------------------
    # Ensure evaluation fields have defaults
    # -----------------------------------------------------------------------
    if "max_new_tokens" not in flat:
        flat["max_new_tokens"] = 512
    if "do_sample" not in flat:
        flat["do_sample"] = False

    # -----------------------------------------------------------------------
    # Ensure logging fields have defaults
    # -----------------------------------------------------------------------
    if "log_every_n_steps" not in flat:
        flat["log_every_n_steps"] = 100
    if "use_wandb" not in flat:
        flat["use_wandb"] = False
    if "use_tensorboard" not in flat:
        flat["use_tensorboard"] = False
    if "log_trainable_params" not in flat:
        flat["log_trainable_params"] = True

    # -----------------------------------------------------------------------
    # Ensure hardware fields have defaults
    # -----------------------------------------------------------------------
    if "gradient_checkpointing" not in flat:
        flat["gradient_checkpointing"] = True

    # -----------------------------------------------------------------------
    # Ensure dataset fields have defaults
    # -----------------------------------------------------------------------
    if "num_train_samples" not in flat:
        flat["num_train_samples"] = 50000
    if "lora_baseline_rank" not in flat:
        flat["lora_baseline_rank"] = rank
    if "sequence_lengths" not in flat:
        flat["sequence_lengths"] = {}

    # -----------------------------------------------------------------------
    # Ensure target_modules has a default
    # -----------------------------------------------------------------------
    if "target_modules" not in flat:
        flat["target_modules"] = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

    # -----------------------------------------------------------------------
    # Ensure eval_datasets has a default
    # -----------------------------------------------------------------------
    if "eval_datasets" not in flat:
        task = flat.get("task", "math")
        if task == "math":
            flat["eval_datasets"] = ["gsm8k", "math"]
        elif task == "commonsense":
            flat["eval_datasets"] = [
                "boolq", "piqa", "social_i_qa", "hellaswag",
                "winogrande", "arc_easy", "arc_challenge", "openbookqa"
            ]
        elif task == "glue":
            flat["eval_datasets"] = ["cola", "rte", "mrpc", "stsb", "qnli", "sst2"]
        else:
            flat["eval_datasets"] = []

    # -----------------------------------------------------------------------
    # Ensure model_name has a default
    # -----------------------------------------------------------------------
    if "model_name" not in flat:
        flat["model_name"] = "mistralai/Mistral-7B-v0.1"

    # -----------------------------------------------------------------------
    # Ensure dataset_name has a default
    # -----------------------------------------------------------------------
    if "dataset_name" not in flat:
        flat["dataset_name"] = "meta-math/MetaMathQA"

    # -----------------------------------------------------------------------
    # Remove internal/private keys (prefixed with _) before constructing Config
    # -----------------------------------------------------------------------
    keys_to_remove = [k for k in flat if k.startswith("_")]
    for k in keys_to_remove:
        del flat[k]

    return flat
