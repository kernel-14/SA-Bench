import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import yaml

# Pre-load the configuration file (cached)
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
_RAW_CONFIG: Optional[Dict[str, Any]] = None

def _load_config() -> Dict[str, Any]:
    global _RAW_CONFIG
    if _RAW_CONFIG is None:
        with open(_CONFIG_PATH, "r") as f:
            _RAW_CONFIG = yaml.safe_load(f)
    return _RAW_CONFIG

@dataclass
class ExperimentConfig:
    """
    Unified configuration container for LoRA‑SB experiments.
    Reads default values from config.yaml and allows overriding any field.
    """
    # Mandatory fields (no defaults)
    task: str
    r: int

    # General defaults (filled from config.yaml)
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"
    init_samples_fraction: float = 0.001
    num_init_samples: Optional[int] = None                        # computed later

    # Model / dataset
    model_name_or_path: str = ""
    target_modules: List[str] = field(default_factory=list)

    # Training hyperparameters
    lr: float = 1e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_seq_length: int = 512
    epochs: int = 1
    warmup_ratio: float = 0.0
    dropout: float = 0.0
    lr_scheduler: str = "linear"
    optimizer: str = "adamw"

    # Additional data / training args (extensible)
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
    training_args: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Load the raw YAML once and parse the relevant section
        raw = _load_config()
        general = raw["general"]

        # Apply general defaults if not already overridden (do later after family detection)
        # Determine the benchmark family from the task string
        family: str
        sub_task: Optional[str] = None
        if self.task.startswith("arithmetic"):
            family = "arithmetic"
        elif self.task.startswith("commonsense"):
            family = "commonsense"
        elif self.task.startswith("glue_"):
            family = "glue"
            sub_task = self.task[5:]   # e.g., "cola"
        else:
            raise ValueError(f"Unknown task: {self.task}")

        # Overwrite general defaults (but respect any user-supplied values)
        # We need to check whether the user explicitly set these; in dataclasses, field with
        # default values are not distinguished from field set with default. So we use a simple
        # approach: if the value equals the class default, we apply from YAML.
        # This is a common pattern to detect user overrides.
        defaults = {
            "seed": 42,
            "device": "cuda",
            "dtype": "bfloat16",
            "init_samples_fraction": 0.001,
        }
        for attr, def_val in defaults.items():
            if getattr(self, attr) == def_val:
                setattr(self, attr, general[attr])

        # Retrieve family-specific config
        family_cfg = raw[family]

        # Model name
        if family == "arithmetic":
            # For arithmetic, model_name_or_path must be provided explicitly
            if not self.model_name_or_path:
                raise ValueError("For arithmetic task, model_name_or_path must be provided.")
            # No default model from YAML because multiple models are used
        elif family == "commonsense":
            if not self.model_name_or_path and self.model_name_or_path == "":
                self.model_name_or_path = family_cfg["model_name"]
        elif family == "glue":
            if not self.model_name_or_path and self.model_name_or_path == "":
                self.model_name_or_path = family_cfg["model_name"]

        # Training defaults (only apply if not already overridden)
        training_defaults = {
            "lr": (1e-4, self.lr),
            "batch_size": (1, self.batch_size),
            "gradient_accumulation_steps": (1, self.gradient_accumulation_steps),
            "max_seq_length": (512, self.max_seq_length),
            "epochs": (1, self.epochs),
            "warmup_ratio": (0.0, self.warmup_ratio),
            "dropout": (0.0, self.dropout),
            "lr_scheduler": ("linear", self.lr_scheduler),
            "optimizer": ("adamw", self.optimizer),
        }

        # For GLUE, batch_size and max_seq_length come from per‑task dicts
        if family == "glue":
            if sub_task is None:
                raise ValueError("GLUE task requires a sub‑task (e.g., glue_cola)")
            # batch_size and max_seq_length are per-task, not the same for all GLUE.
            # We must set them from the YAML, ignoring current self.batch_size if it equals default.
            if self.batch_size == 1:   # default
                self.batch_size = family_cfg["training"]["batch_sizes"][sub_task]
            if self.max_seq_length == 512:  # default
                self.max_seq_length = family_cfg["training"]["max_seq_lengths"][sub_task]
            # For other fields, pull from family_cfg["training"] if they equal defaults
            if self.lr == 1e-4:
                self.lr = family_cfg["training"]["lr"]
            if self.epochs == 1:
                self.epochs = family_cfg["training"]["epochs"]
            if self.warmup_ratio == 0.0:
                self.warmup_ratio = family_cfg["training"]["warmup_ratio"]
            if self.dropout == 0.0:
                self.dropout = family_cfg["training"]["dropout"]
            if self.lr_scheduler == "linear":
                self.lr_scheduler = family_cfg["training"]["lr_scheduler"]
            if self.optimizer == "adamw":
                self.optimizer = family_cfg["training"]["optimizer"]
            # For simplicity, we treat training_defaults as mapping from attr to (default_value, current_value)
            # but we already handled batch_size and max_seq_length separately.
        else:
            # Non-GLUE: load training fields from family_cfg["training"] if they equal defaults
            training_cfg = family_cfg["training"]
            for attr, (def_val, cur_val) in training_defaults.items():
                if cur_val == def_val:   # not user overridden
                    setattr(self, attr, training_cfg[attr])

        # target_modules
        if not self.target_modules:
            self.target_modules = family_cfg["target_modules"]

        # dataset_kwargs can be partially filled; here we set the dataset path if not already present
        if "dataset_path" not in self.dataset_kwargs:
            if family == "arithmetic":
                self.dataset_kwargs["path"] = family_cfg["dataset"]
            elif family == "commonsense":
                self.dataset_kwargs["path"] = family_cfg["dataset"]
            elif family == "glue":
                # For GLUE, we need to set the task-specific path; usually datasets from HF.
                # The YAML only gives dataset name? GLUE tasks are loaded by name via datasets.
                # We'll store the sub_task as dataset_kwargs; the dataset loader will handle.
                self.dataset_kwargs["task"] = sub_task
                self.dataset_kwargs["path"] = "glue"  # standard GLUE

    def set_init_samples(self, total_train_size: int) -> None:
        """Compute the exact number of initialization samples based on dataset size."""
        self.num_init_samples = max(1, int(self.init_samples_fraction * total_train_size))

    def to_dict(self) -> Dict[str, Any]:
        """Serialise all fields to a dictionary suitable for logging (wandb, etc.)."""
        d = asdict(self)
        # Remove None values that might confuse logging
        d = {k: v for k, v in d.items() if v is not None}
        # Add a 'benchmark' field for clarity
        if self.task.startswith("arithmetic"):
            d["benchmark"] = "arithmetic"
        elif self.task.startswith("commonsense"):
            d["benchmark"] = "commonsense"
        elif self.task.startswith("glue_"):
            d["benchmark"] = "glue"
            d["sub_task"] = self.task[5:]
        return d

    # For convenience, provide a string representation
    def __repr__(self) -> str:
        return f"ExperimentConfig(task={self.task}, r={self.r}, lr={self.lr}, bs={self.batch_size})"
