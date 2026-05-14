"""Configuration for LoRA-SB experiments."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ModelConfig:
    """Base model configuration."""
    name: str = "roberta-large"
    dtype: str = "bfloat16"
    use_cache: bool = False


@dataclass
class LoRAConfig:
    """LoRA / LoRA-XS configuration."""
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ["query", "value"])
    use_lora_sb: bool = True
    use_optimal_gradient: bool = True


@dataclass
class InitConfig:
    """Initialization configuration for LoRA-SB."""
    num_samples: int = 50
    sample_fraction: float = 0.001
    batch_size: int = 8
    use_sign: bool = True
    random_seed: int = 42


@dataclass
class TrainingConfig:
    """Training configuration."""
    output_dir: str = "./output"
    num_epochs: int = 30
    batch_size: int = 30
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.06
    max_seq_length: int = 512
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    seed: int = 42
    eval_on_start: bool = False


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str = "glue"
    task_name: Optional[str] = "mrpc"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    max_predict_samples: Optional[int] = None


GLUE_CONFIGS = {
    "cola": {
        "task_name": "cola",
        "batch_size": 30,
        "num_epochs": 30,
        "max_seq_length": 512,
        "learning_rate": 1e-3,
        "metric": "matthews_correlation",
    },
    "mrpc": {
        "task_name": "mrpc",
        "batch_size": 128,
        "num_epochs": 30,
        "max_seq_length": 256,
        "learning_rate": 1e-3,
        "metric": "accuracy",
    },
    "rte": {
        "task_name": "rte",
        "batch_size": 128,
        "num_epochs": 30,
        "max_seq_length": 256,
        "learning_rate": 1e-3,
        "metric": "accuracy",
    },
    "sst2": {
        "task_name": "sst2",
        "batch_size": 128,
        "num_epochs": 30,
        "max_seq_length": 256,
        "learning_rate": 1e-3,
        "metric": "accuracy",
    },
    "qnli": {
        "task_name": "qnli",
        "batch_size": 128,
        "num_epochs": 30,
        "max_seq_length": 512,
        "learning_rate": 1e-3,
        "metric": "accuracy",
    },
    "stsb": {
        "task_name": "stsb",
        "batch_size": 30,
        "num_epochs": 30,
        "max_seq_length": 512,
        "learning_rate": 1e-3,
        "metric": "pearson",
    },
}

MATH_CONFIG = {
    "batch_size": 1,
    "gradient_accumulation_steps": 32,
    "num_epochs": 1,
    "max_seq_length": 512,
    "learning_rate": 1e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.02,
    "dropout": 0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
}

COMMONSENSE_CONFIG = {
    "batch_size": 6,
    "gradient_accumulation_steps": 24,
    "num_epochs": 2,
    "max_seq_length": 256,
    "learning_rate": 2e-3,
    "lr_scheduler_type": "linear",
    "warmup_ratio": 0.02,
    "dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
}

LORA_SB_RANKS = [8, 16, 24]
LORA_XS_RANKS = [8, 16, 24]
LORA_RANK = 8

LORA_ALPHA = 16


def get_glue_config(task_name: str) -> dict:
    if task_name not in GLUE_CONFIGS:
        raise ValueError(f"Unknown GLUE task: {task_name}")
    return GLUE_CONFIGS[task_name]
