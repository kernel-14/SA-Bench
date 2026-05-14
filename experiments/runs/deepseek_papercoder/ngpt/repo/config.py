"""
config.py

Defines the Config dataclass and all nested configuration groups for reproducing
the NGPT paper experiments.  Supports loading from a YAML file that may contain
``!expr`` tags, which are evaluated in a safe namespace containing the ``math``
module and the configuration itself.
"""

import math
import yaml
from dataclasses import dataclass, field, asdict
from typing import Tuple


class _Expr(str):
    """Marker class for YAML ``!expr`` tags – the string will be evaluated later."""
    pass


# ---------------------------------------------------------------------------
# Custom YAML constructor for !expr
# ---------------------------------------------------------------------------
def _expr_constructor(loader: yaml.Loader, node: yaml.Node) -> _Expr:
    value = loader.construct_scalar(node)
    return _Expr(value)


# We register the constructor on the SafeLoader.
yaml.SafeLoader.add_constructor("!expr", _expr_constructor)


# ---------------------------------------------------------------------------
# Nested configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Architecture and normalisation parameters."""
    use_ngpt: bool = False
    vocab_size: int = 32000
    n_layers: int = 24
    d_model: int = 1024
    n_heads: int = 16
    d_k: int = field(init=False)          # derived: d_model // n_heads
    d_mlp: int = field(init=False)        # derived: 4 * d_model
    max_seq_len: int = 1024
    rope_base: float = 10000.0

    # Initialisation standard deviations
    init_scale_norm: float = 0.02
    init_scale_ngpt: float = field(default_factory=lambda: 1.0 / math.sqrt(1024))

    tie_embeddings: bool = False

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        self.d_mlp = 4 * self.d_model
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible "
                f"by n_heads ({self.n_heads})"
            )


@dataclass
class TrainingConfig:
    """Training loop and data loading configuration."""
    batch_size: int = 512
    num_iters: int = 100000
    eval_interval: int = 1000
    num_gpus: int = 64
    use_amp: bool = True
    dtype: str = "bfloat16"


@dataclass
class OptimConfig:
    """Optimizer and learning rate schedule settings."""
    lr: float = 2.0e-3
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1.0e-8
    weight_decay: float = 0.0          # 0.1 for GPT, 0.0 for nGPT
    warmup_steps: int = 0              # 2000 for GPT, 0 for nGPT
    lr_schedule: str = "cosine"
    lr_final: float = 0.0


@dataclass
class nGPTConfig:
    """Hyperparameters specific to the normalised Transformer."""
    # Eigen learning rates (per‑layer per‑dimension vectors)
    eigen_alpha_A_init: float = 0.05
    eigen_alpha_A_scale: float = field(default_factory=lambda: 1.0 / math.sqrt(1024))
    eigen_alpha_M_init: float = 0.05
    eigen_alpha_M_scale: float = field(default_factory=lambda: 1.0 / math.sqrt(1024))

    # QK scaling factors (per attention head)
    s_qk_init: float = 1.0
    s_qk_scale: float = field(default_factory=lambda: 1.0 / math.sqrt(1024))

    # MLP intermediate scaling factors
    s_u_init: float = 1.0
    s_u_scale: float = 1.0
    s_v_init: float = 1.0
    s_v_scale: float = 1.0

    # Logit scaling factor (vocabulary‑wise)
    s_z_init: float = 1.0
    s_z_scale: float = field(default_factory=lambda: 1.0 / math.sqrt(1024))


@dataclass
class DataConfig:
    """Dataset and tokenizer configuration."""
    dataset_path: str = "./data/openwebtext"
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf"
    val_ratio: float = 0.01


@dataclass
class LoggingConfig:
    """Logging and checkpointing settings."""
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    use_wandb: bool = False
    wandb_project: str = "ngpt-reproduction"


# ---------------------------------------------------------------------------
# Top‑level configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Complete configuration container for the NGPT reproduction."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    ngpt: nGPTConfig = field(default_factory=nGPTConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self):
        """Validate constraints that depend on multiple sub‑configs."""
        if self.model.use_ngpt:
            if self.optim.weight_decay != 0.0:
                raise ValueError(
                    "For nGPT, weight_decay must be 0.0 "
                    f"(got {self.optim.weight_decay})"
                )
            if self.optim.warmup_steps != 0:
                raise ValueError(
                    "For nGPT, warmup_steps must be 0 "
                    f"(got {self.optim.warmup_steps})"
                )

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """
        Load configuration from a YAML file.

        The YAML file may contain ``!expr`` tags (e.g., ``!expr "1/math.sqrt(model.d_model)"``).
        These expressions are evaluated in a restricted environment containing the
        ``math`` module and the configuration itself (so that attributes like
        ``model.d_model`` are accessible).

        Args:
            path: Path to the YAML file.

        Returns:
            A fully resolved ``Config`` instance.
        """
        with open(path, "r") as f:
            raw = yaml.load(f, Loader=yaml.SafeLoader)

        # Helper that replaces every _Expr with the number 0.
        # This allows us to build a valid (though incorrect) Config for evaluation.
        def _replace_expr(obj):
            if isinstance(obj, _Expr):
                return 0
            if isinstance(obj, dict):
                return {k: _replace_expr(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_replace_expr(item) for item in obj]
            return obj

        # Build a "dummy" config from the raw dict with expressions replaced by 0.
        # This gives us a proper namespace for evaluating the expressions.
        dummy_raw = _replace_expr(raw)
        # Convert list to tuple for betas if necessary (YAML gives a list).
        optim_dummy = dummy_raw.get("optim", {})
        if "betas" in optim_dummy and isinstance(optim_dummy["betas"], list):
            optim_dummy["betas"] = tuple(optim_dummy["betas"])

        dummy_config = Config(
            model=ModelConfig(**dummy_raw.get("model", {})),
            training=TrainingConfig(**dummy_raw.get("training", {})),
            optim=OptimConfig(**optim_dummy),
            ngpt=nGPTConfig(**dummy_raw.get("ngpt", {})),
            data=DataConfig(**dummy_raw.get("data", {})),
            logging=LoggingConfig(**dummy_raw.get("logging", {})),
        )

        # The local namespace for evaluation: the dummy config itself.
        eval_locals = vars(dummy_config)

        # Recursively traverse the original raw dict and evaluate all _Expr.
        def _evaluate_expr(obj):
            if isinstance(obj, _Expr):
                try:
                    # Safe eval: only `math` is available as a global; no builtins.
                    return eval(obj, {"math": math, "__builtins__": {}}, eval_locals)
                except Exception as exc:
                    raise ValueError(f"Failed to evaluate expression '{obj}': {exc}") from exc
            if isinstance(obj, dict):
                return {k: _evaluate_expr(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_evaluate_expr(item) for item in obj]
            return obj

        clean_raw = _evaluate_expr(raw)
        # Ensure betas is a tuple again.
        if "betas" in clean_raw.get("optim", {}) and isinstance(clean_raw["optim"]["betas"], list):
            clean_raw["optim"]["betas"] = tuple(clean_raw["optim"]["betas"])

        # Build the final, fully resolved config.
        config = cls(
            model=ModelConfig(**clean_raw["model"]),
            training=TrainingConfig(**clean_raw["training"]),
            optim=OptimConfig(**clean_raw["optim"]),
            ngpt=nGPTConfig(**clean_raw["ngpt"]),
            data=DataConfig(**clean_raw["data"]),
            logging=LoggingConfig(**clean_raw["logging"]),
        )
        return config

    def to_dict(self) -> dict:
        """Convert the configuration (recursively) to a plain dictionary."""
        return asdict(self)
