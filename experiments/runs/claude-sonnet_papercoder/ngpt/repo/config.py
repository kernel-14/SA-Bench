```python
## config.py
"""Configuration dataclass for nGPT and GPT experiments.

This module defines the Config dataclass that centralizes all hyperparameters
and experimental settings for reproducing the nGPT paper experiments. All
default values trace back to the paper (Loshchilov et al., nGPT) or the
accompanying config.yaml.

Typical usage:
    # Use a factory method for paper-exact defaults
    config = Config.ngpt_500m(context_length=4096)

    # Load from YAML config file
    config = load_from_yaml("config.yaml", model_key="ngpt_500m")

    # Create ablation variant
    ablation_config = config.replace(normalize_qk=False)
"""

import json
import math
import warnings
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace as dataclass_replace
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


@dataclass
class Config:
    """Centralized configuration for GPT and nGPT experiments.

    All fields have defaults corresponding to the nGPT 0.5B model with 4k
    context length, as described in the paper (Table 2, Table 3, Section 2.6,
    Appendix A.6).

    Attributes:
        model_type: Either "gpt" (baseline) or "ngpt" (normalized transformer).
        n_layers: Number of transformer layers.
        d_model: Model embedding dimension.
        n_heads: Number of attention heads.
        d_mlp: MLP hidden dimension (typically 4 * d_model).
        d_k: Key/query dimension per head (typically d_model // n_heads).
        vocab_size: Vocabulary size (32000 for LLaMA-2, 50257 for GPT-2).
        context_length: Sequence length for training.
        batch_size: Global batch size across all GPUs.
        max_steps: Total number of training iterations.
        optimizer: Optimizer name ("adamw" for GPT, "adam" for nGPT).
        learning_rate: Initial learning rate (the only tuned hyperparameter).
        weight_decay: Weight decay coefficient (0.1 for GPT, 0.0 for nGPT).
        warmup_steps: LR warmup steps (2000 for GPT, 0 for nGPT).
        lr_schedule: Learning rate schedule type ("cosine").
        betas: Adam optimizer momentum coefficients.
        grad_clip: Gradient clipping norm threshold.
        rope_base: Base for Rotary Position Embeddings.
        dtype: Parameter storage dtype ("bfloat16").
        bias: Whether to use bias terms in linear layers.
        alpha_a_init: Initial value for attention eigen learning rates.
        alpha_a_scale: Scale for attention eigen learning rates (controls Adam LR).
        alpha_m_init: Initial value for MLP eigen learning rates.
        alpha_m_scale: Scale for MLP eigen learning rates.
        sqk_init: Initial value for QK scaling factors.
        sqk_scale: Scale for QK scaling factors.
        su_init: Initial value for MLP u-gate scaling factors.
        su_scale: Scale for MLP u-gate scaling factors.
        sv_init: Initial value for MLP v-gate scaling factors.
        sv_scale: Scale for MLP v-gate scaling factors.
        sz_init: Initial value for logit scaling factors.
        sz_scale: Scale for logit scaling factors.
        normalize_qk: Whether to normalize q and k in attention (nGPT).
        use_lerp: Whether to use LERP (True) or SLERP (False) for updates.
        eval_interval: Number of training steps between evaluations.
        eval_steps: Number of validation batches per evaluation.
        downstream_tasks: List of downstream task names for evaluation.
        checkpoint_dir: Directory for saving model checkpoints.
        log_dir: Directory for TensorBoard logs.
        seed: Random seed for reproducibility.
        dataset_name: HuggingFace dataset identifier.
        tokenizer_name: HuggingFace tokenizer identifier.
        tokenizer_fallback: Fallback tokenizer if primary is unavailable.
        train_val_split: Fraction of data used for training.
        cache_dir: Directory for cached tokenized data.
        n_gpus: Number of GPUs for distributed training.
        micro_batch_size: Per-GPU batch size.
        gradient_accumulation_steps: Steps to accumulate before optimizer step.
        use_amp: Whether to use automatic mixed precision.
    """

    # -------------------------------------------------------------------------
    # Model identity
    # -------------------------------------------------------------------------
    model_type: str = "ngpt"
    n_layers: int = 24
    d_model: int = 1024
    n_heads: int = 16
    d_mlp: int = 4096
    d_k: int = 64
    vocab_size: int = 32000

    # -------------------------------------------------------------------------
    # Training context and batch
    # -------------------------------------------------------------------------
    context_length: int = 4096
    batch_size: int = 512
    max_steps: int = 200000

    # -------------------------------------------------------------------------
    # Optimizer settings
    # -------------------------------------------------------------------------
    optimizer: str = "adam"
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    warmup_steps: int = 0
    lr_schedule: str = "cosine"
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    # -------------------------------------------------------------------------
    # Architecture shared settings
    # -------------------------------------------------------------------------
    rope_base: int = 10000
    dtype: str = "bfloat16"
    bias: bool = False

    # -------------------------------------------------------------------------
    # nGPT scaling parameters — Section 2.6, Table 4
    # Stored as actual float values (not strings like "inv_sqrt_d_model").
    # Factory methods compute 1/sqrt(d_model) and store the result.
    # -------------------------------------------------------------------------
    # Eigen learning rates for Attention block (per layer, shape: d_model)
    alpha_a_init: float = 0.05
    alpha_a_scale: float = 1.0 / math.sqrt(1024)  # 1/sqrt(d_model=1024)

    # Eigen learning rates for MLP block (per layer, shape: d_model)
    alpha_m_init: float = 0.05
    alpha_m_scale: float = 1.0 / math.sqrt(1024)  # 1/sqrt(d_model=1024)

    # QK scaling factors (per layer per head, shape: d_k)
    sqk_init: float = 1.0
    sqk_scale: float = 1.0 / math.sqrt(1024)  # 1/sqrt(d_model=1024)

    # MLP u-gate scaling (per layer, shape: d_mlp)
    su_init: float = 1.0
    su_scale: float = 1.0

    # MLP v-gate scaling (per layer, shape: d_mlp)
    sv_init: float = 1.0
    sv_scale: float = 1.0

    # Logit scaling (global, shape: vocab_size)
    sz_init: float = 1.0
    sz_scale: float = 1.0 / math.sqrt(1024)  # 1/sqrt(d_model=1024)

    # -------------------------------------------------------------------------
    # nGPT attention specifics — Section 2.3
    # -------------------------------------------------------------------------
    normalize_qk: bool = True
    use_lerp: bool = True

    # -------------------------------------------------------------------------
    # Evaluation settings
    # -------------------------------------------------------------------------
    eval_interval: int = 500
    eval_steps: int = 100
    downstream_tasks: List[str] = field(
        default_factory=lambda: [
            "hellaswag",
            "piqa",
            "winogrande",
            "arc_easy",
            "wmt14-fr-en",
        ]
    )
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    seed: int = 42

    # -------------------------------------------------------------------------
    # Data settings
    # -------------------------------------------------------------------------
    dataset_name: str = "Skylion007/openwebtext"
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf"
    tokenizer_fallback: str = "gpt2"
    train_val_split: float = 0.9
    cache_dir: str = "data/cache"

    # -------------------------------------------------------------------------
    # Hardware / distributed training settings
    # -------------------------------------------------------------------------
    n_gpus: int = 8
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    use_amp: bool = True

    def __post_init__(self) -> None:
        """Validate configuration consistency after initialization.

        Raises:
            ValueError: If model_type is not "gpt" or "ngpt".
            ValueError: If d_model != n_heads * d_k.
        """
        # Validate model type
        if self.model_type not in ("gpt", "ngpt"):
            raise ValueError(
                f"model_type must be 'gpt' or 'ngpt', got '{self.model_type}'"
            )

        # Validate head dimension consistency
        if self.d_model != self.n_heads * self.d_k:
            raise ValueError(
                f"d_model ({self.d_model}) must equal n_heads ({self.n_heads}) "
                f"* d_k ({self.d_k}) = {self.n_heads * self.d_k}"
            )

        # Warn if MLP dimension deviates from paper's 4x ratio
        if self.d_mlp != 4 * self.d_model:
            warnings.warn(
                f"d_mlp ({self.d_mlp}) != 4 * d_model ({4 * self.d_model}). "
                "The paper uses d_mlp = 4 * d_model. This may be intentional "
                "for ablations.",
                UserWarning,
                stacklevel=2,
            )

        # Warn if context length is not a paper-tested value
        paper_context_lengths = (1024, 4096, 8192)
        if self.context_length not in paper_context_lengths:
            warnings.warn(
                f"context_length ({self.context_length}) is not one of the "
                f"paper-tested values {paper_context_lengths}. Results may "
                "not be directly comparable.",
                UserWarning,
                stacklevel=2,
            )

        # Warn if effective batch size doesn't match global batch size
        effective_batch = (
            self.micro_batch_size
            * self.gradient_accumulation_steps
            * self.n_gpus
        )
        if effective_batch != self.batch_size:
            warnings.warn(
                f"Effective batch size ({effective_batch}) = micro_batch_size "
                f"({self.micro_batch_size}) * gradient_accumulation_steps "
                f"({self.gradient_accumulation_steps}) * n_gpus ({self.n_gpus}) "
                f"!= batch_size ({self.batch_size}). Adjust hardware settings "
                "to match the global batch size.",
                UserWarning,
                stacklevel=2,
            )

        # nGPT-specific validations
        if self.model_type == "ngpt":
            if self.weight_decay != 0.0:
                warnings.warn(
                    f"nGPT should use weight_decay=0.0 (paper Table 3), "
                    f"but got weight_decay={self.weight_decay}.",
                    UserWarning,
                    stacklevel=2,
                )
            if self.warmup_steps != 0:
                warnings.warn(
                    f"nGPT should use warmup_steps=0 (paper Table 3), "
                    f"but got warmup_steps={self.warmup_steps}.",
                    UserWarning,
                    stacklevel=2,
                )

        # GPT-specific validations
        if self.model_type == "gpt":
            if self.optimizer != "adamw":
                warnings.warn(
                    f"GPT baseline should use optimizer='adamw' (paper Table 3), "
                    f"but got optimizer='{self.optimizer}'.",
                    UserWarning,
                    stacklevel=2,
                )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        """Create a Config from a flat dictionary.

        Handles type coercion for fields like betas (list -> tuple) and
        resolves string scale values like "inv_sqrt_d_model" to floats.

        Args:
            d: Flat dictionary mapping field names to values. Unknown keys
               are silently ignored for forward compatibility.

        Returns:
            A Config instance with fields populated from the dictionary.
        """
        import dataclasses

        # Resolve d_model first (needed for scale computations)
        d_model = d.get("d_model", 1024)

        # Resolve string scale values to floats
        resolved = dict(d)
        scale_fields = [
            "alpha_a_scale",
            "alpha_m_scale",
            "sqk_scale",
            "sz_scale",
        ]
        for field_name in scale_fields:
            if field_name in resolved:
                val = resolved[field_name]
                if val == "inv_sqrt_d_model":
                    resolved[field_name] = 1.0 / math.sqrt(d_model)
                elif val == "sqrt_d_model":
                    resolved[field_name] = math.sqrt(d_model)

        # Resolve init values that might be strings
        init_fields = [
            "alpha_a_init",
            "alpha_m_init",
            "sqk_init",
            "su_init",
            "sv_init",
            "sz_init",
        ]
        for field_name in init_fields:
            if field_name in resolved:
                val = resolved[field_name]
                if val == "inv_sqrt_d_model":
                    resolved[field_name] = 1.0 / math.sqrt(d_model)
                elif val == "sqrt_d_model":
                    resolved[field_name] = math.sqrt(d_model)

        # Coerce betas from list to tuple
        if "betas" in resolved and isinstance(resolved["betas"], list):
            resolved["betas"] = tuple(resolved["betas"])

        # Extract only known fields, ignoring unknown keys
        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in resolved.items() if k in known_fields}

        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the Config to a flat dictionary.

        Returns:
            A dictionary representation of the config, suitable for JSON
            serialization. Tuple fields are converted to lists.
        """
        d = asdict(self)
        # Convert tuple to list for JSON compatibility
        if "betas" in d and isinstance(d["betas"], tuple):
            d["betas"] = list(d["betas"])
        return d

    def replace(self, **kwargs: Any) -> "Config":
        """Create a copy of this config with specified fields overridden.

        This is the primary method for creating ablation variants. It delegates
        to dataclasses.replace() and re-runs __post_init__ validation.

        Args:
            **kwargs: Field names and their new values.

        Returns:
            A new Config instance with the specified fields overridden.

        Example:
            ablation_config = config.replace(normalize_qk=False)
        """
        return dataclass_replace(self, **kwargs)

    @classmethod
    def gpt_500m(
        cls,
        context_length: int = 4096,
        learning_rate: Optional[float] = None,
    ) -> "Config":
        """Create a Config for the 0.5B GPT baseline model.

        Architecture and optimization settings match paper Table 2 and Table 3
        exactly. Learning rate defaults are representative values from
        config.yaml Figure 7 sweep.

        Args:
            context_length: Sequence length. Must be 1024, 4096, or 8192.
            learning_rate: Initial learning rate. If None, uses the
                representative value from config.yaml for the given context.

        Returns:
            Config for the 0.5B GPT baseline.
        """
        # Representative learning rates from config.yaml training.gpt section
        lr_map = {
            1024: 3.0e-3,
            4096: 2.0e-3,
            8192: 1.0e-3,
        }
        if learning_rate is None:
            learning_rate = lr_map.get(context_length, 2.0e-3)

        d_model = 1024
        inv_sqrt_d = 1.0 / math.sqrt(d_model)

        return cls(
            model_type="gpt",
            n_layers=24,
            d_model=d_model,
            n_heads=16,
            d_mlp=4096,
            d_k=64,
            vocab_size=32000,
            context_length=context_length,
            batch_size=512,
            max_steps=200000,
            optimizer="adamw",
            learning_rate=learning_rate,
            weight_decay=0.1,
            warmup_steps=2000,
            lr_schedule="cosine",
            betas=(0.9, 0.95),
            grad_clip=1.0,
            rope_base=10000,
            dtype="bfloat16",
            bias=False,
            # nGPT fields — set to defaults even for GPT (ignored by GPT code)
            alpha_a_init=0.05,
            alpha_a_scale=inv_sqrt_d,
            alpha_m_init=0.05,
            alpha_m_scale=inv_sqrt_d,
            sqk_init=1.0,
            sqk_scale=inv_sqrt_d,
            su_init=1.0,
            su_scale=1.0,
            sv_init=1.0,
            sv_scale=1.0,
            sz_init=1.0,
            sz_scale=inv_sqrt_d,
            normalize_qk=True,
            use_lerp=True,
        )

    @classmethod
    def ngpt_500m(
        cls,
        context_length: int = 4096,
        learning_rate: Optional[float] = None,
    ) -> "Config":
        """Create a Config for the 0.5B nGPT model.

        Architecture matches paper Table 2. Scaling parameter initialization
        follows Section 2.6 exactly. Learning rate defaults are representative
        values from config.yaml Figure 7 sweep.

        Args:
            context_length: Sequence length. Must be 1024, 4096, or 8192.
            learning_rate: Initial learning rate. If None, uses the
                representative value from config.yaml for the given context.

        Returns:
            Config for the 0.5B nGPT model.
        """
        # Representative learning rates from config.yaml training.ngpt section
        lr_map = {
            1024: 2.0e-3,
            4096: 1.0e-3,
            8192: 5.0e-4,
        }
        if learning_rate is None:
            learning_rate = lr_map.get(context_length, 1.0e-3)

        d_model = 1024
        inv_sqrt_d = 1.0 / math.sqrt(d_model)  # ≈ 0.03125

        return cls(
            model_type="ngpt",
            n_layers=24,
            d_model=d_model,
            n_heads=16,
            d_mlp=4096,
            d_k=64,
            vocab_size=32000,
            context_length=context_length,
            batch_size=512,
            max_steps=200000,
            optimizer="adam",
            learning_rate=learning_rate,
            weight_decay=0.0,
            warmup_steps=0,
            lr_schedule="cosine",
            betas=(0.9, 0.95),
            grad_clip=1.0,
            rope_base=10000,
            dtype="bfloat16",
            bias=False,
            # nGPT scaling parameters — Section 2.6
            alpha_a_init=0.05,
            alpha_a_scale=inv_sqrt_d,
            alpha_m_init=0.05,
            alpha_m_scale=inv_sqrt_d,
            sqk_init=1.0,
            sqk_scale=inv_sqrt_d,
            su_init=1.0,
            su_scale=1.0,
            sv_init=1.0,
            sv_scale=1.0,
            sz_init=1.0,
            sz_scale=inv_sqrt_d,
            normalize_qk=True,
            use_lerp=True,
        )

    @classmethod
    def gpt_1b(
        cls,
        context_length: int = 4096,
        learning_rate: Optional[float] = None,
    ) -> "Config":
        """Create a Config for the 1B GPT baseline model.

        Architecture and optimization settings match paper Table 2 and Table 3
        exactly.

        Args:
            context_length: Sequence length. Must be 1024, 4096, or 8192.
            learning_rate: Initial learning rate. If None, uses the
                representative value from config.yaml for the given context.

        Returns:
            Config for the 1B GPT baseline.
        """
        # Representative learning rates from config.yaml training.gpt section
        lr_map = {
            1024: 2.0e-3,
            4096: 1.0e-3,
            8192: 5.0e-4,
        }
        if learning_rate is None:
            learning_rate = lr_map.get(context_length, 1.0e-3)

        d_model = 1280
        inv_sqrt_d = 1.0 / math.sqrt(d_model)

        return cls(
            model_type="gpt",
            n_layers=36,
            d_model=d_model,
            n_heads=20,
            d_mlp=5120,
            d_k=64,
            vocab_size=32000,
            context_length=context_length,
            batch_size=512,
            max_steps=200000,
            optimizer="adamw",
            learning_rate=learning_rate,
            weight_decay=0.1,
            warmup_steps=2000,
            lr_schedule="cosine",
            betas=(0.9, 0.95),
            grad_clip=1.0,
            rope_base=10000,
            dtype="bfloat16",
            bias=False,
            # nGPT fields — set to defaults even for GPT (ignored by GPT code)
            alpha_a_init=0.05,
            alpha_a_scale=inv_sqrt_d,
            alpha_m_init=0.05,
            alpha_m_scale=inv_sqrt_d,
            sqk_init=1.0,
            sqk_scale=inv_sqrt_d,
            su_init=1.0,
            su_scale=1.0,
            sv_init=1.0,
            sv_scale=1.0,
            sz_init=1.0,
            sz_scale=inv_sqrt_d,
            normalize_qk=True,
            use_lerp=True,
        )

    @classmethod
    def ngpt_1b(
        cls,
        context_length: int = 4096,
        learning_rate: Optional[float] = None,
    ) -> "Config":
        """Create a Config for the 1B nGPT model.

        Architecture matches paper Table 2. Applies the special-case override
        from Appendix A.7: for 8k context length, alpha_a_init and alpha_m_init
        are set to 0.1 instead of 0.05 to improve training stability.

        Args:
            context_length: Sequence length. Must be 1024, 4096, or 8192.
            learning_rate: Initial learning rate. If None, uses the
                representative value from config.yaml for the given context.

        Returns:
            Config for the 1B nGPT model.
        """
        # Representative learning rates from config.yaml training.ngpt section
        lr_map = {
            1024: 2.0e-3,
            4096: 1.0e-3,
            8192: 5.0e-4,
        }
        if learning_rate is None:
            learning_rate = lr_map.get(context_length, 1.0e-3)

        d_model = 1280
        inv_sqrt_d = 1.0 / math.sqrt(d_model)

        # Special case: 1B model on 8k context uses larger alpha init
        # (Appendix A.7): increases from 0.05 to 0.1 to slow down Adam's
        # effective learning rate on eigen learning rates by ~3x.
        if context_length == 8192:
            alpha_init = 0.1
        else:
            alpha_init = 0.05

        return cls(
            model_type="ngpt",
            n_layers=36,
            d_model=d_model,
            n_heads=20,
            d_mlp=5120,
            d_k=64,
            vocab_size=32000,
            context_length=context_length,
            batch_size=512,
            max_steps=200000,
            optimizer="adam",
            learning_rate=learning_rate,
            weight_decay=0.0,
            warmup_steps=0,
            lr_schedule="cosine",
            betas=(0.9, 0.95),
            grad_clip=1.0,
            rope_base=10000,
            dtype="bfloat16",
            bias=False,
            # nGPT scaling parameters — Section 2.6
            alpha_a_init=alpha_init,
            alpha_a_scale=inv_sqrt_d,
            alpha_m_init=alpha_init,
            alpha_m_scale=inv_sqrt_d,
            sqk_init=1.0,
            sqk_scale=inv_sqrt_d,
            su_init=1.0,
            su_scale=1.0,
            sv_init=1.0,
            sv_scale=1.0,
            sz_init=1.0,
            sz_scale=inv_sqrt_d,
            normalize_qk=True,
            use_lerp=True,
        )

    def __repr__(self) -> str:
        """Return a concise string representation of the config."""
        return (
            f"Config("
            f"model_type={self.model_type!r}, "
            f"n_layers={self.n_layers}, "
            f"d_model={self.d_model}, "
            f"n_heads={self.n_heads}, "
            f"context_length={self.context_length}, "
            f"batch_size={self.batch_size}, "
            f"max_steps={self.max_steps}, "
            f"learning_rate={self.learning_rate}, "
            f"optimizer={self.optimizer!r}"
            f")"
        )


def _resolve_scale_value(
    value: Any,
    d_model: int,
) -> float:
    """Resolve a scale value that may be a string or a float.

    Args:
        value: The value to resolve. Can be a float, int, or one of the
            special strings "inv_sqrt_d_model" or "sqrt_d_model".
        d_model: The model dimension, used for string resolution.

    Returns:
        The resolved float value.

    Raises:
        ValueError: If the string value is not a recognized special string.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if value == "inv_sqrt_d_model":
        return 1.0 / math.sqrt(d_model)
    if value == "sqrt_d_model":
        return math.sqrt(d_model)
    raise ValueError(
        f"Unrecognized scale value: {value!r}. "
        "Expected a float or one of 'inv_sqrt_d_model', 'sqrt_d_model'."
    )


def load_from_yaml(
    yaml_path: str,
    model_key: str,
    context_length: int = 4096,
) -> Config:
    """Load a Config from the project's config.yaml file.

    Handles the nested YAML structure, resolves string scale values, selects
    the appropriate learning rate for the given context length, and returns
    a fully validated Config instance.

    Args:
        yaml_path: Path to the config.yaml file.
        model_key: Key in the "models" section of the YAML, e.g.,
            "ngpt_500m", "gpt_500m", "ngpt_1b", "gpt_1b".
        context_length: Sequence length to use. Determines which
            representative learning rate is selected from the YAML.

    Returns:
        A Config instance populated from the YAML file.

    Raises:
        ImportError: If PyYAML is not installed.
        KeyError: If model_key is not found in the YAML models section.
        ValueError: If the YAML contains unrecognized scale string values.
    """
    if not _YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required to load from YAML. "
            "Install it with: pip install pyyaml"
        )

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # --- Extract model architecture fields ---
    if model_key not in cfg.get("models", {}):
        available = list(cfg.get("models", {}).keys())
        raise KeyError(
            f"Model key '{model_key}' not found in YAML. "
            f"Available keys: {available}"
        )

    model_cfg = cfg["models"][model_key]
    model_type: str = model_cfg["model_type"]
    d_model: int = model_cfg["d_model"]

    flat: Dict[str, Any] = {
        "model_type": model_type,
        "n_layers": model_cfg["n_layers"],
        "d_model": d_model,
        "n_heads": model_cfg["n_heads"],
        "d_mlp": model_cfg["d_mlp"],
        "d_k": model_cfg["d_k"],
        "context_length": context_length,
    }

    # --- Extract shared architecture settings ---
    arch_cfg = cfg.get("architecture", {})
    flat["rope_base"] = arch_cfg.get("rope_base", 10000)
    flat["bias"] = arch_cfg.get("bias", False)
    flat["dtype"] = arch_cfg.get("dtype", "bfloat16")

    # --- Extract training settings ---
    training_cfg = cfg.get("training", {})
    flat["batch_size"] = training_cfg.get("global_batch_size", 512)
    flat["max_steps"] = training_cfg.get("max_steps", 200000)
    flat["eval_interval"] = training_cfg.get("eval_interval", 500)
    flat["eval_steps"] = training_cfg.get("eval_steps", 100)
    flat["grad_clip"] = training_cfg.get("grad_clip", 1.0)

    # Select model-type-specific training settings
    optimizer_section = training_cfg.get(model_type, {