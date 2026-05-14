```python
## main.py
"""Entry point for nGPT and GPT experiment reproduction.

This module wires together all other modules — config loading, data
preparation, model construction, training, evaluation, ablation, and
analysis — through a clean CLI interface.

Supported modes:
    train     — Train a GPT or nGPT model on OpenWebText.
    eval      — Evaluate a trained model (val loss + downstream tasks).
    analysis  — Analyze model internals (embeddings, condition numbers,
                learned parameter distributions).
    ablation  — Run ablation studies from Appendix A.9 (Tables 4, 5, 6).
    all       — Run all of the above in sequence.

Typical usage:
    # Train nGPT 0.5B with 4k context (paper default)
    python main.py --mode train --model_type ngpt --model_size 500m \\
                   --context_length 4096

    # Train GPT baseline for comparison
    python main.py --mode train --model_type gpt --model_size 500m \\
                   --context_length 4096

    # Evaluate a trained checkpoint
    python main.py --mode eval --model_type ngpt --model_size 500m \\
                   --resume outputs/checkpoints/best.pt

    # Run ablation studies (Appendix A.9)
    python main.py --mode ablation --model_type ngpt --model_size 500m

    # Full pipeline
    python main.py --mode all --model_type ngpt --model_size 500m

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=8 main.py --mode train --model_type ngpt \\
             --model_size 500m --context_length 4096
"""

import argparse
import json
import logging
import math
import os
import pathlib
import sys
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn

from config import Config
from data import OpenWebTextDataset
from data import PG19Dataset
from evaluation import AblationRunner
from evaluation import Evaluator
from model import GPTModel
from model import nGPTModel
from trainer import Trainer
import utils


# ---------------------------------------------------------------------------
# Module-level logger (configured after output_dir is known)
# ---------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    """Create a directory and all parent directories if they do not exist.

    Args:
        path: The directory path to create.
    """
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def _resolve_scale_strings(config: Config) -> Config:
    """Resolve string-valued scale fields in a Config to numeric floats.

    The config.yaml file uses symbolic strings like "inv_sqrt_d_model" for
    scale values that depend on model dimensions. This function converts those
    strings to their numeric equivalents after d_model and d_k are known.

    Handled string tokens:
        "inv_sqrt_d_model" -> 1.0 / sqrt(config.d_model)
        "sqrt_d_model"     -> sqrt(config.d_model)
        "sqrt_dk"          -> sqrt(config.d_k)

    Args:
        config: Config instance that may contain string-valued scale fields.

    Returns:
        A new Config instance with all scale fields resolved to floats.
    """
    d_model: int = config.d_model
    d_k: int = config.d_k

    def _resolve(value: Any) -> Any:
        """Resolve a single value, returning it unchanged if not a string token."""
        if not isinstance(value, str):
            return value
        if value == "inv_sqrt_d_model":
            return 1.0 / math.sqrt(d_model)
        if value == "sqrt_d_model":
            return math.sqrt(d_model)
        if value == "sqrt_dk":
            return math.sqrt(d_k)
        # Try parsing as a plain float string
        try:
            return float(value)
        except ValueError:
            return value  # Leave unrecognized strings unchanged

    # Build override dict for fields that might be string tokens
    scale_fields: List[str] = [
        "alpha_a_scale",
        "alpha_m_scale",
        "sqk_scale",
        "sz_scale",
        "alpha_a_init",
        "alpha_m_init",
        "sqk_init",
        "su_init",
        "sv_init",
        "sz_init",
        "su_scale",
        "sv_scale",
    ]

    overrides: Dict[str, Any] = {}
    for field_name in scale_fields:
        current_value = getattr(config, field_name, None)
        if current_value is not None:
            resolved = _resolve(current_value)
            if resolved != current_value:
                overrides[field_name] = resolved

    if overrides:
        return config.replace(**overrides)
    return config


def _lookup_lr(
    model_type: str,
    n_layers: int,
    context_length: int,
) -> float:
    """Look up the representative learning rate from config.yaml.

    Maps (model_type, model_size, context_length) to the representative
    initial learning rate from config.yaml training.gpt / training.ngpt.

    Args:
        model_type: "gpt" or "ngpt".
        n_layers: Number of layers (24 for 0.5B, 36 for 1B).
        context_length: Sequence length (1024, 4096, or 8192).

    Returns:
        Representative initial learning rate as a float.
    """
    # Determine model size from n_layers
    size: str = "500m" if n_layers == 24 else "1b"

    # Map context length to config.yaml key suffix
    ctx_map: Dict[int, str] = {
        1024: "1k",
        4096: "4k",
        8192: "8k",
    }
    ctx_key: str = ctx_map.get(context_length, "4k")

    # Representative learning rates from config.yaml
    # training.gpt.learning_rate_{size}_{ctx} and
    # training.ngpt.learning_rate_{size}_{ctx}
    lr_table: Dict[str, Dict[str, float]] = {
        "gpt": {
            "500m_1k": 3.0e-3,
            "500m_4k": 2.0e-3,
            "500m_8k": 1.0e-3,
            "1b_1k": 2.0e-3,
            "1b_4k": 1.0e-3,
            "1b_8k": 5.0e-4,
        },
        "ngpt": {
            "500m_1k": 2.0e-3,
            "500m_4k": 1.0e-3,
            "500m_8k": 5.0e-4,
            "1b_1k": 2.0e-3,
            "1b_4k": 1.0e-3,
            "1b_8k": 5.0e-4,
        },
    }

    key: str = f"{size}_{ctx_key}"
    return lr_table.get(model_type, {}).get(key, 1.0e-3)


def unwrap_model(
    model: Union[GPTModel, nGPTModel, nn.Module],
) -> Union[GPTModel, nGPTModel]:
    """Unwrap a DDP-wrapped model to access the underlying model instance.

    Args:
        model: A raw GPTModel/nGPTModel or a DDP-wrapped version.

    Returns:
        The underlying GPTModel or nGPTModel instance.
    """
    if hasattr(model, "module"):
        return model.module  # type: ignore[return-value]
    return model  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace with the following attributes:
            config (Optional[str]): Path to YAML config file.
            mode (str): One of "train", "eval", "ablation", "analysis", "all".
            model_type (str): "gpt" or "ngpt".
            model_size (str): "500m" or "1b".
            context_length (int): 1024, 4096, or 8192.
            resume (Optional[str]): Path to checkpoint for resuming training.
            output_dir (str): Root output directory.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce nGPT (Normalized Transformer with Representation "
            "Learning on the Hypersphere) experiments."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to a YAML configuration file. If provided, overrides "
            "factory method defaults. CLI arguments still take precedence "
            "over YAML values."
        ),
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "ablation", "analysis", "all"],
        help=(
            "Execution mode. 'all' runs train → eval → analysis → ablation "
            "in sequence."
        ),
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="ngpt",
        choices=["gpt", "ngpt"],
        help=(
            "Model type. 'gpt' is the baseline Transformer; 'ngpt' is the "
            "Normalized Transformer from the paper."
        ),
    )

    parser.add_argument(
        "--model_size",
        type=str,
        default="500m",
        choices=["500m", "1b"],
        help=(
            "Model size. '500m' uses 24 layers / d_model=1024 (468M params); "
            "'1b' uses 36 layers / d_model=1280 (1026M params). "
            "See paper Table 2."
        ),
    )

    parser.add_argument(
        "--context_length",
        type=int,
        default=4096,
        choices=[1024, 4096, 8192],
        help=(
            "Training context length in tokens. The paper tests 1k, 4k, and "
            "8k. Longer contexts show greater speedup for nGPT (Figure 2)."
        ),
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint file to resume training from. The trainer "
            "will restore model weights, optimizer state, scheduler state, "
            "and training step."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help=(
            "Root output directory. Checkpoints are saved to "
            "output_dir/checkpoints/ and logs to output_dir/logs/."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (config.yaml experiment.seed: 42).",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        default=False,
        help=(
            "After training, run compare_gpt_ngpt() to plot validation loss "
            "curves for both models side by side (reproduces Figure 1)."
        ),
    )

    parser.add_argument(
        "--context_sweep",
        action="store_true",
        default=False,
        help=(
            "Run run_context_length_sweep() to train at 1k/4k/8k context "
            "lengths and plot the speedup comparison (reproduces Figure 2)."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(args: argparse.Namespace) -> Config:
    """Load and resolve the experiment configuration.

    Priority order (highest to lowest):
        1. CLI arguments (--model_type, --context_length, --output_dir, --seed)
        2. YAML file values (if --config is provided)
        3. Factory method defaults (Config.ngpt_500m(), etc.)

    After loading, resolves string-valued scale fields (e.g.,
    "inv_sqrt_d_model") to numeric floats using the resolved d_model.

    Args:
        args: Parsed CLI arguments from parse_args().

    Returns:
        A fully resolved Config instance ready for use.
    """
    # ----------------------------------------------------------------
    # Step 1: Load base config from YAML or factory method
    # ----------------------------------------------------------------
    if args.config is not None:
        # Load from YAML file
        try:
            import yaml  # type: ignore
        except ImportError:
            logger.error(
                "PyYAML is required to load from --config. "
                "Install with: pip install pyyaml"
            )
            sys.exit(1)

        with open(args.config, "r", encoding="utf-8") as f:
            raw_yaml: Dict[str, Any] = yaml.safe_load(f)

        # Flatten the nested YAML structure into a flat dict for Config.from_dict()
        flat_config: Dict[str, Any] = _flatten_yaml_config(
            raw_yaml,
            model_type=args.model_type,
            model_size=args.model_size,
            context_length=args.context_length,
        )
        config: Config = Config.from_dict(flat_config)

    else:
        # Use factory method based on model_type + model_size
        factory_key: str = f"{args.model_type}_{args.model_size}"
        factory_map = {
            "gpt_500m": Config.gpt_500m,
            "ngpt_500m": Config.ngpt_500m,
            "gpt_1b": Config.gpt_1b,
            "ngpt_1b": Config.ngpt_1b,
        }

        if factory_key not in factory_map:
            logger.error(
                "Unknown model configuration: model_type=%s, model_size=%s. "
                "Valid combinations: %s",
                args.model_type,
                args.model_size,
                list(factory_map.keys()),
            )
            sys.exit(1)

        factory_fn = factory_map[factory_key]
        config = factory_fn(context_length=args.context_length)

    # ----------------------------------------------------------------
    # Step 2: Apply CLI overrides (highest priority)
    # ----------------------------------------------------------------
    overrides: Dict[str, Any] = {}

    # model_type and context_length are always applied from CLI
    overrides["model_type"] = args.model_type
    overrides["context_length"] = args.context_length
    overrides["seed"] = args.seed

    # Output directory overrides
    overrides["checkpoint_dir"] = os.path.join(args.output_dir, "checkpoints")
    overrides["log_dir"] = os.path.join(args.output_dir, "logs")
    overrides["cache_dir"] = os.path.join(args.output_dir, "data", "cache")

    config = config.replace(**overrides)

    # ----------------------------------------------------------------
    # Step 3: Resolve string-valued scale fields
    # ----------------------------------------------------------------
    config = _resolve_scale_strings(config)

    # ----------------------------------------------------------------
    # Step 4: Set learning rate from lookup table if not explicitly set
    # ----------------------------------------------------------------
    # Only override if the config's LR is still the factory default
    # (i.e., the user didn't explicitly set it in YAML or CLI)
    looked_up_lr: float = _lookup_lr(
        model_type=config.model_type,
        n_layers=config.n_layers,
        context_length=config.context_length,
    )
    # Use the looked-up LR as the default; factory methods already set this
    # correctly, so this is a no-op in most cases.
    if config.learning_rate == 1.0e-3:  # factory default sentinel
        config = config.replace(learning_rate=looked_up_lr)

    return config


def _flatten_yaml_config(
    raw_yaml: Dict[str, Any],
    model_type: str,
    model_size: str,
    context_length: int,
) -> Dict[str, Any]:
    """Flatten the nested config.yaml structure into a flat dict for Config.from_dict().

    The config.yaml has a nested structure (models, training, architecture,
    etc.). This function extracts the relevant fields and flattens them into
    the flat key-value format expected by Config.from_dict().

    Args:
        raw_yaml: The parsed YAML dictionary.
        model_type: "gpt" or "ngpt" — selects the model config block.
        model_size: "500m" or "1b" — selects the model size block.
        context_length: Sequence length — used to select the LR.

    Returns:
        A flat dictionary suitable for Config.from_dict().
    """
    flat: Dict[str, Any] = {}

    # --- Model architecture ---
    model_key: str = f"{model_type}_{model_size}"
    models_cfg: Dict[str, Any] = raw_yaml.get("models", {})
    if model_key in models_cfg:
        model_cfg: Dict[str, Any] = models_cfg[model_key]
        flat["model_type"] = model_cfg.get("model_type", model_type)
        flat["n_layers"] = model_cfg.get("n_layers", 24)
        flat["d_model"] = model_cfg.get("d_model", 1024)
        flat["n_heads"] = model_cfg.get("n_heads", 16)
        flat["d_mlp"] = model_cfg.get("d_mlp", 4096)
        flat["d_k"] = model_cfg.get("d_k", 64)

    flat["context_length"] = context_length

    # --- Architecture shared settings ---
    arch_cfg: Dict[str, Any] = raw_yaml.get("architecture", {})
    flat["rope_base"] = arch_cfg.get("rope_base", 10000)
    flat["bias"] = arch_cfg.get("bias", False)
    flat["dtype"] = arch_cfg.get("dtype", "bfloat16")

    # --- Training settings ---
    training_cfg: Dict[str, Any] = raw_yaml.get("training", {})
    flat["batch_size"] = training_cfg.get("global_batch_size", 512)
    flat["max_steps"] = training_cfg.get("max_steps", 200000)
    flat["eval_interval"] = training_cfg.get("eval_interval", 500)
    flat["eval_steps"] = training_cfg.get("eval_steps", 100)
    flat["grad_clip"] = training_cfg.get("grad_clip", 1.0)

    # Model-type-specific training settings
    mt_training: Dict[str, Any] = training_cfg.get(model_type, {})
    flat["optimizer"] = mt_training.get(
        "optimizer", "adam" if model_type == "ngpt" else "adamw"
    )
    flat["weight_decay"] = mt_training.get(
        "weight_decay", 0.0 if model_type == "ngpt" else 0.1
    )
    flat["warmup_steps"] = mt_training.get(
        "warmup_steps", 0 if model_type == "ngpt" else 2000
    )
    flat["lr_schedule"] = mt_training.get("lr_schedule", "cosine")
    betas_raw = mt_training.get("betas", [0.9, 0.95])
    flat["betas"] = tuple(betas_raw) if isinstance(betas_raw, list) else betas_raw

    # Learning rate: select based on model size and context length
    size_key: str = "500m" if flat.get("n_layers", 24) == 24 else "1b"
    ctx_key: str = {1024: "1k", 4096: "4k", 8192: "8k"}.get(context_length, "4k")
    lr_field: str = f"learning_rate_{size_key}_{ctx_key}"
    flat["learning_rate"] = mt_training.get(lr_field, 1.0e-3)

    # --- nGPT scaling parameters ---
    ngpt_scaling: Dict[str, Any] = raw_yaml.get("ngpt_scaling", {})
    d_model: int = flat.get("d_model", 1024)

    def _resolve_scale(val: Any) -> Any:
        """Resolve string scale tokens to floats using d_model."""
        if val == "inv_sqrt_d_model":
            return 1.0 / math.sqrt(d_model)
        if val == "sqrt_d_model":
            return math.sqrt(d_model)
        return val

    alpha_a_cfg: Dict[str, Any] = ngpt_scaling.get("alpha_a", {})
    flat["alpha_a_init"] = _resolve_scale(alpha_a_cfg.get("s_init", 0.05))
    flat["alpha_a_scale"] = _resolve_scale(
        alpha_a_cfg.get("s_scale", "inv_sqrt_d_model")
    )

    alpha_m_cfg: Dict[str, Any] = ngpt_scaling.get("alpha_m", {})
    flat["alpha_m_init"] = _resolve_scale(alpha_m_cfg.get("s_init", 0.05))
    flat["alpha_m_scale"] = _resolve_scale(
        alpha_m_cfg.get("s_scale", "inv_sqrt_d_model")
    )

    sqk_cfg: Dict[str, Any] = ngpt_scaling.get("sqk", {})
    flat["sqk_init"] = _resolve_scale(sqk_cfg.get("s_init", 1.0))
    flat["sqk_scale"] = _resolve_scale(
        sqk_cfg.get("s_scale", "inv_sqrt_d_model")
    )

    su_cfg: Dict[str, Any] = ngpt_scaling.get("su", {})
    flat["su_init"] = _resolve_scale(su_cfg.get("s_init", 1.0))
    flat["su_scale"] = _resolve_scale(su_cfg.get("s_scale", 1.0))

    sv_cfg: Dict[str, Any] = ngpt_scaling.get("sv", {})
    flat["sv_init"] = _resolve_scale(sv_cfg.get("s_init", 1.0))
    flat["sv_scale"] = _resolve_scale(sv_cfg.get("s_scale", 1.0))

    sz_cfg: Dict[str, Any] = ngpt_scaling.get("sz", {})
    flat["sz_init"] = _resolve_scale(sz_cfg.get("s_init", 1.0))
    flat["sz_scale"] = _resolve_scale(
        sz_cfg.get("s_scale", "inv_sqrt_d_model")
    )

    # --- nGPT attention specifics ---
    ngpt_attn: Dict[str, Any] = raw_yaml.get("ngpt_attention", {})
    flat["normalize_qk"] = ngpt_attn.get("normalize_qk", True)
    flat["use_lerp"] = ngpt_attn.get("use_lerp", True)

    # --- Data settings ---
    data_cfg: Dict[str, Any] = raw_yaml.get("data", {})
    flat["dataset_name"] = data_cfg.get(
        "dataset_name", "Skylion007/openwebtext"
    )
    flat["tokenizer_name"] = data_cfg.get(
        "tokenizer", "meta-llama/Llama-2-7b-hf"
    )
    flat["tokenizer_fallback"] = data_cfg.get("tokenizer_fallback", "gpt2")
    flat["vocab_size"] = data_cfg.get("vocab_size", 32000)
    flat["train_val_split"] = data_cfg.get("train_val_split", 0.9)

    # --- Evaluation settings ---
    eval_cfg: Dict[str, Any] = raw_yaml.get("evaluation", {})
    flat["downstream_tasks"] = eval_cfg.get(
        "downstream_tasks",
        ["hellaswag", "piqa", "winogrande", "arc_easy", "wmt14-fr-en"],
    )

    # --- Hardware settings ---
    hw_cfg: Dict[str, Any] = raw_yaml.get("hardware", {})
    flat["n_gpus"] = hw_cfg.get("n_gpus", 8)
    flat["micro_batch_size"] = hw_cfg.get("micro_batch_size", 8)
    flat["gradient_accumulation_steps"] = hw_cfg.get(
        "gradient_accumulation_steps", 8
    )
    flat["use_amp"] = hw_cfg.get("use_amp", True)

    # --- Experiment settings ---
    exp_cfg: Dict[str, Any] = raw_yaml.get("experiment", {})
    flat["seed"] = exp_cfg.get("seed", 42)
    flat["checkpoint_dir"] = exp_cfg.get(
        "checkpoint_dir", "outputs/checkpoints"
    )
    flat["log_dir"] = exp_cfg.get("log_dir", "outputs/logs")

    return flat


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

def setup_distributed() -> Tuple[torch.device, bool, int, int]:
    """Initialize distributed training if running under torchrun/DDP.

    Detects the LOCAL_RANK environment variable set by torchrun. If present,
    initializes the NCCL process group and sets the CUDA device for this rank.
    Falls back to single-GPU or CPU if not in a distributed context.

    Returns:
        A tuple (device, is_distributed, rank, world_size) where:
            - device: The torch.device for this process.
            - is_distributed: True if running under DDP.
            - rank: This process's rank (0 for main process).
            - world_size: Total number of processes.
    """
    local_rank_str: Optional[str] = os.environ.get("LOCAL_RANK")

    if local_rank_str is not None:
        # Running under torchrun — initialize DDP
        local_rank: int = int(local_rank_str)

        if not torch.cuda.is_available():
            logger.error(
                "LOCAL_RANK=%d is set but CUDA is not available. "
                "DDP requires CUDA.",
                local_rank,
            )
            sys.exit(1)

        torch.distributed.init_process_group(backend="nccl")
        device: torch.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

        rank: int = torch.distributed.get_rank()
        world_size: int = torch.distributed.get_world_size()

        logger.info(
            "DDP initialized: rank=%d/%d, device=%s",
            rank,
            world_size,
            device,
        )
        return device, True, rank, world_size

    elif torch.cuda.is_available():
        # Single GPU
        device = torch.device("cuda:0")
        return device, False, 0, 1

    else:
        # CPU fallback
        logger.warning(
            "CUDA not available. Running on CPU. "
            "Training will be very slow for large models."
        )
        return torch.device("cpu"), False, 0, 1


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(
    config: Config,
    device: torch.device,
    is_distributed: bool,
    rank: int,
) -> Union[GPTModel, nGPTModel, nn.Module]:
    """Construct and initialize the model, optionally wrapping in DDP.

    For nGPT, normalize_all_weights() is called inside nGPTModel.__init__()
    immediately after weight initialization, so the model is on the
    hypersphere from the start.

    Args:
        config: Experiment configuration.
        device: Target compute device.
        is_distributed: Whether to wrap the model in DDP.
        rank: This process's rank (used for DDP device_ids).

    Returns:
        The constructed model, moved to device and optionally DDP-wrapped.
    """
    if config.model_type == "gpt":
        model: Union[GPTModel, nGPTModel] = GPTModel(config)
    elif config.model_type == "ngpt":
        model = nGPTModel(config)
    else:
        raise ValueError(
            f"Unknown model_type: '{config.model_type}'. "
            "Must be 'gpt' or 'ngpt'."
        )

    # Move to device before DDP wrapping
    model = model.to(device)

    # Wrap in DDP for multi-GPU training
    if is_distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False,
        )
        logger.info(
            "Model wrapped in DistributedDataParallel (rank=%d).", rank
        )

    return model


# ---------------------------------------------------------------------------
# Training mode
# ---------------------------------------------------------------------------

def _run_training(
    config: Config,
    model: Union[GPTModel, nGPTModel, nn.Module],
    dataset: OpenWebTextDataset,
    resume_path: Optional[str],
    rank: int,
) -> Trainer:
    """Run the full training loop.

    Creates a Trainer, optionally loads a checkpoint, and calls trainer.train().
    The Trainer handles all training details including the critical
    post-step normalization for nGPT.

    Args:
        config: Experiment configuration.
        model: The model to train (may be DDP-wrapped).
        dataset: OpenWebText dataset.
        resume_path: Optional path to a checkpoint for resuming training.
        rank: This process's rank (used for logging guards).

    Returns:
        The Trainer instance after training completes. Callers can access
        trainer.val_loss_history for plotting.
    """
    if rank == 0:
        logger.info(
            "Starting training: model_type=%s, n_layers=%d, d_model=%d, "
            "context_length=%d, max_steps=%d, lr=%.2e",
            config.model_type,
            config.n_layers,
            config.d_model,
            config.context_length,
            config.max_steps,
            config.learning_rate,
        )

    trainer: Trainer = Trainer(config, model, dataset)

    # Resume from checkpoint if specified
    if resume_path is not None:
        if rank == 0:
            logger.info("Resuming from checkpoint: %s", resume_path)
        trainer.load_checkpoint(resume_path)

    # Run training loop
    trainer.train()

    return trainer


# ---------------------------------------------------------------------------
# Evaluation mode
# ---------------------------------------------------------------------------

def _run_evaluation(
    config: Config,
    model: Union[GPTModel, nGPTModel, nn.Module],
    dataset: OpenWebTextDataset,
    device: torch.device,
    rank: int,
) -> Dict[str, Any]:
    """Evaluate the model on validation loss and downstream tasks.

    Only rank 0 runs evaluation to avoid duplicate I/O. Returns an empty
    dict on non-main processes.

    Args:
        config: Experiment configuration.
        model: The trained model (may be DDP-wrapped).
        dataset: OpenWebText dataset for validation loss.
        device: