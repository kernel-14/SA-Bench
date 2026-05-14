```python
## main.py
"""Main entry point for OLMoE-1B-7B reproduction.

Orchestrates all training phases (pretraining, SFT, DPO), evaluation
(OLMES, adaptation), and MoE analysis (router saturation, co-activation,
domain/vocabulary specialization) described in the OLMoE paper.

Modes:
    pretrain  - Full pretraining for 5.133T tokens
    ablation  - Controlled design-choice experiments (Section 4)
    sft       - Supervised fine-tuning (Section 4.3, Appendix B)
    dpo       - Direct preference optimization (Section 4.3, Appendix B)
    evaluate  - OLMES post-pretraining or adaptation evaluation (Appendix C)
    analyze   - MoE analysis: router saturation, co-activation,
                domain/vocabulary specialization (Section 5)

Usage:
    # Single GPU:
    python main.py --config config.yaml --mode pretrain

    # Multi-GPU (torchrun):
    torchrun --nproc_per_node=8 main.py --config config.yaml --mode pretrain

    # Ablation experiment:
    python main.py --config config.yaml --mode ablation --ablation granularity

    # SFT from pretrained checkpoint:
    python main.py --config config.yaml --mode sft \
        --checkpoint outputs/pretrain/checkpoint-01223958

    # DPO from SFT checkpoint:
    python main.py --config config.yaml --mode dpo \
        --checkpoint outputs/sft/checkpoint-final

    # OLMES evaluation:
    python main.py --config config.yaml --mode evaluate \
        --checkpoint allenai/OLMoE-1B-7B-0924

    # MoE analysis:
    python main.py --config config.yaml --mode analyze \
        --checkpoint allenai/OLMoE-1B-7B-0924
"""

import argparse
import copy
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed

# ---------------------------------------------------------------------------
# Internal imports — all modules must exist before main.py runs.
# ---------------------------------------------------------------------------
from config import (
    AblationConfig,
    DPOConfig,
    KTOConfig,
    OLMoEConfig,
    SFTConfig,
    TrainingConfig,
)
from model.olmoe_model import OLMoEModel
from utils.distributed import DistributedUtils
from utils.logging_utils import WandbLogger, get_logger, setup_logging
from utils.checkpoint import CheckpointManager

# ---------------------------------------------------------------------------
# Optional OmegaConf import for YAML config loading.
# ---------------------------------------------------------------------------
try:
    from omegaconf import DictConfig, OmegaConf
    OMEGACONF_AVAILABLE: bool = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    OmegaConf = None  # type: ignore[assignment]
    DictConfig = dict  # type: ignore[assignment]
    logging.warning(
        "omegaconf not available. Install with: pip install omegaconf. "
        "Falling back to basic YAML loading."
    )

# ---------------------------------------------------------------------------
# Optional HuggingFace transformers import for tokenizer.
# ---------------------------------------------------------------------------
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE: bool = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None  # type: ignore[assignment,misc]

logger: logging.Logger = get_logger("olmoe.main")

# ---------------------------------------------------------------------------
# Valid mode and ablation names.
# ---------------------------------------------------------------------------
VALID_MODES: List[str] = [
    "pretrain", "ablation", "sft", "dpo", "evaluate", "analyze"
]

VALID_ABLATIONS: List[str] = [
    "moe_vs_dense",
    "granularity",
    "shared_experts",
    "expert_choice_vs_token_choice",
    "sparse_upcycling",
    "load_balancing",
    "router_zloss",
    "dataset",
    "initialization",
    "rmsnorm",
    "qknorm",
    "adamw_eps",
]

# ---------------------------------------------------------------------------
# Default configuration values (fallback when config.yaml is not available).
# These match config.yaml exactly.
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH: str = "config.yaml"
DEFAULT_OUTPUT_DIR: str = "outputs"
DEFAULT_RUN_NAME: str = "olmoe-1b-7b"
DEFAULT_SEED: int = 42
DEFAULT_TOKENIZER_NAME: str = "EleutherAI/gpt-neox-20b"


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for OLMoE training and analysis.

    Returns:
        Parsed argument namespace with all CLI options.
    """
    parser = argparse.ArgumentParser(
        description="OLMoE-1B-7B: Open Mixture-of-Experts Language Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -----------------------------------------------------------------------
    # Required arguments
    # -----------------------------------------------------------------------
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=VALID_MODES,
        default="pretrain",
        help=(
            "Training/evaluation mode. "
            "pretrain: full pretraining; "
            "ablation: design-choice experiments; "
            "sft: instruction tuning; "
            "dpo: preference optimization; "
            "evaluate: OLMES/adaptation evaluation; "
            "analyze: MoE routing analysis."
        ),
    )

    # -----------------------------------------------------------------------
    # Optional arguments
    # -----------------------------------------------------------------------
    parser.add_argument(
        "--ablation",
        type=str,
        choices=VALID_ABLATIONS,
        default=None,
        help=(
            "Ablation experiment name (required when mode=ablation). "
            f"Valid values: {VALID_ABLATIONS}"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint directory or HuggingFace Hub model ID. "
            "Used for evaluate, analyze, sft, and dpo modes. "
            "Example: 'outputs/pretrain/checkpoint-01223958' or "
            "'allenai/OLMoE-1B-7B-0924'."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override the output directory from config.yaml.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Override the W&B run name from config.yaml.",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help=(
            "Config overrides in key=value format. "
            "Example: --overrides model.num_experts=32 pretraining.learning_rate=3e-4"
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set automatically by torchrun).",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Optional path for file logging (rank-specific files will be created).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        default=False,
        help="Disable Weights & Biases logging.",
    )
    parser.add_argument(
        "--eval_tasks",
        nargs="*",
        default=None,
        help=(
            "Specific evaluation tasks to run (for evaluate mode). "
            "If not specified, runs all tasks. "
            "Example: --eval_tasks arc_challenge hellaswag mmlu"
        ),
    )
    parser.add_argument(
        "--analysis_types",
        nargs="*",
        default=None,
        choices=[
            "router_saturation",
            "expert_coactivation",
            "domain_specialization",
            "vocab_specialization",
        ],
        help=(
            "Specific analysis types to run (for analyze mode). "
            "If not specified, runs all four analyses."
        ),
    )
    parser.add_argument(
        "--intermediate_checkpoints",
        nargs="*",
        default=None,
        help=(
            "Paths to intermediate checkpoints for router saturation analysis. "
            "Should correspond to 1%%, 10%%, 20%%, 40%% of pretraining. "
            "Example: --intermediate_checkpoints ckpt-10000 ckpt-120000 ckpt-245000 ckpt-490000"
        ),
    )
    parser.add_argument(
        "--kto",
        action="store_true",
        default=False,
        help="Use KTO instead of DPO for preference tuning (mode=dpo only).",
    )

    args = parser.parse_args()

    # Validate mode-specific requirements.
    if args.mode == "ablation" and args.ablation is None:
        parser.error(
            "--ablation is required when mode=ablation. "
            f"Valid values: {VALID_ABLATIONS}"
        )

    return args


# =============================================================================
# Configuration Loading
# =============================================================================

def load_config(config_path: str, overrides: Optional[List[str]] = None) -> Any:
    """Load and merge configuration from YAML file.

    Args:
        config_path: Path to config.yaml file.
        overrides: Optional list of "key=value" override strings.

    Returns:
        OmegaConf DictConfig (or plain dict if OmegaConf unavailable).

    Raises:
        FileNotFoundError: If config_path does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: '{config_path}'. "
            f"Ensure config.yaml exists in the current directory or "
            f"specify the path with --config."
        )

    if OMEGACONF_AVAILABLE:
        cfg = OmegaConf.load(config_path)
        if overrides:
            override_cfg = OmegaConf.from_dotlist(overrides)
            cfg = OmegaConf.merge(cfg, override_cfg)
        return cfg
    else:
        # Fallback: use PyYAML directly.
        import yaml  # type: ignore[import]
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if overrides:
            for override in overrides:
                if "=" in override:
                    key, value = override.split("=", 1)
                    # Simple dot-notation key setting.
                    keys = key.split(".")
                    d = cfg
                    for k in keys[:-1]:
                        d = d.setdefault(k, {})
                    # Try to parse value as Python literal.
                    try:
                        import ast
                        d[keys[-1]] = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        d[keys[-1]] = value
        return cfg


def cfg_to_dict(cfg: Any) -> Dict[str, Any]:
    """Convert OmegaConf DictConfig or plain dict to a plain Python dict.

    Args:
        cfg: OmegaConf DictConfig or plain dict.

    Returns:
        Plain Python dict.
    """
    if OMEGACONF_AVAILABLE and isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    elif isinstance(cfg, dict):
        return cfg
    else:
        return dict(cfg)


def load_base_configs(
    cfg: Any,
    output_dir_override: Optional[str] = None,
    run_name_override: Optional[str] = None,
) -> Tuple[OLMoEConfig, TrainingConfig]:
    """Extract OLMoEConfig and TrainingConfig from the loaded YAML config.

    Reads from cfg.model and cfg.pretraining sections. Applies any
    command-line overrides for output_dir and run_name.

    Args:
        cfg: Loaded OmegaConf DictConfig or plain dict from config.yaml.
        output_dir_override: Optional override for output directory.
        run_name_override: Optional override for W&B run name.

    Returns:
        Tuple of (OLMoEConfig, TrainingConfig) with all values from config.yaml.

    Raises:
        KeyError: If required config sections are missing.
    """
    cfg_dict: Dict[str, Any] = cfg_to_dict(cfg)

    # Build OLMoEConfig from cfg.model section.
    model_dict: Dict[str, Any] = cfg_dict.get("model", {})
    model_config: OLMoEConfig = OLMoEConfig.from_dict(model_dict)

    # Build TrainingConfig from cfg.pretraining section.
    pretrain_dict: Dict[str, Any] = cfg_dict.get("pretraining", {})

    # Apply output_dir override if provided.
    if output_dir_override is not None:
        pretrain_dict["output_dir"] = output_dir_override

    # Apply run_name override if provided.
    if run_name_override is not None:
        pretrain_dict["run_name"] = run_name_override

    train_config: TrainingConfig = TrainingConfig.from_dict(pretrain_dict)

    # Validate key invariants from config.yaml.
    expected_batch_tokens: int = train_config.batch_size_samples * train_config.seq_len
    if train_config.batch_size_tokens != expected_batch_tokens:
        logger.warning(
            f"batch_size_tokens mismatch: "
            f"batch_size_samples ({train_config.batch_size_samples}) × "
            f"seq_len ({train_config.seq_len}) = {expected_batch_tokens}, "
            f"but batch_size_tokens = {train_config.batch_size_tokens}. "
            f"Using computed value: {expected_batch_tokens}."
        )

    if model_config.hidden_dim % model_config.num_heads != 0:
        raise ValueError(
            f"hidden_dim ({model_config.hidden_dim}) must be divisible by "
            f"num_heads ({model_config.num_heads}). "
            f"Check config.yaml model section."
        )

    if model_config.num_experts < model_config.top_k:
        raise ValueError(
            f"num_experts ({model_config.num_experts}) must be >= "
            f"top_k ({model_config.top_k}). "
            f"Check config.yaml model section."
        )

    logger.info(
        f"Configs loaded: "
        f"hidden_dim={model_config.hidden_dim}, "
        f"num_layers={model_config.num_layers}, "
        f"num_experts={model_config.num_experts}, "
        f"top_k={model_config.top_k}, "
        f"ffn_dim={model_config.ffn_dim}, "
        f"max_steps={train_config.max_steps:,}, "
        f"annealing_steps={train_config.annealing_steps:,}"
    )

    return model_config, train_config


def load_tokenizer(tokenizer_name: str = DEFAULT_TOKENIZER_NAME) -> Any:
    """Load the GPT-NeoX tokenizer (vocab_size=50304).

    Args:
        tokenizer_name: HuggingFace tokenizer identifier.
                        Default: "EleutherAI/gpt-neox-20b" (config.yaml).

    Returns:
        Loaded tokenizer with pad_token set to eos_token.

    Raises:
        ImportError: If transformers library is not installed.
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "HuggingFace 'transformers' library is required for tokenizer loading. "
            "Install with: pip install transformers"
        )

    logger.info(f"Loading tokenizer: '{tokenizer_name}'")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # Set pad token to EOS token if not already set.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Verify vocabulary size matches config.yaml: model.vocab_size = 50304.
    expected_vocab_size: int = 50304
    actual_vocab_size: int = len(tokenizer)
    if actual_vocab_size != expected_vocab_size:
        logger.warning(
            f"Tokenizer vocab_size={actual_vocab_size}, "
            f"expected {expected_vocab_size} (config.yaml: model.vocab_size). "
            f"This may cause embedding dimension mismatches."
        )
    else:
        logger.info(
            f"Tokenizer loaded: vocab_size={actual_vocab_size} ✓ "
            f"(matches config.yaml: model.vocab_size={expected_vocab_size})"
        )

    return tokenizer


def set_random_seed(seed: int, rank: int = 0) -> None:
    """Set random seeds for reproducibility.

    Uses rank-offset to avoid identical data ordering across processes
    while maintaining determinism within each rank.

    Args:
        seed: Base random seed. Default: 42.
        rank: Process rank for offset. Default: 0.
    """
    import random
    import numpy as np

    effective_seed: int = seed + rank
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(effective_seed)
        torch.cuda.manual_seed_all(effective_seed)

    logger.debug(
        f"Random seed set: base={seed}, rank={rank}, effective={effective_seed}"
    )


# =============================================================================
# Model Creation Helpers
# =============================================================================

def create_model(
    model_config: OLMoEConfig,
    device: Optional[torch.device] = None,
) -> OLMoEModel:
    """Create and initialize an OLMoEModel.

    Logs parameter counts after creation.

    Args:
        model_config: OLMoEConfig with all architecture hyperparameters.
        device: Optional device to move the model to. If None, model stays on CPU.

    Returns:
        Initialized OLMoEModel with truncated normal weights.
    """
    logger.info(
        f"Creating OLMoEModel: "
        f"hidden_dim={model_config.hidden_dim}, "
        f"num_layers={model_config.num_layers}, "
        f"num_experts={model_config.num_experts}, "
        f"top_k={model_config.top_k}, "
        f"ffn_dim={model_config.ffn_dim}"
    )

    model: OLMoEModel = OLMoEModel(model_config)

    # Log parameter counts (rank 0 only to avoid duplicate output).
    if DistributedUtils.is_main_process():
        total_params: int = model.num_parameters(active_only=False)
        active_params: int = model.num_parameters(active_only=True)
        logger.info(
            f"OLMoEModel created: "
            f"total_params={total_params:,} ({total_params / 1e9:.2f}B), "
            f"active_params={active_params:,} ({active_params / 1e9:.2f}B)"
        )

    if device is not None:
        model = model.to(device)

    return model


def load_model_from_checkpoint(
    model_config: OLMoEConfig,
    checkpoint_path: str,
    strict: bool = True,
) -> OLMoEModel:
    """Create a model and load weights from a checkpoint.

    Supports both local checkpoint directories and HuggingFace Hub model IDs.

    Args:
        model_config: OLMoEConfig for model architecture.
        checkpoint_path: Local checkpoint directory or HuggingFace Hub model ID.
                         Examples:
                           - "outputs/pretrain/checkpoint-01223958"
                           - "allenai/OLMoE-1B-7B-0924"
        strict: Whether to require exact key matching. Default: True.

    Returns:
        OLMoEModel with weights loaded from the checkpoint.
    """
    model: OLMoEModel = create_model(model_config)
    ckpt_manager: CheckpointManager = CheckpointManager(
        output_dir="outputs_temp_load",
        max_checkpoints=None,
    )
    ckpt_manager.load_model_only(
        path=checkpoint_path,
        model=model,
        strict=strict,
    )
    logger.info(f"Model weights loaded from: '{checkpoint_path}'")
    return model


# =============================================================================
# Mode: Pretraining
# =============================================================================

def run_pretrain(
    cfg: Any,
    args: argparse.Namespace,
) -> None:
    """Run full OLMoE-1B-7B pretraining for 5.133T tokens.

    Implements the pretraining procedure from Section 2 and Appendix B:
      - 5.133T tokens (1.3 epochs of OLMoE-Mix)
      - Three-phase LR: warmup (2500 steps) → cosine → linear annealing (100B tokens)
      - Load balancing loss (α=0.01) + router z-loss (β=0.001)
      - BF16 mixed precision, FSDP ZeRO-3
      - Checkpoints every 5000 steps

    Args:
        cfg: Loaded OmegaConf DictConfig from config.yaml.
        args: Parsed command-line arguments.
    """
    logger.info("=" * 60)
    logger.info("Mode: PRETRAIN")
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # Load configurations.
    # -----------------------------------------------------------------------
    output_dir: str = args.output_dir or os.path.join(
        cfg_to_dict(cfg).get("pretraining", {}).get("output_dir", DEFAULT_OUTPUT_DIR),
        "pretrain",
    )
    run_name: str = args.run_name or "olmoe-1b-7b-pretrain"

    model_config, train_config = load_base_configs(
        cfg,
        output_dir_override=output_dir,
        run_name_override=run_name,
    )

    cfg_dict: Dict[str, Any] = cfg_to_dict(cfg)

    # -----------------------------------------------------------------------
    # Load tokenizer.
    # -----------------------------------------------------------------------
    tokenizer_name: str = cfg_dict.get("pretraining", {}).get(
        "tokenizer_name", DEFAULT_TOKENIZER_NAME
    )
    tokenizer = load_tokenizer(tokenizer_name)

    # -----------------------------------------------------------------------
    # Create DatasetLoader and training DataLoader.
    # -----------------------------------------------------------------------
    from data.dataset_loader import DatasetLoader

    data_loader_factory: DatasetLoader = DatasetLoader(
        config=train_config,
        tokenizer_name=tokenizer_name,
    )

    data_mix: Dict[str, Any] = cfg_dict.get("pretraining", {}).get("data_mix", {})
    train_dataloader = data_loader_factory.load_pretraining_data(data_mix=data_mix)

    # -----------------------------------------------------------------------
    # Create model and wrap with FSDP.
    # -----------------------------------------------------------------------
    model: OLMoEModel = create_model(model_config)

    # Check for existing checkpoint to resume from.
    ckpt_manager: CheckpointManager = CheckpointManager(
        output_dir=output_dir,
        max_checkpoints=None,  # Keep all checkpoints (paper releases every 5000 steps)
    )

    existing_checkpoints: List[str] = ckpt_manager.list_checkpoints()
    resume_step: int = 0

    # Wrap with FSDP before loading checkpoint (FSDP requires wrapping first).
    model = DistributedUtils.setup_fsdp(model, train_config)

    # -----------------------------------------------------------------------
    # Create optimizer, scheduler, and auxiliary losses.
    # -----------------------------------------------------------------------
    from training.optimizer import create_optimizer
    from training.lr_scheduler import LRScheduler
    from training.losses import AuxiliaryLosses

    optimizer = create_optimizer(model, train_config, phase="pretrain")
    scheduler: LRScheduler = LRScheduler(optimizer=optimizer, config=train_config)
    aux_losses: AuxiliaryLosses = AuxiliaryLosses(model_config)

    # Resume from checkpoint if available.
    if existing_checkpoints:
        latest_ckpt: str = existing_checkpoints[-1]
        logger.info(f"Resuming from checkpoint: '{latest_ckpt}'")
        try:
            resume_step = ckpt_manager.load(
                path=latest_ckpt,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            logger.info(f"Resumed from step {resume_step:,}")
        except Exception as exc:
            logger.warning(
                f"Failed to load checkpoint '{latest_ckpt}': "
                f"{type(exc).__name__}: {exc}. "
                f"Starting from scratch."
            )
            resume_step = 0

    # -----------------------------------------------------------------------
    # Create W&B logger.
    # -----------------------------------------------------------------------
    wandb_logger: WandbLogger
    if args.no_wandb:
        wandb_logger = WandbLogger.__new__(WandbLogger)
        wandb_logger.project = cfg_dict.get("pretraining", {}).get("wandb_project", "olmoe")
        wandb_logger.run_name = run_name
        wandb_logger._enabled = False
        wandb_logger._run = None
    else:
        wandb_logger = WandbLogger(
            project=cfg_dict.get("pretraining", {}).get("wandb_project", "olmoe"),
            run_name=run_name,
            config_dict=cfg_dict,
        )

    # -----------------------------------------------------------------------
    # Create in-loop evaluator.
    # -----------------------------------------------------------------------
    from evaluation.evaluator import Evaluator

    device_str: str = (
        f"cuda:{torch.cuda.current_device()}"
        if torch.cuda.is_available()
        else "cpu"
    )
    evaluator: Evaluator = Evaluator(
        model=model,
        tokenizer=tokenizer,
        device=device_str,
    )

    # -----------------------------------------------------------------------
    # Create and run Trainer.
    # -----------------------------------------------------------------------
    from training.trainer import Trainer

    trainer: Trainer = Trainer(
        model=model,
        train_loader=train_dataloader,
        config=train_config,
        aux_losses=aux_losses,
        scheduler=scheduler,
        optimizer=optimizer,
        wandb_logger=wandb_logger,
        checkpoint_manager=ckpt_manager,
        evaluator=evaluator,
    )

    # Set global_step to resume_step if resuming.
    if resume_step > 0:
        trainer.global_step = resume_step
        logger.info(f"Trainer global_step set to {resume_step:,} for resumption.")

    logger.info(
        f"Starting pretraining: "
        f"max_steps={train_config.max_steps:,}, "
        f"total_tokens={train_config.total_tokens:,}, "
        f"batch_size_tokens={train_config.batch_size_tokens:,}, "
        f"world_size={DistributedUtils.get_world_size()}"
    )

    trainer.train()

    logger.info("Pretraining complete.")


# =============================================================================
# Mode: Ablation
# =============================================================================

def apply_ablation_overrides(
    model_config: OLMoEConfig,
    train_config: TrainingConfig,
    cfg_dict: Dict[str, Any],
    ablation_name: str,
) -> Tuple[OLMoEConfig, TrainingConfig]:
    """Apply ablation-specific configuration overrides.

    Modifies model_config and train_config to implement the specific
    ablation experiment from Section 4 of the paper.

    Args:
        model_config: Base OLMoEConfig to modify.
        train_config: Base TrainingConfig to modify.
        cfg_dict: Full config dict from config.yaml (for ablation parameters).
        ablation_name: Name of the ablation experiment.

    Returns:
        Tuple of (modified_model_config, modified_train_config).
    """
    ablations_cfg: Dict[str, Any] = cfg_dict.get("ablations", {})

    # Use dataclass copy to avoid mutating the original.
    import dataclasses
    model_cfg: OLMoEConfig = dataclasses.replace(model_config)
    train_cfg: TrainingConfig = dataclasses.replace(train_config)

    # Set ablation token budget (130B default, except initialization needs ~500B).
    default_ablation_tokens: int = ablations_cfg.get(
        "default_ablation_tokens", 130_000_000_000
    )
    ablation_tokens: int = default_ablation_tokens

    if ablation_name == "initialization":
        # Signal only becomes clear at ~450B tokens (Section 4.2.2, Figure 13).
        ablation_tokens = 500_000_000_000

    # Recompute max_steps for the ablation token budget.
    # We need to create a new TrainingConfig with the modified total_tokens.
    # Since TrainingConfig.max_steps is computed in __post_init__, we rebuild it.
    train_cfg_dict: Dict[str, Any] = train_cfg.to_dict()
    train_cfg_dict["total_tokens"] = ablation_tokens
    train_cfg_dict["annealing_tokens"] = min(
        10_000_000_000,  # 10B annealing for ablations
        ablation_tokens // 10,
    )
    train_cfg = TrainingConfig.from_dict(train_cfg_dict)

    # -----------------------------------------------------------------------
    # Apply ablation-specific modifications.
    # -----------------------------------------------------------------------
    if ablation_name == "moe_vs_dense":
        # Dense model: same active params as MoE but no sparsity.
        # FFN dim = 8192 for dense (vs 1024 per expert for MoE).
        # This is handled by running two separate experiments.
        # The dense config uses num_experts=1, top_k=1, ffn_dim=8192.
        # The MoE config uses the base model_config.
        # main.py runs both and logs results for comparison.
        logger.info(
            "Ablation 'moe_vs