## Code: main.py

```python
## main.py
"""Entry point for Hi-MAR: Hierarchical Masked Autoregressive Models.

Provides a unified CLI for training (ImageNet class-conditional and MS-COCO
text-to-image), evaluation (FID/IS/Precision/Recall/T2I-CompBench), and
qualitative sample generation.

Usage examples:
    # Single-GPU training on ImageNet
    python main.py --config configs/himar_b.yaml --task train_imagenet

    # Multi-GPU distributed training (torchrun)
    torchrun --nproc_per_node=8 main.py --config configs/himar_b.yaml --task train_imagenet

    # Evaluation on ImageNet
    python main.py --config configs/himar_b.yaml --task eval_imagenet --ckpt outputs/checkpoints/epoch_800.pt

    # Text-to-image generation on COCO
    python main.py --config configs/himar_s.yaml --task eval_coco --ckpt outputs/checkpoints/best.pt

    # Qualitative sample generation
    python main.py --config configs/himar_b.yaml --task generate --ckpt outputs/checkpoints/epoch_800.pt
"""

import argparse
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.coco_dataset import COCODataset, build_coco_dataloader
from data.imagenet_dataset import ImageNetDataset, build_imagenet_dataloader
from evaluation.evaluator import Evaluator
from inference.generate import Generator
from models.himar import HiMAR, HiMARConfig
from models.vae_tokenizer import VAETokenizer
from training.trainer import Trainer, TrainerConfig
from utils.misc import (
    AverageMeter,
    count_parameters,
    get_device,
    load_yaml,
    save_yaml,
    set_seed,
    setup_logger,
)


# ---------------------------------------------------------------------------
# Distributed info container
# ---------------------------------------------------------------------------


@dataclass
class DistributedInfo:
    """Container for distributed training state.

    Attributes:
        is_distributed: Whether torch.distributed is active.
        world_size: Total number of processes (GPUs).
        rank: Global rank of this process (0 = primary).
        local_rank: Local rank on this node (used for device assignment).
        device: Compute device for this process.
    """

    is_distributed: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    device: torch.device = field(default_factory=get_device)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the Hi-MAR entry point.

    Returns:
        Parsed argument namespace. All optional arguments have sensible
        defaults so that the most common invocations require only
        ``--config`` and ``--task``.
    """
    parser = argparse.ArgumentParser(
        description="Hi-MAR: Hierarchical Masked Autoregressive Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments.
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file (e.g. configs/himar_b.yaml).",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["train_imagenet", "train_coco", "eval_imagenet", "eval_coco", "generate"],
        help=(
            "Task to execute. "
            "'train_imagenet': class-conditional training on ImageNet. "
            "'train_coco': text-to-image training on MS-COCO. "
            "'eval_imagenet': FID/IS/Precision/Recall evaluation on ImageNet. "
            "'eval_coco': FID/T2I-CompBench evaluation on MS-COCO. "
            "'generate': qualitative sample generation."
        ),
    )

    # Optional arguments.
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to a checkpoint file for evaluation or generation tasks.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override the output directory from config.yaml.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Config default: seed=42.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of DataLoader worker processes. Config default: num_workers=8.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help=(
            "Local GPU rank for distributed training. "
            "Set automatically by torchrun via the LOCAL_RANK environment variable."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override the batch size from config.yaml.",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=None,
        help="Override the CFG scale from config.yaml (inference.cfg.scale).",
    )
    parser.add_argument(
        "--phase1_steps",
        type=int,
        default=None,
        help="Override Phase 1 AR steps from config.yaml (inference.phase1_steps).",
    )
    parser.add_argument(
        "--phase2_steps",
        type=int,
        default=None,
        help="Override Phase 2 AR steps from config.yaml (inference.phase2_steps).",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Override the number of evaluation samples from config.yaml.",
    )
    parser.add_argument(
        "--no_cfg_phase2",
        action="store_true",
        default=False,
        help=(
            "Disable CFG for Phase 2 (w/o CFG setting). "
            "Per paper: 'the CFG is only turned off during the prediction of "
            "dense tokens.' Phase 1 CFG remains enabled."
        ),
    )
    parser.add_argument(
        "--force_regenerate",
        action="store_true",
        default=False,
        help="Force regeneration of images even if they already exist on disk.",
    )
    parser.add_argument(
        "--speed_sweep",
        action="store_true",
        default=False,
        help=(
            "Run the speed/accuracy trade-off sweep (Figure 3 in the paper). "
            "Only applicable with --task eval_imagenet."
        ),
    )

    args: argparse.Namespace = parser.parse_args()

    # Override local_rank from environment variable if set by torchrun.
    # torchrun sets LOCAL_RANK before launching the process.
    env_local_rank: Optional[str] = os.environ.get("LOCAL_RANK")
    if env_local_rank is not None:
        args.local_rank = int(env_local_rank)

    return args


# ---------------------------------------------------------------------------
# Distributed training setup
# ---------------------------------------------------------------------------


def _setup_distributed(local_rank: int) -> DistributedInfo:
    """Initialises torch.distributed if multiple GPUs are available.

    Detects distributed training by checking whether the ``LOCAL_RANK``
    environment variable is set (injected by ``torchrun``) or whether
    ``local_rank > 0``. Uses NCCL backend, which is optimal for H100 GPUs.

    For single-GPU or CPU training, returns a ``DistributedInfo`` with
    ``is_distributed=False`` and ``world_size=1``.

    Args:
        local_rank: Local GPU rank for this process. 0 for single-GPU.
            Set by ``torchrun`` via the ``LOCAL_RANK`` environment variable.

    Returns:
        ``DistributedInfo`` instance with all distributed state populated.
    """
    # Check whether distributed training is requested.
    # torchrun sets WORLD_SIZE > 1 when launching multiple processes.
    world_size_env: Optional[str] = os.environ.get("WORLD_SIZE")
    rank_env: Optional[str] = os.environ.get("RANK")

    is_distributed: bool = (
        world_size_env is not None
        and int(world_size_env) > 1
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 1
    )

    if is_distributed:
        # Initialise the process group with NCCL backend.
        # NCCL is the recommended backend for GPU-to-GPU communication on H100.
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl",
                init_method="env://",
            )

        world_size: int = torch.distributed.get_world_size()
        rank: int = torch.distributed.get_rank()
        local_rank_actual: int = local_rank

        # Bind this process to its assigned GPU.
        torch.cuda.set_device(local_rank_actual)
        device: torch.device = torch.device(f"cuda:{local_rank_actual}")

        return DistributedInfo(
            is_distributed=True,
            world_size=world_size,
            rank=rank,
            local_rank=local_rank_actual,
            device=device,
        )
    else:
        # Single-GPU or CPU training.
        device = get_device()
        return DistributedInfo(
            is_distributed=False,
            world_size=1,
            rank=0,
            local_rank=0,
            device=device,
        )


# ---------------------------------------------------------------------------
# Config loading and resolution
# ---------------------------------------------------------------------------


def _load_and_resolve_config(
    config_path: str,
    args: argparse.Namespace,
    task: str,
) -> Dict[str, Any]:
    """Loads the YAML config and resolves the active model variant.

    Reads the top-level config.yaml (or a variant-specific config), selects
    the active model sub-config based on ``config.active_model``, and flattens
    all relevant fields into a single dictionary that can be used to construct
    ``HiMARConfig``, ``TrainerConfig``, and other dataclasses.

    The resolution priority for overridable fields is:
        CLI args > config.yaml values > dataclass defaults

    Args:
        config_path: Path to the YAML configuration file.
        args: Parsed CLI arguments. Used to apply overrides.
        task: Task identifier ('train_imagenet', 'train_coco', etc.).
            Determines which training sub-config to use for task-dependent
            fields (lr, weight_decay, epochs, etc.).

    Returns:
        Flat dictionary with all resolved configuration values. Keys match
        the field names of ``HiMARConfig`` and ``TrainerConfig``.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        KeyError: If required config fields are missing.
        ValueError: If ``config.active_model`` is not a valid model variant.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: '{config_path}'. "
            "Please provide a valid path via --config."
        )

    raw: Dict[str, Any] = load_yaml(config_path)

    # ------------------------------------------------------------------
    # Resolve active model variant.
    # config.active_model selects from config.models.{himar_b,l,h,s}.
    # ------------------------------------------------------------------
    active_model: str = raw.get("active_model", "himar_b")
    valid_models: Tuple[str, ...] = ("himar_b", "himar_l", "himar_h", "himar_s")
    if active_model not in valid_models:
        raise ValueError(
            f"config.active_model='{active_model}' is not valid. "
            f"Must be one of: {valid_models}."
        )

    models_cfg: Dict[str, Any] = raw.get("models", {})
    if active_model not in models_cfg:
        raise KeyError(
            f"Model variant '{active_model}' not found in config.models. "
            f"Available variants: {list(models_cfg.keys())}."
        )

    model_cfg: Dict[str, Any] = models_cfg[active_model]
    transformer_cfg: Dict[str, Any] = model_cfg.get("transformer", {})
    diff_head1_cfg: Dict[str, Any] = model_cfg.get("diff_head1", {})
    diff_head2_cfg: Dict[str, Any] = model_cfg.get("diff_head2", {})

    # ------------------------------------------------------------------
    # VAE configuration.
    # ------------------------------------------------------------------
    vae_cfg: Dict[str, Any] = raw.get("vae", {})

    # ------------------------------------------------------------------
    # Resolution configuration.
    # ------------------------------------------------------------------
    resolution_cfg: Dict[str, Any] = raw.get("resolution", {})

    # ------------------------------------------------------------------
    # Diffusion configuration.
    # ------------------------------------------------------------------
    diffusion_cfg: Dict[str, Any] = raw.get("diffusion", {})

    # ------------------------------------------------------------------
    # Inference configuration.
    # ------------------------------------------------------------------
    inference_cfg: Dict[str, Any] = raw.get("inference", {})
    cfg_inference: Dict[str, Any] = inference_cfg.get("cfg", {})

    # ------------------------------------------------------------------
    # Task-dependent training configuration.
    # ------------------------------------------------------------------
    training_imagenet_cfg: Dict[str, Any] = raw.get("training_imagenet", {})
    training_coco_cfg: Dict[str, Any] = raw.get("training_coco", {})

    # Select the active training config based on task.
    if "imagenet" in task or task == "generate":
        active_training_cfg: Dict[str, Any] = training_imagenet_cfg
    else:
        active_training_cfg = training_coco_cfg

    # ------------------------------------------------------------------
    # Output configuration.
    # ------------------------------------------------------------------
    output_cfg: Dict[str, Any] = raw.get("output", {})

    # ------------------------------------------------------------------
    # Data paths.
    # ------------------------------------------------------------------
    data_cfg: Dict[str, Any] = raw.get("data", {})

    # ------------------------------------------------------------------
    # Evaluation configuration.
    # ------------------------------------------------------------------
    evaluation_cfg: Dict[str, Any] = raw.get("evaluation", {})

    # ------------------------------------------------------------------
    # Flatten all fields into a single resolved dictionary.
    # ------------------------------------------------------------------
    resolved: Dict[str, Any] = {
        # Model identity.
        "model_type": active_model,

        # Transformer backbone (from active model variant).
        "n_layers": transformer_cfg.get("n_layers", 24),
        "hidden_size": transformer_cfg.get("hidden_size", 768),
        "n_heads": transformer_cfg.get("n_heads", 12),
        "mlp_ratio": transformer_cfg.get("mlp_ratio", 4.0),

        # Diffusion heads (from active model variant).
        "diff_head1_layers": diff_head1_cfg.get("n_layers", 6),
        "diff_head1_hidden": diff_head1_cfg.get("hidden_size", 1024),
        "diff_head2_layers": diff_head2_cfg.get("n_layers", 6),
        "diff_head2_hidden": diff_head2_cfg.get("hidden_size", 512),

        # VAE.
        "vae_ckpt": vae_cfg.get("ckpt", "pretrained/kl16.ckpt"),
        "vae_scale_factor": vae_cfg.get("scale_factor", 0.2325),
        "latent_dim": vae_cfg.get("latent_channels", 16),

        # Resolution.
        "high_res": resolution_cfg.get("high_res", 256),
        "low_res": resolution_cfg.get("low_res", 128),
        "hr_seq_len": resolution_cfg.get("hr_seq_len", 256),
        "lr_seq_len": resolution_cfg.get("lr_seq_len", 64),

        # Diffusion schedule.
        "diff_timesteps": diffusion_cfg.get("timesteps", 100),
        "beta_start": diffusion_cfg.get("beta_start", 0.0001),
        "beta_end": diffusion_cfg.get("beta_end", 0.02),

        # Inference.
        "cfg_scale": inference_cfg.get("cfg", {}).get("scale", 2.9),
        "phase1_steps": inference_cfg.get("phase1_steps", 32),
        "phase2_steps": inference_cfg.get("phase2_steps", 4),
        "phase1_cfg_enabled": cfg_inference.get("phase1_cfg_enabled", True),
        "phase2_cfg_enabled": cfg_inference.get("phase2_cfg_enabled", True),
        "phase2_cfg_disabled_for_nocfg": cfg_inference.get(
            "phase2_cfg_disabled_for_nocfg", True
        ),

        # Speed/accuracy sweep.
        "speed_sweep_phase1_steps": inference_cfg.get(
            "speed_accuracy_sweep", {}
        ).get("phase1_steps_fixed", 32),
        "speed_sweep_phase2_steps": inference_cfg.get(
            "speed_accuracy_sweep", {}
        ).get("phase2_steps_sweep", [1, 2, 4, 6, 8]),
        "speed_sweep_batch_size": inference_cfg.get(
            "speed_accuracy_sweep", {}
        ).get("batch_size", 128),

        # Training — ImageNet.
        "imagenet_lr": training_imagenet_cfg.get("lr_schedule", {}).get(
            "base_lr", 1e-4
        ),
        "imagenet_weight_decay": training_imagenet_cfg.get("optimizer", {}).get(
            "weight_decay", 0.02
        ),
        "imagenet_beta1": training_imagenet_cfg.get("optimizer", {}).get(
            "beta1", 0.9
        ),
        "imagenet_beta2": training_imagenet_cfg.get("optimizer", {}).get(
            "beta2", 0.95
        ),
        "imagenet_epochs": training_imagenet_cfg.get("epochs", 800),
        "imagenet_warmup_epochs": training_imagenet_cfg.get(
            "lr_schedule", {}
        ).get("warmup_epochs", 100),
        "imagenet_batch_size": training_imagenet_cfg.get("batch_size", 2048),
        "imagenet_grad_clip": training_imagenet_cfg.get("grad_clip_norm", 1.0),
        "imagenet_ema_enabled": training_imagenet_cfg.get("ema", {}).get(
            "enabled", False
        ),
        "imagenet_ema_momentum": training_imagenet_cfg.get("ema", {}).get(
            "momentum", 0.9999
        ),
        "mask_ratio_min": training_imagenet_cfg.get("masking", {}).get(
            "phase1", {}
        ).get("ratio_min", 0.7),
        "mask_ratio_max": training_imagenet_cfg.get("masking", {}).get(
            "phase1", {}
        ).get("ratio_max", 1.0),
        "n_classes": training_imagenet_cfg.get("n_classes", 1000),

        # Training — COCO.
        "coco_lr": training_coco_cfg.get("lr_schedule", {}).get("base_lr", 8e-4),
        "coco_weight_decay": training_coco_cfg.get("optimizer", {}).get(
            "weight_decay", 0.03
        ),
        "coco_beta1": training_coco_cfg.get("optimizer", {}).get("beta1", 0.9),
        "coco_beta2": training_coco_cfg.get("optimizer", {}).get("beta2", 0.999),
        "coco_epochs": training_coco_cfg.get("epochs", None),
        "coco_warmup_steps": training_coco_cfg.get("lr_schedule", {}).get(
            "warmup_steps", 8000
        ),
        "coco_batch_size": training_coco_cfg.get("batch_size", None),
        "coco_grad_clip": training_coco_cfg.get("grad_clip_norm", 1.0),
        "coco_ema_enabled": training_coco_cfg.get("ema", {}).get("enabled", True),
        "coco_ema_momentum": training_coco_cfg.get("ema", {}).get(
            "momentum", 0.9999
        ),
        "beta_alpha": training_coco_cfg.get("masking", {}).get("phase1", {}).get(
            "beta_alpha", 4.0
        ),
        "beta_beta": training_coco_cfg.get("masking", {}).get("phase1", {}).get(
            "beta_beta", 1.0
        ),
        "clip_model_name": training_coco_cfg.get("text_encoder", {}).get(
            "model", "openai/clip-vit-large-patch14"
        ),
        "clip_max_length": training_coco_cfg.get("text_encoder", {}).get(
            "max_length", 77
        ),

        # Data paths.
        "imagenet_train_root": data_cfg.get("imagenet", {}).get(
            "train_root", "data/imagenet/train"
        ),
        "imagenet_val_root": data_cfg.get("imagenet", {}).get(
            "val_root", "data/imagenet/val"
        ),
        "coco_train_root": data_cfg.get("coco", {}).get(
            "train_root", "data/coco/train2017"
        ),
        "coco_val_root": data_cfg.get("coco", {}).get(
            "val_root", "data/coco/val2017"
        ),
        "coco_train_ann": data_cfg.get("coco", {}).get(
            "train_ann",
            "data/coco/annotations/captions_train2017.json",
        ),
        "coco_val_ann": data_cfg.get("coco", {}).get(
            "val_ann",
            "data/coco/annotations/captions_val2017.json",
        ),

        # Output paths.
        "output_dir": output_cfg.get("dir", "outputs"),
        "checkpoint_dir": output_cfg.get("checkpoint_dir", "outputs/checkpoints"),
        "log_dir": output_cfg.get("log_dir", "outputs/logs"),
        "sample_dir": output_cfg.get("sample_dir", "outputs/samples"),
        "save_every_epochs": output_cfg.get("save_every_epochs", 50),
        "log_every_steps": output_cfg.get("log_every_steps", 100),

        # Evaluation.
        "eval_imagenet_n_samples": evaluation_cfg.get("imagenet", {}).get(
            "n_samples", 50000
        ),
        "eval_coco_n_samples": evaluation_cfg.get("coco", {}).get(
            "n_samples", 30000
        ),
        "t2i_compbench_enabled": evaluation_cfg.get("coco", {}).get(
            "t2i_compbench", {}
        ).get("enabled", True),

        # Reproducibility.
        "seed": raw.get("seed", 42),
        "num_workers": raw.get("num_workers", 8),
        "pin_memory": raw.get("pin_memory", True),
        "mixed_precision": raw.get("mixed_precision", True),

        # CLIP embedding dimension (fixed by openai/clip-vit-large-patch14).
        "clip_dim": 768,
    }

    # ------------------------------------------------------------------
    # Apply CLI overrides (highest priority).
    # ------------------------------------------------------------------
    if args.output_dir is not None:
        resolved["output_dir"] = args.output_dir
        resolved["checkpoint_dir"] = os.path.join(args.output_dir, "checkpoints")
        resolved["log_dir"] = os.path.join(args.output_dir, "logs")
        resolved["sample_dir"] = os.path.join(args.output_dir, "samples")

    if args.batch_size is not None:
        resolved["imagenet_batch_size"] = args.batch_size
        resolved["coco_batch_size"] = args.batch_size

    if args.cfg_scale is not None:
        resolved["cfg_scale"] = args.cfg_scale

    if args.phase1_steps is not None:
        resolved["phase1_steps"] = args.phase1_steps

    if args.phase2_steps is not None:
        resolved["phase2_steps"] = args.phase2_steps

    if args.n_samples is not None:
        resolved["eval_imagenet_n_samples"] = args.n_samples
        resolved["eval_coco_n_samples"] = args.n_samples

    return resolved


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(
    resolved_cfg: Dict[str, Any],
    args: argparse.Namespace,
    rank: int,
) -> logging.Logger:
    """Configures logging and archives the config for reproducibility.

    Only the primary process (rank 0) writes logs and archives the config.
    Other processes get a minimal logger that discards output.

    Args:
        resolved_cfg: Flat resolved configuration dictionary.
        args: Parsed CLI arguments.
        rank: Global rank of this process. Only rank 0 performs I/O.

    Returns:
        Configured ``logging.Logger`` instance.
    """
    output_dir: str = resolved_cfg["output_dir"]
    log_dir: str = resolved_cfg["log_dir"]
    task: str = args.task

    if rank == 0:
        # Create output directory structure.
        for directory in (
            output_dir,
            log_dir,
            resolved_cfg["checkpoint_dir"],
            resolved_cfg["sample_dir"],
        ):
            os.makedirs(directory, exist_ok=True)

        # Setup logger with file and console handlers.
        log_file: str = os.path.join(log_dir, f"{task}.log")
        logger: logging.Logger = setup_logger("himar", log_file)

        # Archive the config file for reproducibility.
        config_archive_path: str = os.path.join(output_dir, "config_used.yaml")
        if os.path.isfile(args.config):
            shutil.copy(args.config, config_archive_path)
            logger.info(f"Config archived to: {config_archive_path}")

        # Log system information.
        logger.info("=" * 60)
        logger.info("Hi-MAR: Hierarchical Masked Autoregressive Models")
        logger.info("=" * 60)
        logger.info(f"Task: {task}")
        logger.info(f"Config: {args.config}")
        logger.info(f"Active model: {resolved_cfg['model_type']}")
        logger.info(f"Output dir: {output_dir}")
        logger.info(f"Seed: {args.seed}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"Model config: n_layers={resolved_cfg['n_layers']}, "
            f"hidden_size={resolved_cfg['hidden_size']}, "
            f"n_heads={resolved_cfg['n_heads']}"
        )

        return logger
    else:
        # Non-primary processes get a null logger.
        null_logger: logging.Logger = logging.getLogger(f"himar_rank{rank}")
        null_logger.addHandler(logging.NullHandler())
        return null_logger


# ---------------------------------------------------------------------------
# Model construction helpers
# ---------------------------------------------------------------------------


def _build_himar_config(
    resolved_cfg: Dict[str, Any],
    task: str,
) -> HiMARConfig:
    """Constructs a HiMARConfig from the resolved configuration dictionary.

    Maps the flat resolved config fields to the HiMARConfig dataclass fields.
    The masking strategy is set based on the task: 'uniform' for ImageNet,
    'beta' for COCO.

    Args:
        resolved_cfg: Flat resolved configuration dictionary from
            ``_load_and_resolve_config``.
        task: Task identifier. Determines masking strategy.

    Returns:
        Populated ``HiMARConfig`` instance.
    """
    # Determine Phase 1 masking strategy based on task.
    # Paper Section 4.2:
    #   ImageNet: "masking ratio is randomly sampled in [0.7, 1.0] as MAR"
    #   COCO: "randomly sample the masking ratio by Beta distribution (α=4, β=1)"
    phase1_masking: str = "beta" if "coco" in task else "uniform"

    return HiMARConfig(
        model_type=resolved_cfg["model_type"],
        n_layers=resolved_cfg["n_layers"],
        hidden_size=resolved_cfg["hidden_size"],
        n_heads=resolved_cfg["n_heads"],
        mlp_ratio=resolved_cfg["mlp_ratio"],
        diff_head1_layers=resolved_cfg["diff_head1_layers"],
        diff_head1_hidden=resolved_cfg["diff_head1_hidden"],
        diff_head2_layers=resolved_cfg["diff_head2_layers"],
        diff_head2_hidden=resolved_cfg["diff_head2_hidden"],
        n_classes=resolved_cfg["n_classes"],
        latent_dim=resolved_cfg["latent_dim"],
        lr_seq_len=resolved_cfg["lr_seq_len"],
        hr_seq_len=resolved_cfg["hr_seq_len"],
        clip_dim=resolved_cfg["clip_dim"],
        diff_timesteps=resolved_cfg["diff_timesteps"],
        cfg_scale=resolved_cfg["cfg_scale"],
        phase1_steps=resolved_cfg["phase1_steps"],
        phase2_steps=resolved_cfg["phase2_steps"],
        mask_ratio_min=resolved_cfg["mask_ratio_min"],
        mask_ratio_max=resolved_cfg["mask_ratio_max"],
        beta_alpha=resolved_cfg["beta_alpha"],
        beta_beta=resolved_cfg["beta_beta"],
        phase1_masking_strategy=phase1_masking,
        use_ema_for_inference=True,
    )


def _build_trainer_config(
    resolved_cfg: Dict[str, Any],
    task: str,
) -> TrainerConfig:
    """Constructs a TrainerConfig from the resolved configuration dictionary.

    Selects task-dependent fields (lr, weight_decay, epochs, etc.) based on
    whether the task is ImageNet or COCO.

    Args:
        resolved_cfg: Flat resolved configuration dictionary.
        task: Task identifier ('train_imagenet' or 'train_coco').

    Returns:
        Populated ``TrainerConfig`` instance.
    """
    if task == "train_imagenet":
        return TrainerConfig(