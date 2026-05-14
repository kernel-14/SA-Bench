```python
## main.py
"""Entry point for the generative PDE foundation model pipeline.

Supports five operational modes:
  - preprocess:   Convert raw PDE datasets to unified c3p128 float16 HDF5 format
  - train_p2vae:  Stage 1 — train P2VAE autoencoder on all 16 sub-datasets
  - train_fmt:    Stage 2 — train FMT on frozen P2VAE latents
  - finetune:     Few-shot fine-tuning on Kolmogorov turbulence (REPA-E style)
  - evaluate:     Full evaluation suite reproducing Tables 1-3 and Figure 3

Multi-GPU DDP training is launched via torch.multiprocessing.spawn.
Single-GPU and CPU execution are also supported (world_size=1).

Usage examples:
    # Preprocess all datasets
    python main.py --mode preprocess --config config.yaml \
                   --data_dir /data/raw --output_dir /data/processed

    # Train P2VAE-16M on 4 GPUs
    python main.py --mode train_p2vae --config config.yaml \
                   --p2vae_variant p2vae_16m \
                   --data_dir /data/processed --output_dir /checkpoints

    # Train FMT-B on frozen P2VAE-16M
    python main.py --mode train_fmt --config config.yaml \
                   --fmt_variant fmt_b --p2vae_variant p2vae_16m \
                   --p2vae_ckpt /checkpoints/p2vae_final.pt \
                   --data_dir /data/processed --output_dir /checkpoints

    # Fine-tune on Kolmogorov turbulence
    python main.py --mode finetune --config config.yaml \
                   --fmt_variant fmt_b --p2vae_variant p2vae_16m \
                   --p2vae_ckpt /checkpoints/p2vae_final.pt \
                   --fmt_ckpt /checkpoints/fmt_final.pt \
                   --data_dir /data/processed --output_dir /checkpoints

    # Full evaluation
    python main.py --mode evaluate --config config.yaml \
                   --fmt_variant fmt_b --p2vae_variant p2vae_16m \
                   --p2vae_ckpt /checkpoints/p2vae_final.pt \
                   --fmt_ckpt /checkpoints/fmt_final.pt \
                   --data_dir /data/processed --output_dir /results
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# ---------------------------------------------------------------------------
# Project imports — all submodules are imported at the top level.
# main.py is the entry point and has no circular import risk.
# ---------------------------------------------------------------------------
from data.dataset import PDEUnifiedDataset
from data.preprocess import Preprocessor
from evaluation.evaluate import Evaluator
from models.fmt import FMT
from models.p2vae import P2VAE
from training.finetune import FinetuneTrainer
from training.train_fmt import FMTTrainer
from training.train_p2vae import P2VAETrainer
from utils.distributed import cleanup_ddp, is_main_process, setup_ddp

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# All 16 sub-dataset names in the order defined in config.yaml.
# Used to build dataset_paths lists and for per-dataset metric reporting.
ALL_DATASET_NAMES: List[str] = [
    "fno_v5",
    "fno_v4",
    "fno_v3",
    "pa_ns",
    "pa_nsc",
    "pa_swe",
    "pb_cns_low",
    "pb_cns_high",
    "pb_swe",
    "w_gs",
    "w_am",
    "w_swe",
    "w_rb",
    "w_sf",
    "w_tr",
    "w_ve",
]

# Datasets used for long-term rollout evaluation (Table 3).
ROLLOUT_EVAL_DATASETS: List[str] = ["pa_ns", "pb_cns_low", "pb_cns_high"]

# Kolmogorov turbulence dataset name (fine-tuning and Table 2).
KOLMOGOROV_DATASET_NAME: str = "kolmogorov_re222"


# ---------------------------------------------------------------------------
# Config loading and extraction helpers
# ---------------------------------------------------------------------------


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file into a nested Python dict.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Nested dictionary containing all configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ImportError: If PyYAML is not installed.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            "Ensure the path is correct and the file exists."
        )

    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load configuration files. "
            "Install it with: pip install pyyaml"
        ) from exc

    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    logger.info("Configuration loaded from: %s", config_path)
    return config


def extract_p2vae_config(
    config: Dict[str, Any],
    variant: str = "p2vae_16m",
) -> Dict[str, Any]:
    """Merge P2VAE base config, variant-specific config, and training config.

    Produces a flat dict suitable for passing to P2VAE.__init__ and
    P2VAETrainer.__init__. The variant-specific 'base_dim' overrides the
    base config's default.

    Args:
        config: Full configuration dict loaded from config.yaml.
        variant: P2VAE variant name. One of 'p2vae_16m', 'p2vae_87m'.
            From config.yaml: p2vae.variants.p2vae_16m.base_dim = 64,
            p2vae.variants.p2vae_87m.base_dim = 128.

    Returns:
        Flat configuration dict with keys:
            model_type, in_channels, out_channels, base_dim, latent_channels,
            latent_size, channel_multipliers, num_res_blocks, dropout,
            kl_weight, lr (=base_lr), batch_size, total_steps, warmup_ratio,
            weight_decay, beta1, beta2, min_lr_ratio, gradient_clip,
            num_gpus, precision.

    Raises:
        KeyError: If the variant is not found in config.p2vae.variants.
    """
    p2vae_base: Dict[str, Any] = config.get("p2vae", {})
    variants: Dict[str, Any] = p2vae_base.get("variants", {})

    if variant not in variants:
        raise KeyError(
            f"P2VAE variant '{variant}' not found in config.p2vae.variants. "
            f"Available variants: {list(variants.keys())}."
        )

    variant_cfg: Dict[str, Any] = variants[variant]
    training_cfg: Dict[str, Any] = p2vae_base.get("training", {})
    hardware_cfg: Dict[str, Any] = p2vae_base.get("hardware", {})

    # Build merged flat config dict.
    merged: Dict[str, Any] = {
        # Architecture (from p2vae base + variant override)
        "model_type": "p2vae",
        "in_channels": int(p2vae_base.get("in_channels", 3)),
        "out_channels": int(p2vae_base.get("out_channels", 3)),
        "base_dim": int(variant_cfg.get("base_dim", 64)),
        "latent_channels": int(p2vae_base.get("latent_channels", 16)),
        "latent_size": int(p2vae_base.get("latent_size", 16)),
        "channel_multipliers": list(
            p2vae_base.get("channel_multipliers", [1, 2, 4, 4])
        ),
        "num_res_blocks": int(p2vae_base.get("num_res_blocks", 2)),
        "dropout": float(p2vae_base.get("dropout", 0.0)),
        "kl_weight": float(p2vae_base.get("kl_weight", 1e-3)),
        # Training hyperparameters
        "lr": float(training_cfg.get("base_lr", 1e-4)),
        "base_lr": float(training_cfg.get("base_lr", 1e-4)),
        "base_batch_size": int(training_cfg.get("base_batch_size", 256)),
        "batch_size": int(training_cfg.get("base_batch_size", 256)),
        "total_steps": int(training_cfg.get("total_steps", 100000)),
        "warmup_ratio": float(training_cfg.get("warmup_ratio", 0.1)),
        "weight_decay": float(training_cfg.get("weight_decay", 1e-4)),
        "beta1": float(training_cfg.get("beta1", 0.9)),
        "beta2": float(training_cfg.get("beta2", 0.995)),
        "min_lr_ratio": float(training_cfg.get("min_lr_ratio", 0.0)),
        "gradient_clip": float(training_cfg.get("gradient_clip", 1.0)),
        # Hardware
        "num_gpus": int(hardware_cfg.get("num_gpus", 4)),
        "precision": str(hardware_cfg.get("precision", "float16")),
    }

    logger.debug(
        "Extracted P2VAE config for variant '%s': base_dim=%d, kl_weight=%.2e",
        variant,
        merged["base_dim"],
        merged["kl_weight"],
    )

    return merged


def extract_fmt_config(
    config: Dict[str, Any],
    variant: str = "fmt_b",
) -> Dict[str, Any]:
    """Merge FMT base config, variant-specific config, and training config.

    Produces a flat dict suitable for passing to FMT.__init__ and
    FMTTrainer.__init__. The variant-specific embed_dim, depth, and num_heads
    override the base config defaults.

    Args:
        config: Full configuration dict loaded from config.yaml.
        variant: FMT variant name. One of 'fmt_s', 'fmt_b', 'fmt_l'.
            From config.yaml:
                fmt_s: embed_dim=256, depth=6,  num_heads=4
                fmt_b: embed_dim=512, depth=12, num_heads=8
                fmt_l: embed_dim=768, depth=24, num_heads=12

    Returns:
        Flat configuration dict with keys:
            model_type, embed_dim, depth, num_heads, head_dim, mlp_ratio,
            latent_channels, latent_size, patch_size, lr (=base_lr),
            batch_size, total_steps, warmup_ratio, weight_decay, beta1,
            beta2, min_lr_ratio, gradient_clip, euler_steps, freeze_p2vae.

    Raises:
        KeyError: If the variant is not found in config.fmt.variants.
    """
    fmt_base: Dict[str, Any] = config.get("fmt", {})
    variants: Dict[str, Any] = fmt_base.get("variants", {})

    if variant not in variants:
        raise KeyError(
            f"FMT variant '{variant}' not found in config.fmt.variants. "
            f"Available variants: {list(variants.keys())}."
        )

    variant_cfg: Dict[str, Any] = variants[variant]
    training_cfg: Dict[str, Any] = fmt_base.get("training", {})
    hardware_cfg: Dict[str, Any] = fmt_base.get("hardware", {})
    inference_cfg: Dict[str, Any] = fmt_base.get("inference", {})
    pyramid_cfg: Dict[str, Any] = fmt_base.get("temporal_pyramid", {})

    merged: Dict[str, Any] = {
        # Architecture (from fmt base + variant override)
        "model_type": "fmt",
        "embed_dim": int(variant_cfg.get("embed_dim", 512)),
        "depth": int(variant_cfg.get("depth", 12)),
        "num_heads": int(variant_cfg.get("num_heads", 8)),
        "head_dim": int(fmt_base.get("head_dim", 64)),
        "mlp_ratio": float(fmt_base.get("mlp_ratio", 4.0)),
        "latent_channels": int(
            config.get("p2vae", {}).get("latent_channels", 16)
        ),
        "latent_size": int(
            config.get("p2vae", {}).get("latent_size", 16)
        ),
        "patch_size": int(fmt_base.get("patch_size", 1)),
        # Temporal pyramid
        "downsample_factors": list(
            pyramid_cfg.get("downsample_factors", [8, 4, 2, 1])
        ),
        "total_tokens": int(pyramid_cfg.get("total_tokens", 340)),
        # Training hyperparameters
        "lr": float(training_cfg.get("base_lr", 1e-4)),
        "base_lr": float(training_cfg.get("base_lr", 1e-4)),
        "base_batch_size": int(training_cfg.get("base_batch_size", 256)),
        "batch_size": int(training_cfg.get("base_batch_size", 256)),
        "total_steps": int(training_cfg.get("total_steps", 100000)),
        "warmup_ratio": float(training_cfg.get("warmup_ratio", 0.1)),
        "weight_decay": float(training_cfg.get("weight_decay", 0.01)),
        "beta1": float(training_cfg.get("beta1", 0.9)),
        "beta2": float(training_cfg.get("beta2", 0.95)),
        "min_lr_ratio": float(training_cfg.get("min_lr_ratio", 0.0)),
        "gradient_clip": float(training_cfg.get("gradient_clip", 1.0)),
        "freeze_p2vae": bool(training_cfg.get("freeze_p2vae", True)),
        # Inference
        "euler_steps": int(inference_cfg.get("euler_steps", 100)),
        "dt": float(inference_cfg.get("dt", 0.01)),
        # Hardware
        "num_gpus": int(hardware_cfg.get("num_gpus", 4)),
        "precision": str(hardware_cfg.get("precision", "float16")),
    }

    logger.debug(
        "Extracted FMT config for variant '%s': embed_dim=%d, depth=%d, "
        "num_heads=%d",
        variant,
        merged["embed_dim"],
        merged["depth"],
        merged["num_heads"],
    )

    return merged


def extract_finetune_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract fine-tuning configuration from the master config.

    Args:
        config: Full configuration dict loaded from config.yaml.

    Returns:
        Flat configuration dict for FinetuneTrainer with keys:
            total_steps, lambda_vae, stop_gradient_latent, base_lr,
            weight_decay, beta1, beta2, lr_schedule, warmup_ratio,
            min_lr_ratio, gradient_clip, train_trajectories,
            test_trajectories.
    """
    finetune_cfg: Dict[str, Any] = config.get("finetune", {})

    merged: Dict[str, Any] = {
        "total_steps": int(finetune_cfg.get("total_steps", 5000)),
        "lambda_vae": float(finetune_cfg.get("lambda_vae", 1.0)),
        "stop_gradient_latent": bool(
            finetune_cfg.get("stop_gradient_latent", True)
        ),
        "base_lr": float(finetune_cfg.get("base_lr", 1e-4)),
        "lr": float(finetune_cfg.get("base_lr", 1e-4)),
        "weight_decay": float(finetune_cfg.get("weight_decay", 0.01)),
        "beta1": float(finetune_cfg.get("beta1", 0.9)),
        "beta2": float(finetune_cfg.get("beta2", 0.95)),
        "lr_schedule": str(finetune_cfg.get("lr_schedule", "cosine")),
        "warmup_ratio": float(finetune_cfg.get("warmup_ratio", 0.1)),
        "min_lr_ratio": float(finetune_cfg.get("min_lr_ratio", 0.0)),
        "gradient_clip": float(finetune_cfg.get("gradient_clip", 1.0)),
        "train_trajectories": int(
            finetune_cfg.get("train_trajectories", 200)
        ),
        "test_trajectories": int(
            finetune_cfg.get("test_trajectories", 500)
        ),
    }

    return merged


def build_full_config_for_trainer(
    config: Dict[str, Any],
    model_config: Dict[str, Any],
    mode: str,
) -> Dict[str, Any]:
    """Build the full config dict expected by trainer __init__ methods.

    Trainers expect a config dict with top-level keys 'p2vae' or 'fmt',
    plus 'logging' and 'checkpointing'. This function assembles that
    structure from the master config and the extracted model config.

    Args:
        config: Full master configuration dict from config.yaml.
        model_config: Extracted flat model config (from extract_p2vae_config
            or extract_fmt_config).
        mode: Training mode ('train_p2vae', 'train_fmt', 'finetune').

    Returns:
        Config dict with the structure expected by trainer __init__ methods.
    """
    full_config: Dict[str, Any] = {
        "logging": config.get("logging", {}),
        "checkpointing": config.get("checkpointing", {}),
        "data": config.get("data", {}),
        "finetune": config.get("finetune", {}),
        "ensemble": config.get("ensemble", {}),
        "evaluation": config.get("evaluation", {}),
    }

    if mode in ("train_p2vae", "evaluate"):
        # Trainers access config['p2vae']['training'] internally.
        full_config["p2vae"] = config.get("p2vae", {})
        # Inject the resolved variant config into the training sub-dict.
        full_config["p2vae"]["training"] = {
            "base_lr": model_config.get("base_lr", 1e-4),
            "base_batch_size": model_config.get("base_batch_size", 256),
            "batch_size": model_config.get("batch_size", 256),
            "total_steps": model_config.get("total_steps", 100000),
            "warmup_ratio": model_config.get("warmup_ratio", 0.1),
            "weight_decay": model_config.get("weight_decay", 1e-4),
            "beta1": model_config.get("beta1", 0.9),
            "beta2": model_config.get("beta2", 0.995),
            "min_lr_ratio": model_config.get("min_lr_ratio", 0.0),
            "gradient_clip": model_config.get("gradient_clip", 1.0),
        }

    if mode in ("train_fmt", "evaluate"):
        full_config["fmt"] = config.get("fmt", {})
        full_config["fmt"]["training"] = {
            "base_lr": model_config.get("base_lr", 1e-4),
            "base_batch_size": model_config.get("base_batch_size", 256),
            "batch_size": model_config.get("batch_size", 256),
            "total_steps": model_config.get("total_steps", 100000),
            "warmup_ratio": model_config.get("warmup_ratio", 0.1),
            "weight_decay": model_config.get("weight_decay", 0.01),
            "beta1": model_config.get("beta1", 0.9),
            "beta2": model_config.get("beta2", 0.95),
            "min_lr_ratio": model_config.get("min_lr_ratio", 0.0),
            "gradient_clip": model_config.get("gradient_clip", 1.0),
            "freeze_p2vae": model_config.get("freeze_p2vae", True),
        }

    return full_config


# ---------------------------------------------------------------------------
# DataLoader construction helpers
# ---------------------------------------------------------------------------


def build_dataset_paths(data_dir: str, names: List[str]) -> List[str]:
    """Build HDF5 file paths for a list of dataset names.

    Args:
        data_dir: Root directory containing preprocessed HDF5 files.
        names: List of dataset names (e.g., ['fno_v5', 'pa_ns', ...]).

    Returns:
        List of full paths: [data_dir/name.h5 for name in names].
    """
    return [os.path.join(data_dir, f"{name}.h5") for name in names]


def filter_existing_datasets(
    paths: List[str],
    names: List[str],
) -> Tuple[List[str], List[str]]:
    """Filter out dataset paths that do not exist on disk.

    Logs a warning for each missing file. Returns only the paths and names
    for datasets that exist, allowing partial training when some datasets
    have not been preprocessed yet.

    Args:
        paths: List of HDF5 file paths.
        names: Corresponding list of dataset names.

    Returns:
        Tuple of (existing_paths, existing_names) with missing files removed.
    """
    existing_paths: List[str] = []
    existing_names: List[str] = []

    for path, name in zip(paths, names):
        if os.path.isfile(path):
            existing_paths.append(path)
            existing_names.append(name)
        else:
            logger.warning(
                "Dataset file not found (skipping): %s. "
                "Run --mode preprocess first to generate this dataset.",
                path,
            )

    if not existing_paths:
        raise FileNotFoundError(
            f"No dataset files found in {os.path.dirname(paths[0])}. "
            "Run --mode preprocess first to generate all datasets."
        )

    logger.info(
        "Found %d/%d dataset files. Missing: %s",
        len(existing_paths),
        len(paths),
        [n for n in names if n not in existing_names],
    )

    return existing_paths, existing_names


def build_dataloaders(
    data_dir: str,
    dataset_names: List[str],
    config: Dict[str, Any],
    rank: int,
    world_size: int,
    batch_size_per_gpu: int,
    seq_len: int = 4,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders for the unified PDE dataset.

    Creates PDEUnifiedDataset instances for train and val splits, wraps
    them with DistributedSampler for DDP training, and returns DataLoaders.

    Args:
        data_dir: Root directory containing preprocessed HDF5 files.
        dataset_names: List of dataset names to include.
        config: Full master configuration dict (for data settings).
        rank: DDP rank of the current process.
        world_size: Total number of DDP processes.
        batch_size_per_gpu: Per-GPU batch size (total_batch // world_size).
        seq_len: Trajectory sequence length (4 from config).
        num_workers: Number of DataLoader worker processes per GPU.

    Returns:
        Tuple of (train_loader, val_loader).

    Raises:
        FileNotFoundError: If no dataset files are found in data_dir.
    """
    # Build and filter dataset paths.
    all_paths: List[str] = build_dataset_paths(data_dir, dataset_names)
    existing_paths: List[str]
    existing_names: List[str]
    existing_paths, existing_names = filter_existing_datasets(
        all_paths, dataset_names
    )

    # Create train dataset with equal-probability sampling.
    train_dataset: PDEUnifiedDataset = PDEUnifiedDataset(
        dataset_paths=existing_paths,
        dataset_names=existing_names,
        split="train",
        seq_len=seq_len,
    )

    # Create validation dataset.
    val_dataset: PDEUnifiedDataset = PDEUnifiedDataset(
        dataset_paths=existing_paths,
        dataset_names=existing_names,
        split="val",
        seq_len=seq_len,
    )

    # DistributedSampler for DDP: partitions the virtual dataset across GPUs.
    # shuffle=True for training, shuffle=False for validation.
    if world_size > 1:
        train_sampler: DistributedSampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        val_sampler: Optional[DistributedSampler] = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        train_sampler = None  # type: ignore[assignment]
        val_sampler = None  # type: ignore[assignment]

    # Build DataLoaders.
    train_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # shuffle only when no sampler
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    val_loader: DataLoader = DataLoader(
        val_dataset,
        batch_size=batch_size_per_gpu,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    logger.info(
        "DataLoaders built: train=%d virtual samples, val=%d virtual samples, "
        "batch_size_per_gpu=%d, num_workers=%d",
        len(train_dataset),
        len(val_dataset),
        batch_size_per_gpu,
        num_workers,
    )

    return train_loader, val_loader


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = True,
) -> int:
    """Load model weights from a checkpoint file.

    Handles the 'module.' prefix that DDP adds to state dict keys when
    saving from a DDP-wrapped model. The checkpoint format follows the
    shared knowledge specification:
        {'model': state_dict, 'optimizer': ..., 'scheduler': ...,
         'scaler': ..., 'step': int, 'config': dict}

    Args:
        model: Model instance to load weights into.
        checkpoint_path: Path to the checkpoint file.
        device: Device to map the checkpoint tensors to.
        strict: Whether to strictly enforce that the keys in state_dict
            match the keys returned by model.state_dict(). Default True.

    Returns:
        The training step at which the checkpoint was saved (int).
        Returns 0 if the checkpoint does not contain a 'step' key.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}. "
            "Check the path or train from scratch."
        )

    logger.info("Loading checkpoint from: %s", checkpoint_path)

    checkpoint: Dict[str, Any] = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # Extract state dict from checkpoint.
    state_dict: Dict[str, Any] = checkpoint.get("model", checkpoint)

    # Strip 'module.' prefix added by DDP wrapping.
    # This allows loading a DDP-saved checkpoint into a non-DDP model.
    cleaned_state_dict: Dict[str, Any] = {}
    for key, value in state_dict.items():
        clean_key: str = key.replace("module.", "", 1)
        cleaned_state_dict[clean_key] = value

    # Load weights into the model.
    missing_keys: List[str]
    unexpected_keys: List[str]
    missing_keys, unexpected_keys = model.load_state_dict(
        cleaned_state_dict, strict=strict
    )

    if missing_keys:
        logger.warning(
            "Missing keys when loading checkpoint: %s", missing_keys
        )
    if unexpected_keys:
        logger.warning(
            "Unexpected keys when loading checkpoint: %s", unexpected_keys
        )

    step: int = int(checkpoint.get("step", 0))
    logger.info(
        "Checkpoint loaded successfully (step=%d, strict=%s).",
        step,
        strict,
    )

    return step


# ---------------------------------------------------------------------------
# Mode-specific handler functions
# ---------------------------------------------------------------------------


def run_preprocess(
    args: argparse.Namespace,
    config: Dict[str, Any],
) -> None:
    """Run dataset preprocessing for all 16 PDE sub-datasets.

    Converts raw PDE datasets from their native formats to unified
    c3p128 float16 HDF5 files. Only runs on rank 0 (or without DDP).

    The input data directory structure is expected to be:
        args.data_dir/
            fno/          # FNO Navier-Stokes .mat files
            pdearena/     # PDEArena NetCDF4 files
            pdebench/     # PDEBench HDF5 files
            the_well/     # The Well zarr/HDF5 files

    Output HDF5 files are written to args.output_dir.

    Args:
        args: Parsed command-line arguments.
        config: Full master configuration dict from config.yaml.
    """
    data