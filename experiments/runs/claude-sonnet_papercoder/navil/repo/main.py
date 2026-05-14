```python
## main.py
"""Entry point for NaViL: training, evaluation, and ablation experiments.

This module orchestrates the full NaViL pipeline across three execution modes:
- train:     Three-stage training (S1.1 → S1.2 → S2) with checkpoint support.
- evaluate:  Benchmark evaluation on all configured benchmarks.
- ablation:  Design-space experiments from Sections 3.2 and 3.3 of the paper.

Usage:
    # Single-GPU training
    python main.py --config configs/navil_2b.yaml --mode train

    # Distributed training (via accelerate)
    accelerate launch main.py --config configs/navil_2b.yaml --mode train

    # Resume from checkpoint
    python main.py --config configs/navil_2b.yaml --mode train \
        --checkpoint checkpoints/navil_2b/step_000070000 --stage s1_2

    # Evaluation
    python main.py --config configs/navil_2b.yaml --mode evaluate \
        --checkpoint checkpoints/navil_2b/step_000140000

    # Ablation experiments
    python main.py --config configs/navil_2b.yaml --mode ablation
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.utils.data as torch_data
from omegaconf import DictConfig, OmegaConf

from data.dataset import NaViLDataset
from data.multi_scale_packing import MultiScalePacking
from data.preprocessing import ImagePreprocessor
from evaluation.evaluator import BenchmarkEvaluator
from evaluation.metrics import MetricsCalculator
from model.connector import Connector
from model.moe_llm import MoELLM
from model.navil_model import NaViLModel
from model.special_tokens import SpecialTokens
from model.visual_encoder import VisualEncoder
from training.loss import NTPLoss
from training.scheduler import LRScheduler
from training.trainer import NaViLTrainer
from utils.checkpoint import CheckpointManager
from utils.logging_utils import AverageMeter, log_metrics, setup_logger


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the NaViL pipeline.

    Returns:
        Parsed argument namespace with all CLI options.
    """
    parser = argparse.ArgumentParser(
        description="NaViL: Native Multimodal Large Language Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g., configs/navil_2b.yaml).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "ablation"],
        required=True,
        help="Execution mode: 'train', 'evaluate', or 'ablation'.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to checkpoint directory to resume from. "
            "Required for 'evaluate' mode. Optional for 'train' mode."
        ),
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["s1_1", "s1_2", "s2"],
        default=None,
        help=(
            "Training stage to resume from when --checkpoint is provided. "
            "Only meaningful in 'train' mode."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Override the output directory from config. "
            "Sets checkpoint_dir and log_dir under this path."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set automatically by accelerate launch).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def validate_config(config: DictConfig, args: argparse.Namespace) -> None:
    """Validate configuration fields before starting expensive operations.

    Checks architectural constraints, argument consistency, and warns about
    missing data paths. Does not crash on missing paths (they may be remote).

    Args:
        config: OmegaConf DictConfig loaded from the YAML config file.
        args:   Parsed CLI arguments.

    Raises:
        ValueError: If any hard constraint is violated (e.g., head_dim not integer,
                    stage provided without checkpoint).
    """
    logger: logging.Logger = logging.getLogger("navil")

    # ------------------------------------------------------------------ #
    # Architectural constraints                                            #
    # ------------------------------------------------------------------ #
    ve_width: int = int(config.model.visual_encoder.width)
    ve_heads: int = int(config.model.visual_encoder.num_heads)
    if ve_width % ve_heads != 0:
        raise ValueError(
            f"Visual encoder width ({ve_width}) must be divisible by "
            f"num_heads ({ve_heads}). Got remainder {ve_width % ve_heads}. "
            f"For NaViL-2B: 1472 / 23 = 64 exactly."
        )

    llm_width: int = int(config.model.llm.width)
    llm_heads: int = int(config.model.llm.num_heads)
    if llm_width % llm_heads != 0:
        raise ValueError(
            f"LLM width ({llm_width}) must be divisible by "
            f"num_heads ({llm_heads}). Got remainder {llm_width % llm_heads}."
        )

    # ------------------------------------------------------------------ #
    # Argument consistency                                                 #
    # ------------------------------------------------------------------ #
    if args.stage is not None and args.checkpoint is None:
        logger.warning(
            "--stage='%s' provided without --checkpoint. "
            "The --stage argument is only meaningful when resuming from a "
            "checkpoint. It will be ignored.",
            args.stage,
        )

    if args.mode == "evaluate" and args.checkpoint is None:
        raise ValueError(
            "--checkpoint is required for 'evaluate' mode. "
            "Provide the path to a trained model checkpoint directory."
        )

    if args.checkpoint is not None and not Path(args.checkpoint).exists():
        raise ValueError(
            f"Checkpoint path does not exist: {args.checkpoint}. "
            "Verify the path passed to --checkpoint."
        )

    # ------------------------------------------------------------------ #
    # Data path warnings (non-fatal)                                       #
    # ------------------------------------------------------------------ #
    if args.mode in ("train", "ablation"):
        data_paths: DictConfig = config.data.paths
        for name, path in OmegaConf.to_container(data_paths, resolve=True).items():
            if path and not Path(str(path)).exists():
                logger.warning(
                    "Data path for '%s' does not exist: %s. "
                    "This may cause errors during training if this source is used.",
                    name,
                    path,
                )

    # ------------------------------------------------------------------ #
    # Log config summary                                                   #
    # ------------------------------------------------------------------ #
    logger.info(
        "Config validated. Model: %s, LLM: %s, "
        "Visual encoder: depth=%d width=%d heads=%d, "
        "Connector: pixel_shuffle_factor=%d",
        config.model.model_size,
        config.model.llm.name_or_path,
        config.model.visual_encoder.depth,
        config.model.visual_encoder.width,
        config.model.visual_encoder.num_heads,
        config.model.connector.pixel_shuffle_factor,
    )


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(config: DictConfig) -> NaViLModel:
    """Instantiate all model components and assemble NaViLModel.

    Follows the strict dependency order:
    1. SpecialTokens (no dependencies)
    2. VisualEncoder (depends on RoPE2D internally)
    3. Connector (depends on visual_dim and llm_dim)
    4. MoELLM.from_pretrained (loads base LLM and replaces layers with MoE)
    5. NaViLModel (assembles all components, registers special tokens,
       resizes embeddings)

    Args:
        config: OmegaConf DictConfig with model architecture hyperparameters.

    Returns:
        Fully assembled NaViLModel instance ready for training or inference.
    """
    logger: logging.Logger = logging.getLogger("navil")
    logger.info("Building NaViLModel from config...")

    # NaViLModel.__init__ handles all sub-component construction internally.
    # It reads from config.model.{visual_encoder, connector, llm} and
    # config.inference.patch_size.
    model: NaViLModel = NaViLModel(config)

    # Log parameter counts
    total_params: int = sum(p.numel() for p in model.parameters())
    trainable_params: int = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(
        "NaViLModel built: total_params=%.2fB, trainable_params=%.2fB",
        total_params / 1e9,
        trainable_params / 1e9,
    )

    return model


# ---------------------------------------------------------------------------
# Data source resolution
# ---------------------------------------------------------------------------

def build_data_sources(
    config: DictConfig,
    stage: str,
) -> List[Dict[str, Any]]:
    """Resolve data source configurations for a given training stage.

    Reads the stage-specific source list from config and resolves file paths
    from config.data.paths. Returns a list of source dicts ready for
    NaViLDataset construction.

    Args:
        config: OmegaConf DictConfig with training and data sections.
        stage:  Training stage: "s1_1", "s1_2", or "s2".

    Returns:
        List of source configuration dicts, each containing:
        - "name":          Source identifier string.
        - "type":          "webdataset" or "jsonl".
        - "path":          Resolved file path or glob pattern.
        - "weight":        Sampling probability (float).
        - "caption_field": Caption field name (for webdataset sources).
    """
    logger: logging.Logger = logging.getLogger("navil")

    stage_cfg: DictConfig = getattr(config.training, stage)
    raw_sources: List[Any] = OmegaConf.to_container(
        stage_cfg.data.sources, resolve=True
    )
    data_paths: Dict[str, str] = OmegaConf.to_container(
        config.data.paths, resolve=True
    )

    resolved_sources: List[Dict[str, Any]] = []

    for raw_source in raw_sources:
        source_name: str = str(raw_source.get("name", "unknown"))
        source_type: str = str(raw_source.get("type", "webdataset"))
        source_weight: float = float(raw_source.get("weight", 1.0))
        caption_field: str = str(raw_source.get("caption_field", "caption"))

        # Resolve path from config.data.paths using source name as key
        resolved_path: str = str(data_paths.get(source_name, ""))

        if not resolved_path:
            logger.warning(
                "No path found for data source '%s' in config.data.paths. "
                "This source will yield no samples.",
                source_name,
            )

        resolved_source: Dict[str, Any] = {
            "name": source_name,
            "type": source_type,
            "path": resolved_path,
            "weight": source_weight,
            "caption_field": caption_field,
        }
        resolved_sources.append(resolved_source)

        logger.debug(
            "Data source '%s': type=%s, path=%s, weight=%.3f",
            source_name,
            source_type,
            resolved_path,
            source_weight,
        )

    return resolved_sources


# ---------------------------------------------------------------------------
# Data pipeline construction
# ---------------------------------------------------------------------------

def build_data_pipeline(
    config: DictConfig,
    tokenizer: Any,
    special_tokens: SpecialTokens,
    stage: str,
) -> torch_data.DataLoader:
    """Build the full data pipeline for a given training stage.

    Constructs MultiScalePacking → ImagePreprocessor → NaViLDataset →
    DataLoader in sequence, using stage-specific hyperparameters.

    Args:
        config:         OmegaConf DictConfig.
        tokenizer:      HuggingFace tokenizer (already extended with special tokens).
        special_tokens: SpecialTokens instance with populated token_ids.
        stage:          Training stage: "s1_1", "s1_2", or "s2".

    Returns:
        Configured PyTorch DataLoader for the given stage.
    """
    logger: logging.Logger = logging.getLogger("navil")

    stage_cfg: DictConfig = getattr(config.training, stage)
    max_patches: int = int(stage_cfg.max_image_patches)
    batch_size: int = int(stage_cfg.global_batch_size)
    visual_multiscale: bool = bool(stage_cfg.visual_multiscale_packing)
    num_workers: int = int(config.data.num_workers)
    pin_memory: bool = bool(config.data.pin_memory)

    tau: float = float(config.inference.tau)
    min_area_threshold: int = int(config.inference.min_area_threshold)
    patch_size: int = int(config.inference.patch_size)
    max_seq_length: int = int(config.inference.llm_max_seq_length)
    pixel_shuffle_factor: int = int(config.model.connector.pixel_shuffle_factor)

    logger.info(
        "Building data pipeline for stage '%s': "
        "max_patches=%d, batch_size=%d, visual_multiscale=%s",
        stage,
        max_patches,
        batch_size,
        visual_multiscale,
    )

    # ------------------------------------------------------------------ #
    # MultiScalePacking                                                    #
    # ------------------------------------------------------------------ #
    # For NaViL-9B S1.1, visual_multiscale_packing=False.
    # When disabled, use tau=1.0 (no downsampling) to produce single-scale.
    effective_tau: float = tau if visual_multiscale else 1.0

    multi_scale: MultiScalePacking = MultiScalePacking(
        tau=effective_tau,
        min_area_threshold=min_area_threshold,
        patch_size=patch_size,
        max_patches=max_patches,
        special_tokens=special_tokens,
    )

    # ------------------------------------------------------------------ #
    # ImagePreprocessor                                                    #
    # ------------------------------------------------------------------ #
    preprocessor: ImagePreprocessor = ImagePreprocessor(
        patch_size=patch_size,
        max_patches=max_patches,
        multi_scale=multi_scale,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )

    # ------------------------------------------------------------------ #
    # Data sources                                                         #
    # ------------------------------------------------------------------ #
    data_sources: List[Dict[str, Any]] = build_data_sources(config, stage)

    # ------------------------------------------------------------------ #
    # NaViLDataset                                                         #
    # ------------------------------------------------------------------ #
    dataset: NaViLDataset = NaViLDataset(
        data_sources=data_sources,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        max_seq_length=max_seq_length,
        stage=stage,
        pixel_shuffle_factor=pixel_shuffle_factor,
    )

    # ------------------------------------------------------------------ #
    # DataLoader                                                           #
    # ------------------------------------------------------------------ #
    dataloader: torch_data.DataLoader = dataset.build_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    logger.info(
        "Data pipeline for stage '%s' built successfully.",
        stage,
    )

    return dataloader


# ---------------------------------------------------------------------------
# LLM path resolution for ablation experiments
# ---------------------------------------------------------------------------

def get_llm_path_for_size(size_str: str, config: DictConfig) -> str:
    """Map an LLM size string to a HuggingFace model path.

    Args:
        size_str: Size string like "0.5B", "1.8B", "7B".
        config:   OmegaConf DictConfig (used to get the base LLM path).

    Returns:
        HuggingFace model path string.
    """
    # Mapping from size string to HuggingFace model identifiers
    _SIZE_TO_PATH: Dict[str, str] = {
        "0.5B": "internlm/internlm2-0_5b",
        "1.8B": "internlm/internlm2-1_8b",
        "7B": "internlm/internlm2-7b",
        "8B": "Qwen/Qwen3-8B",
    }

    # Use the base LLM path from config as the default for the configured size
    base_path: str = str(config.model.llm.name_or_path)

    return _SIZE_TO_PATH.get(size_str, base_path)


# ---------------------------------------------------------------------------
# Encoder config resolution for ablation experiments
# ---------------------------------------------------------------------------

def get_encoder_config_for_size(size_str: str) -> Tuple[int, int]:
    """Map an encoder size string to (depth, width) hyperparameters.

    Uses the approximation N ≈ 12 * d * w^2 to select configurations
    that achieve approximately the target parameter count.

    Args:
        size_str: Size string like "75M", "150M", "300M", "600M", "1.2B", "2.4B".

    Returns:
        Tuple (depth, width) for the VisualEncoder constructor.
    """
    # Lookup table: (depth, width) pairs that achieve approximately the target size
    # Computed via: N ≈ 12 * d * w^2
    _SIZE_TO_CONFIG: Dict[str, Tuple[int, int]] = {
        "75M":  (6, 1024),    # 12 * 6 * 1024^2 ≈ 75.5M
        "150M": (12, 1024),   # 12 * 12 * 1024^2 ≈ 151M
        "300M": (12, 1472),   # 12 * 12 * 1472^2 ≈ 312M ≈ 300M
        "600M": (24, 1472),   # 12 * 24 * 1472^2 ≈ 624M ≈ 600M
        "1.2B": (32, 1792),   # 12 * 32 * 1792^2 ≈ 1.23B ≈ 1.2B
        "2.4B": (48, 2048),   # 12 * 48 * 2048^2 ≈ 2.41B ≈ 2.4B
    }

    if size_str not in _SIZE_TO_CONFIG:
        # Fallback: use 600M config
        return (24, 1472)

    return _SIZE_TO_CONFIG[size_str]


# ---------------------------------------------------------------------------
# Ablation model construction
# ---------------------------------------------------------------------------

def build_ablation_model(
    config: DictConfig,
    encoder_depth: int,
    encoder_width: int,
    llm_path: str,
    use_pretrained_llm: bool = True,
    use_moe: bool = True,
) -> NaViLModel:
    """Build a model variant for ablation experiments.

    Constructs a NaViLModel with the specified encoder architecture and LLM,
    with optional pretrained initialization and MoE disabling.

    Args:
        config:              Base OmegaConf DictConfig.
        encoder_depth:       Visual encoder depth (number of transformer layers).
        encoder_width:       Visual encoder hidden dimension.
        llm_path:            HuggingFace model path for the base LLM.
        use_pretrained_llm:  If True, load pretrained LLM weights.
                             If False, use random initialization.
        use_moe:             If True, use MoE-extended LLM (standard NaViL).
                             If False, use vanilla LLM (no modality-specific experts).

    Returns:
        NaViLModel instance configured for the ablation condition.
    """
    logger: logging.Logger = logging.getLogger("navil")

    # Create a modified config for this ablation condition
    # OmegaConf.to_container + OmegaConf.create allows deep copy + modification
    config_dict: Dict[str, Any] = OmegaConf.to_container(config, resolve=True)

    # Override encoder architecture
    config_dict["model"]["visual_encoder"]["depth"] = encoder_depth
    config_dict["model"]["visual_encoder"]["width"] = encoder_width
    # Compute mlp_width as 4x width (standard ratio)
    config_dict["model"]["visual_encoder"]["mlp_width"] = encoder_width * 4
    # Compute num_heads: target head_dim=64
    num_heads: int = max(1, encoder_width // 64)
    # Ensure width is divisible by num_heads
    while encoder_width % num_heads != 0 and num_heads > 1:
        num_heads -= 1
    config_dict["model"]["visual_encoder"]["num_heads"] = num_heads

    # Override LLM path
    config_dict["model"]["llm"]["name_or_path"] = llm_path

    ablation_config: DictConfig = OmegaConf.create(config_dict)

    logger.info(
        "Building ablation model: encoder(depth=%d, width=%d, heads=%d), "
        "llm=%s, pretrained=%s, moe=%s",
        encoder_depth,
        encoder_width,
        num_heads,
        llm_path,
        use_pretrained_llm,
        use_moe,
    )

    # Build the model using the modified config
    # NaViLModel.__init__ handles all sub-component construction
    try:
        model: NaViLModel = NaViLModel(ablation_config)

        # If not using pretrained LLM, reinitialize LLM weights randomly
        if not use_pretrained_llm:
            logger.info("Reinitializing LLM weights randomly (from-scratch condition).")
            for name, param in model.llm.named_parameters():
                if param.dim() >= 2:
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)
                else:
                    torch.nn.init.zeros_(param)

        return model

    except Exception as exc:
        logger.error(
            "Failed to build ablation model (encoder_depth=%d, encoder_width=%d, "
            "llm=%s): %s",
            encoder_depth,
            encoder_width,
            llm_path,
            exc,
        )
        raise


# ---------------------------------------------------------------------------
# Ablation training helper
# ---------------------------------------------------------------------------

def train_ablation_model(
    model: NaViLModel,
    train_dataloader: torch_data.DataLoader,
    val_dataloader: Optional[torch_data.DataLoader],
    config: DictConfig,
    steps: int = 10000,
    log_interval: int = 1000,
) -> List[Dict[str, float]]:
    """Train a model for a fixed number of steps and record validation loss.

    Used for all ablation experiments. Trains with a simplified single-stage
    setup (no parameter freezing, constant LR with warmup).

    Args:
        model:            NaViLModel to train.
        train_dataloader: DataLoader for training data.
        val_dataloader:   Optional DataLoader for validation loss measurement.
                          If None, only training loss is recorded.
        config:           OmegaConf DictConfig for optimizer settings.
        steps:            Number of training steps. Default: 10000.
        log_interval:     Steps between validation loss measurements. Default: 1000.

    Returns:
        List of dicts, each containing:
        - "step":       Training step number.
        - "train_loss": Average training loss over the last log_interval steps.
        - "val_loss":   Validation loss (if val_dataloader provided), else None.
    """
    logger: logging.Logger = logging.getLogger("navil")

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.train()

    # All parameters trainable for ablation
    for param in model.parameters():
        param.requires_grad_(True)

    # Optimizer
    beta1: float = float(config.training.optimizer.beta1)
    beta2: float = float(config.training.optimizer.beta2)
    eps: float = float(config.training.optimizer.eps)
    peak_lr: float = 5e-5  # fixed for ablation

    optimizer: torch.optim.AdamW = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=0.05,
    )

    # LR scheduler: constant warmup for ablation
    scheduler: LRScheduler = LRScheduler(
        optimizer=optimizer,
        schedule_type="constant_warmup",
        peak_lr=peak_lr,
        warmup_steps=200,
        total_steps=steps,
        min_lr=0.0,
    )

    # Loss function
    loss_fn: NTPLoss = NTPLoss(ignore_index=-100)

    # Training loop
    import itertools
    data_iter = itertools.cycle(train_dataloader)

    loss_meter: AverageMeter = AverageMeter()
    loss_curve: List[Dict[str, float]] = []

    for step in range(steps):
        try:
            batch: Dict[str, Any] = next(data_iter)
        except StopIteration:
            break

        # Move batch to device
        input_ids: torch.Tensor = batch["input_ids"].to(device)
        labels: torch.Tensor = batch["labels"].to(device)
        attention_mask: torch.Tensor = batch["attention_mask"].to(device)
        modality_mask: torch.Tensor = batch["modality_mask"].to(device)
        pixel_values: Optional[List[Any]] = batch.get("pixel_values", None)
        grid_sizes: Optional[List[Any]] = batch.get("grid_sizes", None)

        optimizer.zero_grad()

        try:
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                modality_mask=modality_mask,
                attention_mask=attention_mask,
                labels=None,
            )

            if hasattr(outputs, "logits"):
                logits: torch.Tensor = outputs.logits
            elif isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                continue

            loss: torch.Tensor = loss_fn(logits, labels)

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                loss_meter.update(loss.item())

        except Exception as exc:
            logger.debug("Ablation train step %d failed: %s", step, exc)
            continue

        # Record at log_interval
        if (step + 1) % log_interval == 0:
            val_loss: Optional[float] = None

            if val_dataloader is not None:
                val_loss = compute_validation_loss(
                    model, val_dataloader, loss_fn, device, max_batches=50
                )

            record: Dict[str, float] = {
                "step": step + 1,
                "train_loss": loss_meter.avg,
                "val_loss": val_loss if val_loss is not None else float("nan"),
            }
            loss_curve.append(record)

            logger.info(
                "Ablation step %d/%d: train_loss=%.4f, val_loss=%s",
                step + 1,
                steps,
                loss_meter.avg,
                f"{val_loss:.4f}" if val_loss is not None else "N/A",
            )
            loss_meter.reset()

    model.train(False)
    return loss_curve


def compute_validation_loss(
    model: NaViLModel,
    val_dataloader: torch_data.DataLoader,
    loss_fn: NTPLoss,
    device: str,
    max_batches: int = 50,
) -> float:
    """Compute average validation loss over a fixed number of batches.

    Args:
        model:          NaViLModel in eval mode.
        val_dataloader: Validation DataLoader.
        loss_fn:        NTPLoss instance.
        device:         Device string.
        max_batches:    Maximum number of batches to evaluate. Default: 50.

    Returns:
        Average validation loss as a float.
    """
    model.eval()
    total_loss: float = 0.0
    num_batches: int = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            if batch_idx >= max_batches:
                break

            try:
                input_ids: torch.Tensor = batch["input_ids"].to(device)
                labels: torch.Tensor = batch["labels"].to(device)
                attention_mask: torch.Tensor = batch["attention_mask"].to(device)
                modality_mask: torch.Tensor = batch["modality_mask"].to(device)
                pixel_values: Optional[List[Any]] = batch.get("pixel_values", None)

                outputs = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    modality_mask=modality_mask,
                    attention_mask=attention_mask,
                    labels=None,
                )

                if hasattr(outputs, "logits"):
                    logits: torch.Tensor = outputs.logits
                elif isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    continue

                loss: torch.Tensor = loss_fn(logits, labels)
                if torch.isfinite(loss):
                    total_loss += loss.item()
                    num_batches += 1

            except Exception:
                continue

    model.train()

    if num_batches == 0:
        return float("nan")

    return total_loss / num_batches


# ---------------------------------------------------------------------------
# Ablation validation dataloader
# ---------------------------------------------------------------------------

def build_validation_dataloader(
    config: DictConfig,
    special_tokens: SpecialTokens,
    tokenizer: Any,
    max_samples: int = 5000,
) -> Optional[torch_data.DataLoader]:
    """Build a small held-out validation dataloader for ablation experiments.

    Uses a small subset of the S1.1 web-scale data as the validation set.
    Returns None if no valid data paths are available.

    Args:
        config:         OmegaConf DictConfig.
        special_tokens: SpecialTokens instance with populated token_ids.