## main.py
"""Entry point for MoE‑POT experiments.

Usage examples:
    # Pre‑training from scratch
    python main.py --mode pretrain --model_size small
        --data_root /path/to/data --output_dir ./results

    # Fine‑tuning on a downstream task
    python main.py --mode finetune --pretrained_ckpt ./results/pretrain/best.pt
        --task_name NS-1e-4 --output_dir ./results

    # Downstream adaptation
    python main.py --mode downstream --pretrained_ckpt ./results/pretrain/best.pt
        --task_name PDEArena --output_dir ./results

    # Zero‑shot / saved model evaluation
    python main.py --mode eval --pretrained_ckpt ./results/pretrain/best.pt
        --eval_dataset SWE --output_dir ./results/eval_only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Union

import torch

from config import Config
from data_loader import DatasetLoader
from evaluation import Evaluation
from model import Model
from trainer import Trainer
from utils import Utils

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging(output_dir: Path, mode: str) -> None:
    """Configure root logger to write to both console and a file."""
    log_dir = output_dir / mode
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"

    # Set root logger level
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers (in case of previous runs)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_fmt)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Metrics saving
# ---------------------------------------------------------------------------
def save_metrics(
    metrics: Dict[str, float],
    mode: str,
    output_dir: Path,
    task_name: Optional[str] = None,
) -> None:
    """Write evaluation metrics to a JSON file.

    The file is placed inside ``output_dir/<mode>/[task_name/]``.
    """
    subdir = output_dir / mode
    if task_name:
        subdir = subdir / task_name
    subdir.mkdir(parents=True, exist_ok=True)

    path = subdir / "metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logging.info("Metrics saved to %s", path)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="MoE‑POT pre‑training / fine‑tuning / evaluation"
    )

    # Required mode
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["pretrain", "finetune", "downstream", "eval"],
        help="Operating mode",
    )

    # Model and data overrides
    parser.add_argument(
        "--model_size",
        type=str,
        default=None,
        choices=["tiny", "small", "medium"],
        help="Override model size from YAML (default: small)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to the root data directory (default from config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to the YAML configuration file",
    )

    # Checkpoint path (required for fine‑tune, downstream, eval)
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        default=None,
        help="Path to pre‑trained model checkpoint (.pt file)",
    )

    # Task‑specific arguments
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="Name of the dataset for fine‑tuning / downstream / single‑dataset eval",
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default=None,
        help="If set, evaluate only this dataset in eval mode; otherwise all validation sets",
    )

    # GPU selection
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device index (default: 0)",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    logging.info("Loading configuration from %s", args.config)
    cfg = Config.from_yaml(args.config)

    # Override from CLI arguments if provided
    if args.model_size is not None:
        cfg.model_size = args.model_size
        # After model_size change, __post_init__ will be called again
        # because it's a dataclass; we need to re‑trigger it.
        # Since Config is frozen=False, we can call __post_init__ manually.
        cfg.__post_init__()
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging early so that all subsequent messages are captured
    setup_logging(output_dir, args.mode)

    logging.info("Configuration:\n%s", cfg)

    # ------------------------------------------------------------------
    # 2. Reproducibility & device
    # ------------------------------------------------------------------
    Utils.set_seed(cfg.seed)

    gpu_idx = args.gpu
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_idx}")
        torch.cuda.set_device(device)
        logging.info("Using GPU: %s", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        logging.info("CUDA not available, using CPU")

    # ------------------------------------------------------------------
    # 3. Data preparation
    # ------------------------------------------------------------------
    logging.info("Initialising data loaders …")
    data_loader = DatasetLoader(cfg)

    # Determine loaders based on mode
    if args.mode == "pretrain":
        # Load full multi‑dataset train/val
        train_loader, val_loaders = data_loader.load_pretrain_data()
        train_single = False
        task_name = None
    elif args.mode in ("finetune", "downstream"):
        if args.task_name is None:
            raise ValueError("--task_name must be provided for fine‑tune / downstream mode")
        # Load single‑dataset loaders
        train_loader = data_loader.load_single_task(args.task_name, "train")
        val_loader = data_loader.load_single_task(args.task_name, "test")
        # Trainer expects a dict of validation loaders
        val_loaders = {args.task_name: val_loader}
        task_name = args.task_name
        train_single = True
    elif args.mode == "eval":
        # In eval mode we only need validation loaders;
        # training loader is irrelevant.
        if args.eval_dataset is not None:
            val_loader = data_loader.load_single_task(args.eval_dataset, "test")
            val_loaders = {args.eval_dataset: val_loader}
            logging.info("Evaluating only dataset: %s", args.eval_dataset)
        else:
            # Load all validation sets from pre‑training
            _, val_loaders = data_loader.load_pretrain_data()
            logging.info("Evaluating on all pre‑training validation sets")
        train_loader = None
        train_single = False
        task_name = None
    else:
        raise NotImplementedError(f"Unknown mode: {args.mode}")

    # ------------------------------------------------------------------
    # 4. Model instantiation
    # ------------------------------------------------------------------
    logging.info("Building MoE‑POT model (size: %s) …", cfg.model_size)
    model = Model(cfg)
    model = model.to(device)

    # If a checkpoint is provided, load it now
    if args.pretrained_ckpt is not None:
        ckpt = Path(args.pretrained_ckpt)
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.pretrained_ckpt}")
        logging.info("Loading model weights from %s", ckpt)
        state_dict = torch.load(ckpt, map_location=device)
        model.load_state_dict(state_dict, strict=True)
    elif args.mode != "pretrain":
        # All modes except pretrain require a checkpoint
        raise ValueError(
            f"--pretrained_ckpt is required for mode '{args.mode}'"
        )

    # Freeze router for fine‑tune and downstream (already during training,
    # but we also do it here for consistency)
    if args.mode in ("finetune", "downstream"):
        logging.info("Freezing router parameters …")
        model.freeze_router()
    else:
        # Ensure router is unfrozen for pre‑training
        model.unfreeze_router()

    # ------------------------------------------------------------------
    # 5. Training (if applicable)
    # ------------------------------------------------------------------
    if args.mode != "eval":
        trainer = Trainer(
            model=model,
            config=cfg,
            train_loader=train_loader,
            val_loaders=val_loaders,
        )

        if args.mode == "pretrain":
            logging.info("Starting pre‑training …")
            trainer.pretrain()
        elif args.mode == "finetune":
            logging.info("Starting fine‑tuning on %s …", task_name)
            trainer.finetune(task_name)
        elif args.mode == "downstream":
            logging.info("Starting downstream adaptation on %s …", task_name)
            trainer.downstream(task_name)
        else:
            raise RuntimeError("Unreachable")

        # Reload best checkpoint if available (for evaluation)
        if args.mode == "pretrain":
            best_ckpt_path = output_dir / "pretrain" / "best.pt"
        else:
            # fine‑tune / downstream saves inside subdirectories
            best_ckpt_path = output_dir / args.mode / task_name / "best.pt"
        if best_ckpt_path.exists():
            logging.info("Loading best checkpoint from %s", best_ckpt_path)
            state_dict = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(state_dict, strict=True)

    # ------------------------------------------------------------------
    # 6. Evaluation
    # ------------------------------------------------------------------
    logging.info("Evaluating model …")
    evaluator = Evaluation(model, val_loaders, cfg)
    # L2RE computation
    l2re_metrics = evaluator.compute_l2re(num_rollout_steps=cfg.rollout_steps)
    logging.info("L2 Relative Error (L2RE): %s", l2re_metrics)

    # Inference time (only if GPU available and we care)
    if device.type == "cuda":
        inf_time_ms = evaluator.compute_inference_time()
        logging.info("Single‑step inference time: %.2f ms", inf_time_ms)
        l2re_metrics["inference_time_ms"] = inf_time_ms

    # Save metrics to disk
    _mode_label = args.mode if args.mode != "eval" else "eval"
    save_metrics(
        l2re_metrics,
        mode=_mode_label,
        output_dir=output_dir,
        task_name=task_name if args.mode != "eval" else args.eval_dataset,
    )

    # Also save a copy of the configuration used
    cfg_copy_path = output_dir / _mode_label / "config.yaml"
    if task_name:
        cfg_copy_path = output_dir / _mode_label / task_name / "config.yaml"
    cfg_copy_path.parent.mkdir(parents=True, exist_ok=True)
    import yaml
    with open(cfg_copy_path, "w") as f:
        yaml.dump(cfg.__dict__, f, default_flow_style=False)
    logging.info("Experiment finished. Results saved in %s", output_dir)


if __name__ == "__main__":
    main()
