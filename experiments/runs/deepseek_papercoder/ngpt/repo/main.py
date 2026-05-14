#!/usr/bin/env python3
"""
main.py

Entry point for reproducing the NGPT paper experiments.  This script
orchestrates loading the configuration, setting up the distributed
environment, creating data loaders, building the model, and either
training or evaluating (or both) the baseline GPT / nGPT models.

Usage examples
--------------
Train nGPT with 0.5B model (1k context) from scratch:
    python main.py --config config.yaml --mode train

Resume training from a checkpoint:
    python main.py --config config.yaml --mode train --checkpoint ./checkpoints/ckpt_0100000.pt

Evaluate a trained model on downstream tasks:
    python main.py --config config.yaml --mode eval --checkpoint ./checkpoints/ckpt_0100000.pt

Distributed launch (single node, 8 GPUs):
    python -m torch.distributed.launch --nproc_per_node=8 main.py --config config.yaml --mode train
"""

import argparse
import logging
import os
import sys
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from config import Config
from utils import set_seed
from dataset_loader import DatasetLoader
from model import GPTModel
from trainer import Trainer
from evaluation import Evaluator


# ---------------------------------------------------------------------------
# Helper to override config fields via command‑line
# ---------------------------------------------------------------------------
def _override_config(config: Config, args: argparse.Namespace) -> None:
    """
    Override configuration attributes from command‑line arguments.
    Only simple attributes (no nested) are supported for brevity.
    """
    # Common direct overrides
    if hasattr(args, "lr") and args.lr is not None:
        config.optim.lr = float(args.lr)
    if hasattr(args, "batch_size") and args.batch_size is not None:
        config.training.batch_size = int(args.batch_size)
    if hasattr(args, "num_iters") and args.num_iters is not None:
        config.training.num_iters = int(args.num_iters)
    if hasattr(args, "max_seq_len") and args.max_seq_len is not None:
        config.model.max_seq_len = int(args.max_seq_len)
        # Recompute derived fields if necessary (d_k, rope_freqs already handle via buffer)
    if hasattr(args, "n_layers") and args.n_layers is not None:
        config.model.n_layers = int(args.n_layers)
    if hasattr(args, "d_model") and args.d_model is not None:
        config.model.d_model = int(args.d_model)
    if hasattr(args, "n_heads") and args.n_heads is not None:
        config.model.n_heads = int(args.n_heads)
    if hasattr(args, "weight_decay") and args.weight_decay is not None:
        config.optim.weight_decay = float(args.weight_decay)
    if hasattr(args, "warmup_steps") and args.warmup_steps is not None:
        config.optim.warmup_steps = int(args.warmup_steps)


# ---------------------------------------------------------------------------
# Distributed environment initialisation
# ---------------------------------------------------------------------------
def _init_distributed() -> tuple[bool, int, torch.device]:
    """
    Initialize the PyTorch distributed process group (if launched with
    torchrun or torch.distributed.launch).  Returns a tuple
    (is_distributed, local_rank, device).
    """
    # Detect whether we are in a distributed launch.
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        # Initialize the process group.
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return True, local_rank, device
    else:
        # Single‑GPU or CPU fallback.
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return False, local_rank, device


# ---------------------------------------------------------------------------
# Model loading with optional DDP wrapping
# ---------------------------------------------------------------------------
def _create_model(config: Config, local_rank: int, is_distributed: bool) -> nn.Module:
    """
    Instantiate the GPT/nGPT model, move it to the appropriate device,
    and optionally wrap it with DistributedDataParallel.
    """
    raw_model = GPTModel(config).to(local_rank if torch.cuda.is_available() else "cpu")
    if is_distributed:
        model = nn.parallel.DistributedDataParallel(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    else:
        model = raw_model
    return model


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce nGPT: Normalized Transformer with Representation Learning on the Hypersphere",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "train_eval"],
        help="Operation mode: train, evaluate, or both.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (for resuming training or evaluation).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config).",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (automatically provided by torch.distributed.launch).",
    )

    # Optional overrides for key hyperparameters (useful for quick grid searches)
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate.")
    parser.add_argument("--batch_size", type=int, default=None, help="Global batch size.")
    parser.add_argument("--num_iters", type=int, default=None, help="Number of training iterations.")
    parser.add_argument("--max_seq_len", type=int, default=None, help="Maximum sequence length.")
    parser.add_argument("--n_layers", type=int, default=None, help="Number of Transformer layers.")
    parser.add_argument("--d_model", type=int, default=None, help="Model dimension.")
    parser.add_argument("--n_heads", type=int, default=None, help="Number of attention heads.")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay (AdamW).")
    parser.add_argument("--warmup_steps", type=int, default=None, help="LR warmup steps.")

    # Logging / checkpoint directory overrides
    parser.add_argument("--log_dir", type=str, default=None, help="Override logging directory.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Override checkpoint directory.")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    config = Config.from_yaml(args.config)

    # Apply command‑line overrides
    _override_config(config, args)

    # Override logging/checkpoint dirs if provided
    if args.log_dir is not None:
        config.logging.log_dir = args.log_dir
    if args.checkpoint_dir is not None:
        config.logging.checkpoint_dir = args.checkpoint_dir

    # Set random seed (command‑line takes precedence over config)
    seed = args.seed if args.seed is not None else getattr(config.logging, "seed", 42)
    set_seed(seed)

    # ------------------------------------------------------------------
    # 2. Distributed environment
    # ------------------------------------------------------------------
    is_distributed, local_rank, device = _init_distributed()

    # Re‑set the seed after device initialisation to ensure reproducibility
    # (especially for CUDA operations).
    set_seed(seed)

    # ------------------------------------------------------------------
    # 3. Logging setup (only on rank 0)
    # ------------------------------------------------------------------
    rank = local_rank if is_distributed else 0
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(config.logging.log_dir, "train.log")),
                logging.StreamHandler(sys.stdout),
            ],
        )
        logging.info(f"Configuration:\n{config.to_dict()}")
    else:
        # Suppress logging on non‑master processes
        logging.basicConfig(level=logging.ERROR)

    # ------------------------------------------------------------------
    # 4. Data loaders
    # ------------------------------------------------------------------
    # DatasetLoader handles downloading, tokenization, caching.
    train_loader = DatasetLoader(config, split="train")
    val_loader = DatasetLoader(config, split="val")

    # We need the DataLoader objects for the Trainer.
    train_dl = train_loader.dataloader
    val_dl = val_loader.dataloader

    # ------------------------------------------------------------------
    # 5. Model creation (and optional DDP wrapping)
    # ------------------------------------------------------------------
    model = _create_model(config, local_rank, is_distributed)

    # ------------------------------------------------------------------
    # 6. Evaluator instance (shared if needed)
    # ------------------------------------------------------------------
    evaluator = None
    if "eval" in args.mode:
        # The Evaluator works with the raw (unwrapped) model.
        # If DDP is active, the raw model is model.module.
        raw_model = model.module if is_distributed else model
        evaluator = Evaluator(raw_model, config)
        # Put the evaluator's model into eval mode (it will be switched later if training).
        raw_model.eval()

    # ------------------------------------------------------------------
    # 7. Training (if requested)
    # ------------------------------------------------------------------
    if "train" in args.mode:
        # Create the Trainer. Note: the Trainer will create the optimizer and
        # scheduler using the (possibly DDP‑wrapped) model.parameters().
        trainer = Trainer(
            model=model,
            config=config,
            train_loader=train_dl,
            val_loader=val_dl,
            evaluator=evaluator,
        )

        # If a checkpoint is provided, load the model, optimizer, and scheduler states.
        if args.checkpoint is not None:
            resume_step = trainer.load_checkpoint(args.checkpoint)
            logging.info(f"Resumed from checkpoint at step {resume_step}")

        # Launch the training loop.
        trainer.train()

        # After training, the raw model may be needed for evaluation.
        # The Trainer already saved checkpoints; we can re‑obtain the raw model
        # for the evaluator if "train_eval" is chosen.
        if args.mode == "train_eval" and evaluator is None:
            raw_model = model.module if is_distributed else model
            evaluator = Evaluator(raw_model, config)

    # ------------------------------------------------------------------
    # 8. Final evaluation (eval only or after training)
    # ------------------------------------------------------------------
    if "eval" in args.mode:
        # If we trained first, we already have an evaluator; otherwise,
        # create a fresh one from the loaded checkpoint.
        if args.mode == "eval":
            # For pure eval mode, we need to load the checkpoint.
            if args.checkpoint is None:
                raise ValueError("--checkpoint is required for eval mode.")
            raw_model = model.module if is_distributed else model
            evaluator = Evaluator(raw_model, config)
            # Load the saved weights
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
            # The checkpoint contains 'model_state_dict' (from our trainer).
            raw_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            logging.info(f"Loaded weights from {args.checkpoint} for evaluation.")

        # Ensure model is in evaluation mode
        raw_model = model.module if is_distributed else model
        raw_model.eval()

        if evaluator is None:
            evaluator = Evaluator(raw_model, config)

        # Run downstream tasks
        metrics = evaluator.evaluate_downstream()
        logging.info(f"Downstream evaluation results: {metrics}")

        # Optionally save metrics to a file
        eval_metrics_path = os.path.join(config.logging.log_dir, "eval_metrics.json")
        import json
        with open(eval_metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logging.info(f"Metrics saved to {eval_metrics_path}")

    # ------------------------------------------------------------------
    # 9. Cleanup
    # ------------------------------------------------------------------
    if is_distributed:
        dist.destroy_process_group()

    logging.info("Done.")


if __name__ == "__main__":
    main()
