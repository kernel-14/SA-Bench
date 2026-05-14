# main.py
"""
Entry point for SAM 2 reproduction pipeline.

Orchestrates data loading, model building, training (pre‑training and full
joint training), and evaluation according to the YAML configuration
(``config.yaml``).  Supports incremental execution via command‑line flags.

Usage::

    python main.py --stage all          # run all stages
    python main.py --stage pretrain     # only pre‑training
    python main.py --stage evaluate --eval_type interactive_offline

The pipeline strictly follows the experimental protocol described in the
SAM 2 paper (Sections 4‑7 and Appendices D‑F).
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import os
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch

# Project imports (assume they are installed or on PYTHONPATH)
from config import Config
from data.video_dataset import VideoDataset
from data.image_dataset import ImageDataset
from model.sam2 import SAM2Model
from training.trainer import Trainer
from evaluation.evaluator import Evaluator

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sam2.main")

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments.

    Returns:
        Namespace with ``config``, ``stage``, ``eval_type``, ``checkpoint``.
    """
    parser = argparse.ArgumentParser(
        description="SAM 2 – Segment Anything in Images and Videos (reproduction)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "pretrain", "train", "evaluate", "pretrain+train"],
        help="Which phase(s) to execute.",
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        default="all",
        choices=["all", "interactive_offline", "interactive_online",
                 "semi_supervised", "image_segmentation"],
        help="Type of evaluation to run (ignored unless stage includes 'evaluate').",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a saved checkpoint (model_state_dict) for evaluation or fine‑tuning.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class Main:
    """
    Top‑level pipeline orchestrator.

    Builds datasets, model, trainer, and evaluator from a :class:`Config`
    object and provides a single :meth:`run_pipeline` method that executes
    the requested phases.

    Args:
        config: fully‑parsed configuration (AttrDict) as returned by
            ``Config("config.yaml")``.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

        # ------------------------------------------------------------------
        #  Device selection
        # ------------------------------------------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # ------------------------------------------------------------------
        #  Reproducibility
        # ------------------------------------------------------------------
        seed = getattr(config, "seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # ------------------------------------------------------------------
        #  Datasets
        # ------------------------------------------------------------------
        self._build_datasets()

        # ------------------------------------------------------------------
        #  Model
        # ------------------------------------------------------------------
        self.model = SAM2Model(config._cfg)   # pass the underlying AttrDict
        logger.info("SAM 2 model built.")

        # ------------------------------------------------------------------
        #  Trainer (only when training stages are selected)
        # ------------------------------------------------------------------
        self.trainer = None
        self.evaluator = None

    def _build_datasets(self) -> None:
        """
        Collect all video roots and build the shared :class:`VideoDataset` and
        :class:`ImageDataset` objects.

        The internal dataset root is omitted if ``None``.
        """
        data_cfg = self.config.data

        # --- Video datasets ---
        video_roots: Dict[str, str] = {}
        for ds_name, root_key in [
            ("davis", "davis_root"),
            ("mose", "mose_root"),
            ("ytvos", "ytv_root"),
            ("sav", "sav_root"),
        ]:
            root = data_cfg.get(root_key)
            if root is not None:
                video_roots[ds_name] = root

        if not video_roots:
            logger.warning(
                "No video dataset roots provided – only image training is possible."
            )
            self.video_dataset = None
        else:
            self.video_dataset = VideoDataset(
                root_paths=video_roots,
                config=self.config,
                split="train",
                augment=True,
            )
            logger.info(
                f"Video dataset built with {len(self.video_dataset)} clips "
                f"from {list(video_roots.keys())}."
            )

        # --- Image dataset (SA‑1B) ---
        if data_cfg.sa1b_root is not None:
            self.image_dataset = ImageDataset(
                root=data_cfg.sa1b_root,
                config=self.config,
                train=True,
            )
            logger.info(f"Image dataset built with {len(self.image_dataset)} images.")
        else:
            self.image_dataset = None
            if data_cfg.internal_root is not None:
                logger.warning(
                    "SA‑1B root not set; image training will be skipped. "
                    "Internal dataset is also unavailable."
                )

    # ------------------------------------------------------------------
    #  Pipeline execution
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        stage: str,
        eval_type: str = "all",
        checkpoint_path: Optional[str] = None,
    ) -> None:
        """
        Execute the requested stage(s) in the correct order.

        Args:
            stage: one of ``"all"``, ``"pretrain"``, ``"train"``, ``"evaluate"``,
                ``"pretrain+train"``.
            eval_type: evaluation sub‑type (ignored unless ``stage`` includes
                ``"evaluate"``).
            checkpoint_path: optional path to a checkpoint file (``*.pth``) to
                load before training/evaluation.
        """
        # ------------------------------------------------------------------
        #  Load checkpoint if provided
        # ------------------------------------------------------------------
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            if "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
                logger.info(f"Loaded model weights from {checkpoint_path}")
            else:
                logger.warning(
                    f"Checkpoint {checkpoint_path} does not contain "
                    "'model_state_dict'; ignoring."
                )

        # ------------------------------------------------------------------
        #  Pre‑training on SA‑1B (image only)
        # ------------------------------------------------------------------
        if stage in ("all", "pretrain", "pretrain+train"):
            if self.image_dataset is None:
                raise RuntimeError(
                    "Image dataset (SA‑1B) is required for pre‑training, "
                    "but sa1b_root is not set in config."
                )
            if self.trainer is None:
                self.trainer = Trainer(model=self.model, config=self.config)
            logger.info("===== Starting pre‑training on SA‑1B =====")
            self.trainer.pretrain(self.image_dataset)
            torch.save(
                {"model_state_dict": self.model.state_dict()},
                "pretrained_sam2.pth",
            )
            logger.info("Pre‑trained model saved to pretrained_sam2.pth")

        # ------------------------------------------------------------------
        #  Full training (image + video)
        # ------------------------------------------------------------------
        if stage in ("all", "train", "pretrain+train"):
            if self.video_dataset is None or self.image_dataset is None:
                raise RuntimeError(
                    "Both video and image datasets are required for full training."
                )
            if self.trainer is None:
                self.trainer = Trainer(model=self.model, config=self.config)
            logger.info("===== Starting full joint training =====")
            self.trainer.train_full(self.video_dataset, self.image_dataset)
            torch.save(
                {"model_state_dict": self.model.state_dict()},
                "sam2_full.pth",
            )
            logger.info("Fully trained model saved to sam2_full.pth")

        # ------------------------------------------------------------------
        #  Evaluation
        # ------------------------------------------------------------------
        if stage in ("all", "evaluate") or stage == "evaluate":
            self._run_evaluation(eval_type)

    # ------------------------------------------------------------------
    #  Evaluation helper
    # ------------------------------------------------------------------

    def _run_evaluation(self, eval_type: str) -> None:
        """
        Execute the requested evaluation sub‑protocol(s).

        Args:
            eval_type: one of ``"all"``, ``"interactive_offline"``,
                ``"interactive_online"``, ``"semi_supervised"``,
                ``"image_segmentation"``.
        """
        if self.trainer is not None:
            # Ensure model is in eval mode and on the correct device.
            self.model.eval()
            self.model.to(self.device)

        # The Evaluator expects the raw configuration (AttrDict)
        self.evaluator = Evaluator(
            model=self.model,
            config=self.config._cfg,   # underlying AttrDict (dict‑compatible)
            device=str(self.device),
        )

        if eval_type == "all":
            logger.info("===== Running full evaluation suite =====")
            results = self.evaluator.evaluate_all()
            self._print_results(results)
        else:
            eval_cfg = self.config.evaluation
            if eval_type == "interactive_offline":
                ds_list = eval_cfg.interactive_offline.datasets
                self._evaluate_single(
                    ds_list, self.evaluator.evaluate_interactive_offline
                )
            elif eval_type == "interactive_online":
                ds_list = eval_cfg.interactive_online.datasets
                self._evaluate_single(
                    ds_list, self.evaluator.evaluate_interactive_online
                )
            elif eval_type == "semi_supervised":
                ds_list = eval_cfg.semi_supervised.datasets
                self._evaluate_single(
                    ds_list, self.evaluator.evaluate_semi_supervised
                )
            elif eval_type == "image_segmentation":
                img_datasets = eval_cfg.image_segmentation.datasets
                if isinstance(img_datasets, str):
                    # Placeholder – full list should be provided in config
                    img_datasets = ["ADE20K", "Cityscapes"]
                self._evaluate_single(
                    img_datasets, self.evaluator.evaluate_image_segmentation
                )
            else:
                raise ValueError(f"Unknown eval_type: {eval_type}")

    @staticmethod
    def _evaluate_single(datasets: List[str], func) -> None:
        """
        Iterate over a list of dataset names, call the evaluation function,
        and print the results.  Catches :class:`ValueError` when a dataset
        is not found and continues.
        """
        for ds in datasets:
            try:
                logger.info(f"Evaluating on {ds} ...")
                res = func(ds)
                logger.info(f"Result for {ds}: {res}")
            except (ValueError, FileNotFoundError) as e:
                logger.warning(f"Skipping dataset '{ds}': {e}")

    @staticmethod
    def _print_results(results: Dict[str, Any]) -> None:
        """Pretty‑print the evaluation results dictionary."""
        import json
        logger.info("Evaluation results:\n" + json.dumps(results, indent=2))


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    # 1. Load configuration
    if not os.path.isfile(args.config):
        sys.exit(f"Configuration file not found: {args.config}")
    cfg = Config(args.config)

    # 2. Instantiate pipeline and run
    main = Main(cfg)
    main.run_pipeline(
        stage=args.stage,
        eval_type=args.eval_type,
        checkpoint_path=args.checkpoint,
    )
