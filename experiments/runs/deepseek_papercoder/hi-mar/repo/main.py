"""
main.py

Entry point for the Hi‑MAR reproduction pipeline.  Coordinates configuration,
environment setup, training, inference, and evaluation for both ImageNet
(class‑conditional) and MS‑COCO (text‑to‑image) datasets.
"""

import argparse
import logging
import os
import sys
from typing import Optional, Union

import numpy as np
import torch
import torch.utils.data
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset

# Project modules (assumed to be importable from the same package)
from config import Config, load_config, ImageNetTrainConfig, CocoTrainConfig, InferencePhaseConfig
from utils import seed_everything, setup_logger
from vae_tokenizer import VAETokenizer
from masking import TokenMasker
from model import HiMARTransformer
from diffusion_heads import MLPDiffusionHead, DiffusionTransformerHead
from dataset_loader import DatasetLoader, COCODataset
from trainer import Trainer
from inference import InferenceRunner
from evaluation import Evaluator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Main orchestrator class
# ---------------------------------------------------------------------------

class Main:
    """Central controller for the Hi‑MAR experiment."""

    def __init__(self, config: Config, args: argparse.Namespace) -> None:
        self.config = config
        self.args = args

        # Extract typed sub‑configs
        self.global_cfg = config.global_config
        self.model_cfg = config.model
        self.data_cfg = config.data
        self.mask_cfg = config.masking

        # Select dataset‑specific training / inference configs
        if args.dataset == "imagenet":
            self.train_cfg_dataset: Union[ImageNetTrainConfig, CocoTrainConfig] = config.training.imagenet
            self.infer_cfg: InferencePhaseConfig = config.inference.imagenet
        else:
            self.train_cfg_dataset = config.training.coco
            self.infer_cfg = config.inference.coco

        # ------------------------------------------------------------------
        #  1. Deterministic seed & logging
        # ------------------------------------------------------------------
        seed_everything(self.global_cfg.seed)
        os.makedirs(self.global_cfg.output_dir, exist_ok=True)
        setup_logger(__name__, self.global_cfg.output_dir, level=logging.INFO)
        logger.info("Starting Hi‑MAR experiment")
        logger.info("Mode: %s  |  Dataset: %s", args.mode, args.dataset)

        # ------------------------------------------------------------------
        #  2. Device & Accelerator (mixed precision)
        # ------------------------------------------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mixed_precision = getattr(self.train_cfg_dataset, "mixed_precision", "bf16")
        if args.debug:
            mixed_precision = "no"
        self.accelerator = Accelerator(mixed_precision=mixed_precision)

        # ------------------------------------------------------------------
        #  3. VAE tokenizer (frozen KL‑16)
        # ------------------------------------------------------------------
        vae_path = self.global_cfg.vae_path
        if not os.path.isfile(vae_path):
            logger.error(
                "VAE checkpoint not found at '%s'.  Please obtain or train the MAR KL‑16 VAE.",
                vae_path,
            )
            sys.exit(1)
        self.vae = VAETokenizer(vae_path, self.device)

        # ------------------------------------------------------------------
        #  4. Masking utility
        # ------------------------------------------------------------------
        self.token_masker = TokenMasker(self.mask_cfg, dataset=self.args.dataset)

        # ------------------------------------------------------------------
        #  5. Model components
        # ------------------------------------------------------------------
        # Number of classes / text encoder dimension
        num_classes = 1000 if args.dataset == "imagenet" else 0
        text_encoder_dim = 0
        if args.dataset == "coco":
            # CLIP‑ViT‑L/14 output dimension
            text_encoder_dim = 768

        self.model = HiMARTransformer(
            config=self.model_cfg,
            latent_dim=self.global_cfg.latent_dim,
            num_classes=num_classes,
            text_encoder_dim=text_encoder_dim,
        )
        self.head1 = MLPDiffusionHead(self.model_cfg, latent_dim=self.global_cfg.latent_dim)
        self.head2 = DiffusionTransformerHead(self.model_cfg, latent_dim=self.global_cfg.latent_dim)

        # Will hold the Trainer after training (needed for EMA application)
        self.trainer: Optional[Trainer] = None

    # ------------------------------------------------------------------ #
    #  Data loading helpers
    # ------------------------------------------------------------------ #

    def _build_data_loader(self, split: str) -> DataLoader:
        """Return a training or validation DataLoader for the chosen dataset."""
        batch_size = self.train_cfg_dataset.batch_size
        if batch_size is None:
            if self.args.dataset == "imagenet":
                batch_size = 256
            else:
                batch_size = 256
        # Override for debug
        if self.args.debug:
            batch_size = 4

        loader = DatasetLoader(
            config=self.data_cfg,
            vae_tokenizer=self.vae,
            dataset=self.args.dataset,
            batch_size=batch_size,
            num_workers=4,
        )
        return loader.get_train_loader() if split == "train" else loader.get_val_loader()

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #

    def _determine_epochs(self, train_loader: DataLoader) -> int:
        """Return the number of epochs to train for."""
        if self.args.dataset == "imagenet":
            return self.train_cfg_dataset.epochs
        else:   # coco
            total_steps = getattr(self.train_cfg_dataset, "total_steps", None)
            if total_steps is not None:
                steps_per_epoch = len(train_loader)
                return int(np.ceil(total_steps / steps_per_epoch))
            # Fallback if total_steps is None – use a sensible default
            logger.warning(
                "No total_steps defined for COCO training.  Using default 400 epochs."
            )
            return 400

    def _setup_trainer(self, train_loader: DataLoader, epochs: int) -> Trainer:
        """Instantiate the Trainer with all required components."""
        return Trainer(
            model=self.model,
            head1=self.head1,
            head2=self.head2,
            vae=self.vae,
            dataloader=train_loader,
            train_config=self.train_cfg_dataset,
            token_masker=self.token_masker,
            accelerator=self.accelerator,
            dataset_type=self.args.dataset,
            epochs=epochs,
            ema_momentum=getattr(self.train_cfg_dataset, "ema_momentum", 0.9999),
            gradient_clip=getattr(self.train_cfg_dataset, "gradient_clip", None),
        )

    # ------------------------------------------------------------------ #
    #  Checkpoint I/O
    # ------------------------------------------------------------------ #

    def _save_checkpoint(self, trainer: Trainer, path: str) -> None:
        """Persist model weights, optimizer, scheduler, and EMA shadows."""
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_head1 = self.accelerator.unwrap_model(self.head1)
        unwrapped_head2 = self.accelerator.unwrap_model(self.head2)

        state = {
            "model": unwrapped_model.state_dict(),
            "head1": unwrapped_head1.state_dict(),
            "head2": unwrapped_head2.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "scheduler": trainer.scheduler.state_dict(),
        }
        if trainer.ema is not None:
            state["ema_shadow"] = trainer.ema.shadow

        torch.save(state, path)
        logger.info("Checkpoint saved to %s", path)

    def _load_checkpoint(
        self,
        path: str,
        trainer: Optional[Trainer] = None,
        load_weights_only: bool = False,
    ) -> None:
        """Load a checkpoint into the models and (optionally) the trainer."""
        if not os.path.isfile(path):
            logger.error("Checkpoint file not found: %s", path)
            raise FileNotFoundError(path)

        checkpoint = torch.load(path, map_location=self.device)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_head1 = self.accelerator.unwrap_model(self.head1)
        unwrapped_head2 = self.accelerator.unwrap_model(self.head2)

        unwrapped_model.load_state_dict(checkpoint["model"])
        unwrapped_head1.load_state_dict(checkpoint["head1"])
        unwrapped_head2.load_state_dict(checkpoint["head2"])

        if not load_weights_only and trainer is not None:
            trainer.optimizer.load_state_dict(checkpoint["optimizer"])
            trainer.scheduler.load_state_dict(checkpoint["scheduler"])
            if "ema_shadow" in checkpoint and trainer.ema is not None:
                trainer.ema.shadow = checkpoint["ema_shadow"]

        logger.info("Checkpoint loaded from %s", path)

    # ------------------------------------------------------------------ #
    #  Evaluation data loaders (dataset‑specific)
    # ------------------------------------------------------------------ #

    def _create_imagenet_eval_loader(self) -> DataLoader:
        """
        Generator that yields 50,000 class IDs (50 per class), used to guide
        generation of the 50K evaluation images.
        """
        class_ids = []
        for c in range(1000):
            class_ids.extend([c] * 50)

        class EvalDataset(Dataset):
            def __init__(self, ids):
                self.ids = ids

            def __len__(self):
                return len(self.ids)

            def __getitem__(self, idx):
                return {"class_id": torch.tensor(self.ids[idx], dtype=torch.long)}

        dataset = EvalDataset(class_ids)
        return DataLoader(dataset, batch_size=50, shuffle=False, num_workers=0)

    def _create_coco_eval_loader(self) -> DataLoader:
        """
        Randomly select 30,000 prompts from the COCO validation set.
        Requires pre‑computed CLIP embeddings.
        """
        emb_path = os.path.join(self.data_cfg.coco_root, "coco_clip_embeddings.pkl")
        if not os.path.isfile(emb_path):
            raise FileNotFoundError(
                f"COCO CLIP embeddings not found at '{emb_path}'. "
                "Run 'build_coco_embeddings' to generate them."
            )

        coco_val = COCODataset(
            root=self.data_cfg.coco_root,
            ann_file=self.data_cfg.coco_ann_file,
            embeddings_path=emb_path,
            split="val",
        )
        total = len(coco_val)
        n = min(30000, total)
        rng = np.random.default_rng(self.global_cfg.seed)
        indices = rng.choice(total, size=n, replace=False)
        subset = torch.utils.data.Subset(coco_val, indices)
        return DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)

    # ------------------------------------------------------------------ #
    #  Execution of the chosen mode
    # ------------------------------------------------------------------ #

    def run_training(self, resume: Optional[str] = None) -> None:
        """Full training routine (possibly resuming from a checkpoint)."""
        train_loader = self._build_data_loader("train")
        epochs = self._determine_epochs(train_loader)
        logger.info(
            "Training for %d epochs (%d steps / epoch)",
            epochs,
            len(train_loader),
        )

        self.trainer = self._setup_trainer(train_loader, epochs)

        if resume is not None:
            self._load_checkpoint(resume, self.trainer, load_weights_only=False)

        try:
            self.trainer.train()
        finally:
            # Save final checkpoint regardless of early termination
            save_path = os.path.join(self.global_cfg.output_dir, "final_checkpoint.pt")
            self._save_checkpoint(self.trainer, save_path)

        logger.info("Training completed.")

    def run_evaluation(self, checkpoint_path: Optional[str] = None) -> None:
        """
        Generate images and compute quantitative metrics (FID, IS, P/R).
        """
        # Load checkpoint weights if provided
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path, trainer=None, load_weights_only=True)

        # Apply EMA weights if they exist (e.g., after training)
        if self.trainer is not None and self.trainer.ema is not None:
            logger.info("Applying EMA weights for evaluation.")
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_head1 = self.accelerator.unwrap_model(self.head1)
            unwrapped_head2 = self.accelerator.unwrap_model(self.head2)
            self.trainer.ema.apply([unwrapped_model, unwrapped_head1, unwrapped_head2])

        # Inference runner
        inference_runner = InferenceRunner(
            model=self.model,
            head1=self.head1,
            head2=self.head2,
            vae=self.vae,
            config=self.infer_cfg,
            token_masker=self.token_masker,
        )

        # ---- Build dataloader for prompts ----
        if self.args.dataset == "imagenet":
            eval_loader = self._create_imagenet_eval_loader()
        else:
            eval_loader = self._create_coco_eval_loader()

        # Generate all images
        logger.info("Generating images for evaluation...")
        generated_images = inference_runner.run_on_dataloader(eval_loader, use_cfg=True)

        # ---- Evaluator ----
        real_dir = (
            os.path.join(self.data_cfg.imagenet_root, "train")
            if self.args.dataset == "imagenet"
            else os.path.join(self.data_cfg.coco_root, "val2017")
        )
        cache_dir = os.path.join(self.global_cfg.output_dir, "eval_cache")

        evaluator = Evaluator(
            real_images_dir=real_dir,
            cache_dir=cache_dir,
            image_size=self.global_cfg.image_res_high,
            batch_size=64,
            device=str(self.device),
        )

        # Compute metrics
        fid = evaluator.compute_fid(generated_images)
        logger.info("FID: %.4f", fid)

        if self.args.dataset == "imagenet":
            is_score = evaluator.compute_is(generated_images)
            prec, rec = evaluator.compute_precision_recall(generated_images)
            logger.info(
                "Inception Score: %.2f  Precision: %.3f  Recall: %.3f",
                is_score,
                prec,
                rec,
            )
        else:
            # For COCO only FID is reported, but you can add others.
            logger.info("COCO evaluation finished.")

    # ------------------------------------------------------------------ #
    #  Main experiment dispatcher
    # ------------------------------------------------------------------ #

    def run_experiment(self) -> None:
        """Dispatch to training, evaluation, or both based on CLI arguments."""
        mode = self.args.mode

        if mode in ("train", "train_eval"):
            self.run_training(resume=self.args.resume)

        if mode in ("eval", "train_eval"):
            # For 'eval', use the provided resume checkpoint; for 'train_eval',
            # the models are already in memory from training.
            ckpt = None
            if mode == "eval" and self.args.resume:
                ckpt = self.args.resume
            self.run_evaluation(checkpoint_path=ckpt)


# ---------------------------------------------------------------------------
#  Command‑line interface
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hi‑MAR reproduction script")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "train_eval"],
        help="Operation mode: train, eval, or both",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        choices=["imagenet", "coco"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training or load for evaluation",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (small batch, no mixed precision, few steps)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    # ------ Debug overrides ------
    if args.debug:
        logger = logging.getLogger(__name__)
        logger.info("Running in DEBUG mode – tweaking configuration for fast test.")
        config.training.imagenet.batch_size = 4
        config.training.coco.batch_size = 4
        config.training.imagenet.mixed_precision = "no"
        config.training.coco.mixed_precision = "no"
        config.inference.imagenet.phase1_steps = 2
        config.inference.imagenet.phase2_steps = 1
        config.inference.imagenet.inner_diffusion_steps = 1
        config.inference.coco.phase1_steps = 2
        config.inference.coco.phase2_steps = 1
        config.inference.coco.inner_diffusion_steps = 1

    main = Main(config, args)
    main.run_experiment()
