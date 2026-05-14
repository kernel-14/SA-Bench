"""
main.py

Entry point for the NFIG reproduction pipeline.
Parses command-line arguments, loads the YAML configuration,
instantiates all required components (datasets, models, trainers, evaluator),
and executes the selected pipeline stage.

Usage:
    python main.py --config config.yaml --stage all
    python main.py --config config.yaml --stage tokenizer_train
    python main.py --config config.yaml --stage extract_tokens
    python main.py --config config.yaml --stage generator_train
    python main.py --config config.yaml --stage evaluate
"""

import argparse
import os
import random
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Project modules
from utils.helpers import (
    load_config,
    extract_tokens,
    save_tokens,
    load_tokens,
    collate_token_batch,
)
from data.dataset import ImageNetDataset
from models.fr_vae import FRVAE
from models.discriminator import DinoDiscriminator
from models.ar_transformer import VARTransformer
from trainers.tokenizer_trainer import TokenizerTrainer
from trainers.generator_trainer import GeneratorTrainer
from evaluation.evaluate import Evaluator


# ----------------------------------------------------------------------
# Minimal token dataset for generator training
# ----------------------------------------------------------------------
class TokenDataset(Dataset):
    """
    Wraps the saved tokenised data (from `save_tokens`) into a PyTorch Dataset.

    Each item is a tuple: (list of per‑scale tensors, label).
    """
    def __init__(self, file_path: str) -> None:
        super().__init__()
        token_dict, labels = load_tokens(file_path)

        # Sort scale keys numerically (assume keys like 'tokens_scale_0')
        scale_keys = sorted(
            [k for k in token_dict.keys() if k.startswith('tokens_scale_')],
            key=lambda x: int(x.split('_')[-1]),
        )
        self.tokens: List[torch.Tensor] = [token_dict[k] for k in scale_keys]
        self.labels: torch.Tensor = labels

        # Sanity check: all token tensors should have the same first dimension
        assert all(t.size(0) == self.labels.size(0) for t in self.tokens), \
            "Mismatched token and label lengths."

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], int]:
        token_list = [self.tokens[j][idx] for j in range(len(self.tokens))]
        label = self.labels[idx].item()
        return token_list, label


# ----------------------------------------------------------------------
# Main orchestrator
# ----------------------------------------------------------------------
class Main:
    """
    Reproducibility orchestrator for the NFIG framework.
    
    Args:
        config_path: Filesystem path to the configuration YAML file.
    """

    def __init__(self, config_path: str) -> None:
        # ---- Load configuration ----
        self.config: Dict = load_config(config_path)
        print("[Main] Configuration loaded.")

        # ---- Device setup ----
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"[Main] Using device: {self.device}")

        # ---- Reproducibility seeds ----
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Deterministic algorithms (may impact performance; can be disabled via config if needed)
        # We keep it for faithful reproduction.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # ---- Checkpoints directory ----
        self.checkpoints_dir = self.config["logging"]["checkpoints_dir"]
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public dispatcher
    # ------------------------------------------------------------------
    def run_experiment(self, stage: str) -> None:
        """Execute one or more pipeline stages."""
        if stage in ("tokenizer_train", "all"):
            self.train_tokenizer()
        if stage in ("extract_tokens", "all"):
            self.extract_tokens()
        if stage in ("generator_train", "all"):
            self.train_generator()
        if stage in ("evaluate", "all"):
            self.evaluate_all()

    # ------------------------------------------------------------------
    # Stage 1: Train FR‑VAE tokenizer + DINO discriminator
    # ------------------------------------------------------------------
    def train_tokenizer(self) -> None:
        print("[Stage] Tokenizer training")
        cfg = self.config

        # ---- Data loaders ----
        train_transform = ImageNetDataset.default_transform("train", cfg["data"]["image_size"])
        val_transform = ImageNetDataset.default_transform("val", cfg["data"]["image_size"])

        train_dataset = ImageNetDataset(
            root=cfg["data"]["data_root"],
            split="train",
            transform=train_transform,
        )
        val_dataset = ImageNetDataset(
            root=cfg["data"]["data_root"],
            split="val",
            transform=val_transform,
        )

        batch_size = cfg["tokenizer"]["tokenizer_batch_size"]
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # ---- Build FR‑VAE ----
        fr_vae = FRVAE(cfg).to(self.device)

        # ---- Build DINO discriminator ----
        # Load a separate frozen DINOv2 backbone.
        dino_backbone = self._load_dinov2_backbone(cfg["discriminator"]["dino_backbone"])
        discriminator = DinoDiscriminator(dino_backbone).to(self.device)

        # ---- Trainer ----
        trainer = TokenizerTrainer(
            model=fr_vae,
            disc=discriminator,
            config=cfg,
        )

        # ---- Launch training ----
        trainer.train(train_loader, val_loader)

        print("[Stage] Tokenizer training completed. Best checkpoint saved.")

    # ------------------------------------------------------------------
    # Stage 2: Extract discrete tokens with trained FR‑VAE
    # ------------------------------------------------------------------
    def extract_tokens(self) -> None:
        print("[Stage] Token extraction")
        cfg = self.config

        # ---- Rebuild and load trained tokenizer ----
        best_ckpt = os.path.join(self.checkpoints_dir, "best.pt")
        if not os.path.isfile(best_ckpt):
            raise FileNotFoundError(
                f"Tokenizer checkpoint not found at {best_ckpt}. "
                f"Please run tokenizer training first."
            )

        checkpoint = torch.load(best_ckpt, map_location=self.device)
        fr_vae = FRVAE(cfg).to(self.device)
        fr_vae.load_state_dict(checkpoint["fr_vae_state_dict"])
        fr_vae.eval()

        # ---- Data loaders ----
        transform = ImageNetDataset.default_transform("train", cfg["data"]["image_size"])
        for split, out_name in [("train", "train_tokens.pt"), ("val", "val_tokens.pt")]:
            dataset = ImageNetDataset(
                root=cfg["data"]["data_root"],
                split=split,
                transform=transform,
            )
            # Use a modest batch size because we only need forward passes.
            loader = DataLoader(
                dataset,
                batch_size=cfg["tokenizer"]["tokenizer_batch_size"],
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

            all_tokens, all_labels = extract_tokens(fr_vae, loader, self.device)
            out_path = os.path.join(self.checkpoints_dir, out_name)
            save_tokens(all_tokens, all_labels, out_path)
            print(f"[Stage] {split} tokens saved to {out_path} "

    # ------------------------------------------------------------------
    # Stage 3: Train the Next‑Frequency Prediction transformer
    # ------------------------------------------------------------------
    def train_generator(self) -> None:
        print("[Stage] Generator training")
        cfg = self.config

        # ---- Tokenised datasets ----
        train_tokens_path = os.path.join(self.checkpoints_dir, "train_tokens.pt")
        val_tokens_path = os.path.join(self.checkpoints_dir, "val_tokens.pt")
        for fpath in (train_tokens_path, val_tokens_path):
            if not os.path.isfile(fpath):
                raise FileNotFoundError(
                    f"Token file {fpath} not found. "
                    f"Please run token extraction first."
                )

        train_dataset = TokenDataset(train_tokens_path)
        val_dataset = TokenDataset(val_tokens_path)

        batch_size = cfg["training_generator"]["batch_size"]
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_token_batch,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_token_batch,
        )

        # ---- Build the AR transformer ----
        model = VARTransformer(cfg).to(self.device)

        # ---- Trainer ----
        trainer = GeneratorTrainer(model=model, config=cfg)

        # ---- Launch training ----
        trainer.train(train_loader, val_loader)

        print("[Stage] Generator training completed. Checkpoints saved.")

    # ------------------------------------------------------------------
    # Stage 4: Evaluation (FID, IS, Precision, Recall)
    # ------------------------------------------------------------------
    def evaluate_all(self) -> None:
        print("[Stage] Evaluation")
        cfg = self.config

        # ---- Load generator ----
        gen_ckpt = os.path.join(self.checkpoints_dir, "generator_best.pt")
        if not os.path.isfile(gen_ckpt):
            # Fallback to final if best does not exist
            gen_ckpt = os.path.join(self.checkpoints_dir, "generator_final.pth")
        if not os.path.isfile(gen_ckpt):
            raise FileNotFoundError(
                f"Generator checkpoint not found. Expected at "
                f"{os.path.join(self.checkpoints_dir, 'generator_best.pt')} or "
                f"{os.path.join(self.checkpoints_dir, 'generator_final.pth')}."
            )
        gen_state = torch.load(gen_ckpt, map_location=self.device)
        gen_model = VARTransformer(cfg).to(self.device)
        gen_model.load_state_dict(gen_state["model_state_dict"])
        gen_model.eval()

        # ---- Load tokenizer (FR‑VAE) ----
        tok_ckpt = os.path.join(self.checkpoints_dir, "best.pt")
        if not os.path.isfile(tok_ckpt):
            raise FileNotFoundError(
                f"Tokenizer checkpoint not found at {tok_ckpt}. "
                f"Please complete tokenizer training."
            )
        tok_state = torch.load(tok_ckpt, map_location=self.device)
        tokenizer = FRVAE(cfg).to(self.device)
        tokenizer.load_state_dict(tok_state["fr_vae_state_dict"])
        tokenizer.eval()

        # ---- Evaluator ----
        evaluator = Evaluator(
            gen_model=gen_model,
            tokenizer=tokenizer,
            config=cfg,
        )

        # ---- Compute metrics ----
        fid, is_score = evaluator.compute_fid_is()
        precision, recall = evaluator.compute_precision_recall()

        print("\n" + "=" * 50)
        print("       FINAL EVALUATION RESULTS")
        print("=" * 50)
        print(f"  FID       : {fid:.4f}")
        print(f"  IS        : {is_score:.2f}")
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print("=" * 50)

    # ------------------------------------------------------------------
    # Helper: load a DINOv2 backbone for the discriminator
    # ------------------------------------------------------------------
    @staticmethod
    def _load_dinov2_backbone(backbone_name: str) -> nn.Module:
        """Load a pretrained DINOv2 VisionTransformer from torchvision."""
        import torchvision.models as tv_models
        if backbone_name == "dinov2_vitb14":
            # DINOv2 ViT-B/14 weights
            try:
                # torchvision >= 0.17
                weights = tv_models.DINOv2_ViT_B14_Weights.IMAGENET1K
            except AttributeError:
                raise RuntimeError(
                    "DINOv2 weights require torchvision >= 0.17. "
                    "Please upgrade or use a compatible checkpoint."
                )
            model = tv_models.dinov2_vitb14(weights=weights)
        else:
            raise ValueError(f"Unsupported DINO backbone: {backbone_name}")

        # Discard the classification head (if any)
        if hasattr(model, "heads"):
            model.heads = nn.Identity()
        return model


# ----------------------------------------------------------------------
# Command‑line interface
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="NFIG Reproduction Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["tokenizer_train", "extract_tokens", "generator_train", "evaluate", "all"],
        help="Which stage(s) to execute (default: all)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"Error: Configuration file not found at {args.config}")
        sys.exit(1)

    orchestrator = Main(args.config)
    orchestrator.run_experiment(args.stage)


if __name__ == "__main__":
    main()
