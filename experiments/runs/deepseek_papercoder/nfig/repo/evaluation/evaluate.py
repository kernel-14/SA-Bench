"""
evaluation/evaluate.py

Evaluator for the NFIG reproduction project. Generates 50k class‑conditional images
using the trained VARTransformer and FR‑VAE, extracts Inception features, and computes
FID, Inception Score, Precision, and Recall following the protocols described in the
paper. All hyperparameters (e.g., CFG scale, top‑k) are read from the configuration
dictionary loaded via config.yaml.

The evaluator caches generated images and features so that multiple metric calls
do not regenerate the entire set.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import inception_v3
from tqdm import tqdm

from data.dataset import ImageNetDataset
from models.ar_transformer import VARTransformer
from models.fr_vae import FRVAE
from utils.metrics import compute_fid, compute_is, compute_precision_recall


class InceptionFeatureExtractor(nn.Module):
    """
    A wrapper around torchvision's pretrained InceptionV3 that returns the
    'pool3' features (2048‑dimensional) used for FID and Precision/Recall.
    The model is frozen and set to evaluation mode automatically.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = inception_v3(
            pretrained=True, transform_input=False, aux_logits=False
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Register a forward hook to capture the pool3 output (just before
        # the final classification layer).
        self.pool3: Optional[torch.Tensor] = None
        self.model.avgpool.register_forward_hook(self._hook)

    def _hook(
        self,
        module: nn.Module,
        input: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # avgpool output is (B, 2048, 1, 1) -> flatten to (B, 2048)
        self.pool3 = output.view(output.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Batch of images in range [0, 1] (or properly normalized).
        Returns:
            pool3 features of shape (B, 2048).
        """
        with torch.no_grad():
            _ = self.model(x)
        if self.pool3 is None:
            raise RuntimeError("Pool3 hook did not capture features.")
        return self.pool3.clone()


class Evaluator:
    """
    Evaluator for the NFIG autoregressive generation model.

    Args:
        gen_model: Trained VARTransformer (generator).
        tokenizer: Trained FRVAE tokenizer (used to decode token maps into images).
        config: Global configuration dictionary (contents of config.yaml).
    """

    def __init__(
        self,
        gen_model: VARTransformer,
        tokenizer: FRVAE,
        config: Dict,
    ) -> None:
        self.gen_model = gen_model
        self.tokenizer = tokenizer
        self.config = config

        # ---- Inference parameters ----
        infer_cfg = config["inference"]
        self.cfg_scale: float = infer_cfg.get("cfg_scale", 4.5)
        self.top_k: int = infer_cfg.get("top_k", 990)

        # ---- Dataset parameters ----
        data_cfg = config["data"]
        self.num_classes: int = data_cfg["num_classes"]
        self.data_root: str = data_cfg["data_root"]
        self.image_size: int = data_cfg.get("image_size", 256)

        # ---- Device ----
        self.device = next(self.gen_model.parameters()).device

        # ---- Inception feature extractors ----
        # One for FID/P, R features, another is loaded inside compute_is,
        # but we can reuse the extractor for both to save memory.
        # We'll use the extractor for FID/P,R and compute_is will load its own Inception.
        self.inception_extractor = InceptionFeatureExtractor().to(self.device)

        # ---- Normalisation transformation for Inception input ----
        # The extractor expects tensors in [0, 1]. Our generated images are in [-1, 1],
        # so we need to rescale.
        self.inception_norm = transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # ---- Caches ----
        self._real_features: Optional[torch.Tensor] = None
        self._fake_features: Optional[torch.Tensor] = None
        self._fake_images: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Reference (real) feature extraction
    # ------------------------------------------------------------------
    def _get_reference_features(self) -> torch.Tensor:
        """Compute (or load cached) Inception features for 50k training images."""
        if self._real_features is not None:
            return self._real_features

        # Build DataLoader for the training split
        train_transform = ImageNetDataset.default_transform("train", self.image_size)
        train_dataset = ImageNetDataset(
            root=self.data_root, split="train", transform=train_transform
        )
        loader = DataLoader(
            train_dataset,
            batch_size=128,
            num_workers=4,
            shuffle=False,          # deterministic order
            pin_memory=True,
        )

        all_features: List[torch.Tensor] = []
        total_needed = 50000
        collected = 0
        print("[Evaluator] Extracting reference features from training set...")
        with tqdm(total=total_needed, desc="Ref Feats") as pbar:
            for images, _ in loader:
                images = images.to(self.device, non_blocking=True)
                # Convert from [-1, 1] to [0, 1] and apply Inception normalisation
                im_01 = (images + 1.0) / 2.0
                im_norm = self.inception_norm(im_01)

                feats = self.inception_extractor(im_norm).cpu()
                all_features.append(feats)

                collected += feats.size(0)
                pbar.update(feats.size(0))

                if collected >= total_needed:
                    break

        all_features = torch.cat(all_features, dim=0)[:total_needed].detach()
        self._real_features = all_features
        return all_features

    # ------------------------------------------------------------------
    # Image reconstruction from token maps
    # ------------------------------------------------------------------
    def _reconstruct_images(
        self,
        token_maps: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Convert a list of per‑scale token maps into RGB images.

        Args:
            token_maps: List of tensors, one per frequency band, each of shape
                (B, h_i, w_i) containing codebook indices.

        Returns:
            Reconstructed images of shape (B, 3, H, W) in [-1, 1].
        """
        B = token_maps[0].size(0)
        Hp, Wp = self.tokenizer.H_prime, self.tokenizer.W_prime  # both 16
        C = self.tokenizer.latent_dim                              # e.g., 256

        # Shared codebook
        codebook = self.tokenizer.quantizer.codebook  # (K, C)

        # Reconstruct composite feature map f_hat
        f_hat = torch.zeros(B, C, Hp, Wp, device=self.device, dtype=torch.float32)

        for i, token_map in enumerate(token_maps):
            # token_map: (B, h_i, w_i)
            h_i, w_i = token_map.shape[1], token_map.shape[2]

            # Look up vectors
            indices_flat = token_map.reshape(-1)                 # (B * h_i * w_i,)
            vectors = codebook[indices_flat]                     # (B * h_i * w_i, C)
            vectors = vectors.reshape(B, h_i, w_i, C).permute(0, 3, 1, 2)  # (B, C, h_i, w_i)

            # Upsample to (Hp, Wp)
            upsampled = F.interpolate(
                vectors,
                size=(Hp, Wp),
                mode="bilinear",
                align_corners=False,
            )                                                   # (B, C, Hp, Wp)

            f_hat = f_hat + upsampled

        # Decode to image space
        with torch.no_grad():
            images = self.tokenizer.decode(f_hat)                 # (B, 3, 256, 256)
        return images

    # ------------------------------------------------------------------
    # Feature extraction from generated images
    # ------------------------------------------------------------------
    def _extract_inception_features(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert a batch of generated images (in [-1, 1]) into Inception features.

        Args:
            images: Tensor of shape (B, 3, 256, 256).

        Returns:
            Feature tensor of shape (B, 2048).
        """
        # Preprocess: [-1, 1] -> [0, 1] -> Inception norm
        im_01 = (images + 1.0) / 2.0
        im_norm = self.inception_norm(im_01)
        feats = self.inception_extractor(im_norm)
        return feats

    # ------------------------------------------------------------------
    # Generate all 50k class‑conditional images (main generation routine)
    # ------------------------------------------------------------------
    def _generate_all(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate 50 images per class, decode, and extract features.

        Returns:
            Tuple (fake_features, fake_images):
                - fake_features: tensor of shape (50000, 2048)
                - fake_images: tensor of shape (50000, 3, 256, 256)
        """
        if self._fake_features is not None and self._fake_images is not None:
            return self._fake_features, self._fake_images

        self.gen_model.eval()
        num_per_class = 50
        total = self.num_classes * num_per_class  # 50000

        all_images: List[torch.Tensor] = []
        all_features: List[torch.Tensor] = []

        print(
            f"[Evaluator] Generating {total} images "
            f"({num_per_class} per class, CFG={self.cfg_scale:.1f}, top_k={self.top_k})"
        )

        with torch.no_grad():
            for class_idx in tqdm(range(self.num_classes), desc="Generating"):
                # Accumulate token maps for 50 images of this class
                scale_lists: List[List[torch.Tensor]] = None  # will be list of per‑scale lists

                for _ in range(num_per_class):
                    # generate returns list of tensors, each (1, n_i)
                    token_list = self.gen_model.generate(
                        class_label=class_idx,
                        top_k=self.top_k,
                        cfg_scale=self.cfg_scale,
                    )
                    # Move to device (they are already on model's device)
                    if scale_lists is None:
                        scale_lists = [[] for _ in range(len(token_list))]
                    for scale_idx, tok in enumerate(token_list):
                        scale_lists[scale_idx].append(tok.detach())

                # Stack per‑scale lists into batched tensors (50, n_i)
                batched_tokens = [torch.cat(sl, dim=0) for sl in scale_lists]

                # Reconstruct images
                images = self._reconstruct_images(batched_tokens)  # (50, 3, 256, 256)
                all_images.append(images.cpu())

                # Extract Inception features (on GPU)
                feats = self._extract_inception_features(images).cpu()  # (50, 2048)
                all_features.append(feats)

        fake_images = torch.cat(all_images, dim=0)       # (50000, 3, 256, 256)
        fake_features = torch.cat(all_features, dim=0)   # (50000, 2048)

        # Clamp to valid range (just in case)
        fake_images = torch.clamp(fake_images, -1.0, 1.0)

        self._fake_images = fake_images
        self._fake_features = fake_features
        return fake_features, fake_images

    # ------------------------------------------------------------------
    # Public metric methods
    # ------------------------------------------------------------------
    def compute_fid_is(self) -> Tuple[float, float]:
        """
        Compute Fréchet Inception Distance (FID) and Inception Score (IS).

        Returns:
            (fid, inception_score) tuple of floats.
        """
        # Ensure generated data is available
        fake_features, fake_images = self._generate_all()
        real_features = self._get_reference_features()

        # FID (uses features already on CPU)
        fid = compute_fid(real_features, fake_features)

        # IS (needs raw images on the same device as the metrics function)
        # compute_is internally moves images to GPU if necessary; we provide CPU images.
        is_score = compute_is(fake_images, batch_size=64)

        return fid, is_score

    def compute_precision_recall(self) -> Tuple[float, float]:
        """
        Compute Precision and Recall using the k‑NN manifold method.

        Returns:
            (precision, recall) tuple of floats.
        """
        fake_features, _ = self._generate_all()
        real_features = self._get_reference_features()

        # Both feature tensors are on CPU; metrics function expects device consistency.
        precision, recall = compute_precision_recall(real_features, fake_features)
        return precision, recall


# ----------------------------------------------------------------------
# Minimal self‑test (not executed when imported)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <config_path>")
        sys.exit(1)

    from utils.helpers import load_config
    from models.fr_vae import FRVAE
    from models.ar_transformer import VARTransformer

    cfg = load_config(sys.argv[1])

    # Dummy models for testing (normally they would be loaded from checkpoints)
    # In practice, you would load pretrained weights:
    # tokenizer = FRVAE(cfg)
    # tokenizer.load_state_dict(torch.load("fr_vae_best.pth")["fr_vae_state_dict"])
    # gen_model = VARTransformer(cfg)
    # gen_model.load_state_dict(torch.load("generator_best.pth")["model_state_dict"])
    # Here we just create random models to test the evaluator’s logic.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = FRVAE(cfg).to(device)
    gen_model = VARTransformer(cfg).to(device)

    evaluator = Evaluator(gen_model, tokenizer, cfg)
    print("Starting FID/IS computation...")
    fid, is_ = evaluator.compute_fid_is()
    print(f"FID = {fid:.4f}, IS = {is_:.2f}")
    pr, rc = evaluator.compute_precision_recall()
    print(f"Precision = {pr:.4f}, Recall = {rc:.4f}")
