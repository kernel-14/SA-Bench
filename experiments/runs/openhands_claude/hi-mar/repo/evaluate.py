"""
Evaluation for Hi-MAR.

Metrics:
  - FID (Fréchet Inception Distance)
  - IS  (Inception Score)
  - Precision / Recall

Evaluation protocol (paper §4.2):
  - ImageNet: 50K generated samples, FID/IS/Precision/Recall
  - MS-COCO:  30K generated samples from random validation prompts, FID

Usage:
  python evaluate.py --checkpoint outputs/checkpoint_epoch0799.pt \
                     --dataset imagenet --model Hi-MAR-B \
                     --data_path /data/imagenet --num_samples 50000
"""

import argparse
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm

from config import (
    Config,
    coco_small_config,
    imagenet_base_config,
    imagenet_huge_config,
    imagenet_large_config,
)
from data import VAETokenizer, build_imagenet_loaders
from model import HiMAR, build_himar


# ---------------------------------------------------------------------------
# Inception network for FID / IS
# ---------------------------------------------------------------------------

class InceptionV3Features(nn.Module):
    """Extracts pool3 features from InceptionV3 for FID computation."""

    def __init__(self):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights
        inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        inception.eval()

        # Keep layers up to pool3 (2048-dim features)
        self.layers = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        # For IS: keep the full classifier
        self.fc = inception.fc

        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) in [0, 1]
        Returns:
            features: (B, 2048) pool3 features
            logits:   (B, 1000) class logits
        """
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        # Normalize to [-1, 1] as expected by InceptionV3
        x = 2 * x - 1
        features = self.layers(x).flatten(1)
        logits = self.fc(features)
        return features, logits


# ---------------------------------------------------------------------------
# FID computation
# ---------------------------------------------------------------------------

def compute_fid(
    real_features: np.ndarray,
    fake_features: np.ndarray,
) -> float:
    """Compute FID between two sets of features."""
    from scipy import linalg

    mu_r = real_features.mean(axis=0)
    mu_f = fake_features.mean(axis=0)
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_f = np.cov(fake_features, rowvar=False)

    diff = mu_r - mu_f
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean)
    return float(fid)


# ---------------------------------------------------------------------------
# IS computation
# ---------------------------------------------------------------------------

def compute_inception_score(
    logits: np.ndarray,
    num_splits: int = 10,
) -> Tuple[float, float]:
    """Compute Inception Score from class logits."""
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)

    scores = []
    n = len(probs)
    split_size = n // num_splits
    for i in range(num_splits):
        p = probs[i * split_size: (i + 1) * split_size]
        q = p.mean(axis=0, keepdims=True)
        kl = p * (np.log(p + 1e-10) - np.log(q + 1e-10))
        scores.append(np.exp(kl.sum(axis=1).mean()))

    return float(np.mean(scores)), float(np.std(scores))


# ---------------------------------------------------------------------------
# Precision / Recall
# ---------------------------------------------------------------------------

def compute_precision_recall(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    k: int = 3,
) -> Tuple[float, float]:
    """
    Improved Precision and Recall (Kynkäänniemi et al., 2019).
    Uses k-NN manifold estimation.
    """
    real_t = torch.from_numpy(real_features).float()
    fake_t = torch.from_numpy(fake_features).float()

    def knn_distances(a: torch.Tensor, b: torch.Tensor, k: int) -> torch.Tensor:
        # Compute pairwise distances in chunks to avoid OOM
        chunk = 1000
        dists = []
        for i in range(0, len(a), chunk):
            d = torch.cdist(a[i:i+chunk], b)
            kth, _ = d.kthvalue(k + 1, dim=1)
            dists.append(kth)
        return torch.cat(dists)

    real_knn = knn_distances(real_t, real_t, k)
    fake_knn = knn_distances(fake_t, fake_t, k)

    # Precision: fraction of fake samples in real manifold
    d_fake_to_real = torch.cdist(fake_t, real_t)
    precision = (d_fake_to_real.min(dim=1).values <= real_knn.unsqueeze(0)).float().mean().item()

    # Recall: fraction of real samples in fake manifold
    d_real_to_fake = torch.cdist(real_t, fake_t)
    recall = (d_real_to_fake.min(dim=1).values <= fake_knn.unsqueeze(0)).float().mean().item()

    return precision, recall


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    images: torch.Tensor,
    inception: InceptionV3Features,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract Inception features and logits from a tensor of images."""
    inception = inception.to(device)
    all_features = []
    all_logits = []

    dataset = TensorDataset(images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for (batch,) in tqdm(loader, desc="Extracting features"):
        batch = batch.to(device)
        # Normalize from [-1, 1] to [0, 1]
        batch = (batch + 1) / 2
        features, logits = inception(batch)
        all_features.append(features.cpu().numpy())
        all_logits.append(logits.cpu().numpy())

    return np.concatenate(all_features), np.concatenate(all_logits)


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_samples(
    model: HiMAR,
    vae: VAETokenizer,
    num_samples: int,
    batch_size: int,
    device: torch.device,
    num_classes: int = 1000,
    steps_phase1: int = 32,
    steps_phase2: int = 4,
    diff_steps: int = 100,
    cfg_scale: float = 1.5,
    temperature: float = 1.0,
    class_labels: Optional[torch.Tensor] = None,
    text_embeds: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Generate num_samples images and return as (N, 3, H, W) tensor in [-1, 1].
    """
    model.eval()
    all_images = []
    generated = 0

    h_l = w_l = int(math.isqrt(model.num_tokens_large))

    while generated < num_samples:
        bs = min(batch_size, num_samples - generated)

        if class_labels is not None:
            labels_batch = class_labels[generated:generated + bs].to(device)
            text_batch = None
        elif text_embeds is not None:
            labels_batch = None
            text_batch = text_embeds[generated:generated + bs].to(device)
        else:
            # Random class labels for ImageNet
            labels_batch = torch.randint(0, num_classes, (bs,), device=device)
            text_batch = None

        tokens = model.generate(
            batch_size=bs,
            class_labels=labels_batch,
            text_embeds=text_batch,
            steps_phase1=steps_phase1,
            steps_phase2=steps_phase2,
            diff_steps=diff_steps,
            cfg_scale=cfg_scale,
            temperature=temperature,
            device=device,
        )

        images = vae.decode(tokens, h_l, w_l)  # (bs, 3, H, W)
        all_images.append(images.cpu())
        generated += bs

    return torch.cat(all_images, dim=0)[:num_samples]


# ---------------------------------------------------------------------------
# Real image feature extraction from dataset
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_real_features(
    data_path: str,
    num_samples: int,
    inception: InceptionV3Features,
    device: torch.device,
    image_size: int = 256,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract Inception features from real ImageNet validation images."""
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    from torchvision import datasets
    dataset = datasets.ImageFolder(
        root=os.path.join(data_path, "val"),
        transform=transform,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_features = []
    all_logits = []
    count = 0

    inception = inception.to(device)
    for images, _ in tqdm(loader, desc="Real features"):
        if count >= num_samples:
            break
        images = images.to(device)
        features, logits = inception(images)
        all_features.append(features.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
        count += images.shape[0]

    features = np.concatenate(all_features)[:num_samples]
    logits = np.concatenate(all_logits)[:num_samples]
    return features, logits


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load config ----
    if args.dataset == "imagenet":
        if args.model == "Hi-MAR-B":
            cfg = imagenet_base_config()
        elif args.model == "Hi-MAR-L":
            cfg = imagenet_large_config()
        else:
            cfg = imagenet_huge_config()
    else:
        cfg = coco_small_config()

    cfg.model.model_name = args.model
    cfg.model.vae_path = args.vae_path

    # ---- Build model ----
    model = build_himar(
        model_name=cfg.model.model_name,
        token_dim=cfg.model.token_dim,
        num_tokens_small=cfg.model.num_tokens_small,
        num_tokens_large=cfg.model.num_tokens_large,
        num_classes=cfg.model.num_classes,
        text_embed_dim=cfg.model.text_embed_dim,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("ema", ckpt.get("model", ckpt))
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ---- VAE ----
    vae = VAETokenizer(vae_path=cfg.model.vae_path, device=str(device))

    # ---- Inception model ----
    inception = InceptionV3Features().to(device)

    # ---- Generate samples ----
    print(f"Generating {args.num_samples} samples...")
    fake_images = generate_samples(
        model=model,
        vae=vae,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=device,
        num_classes=cfg.model.num_classes,
        steps_phase1=args.steps_phase1,
        steps_phase2=args.steps_phase2,
        diff_steps=args.diff_steps,
        cfg_scale=args.cfg_scale,
    )

    # ---- Extract fake features ----
    print("Extracting features from generated images...")
    fake_features, fake_logits = extract_features(fake_images, inception, device, args.batch_size)

    # ---- Extract real features ----
    print("Extracting features from real images...")
    real_features, real_logits = extract_real_features(
        args.data_path, args.num_samples, inception, device,
        image_size=cfg.model.image_size_large, batch_size=args.batch_size,
    )

    # ---- Compute metrics ----
    fid = compute_fid(real_features, fake_features)
    is_mean, is_std = compute_inception_score(fake_logits)
    precision, recall = compute_precision_recall(real_features, fake_features)

    print(f"\n{'='*50}")
    print(f"Model:     {args.model}")
    print(f"CFG scale: {args.cfg_scale}")
    print(f"Samples:   {args.num_samples}")
    print(f"{'='*50}")
    print(f"FID:       {fid:.3f}")
    print(f"IS:        {is_mean:.2f} ± {is_std:.2f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"{'='*50}")

    # Save results
    results = {
        "model": args.model,
        "cfg_scale": args.cfg_scale,
        "num_samples": args.num_samples,
        "fid": fid,
        "is_mean": is_mean,
        "is_std": is_std,
        "precision": precision,
        "recall": recall,
    }
    out_path = Path(args.output_dir) / "eval_results.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, out_path)
    print(f"Results saved to {out_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Hi-MAR")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet", "coco"])
    parser.add_argument("--model", type=str, default="Hi-MAR-B",
                        choices=["Hi-MAR-S", "Hi-MAR-B", "Hi-MAR-L", "Hi-MAR-H"])
    parser.add_argument("--data_path", type=str, default="/data/imagenet")
    parser.add_argument("--vae_path", type=str, default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--steps_phase1", type=int, default=32)
    parser.add_argument("--steps_phase2", type=int, default=4)
    parser.add_argument("--diff_steps", type=int, default=100)
    parser.add_argument("--cfg_scale", type=float, default=1.5)
    parser.add_argument("--output_dir", type=str, default="eval_outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
