"""
Evaluation script for NFIG.

Computes FID, IS, Precision, and Recall on ImageNet 256x256.
Also computes reconstruction FID (rFID) for the FR-VAE tokenizer.

Metrics:
- FID: Fréchet Inception Distance (lower is better)
- IS: Inception Score (higher is better)
- Precision: sample fidelity (higher is better)
- Recall: diversity coverage (higher is better)
- rFID: reconstruction FID for tokenizer quality
"""

import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from config import NFIGConfig, config_600m
from data import build_imagenet_dataset, build_dataloader
from models.fr_vae import FRVAE
from models.transformer import NFIGTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NFIG")
    parser.add_argument("--mode", type=str, choices=["rfid", "gfid", "both"], default="both")
    parser.add_argument("--tokenizer-ckpt", type=str, required=True)
    parser.add_argument("--transformer-ckpt", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="/data/imagenet")
    parser.add_argument("--generated-dir", type=str, default=None,
                        help="Directory of pre-generated images (skip generation)")
    parser.add_argument("--fid-stats-path", type=str, default="./fid_stats/imagenet256.npz")
    parser.add_argument("--model-size", type=str, default="310M", choices=["310M", "600M"])
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=4.5)
    parser.add_argument("--top-k", type=int, default=990)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


class GeneratedImageDataset(Dataset):
    """Dataset for loading pre-generated images from a directory."""

    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self.paths = []
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.paths.append(os.path.join(dirpath, fname))
        self.paths.sort()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def compute_inception_features(
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 50000,
) -> np.ndarray:
    """Extract Inception-v3 features for FID computation."""
    try:
        from torchvision.models import inception_v3
    except ImportError:
        raise ImportError("torchvision required for FID computation")

    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()
    # Remove final classification layer
    inception.fc = nn.Identity()

    features = []
    n_collected = 0

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                imgs = batch[0]
            else:
                imgs = batch
            imgs = imgs.to(device)

            # Resize to 299x299 for Inception
            imgs_resized = nn.functional.interpolate(
                imgs, size=(299, 299), mode="bilinear", align_corners=False
            )
            # Normalize to [0, 1] if in [-1, 1]
            if imgs_resized.min() < 0:
                imgs_resized = imgs_resized * 0.5 + 0.5

            feat = inception(imgs_resized)
            features.append(feat.cpu().numpy())
            n_collected += imgs.shape[0]
            if n_collected >= max_samples:
                break

    return np.concatenate(features, axis=0)[:max_samples]


def compute_fid(
    features_real: np.ndarray,
    features_fake: np.ndarray,
) -> float:
    """Compute FID between two sets of Inception features."""
    from scipy import linalg

    mu1, sigma1 = features_real.mean(axis=0), np.cov(features_real, rowvar=False)
    mu2, sigma2 = features_fake.mean(axis=0), np.cov(features_fake, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def compute_inception_score(
    loader: DataLoader,
    device: torch.device,
    num_splits: int = 10,
    max_samples: int = 50000,
) -> Tuple[float, float]:
    """Compute Inception Score (IS)."""
    try:
        from torchvision.models import inception_v3
    except ImportError:
        raise ImportError("torchvision required for IS computation")

    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()

    probs_list = []
    n_collected = 0

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                imgs = batch[0]
            else:
                imgs = batch
            imgs = imgs.to(device)

            imgs_resized = nn.functional.interpolate(
                imgs, size=(299, 299), mode="bilinear", align_corners=False
            )
            if imgs_resized.min() < 0:
                imgs_resized = imgs_resized * 0.5 + 0.5

            logits = inception(imgs_resized)
            probs = torch.softmax(logits, dim=-1)
            probs_list.append(probs.cpu().numpy())
            n_collected += imgs.shape[0]
            if n_collected >= max_samples:
                break

    probs = np.concatenate(probs_list, axis=0)[:max_samples]
    n = probs.shape[0]
    split_size = n // num_splits

    scores = []
    for i in range(num_splits):
        p = probs[i * split_size:(i + 1) * split_size]
        p_marginal = p.mean(axis=0, keepdims=True)
        kl = p * (np.log(p + 1e-10) - np.log(p_marginal + 1e-10))
        scores.append(np.exp(kl.sum(axis=1).mean()))

    return float(np.mean(scores)), float(np.std(scores))


def compute_precision_recall(
    features_real: np.ndarray,
    features_fake: np.ndarray,
    k: int = 3,
) -> Tuple[float, float]:
    """
    Compute Precision and Recall using k-NN manifold estimation.
    Based on Kynkäänniemi et al. (2019).
    """
    from sklearn.neighbors import NearestNeighbors

    # Fit k-NN on real features
    nn_real = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    nn_real.fit(features_real)
    real_distances, _ = nn_real.kneighbors(features_real)
    real_radii = real_distances[:, -1]  # k-th neighbor distance

    # Fit k-NN on fake features
    nn_fake = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    nn_fake.fit(features_fake)
    fake_distances, _ = nn_fake.kneighbors(features_fake)
    fake_radii = fake_distances[:, -1]

    # Precision: fraction of fake samples in real manifold
    distances_fake_to_real, _ = nn_real.kneighbors(features_fake, n_neighbors=1)
    precision = (distances_fake_to_real[:, 0] <= real_radii[
        nn_real.kneighbors(features_fake, n_neighbors=1)[1][:, 0]
    ]).mean()

    # Recall: fraction of real samples in fake manifold
    distances_real_to_fake, _ = nn_fake.kneighbors(features_real, n_neighbors=1)
    recall = (distances_real_to_fake[:, 0] <= fake_radii[
        nn_fake.kneighbors(features_real, n_neighbors=1)[1][:, 0]
    ]).mean()

    return float(precision), float(recall)


@torch.no_grad()
def evaluate_reconstruction_fid(
    tokenizer: FRVAE,
    data_root: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    num_samples: int = 50000,
) -> float:
    """Compute reconstruction FID (rFID) for the FR-VAE tokenizer."""
    val_dataset = build_imagenet_dataset(data_root, 256, "val")
    val_loader, _ = build_dataloader(val_dataset, batch_size, num_workers, is_train=False)

    tokenizer.eval()
    real_features = []
    rec_features = []

    try:
        from torchvision.models import inception_v3
        inception = inception_v3(pretrained=True, transform_input=False).to(device)
        inception.eval()
        inception.fc = nn.Identity()
    except ImportError:
        raise ImportError("torchvision required for rFID computation")

    n_collected = 0
    for images, _ in val_loader:
        images = images.to(device)

        # Reconstruct
        with torch.no_grad():
            x_rec, _, _, _, _, _ = tokenizer(images)
            x_rec = x_rec.clamp(-1, 1)

        # Extract features
        def extract_features(imgs):
            imgs_r = nn.functional.interpolate(imgs, size=(299, 299), mode="bilinear", align_corners=False)
            if imgs_r.min() < 0:
                imgs_r = imgs_r * 0.5 + 0.5
            return inception(imgs_r).cpu().numpy()

        real_features.append(extract_features(images))
        rec_features.append(extract_features(x_rec))
        n_collected += images.shape[0]
        if n_collected >= num_samples:
            break

    real_features = np.concatenate(real_features, axis=0)[:num_samples]
    rec_features = np.concatenate(rec_features, axis=0)[:num_samples]

    rfid = compute_fid(real_features, rec_features)
    return rfid


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = NFIGConfig()

    # Load tokenizer
    tokenizer = FRVAE(
        image_size=cfg.tokenizer.image_size,
        in_channels=cfg.tokenizer.in_channels,
        z_channels=cfg.tokenizer.z_channels,
        ch=cfg.tokenizer.ch,
        ch_mult=cfg.tokenizer.ch_mult,
        num_res_blocks=cfg.tokenizer.num_res_blocks,
        attn_resolutions=cfg.tokenizer.attn_resolutions,
        codebook_size=cfg.tokenizer.codebook_size,
        scale_factors=cfg.tokenizer.scale_factors,
        feature_map_size=cfg.tokenizer.feature_map_size,
    ).to(device)
    tokenizer.load_state_dict(torch.load(args.tokenizer_ckpt, map_location=device))
    tokenizer.eval()

    results = {}

    # Reconstruction FID
    if args.mode in ("rfid", "both"):
        print("Computing reconstruction FID...")
        rfid = evaluate_reconstruction_fid(
            tokenizer=tokenizer,
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            num_samples=args.num_samples,
        )
        results["rFID"] = rfid
        print(f"rFID: {rfid:.4f}")

    # Generation FID
    if args.mode in ("gfid", "both"):
        if args.generated_dir is None and args.transformer_ckpt is None:
            print("Either --generated-dir or --transformer-ckpt required for gFID evaluation")
            return

        # Load real image features
        print("Extracting real image features...")
        val_dataset = build_imagenet_dataset(args.data_root, 256, "val")
        val_loader, _ = build_dataloader(
            val_dataset, args.batch_size, args.num_workers, is_train=False
        )
        real_features = compute_inception_features(val_loader, device, args.num_samples)

        # Load or generate fake images
        if args.generated_dir is not None:
            print(f"Loading generated images from {args.generated_dir}...")
            gen_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            gen_dataset = GeneratedImageDataset(args.generated_dir, gen_transform)
            gen_loader = DataLoader(
                gen_dataset, batch_size=args.batch_size, num_workers=args.num_workers
            )
            fake_features = compute_inception_features(gen_loader, device, args.num_samples)
            is_mean, is_std = compute_inception_score(gen_loader, device, max_samples=args.num_samples)
        else:
            # Generate images on the fly
            from generate import load_transformer, generate_images

            print("Loading transformer...")
            transformer = load_transformer(args.transformer_ckpt, cfg, args.model_size, device)

            print("Generating images...")
            all_fake_images = []
            n_generated = 0
            class_idx = 0

            while n_generated < args.num_samples:
                batch_size = min(args.batch_size, args.num_samples - n_generated)
                labels = torch.tensor(
                    [class_idx % 1000] * batch_size, dtype=torch.long
                )
                imgs = generate_images(
                    transformer=transformer,
                    tokenizer=tokenizer,
                    class_labels=labels,
                    cfg_scale=args.cfg_scale,
                    top_k=args.top_k,
                    device=device,
                )
                all_fake_images.append(imgs.cpu())
                n_generated += batch_size
                class_idx += batch_size

            fake_tensor = torch.cat(all_fake_images, dim=0)[:args.num_samples]

            # Extract features from generated images
            from torch.utils.data import TensorDataset
            fake_dataset = TensorDataset(fake_tensor)
            fake_loader = DataLoader(fake_dataset, batch_size=args.batch_size)
            fake_features = compute_inception_features(fake_loader, device, args.num_samples)
            is_mean, is_std = compute_inception_score(fake_loader, device, max_samples=args.num_samples)

        gfid = compute_fid(real_features, fake_features)
        precision, recall = compute_precision_recall(real_features, fake_features)

        results["gFID"] = gfid
        results["IS_mean"] = is_mean
        results["IS_std"] = is_std
        results["Precision"] = precision
        results["Recall"] = recall

        print(f"gFID: {gfid:.4f}")
        print(f"IS: {is_mean:.2f} ± {is_std:.2f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")

    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()
