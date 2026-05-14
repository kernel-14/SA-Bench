"""
evaluation.py

Evaluator class for computing FID, Inception Score, and Precision/Recall
on a list of generated image tensors.  Follows the standard protocols used
in the Hi‑MAR paper (Heusel et al., Salimans et al., Kynkäänniemi et al.).

Uses InceptionV3 features exactly as in `clean-fid` for consistency,
but implements the metrics manually to avoid file‑based API requirements.
Real‑dataset statistics are cached for efficiency.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as T
from PIL import Image
from scipy import linalg
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder  # kept for potential future use
from torchvision.models import inception_v3, Inception_V3_Weights
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Helper: image recursion dataset
# ---------------------------------------------------------------------------

class _RecursiveImageFolder(Dataset):
    """
    Recursively discovers all image files under a root directory and yields
    PIL images.  Suitable for COCO‑style directories without sub‑folder layout.
    """

    def __init__(self, root: str, transform: Optional[T.Compose] = None) -> None:
        super().__init__()
        self.transform = transform
        self.samples: List[str] = []
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append(os.path.join(dirpath, fname))
        if not self.samples:
            raise FileNotFoundError(f"No image files found under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tensor:
        path = self.samples[idx]
        pil_img = Image.open(path).convert("RGB")
        if self.transform is not None:
            return self.transform(pil_img)
        return T.ToTensor()(pil_img)  # fallback


# ---------------------------------------------------------------------------
#  Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Compute FID, IS, Precision/Recall for generated image tensors.

    Parameters
    ----------
    real_images_dir : str
        Path to the root directory of the real (training) images.
    cache_dir : str
        Directory where pre‑computed reference statistics will be saved / loaded.
    image_size : int
        Resolution of the generated images (e.g. 256).  Images are resized
        to 299×299 before Inception processing.
    batch_size : int
        Batch size for feature extraction.
    device : str
        Torch device to use (``"cuda"`` or ``"cpu"``).
    """

    def __init__(
        self,
        real_images_dir: str,
        cache_dir: str,
        image_size: int = 256,
        batch_size: int = 64,
        device: str = "cuda",
    ) -> None:
        if not os.path.isdir(real_images_dir):
            raise FileNotFoundError(f"Real images directory not found: {real_images_dir}")

        self.real_dir = real_images_dir
        self.cache_dir = cache_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

        # Inception preprocessing pipeline (standard for FID/IS)
        self.inception_transform = T.Compose([
            T.Resize((299, 299)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # Two InceptionV3 models:
        # - fid_model returns pool3 features (2048-D)
        # - is_model returns classification logits (1000-D)
        self.fid_model = inception_v3(
            weights=Inception_V3_Weights.DEFAULT, transform_input=True
        ).to(self.device)
        self.fid_model.fc = torch.nn.Identity()
        self.fid_model.eval()

        self.is_model = inception_v3(
            weights=Inception_V3_Weights.DEFAULT, transform_input=True
        ).to(self.device)
        self.is_model.eval()

        # Identify cache file via directory hash
        dir_hash = hashlib.md5(self.real_dir.encode()).hexdigest()
        self.stats_path = os.path.join(cache_dir, f"ref_stats_{dir_hash}.npz")

        # Lazy‑loaded reference data
        self._ref_mu: Optional[np.ndarray] = None
        self._ref_sigma: Optional[np.ndarray] = None
        self._ref_features_raw: Optional[np.ndarray] = None   # (N_real, 2048)

        # Pre‑computation may be triggered on first call
        self._stats_loaded = False

        logger.info("Evaluator initialized for dataset at %s", real_images_dir)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def compute_fid(self, generated_images: List[Tensor]) -> float:
        """Compute Fréchet Inception Distance between generated and real image features."""
        self._ensure_stats()

        feat_gen = self._extract_inception_features(generated_images, output_type="pool3")
        mu_gen = np.mean(feat_gen, axis=0)
        sigma_gen = np.cov(feat_gen, rowvar=False)

        mu_ref = self._ref_mu
        sigma_ref = self._ref_sigma

        diff = np.sum((mu_gen - mu_ref) ** 2)
        # Compute matrix square root of (sigma_gen @ sigma_ref)
        sqrtm_term = _safe_sqrtm(sigma_gen @ sigma_ref)
        trace = np.trace(sigma_gen + sigma_ref - 2.0 * sqrtm_term)
        fid = diff + trace
        return float(fid)

    def compute_is(self, generated_images: List[Tensor]) -> float:
        """Compute Inception Score for the generated images."""
        logits = self._extract_inception_features(generated_images, output_type="logits")
        # logits shape: (N, 1000)
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
        # marginal: p(y)
        p_y = np.mean(probs, axis=0, keepdims=True)
        # KL divergence per sample
        kl = np.sum(probs * (np.log(probs + 1e-8) - np.log(p_y + 1e-8)), axis=-1)
        is_mean = np.exp(np.mean(kl))
        return float(is_mean)

    def compute_precision_recall(self, generated_images: List[Tensor]) -> Tuple[float, float]:
        """Compute improved Precision and Recall (Kynkäänniemi et al.)."""
        self._ensure_stats()

        feat_gen = self._extract_inception_features(generated_images, output_type="pool3")
        feat_real = self._ref_features_raw  # (N_real, 2048), already loaded

        # Use k=3 for nearest neighbor radii, as common.
        k = 3

        # Radii of real manifold: distance to k-th nearest neighbor in real
        nn_real = NearestNeighbors(n_neighbors=k).fit(feat_real)
        dists_real, _ = nn_real.kneighbors(feat_real)
        r_real = dists_real[:, -1]   # (N_real,)

        # Precision: for each generated point, check if distance to nearest real <= r_real[nearest]
        nn_gen2real = NearestNeighbors(n_neighbors=1).fit(feat_real)
        dists_gen, indices = nn_gen2real.kneighbors(feat_gen)
        dists_gen = dists_gen[:, 0]          # (N_gen,)
        nearest_real_radius = r_real[indices[:, 0]]
        precision = float(np.mean(dists_gen <= nearest_real_radius))

        # Radii of generated manifold
        nn_gen = NearestNeighbors(n_neighbors=k).fit(feat_gen)
        dists_gen_k, _ = nn_gen.kneighbors(feat_gen)
        r_gen = dists_gen_k[:, -1]           # (N_gen,)

        # Recall: for each real point, check distance to nearest gen <= r_gen[nearest]
        nn_real2gen = NearestNeighbors(n_neighbors=1).fit(feat_gen)
        dists_real2gen, indices_r = nn_real2gen.kneighbors(feat_real)
        dists_real2gen = dists_real2gen[:, 0]
        nearest_gen_radius = r_gen[indices_r[:, 0]]
        recall = float(np.mean(dists_real2gen <= nearest_gen_radius))

        return precision, recall

    # ------------------------------------------------------------------ #
    #  Internal helpers – feature extraction
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _extract_inception_features(
        self,
        images: List[Tensor],
        output_type: str = "pool3",
    ) -> np.ndarray:
        """
        Convert a list of image tensors (C, H, W, values in [0, 1]) into Inception features.

        Parameters
        ----------
        images : List[Tensor]
            Each tensor is of shape ``(3, image_size, image_size)`` with values in [0, 1].
        output_type : str
            ``"pool3"`` (2048‑D) or ``"logits"`` (1000‑D).

        Returns
        -------
        np.ndarray
            Array of shape ``(N, feat_dim)``.
        """
        if output_type not in ("pool3", "logits"):
            raise ValueError(f"output_type must be 'pool3' or 'logits', got {output_type}")

        # Convert tensors to PIL images
        pil_images = []
        for t in images:
            # Ensure CHW format and [0,1] range.
            img = t.detach().cpu()
            if img.ndim != 3 or img.shape[0] != 3:
                raise ValueError(f"Expected (3,H,W) tensor, got shape {img.shape}")
            img = (img.clamp(0.0, 1.0) * 255).round().byte()
            pil_img = Image.fromarray(img.permute(1, 2, 0).numpy(), mode="RGB")
            pil_images.append(pil_img)

        # Create a simple dataset that applies the Inception transform
        class ImageListDataset(Dataset):
            def __init__(self, imgs, transform):
                self.imgs = imgs
                self.transform = transform

            def __len__(self):
                return len(self.imgs)

            def __getitem__(self, idx):
                return self.transform(self.imgs[idx])

        dataset = ImageListDataset(pil_images, self.inception_transform)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        model = self.fid_model if output_type == "pool3" else self.is_model
        features = []
        for batch in tqdm(loader, desc=f"Extracting {output_type} features"):
            batch = batch.to(self.device, non_blocking=True)
            output = model(batch)
            features.append(output.cpu().numpy())

        return np.concatenate(features, axis=0)

    # ------------------------------------------------------------------ #
    #  Real‑dataset statistics handling
    # ------------------------------------------------------------------ #

    def _ensure_stats(self) -> None:
        """Load or compute reference stats and store in memory."""
        if self._stats_loaded:
            return

        if os.path.exists(self.stats_path):
            logger.info("Loading cached reference stats from %s", self.stats_path)
            data = np.load(self.stats_path, allow_pickle=True)
            self._ref_mu = data["mu"]
            self._ref_sigma = data["sigma"]
            self._ref_features_raw = data["features"]
            self._stats_loaded = True
        else:
            logger.info("Cached stats not found. Computing reference statistics...")
            self._compute_real_stats()
            self._stats_loaded = True

    def _compute_real_stats(self) -> None:
        """Compute and persist mu, sigma, and a subset of features for PR."""
        # Gather all real images
        logger.info("Loading real images from %s", self.real_dir)

        # Determine dataset type: if subfolders exist -> ImageNet-style,
        # otherwise use recursive listing.
        if any(os.path.isdir(os.path.join(self.real_dir, d))
               for d in os.listdir(self.real_dir)):
            # Use ImageFolder (assumes class subdirectories)
            dataset = ImageFolder(
                root=self.real_dir,
                transform=self.inception_transform,
            )
        else:
            dataset = _RecursiveImageFolder(
                root=self.real_dir,
                transform=self.inception_transform,
            )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        all_features = []
        for batch in tqdm(loader, desc="Extracting real pool3 features"):
            if isinstance(batch, (tuple, list)):
                batch = batch[0]  # ImageFolder returns (image, label)
            batch = batch.to(self.device, non_blocking=True)
            with torch.no_grad():
                features = self.fid_model(batch).cpu().numpy()
            all_features.append(features)

        all_features = np.concatenate(all_features, axis=0)
        N = all_features.shape[0]
        logger.info("Collected %d real feature vectors", N)

        # Compute FID statistics (mu, sigma)
        mu = np.mean(all_features, axis=0)
        sigma = np.cov(all_features, rowvar=False)

        # For Precision/Recall, store a subset if too large (50k).
        # We use the full set if it's under 100k, otherwise sample 50k.
        if N <= 100_000:
            features_subset = all_features
        else:
            logger.info("Sampling 50,000 real features for PR evaluation")
            indices = np.random.default_rng(42).choice(N, size=50_000, replace=False)
            features_subset = all_features[indices]

        # Save to cache file
        np.savez_compressed(
            self.stats_path,
            mu=mu,
            sigma=sigma,
            features=features_subset,
        )
        logger.info("Reference statistics saved to %s", self.stats_path)

        # Set internal caches
        self._ref_mu = mu
        self._ref_sigma = sigma
        self._ref_features_raw = features_subset


# ---------------------------------------------------------------------------
#  Utility: safe matrix square root
# ---------------------------------------------------------------------------

def _safe_sqrtm(mat: np.ndarray) -> np.ndarray:
    """
    Compute the matrix square root using scipy.linalg.sqrtm and fall back to
    eigenvalue clipping for numerical stability.
    """
    # If matrix contains NaN/Inf (should not happen), return zeros
    if not np.all(np.isfinite(mat)):
        logger.warning("Non-finite values in matrix for sqrtm; returning zero matrix.")
        return np.zeros_like(mat)

    try:
        # scipy.linalg.sqrtm returns a tuple (T, error) in some versions; handle
        result = linalg.sqrtm(mat)
        if isinstance(result, tuple):
            T = result[0]
        else:
            T = result
        if np.iscomplexobj(T):
            logger.warning("sqrtm produced complex values; taking real part.")
            T = T.real
        return T
    except Exception as e:
        logger.warning("sqrtm failed (%s). Falling back to eigenvalue clipping.", e)
        eigvals, eigvecs = linalg.eigh(mat)
        eigvals = np.clip(eigvals, a_min=0, a_max=None)
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


