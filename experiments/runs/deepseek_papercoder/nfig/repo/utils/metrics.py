"""
utils/metrics.py

Evaluation metrics for the NFIG reproduction project: Fréchet Inception Distance (FID),
Inception Score (IS), and Precision/Recall (k‑NN manifold).

All functions operate on PyTorch tensors and are designed to be GPU‑friendly.
The module is self‑contained and does not require internal project dependencies.
It relies on torchvision's pretrained InceptionV3 for IS, while FID and Precision/Recall
accept pre‑extracted feature vectors (typically 2048‑dim Inception pool3 features).
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from typing import Tuple


def compute_fid(real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
    """
    Compute the Fréchet Inception Distance (FID) between two sets of features.

    The features should come from the same feature extractor (e.g., InceptionV3
    pool3 layer, 2048 dimensions).  The covariance matrices are computed with
    population covariance (ddof=0) to match the standard pytorch‑fid implementation.

    Numerical stability is ensured by using double precision for covariance and
    eigenvalue decomposition.

    Args:
        real_features: Tensor of shape (N_real, D) with real image features.
        fake_features: Tensor of shape (N_fake, D) with generated image features.

    Returns:
        Scalar FID value (float).
    """
    assert real_features.ndim == 2 and fake_features.ndim == 2, \
        "Features must be 2‑dimensional (N, D)."
    assert real_features.size(1) == fake_features.size(1), \
        "Feature dimensions must match."

    # Convert to double for numerical stability
    real = real_features.double()
    fake = fake_features.double()

    # Means
    mu_real = real.mean(dim=0)
    mu_fake = fake.mean(dim=0)
    diff = torch.sum((mu_real - mu_fake) ** 2)

    # Covariance matrices (population covariance, ddof=0)
    sigma_real = torch.cov(real.T, correction=0)
    sigma_fake = torch.cov(fake.T, correction=0)

    # Product of covariances: σ_real · σ_fake
    cov_product = sigma_real @ sigma_fake

    # Matrix square root via eigendecomposition; the product is symmetric in theory.
    # To handle possible tiny negative eigenvalues, we clamp to zero.
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_product)
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    cov_mean = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T

    # FID formula
    fid = diff + torch.trace(sigma_real + sigma_fake - 2 * cov_mean)
    return fid.item()


def compute_is(images: torch.Tensor, batch_size: int = 32) -> float:
    """
    Compute the Inception Score (IS) for a set of generated images.

    The images are assumed to be normalized to [-1, 1] (the standard range
    used throughout the NFIG pipeline).  They are automatically preprocessed
    to match the InceptionV3 input requirements (resize/center‑crop to 299×299,
    normalization using standard ImageNet statistics).

    Args:
        images: Tensor of shape (N, 3, H, W), pixel values in [-1, 1].
        batch_size: Number of images to process at once; reduces GPU memory usage.

    Returns:
        Inception Score (higher is better).
    """
    device = images.device
    n = images.size(0)

    # ------------------------------------------------------------------
    # 1. Prepare Inception preprocessing pipeline
    # ------------------------------------------------------------------
    # Convert from [-1, 1] to [0, 1]
    def unnorm_to_01(batch):
        return (batch + 1.0) / 2.0

    # Standard Inception normalization
    img_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Full transform: resize → center‑crop → normalize
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # ------------------------------------------------------------------
    # 2. Load pretrained InceptionV3 (aux_logits=False for eval mode)
    # ------------------------------------------------------------------
    model = inception_v3(pretrained=True, transform_input=False, aux_logits=False)
    model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # 3. Process images in batches
    # ------------------------------------------------------------------
    all_probs = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = images[start:end]

            # Clamp to valid range before conversion (just in case)
            batch = torch.clamp(batch, -1.0, 1.0)
            batch = unnorm_to_01(batch)

            # Apply Inception preprocessing (Resize/CenterCrop/Normalize)
            # The compose expects images in [0,1]
            batch_proc = preprocess(batch)

            # Forward through Inception
            logits = model(batch_proc)                     # (B, 1000)
            probs = F.softmax(logits, dim=1)               # (B, 1000)
            all_probs.append(probs.detach().cpu())

    # Stack all probabilities
    all_probs = torch.cat(all_probs, dim=0)                # (N, 1000)

    # ------------------------------------------------------------------
    # 4. Compute Inception Score
    # ------------------------------------------------------------------
    # Marginal distribution over classes
    p_y = all_probs.mean(dim=0, keepdim=True)              # (1, 1000)

    # KL divergence between conditional and marginal
    log_p_yx = torch.log(all_probs)                        # (N, 1000)
    log_p_y = torch.log(p_y)                               # (1, 1000) → broadcast
    kl = all_probs * (log_p_yx - log_p_y)                  # (N, 1000)
    kl_per_sample = kl.sum(dim=1)                          # (N,)

    # Inception Score = exp( mean KL )
    is_score = torch.exp(kl_per_sample.mean())             # scalaar
    return is_score.item()


def compute_precision_recall(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    k: int = 3
) -> Tuple[float, float]:
    """
    Compute improved Precision and Recall using the k‑NN manifold method.

    For each sample, the distance to its k‑th nearest neighbour within the
    same set (excluding itself) defines a manifold radius.  Precision is the
    fraction of fake samples that fall within the radii of their nearest real
    neighbours; Recall is the fraction of real samples that fall within the
    radii of their nearest fake neighbours.

    Large feature sets are handled by chunking to avoid O(n²) memory overhead.

    Args:
        real_features: Tensor of shape (N_real, D) of real image features.
        fake_features: Tensor of shape (N_fake, D) of generated image features.
        k: Neighbourhood size for manifold radius (default 3).

    Returns:
        A tuple (precision, recall) as floats.
    """
    N_real, D = real_features.shape
    N_fake, _ = fake_features.shape
    device = real_features.device

    # Chunk size to limit distance matrix memory (adjust based on GPU memory)
    CHUNK_SIZE = 1000

    # ------------------------------------------------------------------
    # 1. Compute manifold radii for real samples
    # ------------------------------------------------------------------
    radii_real = torch.empty(N_real, device=device)

    for start in range(0, N_real, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, N_real)
        chunk_real = real_features[start:end]               # (c, D)
        # Distance from chunk to all real features
        dist = torch.cdist(chunk_real, real_features)       # (c, N_real)

        # Set self‑distances to infinity (diagonal entries within this chunk)
        rows = torch.arange(start, end, device=device)
        dist[rows - start, rows] = float('inf')

        # Find k‑th smallest distance per row
        topk_vals, _ = torch.topk(dist, k, dim=1, largest=False)  # (c, k)
        radii_real[start:end] = topk_vals[:, -1]                  # k‑th value

    # ------------------------------------------------------------------
    # 2. Compute manifold radii for fake samples
    # ------------------------------------------------------------------
    radii_fake = torch.empty(N_fake, device=device)

    for start in range(0, N_fake, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, N_fake)
        chunk_fake = fake_features[start:end]
        dist = torch.cdist(chunk_fake, fake_features)
        rows = torch.arange(start, end, device=device)
        dist[rows - start, rows] = float('inf')
        topk_vals, _ = torch.topk(dist, k, dim=1, largest=False)
        radii_fake[start:end] = topk_vals[:, -1]

    # ------------------------------------------------------------------
    # 3. Precision: fraction of fakes within real manifold
    # ------------------------------------------------------------------
    precision_count = 0

    for start in range(0, N_fake, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, N_fake)
        chunk_fake = fake_features[start:end]                # (c, D)
        dist_f2r = torch.cdist(chunk_fake, real_features)    # (c, N_real)

        # For each fake sample, find nearest real sample and its radius
        min_dist, min_idx = torch.min(dist_f2r, dim=1)      # c
        # Check if min_dist <= radius of that nearest real
        inside = min_dist <= radii_real[min_idx]             # c bool
        precision_count += inside.sum().item()

    precision = precision_count / N_fake

    # ------------------------------------------------------------------
    # 4. Recall: fraction of reals within fake manifold
    # ------------------------------------------------------------------
    recall_count = 0

    for start in range(0, N_real, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, N_real)
        chunk_real = real_features[start:end]                # (c, D)
        dist_r2f = torch.cdist(chunk_real, fake_features)    # (c, N_fake)

        min_dist, min_idx = torch.min(dist_r2f, dim=1)
        inside = min_dist <= radii_fake[min_idx]
        recall_count += inside.sum().item()

    recall = recall_count / N_real

    return (precision, recall)

