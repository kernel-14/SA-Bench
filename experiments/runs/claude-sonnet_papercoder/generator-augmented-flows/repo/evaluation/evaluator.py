## evaluation/evaluator.py
"""Evaluation pipeline for iCT-GC consistency model experiments.

This module implements the ``Evaluator`` class, which computes the three
evaluation metrics reported in the paper: FID, KID (×10²), and IS.

The evaluation protocol follows Appendix D exactly:
    - Compare 50,000 generated images vs 50,000 training images
    - Run 5 independent evaluation runs for confidence intervals
    - Use TorchMetrics implementations for all three metrics
    - Use single-step generation (one NFE — the core consistency model property)

Confidence intervals (Table 1): "averaged on five runs by sampling new sets
of training images, and new sets of generated images from the same model."

Config values used (from config.yaml defaults):
    sigma_max:        80.0     (sigma_T for single-step generation)
    num_eval_samples: 50000    (images per evaluation run)
    num_eval_runs:    5        (runs for confidence intervals)
    batch_size:       512      (generation batch size)
    device:           cuda
    image_size:       32       (spatial resolution)
    in_channels:      3        (RGB)

Typical usage::

    evaluator = Evaluator(config, model, real_loader)

    # During training (fast, single run):
    metrics = evaluator.evaluate(num_samples=10000, num_runs=1)

    # Final evaluation (full protocol):
    metrics = evaluator.evaluate(num_samples=50000, num_runs=5)
    print(f"FID: {metrics['fid_mean']:.2f} ± {metrics['fid_std']:.2f}")
    print(f"KID: {metrics['kid_mean']:.2f} ± {metrics['kid_std']:.2f}")
    print(f"IS:  {metrics['is_mean']:.2f} ± {metrics['is_std']:.2f}")
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore

from models.consistency_model import ConsistencyModel
from utils.helpers import normalize_images


class Evaluator:
    """Computes FID, KID (×10²), and IS for a trained consistency model.

    Implements the evaluation protocol from Appendix D of the paper:
    50,000 generated images vs 50,000 training images, averaged over 5
    independent runs to produce mean ± std confidence intervals.

    All three metrics use TorchMetrics with ``normalize=True``, which
    accepts float tensors in ``[0, 1]`` rather than uint8 in ``[0, 255]``.

    The model is used in single-step generation mode (one NFE):
        x_0_hat = f_θ(σ_T · z, σ_T),  z ~ N(0, I)

    This is the core property of consistency models that distinguishes them
    from multi-step diffusion models.

    Attributes:
        config: Configuration object with evaluation hyperparameters.
        model: Trained ConsistencyModel for single-step image generation.
        real_loader: DataLoader over the training set for real image features.
        device: Target device for model inference and metric computation.
        fid: TorchMetrics FrechetInceptionDistance instance.
        kid: TorchMetrics KernelInceptionDistance instance.
        is_metric: TorchMetrics InceptionScore instance.
        sigma_max: Maximum noise level σ_T for generation (default 80.0).
        gen_batch_size: Batch size used during image generation.
        image_size: Spatial resolution of generated images.
        in_channels: Number of image channels (3 for RGB).
        num_eval_samples: Number of images per evaluation run (default 50000).
        num_eval_runs: Number of runs for confidence intervals (default 5).
    """

    def __init__(
        self,
        config: Any,
        model: ConsistencyModel,
        real_loader: DataLoader,
    ) -> None:
        """Initialise the evaluator with model, data loader, and metrics.

        Instantiates TorchMetrics objects and moves them to the target device.
        The model is moved to eval mode but not modified structurally.

        Args:
            config: Configuration object. Must expose the following attributes
                (all present in config.yaml):
                - ``device`` (str): 'cuda' or 'cpu'.
                - ``sigma_max`` (float): Maximum noise level σ_T (default 80.0).
                - ``batch_size`` (int): Batch size for generation (e.g. 512).
                - ``image_size`` (int): Spatial resolution (e.g. 32 or 64).
                - ``in_channels`` (int): Number of image channels (default 3).
                - ``num_eval_samples`` (int): Images per run (default 50000).
                - ``num_eval_runs`` (int): Number of runs (default 5).
            model: Trained ``ConsistencyModel`` instance. Must implement
                ``generate(z, sigma_T)`` for single-step synthesis.
            real_loader: DataLoader yielding ``(images, labels)`` tuples
                where images are in ``[-1, 1]`` (from the training transform).
                Used to provide real image statistics for FID and KID.

        Raises:
            TypeError: If ``model`` is not a ``ConsistencyModel`` instance.
        """
        if not isinstance(model, ConsistencyModel):
            raise TypeError(
                f"Expected 'model' to be a ConsistencyModel instance, "
                f"got {type(model).__name__}."
            )

        self.config: Any = config
        self.model: ConsistencyModel = model
        self.real_loader: DataLoader = real_loader

        # --- Device ---
        self.device: torch.device = torch.device(
            str(getattr(config, "device", "cuda"))
        )

        # --- Cached hyperparameters from config ---
        self.sigma_max: float = float(getattr(config, "sigma_max", 80.0))
        self.gen_batch_size: int = int(getattr(config, "batch_size", 256))
        self.image_size: int = int(getattr(config, "image_size", 32))
        self.in_channels: int = int(getattr(config, "in_channels", 3))
        self.num_eval_samples: int = int(
            getattr(config, "num_eval_samples", 50000)
        )
        self.num_eval_runs: int = int(getattr(config, "num_eval_runs", 5))

        # --- Move model to device and set to eval mode ---
        self.model = self.model.to(self.device)
        self.model.eval()

        # --- KID subset_size: must be <= num_eval_samples ---
        # Standard choice is 1000 for stability with 50k images.
        # Clamp to num_eval_samples in case of small evaluation sets.
        kid_subset_size: int = min(1000, self.num_eval_samples)

        # --- TorchMetrics instances ---
        # normalize=True: accepts float tensors in [0, 1] (not uint8 [0, 255])
        # feature=2048: use the final Inception-v3 pooling layer (standard for FID)
        self.fid: FrechetInceptionDistance = FrechetInceptionDistance(
            feature=2048,
            normalize=True,
        ).to(self.device)

        self.kid: KernelInceptionDistance = KernelInceptionDistance(
            subset_size=kid_subset_size,
            normalize=True,
        ).to(self.device)

        self.is_metric: InceptionScore = InceptionScore(
            normalize=True,
        ).to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        num_samples: int = 50000,
        num_runs: int = 5,
    ) -> Dict[str, float]:
        """Run the full evaluation protocol and return aggregated metrics.

        Implements the paper's evaluation protocol (Appendix D):
        - Generate ``num_samples`` images per run using single-step synthesis
        - Compare against ``num_samples`` real training images
        - Repeat ``num_runs`` times with fresh random samples each run
        - Return mean ± std across runs

        The model is set to eval mode at the start and restored to train mode
        at the end, so this method is safe to call during training.

        Args:
            num_samples: Number of generated and real images per run.
                Default 50,000 matches the paper's protocol. Can be reduced
                (e.g. 10,000) for faster intermediate evaluations during
                training.
            num_runs: Number of independent evaluation runs. Default 5
                matches the paper's confidence interval protocol. Use 1
                for fast intermediate evaluations (std will be 0.0).

        Returns:
            Dictionary with the following keys and float values:
                - ``'fid_mean'``: Mean FID across runs (lower is better).
                - ``'fid_std'``: Std of FID across runs.
                - ``'kid_mean'``: Mean KID ×10² across runs (lower is better).
                - ``'kid_std'``: Std of KID ×10² across runs.
                - ``'is_mean'``: Mean Inception Score across runs (higher is better).
                - ``'is_std'``: Std of IS across runs.

        Example::

            metrics = evaluator.evaluate(num_samples=50000, num_runs=5)
            # Expected for iCT-GC (μ=0.5) on CIFAR-10 (Table 1):
            # FID: 5.95 ± 0.05
            # KID: 0.26 ± 0.02  (×10²)
            # IS:  9.10 ± 0.05
        """
        # Clamp num_samples to a positive value
        num_samples = max(1, num_samples)
        num_runs = max(1, num_runs)

        # Adjust KID subset_size if num_samples is smaller than the default
        kid_subset_size: int = min(1000, num_samples)
        if self.kid.subset_size != kid_subset_size:
            # Re-instantiate KID with the correct subset_size
            self.kid = KernelInceptionDistance(
                subset_size=kid_subset_size,
                normalize=True,
            ).to(self.device)

        # Set model to eval mode for generation
        was_training: bool = self.model.training
        self.model.eval()

        fid_scores: List[float] = []
        kid_mean_scores: List[float] = []
        kid_std_scores: List[float] = []
        is_mean_scores: List[float] = []
        is_std_scores: List[float] = []

        try:
            for run_idx in range(num_runs):
                print(
                    f"[Evaluator] Run {run_idx + 1}/{num_runs}: "
                    f"generating {num_samples} images..."
                )

                # --- Generate fake images ---
                # Shape: (num_samples, C, H, W) in [0, 1], on CPU
                fake_images: torch.Tensor = self.generate_samples(num_samples)

                # --- Reset all metrics before each run ---
                # TorchMetrics accumulates state; must reset between runs.
                self.fid.reset()
                self.kid.reset()
                self.is_metric.reset()

                # --- Feed real images to FID and KID ---
                # Each run uses a fresh (potentially different) subset of real
                # images, matching the paper's protocol of "sampling new sets
                # of training images" per run.
                self._update_real_features(num_samples=num_samples)

                # --- Feed fake images to all three metrics ---
                self._update_fake_features(fake_images)

                # --- Compute metrics ---
                fid_val: float = self.compute_fid(fake_images)
                kid_mean_val: float
                kid_std_val: float
                kid_mean_val, kid_std_val = self.compute_kid(fake_images)
                is_mean_val: float
                is_std_val: float
                is_mean_val, is_std_val = self.compute_is(fake_images)

                fid_scores.append(fid_val)
                kid_mean_scores.append(kid_mean_val)
                kid_std_scores.append(kid_std_val)
                is_mean_scores.append(is_mean_val)
                is_std_scores.append(is_std_val)

                print(
                    f"[Evaluator] Run {run_idx + 1}/{num_runs}: "
                    f"FID={fid_val:.4f}, "
                    f"KID={kid_mean_val:.4f}(×10²), "
                    f"IS={is_mean_val:.4f}"
                )

        finally:
            # Always restore model training mode
            if was_training:
                self.model.train()

        # --- Aggregate across runs ---
        # For a single run, std is 0.0 (not NaN)
        fid_arr: np.ndarray = np.array(fid_scores, dtype=np.float64)
        kid_mean_arr: np.ndarray = np.array(kid_mean_scores, dtype=np.float64)
        is_mean_arr: np.ndarray = np.array(is_mean_scores, dtype=np.float64)

        fid_mean: float = float(np.mean(fid_arr))
        fid_std: float = float(np.std(fid_arr)) if num_runs > 1 else 0.0

        kid_mean: float = float(np.mean(kid_mean_arr))
        kid_std: float = float(np.std(kid_mean_arr)) if num_runs > 1 else 0.0

        is_mean: float = float(np.mean(is_mean_arr))
        is_std: float = float(np.std(is_mean_arr)) if num_runs > 1 else 0.0

        results: Dict[str, float] = {
            "fid_mean": fid_mean,
            "fid_std": fid_std,
            "kid_mean": kid_mean,
            "kid_std": kid_std,
            "is_mean": is_mean,
            "is_std": is_std,
        }

        print(
            f"[Evaluator] Final results ({num_runs} run(s)):\n"
            f"  FID: {fid_mean:.4f} ± {fid_std:.4f}\n"
            f"  KID (×10²): {kid_mean:.4f} ± {kid_std:.4f}\n"
            f"  IS:  {is_mean:.4f} ± {is_std:.4f}"
        )

        return results

    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """Generate images using single-step consistency model synthesis.

        Generates ``num_samples`` images by sampling noise ``z ~ N(0, I)``
        and applying the consistency model in a single forward pass:
            x_0_hat = f_θ(σ_T · z, σ_T)

        This is the core property of consistency models: one NFE (neural
        function evaluation) maps noise directly to data.

        The output is returned on CPU to conserve GPU memory. TorchMetrics
        will move tensors to the metric device internally.

        Args:
            num_samples: Total number of images to generate. Must be positive.
                Typically 50,000 for final evaluation or 10,000 for
                intermediate monitoring.

        Returns:
            Float tensor of shape ``(num_samples, C, H, W)`` with values
            in ``[0, 1]``. On CPU.
        """
        all_samples: List[torch.Tensor] = []
        generated: int = 0

        with torch.no_grad():
            pbar = tqdm(
                total=num_samples,
                desc="Generating",
                unit="img",
                leave=False,
                dynamic_ncols=True,
            )

            while generated < num_samples:
                # Determine batch size for this iteration
                remaining: int = num_samples - generated
                current_batch_size: int = min(self.gen_batch_size, remaining)

                # Sample standard Gaussian noise: z ~ N(0, I)
                z: torch.Tensor = torch.randn(
                    current_batch_size,
                    self.in_channels,
                    self.image_size,
                    self.image_size,
                    device=self.device,
                )

                # Single-step generation: x_0_hat = f_θ(σ_T · z, σ_T)
                # Returns (B, C, H, W) in [0, 1] after clipping and mapping
                samples: torch.Tensor = self._single_step_sample(z)

                # Move to CPU to conserve GPU memory
                all_samples.append(samples.cpu())
                generated += current_batch_size
                pbar.update(current_batch_size)

            pbar.close()

        # Concatenate all batches and trim to exactly num_samples
        all_images: torch.Tensor = torch.cat(all_samples, dim=0)
        return all_images[:num_samples]

    def compute_fid(self, fake_images: torch.Tensor) -> float:
        """Compute FID using the current metric state.

        Reads the FID value from the already-updated metric state. The
        metric must have been updated with both real and fake images before
        calling this method (via ``_update_real_features`` and
        ``_update_fake_features``).

        Args:
            fake_images: Unused parameter kept for API consistency with
                ``compute_kid`` and ``compute_is``. The metric state is
                read directly from ``self.fid``.

        Returns:
            FID score as a Python float. Lower is better.
            Typical values: iCT-IC ≈ 7.42, iCT-GC ≈ 5.95 on CIFAR-10.
        """
        fid_value: torch.Tensor = self.fid.compute()
        return float(fid_value.item())

    def compute_kid(
        self, fake_images: torch.Tensor
    ) -> Tuple[float, float]:
        """Compute KID ×10² using the current metric state.

        Reads the KID value from the already-updated metric state. The
        metric must have been updated with both real and fake images before
        calling this method.

        The paper reports KID ×10² (i.e., the raw KID value multiplied by
        100). TorchMetrics returns the raw KID; this method applies the ×100
        scaling.

        Args:
            fake_images: Unused parameter kept for API consistency.
                The metric state is read directly from ``self.kid``.

        Returns:
            Tuple ``(kid_mean * 100, kid_std * 100)`` as Python floats.
            Lower is better.
            Typical values: iCT-IC ≈ 0.44, iCT-GC ≈ 0.26 on CIFAR-10.
        """
        kid_mean: torch.Tensor
        kid_std: torch.Tensor
        kid_mean, kid_std = self.kid.compute()

        # Scale by 100 to match the paper's ×10² reporting convention
        return float(kid_mean.item()) * 100.0, float(kid_std.item()) * 100.0

    def compute_is(
        self, fake_images: torch.Tensor
    ) -> Tuple[float, float]:
        """Compute Inception Score using the current metric state.

        Reads the IS value from the already-updated metric state. IS only
        uses fake images (no real image comparison), so only
        ``_update_fake_features`` needs to have been called before this.

        Args:
            fake_images: Unused parameter kept for API consistency.
                The metric state is read directly from ``self.is_metric``.

        Returns:
            Tuple ``(is_mean, is_std)`` as Python floats. Higher is better.
            Typical values: iCT-IC ≈ 8.76, iCT-GC ≈ 9.10 on CIFAR-10.
        """
        is_mean: torch.Tensor
        is_std: torch.Tensor
        is_mean, is_std = self.is_metric.compute()

        return float(is_mean.item()), float(is_std.item())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _single_step_sample(self, z: torch.Tensor) -> torch.Tensor:
        """Generate a batch of images from noise in a single forward pass.

        Implements the consistency model's single-step generation:
            x_T = σ_T · z                    (scale noise to σ_T level)
            x̂_0 = f_θ(x_T, σ_T)             (single-step denoising)

        The output is clipped to ``[-1, 1]`` (neural network outputs can
        slightly exceed this range) and then mapped to ``[0, 1]`` for
        TorchMetrics compatibility.

        Args:
            z: Standard Gaussian noise of shape ``(B, C, H, W)`` on
                ``self.device``. Must be sampled as ``torch.randn(...)``
                (not pre-scaled by σ_T — scaling happens inside
                ``ConsistencyModel.generate``).

        Returns:
            Float tensor of shape ``(B, C, H, W)`` with values in ``[0, 1]``.
            On the same device as ``z`` (``self.device``).
        """
        # Single-step generation via ConsistencyModel.generate
        # Internally computes: forward(sigma_T * z, sigma_T * ones(B))
        x_0_hat: torch.Tensor = self.model.generate(z, sigma_T=self.sigma_max)

        # Clip to [-1, 1]: neural network outputs can slightly exceed this range
        # due to the c_skip * x + c_out * F_theta parametrization
        x_0_hat = x_0_hat.clamp(-1.0, 1.0)

        # Map from [-1, 1] to [0, 1] for TorchMetrics (normalize=True mode)
        # normalize_images: (x + 1.0) / 2.0, then clamp to [0, 1]
        x_0_hat = normalize_images(x_0_hat)

        return x_0_hat

    def _update_real_features(self, num_samples: int = 50000) -> None:
        """Feed real images to FID and KID metrics.

        Iterates over ``self.real_loader`` and feeds up to ``num_samples``
        real images to the FID and KID metric objects. Images from the
        DataLoader are in ``[-1, 1]`` (from the training transform) and are
        converted to ``[0, 1]`` before being passed to TorchMetrics.

        This method is called at the start of each evaluation run (after
        ``self.fid.reset()`` and ``self.kid.reset()``), matching the paper's
        protocol of "sampling new sets of training images" per run.

        IS does not use real images, so it is not updated here.

        Args:
            num_samples: Maximum number of real images to feed. Typically
                50,000 (``config.num_eval_samples``). Stops early if the
                DataLoader is exhausted before reaching this count.
        """
        real_count: int = 0

        with torch.no_grad():
            for batch in self.real_loader:
                # Extract images from (images, labels) or (images,) tuples
                if isinstance(batch, (list, tuple)):
                    real_images: torch.Tensor = batch[0]
                else:
                    real_images = batch

                # Move to device for metric update
                real_images = real_images.to(self.device)

                # Convert from [-1, 1] (training range) to [0, 1] (metric range)
                real_images_01: torch.Tensor = normalize_images(real_images)

                # Ensure float32 (some DataLoaders may return float16 or uint8)
                real_images_01 = real_images_01.float()

                # Determine how many images to use from this batch
                remaining: int = num_samples - real_count
                if remaining <= 0:
                    break

                # Trim batch if it would exceed num_samples
                if real_images_01.shape[0] > remaining:
                    real_images_01 = real_images_01[:remaining]

                # Update FID and KID with real images
                # real=True marks these as the reference distribution
                self.fid.update(real_images_01, real=True)
                self.kid.update(real_images_01, real=True)

                real_count += real_images_01.shape[0]

                if real_count >= num_samples:
                    break

    def _update_fake_features(self, fake_images: torch.Tensor) -> None:
        """Feed generated images to FID, KID, and IS metrics.

        Feeds the pre-generated fake images to all three metric objects in
        batches to avoid OOM errors. The images are already in ``[0, 1]``
        (output of ``generate_samples``).

        Args:
            fake_images: Float tensor of shape ``(N, C, H, W)`` with values
                in ``[0, 1]``. Typically on CPU (output of ``generate_samples``).
                Will be moved to ``self.device`` in batches.
        """
        num_fake: int = fake_images.shape[0]
        update_batch_size: int = min(self.gen_batch_size, num_fake)

        with torch.no_grad():
            for start_idx in range(0, num_fake, update_batch_size):
                end_idx: int = min(start_idx + update_batch_size, num_fake)
                batch: torch.Tensor = fake_images[start_idx:end_idx].to(
                    self.device
                )

                # Ensure float32
                batch = batch.float()

                # Update FID and KID with fake images (real=False)
                self.fid.update(batch, real=False)
                self.kid.update(batch, real=False)

                # IS only uses fake images (no real=True/False distinction)
                self.is_metric.update(batch)
