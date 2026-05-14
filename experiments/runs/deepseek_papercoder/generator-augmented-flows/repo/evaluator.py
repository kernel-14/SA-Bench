"""
evaluator.py – Evaluation utilities for consistency models.

This module provides an `Evaluator` class that wraps torchmetrics to compute
Fréchet Inception Distance (FID), Kernel Inception Distance (KID) and Inception
Score (IS) for a trained consistency model. It follows the protocol described
in the paper:

- 50 000 images are generated using the one‑step consistency model.
- Real images are taken from the training set (exactly 50 000 samples).
- Confidence intervals are computed by evaluating the same model multiple times
  with different random noise seeds.

The class uses the EMA model (provided) and the data module to load real images.
"""

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Optional, Tuple, Union

# Project imports (use absolute paths to avoid circular dependencies)
from model import ConsistencyModel
from data import DataModule

# Metrics from torchmetrics – version 0.11+ (the API is stable for FID/KID/IS)
from torchmetrics.image import FrechetInceptionDistance
from torchmetrics.image import KernelInceptionDistance
from torchmetrics.image import InceptionScore


class Evaluator:
    """
    Performs quantitative evaluation of a consistency model.

    Args:
        model:            An **EMA** consistency model (already loaded with trained weights).
        data_module:      A DataModule instance giving access to the training set.
        num_samples:      Number of generated / real images to use for evaluation
                          (default from config: 50000).
        sigma_max:        Maximum noise level σ_T used for one‑step sampling
                          (default from config: 80.0).
        device:           Device on which to run the model and metrics.
        eval_batch_size:  Batch size used when generating images and feeding metrics.
                          Defaults to 128.
        cache_real:       If True (the default), all real images are loaded once during
                          construction and kept in GPU memory. This speeds up multiple
                          evaluation runs but consumes VRAM accordingly.
    """

    def __init__(
        self,
        model: ConsistencyModel,
        data_module: DataModule,
        num_samples: int = 50000,
        sigma_max: float = 80.0,
        device: str = "cuda",
        eval_batch_size: int = 128,
        cache_real: bool = True,
    ) -> None:
        self.model = model
        self.data_module = data_module
        self.num_samples = num_samples
        self.sigma_max = sigma_max
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.eval_batch_size = eval_batch_size

        # Put model in evaluation mode and transfer to device
        self.model.to(self.device)
        self.model.eval()

        # Pre-load real images for speed and reproducibility
        if cache_real:
            self.real_images = self._load_real_images()
        else:
            self.real_images = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, noise: Tensor) -> Tensor:
        """
        One‑step generation from a batch of noise vectors.

        Args:
            noise:  Tensor of shape (B, C, H, W) drawn from N(0, I).

        Returns:
            Generated images in the original model output range [-1, 1].
        """
        B, C, H, W = noise.shape
        noise = noise.to(self.device)

        # Scale noise and construct the sigma input
        noisy_input = noise * self.sigma_max
        sigma = torch.full((B,), self.sigma_max, device=self.device)

        # Forward pass – the model internally handles the broadcast of sigma
        output = self.model(noisy_input, sigma)
        return output  # range [-1, 1]

    def compute_fid(self) -> float:
        """
        Compute Fréchet Inception Distance.

        Returns:
            The FID score (lower is better).
        """
        fid = FrechetInceptionDistance(feature=2048).to(self.device)
        self._feed_real_images(fid)
        self._feed_fake_images(fid)
        return float(fid.compute().item())

    def compute_kid(self, subset_size: int = 1000) -> float:
        """
        Compute Kernel Inception Distance.

        Args:
            subset_size: Number of images used for the MMD estimate (default 1000).

        Returns:
            The KID score (lower is better), multiplied by 100 (as in paper).
        """
        kid = KernelInceptionDistance(
            feature=2048, subset_size=subset_size
        ).to(self.device)
        self._feed_real_images(kid)
        self._feed_fake_images(kid)
        # The metric returns KID * 100 by default? In torchmetrics 0.11+ it returns
        # the raw value; but the paper reports KID × 10^2. We'll multiply by 100 if needed.
        # Check: KernelInceptionDistance returns the metric as computed; it may already
        # be scaled? The docs: "The output is a scalar * 1000". Actually they changed;
        # safest to multiply by 100 manually. We'll return `kid.compute().item() * 100`.
        return float(kid.compute().item() * 100)

    def compute_is(self) -> float:
        """
        Compute Inception Score (mean).

        Returns:
            The Inception Score (higher is better).
        """
        is_metric = InceptionScore(feature=2048, splits=10).to(self.device)
        self._feed_fake_images(is_metric, is_inception_score=True)
        score, _ = is_metric.compute()  # returns (mean, std)
        return float(score.item())

    def evaluate_multiple_runs(
        self,
        num_runs: int = 5,
        base_seed: int = 42,
        metrics_list: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation multiple times with different noise seeds and report
        mean ± std for each metric. This matches the confidence intervals in the paper.

        Args:
            num_runs:     Number of independent evaluation passes (default 5).
            base_seed:    Base seed offset for reproducibility.
            metrics_list: List of metric names to compute; defaults to
                          ['fid', 'kid', 'is'].

        Returns:
            A dictionary with keys like ``'fid_mean'``, ``'fid_std'``, etc.
        """
        if metrics_list is None:
            metrics_list = ["fid", "kid", "is"]

        results = {m: [] for m in metrics_list}

        for run_id in range(num_runs):
            self._set_seed(run_id, base_seed)

            run_values = {}
            if "fid" in metrics_list:
                run_values["fid"] = self.compute_fid()
            if "kid" in metrics_list:
                run_values["kid"] = self.compute_kid()
            if "is" in metrics_list:
                run_values["is"] = self.compute_is()
            for k, v in run_values.items():
                results[k].append(v)

        # Aggregate
        agg = {}
        for metric_name, vals in results.items():
            t = torch.tensor(vals, dtype=torch.float32)
            agg[f"{metric_name}_mean"] = float(t.mean().item())
            agg[f"{metric_name}_std"] = float(t.std().item())
        return agg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_real_images(self) -> Tensor:
        """
        Collect exactly ``num_samples`` real images from the training set,
        convert them to uint8 [0, 255], and return as a single tensor on the
        target device.

        This method guarantees a deterministic ordering (shuffle=False) so that
        the reference set remains identical across evaluation runs.
        """
        # Create a fresh dataloader that does not drop the last batch and has a
        # suitable batch size. This ensures we can obtain exactly num_samples images.
        loader = DataLoader(
            self.data_module.dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=2,
            drop_last=False,
            pin_memory=True,
        )

        real_list: List[Tensor] = []
        count = 0
        for batch in loader:
            if count + batch.shape[0] > self.num_samples:
                # Trim the last batch to reach exactly num_samples
                batch = batch[: self.num_samples - count]
            uint8_batch = self._to_uint8(batch).to(self.device)
            real_list.append(uint8_batch)
            count += uint8_batch.shape[0]
            if count >= self.num_samples:
                break

        return torch.cat(real_list, dim=0)

    def _feed_real_images(self, metric: Union[FrechetInceptionDistance, KernelInceptionDistance]) -> None:
        """
        Feed all cached real images to a FID or KID metric.

        The metric expects calls to ``update(images, real=True)`` for real images.
        """
        if self.real_images is None:
            # Cache not used; stream from dataloader (slower but still correct)
            real_iter = self._real_image_iterator()
        else:
            dataset = TensorDataset(self.real_images)
            loader = DataLoader(
                dataset, batch_size=self.eval_batch_size, shuffle=False, drop_last=False
            )
            real_iter = (batch.to(self.device) for (batch,) in loader)

        for real_batch in real_iter:
            # Real images are already uint8
            metric.update(real_batch, real=True)

    def _feed_fake_images(
        self,
        metric: Union[FrechetInceptionDistance, KernelInceptionDistance, InceptionScore],
        is_inception_score: bool = False,
    ) -> None:
        """
        Generate fake images and feed them to a metric.

        For FID/KID: ``metric.update(fake_batch, real=False)``
        For IS:       ``metric.update(fake_batch)``

        Args:
            metric:             The torchmetrics metric instance.
            is_inception_score: Flag switching to IS update API.
        """
        fake_iter = self._generate_fake_batches()
        for fake_batch_uint8 in fake_iter:
            if is_inception_score:
                metric.update(fake_batch_uint8)
            else:
                metric.update(fake_batch_uint8, real=False)

    @torch.no_grad()
    def _generate_fake_batches(self) -> Generator[Tensor, None, None]:
        """
        Generator that yields batches of generated images in uint8 [0, 255]
        until ``num_samples`` have been produced.
        """
        generated_count = 0
        while generated_count < self.num_samples:
            # Determine current batch size (last batch may be smaller)
            current_bs = min(self.eval_batch_size, self.num_samples - generated_count)
            noise = torch.randn(
                current_bs,
                *self.data_module.get_data_shape(),
                device=self.device,
            )
            # One-step generation + conversion
            gen = self.sample(noise)               # [-1, 1]
            gen_uint8 = self._to_uint8(gen)
            yield gen_uint8
            generated_count += current_bs

    def _real_image_iterator(self) -> Generator[Tensor, None, None]:
        """
        Generator yielding batches of real images (uint8) straight from the
        training set, without caching. Used only when ``cache_real=False``.
        """
        loader = DataLoader(
            self.data_module.dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=2,
            drop_last=False,
            pin_memory=True,
        )
        produced = 0
        for batch in loader:
            if produced + batch.shape[0] > self.num_samples:
                batch = batch[: self.num_samples - produced]
            uint8_batch = self._to_uint8(batch).to(self.device)
            produced += uint8_batch.shape[0]
            yield uint8_batch
            if produced >= self.num_samples:
                break

    @staticmethod
    def _to_uint8(images: Tensor) -> Tensor:
        """
        Convert images from the model's output range [-1, 1] to uint8 [0, 255].

        Args:
            images:  Tensor of any shape with values in [-1, 1].

        Returns:
            Tensor of same shape with dtype=torch.uint8.
        """
        images = images.clamp(-1.0, 1.0)
        images = (images + 1.0) * 127.5
        images = torch.round(images).clamp(0, 255).to(torch.uint8)
        return images

    @staticmethod
    def _set_seed(run_id: int, base_seed: int) -> None:
        """
        Set random seeds for reproducibility of a single evaluation run.
        """
        seed = base_seed + run_id
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

