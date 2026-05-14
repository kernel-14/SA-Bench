```python
## evaluation/evaluator.py
"""Evaluation module for the NFIG framework.

Orchestrates two distinct evaluation workflows:
  1. Reconstruction quality (rFID): encode-decode ImageNet val images via FR-VAE
     and compute FID between reconstructed and original images.
  2. Generation quality (gFID, IS, Precision, Recall): generate 50,000 images
     using the full NFIG pipeline and compute metrics against ImageNet val.

Paper targets (Table 2):
    rFID:      0.85
    gFID:      2.81
    IS:        332.42
    Precision: 0.77
    Recall:    0.59

Config values used (config.yaml evaluation section):
    num_samples:   50000   (standard for ImageNet FID)
    cfg_scale:     4.5     (paper Section 4.1)
    top_k:         990     (paper Section 4.1)
    output_dir:    'generated_samples'
    fid_reference: 'data/imagenet_val_fid_stats.npz'
    metrics:       ['gfid', 'rfid', 'is', 'precision', 'recall']
"""

import json
import logging
import os
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple

import torch
import torchvision
from torch import Tensor
from torch.utils.data import DataLoader

from inference.sampler import NFIGSampler
from models.frvae.frvae import FRVAE
from utils.config import Config

# Configure module-level logger.
logger: logging.Logger = logging.getLogger(__name__)


class Evaluator:
    """Runs the full NFIG evaluation pipeline.

    Computes reconstruction FID (rFID) by encoding and decoding ImageNet
    validation images through the FR-VAE, and generation metrics (gFID, IS,
    Precision, Recall) by generating 50,000 images with the full NFIG pipeline
    and comparing against the ImageNet validation set.

    All model forward passes run under torch.no_grad() to prevent gradient
    accumulation. Both models are set to eval() mode in __init__ and kept
    there throughout all evaluation operations.

    Attributes:
        frvae: Trained FRVAE tokenizer/decoder in eval mode.
        sampler: NFIGSampler wrapping the trained transformer and FRVAE.
        val_loader: DataLoader for ImageNet validation split (50k images).
        config: Root Config dataclass loaded from config.yaml.
        num_samples: Number of samples for FID computation (50000 from config).
        cfg_scale: CFG scale for generation (4.5 from config).
        top_k: Top-k sampling parameter (990 from config).
        output_dir: Directory for saving generated images.
        fid_reference: Path to precomputed ImageNet val FID statistics.
        device: Target device inferred from config.training.device.
        _generation_batch_size: Batch size for image generation (memory-aware).
    """

    # Default batch size for generation — chosen to fit in GPU memory for 310M model.
    # Smaller than training batch_size=768 since inference requires storing activations
    # for the full 680-token sequence without gradient checkpointing.
    _DEFAULT_GENERATION_BATCH_SIZE: int = 16

    # Log progress every N generation batches.
    _LOG_EVERY_N_BATCHES: int = 50

    # Image format for saving generated samples.
    _IMAGE_FORMAT: str = "png"

    # Subdirectory names within output_dir.
    _GENERATED_SUBDIR: str = "generated"
    _RECONSTRUCTED_SUBDIR: str = "reconstructed"
    _REAL_SUBDIR: str = "real"

    def __init__(
        self,
        frvae: FRVAE,
        sampler: NFIGSampler,
        val_loader: DataLoader,
        config: Config,
    ) -> None:
        """Initialize the Evaluator.

        Sets both models to eval mode, extracts evaluation hyperparameters
        from config, and creates the output directory.

        Args:
            frvae: Trained FRVAE model. Will be set to eval() mode.
                Must be on the target device before passing to this constructor.
                Used for rFID computation (encode-decode) and as the decoder
                in the generation pipeline (via NFIGSampler).
            sampler: NFIGSampler wrapping the trained NFIGTransformer and FRVAE.
                Used for gFID/IS/Precision/Recall computation.
                The sampler's transformer and frvae are already set to eval()
                by the NFIGSampler constructor.
            val_loader: DataLoader for ImageNet validation split.
                Yields (images, labels) with images normalized to [-1, 1].
                Used for rFID computation and as the real image reference.
                From config.data.val_dir with config.data.image_size=256.
            config: Root Config dataclass populated from config.yaml.
                Key evaluation parameters are read from config.evaluation.
        """
        # --- Store references ---
        self.frvae: FRVAE = frvae
        self.sampler: NFIGSampler = sampler
        self.val_loader: DataLoader = val_loader
        self.config: Config = config

        # --- Freeze models for evaluation ---
        # eval() disables dropout and uses running stats for any BN layers.
        # requires_grad_(False) prevents gradient tape allocation.
        self.frvae.eval()
        self.frvae.requires_grad_(False)
        # NFIGSampler already sets transformer and frvae to eval() in its __init__.
        # Explicitly set here for safety in case sampler was modified externally.
        self.sampler.transformer.eval()
        self.sampler.transformer.requires_grad_(False)

        # --- Extract evaluation hyperparameters from config ---
        # All values from config.yaml evaluation section.
        self.num_samples: int = config.evaluation.num_samples          # 50000
        self.cfg_scale: float = config.evaluation.cfg_scale            # 4.5
        self.top_k: int = config.evaluation.top_k                      # 990
        self.output_dir: str = config.evaluation.output_dir            # 'generated_samples'
        self.fid_reference: str = config.evaluation.fid_reference      # path to .npz stats

        # --- Determine target device ---
        # Use config.training.device; fall back to cuda if available.
        if config.training.device == "cuda" and torch.cuda.is_available():
            self.device: torch.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # --- Generation batch size ---
        # Smaller than training batch_size to fit inference in GPU memory.
        # The 310M parameter model with 680 tokens requires ~4GB at bf16 per batch of 16.
        self._generation_batch_size: int = self._DEFAULT_GENERATION_BATCH_SIZE

        # --- Create output directory structure ---
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(
            os.path.join(self.output_dir, self._GENERATED_SUBDIR), exist_ok=True
        )
        os.makedirs(
            os.path.join(self.output_dir, self._RECONSTRUCTED_SUBDIR), exist_ok=True
        )
        os.makedirs(
            os.path.join(self.output_dir, self._REAL_SUBDIR), exist_ok=True
        )

        logger.info(
            "Evaluator initialized. "
            "num_samples=%d, cfg_scale=%.1f, top_k=%d, output_dir='%s'",
            self.num_samples,
            self.cfg_scale,
            self.top_k,
            self.output_dir,
        )

    # ---------------------------------------------------------------------- #
    # Public interface
    # ---------------------------------------------------------------------- #

    def compute_rfid(self, num_samples: int = 50000) -> float:
        """Compute reconstruction FID (rFID) for the FR-VAE tokenizer.

        Encodes and decodes ImageNet validation images through the FR-VAE,
        saves both original and reconstructed images to disk, then computes
        FID between the two sets using clean-fid.

        Target rFID from paper Table 2: 0.85

        Args:
            num_samples: Number of validation images to use for rFID computation.
                From config.evaluation.num_samples = 50000.
                Capped at the total number of validation images if smaller.

        Returns:
            Scalar rFID value (lower is better; target: 0.85).
            Returns float('inf') if FID computation fails (e.g., missing dependency).
        """
        logger.info("Computing rFID with %d samples...", num_samples)

        # Directories for saving real and reconstructed images.
        real_dir: str = os.path.join(self.output_dir, self._REAL_SUBDIR)
        recon_dir: str = os.path.join(self.output_dir, self._RECONSTRUCTED_SUBDIR)

        # Clear existing images to avoid stale data from previous runs.
        self._clear_directory(real_dir)
        self._clear_directory(recon_dir)

        # Collect and save original + reconstructed image pairs.
        original_images: List[Tensor]
        reconstructed_images: List[Tensor]
        original_images, reconstructed_images = self._compute_reconstruction_images(
            loader=self.val_loader,
            num_samples=num_samples,
        )

        # Save images to disk for FID computation.
        logger.info(
            "Saving %d original and reconstructed image pairs to disk...",
            len(original_images),
        )
        self._save_images_to_dir(original_images, real_dir)
        self._save_images_to_dir(reconstructed_images, recon_dir)

        # Compute FID between reconstructed and real images.
        rfid: float = self._compute_fid_from_dirs(
            generated_dir=recon_dir,
            real_dir=real_dir,
        )

        logger.info("rFID = %.4f (target: %.2f)", rfid, 0.85)
        return rfid

    def compute_gfid_is_prec_rec(
        self,
        num_samples: int = 50000,
        cfg_scale: float = 4.5,
        top_k: int = 990,
    ) -> Dict[str, float]:
        """Generate images and compute gFID, IS, Precision, and Recall.

        Generates `num_samples` images using the full NFIG pipeline (transformer
        + FR-VAE decoder) with CFG and top-k sampling, saves them to disk, then
        computes all four generation quality metrics.

        Target values from paper Table 2:
            gFID:      2.81
            IS:        332.42
            Precision: 0.77
            Recall:    0.59

        Args:
            num_samples: Number of images to generate.
                From config.evaluation.num_samples = 50000.
            cfg_scale: CFG scale for generation.
                From config.evaluation.cfg_scale = 4.5 (paper Section 4.1).
            top_k: Top-k sampling parameter.
                From config.evaluation.top_k = 990 (paper Section 4.1).

        Returns:
            Dictionary with keys:
                - 'gfid': generation FID (lower is better; target: 2.81)
                - 'is': Inception Score (higher is better; target: 332.42)
                - 'precision': Precision (higher is better; target: 0.77)
                - 'recall': Recall (higher is better; target: 0.59)
        """
        logger.info(
            "Computing gFID/IS/Precision/Recall with %d samples "
            "(cfg_scale=%.1f, top_k=%d)...",
            num_samples,
            cfg_scale,
            top_k,
        )

        # Directory for saving generated images.
        generated_dir: str = os.path.join(self.output_dir, self._GENERATED_SUBDIR)
        self._clear_directory(generated_dir)

        # Generate and save all images to disk.
        self.generate_samples(
            num_samples=num_samples,
            cfg_scale=cfg_scale,
            top_k=top_k,
            save_dir=generated_dir,
        )

        # Compute all generation metrics.
        metrics: Dict[str, float] = self._compute_generation_metrics(
            generated_dir=generated_dir,
            num_samples=num_samples,
        )

        logger.info(
            "Generation metrics — gFID: %.4f (target: 2.81), "
            "IS: %.2f (target: 332.42), "
            "Precision: %.4f (target: 0.77), "
            "Recall: %.4f (target: 0.59)",
            metrics.get("gfid", float("inf")),
            metrics.get("is", 0.0),
            metrics.get("precision", 0.0),
            metrics.get("recall", 0.0),
        )

        return metrics

    def generate_samples(
        self,
        num_samples: int,
        cfg_scale: float,
        top_k: int,
        save_dir: str,
    ) -> None:
        """Generate `num_samples` images and save them to `save_dir` as PNG files.

        Generates images in batches using the NFIGSampler. Class labels are
        distributed uniformly across all 1000 ImageNet classes (50 images per
        class for 50,000 total samples). Images are saved as zero-padded PNG
        files (e.g., 000000.png, 000001.png, ...).

        Args:
            num_samples: Total number of images to generate.
                For FID evaluation: 50000 (config.evaluation.num_samples).
            cfg_scale: CFG scale for generation (4.5 from config).
            top_k: Top-k sampling parameter (990 from config).
            save_dir: Directory path where generated PNG images will be saved.
                Created if it does not exist.

        Side effects:
            Saves `num_samples` PNG files to `save_dir`.
            Logs progress every `_LOG_EVERY_N_BATCHES` batches.
        """
        os.makedirs(save_dir, exist_ok=True)

        num_classes: int = self.config.nfig.num_classes  # 1000

        # Build class label schedule: uniform distribution across 1000 classes.
        # For 50000 samples: 50 images per class.
        # For other counts: distribute as evenly as possible.
        images_per_class: int = max(1, num_samples // num_classes)
        remainder: int = num_samples - images_per_class * num_classes

        # Build the full class label tensor.
        # First: images_per_class copies of each class [0..999]
        # Then: remainder additional images from classes [0..remainder-1]
        class_labels_list: List[Tensor] = [
            torch.arange(num_classes, dtype=torch.long).repeat(images_per_class)
        ]
        if remainder > 0:
            class_labels_list.append(
                torch.arange(remainder, dtype=torch.long)
            )
        all_class_labels: Tensor = torch.cat(class_labels_list, dim=0)  # [num_samples]

        # Truncate or pad to exactly num_samples.
        all_class_labels = all_class_labels[:num_samples]

        logger.info(
            "Generating %d images (%d classes × ~%d images/class) "
            "with cfg_scale=%.1f, top_k=%d...",
            num_samples,
            num_classes,
            images_per_class,
            cfg_scale,
            top_k,
        )

        total_saved: int = 0
        batch_size: int = self._generation_batch_size
        num_batches: int = (num_samples + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            batch_start: int = batch_idx * batch_size
            batch_end: int = min(batch_start + batch_size, num_samples)
            current_batch_size: int = batch_end - batch_start

            # Get class labels for this batch and move to device.
            batch_labels: Tensor = all_class_labels[batch_start:batch_end].to(
                self.device
            )

            # Generate images via NFIGSampler.
            # sampler.sample() runs under @torch.no_grad() internally.
            # Returns Tensor [B, 3, 256, 256] in [-1, 1].
            try:
                generated: Tensor = self.sampler.sample(
                    class_labels=batch_labels,
                    cfg_scale=cfg_scale,
                    top_k=top_k,
                )
            except RuntimeError as exc:
                # Handle GPU OOM by reducing batch size and retrying.
                if "out of memory" in str(exc).lower() and batch_size > 1:
                    logger.warning(
                        "GPU OOM at batch_size=%d. Reducing to %d and retrying.",
                        batch_size,
                        batch_size // 2,
                    )
                    batch_size = batch_size // 2
                    self._generation_batch_size = batch_size
                    torch.cuda.empty_cache()

                    # Retry with smaller batch size.
                    generated = self.sampler.sample(
                        class_labels=batch_labels[:batch_size],
                        cfg_scale=cfg_scale,
                        top_k=top_k,
                    )
                    # Adjust batch_end for the reduced batch.
                    batch_end = batch_start + min(batch_size, current_batch_size)
                    current_batch_size = batch_end - batch_start
                    generated = generated[:current_batch_size]
                else:
                    raise

            # Denormalize from [-1, 1] to [0, 1] for saving.
            # config.data.mean=0.5, config.data.std=0.5 → inverse: x*0.5+0.5
            images_01: Tensor = (generated.float() * 0.5 + 0.5).clamp(0.0, 1.0)
            images_01 = images_01.cpu()

            # Save each image as a PNG file.
            for local_idx in range(images_01.shape[0]):
                global_idx: int = batch_start + local_idx
                save_path: str = os.path.join(
                    save_dir, f"{global_idx:06d}.{self._IMAGE_FORMAT}"
                )
                # torchvision.utils.save_image expects [C, H, W] float in [0, 1].
                torchvision.utils.save_image(images_01[local_idx], save_path)

            total_saved += images_01.shape[0]

            # Log progress.
            if (batch_idx + 1) % self._LOG_EVERY_N_BATCHES == 0 or batch_idx == num_batches - 1:
                logger.info(
                    "Generation progress: %d/%d images saved (%.1f%%)",
                    total_saved,
                    num_samples,
                    100.0 * total_saved / num_samples,
                )

        logger.info(
            "Generation complete. %d images saved to '%s'.",
            total_saved,
            save_dir,
        )

    def _compute_reconstruction_images(
        self,
        loader: DataLoader,
        num_samples: int = 50000,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Encode and decode validation images through FR-VAE.

        Iterates over the validation DataLoader, runs each batch through the
        FR-VAE encode-decode pipeline, and returns paired lists of original
        and reconstructed images as uint8 tensors in [0, 255].

        Args:
            loader: DataLoader yielding (images, labels) batches.
                images: Tensor [B, 3, H, W] normalized to [-1, 1].
            num_samples: Maximum number of images to process.
                Stops early once this many images have been collected.

        Returns:
            Tuple of two lists:
                - original_images: List of Tensor [3, H, W] uint8 in [0, 255].
                  Real ImageNet validation images.
                - reconstructed_images: List of Tensor [3, H, W] uint8 in [0, 255].
                  FR-VAE reconstructions of the same images.
            Both lists have the same length (min(num_samples, dataset_size)).
        """
        original_images: List[Tensor] = []
        reconstructed_images: List[Tensor] = []
        total_collected: int = 0

        logger.info(
            "Collecting %d reconstruction pairs from validation set...", num_samples
        )

        with torch.no_grad():
            for batch_idx, (images, _labels) in enumerate(loader):
                # Move images to device.
                images = images.to(self.device, non_blocking=True)  # [B, 3, H, W]

                # FR-VAE forward pass: encode → quantize → decode.
                # FRVAE.forward() returns (x_hat, f, f_tilde).
                x_hat: Tensor
                x_hat, _f, _f_tilde = self.frvae.forward(images)
                # x_hat: [B, 3, H, W] in [-1, 1]

                # Denormalize both to uint8 [0, 255].
                # config.data.mean=0.5, config.data.std=0.5 → inverse: x*0.5+0.5
                orig_uint8: Tensor = self._denormalize_to_uint8(images)    # [B, 3, H, W]
                recon_uint8: Tensor = self._denormalize_to_uint8(x_hat)    # [B, 3, H, W]

                # Move to CPU and split into individual images.
                orig_uint8_cpu: Tensor = orig_uint8.cpu()
                recon_uint8_cpu: Tensor = recon_uint8.cpu()

                for i in range(orig_uint8_cpu.shape[0]):
                    if total_collected >= num_samples:
                        break
                    original_images.append(orig_uint8_cpu[i])       # [3, H, W] uint8
                    reconstructed_images.append(recon_uint8_cpu[i])  # [3, H, W] uint8
                    total_collected += 1

                if total_collected >= num_samples:
                    break

                # Log progress every 100 batches.
                if (batch_idx + 1) % 100 == 0:
                    logger.info(
                        "Reconstruction progress: %d/%d images collected.",
                        total_collected,
                        num_samples,
                    )

        logger.info(
            "Collected %d reconstruction pairs.", total_collected
        )

        return original_images, reconstructed_images

    def run_full_evaluation(self) -> Dict[str, float]:
        """Run the complete evaluation pipeline and return all metrics.

        Orchestrates both rFID (reconstruction quality) and gFID/IS/Precision/
        Recall (generation quality) evaluations. Saves all metrics to a JSON
        file in the output directory for reproducibility.

        Returns:
            Dictionary with all evaluation metrics:
                - 'rfid': Reconstruction FID (target: 0.85)
                - 'gfid': Generation FID (target: 2.81)
                - 'is': Inception Score (target: 332.42)
                - 'precision': Precision (target: 0.77)
                - 'recall': Recall (target: 0.59)

        Side effects:
            Saves metrics to `output_dir/metrics.json`.
            Saves generated images to `output_dir/generated/`.
            Saves reconstructed images to `output_dir/reconstructed/`.
            Saves real images to `output_dir/real/`.
        """
        logger.info("=" * 60)
        logger.info("Starting full NFIG evaluation pipeline.")
        logger.info("=" * 60)

        results: Dict[str, float] = {}

        # ------------------------------------------------------------------ #
        # 1. Reconstruction FID (rFID)
        # ------------------------------------------------------------------ #
        logger.info("--- Phase 1: Reconstruction FID (rFID) ---")
        rfid: float = self.compute_rfid(num_samples=self.num_samples)
        results["rfid"] = rfid
        logger.info("rFID = %.4f (paper target: 0.85)", rfid)

        # ------------------------------------------------------------------ #
        # 2. Generation metrics (gFID, IS, Precision, Recall)
        # ------------------------------------------------------------------ #
        logger.info("--- Phase 2: Generation Metrics (gFID, IS, Precision, Recall) ---")
        gen_metrics: Dict[str, float] = self.compute_gfid_is_prec_rec(
            num_samples=self.num_samples,
            cfg_scale=self.cfg_scale,
            top_k=self.top_k,
        )
        results.update(gen_metrics)

        logger.info("gFID      = %.4f (paper target: 2.81)", results.get("gfid", float("inf")))
        logger.info("IS        = %.2f (paper target: 332.42)", results.get("is", 0.0))
        logger.info("Precision = %.4f (paper target: 0.77)", results.get("precision", 0.0))
        logger.info("Recall    = %.4f (paper target: 0.59)", results.get("recall", 0.0))

        # ------------------------------------------------------------------ #
        # 3. Save metrics to JSON
        # ------------------------------------------------------------------ #
        metrics_path: str = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Metrics saved to '%s'.", metrics_path)

        logger.info("=" * 60)
        logger.info("Evaluation complete.")
        logger.info("=" * 60)

        return results

    # ---------------------------------------------------------------------- #
    # Private helper methods
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _denormalize_to_uint8(tensor: Tensor) -> Tensor:
        """Convert a normalized image tensor from [-1, 1] to uint8 [0, 255].

        Applies the inverse of the training pipeline normalization:
            config.data.mean = [0.5, 0.5, 0.5]
            config.data.std  = [0.5, 0.5, 0.5]
        Inverse: pixel = (tensor * 0.5 + 0.5) * 255

        Args:
            tensor: Float tensor of shape (..., 3, H, W) with values in [-1, 1].
                Typically a batch [B, 3, H, W] or single image [3, H, W].

        Returns:
            uint8 tensor of the same shape with values in [0, 255].
        """
        # Step 1: [-1, 1] → [0, 1]
        tensor_01: Tensor = tensor.float() * 0.5 + 0.5

        # Step 2: [0, 1] → [0, 255], clamp to valid range, convert to uint8.
        tensor_255: Tensor = (tensor_01 * 255.0).clamp(0.0, 255.0)

        return tensor_255.to(torch.uint8)

    def _save_images_to_dir(
        self,
        images: List[Tensor],
        save_dir: str,
    ) -> None:
        """Save a list of uint8 image tensors to a directory as PNG files.

        Args:
            images: List of Tensor [3, H, W] uint8 in [0, 255].
                Each tensor represents one image.
            save_dir: Directory path where PNG files will be saved.
                Must already exist (created by caller).

        Side effects:
            Saves len(images) PNG files to save_dir with zero-padded names.
        """
        for idx, img_uint8 in enumerate(images):
            save_path: str = os.path.join(
                save_dir, f"{idx:06d}.{self._IMAGE_FORMAT}"
            )
            # torchvision.utils.save_image expects float [0, 1] or uint8 [0, 255].
            # Convert uint8 to float [0, 1] for save_image.
            img_float: Tensor = img_uint8.float() / 255.0
            torchvision.utils.save_image(img_float, save_path)

    def _compute_fid_from_dirs(
        self,
        generated_dir: str,
        real_dir: Optional[str] = None,
    ) -> float:
        """Compute FID between generated and real image directories.

        Attempts to use clean-fid for FID computation. Falls back to
        torch-fidelity if clean-fid is not available. Returns float('inf')
        if neither library is available.

        For rFID: compares reconstructed images (generated_dir) against
            real images (real_dir).
        For gFID: compares generated images (generated_dir) against
            ImageNet val reference statistics (fid_reference) or real_dir.

        Args:
            generated_dir: Path to directory containing generated/reconstructed
                PNG images. Must contain at least 2048 images for reliable FID.
            real_dir: Optional path to directory containing real images.
                If None, uses self.fid_reference (precomputed stats file).
                If fid_reference also doesn't exist, computes stats from
                val_loader images saved to a temporary directory.

        Returns:
            Scalar FID value (lower is better).
            Returns float('inf') if computation fails.
        """
        # Try clean-fid first (preferred for ImageNet benchmarks).
        try:
            from cleanfid import fid as cleanfid

            if real_dir is not None and os.path.isdir(real_dir):
                # Compare two directories directly.
                score: float = cleanfid.compute_fid(
                    fdir1=generated_dir,
                    fdir2=real_dir,
                    mode="clean",
                    num_workers=min(8, os.cpu_count() or 4),
                    device=str(self.device),
                    verbose=False,
                )
            elif os.path.exists(self.fid_reference):
                # Use precomputed reference statistics.
                score = cleanfid.compute_fid(
                    fdir1=generated_dir,
                    dataset_name=None,
                    dataset_res=self.config.data.image_size,
                    mode="clean",
                    num_workers=min(8, os.cpu_count() or 4),
                    device=str(self.device),
                    verbose=False,
                    custom_feat_extractor=None,
                    custom_image_tranform=None,
                    dataset_split="custom",
                    custom_stats_file=self.fid_reference,
                )
            else:
                # Fall back to computing stats from val_loader images.
                logger.warning(
                    "FID reference stats file '%s' not found. "
                    "Computing FID from val_loader images (slower).",
                    self.fid_reference,
                )
                real_temp_dir: str = os.path.join(
                    self.output_dir, self._REAL_SUBDIR
                )
                if not os.listdir(real_temp_dir):
                    # Save real images if not already saved.
                    self._save_real_images_for_fid(
                