## Code: evaluation/evaluator.py

```python
## evaluation/evaluator.py
"""Quantitative evaluation hub for Hi-MAR hierarchical masked autoregressive models.

This module implements the ``Evaluator`` class, which orchestrates image
generation via ``Generator``, saves outputs to disk, and computes all metrics
reported in the paper:

- FID (Fréchet Inception Distance) — primary metric for both ImageNet and COCO
- IS (Inception Score) — ImageNet Table 2
- Precision / Recall — ImageNet Table 2
- T2I-CompBench compositional scores — COCO Table 4
- Speed / accuracy trade-off curves — Figure 3 and Figure 4

Every metric value in Tables 2, 3, 4, and Figures 3, 4 flows through this class.

Configuration alignment (config.yaml):
    evaluation.imagenet.n_samples          = 50000
    evaluation.coco.n_samples              = 30000
    inference.phase1_steps                 = 32
    inference.phase2_steps                 = 4
    inference.cfg.scale                    = 2.9
    inference.speed_accuracy_sweep.batch_size = 128
    inference.speed_accuracy_sweep.phase2_steps_sweep = [1, 2, 4, 6, 8]
    data.imagenet.val_root                 = "data/imagenet/val"
    data.coco.val_root                     = "data/coco/val2017"
    seed                                   = 42

Paper reference: Section 4.2 (Evaluation Metrics), Section 4.5 (Experimental Analysis).
"""

import json
import logging
import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from inference.generate import Generator
from models.himar import HiMAR
from models.vae_tokenizer import VAETokenizer
from utils.misc import setup_logger


class Evaluator:
    """Quantitative evaluation hub for Hi-MAR image generation.

    Orchestrates image generation via ``Generator``, saves outputs to disk,
    and computes all metrics reported in the paper. The evaluator is purely
    read-only with respect to the model — it never modifies model parameters
    or calls ``model.train()``.

    All public methods:
    - Call ``model.eval()`` at entry to ensure deterministic behaviour.
    - Wrap generation in ``torch.no_grad()`` to prevent gradient accumulation.
    - Reuse previously generated images when ``force_regenerate=False`` to
      avoid redundant generation across metric calls.

    Directory structure created at construction:
        output_dir/
            fake/          ← generated images for ImageNet FID/IS/PR
            real/          ← real images for PR (if needed)
            coco_fake/     ← generated images for COCO FID / T2I-CompBench
            coco_real/     ← real COCO validation images for FID reference
            speed_bench/   ← per-step-config images for speed/accuracy sweep

    Attributes:
        model: Trained HiMAR model. Set to eval mode at construction.
        vae: Frozen VAETokenizer. Used only for decoding inside Generator.
        generator: Generator instance. All image synthesis goes through this.
        output_dir: Base path for all saved artifacts.
        device: Compute device for all tensor operations.
        fake_dir: Path to generated ImageNet images.
        real_dir: Path to real ImageNet validation images (for PR).
        coco_fake_dir: Path to generated COCO images.
        coco_real_dir: Path to real COCO validation images (for FID reference).
        speed_bench_dir: Path for speed/accuracy sweep images.
        logger: Python logger for console and file output.
    """

    def __init__(
        self,
        model: HiMAR,
        vae: VAETokenizer,
        generator: Generator,
        output_dir: str,
        device: torch.device,
    ) -> None:
        """Initialises the Evaluator and creates output directory structure.

        Args:
            model: Trained HiMAR model. Will be set to eval mode immediately.
                The EMA transformer is used for generation when
                ``model.config.use_ema_for_inference=True`` (default).
            vae: Frozen VAETokenizer. Never trained; used only for decoding
                inside the Generator.
            generator: Generator instance wrapping the model and VAE. All
                image synthesis is delegated to this object.
            output_dir: Root directory for all evaluation artifacts (generated
                images, metric results, logs). Created if it does not exist.
                Config: ``output.dir = "outputs"``.
            device: Compute device for all tensor operations. Should match
                the device on which ``model`` and ``vae`` reside.
        """
        self.model: HiMAR = model
        self.vae: VAETokenizer = vae
        self.generator: Generator = generator
        self.output_dir: str = output_dir
        self.device: torch.device = device

        # Set model to eval mode — disables dropout and batch norm training
        # behaviour. Critical for deterministic, reproducible evaluation.
        self.model.eval()

        # ------------------------------------------------------------------
        # Create output directory structure.
        # All directories are created with exist_ok=True to allow repeated
        # evaluation runs without errors.
        # ------------------------------------------------------------------
        self.fake_dir: str = os.path.join(output_dir, "fake")
        self.real_dir: str = os.path.join(output_dir, "real")
        self.coco_fake_dir: str = os.path.join(output_dir, "coco_fake")
        self.coco_real_dir: str = os.path.join(output_dir, "coco_real")
        self.speed_bench_dir: str = os.path.join(output_dir, "speed_bench")

        for directory in (
            self.fake_dir,
            self.real_dir,
            self.coco_fake_dir,
            self.coco_real_dir,
            self.speed_bench_dir,
        ):
            os.makedirs(directory, exist_ok=True)

        # ------------------------------------------------------------------
        # Setup logger.
        # ------------------------------------------------------------------
        log_dir: str = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file: str = os.path.join(log_dir, "evaluator.log")
        self.logger: logging.Logger = setup_logger("evaluator", log_file)

        self.logger.info(
            f"Evaluator initialised. Output dir: {output_dir}, "
            f"Device: {device}."
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_images(
        self,
        images: torch.Tensor,
        out_dir: str,
        start_idx: int,
    ) -> None:
        """Saves a batch of generated images to disk as PNG files.

        Converts images from the VAE decoder's ``[-1, 1]`` float range to
        ``[0, 255]`` uint8 and saves each as a zero-padded PNG file.

        Filename convention: ``{start_idx + i:06d}.png`` (6-digit zero-padding
        ensures lexicographic sort order matches generation order, which is
        required for reproducibility when computing FID across multiple runs).

        Args:
            images: Float tensor of shape ``[B, 3, H, W]`` in ``[-1, 1]``
                range. Produced by ``VAETokenizer.decode()`` via the Generator.
            out_dir: Directory to save images. Must already exist.
            start_idx: Global offset for filename numbering. Pass the total
                number of images already saved to ensure unique filenames
                across batches.
        """
        # Detach from computation graph and move to CPU for numpy conversion.
        images_cpu: torch.Tensor = images.detach().cpu()

        # Rescale from [-1, 1] to [0, 255] uint8.
        # Clamp to [0, 255] to handle minor out-of-range values from the VAE.
        images_uint8: torch.Tensor = (
            (images_cpu + 1.0) / 2.0 * 255.0
        ).clamp(0.0, 255.0).to(torch.uint8)

        # Convert to numpy: [B, C, H, W] → [B, H, W, C] for PIL.
        images_np: np.ndarray = images_uint8.permute(0, 2, 3, 1).numpy()

        batch_size: int = images_np.shape[0]
        for i in range(batch_size):
            filename: str = os.path.join(out_dir, f"{start_idx + i:06d}.png")
            pil_image: Image.Image = Image.fromarray(images_np[i])
            pil_image.save(filename, format="PNG")

    def _count_images_in_dir(self, directory: str) -> int:
        """Counts the number of PNG/JPEG image files in a directory.

        Used to check whether a directory already contains enough generated
        images to skip regeneration (when ``force_regenerate=False``).

        Args:
            directory: Path to the directory to inspect.

        Returns:
            Number of image files (PNG or JPEG) in the directory.
        """
        if not os.path.isdir(directory):
            return 0
        image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
        return sum(
            1 for f in os.listdir(directory)
            if f.endswith(image_extensions)
        )

    def _generate_imagenet_images(
        self,
        fake_dir: str,
        n_samples: int = 50000,
        cfg_scale: float = 2.9,
        batch_size: int = 128,
        phase1_steps: int = 32,
        phase2_steps: int = 4,
        phase2_cfg_scale: Optional[float] = None,
        force_regenerate: bool = False,
        seed: int = 42,
    ) -> None:
        """Generates ImageNet class-conditional images and saves them to disk.

        Distributes generation evenly across all 1000 ImageNet classes
        (50 images per class for 50K total). This matches standard ImageNet
        evaluation practice and ensures all classes are represented.

        Skips generation if ``fake_dir`` already contains ``n_samples`` images
        and ``force_regenerate=False``.

        Args:
            fake_dir: Directory to save generated images.
            n_samples: Total number of images to generate.
                Config: ``evaluation.imagenet.n_samples = 50000``.
            cfg_scale: CFG scale for Phase 1.
                Config: ``inference.cfg.scale = 2.9``.
            batch_size: Generation batch size.
                Config: ``inference.speed_accuracy_sweep.batch_size = 128``.
            phase1_steps: Phase 1 AR steps.
                Config: ``inference.phase1_steps = 32``.
            phase2_steps: Phase 2 AR steps.
                Config: ``inference.phase2_steps = 4``.
            phase2_cfg_scale: CFG scale for Phase 2. If None, uses cfg_scale.
            force_regenerate: If True, regenerate even if images already exist.
            seed: Random seed for reproducibility. Config: ``seed = 42``.
        """
        # Check if images already exist.
        existing_count: int = self._count_images_in_dir(fake_dir)
        if not force_regenerate and existing_count >= n_samples:
            self.logger.info(
                f"Found {existing_count} existing images in {fake_dir}. "
                f"Skipping generation (force_regenerate=False)."
            )
            return

        self.logger.info(
            f"Generating {n_samples} ImageNet images to {fake_dir}. "
            f"CFG scale: {cfg_scale}, Phase1 steps: {phase1_steps}, "
            f"Phase2 steps: {phase2_steps}."
        )

        # Set seed for reproducibility.
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.model.eval()
        n_classes: int = self.model.config.n_classes  # 1000

        # Resolve Phase 2 CFG scale.
        p2_cfg: float = phase2_cfg_scale if phase2_cfg_scale is not None else cfg_scale

        n_generated: int = 0
        pbar = tqdm(total=n_samples, desc="Generating ImageNet images")

        with torch.no_grad():
            with torch.cuda.amp.autocast(
                enabled=(self.device.type == "cuda")
            ):
                while n_generated < n_samples:
                    # Build a batch of class IDs cycling through all 1000 classes.
                    # This ensures even class distribution across the 50K samples.
                    remaining: int = n_samples - n_generated
                    current_batch_size: int = min(batch_size, remaining)

                    # Cycle through classes: sample i gets class (i % n_classes).
                    class_ids: torch.Tensor = torch.tensor(
                        [
                            (n_generated + j) % n_classes
                            for j in range(current_batch_size)
                        ],
                        dtype=torch.long,
                        device=self.device,
                    )  # [current_batch_size]

                    # Generate images.
                    images: torch.Tensor = self.generator.generate_imagenet(
                        class_ids=class_ids,
                        cfg_scale=cfg_scale,
                        phase1_steps=phase1_steps,
                        phase2_steps=phase2_steps,
                        phase2_cfg_scale=p2_cfg,
                    )
                    # images: [current_batch_size, 3, 256, 256]

                    # Check for NaN/Inf values and skip if found.
                    if torch.isnan(images).any() or torch.isinf(images).any():
                        self.logger.warning(
                            f"NaN/Inf detected in generated images at step "
                            f"{n_generated}. Skipping this batch."
                        )
                        n_generated += current_batch_size
                        pbar.update(current_batch_size)
                        continue

                    # Save images to disk.
                    self._save_images(images, fake_dir, start_idx=n_generated)
                    n_generated += current_batch_size
                    pbar.update(current_batch_size)

        pbar.close()
        self.logger.info(
            f"Generated {n_generated} ImageNet images to {fake_dir}."
        )

    def _generate_coco_images(
        self,
        text_embeddings: torch.Tensor,
        fake_dir: str,
        captions: Optional[List[str]] = None,
        cfg_scale: float = 2.9,
        batch_size: int = 128,
        phase1_steps: int = 32,
        phase2_steps: int = 4,
        null_text_embedding: Optional[torch.Tensor] = None,
        force_regenerate: bool = False,
        seed: int = 42,
    ) -> Optional[str]:
        """Generates COCO text-conditional images and saves them to disk.

        Also saves a prompt mapping JSON file (``prompts.json``) in ``fake_dir``
        for use by T2I-CompBench evaluation.

        Skips generation if ``fake_dir`` already contains enough images and
        ``force_regenerate=False``.

        Args:
            text_embeddings: CLIP text embeddings for all prompts,
                shape ``[N, 77, 768]``. Must be on CPU (moved to device
                inside generation loop).
            fake_dir: Directory to save generated images.
            captions: Optional list of N caption strings corresponding to
                ``text_embeddings``. Used to build the T2I-CompBench prompt
                mapping JSON. If None, prompt mapping is not saved.
            cfg_scale: CFG scale for Phase 1.
                Config: ``inference.cfg.scale = 2.9``.
            batch_size: Generation batch size.
            phase1_steps: Phase 1 AR steps.
                Config: ``inference.phase1_steps = 32``.
            phase2_steps: Phase 2 AR steps.
                Config: ``inference.phase2_steps = 4``.
            null_text_embedding: CLIP embedding of empty string for CFG,
                shape ``[1, 77, 768]``. If None, zeros are used.
            force_regenerate: If True, regenerate even if images already exist.
            seed: Random seed for reproducibility.

        Returns:
            Path to the saved ``prompts.json`` file, or None if captions
            were not provided.
        """
        n_samples: int = text_embeddings.shape[0]

        # Check if images already exist.
        existing_count: int = self._count_images_in_dir(fake_dir)
        if not force_regenerate and existing_count >= n_samples:
            self.logger.info(
                f"Found {existing_count} existing images in {fake_dir}. "
                f"Skipping generation (force_regenerate=False)."
            )
            # Return existing prompt file path if it exists.
            prompt_json_path: str = os.path.join(fake_dir, "prompts.json")
            return prompt_json_path if os.path.isfile(prompt_json_path) else None

        self.logger.info(
            f"Generating {n_samples} COCO images to {fake_dir}. "
            f"CFG scale: {cfg_scale}, Phase1 steps: {phase1_steps}, "
            f"Phase2 steps: {phase2_steps}."
        )

        # Set seed for reproducibility.
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.model.eval()

        # Build prompt mapping for T2I-CompBench.
        prompt_map: Dict[str, str] = {}

        n_generated: int = 0
        pbar = tqdm(total=n_samples, desc="Generating COCO images")

        with torch.no_grad():
            with torch.cuda.amp.autocast(
                enabled=(self.device.type == "cuda")
            ):
                for i in range(0, n_samples, batch_size):
                    batch_end: int = min(i + batch_size, n_samples)
                    batch_emb: torch.Tensor = text_embeddings[i:batch_end].to(
                        self.device
                    )  # [B, 77, 768]

                    # Generate images.
                    images: torch.Tensor = self.generator.generate_coco(
                        text_embeddings=batch_emb,
                        cfg_scale=cfg_scale,
                        phase1_steps=phase1_steps,
                        phase2_steps=phase2_steps,
                        null_text_embedding=null_text_embedding,
                    )
                    # images: [B, 3, 256, 256]

                    # Check for NaN/Inf values.
                    if torch.isnan(images).any() or torch.isinf(images).any():
                        self.logger.warning(
                            f"NaN/Inf detected in COCO generated images at "
                            f"step {n_generated}. Skipping this batch."
                        )
                        n_generated += batch_end - i
                        pbar.update(batch_end - i)
                        continue

                    # Save images to disk.
                    self._save_images(images, fake_dir, start_idx=n_generated)

                    # Build prompt mapping entries.
                    if captions is not None:
                        for j in range(batch_end - i):
                            global_idx: int = n_generated + j
                            if global_idx < len(captions):
                                filename: str = f"{global_idx:06d}.png"
                                prompt_map[filename] = captions[global_idx]

                    n_generated += batch_end - i
                    pbar.update(batch_end - i)

        pbar.close()
        self.logger.info(
            f"Generated {n_generated} COCO images to {fake_dir}."
        )

        # Save prompt mapping JSON for T2I-CompBench.
        prompt_json_path_out: Optional[str] = None
        if captions is not None and prompt_map:
            prompt_json_path_out = os.path.join(fake_dir, "prompts.json")
            with open(prompt_json_path_out, "w", encoding="utf-8") as f:
                json.dump(prompt_map, f, ensure_ascii=False, indent=2)
            self.logger.info(
                f"Saved prompt mapping to {prompt_json_path_out} "
                f"({len(prompt_map)} entries)."
            )

        return prompt_json_path_out

    def _run_clean_fid(
        self,
        real_dir: str,
        fake_dir: str,
        mode: str = "clean",
    ) -> float:
        """Computes FID between real and fake image directories using clean-fid.

        Uses ``cleanfid.fid.compute_fid()`` which applies a consistent
        antialiased resize to 299×299 before InceptionV3 feature extraction.
        This avoids the PIL vs. OpenCV discrepancy that inflates FID scores
        in naive implementations.

        Paper reference (Section 4.2):
            "we use Fréchet Inception Distance (FID) ... on 50K generated
            samples to measure the image quality on ImageNet."

        Args:
            real_dir: Directory containing real reference images.
            fake_dir: Directory containing generated images.
            mode: clean-fid evaluation mode. ``"clean"`` uses the antialiased
                resize pipeline. Default: ``"clean"``.

        Returns:
            Scalar FID score (lower is better).

        Raises:
            ImportError: If ``cleanfid`` is not installed.
            FileNotFoundError: If either directory does not exist or is empty.
        """
        try:
            from cleanfid import fid as cleanfid_module
        except ImportError as exc:
            raise ImportError(
                "cleanfid is required for FID computation. "
                "Install it with: pip install clean-fid==0.1.35"
            ) from exc

        if not os.path.isdir(real_dir):
            raise FileNotFoundError(
                f"Real image directory not found: '{real_dir}'. "
                "Please ensure the dataset is available at the configured path."
            )
        if not os.path.isdir(fake_dir):
            raise FileNotFoundError(
                f"Fake image directory not found: '{fake_dir}'."
            )

        real_count: int = self._count_images_in_dir(real_dir)
        fake_count: int = self._count_images_in_dir(fake_dir)
        self.logger.info(
            f"Computing FID: {real_count} real images in {real_dir}, "
            f"{fake_count} fake images in {fake_dir}."
        )

        fid_score: float = cleanfid_module.compute_fid(
            fdir1=real_dir,
            fdir2=fake_dir,
            mode=mode,
            num_workers=8,
            batch_size=128,
            device=self.device,
            verbose=True,
        )

        self.logger.info(f"FID score: {fid_score:.4f}")
        return float(fid_score)

    # ------------------------------------------------------------------
    # Public API: metric computation methods
    # ------------------------------------------------------------------

    def compute_fid_imagenet(
        self,
        dataloader: Any,
        n_samples: int = 50000,
        cfg_scale: float = 2.9,
        batch_size: int = 128,
        phase1_steps: int = 32,
        phase2_steps: int = 4,
        phase2_cfg_scale: Optional[float] = None,
        real_dir: Optional[str] = None,
        force_regenerate: bool = False,
        seed: int = 42,
    ) -> float:
        """Computes FID for ImageNet class-conditional generation.

        Generates ``n_samples`` images distributed evenly across all 1000
        ImageNet classes, saves them to ``self.fake_dir``, and computes FID
        against the ImageNet validation set using clean-fid.

        Paper reference (Section 4.2):
            "we use Fréchet Inception Distance (FID) ... on 50K generated
            samples to measure the image quality on ImageNet."

        Config alignment:
            evaluation.imagenet.n_samples = 50000
            inference.phase1_steps = 32
            inference.phase2_steps = 4
            inference.cfg.scale = 2.9
            data.imagenet.val_root = "data/imagenet/val"

        Args:
            dataloader: ImageNet validation DataLoader. Used to determine the
                real image directory path (``dataloader.dataset.root``).
                The actual images are read from disk by clean-fid, not from
                the DataLoader.
            n_samples: Total number of images to generate.
                Config: ``evaluation.imagenet.n_samples = 50000``.
                Default: 50000.
            cfg_scale: CFG scale for Phase 1.
                Config: ``inference.cfg.scale = 2.9``. Default: 2.9.
            batch_size: Generation batch size.
                Config: ``inference.speed_accuracy_sweep.batch_size = 128``.
                Default: 128.
            phase1_steps: Phase 1 AR steps.
                Config: ``inference.phase1_steps = 32``. Default: 32.
            phase2_steps: Phase 2 AR steps.
                Config: ``inference.phase2_steps = 4``. Default: 4.
            phase2_cfg_scale: CFG scale for Phase 2. If None, uses cfg_scale.
                Pass ``1.0`` for the w/o CFG setting (Phase 2 CFG disabled).
                Default: None.
            real_dir: Path to real ImageNet validation images. If None,
                attempts to extract from ``dataloader.dataset.root``.
                Config: ``data.imagenet.val_root = "data/imagenet/val"``.
            force_regenerate: If True, regenerate images even if they already
                exist in ``self.fake_dir``. Default: False.
            seed: Random seed for reproducibility.
                Config: ``seed = 42``. Default: 42.

        Returns:
            Scalar FID score (lower is better). Hi-MAR-B achieves 1.93 with
            CFG per Table 2 of the paper.
        """
        self.model.eval()

        # ------------------------------------------------------------------
        # Step 1: Generate images (or reuse existing ones).
        # ------------------------------------------------------------------
        self._generate_imagenet_images(
            fake_dir=self.fake_dir,
            n_samples=n_samples,
            cfg_scale=cfg_scale,
            batch_size=batch_size,
            phase1_steps=phase1_steps,
            phase2_steps=phase2_steps,
            phase2_cfg_scale=phase2_cfg_scale,
            force_regenerate=force_regenerate,
            seed=seed,
        )

        # ------------------------------------------------------------------
        # Step 2: Determine real image directory.
        # ------------------------------------------------------------------
        if real_dir is None:
            # Try to extract from dataloader dataset.
            if hasattr(dataloader, "dataset") and hasattr(
                dataloader.dataset, "root"
            ):
                real_dir = dataloader.dataset.root
            else:
                raise ValueError(
                    "real_dir must be provided or dataloader.dataset.root "
                    "must be accessible. Set data.imagenet.val_root in config.yaml."
                )

        # ------------------------------------------------------------------
        # Step 3: Compute FID via clean-fid.
        # ------------------------------------------------------------------
        fid_score: float = self._run_clean_fid(
            real_dir=real_dir,
            fake_dir=self.fake_dir,
        )

        return fid_score

    def compute_fid_coco(
        self,
        dataloader: Any,
        n_samples: int = 30000,
        cfg_scale: float = 2.9,
        batch_size: int = 128,
        phase1_steps: int = 32,
        phase2_steps: int = 4,
        null_text_embedding: Optional[torch.Tensor] = None,
        real_dir: Optional[str] = None,
        force_regenerate: bool = False,
        seed: int = 42,
    ) -> float:
        """Computes FID for MS-COCO text-to-image generation.

        Randomly draws ``n_samples`` prompts from the COCO validation set,
        generates images, and computes FID following the U-ViT evaluation
        protocol.

        Paper reference (Section 4.2):
            "we randomly draw 30K prompts from the validation set and generate
            samples on these prompts as U-ViT. We report the FID score as the
            main metric."

        Config alignment:
            evaluation.coco.n_samples = 30000
            inference.phase1_steps = 32
            inference.phase2_steps = 4
            inference.cfg.scale = 2.9
            data.coco.val_root = "data/coco/val2017"

        Args:
            dataloader: COCO validation DataLoader returning
                ``(img_256, img_128, text_emb)`` tuples. The first
                ``n_samples`` batches are used (DataLoader should be shuffled
                for the random draw).
            n_samples: Number of prompts to draw and images to generate.
                Config: ``evaluation.coco.n_samples = 30000``. Default: 30000.
            cfg_scale: CFG scale for Phase 1.
                Config: ``inference.cfg.scale = 2.9``. Default: 2.9.
            batch_size: Generation batch size. Default: 128.
            phase1_steps: Phase 1 AR steps.
                Config: ``inference.phase1_steps = 32``. Default: 32.
            phase2_steps: Phase 2 AR steps.
                Config: ``inference.phase2_steps = 4``. Default: 4.
            null_text_embedding: CLIP embedding of empty string for CFG,
                shape ``[1, 77, 768]``. If None, zeros are used (suboptimal).
                Pass ``coco_dataset.null_text_embedding`` for best results.
            real_dir: Path to real COCO validation images. If None, attempts
                to extract from ``dataloader.dataset.root``.
                Config: ``data.coco.val_root = "data/coco/val2017"``.
            force_regenerate: If True, regenerate images even if they already
                exist. Default: False.
            seed: Random seed for reproducibility.
                Config: ``seed = 42``. Default: 42.

        Returns:
            Scalar FID score (lower is better). Hi-MAR-S achieves 4.77 per
            Table 3 of the paper.
        """
        self.model.eval()

        # ------------------------------------------------------------------
        # Step 1: Collect n_samples text embeddings from the validation loader.
        # The DataLoader's shuffle=True provides the random draw of 30K prompts.
        # ------------------------------------------------------------------
        self.logger.info(
            f"Collecting {n_samples} COCO text embeddings from validation set."
        )
        text_embeddings_list: List[torch.Tensor] = []
        total_collected: int = 0

        for batch in tqdm(dataloader, desc="Collecting text embeddings"):
            # batch: (img_256, img_128, text_emb) from COCODataset.collate_fn
            text_emb: torch.Tensor = batch[2]  # [B, 77, 768] on CPU
            text_embeddings_list.append(text