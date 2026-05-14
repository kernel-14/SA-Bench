```python
## evaluation/image_eval.py
"""Image segmentation evaluator for SAM 2 zero-shot evaluation.

Implements ImageEvaluator for 1-click and 5-click mIoU evaluation across
37 zero-shot image datasets (23 original SAM datasets + 14 new video datasets
evaluated as images), as described in Section 6.3 and Appendix F.4 of the
SAM 2 paper.

Key design decisions:
    - torch.compile applied to image encoder once in __init__ (Appendix D.3)
    - Image encoder runs once per image, reused for all masks (Section 4)
    - Empty memory bank for all image evaluations (model behaves like SAM)
    - Initial click at GT mask center; correction clicks at error region center
    - Accumulate all clicks across iterations (full history fed to decoder)
    - bfloat16 mixed precision throughout (Appendix D.3, Table 12)

Config references (config.yaml evaluation section):
    evaluation.image_segmentation.num_click_settings: [1, 5]
    evaluation.benchmarking.image_batch_size: 10
    evaluation.benchmarking.compile_image_encoder: true
    evaluation.benchmarking.precision: "bfloat16"
    evaluation.image_segmentation.num_sa23_datasets: 23
    evaluation.image_segmentation.num_new_video_datasets: 14
    evaluation.image_segmentation.total_datasets: 37

Paper references:
    Section 6.3: "We evaluate SAM 2 on the Segment Anything task across
        37 zero-shot datasets."
    Appendix D.3: "We compile the image encoder with torch.compile for all
        SAM 2 models ... The FPS measurements for the SA task were conducted
        using a batch size of 10 images."
    Appendix F.1.3: "the initial click is placed on the object center and
        subsequent clicks are obtained from the center of the error region."
    Table 5, Table 16: 1-click and 5-click mIoU results.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from datasets import PromptInput
from evaluation.metrics import MIoUMetric
from models.memory_bank import MemoryBank
from models.sam2 import SAM2FrameOutput, SAM2Model
from utils.click_sampler import ClickSampler
from utils.mask_utils import MaskUtils

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default constants from config.yaml evaluation section
# ---------------------------------------------------------------------------

_DEFAULT_NUM_CLICK_SETTINGS: List[int] = [1, 5]
_DEFAULT_IMAGE_BATCH_SIZE: int = 10
_DEFAULT_COMPILE_IMAGE_ENCODER: bool = True
_DEFAULT_PRECISION: str = "bfloat16"
_DEFAULT_OCCLUSION_THRESHOLD: float = 0.5
_DEFAULT_MASK_THRESHOLD: float = 0.0
_DEFAULT_DEVICE: str = "cuda"

# Number of warmup batches before FPS timing
_FPS_WARMUP_BATCHES: int = 10
# Number of timed batches for FPS measurement
_FPS_TIMED_BATCHES: int = 100


class ImageEvaluator:
    """Evaluates SAM 2 on zero-shot image segmentation across 37 datasets.

    Implements the interactive image segmentation evaluation protocol from
    Section 6.3 and Appendix F.4 of the SAM 2 paper. Measures 1-click and
    5-click mIoU by simulating iterative click-based segmentation.

    The image encoder is compiled with torch.compile in __init__ for maximum
    inference speed, matching the benchmarking setup in Appendix D.3.

    For each image:
        1. Run image encoder once → (frame_embed, skip_features)
        2. For each GT mask instance:
           a. Place initial click at GT mask center
           b. Run mask decoder → get prediction
           c. Place correction clicks at error region center (for n_clicks > 1)
           d. Re-run mask decoder with accumulated clicks
           e. Record final IoU

    The memory bank is always empty for image evaluation — the model behaves
    identically to SAM (Section 4: "When applied to images, the memory is
    empty and the model behaves like SAM").

    Args:
        model: Initialized SAM2Model. Will be moved to device and set to eval
            mode. The image encoder will be compiled with torch.compile if
            compile_image_encoder is True in config.
        config: Full config dict loaded from config.yaml. The evaluator reads
            from config['evaluation']['image_segmentation'] and
            config['evaluation']['benchmarking'].
        device: Target device string (e.g., "cuda:0", "cuda", "cpu").
            Defaults to "cuda".

    Example:
        evaluator = ImageEvaluator(model=sam2, config=cfg, device="cuda")
        results = evaluator.evaluate(dataset=lvis_dataset, n_clicks=1)
        print(f"1-click mIoU: {results['miou']:.1f}")
        all_results = evaluator.evaluate_all_click_settings(lvis_dataset, "LVIS")
        print(f"1-click: {all_results['1_click_miou']:.1f}, "
              f"5-click: {all_results['5_click_miou']:.1f}")
    """

    def __init__(
        self,
        model: SAM2Model,
        config: Dict[str, Any],
        device: str = _DEFAULT_DEVICE,
    ) -> None:
        self.model: SAM2Model = model
        self.config: Dict[str, Any] = config
        self.device: str = device

        # ------------------------------------------------------------------
        # Extract evaluation settings from config with defaults
        # ------------------------------------------------------------------
        eval_cfg: Dict[str, Any] = config.get("evaluation", {})
        image_seg_cfg: Dict[str, Any] = eval_cfg.get("image_segmentation", {})
        benchmarking_cfg: Dict[str, Any] = eval_cfg.get("benchmarking", {})

        # Number of click settings to evaluate: [1, 5] per Table 5
        num_click_settings_raw: Any = image_seg_cfg.get(
            "num_click_settings", _DEFAULT_NUM_CLICK_SETTINGS
        )
        if isinstance(num_click_settings_raw, (list, tuple)):
            self.num_click_settings: List[int] = [
                int(x) for x in num_click_settings_raw
            ]
        else:
            self.num_click_settings = _DEFAULT_NUM_CLICK_SETTINGS

        # Batch size for FPS measurement (Appendix D.3: batch_size=10 yields highest FPS)
        self.batch_size_fps: int = int(
            benchmarking_cfg.get("image_batch_size", _DEFAULT_IMAGE_BATCH_SIZE)
        )

        # Whether to compile image encoder with torch.compile (Appendix D.3)
        self.compile_image_encoder: bool = bool(
            benchmarking_cfg.get(
                "compile_image_encoder", _DEFAULT_COMPILE_IMAGE_ENCODER
            )
        )

        # Mixed precision setting (Appendix D.3, Table 12: bfloat16)
        self.precision: str = str(
            benchmarking_cfg.get("precision", _DEFAULT_PRECISION)
        )

        # Occlusion threshold: if occlusion_score > this, output empty mask
        self.occlusion_threshold: float = float(
            benchmarking_cfg.get("occlusion_threshold", _DEFAULT_OCCLUSION_THRESHOLD)
        )

        # Mask threshold for binarizing logits (config: model.mask_threshold: 0.0)
        model_cfg: Dict[str, Any] = config.get("model", {})
        self.mask_threshold: float = float(
            model_cfg.get("mask_threshold", _DEFAULT_MASK_THRESHOLD)
        )

        # ------------------------------------------------------------------
        # Move model to device and set to eval mode
        # ------------------------------------------------------------------
        self.model = self.model.to(device)
        self.model.eval()

        # ------------------------------------------------------------------
        # Apply torch.compile to image encoder for maximum inference speed
        # From Appendix D.3: "We compile the image encoder with torch.compile
        # for all SAM 2 models and do the same for SAM and HQ-SAM for direct
        # comparison on the SA task."
        # ------------------------------------------------------------------
        if self.compile_image_encoder and torch.cuda.is_available():
            try:
                self.model.image_encoder = torch.compile(
                    self.model.image_encoder,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
                logger.info(
                    "ImageEvaluator: torch.compile applied to image encoder "
                    "(mode='reduce-overhead')."
                )
            except Exception as exc:
                logger.warning(
                    "ImageEvaluator: torch.compile failed: %s. "
                    "Proceeding without compilation.",
                    exc,
                )
        elif self.compile_image_encoder:
            logger.info(
                "ImageEvaluator: torch.compile requested but CUDA not available. "
                "Skipping compilation."
            )

        # ------------------------------------------------------------------
        # Instantiate shared utilities
        # ------------------------------------------------------------------
        self.click_sampler: ClickSampler = ClickSampler()
        self.miou_metric: MIoUMetric = MIoUMetric()
        self.mask_utils: MaskUtils = MaskUtils()

        logger.info(
            "ImageEvaluator initialized: device=%s, num_click_settings=%s, "
            "compile_encoder=%s, precision=%s, batch_size_fps=%d",
            device,
            self.num_click_settings,
            self.compile_image_encoder,
            self.precision,
            self.batch_size_fps,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        dataset: Any,
        n_clicks: int = 1,
    ) -> Dict[str, Any]:
        """Run zero-shot image segmentation evaluation for a given click count.

        Iterates over all images in the dataset, evaluates each GT mask
        instance with iterative click-based segmentation, and returns the
        mean IoU across all instances.

        The image encoder runs once per image and its output is reused for
        all mask instances in that image (efficiency). The memory bank is
        always empty (image mode — model behaves like SAM).

        Args:
            dataset: Any iterable dataset yielding sample dicts with keys:
                - "image": Tensor[C, H, W] float32, ImageNet-normalized
                - "masks": List[Tensor[H, W]] binary float32, one per instance
                - "image_id": str unique identifier
                - "dataset_name": str name of the source dataset
                - "original_size": Tuple[int, int] (H, W)
                Compatible with ImageDataset.__getitem__ output format.
            n_clicks: Number of clicks to simulate per mask instance.
                Use 1 for 1-click mIoU and 5 for 5-click mIoU (Table 5).
                Defaults to 1.

        Returns:
            Dict with keys:
                - "miou": float — mean IoU across all instances (0-100 scale)
                - "n_clicks": int — number of clicks used
                - "num_images": int — number of images evaluated
                - "num_instances": int — total number of mask instances evaluated
                - "all_ious": List[float] — per-instance IoU values (0-1 scale)

        Note:
            Returns miou=0.0 if no valid instances are found in the dataset.
        """
        self.model.eval()

        all_ious: List[float] = []
        num_images: int = 0
        num_instances: int = 0

        # Determine autocast dtype
        autocast_dtype: torch.dtype = (
            torch.bfloat16 if self.precision == "bfloat16" else torch.float32
        )
        use_autocast: bool = (
            self.precision in ("bfloat16", "float16")
            and torch.cuda.is_available()
        )

        with torch.no_grad():
            for sample in dataset:
                # ------------------------------------------------------------------
                # Extract image and masks from sample dict
                # ------------------------------------------------------------------
                image: Optional[Tensor] = None
                gt_masks: List[Tensor] = []

                if isinstance(sample, dict):
                    image = sample.get("image")
                    masks_raw: Any = sample.get("masks", [])
                    if isinstance(masks_raw, (list, tuple)):
                        gt_masks = [
                            m for m in masks_raw
                            if isinstance(m, Tensor) and m.numel() > 0
                        ]
                    elif isinstance(masks_raw, Tensor):
                        # Stacked tensor [N, H, W] — split into list
                        if masks_raw.ndim == 3:
                            gt_masks = [masks_raw[i] for i in range(masks_raw.shape[0])]
                        elif masks_raw.ndim == 2:
                            gt_masks = [masks_raw]
                else:
                    logger.warning(
                        "ImageEvaluator.evaluate: Expected dict sample, "
                        "got %s. Skipping.",
                        type(sample).__name__,
                    )
                    continue

                if image is None:
                    logger.debug(
                        "ImageEvaluator.evaluate: Sample has no 'image' key. "
                        "Skipping."
                    )
                    continue

                if len(gt_masks) == 0:
                    logger.debug(
                        "ImageEvaluator.evaluate: Sample has no valid masks. "
                        "Skipping."
                    )
                    num_images += 1
                    continue

                # ------------------------------------------------------------------
                # Evaluate this image
                # ------------------------------------------------------------------
                try:
                    if use_autocast:
                        with torch.autocast(
                            device_type="cuda", dtype=autocast_dtype
                        ):
                            image_result: Dict[str, Any] = self._evaluate_image(
                                image=image,
                                gt_masks=gt_masks,
                                n_clicks=n_clicks,
                            )
                    else:
                        image_result = self._evaluate_image(
                            image=image,
                            gt_masks=gt_masks,
                            n_clicks=n_clicks,
                        )
                except Exception as exc:
                    logger.warning(
                        "ImageEvaluator.evaluate: Failed to evaluate image: %s. "
                        "Skipping.",
                        exc,
                    )
                    num_images += 1
                    continue

                # Accumulate per-instance IoU values
                per_mask_ious: List[float] = image_result.get("per_mask_ious", [])
                all_ious.extend(per_mask_ious)
                num_instances += len(per_mask_ious)
                num_images += 1

                if num_images % 100 == 0:
                    current_miou: float = (
                        float(np.mean(all_ious)) * 100.0 if all_ious else 0.0
                    )
                    logger.debug(
                        "ImageEvaluator: %d images, %d instances, "
                        "running mIoU=%.2f",
                        num_images,
                        num_instances,
                        current_miou,
                    )

        # ------------------------------------------------------------------
        # Compute final mean IoU
        # ------------------------------------------------------------------
        if len(all_ious) == 0:
            logger.warning(
                "ImageEvaluator.evaluate: No valid instances found. "
                "Returning mIoU=0.0."
            )
            return {
                "miou": 0.0,
                "n_clicks": n_clicks,
                "num_images": num_images,
                "num_instances": 0,
                "all_ious": [],
            }

        # Paper reports mIoU on 0-100 scale (e.g., 58.9 in Table 5)
        mean_iou_0_1: float = float(np.mean(all_ious))
        mean_iou_pct: float = mean_iou_0_1 * 100.0

        logger.info(
            "ImageEvaluator.evaluate: n_clicks=%d | mIoU=%.2f%% | "
            "%d images, %d instances",
            n_clicks,
            mean_iou_pct,
            num_images,
            num_instances,
        )

        return {
            "miou": mean_iou_pct,
            "n_clicks": n_clicks,
            "num_images": num_images,
            "num_instances": num_instances,
            "all_ious": all_ious,
        }

    def _evaluate_image(
        self,
        image: Tensor,
        gt_masks: List[Tensor],
        n_clicks: int,
    ) -> Dict[str, Any]:
        """Evaluate a single image across all its GT mask instances.

        Runs the image encoder once and reuses its output for all mask
        instances. The memory bank is reset before each image to ensure
        clean image-mode inference (no temporal context).

        From Section 4: "The image encoder is only run once for the entire
        interaction and its role is to provide unconditioned tokens
        (feature embeddings) representing each frame."

        Args:
            image: Input image tensor of shape [C, H, W] or [1, C, H, W],
                dtype float32, ImageNet-normalized.
            gt_masks: List of GT binary mask tensors, each of shape [H, W].
                Empty masks (all zeros) are skipped.
            n_clicks: Number of clicks to simulate per mask instance.

        Returns:
            Dict with keys:
                - "per_mask_ious": List[float] — IoU for each valid mask
                  instance (values in [0, 1]).
                - "mean_iou": float — mean IoU over valid instances.
        """
        # ------------------------------------------------------------------
        # Reset memory bank: image mode = empty memory = SAM behavior
        # Section 4: "When applied to images, the memory is empty and the
        # model behaves like SAM."
        # ------------------------------------------------------------------
        self.model.reset_memory()

        # ------------------------------------------------------------------
        # Ensure image has batch dimension: [1, C, H, W]
        # ------------------------------------------------------------------
        if image.ndim == 3:
            image_batched: Tensor = image.unsqueeze(0).to(self.device)
        elif image.ndim == 4:
            image_batched = image.to(self.device)
        else:
            logger.warning(
                "_evaluate_image: Unexpected image shape %s. Skipping.",
                image.shape,
            )
            return {"per_mask_ious": [], "mean_iou": 0.0}

        # ------------------------------------------------------------------
        # Run image encoder ONCE per image (shared across all mask instances)
        # ------------------------------------------------------------------
        frame_embed: Tensor
        skip_features: List[Tensor]
        frame_embed, skip_features = self.model.forward_image(image_batched)
        # frame_embed: [1, C, H/16, W/16]
        # skip_features: [stride4_feat, stride8_feat]

        # ------------------------------------------------------------------
        # Evaluate each GT mask instance independently
        # ------------------------------------------------------------------
        per_mask_ious: List[float] = []

        for gt_mask in gt_masks:
            # Skip empty masks (no valid object to segment)
            gt_mask_np: np.ndarray = gt_mask.detach().cpu().numpy().astype(bool)
            if not gt_mask_np.any():
                logger.debug(
                    "_evaluate_image: Skipping empty GT mask."
                )
                continue

            # Evaluate this mask instance with iterative click simulation
            try:
                iou: float = self._iterative_click_eval(
                    frame_embed=frame_embed,
                    skip_features=skip_features,
                    gt_mask=gt_mask,
                    n_clicks=n_clicks,
                )
                per_mask_ious.append(iou)
            except Exception as exc:
                logger.debug(
                    "_evaluate_image: Failed to evaluate mask instance: %s. "
                    "Skipping.",
                    exc,
                )
                continue

        mean_iou: float = float(np.mean(per_mask_ious)) if per_mask_ious else 0.0

        return {
            "per_mask_ious": per_mask_ious,
            "mean_iou": mean_iou,
        }

    def _iterative_click_eval(
        self,
        frame_embed: Tensor,
        skip_features: List[Tensor],
        gt_mask: Tensor,
        n_clicks: int,
    ) -> float:
        """Simulate iterative click-based segmentation for one GT mask instance.

        Implements the evaluation protocol from Appendix F.1.3:
            - Click 1: initial positive click at GT mask center
            - Clicks 2..n: correction clicks at error region center
            - All accumulated clicks are fed to the decoder at each iteration

        The frame embedding is reused across all click iterations (encoder
        ran once per image). The memory bank is always empty (image mode).

        From Appendix F.1.3: "the initial click is placed on the object center
        and subsequent clicks are obtained from the center of the error region."

        Args:
            frame_embed: Pre-computed frame embedding from the image encoder,
                shape [1, C, H/16, W/16]. Reused across all click iterations.
            skip_features: Pre-computed skip connection features from the image
                encoder. List of [stride4_feat, stride8_feat].
            gt_mask: Ground-truth binary mask of shape [H, W], dtype float32
                or bool. Must be non-empty (caller guarantees this).
            n_clicks: Total number of clicks to simulate. Must be >= 1.

        Returns:
            Final IoU (float in [0.0, 1.0]) after all n_clicks have been
            applied. Returns 0.0 if the evaluation fails.
        """
        if n_clicks < 1:
            n_clicks = 1

        # ------------------------------------------------------------------
        # Prepare GT mask as numpy array for click sampling and IoU computation
        # ------------------------------------------------------------------
        gt_mask_np: np.ndarray = gt_mask.detach().cpu().numpy().astype(bool)

        # ------------------------------------------------------------------
        # Accumulate all click coordinates and labels across iterations
        # ------------------------------------------------------------------
        all_click_coords: List[Tensor] = []   # Each: [2] (x, y) float32
        all_click_labels: List[Tensor] = []   # Each: scalar int64

        # Current predicted mask (updated after each click iteration)
        current_pred_mask_np: Optional[np.ndarray] = None

        # Final IoU after all clicks
        final_iou: float = 0.0

        # ------------------------------------------------------------------
        # Create empty memory bank for image-mode inference
        # ------------------------------------------------------------------
        empty_memory_bank: MemoryBank = self._create_empty_memory_bank()

        for click_idx in range(n_clicks):
            # ------------------------------------------------------------------
            # Determine click coordinates and label for this iteration
            # ------------------------------------------------------------------
            if click_idx == 0:
                # Click 1: initial positive click at GT mask center
                # Appendix F.1.3: "the initial click is placed on the object center"
                center_yx: Tensor = self.click_sampler.get_center_click(gt_mask_np)
                # get_center_click returns [row, col] = [y, x]
                # PromptEncoder expects (x, y) pixel coordinates
                click_xy: Tensor = torch.tensor(
                    [float(center_yx[1]), float(center_yx[0])],
                    dtype=torch.float32,
                )
                click_label: int = 1  # Positive click

            else:
                # Clicks 2..n: correction clicks at error region center
                # Appendix F.1.3: "subsequent clicks are obtained from the
                # center of the error region"
                if current_pred_mask_np is None:
                    # No prediction yet — fall back to GT center
                    center_yx = self.click_sampler.get_center_click(gt_mask_np)
                    click_xy = torch.tensor(
                        [float(center_yx[1]), float(center_yx[0])],
                        dtype=torch.float32,
                    )
                    click_label = 1
                else:
                    # Get correction click at error region center
                    error_yx: Tensor
                    error_label: int
                    error_yx, error_label = self.click_sampler.get_error_region_click(
                        gt_mask=gt_mask_np,
                        pred_mask=current_pred_mask_np,
                    )
                    # Convert [row, col] to (x, y)
                    click_xy = torch.tensor(
                        [float(error_yx[1]), float(error_yx[0])],
                        dtype=torch.float32,
                    )
                    click_label = error_label

            # Accumulate this click
            all_click_coords.append(click_xy)
            all_click_labels.append(
                torch.tensor(click_label, dtype=torch.long)
            )

            # ------------------------------------------------------------------
            # Build PromptInput with ALL accumulated clicks
            # The full click history is fed to the decoder at each iteration.
            # This matches SAM's evaluation protocol where all previous clicks
            # are included as context.
            # ------------------------------------------------------------------
            # Stack accumulated clicks: [k, 2] and [k]
            points_tensor: Tensor = torch.stack(
                all_click_coords, dim=0
            ).unsqueeze(0).to(self.device)
            # Shape: [1, k, 2] — batch dim added for PromptEncoder

            labels_tensor: Tensor = torch.stack(
                all_click_labels, dim=0
            ).unsqueeze(0).to(self.device)
            # Shape: [1, k]

            prompt: PromptInput = PromptInput(
                points=points_tensor.squeeze(0),   # [k, 2]
                point_labels=labels_tensor.squeeze(0),  # [k]
                frame_idx=0,
            )

            # ------------------------------------------------------------------
            # Run mask decoder with accumulated prompts
            # Memory bank is always empty for image evaluation
            # ------------------------------------------------------------------
            try:
                frame_output: SAM2FrameOutput = self.model.forward_video_frame(
                    frame_embed=frame_embed,
                    skip_features=skip_features,
                    prompts=prompt,
                    memory_bank=empty_memory_bank,
                )
            except Exception as exc:
                logger.debug(
                    "_iterative_click_eval: forward_video_frame failed at "
                    "click %d: %s. Returning current IoU.",
                    click_idx + 1,
                    exc,
                )
                break

            # ------------------------------------------------------------------
            # Extract predicted mask from output
            # Use the mask with highest predicted IoU (selected_mask_idx)
            # ------------------------------------------------------------------
            selected_idx: int = frame_output.selected_mask_idx
            pred_mask_logits: Tensor = frame_output.masks[0, selected_idx]
            # Shape: [H_out, W_out] — logits before sigmoid

            # Check occlusion: if model predicts object is absent, use empty mask
            occlusion_prob: float = float(
                torch.sigmoid(frame_output.occlusion_score[0, 0]).item()
            )
            if occlusion_prob > self.occlusion_threshold:
                # Model predicts object is not visible — empty prediction
                pred_mask_np: np.ndarray = np.zeros_like(gt_mask_np, dtype=bool)
            else:
                # Binarize logits: logit > mask_threshold (0.0) ↔ sigmoid > 0.5
                pred_mask_binary: Tensor = (
                    pred_mask_logits > self.mask_threshold
                )

                # Resize predicted mask to match GT mask spatial dimensions if needed
                gt_h: int = gt_mask_np.shape[0]
                gt_w: int = gt_mask_np.shape[1]
                pred_h: int = pred_mask_binary.shape[0]
                pred_w: int = pred_mask_binary.shape[1]

                if pred_h != gt_h or pred_w != gt_w:
                    # Resize using nearest-neighbor to preserve binary values
                    pred_mask_float: Tensor = pred_mask_binary.float().unsqueeze(0).unsqueeze(0)
                    # [1, 1, pred_h, pred_w]
                    pred_mask_resized: Tensor = torch.nn.functional.interpolate(
                        pred_mask_float,
                        size=(gt_h, gt_w),
                        mode="nearest",
                    ).squeeze(0).squeeze(0)
                    # [gt_h, gt_w]
                    pred_mask_np = (pred_mask_resized > 0.5).detach().cpu().numpy()
                else:
                    pred_mask_np = pred_mask_binary.detach().cpu().numpy()

            # ------------------------------------------------------------------
            # Compute IoU between prediction and GT
            # ------------------------------------------------------------------
            current_iou: float = self.mask_utils.compute_iou_pair(
                pred_mask_np, gt_mask_np
            )
            final_iou = current_iou

            # Update current prediction for next correction click
            current_pred_mask_np = pred_mask_np

        return final_iou

    def evaluate_all_click_settings(
        self,
        dataset: Any,
        dataset_name: str = "",
    ) -> Dict[str, float]:
        """Run evaluation for all click settings (1-click and 5-click).

        Convenience method that calls evaluate() for each n_clicks in
        self.num_click_settings and returns a combined result dict.

        Matches the reporting format in Table 5 and Table 16 of the paper,
        which report both 1-click and 5-click mIoU as "X.X (Y.Y)".

        Args:
            dataset: Any iterable dataset yielding sample dicts compatible
                with evaluate(). See evaluate() for the expected format.
            dataset_name: Optional dataset name for logging. Defaults to "".

        Returns:
            Dict with keys:
                - "{n}_click_miou": float — mIoU for each click setting
                  (0-100 scale, e.g., 58.9 for 1-click on SA-23)
                - "{n}_click_num_instances": int — number of instances evaluated
                - "{n}_click_num_images": int — number of images evaluated
            Example:
                {
                    "1_click_miou": 58.9,
                    "5_click_miou": 81.7,
                    "1_click_num_instances": 12345,
                    "5_click_num_instances": 12345,
                }
        """
        results: Dict[str, float] = {}

        for n_clicks in self.num_click_settings:
            logger.info(
                "ImageEvaluator.evaluate_all_click_settings: "
                "Evaluating dataset='%s' with n_clicks=%d...",
                dataset_name,
                n_clicks,
            )

            try:
                click_result: Dict[str, Any] = self.evaluate(
                    dataset=dataset,
                    n_clicks=n_clicks,
                )
            except Exception as exc:
                logger.warning(
                    "ImageEvaluator.evaluate_all_click_settings: "
                    "Evaluation failed for n_clicks=%d