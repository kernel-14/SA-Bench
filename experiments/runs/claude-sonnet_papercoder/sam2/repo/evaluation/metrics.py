## evaluation/metrics.py
"""Evaluation metrics for SAM 2: J&F for video segmentation and mIoU for images.

This module implements the core evaluation metrics used throughout the SAM 2
paper. It is consumed by InteractiveEvaluator, VOSEvaluator, and ImageEvaluator.

Metrics implemented:
    - J (Jaccard/IoU): region similarity — intersection over union
    - F (F-measure): contour accuracy — boundary F1 score with tolerance
    - J&F: average of J and F, primary video segmentation metric
    - G: YouTubeVOS 2019 metric = (Js + Fs + Ju + Fu) / 4
    - VOST I-metric: J only, per VOST official protocol
    - mIoU: mean IoU over instances, primary image segmentation metric

All methods operate on numpy.ndarray inputs (not tensors) for compatibility
with evaluation code outside PyTorch autograd.

Config references (config.yaml):
    evaluation.interactive.online_iou_threshold: 0.75
    evaluation.semi_supervised_vos.vost_metric: "J_only"

Paper references:
    Section 6: "We report the standard J&F metric (Pont-Tuset et al., 2017)
        for video and mIoU metric for image tasks."
    Section 6.2: "except for on VOST (Tokmakov et al., 2022), where we report
        the J metric following its protocol."
    Table 6: G metric for YouTubeVOS 2019 val evaluation.
    Appendix F.1.3: "If a dataset provides an official evaluation toolkit,
        we use it for evaluation."
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Boundary tolerance coefficient (DAVIS evaluation toolkit convention)
# bound_th = max(1, round(BOUND_TH_COEFF * max(H, W)))
# For 1024×1024: bound_th = max(1, round(0.008 * 1024)) = 8
_BOUND_TH_COEFF: float = 0.008

# Minimum boundary tolerance in pixels
_BOUND_TH_MIN: int = 1

# Epsilon for division stability
_EPS: float = 1e-8


# ---------------------------------------------------------------------------
# JFMetric
# ---------------------------------------------------------------------------


class JFMetric:
    """Computes J&F metric for video object segmentation evaluation.

    Implements the J&F metric from Pont-Tuset et al. (2017) as used throughout
    the SAM 2 paper for video segmentation evaluation. Also provides the G
    metric for YouTubeVOS 2019 and the I-only metric for VOST.

    All methods are stateless and operate on numpy.ndarray inputs.
    The class is instantiated once and reused across all video evaluations.

    Coordinate convention: masks are 2D binary arrays of shape (H, W).
    Values are treated as boolean (any non-zero value = foreground).

    Example:
        metric = JFMetric()
        j = metric.compute_j(pred_mask, gt_mask)
        f = metric.compute_f(pred_mask, gt_mask)
        jf = metric.compute_jf(pred_mask, gt_mask)
        seq_result = metric.compute_sequence_jf(pred_list, gt_list)
    """

    def __init__(self) -> None:
        """Initialize JFMetric. No state to initialize."""
        pass

    # ------------------------------------------------------------------
    # Core per-frame metrics
    # ------------------------------------------------------------------

    def compute_j(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
    ) -> float:
        """Compute Jaccard index (IoU / region similarity) between two masks.

        The J metric measures region overlap between predicted and GT masks.
        It is the primary component of the J&F metric used in DAVIS evaluation.

        Formula: J = |pred ∩ gt| / |pred ∪ gt|

        Edge cases (matching DAVIS evaluation toolkit convention):
            - Both masks empty (all zeros): return 1.0 (perfect agreement on absence)
            - Exactly one mask empty: return 0.0 (complete disagreement)

        This convention is important for occluded frames in SA-V where the
        object is absent (42.5% disappearance rate per Section 5.2).

        Args:
            pred_mask: Predicted binary mask of shape (H, W). Any dtype;
                treated as boolean (non-zero = foreground).
            gt_mask: Ground-truth binary mask of shape (H, W). Same shape
                and dtype conventions as pred_mask.

        Returns:
            Jaccard index as a float in [0.0, 1.0].
        """
        pred_bool: np.ndarray = pred_mask.astype(bool)
        gt_bool: np.ndarray = gt_mask.astype(bool)

        pred_empty: bool = not pred_bool.any()
        gt_empty: bool = not gt_bool.any()

        # Both empty: perfect agreement on absence
        if pred_empty and gt_empty:
            return 1.0

        # Exactly one empty: complete disagreement
        if pred_empty or gt_empty:
            return 0.0

        # Standard IoU computation
        intersection: np.ndarray = np.logical_and(pred_bool, gt_bool)
        union: np.ndarray = np.logical_or(pred_bool, gt_bool)

        intersection_sum: int = int(intersection.sum())
        union_sum: int = int(union.sum())

        if union_sum == 0:
            return 1.0

        return float(intersection_sum) / float(union_sum)

    def compute_f(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
    ) -> float:
        """Compute boundary F-measure (contour accuracy) between two masks.

        The F metric measures how well the predicted mask boundary aligns with
        the GT mask boundary, using a tolerance-based matching scheme. This
        captures fine-grained boundary accuracy that J alone misses.

        The tolerance is proportional to image size:
            bound_th = max(1, round(0.008 × max(H, W)))
        For 1024×1024: bound_th = 8 pixels.

        Edge cases (matching DAVIS evaluation toolkit convention):
            - Both masks empty: return 1.0
            - Exactly one mask empty: return 0.0

        Args:
            pred_mask: Predicted binary mask of shape (H, W).
            gt_mask: Ground-truth binary mask of shape (H, W).

        Returns:
            Boundary F-measure as a float in [0.0, 1.0].
        """
        pred_bool: np.ndarray = pred_mask.astype(bool)
        gt_bool: np.ndarray = gt_mask.astype(bool)

        pred_empty: bool = not pred_bool.any()
        gt_empty: bool = not gt_bool.any()

        # Both empty: perfect agreement on absence
        if pred_empty and gt_empty:
            return 1.0

        # Exactly one empty: complete disagreement
        if pred_empty or gt_empty:
            return 0.0

        return self._boundary_f_measure(pred_bool, gt_bool)

    def compute_jf(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
    ) -> float:
        """Compute J&F metric (average of J and F) for a single frame.

        J&F = (J + F) / 2

        This is the primary per-frame metric used in DAVIS evaluation and
        throughout the SAM 2 paper for video segmentation benchmarks.

        Args:
            pred_mask: Predicted binary mask of shape (H, W).
            gt_mask: Ground-truth binary mask of shape (H, W).

        Returns:
            J&F metric as a float in [0.0, 1.0].
        """
        j: float = self.compute_j(pred_mask, gt_mask)
        f: float = self.compute_f(pred_mask, gt_mask)
        return (j + f) / 2.0

    # ------------------------------------------------------------------
    # Sequence-level metric
    # ------------------------------------------------------------------

    def compute_sequence_jf(
        self,
        pred_masks: List[np.ndarray],
        gt_masks: List[np.ndarray],
    ) -> Dict[str, float]:
        """Compute J&F metric averaged over all frames in a video sequence.

        Implements the sequence-level averaging protocol from DAVIS evaluation:
        1. Compute J and F per frame
        2. Skip frames where GT mask is all zeros (object absent/occluded)
        3. Average J and F over valid frames
        4. Return J, F, and J&F

        This matches the DAVIS evaluation toolkit behavior for occluded frames.
        The SA-V dataset has a 42.5% disappearance rate (Section 5.2), making
        correct handling of absent-object frames critical for accurate evaluation.

        For multi-object videos, this method handles a single object's sequence.
        The caller (VOSEvaluator) is responsible for averaging across objects.

        Args:
            pred_masks: List of T predicted binary masks, each of shape (H, W).
                May contain all-zero masks for frames where the model predicts
                the object is absent.
            gt_masks: List of T ground-truth binary masks, each of shape (H, W).
                All-zero masks indicate the object is absent in that frame
                (occluded or out of frame).

        Returns:
            Dict with keys:
                - "J": Mean Jaccard index over valid frames (float in [0, 1])
                - "F": Mean boundary F-measure over valid frames (float in [0, 1])
                - "JF": Mean J&F over valid frames = (J + F) / 2 (float in [0, 1])
            Returns {"J": 0.0, "F": 0.0, "JF": 0.0} if no valid frames exist.

        Raises:
            ValueError: If pred_masks and gt_masks have different lengths.
        """
        if len(pred_masks) != len(gt_masks):
            raise ValueError(
                f"pred_masks length {len(pred_masks)} != "
                f"gt_masks length {len(gt_masks)}. "
                "Both lists must have the same number of frames."
            )

        if len(pred_masks) == 0:
            logger.warning(
                "compute_sequence_jf: Empty mask lists provided. "
                "Returning zero metrics."
            )
            return {"J": 0.0, "F": 0.0, "JF": 0.0}

        j_values: List[float] = []
        f_values: List[float] = []

        for t, (pred, gt) in enumerate(zip(pred_masks, gt_masks)):
            gt_bool: np.ndarray = gt.astype(bool)

            # Skip frames where GT mask is all zeros (object absent/occluded)
            # This matches DAVIS evaluation toolkit behavior
            if not gt_bool.any():
                logger.debug(
                    "compute_sequence_jf: Skipping frame %d — GT mask is empty "
                    "(object absent/occluded).",
                    t,
                )
                continue

            j_t: float = self.compute_j(pred, gt)
            f_t: float = self.compute_f(pred, gt)

            j_values.append(j_t)
            f_values.append(f_t)

        if len(j_values) == 0:
            # All frames were occluded — return zero metrics
            logger.debug(
                "compute_sequence_jf: All frames were occluded. "
                "Returning zero metrics."
            )
            return {"J": 0.0, "F": 0.0, "JF": 0.0}

        mean_j: float = float(np.mean(j_values))
        mean_f: float = float(np.mean(f_values))
        mean_jf: float = (mean_j + mean_f) / 2.0

        return {"J": mean_j, "F": mean_f, "JF": mean_jf}

    # ------------------------------------------------------------------
    # YouTubeVOS G metric
    # ------------------------------------------------------------------

    def compute_g(
        self,
        Js: float,
        Fs: float,
        Ju: float,
        Fu: float,
    ) -> float:
        """Compute the YouTubeVOS 2019 G metric from four pre-computed averages.

        The G metric is the official YouTubeVOS evaluation metric used in
        Table 6 and Table 19d of the SAM 2 paper. It averages J and F scores
        across seen and unseen object categories:

            G = (Js + Fs + Ju + Fu) / 4

        Where:
            Js: Mean J score on seen categories (present in training)
            Fs: Mean F score on seen categories
            Ju: Mean J score on unseen categories (not in training)
            Fu: Mean F score on unseen categories

        The per-sequence J and F values are computed by compute_sequence_jf(),
        and the category splitting (seen vs. unseen) is handled by VOSEvaluator.
        This method takes the four pre-computed category-level averages.

        Paper reference: Table 6 — "G" column for YouTubeVOS 2019 val.

        Args:
            Js: Mean J score on seen categories, float in [0, 1].
            Fs: Mean F score on seen categories, float in [0, 1].
            Ju: Mean J score on unseen categories, float in [0, 1].
            Fu: Mean F score on unseen categories, float in [0, 1].

        Returns:
            G metric as a float in [0, 1].

        Example:
            # SAM 2 (Hiera-B+) on YouTubeVOS 2019 val: G = 88.6 (Table 6)
            g = metric.compute_g(Js=87.1, Fs=91.6, Ju=83.9, Fu=91.9)
            # g ≈ 88.625
        """
        return (Js + Fs + Ju + Fu) / 4.0

    # ------------------------------------------------------------------
    # VOST I-only metric
    # ------------------------------------------------------------------

    def compute_vost_metric(
        self,
        pred_masks: List[np.ndarray],
        gt_masks: List[np.ndarray],
    ) -> float:
        """Compute the VOST I-metric (J only) for a video sequence.

        VOST (Video Object Segmentation under Transformations) uses only the
        J (Jaccard/IoU) metric, not J&F. From the paper:
        "except for on VOST (Tokmakov et al., 2022), where we report the J
        metric following its protocol."

        Config reference: evaluation.semi_supervised_vos.vost_metric: "J_only"

        VOST has dense annotations (all frames annotated), so no frames are
        skipped for absent objects. However, the standard occluded-frame
        handling from compute_sequence_jf is still applied for consistency.

        Args:
            pred_masks: List of T predicted binary masks, each of shape (H, W).
            gt_masks: List of T ground-truth binary masks, each of shape (H, W).

        Returns:
            Mean J (Jaccard) score over valid frames as a float in [0, 1].
            Returns 0.0 if no valid frames exist.

        Raises:
            ValueError: If pred_masks and gt_masks have different lengths.
        """
        # Reuse compute_sequence_jf and extract only the J component
        result: Dict[str, float] = self.compute_sequence_jf(pred_masks, gt_masks)
        return result["J"]

    # ------------------------------------------------------------------
    # Private helper: boundary F-measure
    # ------------------------------------------------------------------

    def _boundary_f_measure(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
    ) -> float:
        """Compute boundary F-measure between two non-empty binary masks.

        Implements the boundary F-measure from Pont-Tuset et al. (2017):
        1. Extract boundaries of pred and gt via morphological erosion
        2. Dilate each boundary by tolerance bound_th pixels
        3. Compute precision: fraction of pred boundary within dilated GT boundary
        4. Compute recall: fraction of GT boundary within dilated pred boundary
        5. F = 2 × precision × recall / (precision + recall)

        Boundary extraction:
            boundary(mask) = mask AND NOT(erode(mask, 3×3 cross))

        Dilation tolerance:
            bound_th = max(1, round(0.008 × max(H, W)))

        This matches the DAVIS evaluation toolkit implementation.

        Args:
            pred: Non-empty predicted binary mask of shape (H, W), dtype bool.
                Caller guarantees this is non-empty.
            gt: Non-empty ground-truth binary mask of shape (H, W), dtype bool.
                Caller guarantees this is non-empty.

        Returns:
            Boundary F-measure as a float in [0.0, 1.0].
            Returns 0.0 if either boundary is empty after extraction.
        """
        H: int = pred.shape[0]
        W: int = pred.shape[1]

        # Compute boundary tolerance proportional to image size
        # DAVIS convention: bound_th = max(1, round(0.008 * max(H, W)))
        bound_th: int = max(_BOUND_TH_MIN, round(_BOUND_TH_COEFF * max(H, W)))

        # ------------------------------------------------------------------
        # Step 1: Extract boundaries via morphological erosion
        # boundary = mask AND NOT(erode(mask))
        # Using default 3×3 cross structuring element for erosion
        # ------------------------------------------------------------------
        pred_boundary: np.ndarray = self._extract_boundary(pred)
        gt_boundary: np.ndarray = self._extract_boundary(gt)

        # Handle degenerate case: empty boundary (e.g., single-pixel mask)
        if not pred_boundary.any() and not gt_boundary.any():
            # Both boundaries empty — treat as perfect match
            return 1.0

        if not pred_boundary.any() or not gt_boundary.any():
            # One boundary empty — return 0.0
            return 0.0

        # ------------------------------------------------------------------
        # Step 2: Dilate boundaries by tolerance bound_th
        # Creates a "band" around each boundary for tolerance matching
        # ------------------------------------------------------------------
        # Build disk structuring element of radius bound_th
        disk_struct: np.ndarray = self._make_disk(bound_th)

        dilated_gt: np.ndarray = binary_dilation(
            gt_boundary, structure=disk_struct
        ).astype(bool)

        dilated_pred: np.ndarray = binary_dilation(
            pred_boundary, structure=disk_struct
        ).astype(bool)

        # ------------------------------------------------------------------
        # Step 3: Compute precision and recall
        # Precision: fraction of pred boundary pixels within dilated GT boundary
        # Recall: fraction of GT boundary pixels within dilated pred boundary
        # ------------------------------------------------------------------
        pred_boundary_count: int = int(pred_boundary.sum())
        gt_boundary_count: int = int(gt_boundary.sum())

        # Precision: how many pred boundary pixels are within tolerance of GT boundary
        precision_hits: int = int(
            np.logical_and(pred_boundary, dilated_gt).sum()
        )
        precision: float = (
            float(precision_hits) / float(pred_boundary_count)
            if pred_boundary_count > 0
            else 0.0
        )

        # Recall: how many GT boundary pixels are within tolerance of pred boundary
        recall_hits: int = int(
            np.logical_and(gt_boundary, dilated_pred).sum()
        )
        recall: float = (
            float(recall_hits) / float(gt_boundary_count)
            if gt_boundary_count > 0
            else 0.0
        )

        # ------------------------------------------------------------------
        # Step 4: Compute F-measure
        # F = 2 × precision × recall / (precision + recall)
        # ------------------------------------------------------------------
        denom: float = precision + recall
        if denom < _EPS:
            return 0.0

        f_measure: float = 2.0 * precision * recall / denom
        return float(np.clip(f_measure, 0.0, 1.0))

    @staticmethod
    def _extract_boundary(mask: np.ndarray) -> np.ndarray:
        """Extract the boundary of a binary mask via morphological erosion.

        Boundary pixels are those in the mask that would be removed by erosion:
            boundary = mask AND NOT(erode(mask, 3×3 cross))

        This gives a 1-pixel-wide boundary along the inner edge of the mask.

        Args:
            mask: Binary mask of shape (H, W), dtype bool.

        Returns:
            Binary boundary mask of shape (H, W), dtype bool.
            True pixels are on the boundary of the input mask.
        """
        # Use default structuring element (3×3 cross / diamond shape)
        # scipy.ndimage.binary_erosion default structure is a 3×3 cross
        eroded: np.ndarray = binary_erosion(mask).astype(bool)
        boundary: np.ndarray = np.logical_and(mask, np.logical_not(eroded))
        return boundary

    @staticmethod
    def _make_disk(radius: int) -> np.ndarray:
        """Create a circular disk structuring element of the given radius.

        Used for boundary dilation in the F-measure computation. The disk
        approximates a circle with the given radius using a boolean grid.

        Args:
            radius: Radius of the disk in pixels. Must be >= 1.

        Returns:
            Boolean numpy array of shape (2*radius+1, 2*radius+1) where
            True pixels form a filled circle of the given radius.
        """
        if radius < 1:
            # Degenerate case: return 3×3 cross (minimum structuring element)
            struct: np.ndarray = np.array(
                [[False, True, False],
                 [True,  True, True],
                 [False, True, False]],
                dtype=bool,
            )
            return struct

        # Create coordinate grid centered at (radius, radius)
        size: int = 2 * radius + 1
        y_coords, x_coords = np.ogrid[-radius:radius + 1, -radius:radius + 1]

        # Disk: all pixels within radius distance from center
        disk: np.ndarray = (x_coords ** 2 + y_coords ** 2 <= radius ** 2)
        return disk.astype(bool)


# ---------------------------------------------------------------------------
# MIoUMetric
# ---------------------------------------------------------------------------


class MIoUMetric:
    """Computes mean IoU metric for image segmentation evaluation.

    Implements the mIoU metric used for the Segment Anything (SA) task
    evaluation across 37 zero-shot datasets (Section 6.3, Table 5, Table 16).

    The paper reports 1-click and 5-click mIoU:
        "We report the average 1- and 5-click mIoU of SAM 2 compared to SAM"

    The mIoU here is mean IoU over instances (not semantic classes), consistent
    with SAM's evaluation protocol (Kirillov et al., 2023).

    All methods are stateless and operate on numpy.ndarray inputs.
    The class is instantiated once and reused across all image evaluations.

    Example:
        metric = MIoUMetric()
        iou = metric.compute_iou(pred_mask, gt_mask)
        miou = metric.compute_miou(pred_masks_list, gt_masks_list)
    """

    def __init__(self) -> None:
        """Initialize MIoUMetric. No state to initialize."""
        pass

    def compute_iou(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
    ) -> float:
        """Compute intersection-over-union (IoU) between two binary masks.

        This is the per-instance IoU used to compute mIoU across all instances
        in a dataset. Semantically identical to JFMetric.compute_j() but kept
        as a separate method for clarity in the image segmentation context.

        Formula: IoU = |pred ∩ gt| / |pred ∪ gt|

        Edge cases (consistent with JFMetric.compute_j):
            - Both masks empty: return 1.0 (perfect agreement on absence)
            - Exactly one mask empty: return 0.0

        Args:
            pred_mask: Predicted binary mask of shape (H, W). Any dtype;
                treated as boolean (non-zero = foreground).
            gt_mask: Ground-truth binary mask of shape (H, W). Same shape
                and dtype conventions as pred_mask.

        Returns:
            IoU as a float in [0.0, 1.0].
        """
        pred_bool: np.ndarray = pred_mask.astype(bool)
        gt_bool: np.ndarray = gt_mask.astype(bool)

        pred_empty: bool = not pred_bool.any()
        gt_empty: bool = not gt_bool.any()

        # Both empty: perfect agreement on absence
        if pred_empty and gt_empty:
            return 1.0

        # Exactly one empty: complete disagreement
        if pred_empty or gt_empty:
            return 0.0

        # Standard IoU computation using float64 for precision
        intersection: np.ndarray = np.logical_and(pred_bool, gt_bool)
        union: np.ndarray = np.logical_or(pred_bool, gt_bool)

        intersection_sum: int = int(intersection.sum())
        union_sum: int = int(union.sum())

        if union_sum == 0:
            return 1.0

        return float(intersection_sum) / float(union_sum)

    def compute_miou(
        self,
        pred_masks: List[np.ndarray],
        gt_masks: List[np.ndarray],
    ) -> float:
        """Compute mean IoU over a list of (predicted, ground-truth) mask pairs.

        Computes IoU for each pair and returns the mean. This gives the mean
        IoU across all instances in a dataset, which is the primary metric
        reported in Table 5 and Table 16 of the SAM 2 paper.

        Usage context: In ImageEvaluator, this is called after each click
        iteration. The final reported mIoU is the average over all instances
        across all images in a dataset.

        Args:
            pred_masks: List of N predicted binary masks, each of shape (H, W).
                One mask per instance/object.
            gt_masks: List of N ground-truth binary masks, each of shape (H, W).
                Must have the same length as pred_masks.

        Returns:
            Mean IoU as a float in [0.0, 1.0].
            Returns 0.0 if the input lists are empty.

        Raises:
            ValueError: If pred_masks and gt_masks have different lengths.
        """
        if len(pred_masks) != len(gt_masks):
            raise ValueError(
                f"pred_masks length {len(pred_masks)} != "
                f"gt_masks length {len(gt_masks)}. "
                "Both lists must have the same number of instances."
            )

        if len(pred_masks) == 0:
            logger.warning(
                "compute_miou: Empty mask lists provided. Returning 0.0."
            )
            return 0.0

        ious: List[float] = [
            self.compute_iou(pred, gt)
            for pred, gt in zip(pred_masks, gt_masks)
        ]

        return float(np.mean(ious))
