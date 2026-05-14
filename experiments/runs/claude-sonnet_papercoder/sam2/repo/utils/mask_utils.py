## utils/mask_utils.py
"""Mask utility functions for SAM 2 reproduction.

This module implements the MaskUtils class providing foundational mask
operations used across training (PromptSampler), evaluation (IoU computation
in all evaluators), and post-processing (component removal and hole filling
for auto masklet generation as described in Appendix E.1 of the paper).

All methods operate on numpy.ndarray inputs (not tensors) to stay compatible
with evaluation code that works outside PyTorch autograd.

Config references:
    data_engine.auto_masklet.min_component_area_px: 200
    data_engine.auto_masklet.max_hole_area_px: 200
    evaluation.interactive.online_iou_threshold: 0.75
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pycocotools.mask as coco_mask
from scipy import ndimage as ndi
from skimage import measure


class MaskUtils:
    """Utility class for binary mask operations used throughout SAM 2.

    All methods are stateless and operate on numpy.ndarray inputs.
    The class is instantiated once and reused across training and evaluation.

    Coordinate convention: (row, col) = (y, x) throughout, consistent with
    numpy indexing and ClickSampler.

    Example:
        mask_utils = MaskUtils()
        iou = mask_utils.compute_iou_pair(pred_mask, gt_mask)
        center = mask_utils.get_mask_center(gt_mask)
    """

    def __init__(self) -> None:
        """Initialize MaskUtils. No state to initialize."""
        pass

    # ------------------------------------------------------------------
    # RLE encode / decode
    # ------------------------------------------------------------------

    def mask_to_rle(self, mask: np.ndarray) -> Dict:
        """Encode a binary mask to COCO RLE format.

        Uses pycocotools for COCO-compatible encoding. The mask is converted
        to a Fortran-contiguous uint8 array as required by pycocotools.

        Args:
            mask: 2D binary ndarray of shape (H, W), dtype bool or uint8.

        Returns:
            Dict with keys:
                - "counts": bytes object containing the RLE encoding.
                - "size": [H, W] list of integers.
        """
        if mask.ndim != 2:
            raise ValueError(
                f"Expected 2D mask, got shape {mask.shape}"
            )
        # pycocotools requires Fortran-contiguous uint8 array
        mask_uint8 = np.asfortranarray(mask.astype(np.uint8))
        rle = coco_mask.encode(mask_uint8)
        # Ensure counts is bytes (pycocotools may return bytes or str)
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("utf-8")
        return rle

    def rle_to_mask(self, rle: Dict) -> np.ndarray:
        """Decode a COCO RLE dict back to a binary mask.

        Args:
            rle: Dict with keys "counts" (bytes or str) and "size" [H, W].

        Returns:
            2D binary ndarray of shape (H, W), dtype bool, C-contiguous.
        """
        # pycocotools accepts both bytes and str for counts
        rle_copy = dict(rle)
        if isinstance(rle_copy["counts"], str):
            rle_copy["counts"] = rle_copy["counts"].encode("utf-8")

        decoded = coco_mask.decode(rle_copy)  # Fortran-contiguous uint8
        # Convert to C-contiguous bool array
        return np.ascontiguousarray(decoded).astype(bool)

    # ------------------------------------------------------------------
    # IoU computation
    # ------------------------------------------------------------------

    def compute_iou_pair(
        self,
        mask1: np.ndarray,
        mask2: np.ndarray,
    ) -> float:
        """Compute intersection-over-union (Jaccard index) between two masks.

        This is the J metric used in J&F computation (Pont-Tuset et al., 2017).
        Also used in the SAM+XMem++ baseline reconstruction loop where clicks
        are sampled until IoU > 0.8 (Appendix F.1.4).

        Args:
            mask1: 2D binary ndarray of shape (H, W).
            mask2: 2D binary ndarray of shape (H, W).

        Returns:
            IoU as a float in [0.0, 1.0].
            Returns 1.0 if both masks are empty (perfect match).
            Returns 0.0 if exactly one mask is empty.
        """
        mask1_bool = mask1.astype(bool)
        mask2_bool = mask2.astype(bool)

        intersection = np.logical_and(mask1_bool, mask2_bool).sum()
        union = np.logical_or(mask1_bool, mask2_bool).sum()

        if union == 0:
            # Both masks are empty — treat as perfect match
            return 1.0

        return float(intersection) / float(union)

    # ------------------------------------------------------------------
    # Post-processing: small component removal
    # ------------------------------------------------------------------

    def remove_small_components(
        self,
        mask: np.ndarray,
        min_area: int = 200,
    ) -> np.ndarray:
        """Remove disconnected components with area smaller than min_area.

        From Appendix E.1: "we remove tiny disconnected components with areas
        smaller than 200 pixels." Applied as post-processing to auto masklets.

        Config reference: data_engine.auto_masklet.min_component_area_px: 200

        Args:
            mask: 2D binary ndarray of shape (H, W).
            min_area: Minimum component area in pixels. Components strictly
                smaller than this value are removed. Defaults to 200.

        Returns:
            Cleaned binary ndarray of shape (H, W), dtype bool.
            Returns the input unchanged if it is empty.
        """
        mask_bool = mask.astype(bool)

        if not mask_bool.any():
            return mask_bool

        # Label connected components using 4-connectivity (conservative)
        # structure=None uses cross-shaped (4-connectivity) structuring element
        labeled, num_features = ndi.label(mask_bool)

        if num_features == 0:
            return mask_bool

        # Compute area of each component (label 0 is background)
        component_sizes = np.bincount(labeled.ravel())

        # Build mask of components to keep (area >= min_area)
        # component_sizes[0] is background, skip it
        keep_labels = np.where(component_sizes >= min_area)[0]
        # Always exclude background label 0
        keep_labels = keep_labels[keep_labels != 0]

        if len(keep_labels) == 0:
            return np.zeros_like(mask_bool)

        # Reconstruct mask from kept components
        cleaned = np.isin(labeled, keep_labels)
        return cleaned.astype(bool)

    # ------------------------------------------------------------------
    # Post-processing: hole filling
    # ------------------------------------------------------------------

    def fill_holes(
        self,
        mask: np.ndarray,
        max_hole_area: int = 200,
    ) -> np.ndarray:
        """Fill holes in a segmentation mask if hole area < max_hole_area.

        From Appendix E.1: "we fill in holes in segmentation masks if the area
        of the hole is less than 200 pixels." Applied as the second
        post-processing step for auto masklets.

        Config reference: data_engine.auto_masklet.max_hole_area_px: 200

        A "hole" is a connected region of background pixels fully enclosed by
        foreground pixels (i.e., not touching the image border).

        Args:
            mask: 2D binary ndarray of shape (H, W).
            max_hole_area: Maximum hole area in pixels. Holes strictly smaller
                than this value are filled. Defaults to 200.

        Returns:
            Filled binary ndarray of shape (H, W), dtype bool.
        """
        mask_bool = mask.astype(bool)

        if not mask_bool.any():
            return mask_bool

        h, w = mask_bool.shape

        # Invert mask to find background regions
        inverted = ~mask_bool

        # Label connected components of the inverted mask (background regions)
        labeled_inv, num_features = ndi.label(inverted)

        if num_features == 0:
            # Mask is fully filled — nothing to do
            return mask_bool

        # Identify which components touch the image border
        # A component touches the border if any pixel is on the edge rows/cols
        border_labels = set()
        border_labels.update(labeled_inv[0, :].tolist())    # top row
        border_labels.update(labeled_inv[-1, :].tolist())   # bottom row
        border_labels.update(labeled_inv[:, 0].tolist())    # left col
        border_labels.update(labeled_inv[:, -1].tolist())   # right col
        border_labels.discard(0)  # label 0 is foreground in inverted mask

        # Compute area of each background component
        component_sizes = np.bincount(labeled_inv.ravel())

        # Fill holes: background components that do NOT touch the border
        # and have area < max_hole_area
        filled = mask_bool.copy()
        for label_id in range(1, num_features + 1):
            if label_id in border_labels:
                # This is the true background — do not fill
                continue
            area = component_sizes[label_id] if label_id < len(component_sizes) else 0
            if area < max_hole_area:
                # This is a small enclosed hole — fill it
                filled[labeled_inv == label_id] = True

        return filled.astype(bool)

    # ------------------------------------------------------------------
    # Mask center extraction
    # ------------------------------------------------------------------

    def get_mask_center(
        self,
        mask: np.ndarray,
    ) -> Tuple[int, int]:
        """Get the centroid of a binary mask for initial click placement.

        From Appendix F.1.2: "we place an initial click on the first frame at
        the center of the object's ground-truth mask."
        From Appendix F.1.3: "the initial click is placed on the object center."

        The centroid is more robust than the bounding box center for irregular
        shapes.

        Args:
            mask: 2D binary ndarray of shape (H, W).

        Returns:
            (row, col) tuple of integers representing the centroid.
            Returns (H//2, W//2) as fallback for empty masks.
        """
        h, w = mask.shape
        mask_bool = mask.astype(bool)

        ys, xs = np.where(mask_bool)

        if len(ys) == 0:
            # Empty mask — return image center as fallback
            return (h // 2, w // 2)

        cy = int(round(float(np.mean(ys))))
        cx = int(round(float(np.mean(xs))))

        # Clamp to valid range
        cy = max(0, min(cy, h - 1))
        cx = max(0, min(cx, w - 1))

        return (cy, cx)

    # ------------------------------------------------------------------
    # Error region computation
    # ------------------------------------------------------------------

    def get_error_region(
        self,
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
    ) -> np.ndarray:
        """Compute the dominant error region between GT and predicted masks.

        From Appendix F.1.2: "we interactively add clicks based on the center
        of the error region (between the ground-truth mask and the predicted
        segments)." Also used in PromptSampler.sample_correction_clicks()
        during training (Appendix D.2.2).

        The dominant error region is whichever is larger between:
        - False negatives (FN): GT foreground missed by prediction
          → requires a positive correction click
        - False positives (FP): predicted foreground not in GT
          → requires a negative correction click

        The caller (ClickSampler.get_error_region_click) extracts the centroid
        of the returned region and determines the click label.

        Args:
            gt_mask: 2D binary ndarray of shape (H, W), ground-truth mask.
            pred_mask: 2D binary ndarray of shape (H, W), predicted mask.

        Returns:
            2D binary ndarray of shape (H, W), dtype bool, representing the
            dominant error region. Returns an all-zero mask if prediction is
            perfect (no errors).
        """
        gt_bool = gt_mask.astype(bool)
        pred_bool = pred_mask.astype(bool)

        # False negatives: GT foreground that prediction missed
        fn_region = gt_bool & ~pred_bool

        # False positives: predicted foreground not in GT
        fp_region = ~gt_bool & pred_bool

        fn_area = fn_region.sum()
        fp_area = fp_region.sum()

        if fn_area == 0 and fp_area == 0:
            # Perfect prediction — return empty error region
            return np.zeros_like(gt_bool, dtype=bool)

        # Return the larger error region
        if fn_area >= fp_area:
            return fn_region.astype(bool)
        else:
            return fp_region.astype(bool)

    # ------------------------------------------------------------------
    # Batch IoU computation
    # ------------------------------------------------------------------

    def compute_iou_batch(
        self,
        pred_masks: List[np.ndarray],
        gt_masks: List[np.ndarray],
    ) -> List[float]:
        """Compute IoU for a list of (pred, gt) mask pairs.

        Convenience wrapper around compute_iou_pair for batch evaluation.

        Args:
            pred_masks: List of 2D binary ndarrays.
            gt_masks: List of 2D binary ndarrays, same length as pred_masks.

        Returns:
            List of IoU floats, one per pair.

        Raises:
            ValueError: If pred_masks and gt_masks have different lengths.
        """
        if len(pred_masks) != len(gt_masks):
            raise ValueError(
                f"pred_masks length {len(pred_masks)} != "
                f"gt_masks length {len(gt_masks)}"
            )
        return [
            self.compute_iou_pair(p, g)
            for p, g in zip(pred_masks, gt_masks)
        ]

    # ------------------------------------------------------------------
    # Mask area utilities
    # ------------------------------------------------------------------

    def get_mask_area(self, mask: np.ndarray) -> int:
        """Compute the area (number of foreground pixels) of a binary mask.

        Args:
            mask: 2D binary ndarray of shape (H, W).

        Returns:
            Integer count of foreground pixels.
        """
        return int(mask.astype(bool).sum())

    def get_bounding_box(
        self,
        mask: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Compute the tight bounding box of a binary mask.

        Used by PromptSampler._mask_to_box() for bounding box prompt
        generation during training (25% probability per Appendix D.2.2).

        Args:
            mask: 2D binary ndarray of shape (H, W).

        Returns:
            (y_min, x_min, y_max, x_max) tuple of integers, or None if the
            mask is empty.
        """
        mask_bool = mask.astype(bool)
        ys, xs = np.where(mask_bool)

        if len(ys) == 0:
            return None

        y_min = int(ys.min())
        y_max = int(ys.max())
        x_min = int(xs.min())
        x_max = int(xs.max())

        return (y_min, x_min, y_max, x_max)

    # ------------------------------------------------------------------
    # Mask normalization and conversion helpers
    # ------------------------------------------------------------------

    def normalize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert a mask of any numeric dtype to a bool ndarray.

        Treats any non-zero value as foreground.

        Args:
            mask: 2D ndarray of any dtype.

        Returns:
            2D bool ndarray of the same shape.
        """
        return mask.astype(bool)

    def mask_to_uint8(self, mask: np.ndarray) -> np.ndarray:
        """Convert a binary mask to uint8 (0 or 255) for visualization.

        Args:
            mask: 2D binary ndarray of shape (H, W).

        Returns:
            2D uint8 ndarray with values 0 or 255.
        """
        return (mask.astype(bool) * 255).astype(np.uint8)

    def resize_mask(
        self,
        mask: np.ndarray,
        target_h: int,
        target_w: int,
    ) -> np.ndarray:
        """Resize a binary mask to a target resolution using nearest-neighbor.

        Nearest-neighbor interpolation preserves binary values without
        introducing intermediate gray values.

        Args:
            mask: 2D binary ndarray of shape (H, W).
            target_h: Target height in pixels.
            target_w: Target width in pixels.

        Returns:
            Resized binary ndarray of shape (target_h, target_w), dtype bool.
        """
        import cv2  # Lazy import to avoid hard dependency at module level

        mask_uint8 = mask.astype(np.uint8) * 255
        resized = cv2.resize(
            mask_uint8,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )
        return (resized > 127).astype(bool)

    # ------------------------------------------------------------------
    # Disappearance rate computation (SA-V dataset statistics)
    # ------------------------------------------------------------------

    def compute_disappearance_rate(
        self,
        masklet: List[Optional[np.ndarray]],
    ) -> bool:
        """Check whether a masklet disappears in at least one frame.

        From Section 5.2: "The disappearance rate in SA-V Manual is 42.5%,
        the percentage of annotated masklets that disappear in at least one
        frame and then re-appear."

        A masklet "disappears" if there exists a frame with no mask (None or
        all-zero) that is preceded and followed by frames with valid masks.

        Args:
            masklet: List of per-frame masks. None or all-zero mask indicates
                the object is not visible in that frame.

        Returns:
            True if the object disappears in at least one frame and
            re-appears in a later frame.
        """
        def _is_visible(m: Optional[np.ndarray]) -> bool:
            if m is None:
                return False
            return bool(m.astype(bool).any())

        visibility = [_is_visible(m) for m in masklet]

        # Check for disappear-reappear pattern: True ... False ... True
        seen_visible = False
        seen_invisible_after_visible = False

        for v in visibility:
            if v:
                if seen_invisible_after_visible:
                    return True  # Reappeared after disappearing
                seen_visible = True
            else:
                if seen_visible:
                    seen_invisible_after_visible = True

        return False
