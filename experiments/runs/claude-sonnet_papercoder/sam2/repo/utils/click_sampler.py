## utils/click_sampler.py
"""Click sampler for SAM 2 interactive segmentation reproduction.

This module implements ClickSampler, which provides deterministic click
coordinate generation for both training (via PromptSampler) and evaluation
(via InteractiveEvaluator, VOSEvaluator, ImageEvaluator).

The click placement strategy follows the paper exactly:
- Initial click: center of the GT mask (distance transform maximum)
- Correction clicks: center of the dominant error region (FN or FP)

Paper references:
    - Appendix F.1.2: "we place an initial click on the first frame at the
      center of the object's ground-truth mask and then interactively add
      two more clicks based on the center of the error region"
    - Appendix F.1.3: "the initial click is placed on the object center and
      subsequent clicks are obtained from the center of the error region"
    - Section 4: iterative click simulation during training

Config references:
    evaluation.interactive.num_clicks_per_frame: 3
    pretrain.interactive_clicks.num_correction_clicks: 7
"""

from typing import Tuple, Union

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


class ClickSampler:
    """Deterministic click coordinate generator for interactive segmentation.

    Implements the click placement strategy described in the SAM 2 paper:
    initial clicks are placed at the distance-transform centroid of the GT
    mask (the point inside the mask farthest from any boundary), and
    correction clicks are placed at the centroid of the dominant error region
    (whichever of false-negative or false-positive is larger by pixel count).

    All methods are stateless and deterministic given the input masks.
    No randomness is introduced here — the 10% random click probability
    described in Appendix D.2.2 is handled by PromptSampler.

    Coordinate convention: all returned coordinates are in [y, x] (row, col)
    image-space format. Callers (PromptSampler, evaluators) are responsible
    for converting to [x, y] when passing to PromptEncoder.

    Example:
        sampler = ClickSampler()
        # Initial click at GT mask center
        click = sampler.get_center_click(gt_mask)  # Tensor([y, x])
        # Correction click at error region center
        click, label = sampler.get_error_region_click(gt_mask, pred_mask)
        # Full n-click sequence
        coords, labels = sampler.sample_n_clicks(gt_mask, pred_mask, n=3)
    """

    def __init__(self) -> None:
        """Initialize ClickSampler. No state to initialize."""
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy_bool(mask: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert a mask tensor or array to a 2D boolean numpy array.

        Handles:
        - torch.Tensor of shape [H, W], [1, H, W], or [1, 1, H, W]
        - numpy.ndarray of shape [H, W] or [1, H, W]
        - Float masks: thresholded at 0.5
        - Bool masks: passed through directly

        Args:
            mask: Binary mask as tensor or ndarray.

        Returns:
            2D boolean numpy ndarray of shape [H, W].

        Raises:
            ValueError: If the mask cannot be squeezed to 2D.
        """
        if isinstance(mask, torch.Tensor):
            arr = mask.detach().cpu().numpy()
        else:
            arr = np.asarray(mask)

        # Squeeze leading singleton dimensions
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr.squeeze(0)

        if arr.ndim != 2:
            raise ValueError(
                f"Cannot convert mask of shape {arr.shape} to 2D. "
                "Expected [H, W], [1, H, W], or [1, 1, H, W]."
            )

        # Threshold float masks at 0.5; bool/int masks are cast directly
        if arr.dtype in (np.float32, np.float64, np.float16):
            return arr >= 0.5
        return arr.astype(bool)

    @staticmethod
    def _distance_transform_centroid(
        binary_mask: np.ndarray,
    ) -> Tuple[int, int]:
        """Find the point inside a binary mask farthest from any boundary.

        Uses scipy's Euclidean distance transform. The maximum of the
        distance transform gives the most "central" point guaranteed to lie
        on the foreground — more robust than the raw geometric centroid for
        irregular shapes (e.g., a donut-shaped mask where the centroid falls
        in the hole).

        This is the standard approach used in SAM's evaluation code and is
        the correct interpretation of "center of the mask" in the paper.

        Args:
            binary_mask: 2D boolean ndarray of shape [H, W]. Must be non-empty
                (at least one True pixel).

        Returns:
            (row, col) tuple of integers — the pixel with maximum distance
            to the nearest background pixel.
        """
        # Compute Euclidean distance transform: each foreground pixel gets
        # its distance to the nearest background pixel
        dist = distance_transform_edt(binary_mask)

        # Find the location of the maximum distance value
        # np.argmax returns a flat index; unravel to (row, col)
        flat_idx = int(np.argmax(dist))
        h, w = binary_mask.shape
        row = flat_idx // w
        col = flat_idx % w

        return (row, col)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_center_click(
        self,
        mask: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        """Get the initial click coordinate at the center of a GT mask.

        Places the click at the point inside the mask that is farthest from
        any boundary (distance transform maximum). This is the most
        informative initial click location for interactive segmentation.

        Paper reference: Appendix F.1.2 — "we place an initial click on the
        first frame at the center of the object's ground-truth mask".
        Appendix F.1.3 — "the initial click is placed on the object center".

        Args:
            mask: Binary GT mask as torch.Tensor or numpy.ndarray.
                Accepted shapes: [H, W], [1, H, W], [1, 1, H, W].
                Accepted dtypes: bool, uint8, float32 (thresholded at 0.5).

        Returns:
            torch.Tensor of shape [2] containing [y, x] pixel coordinates
            (row, col convention). dtype=torch.long.
            Returns image center [H//2, W//2] as fallback for empty masks.
        """
        mask_np = self._to_numpy_bool(mask)
        h, w = mask_np.shape

        if not mask_np.any():
            # Empty mask — return image center as safe fallback
            row = h // 2
            col = w // 2
        else:
            row, col = self._distance_transform_centroid(mask_np)

        return torch.tensor([row, col], dtype=torch.long)

    def get_error_region_click(
        self,
        gt_mask: Union[torch.Tensor, np.ndarray],
        pred_mask: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[torch.Tensor, int]:
        """Get a correction click at the center of the dominant error region.

        Computes false-negative (FN) and false-positive (FP) regions between
        the GT and predicted masks. Selects the larger region (by pixel count)
        and places the click at its distance-transform centroid.

        - FN region (GT foreground missed by prediction) → positive click (label=1)
        - FP region (predicted foreground not in GT) → negative click (label=0)

        Paper reference: Appendix F.1.2 — "we interactively add clicks based
        on the center of the error region (between the ground-truth mask and
        the predicted segments on the frame being prompted)".

        Args:
            gt_mask: Binary ground-truth mask. Accepted shapes/dtypes same
                as get_center_click().
            pred_mask: Binary predicted mask. Accepted shapes/dtypes same
                as get_center_click().

        Returns:
            Tuple of:
                - click_coords: torch.Tensor of shape [2] containing [y, x]
                  pixel coordinates. dtype=torch.long.
                - label: int — 1 for positive click (correcting FN region),
                  0 for negative click (correcting FP region).

            Fallback when both regions are empty (perfect prediction):
                Returns GT mask center with label=1.
        """
        gt_np = self._to_numpy_bool(gt_mask)
        pred_np = self._to_numpy_bool(pred_mask)

        # Compute error regions
        fn_region: np.ndarray = gt_np & ~pred_np   # False negatives
        fp_region: np.ndarray = ~gt_np & pred_np   # False positives

        fn_area: int = int(fn_region.sum())
        fp_area: int = int(fp_region.sum())

        if fn_area == 0 and fp_area == 0:
            # Perfect prediction — fall back to GT mask center with positive label
            click_coords = self.get_center_click(gt_mask)
            return click_coords, 1

        # Select the larger error region
        if fn_area >= fp_area:
            selected_region = fn_region
            label: int = 1  # Positive click to correct missed foreground
        else:
            selected_region = fp_region
            label = 0  # Negative click to correct spurious foreground

        # Find the most central point in the selected error region
        row, col = self._distance_transform_centroid(selected_region)
        click_coords = torch.tensor([row, col], dtype=torch.long)

        return click_coords, label

    def sample_n_clicks(
        self,
        gt_mask: Union[torch.Tensor, np.ndarray],
        pred_mask: Union[torch.Tensor, np.ndarray],
        n: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a sequence of n clicks for interactive segmentation.

        Generates the full click sequence used during training and evaluation:
        - Click 1: initial positive click at the GT mask center
        - Clicks 2..n: correction clicks at the dominant error region center

        Note: This method does NOT run the model between clicks. All
        correction clicks are sampled from the same static error between
        gt_mask and pred_mask. For evaluation, the evaluators call
        get_center_click() and get_error_region_click() individually between
        model forward passes to update pred_mask.

        Paper reference: Section 4 — "we simulate interactive prompting of
        the model... probabilistically receive corrective clicks which are
        sampled using the ground-truth masklet and model predictions during
        training". Appendix F.1.2 — N_click = 3 clicks per frame.

        Config reference:
            evaluation.interactive.num_clicks_per_frame: 3
            pretrain.interactive_clicks.num_correction_clicks: 7

        Args:
            gt_mask: Binary ground-truth mask. Accepted shapes/dtypes same
                as get_center_click().
            pred_mask: Binary predicted mask used for correction clicks.
                Accepted shapes/dtypes same as get_center_click().
            n: Total number of clicks to generate. Must be >= 1.

        Returns:
            Tuple of:
                - coords: torch.Tensor of shape [n, 2] — each row is [y, x].
                  dtype=torch.long.
                - labels: torch.Tensor of shape [n] — each element is 0
                  (negative) or 1 (positive). dtype=torch.long.

        Raises:
            ValueError: If n < 1.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}.")

        coords_list: list = []
        labels_list: list = []

        # Click 1: initial positive click at GT mask center
        initial_click = self.get_center_click(gt_mask)
        coords_list.append(initial_click)
        labels_list.append(torch.tensor(1, dtype=torch.long))

        # Clicks 2..n: correction clicks at error region center
        for _ in range(n - 1):
            correction_click, label = self.get_error_region_click(
                gt_mask, pred_mask
            )
            coords_list.append(correction_click)
            labels_list.append(torch.tensor(label, dtype=torch.long))

        # Stack into tensors of shape [n, 2] and [n]
        coords = torch.stack(coords_list, dim=0)    # [n, 2]
        labels = torch.stack(labels_list, dim=0)    # [n]

        return coords, labels
