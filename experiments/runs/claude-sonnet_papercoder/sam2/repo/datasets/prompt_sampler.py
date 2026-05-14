## datasets/prompt_sampler.py
"""Prompt sampler for SAM 2 interactive segmentation training.

This module implements PromptSampler, which simulates the interactive
annotation experience described in the SAM 2 paper during training. It
generates initial prompts (GT mask, click, or bounding box) and corrective
clicks from error regions between ground-truth and model predictions.

Config references (config.yaml):
    training.prompt_probabilities.gt_mask: 0.50
    training.prompt_probabilities.positive_click: 0.25
    training.prompt_probabilities.bounding_box: 0.25
    training.correction_click_random_prob: 0.10
    model.mask_threshold: 0.0

Paper references:
    Section 4: "Initial prompts to the model can be the ground-truth mask
        with probability 0.5, a positive click sampled from the ground-truth
        mask with probability 0.25, or a bounding box input with probability
        0.25."
    Appendix D.2.2: "with a small probability of 10%, we randomly sample
        clicks from the ground truth mask, irrespective of the model
        prediction, to allow additional flexibility in mask refinement."
    Appendix D.2.2: "we simulate interactive prompting of the model...
        probabilistically receive corrective clicks which are sampled using
        the ground-truth masklet and model predictions during training."
"""

import logging
import random
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from utils.click_sampler import ClickSampler
from utils.mask_utils import MaskUtils

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PromptInput is imported from datasets/__init__.py at call time to avoid
# circular imports. The class is defined there as the shared data contract.
# ---------------------------------------------------------------------------


def _get_prompt_input_class():
    """Lazily import PromptInput to avoid circular import at module load time.

    Returns:
        The PromptInput dataclass from datasets/__init__.py.
    """
    from datasets import PromptInput as _PromptInput
    return _PromptInput


# ---------------------------------------------------------------------------
# PromptSampler
# ---------------------------------------------------------------------------


class PromptSampler:
    """Simulates interactive prompt generation for SAM 2 training.

    Generates initial prompts (GT mask, positive click, or bounding box) and
    corrective clicks (from error regions or random GT positions) to train
    SAM 2's interactive segmentation capability.

    All sampling is stateless — each call is independent, making this class
    safe for use with multi-worker DataLoaders. All operations run on CPU
    tensors; the Trainer moves PromptInput tensors to GPU.

    Config references (config.yaml):
        training.prompt_probabilities.gt_mask: 0.50
        training.prompt_probabilities.positive_click: 0.25
        training.prompt_probabilities.bounding_box: 0.25
        training.correction_click_random_prob: 0.10

    Args:
        gt_mask_prob: Probability of using GT mask as initial prompt.
            Defaults to 0.50 (config.training.prompt_probabilities.gt_mask).
        click_prob: Probability of using a positive click as initial prompt.
            Defaults to 0.25 (config.training.prompt_probabilities.positive_click).
        box_prob: Probability of using a bounding box as initial prompt.
            Defaults to 0.25 (config.training.prompt_probabilities.bounding_box).
        correction_click_random_prob: Probability of sampling a random GT click
            instead of an error-region click during correction.
            Defaults to 0.10 (config.training.correction_click_random_prob).
        mask_threshold: Threshold for binarizing predicted mask logits/probs
            before computing error regions. Defaults to 0.0 (sigmoid output > 0.5
            for logits; config.model.mask_threshold: 0.0).

    Raises:
        ValueError: If gt_mask_prob + click_prob + box_prob does not sum to 1.0
            (within floating-point tolerance of 1e-6).

    Example:
        sampler = PromptSampler(
            gt_mask_prob=0.50,
            click_prob=0.25,
            box_prob=0.25,
        )
        gt_mask = torch.zeros(1024, 1024)
        gt_mask[200:400, 300:600] = 1.0
        prompt = sampler.sample_initial_prompt(gt_mask, frame_idx=0)
        # prompt.masks is set (50% chance), or prompt.points (25%), or prompt.boxes (25%)
    """

    def __init__(
        self,
        gt_mask_prob: float = 0.50,
        click_prob: float = 0.25,
        box_prob: float = 0.25,
        correction_click_random_prob: float = 0.10,
        mask_threshold: float = 0.0,
    ) -> None:
        # Validate probabilities sum to 1.0
        total_prob: float = gt_mask_prob + click_prob + box_prob
        if abs(total_prob - 1.0) > 1e-6:
            raise ValueError(
                f"gt_mask_prob + click_prob + box_prob must sum to 1.0, "
                f"got {gt_mask_prob} + {click_prob} + {box_prob} = {total_prob:.6f}."
            )

        if not 0.0 <= correction_click_random_prob <= 1.0:
            raise ValueError(
                f"correction_click_random_prob must be in [0, 1], "
                f"got {correction_click_random_prob}."
            )

        self.gt_mask_prob: float = gt_mask_prob
        self.click_prob: float = click_prob
        self.box_prob: float = box_prob
        self.correction_click_random_prob: float = correction_click_random_prob
        self.mask_threshold: float = mask_threshold

        # Cumulative probability boundaries for branch selection
        # r < gt_mask_prob                          → GT mask
        # gt_mask_prob <= r < gt_mask_prob+click_prob → click
        # r >= gt_mask_prob + click_prob             → box
        self._click_boundary: float = gt_mask_prob + click_prob

        # Owned utilities — not passed in, instantiated here
        self.click_sampler: ClickSampler = ClickSampler()
        self.mask_utils: MaskUtils = MaskUtils()

        logger.debug(
            "PromptSampler initialized: gt_mask_prob=%.2f, click_prob=%.2f, "
            "box_prob=%.2f, correction_click_random_prob=%.2f, "
            "mask_threshold=%.2f",
            gt_mask_prob,
            click_prob,
            box_prob,
            correction_click_random_prob,
            mask_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_initial_prompt(
        self,
        gt_mask: Tensor,
        frame_idx: int = 0,
    ) -> "PromptInput":  # type: ignore[name-defined]
        """Sample an initial prompt for the first prompted frame.

        Randomly selects one of three prompt types based on the configured
        probabilities:
            - GT mask (50%): wraps the ground-truth mask directly
            - Positive click (25%): places a click at the GT mask centroid
            - Bounding box (25%): computes the tight bounding box of the GT mask

        Paper reference: Section 4 — "Initial prompts to the model can be the
        ground-truth mask with probability 0.5, a positive click sampled from
        the ground-truth mask with probability 0.25, or a bounding box input
        with probability 0.25."

        Args:
            gt_mask: Ground-truth binary mask of shape [H, W], dtype float32
                or bool. Values should be 0 (background) or 1 (foreground).
                An all-zero mask (occluded frame) returns an empty PromptInput.
            frame_idx: Integer index of the video frame this prompt applies to.
                Defaults to 0 (first frame).

        Returns:
            PromptInput with exactly one of {masks, points+point_labels, boxes}
            populated, and frame_idx set. Returns an empty PromptInput (all
            fields None) if gt_mask is all zeros (occluded/empty frame).
        """
        PromptInput = _get_prompt_input_class()

        # Handle empty mask (occluded frame) — return empty prompt
        gt_mask_np = gt_mask.detach().cpu().numpy() if isinstance(gt_mask, Tensor) else gt_mask
        if not gt_mask_np.astype(bool).any():
            logger.debug(
                "sample_initial_prompt: gt_mask is empty at frame_idx=%d. "
                "Returning empty PromptInput.",
                frame_idx,
            )
            return PromptInput(frame_idx=frame_idx)

        # Draw random value to select prompt type
        r: float = random.random()

        if r < self.gt_mask_prob:
            # --- GT mask prompt (50%) ---
            return self._build_mask_prompt(gt_mask, frame_idx)

        elif r < self._click_boundary:
            # --- Positive click prompt (25%) ---
            return self._build_click_prompt(gt_mask, frame_idx)

        else:
            # --- Bounding box prompt (25%) ---
            return self._build_box_prompt(gt_mask, frame_idx)

    def sample_correction_clicks(
        self,
        gt_mask: Tensor,
        pred_mask: Tensor,
        num_clicks: int,
        frame_idx: int = 0,
    ) -> "PromptInput":  # type: ignore[name-defined]
        """Sample corrective clicks from the error region between GT and prediction.

        Generates `num_clicks` corrective clicks. Each click is either:
            - Error-region click (90%): placed at the centroid of the largest
              false-negative or false-positive region
            - Random GT click (10%): placed at the GT mask centroid regardless
              of the prediction (for training flexibility)

        Paper reference: Appendix D.2.2 — "with a small probability of 10%,
        we randomly sample clicks from the ground truth mask, irrespective of
        the model prediction, to allow additional flexibility in mask refinement."

        Args:
            gt_mask: Ground-truth binary mask of shape [H, W], dtype float32
                or bool. Values should be 0 (background) or 1 (foreground).
            pred_mask: Predicted mask of shape [H, W], dtype float32 (logits,
                probabilities, or binary). Will be binarized internally using
                mask_threshold before computing error regions.
            num_clicks: Number of corrective clicks to generate. Must be >= 1.
                During pre-training: 7 (Table 12). During evaluation: 3
                (config.evaluation.interactive.num_clicks_per_frame: 3).
            frame_idx: Integer index of the video frame this prompt applies to.
                Defaults to 0.

        Returns:
            PromptInput with points=[num_clicks, 2] and point_labels=[num_clicks]
            populated, and frame_idx set. Returns an empty PromptInput if both
            gt_mask and pred_mask are empty (no valid correction possible).

        Raises:
            ValueError: If num_clicks < 1.
        """
        if num_clicks < 1:
            raise ValueError(f"num_clicks must be >= 1, got {num_clicks}.")

        PromptInput = _get_prompt_input_class()

        # Handle empty GT mask — no valid correction possible
        gt_mask_np = gt_mask.detach().cpu().numpy() if isinstance(gt_mask, Tensor) else gt_mask
        if not gt_mask_np.astype(bool).any():
            logger.debug(
                "sample_correction_clicks: gt_mask is empty at frame_idx=%d. "
                "Returning empty PromptInput.",
                frame_idx,
            )
            return PromptInput(frame_idx=frame_idx)

        # Binarize predicted mask using mask_threshold
        # For logits (mask_threshold=0.0): sigmoid(logit) > 0.5 ↔ logit > 0.0
        # For probabilities: prob > 0.5
        pred_mask_binary: Tensor = self._binarize_mask(pred_mask)

        # Accumulate click coordinates and labels
        coords_list: List[Tensor] = []
        labels_list: List[Tensor] = []

        for click_idx in range(num_clicks):
            r: float = random.random()

            if r < self.correction_click_random_prob:
                # --- Random GT click (10%) ---
                # Place click at GT mask centroid regardless of prediction
                click_coords: Tensor = self._sample_click_from_mask(gt_mask)
                click_label: int = 1  # always positive for GT click

                logger.debug(
                    "sample_correction_clicks: random GT click %d/%d at "
                    "frame_idx=%d, coords=%s",
                    click_idx + 1,
                    num_clicks,
                    frame_idx,
                    click_coords.tolist(),
                )
            else:
                # --- Error-region click (90%) ---
                # Place click at centroid of largest FN or FP region
                click_coords, click_label = self.click_sampler.get_error_region_click(
                    gt_mask=gt_mask,
                    pred_mask=pred_mask_binary,
                )

                logger.debug(
                    "sample_correction_clicks: error-region click %d/%d at "
                    "frame_idx=%d, coords=%s, label=%d",
                    click_idx + 1,
                    num_clicks,
                    frame_idx,
                    click_coords.tolist(),
                    click_label,
                )

            # Convert (row, col) from ClickSampler to (x, y) for PromptEncoder
            # ClickSampler returns [row, col] = [y, x]; PromptEncoder expects [x, y]
            coords_xy: Tensor = torch.tensor(
                [float(click_coords[1]), float(click_coords[0])],
                dtype=torch.float32,
            )

            coords_list.append(coords_xy)
            labels_list.append(
                torch.tensor(click_label, dtype=torch.long)
            )

        # Stack into tensors: [num_clicks, 2] and [num_clicks]
        points: Tensor = torch.stack(coords_list, dim=0)    # [num_clicks, 2]
        point_labels: Tensor = torch.stack(labels_list, dim=0)  # [num_clicks]

        return PromptInput(
            points=points,
            point_labels=point_labels,
            frame_idx=frame_idx,
        )

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------

    def _build_mask_prompt(
        self,
        gt_mask: Tensor,
        frame_idx: int,
    ) -> "PromptInput":  # type: ignore[name-defined]
        """Build a GT mask PromptInput.

        Wraps the ground-truth mask as a [1, H, W] float32 tensor for the
        PromptEncoder's mask downscaling pathway.

        Args:
            gt_mask: Ground-truth binary mask of shape [H, W].
            frame_idx: Frame index for the prompt.

        Returns:
            PromptInput with masks=[1, H, W] float32 and frame_idx set.
        """
        PromptInput = _get_prompt_input_class()

        # Ensure float32 and add channel dimension: [H, W] → [1, H, W]
        mask_tensor: Tensor = gt_mask.float()
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)  # [1, H, W]
        elif mask_tensor.ndim == 3 and mask_tensor.shape[0] != 1:
            # Take first channel if multi-channel mask provided
            mask_tensor = mask_tensor[0:1]

        return PromptInput(
            masks=mask_tensor,
            frame_idx=frame_idx,
        )

    def _build_click_prompt(
        self,
        gt_mask: Tensor,
        frame_idx: int,
    ) -> "PromptInput":  # type: ignore[name-defined]
        """Build a positive click PromptInput at the GT mask centroid.

        Delegates to ClickSampler.get_center_click() for coordinate generation,
        then converts from (row, col) to (x, y) for PromptEncoder compatibility.

        Args:
            gt_mask: Ground-truth binary mask of shape [H, W].
            frame_idx: Frame index for the prompt.

        Returns:
            PromptInput with points=[1, 2] float32 (x, y) and
            point_labels=[1] int64 (value=1 for positive) and frame_idx set.
        """
        PromptInput = _get_prompt_input_class()

        # get_center_click returns [row, col] = [y, x]
        center_yx: Tensor = self._sample_click_from_mask(gt_mask)

        # Convert to (x, y) for PromptEncoder: x=col, y=row
        coords_xy: Tensor = torch.tensor(
            [float(center_yx[1]), float(center_yx[0])],
            dtype=torch.float32,
        ).unsqueeze(0)  # [1, 2]

        point_labels: Tensor = torch.tensor([1], dtype=torch.long)  # [1] positive

        return PromptInput(
            points=coords_xy,
            point_labels=point_labels,
            frame_idx=frame_idx,
        )

    def _build_box_prompt(
        self,
        gt_mask: Tensor,
        frame_idx: int,
    ) -> "PromptInput":  # type: ignore[name-defined]
        """Build a bounding box PromptInput from the GT mask.

        Computes the tight axis-aligned bounding box of the GT mask foreground
        pixels. Falls back to a click prompt if the bounding box cannot be
        computed (degenerate mask).

        Args:
            gt_mask: Ground-truth binary mask of shape [H, W].
            frame_idx: Frame index for the prompt.

        Returns:
            PromptInput with boxes=[4] float32 (x1, y1, x2, y2) and
            frame_idx set. Falls back to click prompt if box extraction fails.
        """
        PromptInput = _get_prompt_input_class()

        box: Optional[Tensor] = self._mask_to_box(gt_mask)

        if box is None:
            # Degenerate mask — fall back to click prompt
            logger.debug(
                "_build_box_prompt: could not extract bounding box from mask "
                "at frame_idx=%d. Falling back to click prompt.",
                frame_idx,
            )
            return self._build_click_prompt(gt_mask, frame_idx)

        return PromptInput(
            boxes=box,
            frame_idx=frame_idx,
        )

    def _mask_to_box(self, mask: Tensor) -> Optional[Tensor]:
        """Compute the tight bounding box of a binary mask.

        Finds all foreground pixels and returns the tight axis-aligned bounding
        box as (x1, y1, x2, y2) in pixel coordinates.

        Note: The paper does not specify any padding or jitter on the bounding
        box during training — the box is kept tight around the mask.

        Args:
            mask: Binary mask of shape [H, W], dtype float32 or bool.
                Values should be 0 (background) or 1 (foreground).

        Returns:
            Tensor of shape [4] with values [x1, y1, x2, y2] as float32,
            where (x1, y1) is the top-left corner and (x2, y2) is the
            bottom-right corner in pixel coordinates.
            Returns None if the mask is empty (no foreground pixels).
        """
        import numpy as np

        # Convert to numpy for MaskUtils compatibility
        if isinstance(mask, Tensor):
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = np.asarray(mask)

        # Delegate to MaskUtils.get_bounding_box which returns (y_min, x_min, y_max, x_max)
        bbox = self.mask_utils.get_bounding_box(mask_np)

        if bbox is None:
            return None

        y_min, x_min, y_max, x_max = bbox

        # Convert to (x1, y1, x2, y2) format expected by PromptEncoder
        box_tensor: Tensor = torch.tensor(
            [float(x_min), float(y_min), float(x_max), float(y_max)],
            dtype=torch.float32,
        )

        return box_tensor

    def _sample_click_from_mask(self, mask: Tensor) -> Tensor:
        """Get the center click coordinate from a binary mask.

        Convenience wrapper around ClickSampler.get_center_click() that
        returns the distance-transform centroid of the mask foreground.

        Args:
            mask: Binary mask of shape [H, W], dtype float32 or bool.

        Returns:
            Tensor of shape [2] containing [row, col] (y, x) pixel coordinates
            as torch.long. The caller is responsible for converting to (x, y)
            format if needed by PromptEncoder.
        """
        return self.click_sampler.get_center_click(mask)

    def _binarize_mask(self, mask: Tensor) -> Tensor:
        """Binarize a predicted mask using the configured mask_threshold.

        Handles three input formats:
            - Raw logits (mask_threshold=0.0): applies sigmoid then thresholds at 0.5
              (equivalent to logit > 0.0)
            - Probabilities in [0, 1]: thresholds at 0.5
            - Already binary {0, 1}: returned as-is

        Config reference: model.mask_threshold: 0.0

        Args:
            mask: Predicted mask of shape [H, W] or [1, H, W] or [B, 1, H, W],
                dtype float32. Can be logits, probabilities, or binary.

        Returns:
            Binary mask of same shape as input, dtype float32 with values
            in {0.0, 1.0}.
        """
        mask_float: Tensor = mask.float()

        # Detect if values are logits (can be negative) or probabilities [0, 1]
        min_val: float = float(mask_float.min().item())
        max_val: float = float(mask_float.max().item())

        if min_val < -0.5 or max_val > 1.5:
            # Likely raw logits — apply sigmoid then threshold at 0.5
            # This is equivalent to logit > mask_threshold (0.0 by default)
            mask_prob: Tensor = torch.sigmoid(mask_float)
            return (mask_prob > 0.5).float()
        else:
            # Probabilities or already binary — threshold at 0.5
            return (mask_float > 0.5).float()

    # ------------------------------------------------------------------
    # Utility / inspection methods
    # ------------------------------------------------------------------

    def get_prompt_type_probabilities(self) -> dict:
        """Return the configured prompt type probabilities as a dict.

        Useful for logging and debugging the training configuration.

        Returns:
            Dict with keys 'gt_mask', 'click', 'box', 'correction_random'
            mapping to their respective probabilities.
        """
        return {
            "gt_mask": self.gt_mask_prob,
            "click": self.click_prob,
            "box": self.box_prob,
            "correction_random": self.correction_click_random_prob,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"gt_mask_prob={self.gt_mask_prob}, "
            f"click_prob={self.click_prob}, "
            f"box_prob={self.box_prob}, "
            f"correction_click_random_prob={self.correction_click_random_prob}, "
            f"mask_threshold={self.mask_threshold})"
        )
