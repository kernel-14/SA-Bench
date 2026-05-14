# evaluation/click_simulator.py
"""
Click simulation strategies for interactive segmentation.

Implements the ``ClickSimulator`` class described in the SAM 2 paper
(Sections 6.1, 6.2, 6.3 and Appendix F.1.2).  The methods produce a list
of click dictionaries – each dictionary specifies a pixel coordinate and
a positive/negative label – following the paper’s rules:

* Initial click: centroid of the ground‑truth mask.
* Correction clicks: centroid of the largest connected component of the
  error region between the binarised prediction and the ground truth.
  If the error region is empty, a random positive click from the ground
  truth is sampled with probability ``random_gt_prob`` (used only during
  training).

All inputs are expected to be 2D ``numpy`` arrays (``uint8``, ``bool``,
or ``float32``).  If a ``torch.Tensor`` is passed, it is automatically
transferred to CPU and converted to ``numpy``.

Typical usage::

    sim = ClickSimulator()
    initial = sim.generate_initial_clicks(gt_mask)          # returns list
    correction = sim.generate_correction_clicks(pred, gt)   # returns list
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import torch  # noqa: F401  # only used for is_tensor check
except ImportError:
    torch = None

__all__ = ["ClickSimulator"]

# ---------------------------------------------------------------------------
# Type alias for a single click dictionary
# ---------------------------------------------------------------------------
ClickDict = Dict[str, Union[int, bool]]


# ---------------------------------------------------------------------------
# ClickSimulator class
# ---------------------------------------------------------------------------

class ClickSimulator:
    """
    Stateless click generator for interactive segmentation.

    The underlying strategy (``"largest_error_centroid"``) adheres to the
    paper: for correction clicks the error map is partitioned into connected
    components, the largest component by area is chosen, and its centroid is
    used.  The positive/negative label is determined by the ground‑truth
    value at that location.

    Args:
        strategy: reserved for future extensions; currently only
            ``"largest_error_centroid"`` is supported.  (Default: ``"largest_error_centroid"``)
    """

    def __init__(self, strategy: str = "largest_error_centroid") -> None:
        if strategy != "largest_error_centroid":
            logging.warning(
                f"ClickSimulator: unknown strategy '{strategy}'. "
                f"Falling back to 'largest_error_centroid'."
            )
        self.strategy = strategy

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(array_like: Any) -> np.ndarray:
        """
        Convert a torch Tensor or array-like object to a CPU numpy array.

        If the input is a torch Tensor, it is detached, moved to CPU, and
        converted to numpy.  Otherwise ``np.asarray`` is used.

        Args:
            array_like: a numpy array or a torch Tensor.

        Returns:
            ``np.ndarray`` representation of the input.
        """
        if torch is not None and torch.is_tensor(array_like):
            return array_like.detach().cpu().numpy()
        return np.asarray(array_like)

    @staticmethod
    def _centroid_of_mask(mask: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Compute the spatial centroid (x, y) of a binary mask via
        ``cv2.moments``.

        Args:
            mask: 2D binary or integer ``numpy`` array where non‑zero
                pixels belong to the object.

        Returns:
            ``(x, y)`` pixel coordinates, or ``None`` if the mask has
            no positive pixels or the moment ``m00`` is zero.
        """
        if not mask.any():
            return None
        moments = cv2.moments(mask.astype(np.uint8), binaryImage=True)
        area = moments["m00"]
        if area == 0:
            return None
        cx = int(moments["m10"] / area)
        cy = int(moments["m01"] / area)
        return cx, cy

    @staticmethod
    def _largest_error_component(
        error_mask: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Find the largest connected component (by area) in a binary error map.

        Uses ``cv2.connectedComponentsWithStats`` with 8‑connectivity.
        Background (label 0) is excluded.

        Args:
            error_mask: 2D ``numpy`` array (``uint8`` or ``bool``) where
                ``True``/1 indicates an error pixel.

        Returns:
            A tuple ``(component_mask, label)``.
            - ``component_mask``: ``bool`` array of the same spatial size
              as ``error_mask``, with ``True`` only for the largest component.
            - ``label``: integer label of that component in the connected
              components labeling.
            If the error mask has no positive pixels or contains only the
            background, ``(None, None)`` is returned.
        """
        error_uint8 = error_mask.astype(np.uint8)
        if not error_uint8.any():
            return None, None

        retval, labels = cv2.connectedComponents(error_uint8, connectivity=8)
        if retval < 2:  # only background label exists
            return None, None

        # Compute area for each label (skip 0)
        areas = np.bincount(labels.flat, minlength=retval)
        areas[0] = 0  # ignore background
        largest_label = areas.argmax()

        component_mask = (labels == largest_label)
        return component_mask, largest_label

    @staticmethod
    def _random_point_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Uniformly sample a random pixel location where ``mask`` is positive.

        Args:
            mask: 2D binary ``numpy`` array.

        Returns:
            ``(x, y)`` pixel coordinates, or ``None`` if no positive
            pixels exist.
        """
        ys, xs = mask.nonzero()
        if len(ys) == 0:
            return None
        idx = np.random.randint(len(ys))
        return int(xs[idx]), int(ys[idx])

    # ------------------------------------------------------------------
    #  Public methods matching the design interface
    # ------------------------------------------------------------------

    def generate_initial_clicks(self, mask: Any) -> List[ClickDict]:
        """
        Generate the initial click(s) from the ground‑truth mask.

        For SAM 2, a single positive click is placed at the spatial centroid
        of the mask.  If the mask is entirely empty (i.e., the object is not
        visible in this frame), an empty list is returned.

        Args:
            mask: 2D ground‑truth mask as numpy array or torch Tensor.
                Non‑zero values indicate the object.

        Returns:
            A list of click dictionaries.  For a non‑empty mask the list
            contains one element:
                ``{"x": x, "y": y, "positive": True}``.
            If the mask is empty, the list is empty.

        Example::

            sim = ClickSimulator()
            clicks = sim.generate_initial_clicks(gt_mask)
            # clicks → [{"x": 512, "y": 320, "positive": True}]
        """
        gt = self._to_numpy(mask)
        centroid = self._centroid_of_mask(gt)
        if centroid is None:
            return []
        return [{"x": centroid[0], "y": centroid[1], "positive": True}]

    def generate_correction_clicks(
        self,
        pred_mask: Any,
        gt_mask: Any,
        random_gt_prob: float = 0.0,
    ) -> List[ClickDict]:
        """
        Generate a correction click based on the error between prediction and
        ground truth.

        The prediction is binarised using a threshold of 0.5 (if it contains
        float values).  The error map is ``pred_bin XOR gt_mask``.  If the
        error map is non‑empty, the largest connected component is found and
        its centroid is returned as a click; the click is positive if the
        ground‑truth value at that centroid is ``True``, negative otherwise.

        If the error map is **empty** (perfect prediction), there are two
        behaviours controlled by ``random_gt_prob``:

        - With probability ``random_gt_prob`` (typically ``0.1`` during
          training), a random positive click is sampled from the ground‑truth
          mask (used to inject diversity).
        - Otherwise, an empty list is returned (no further clicks are needed).

        Args:
            pred_mask: Predicted mask, 2D array (or torch Tensor).
                If it contains float values, a threshold of ``0.5`` is applied.
            gt_mask: Ground‑truth mask, same spatial size as ``pred_mask``.
            random_gt_prob: Probability (between 0 and 1) of sampling a random
                ground‑truth click when the error map is empty (training only).

        Returns:
            A list containing a single click dictionary, or an empty list if
            no valid click can be produced.  The click dict has keys
            ``"x"``, ``"y"``, and ``"positive"`` (``bool``).
        """
        pred = self._to_numpy(pred_mask)
        gt = self._to_numpy(gt_mask)

        # Binarise prediction if it contains float values
        if np.issubdtype(pred.dtype, np.floating):
            pred_bin = pred > 0.5
        else:
            pred_bin = pred.astype(bool)

        error = np.logical_xor(pred_bin, gt)

        # Case 1: no error region
        if not error.any():
            if random_gt_prob > 0 and np.random.rand() < random_gt_prob:
                pt = self._random_point_from_mask(gt)
                if pt is not None:
                    return [{"x": pt[0], "y": pt[1], "positive": True}]
            return []

        # Case 2: error exists → find largest component centroid
        component_mask, _ = self._largest_error_component(error)
        if component_mask is None:
            # Should not happen because error.any() was True, but guard anyway
            return []

        centroid = self._centroid_of_mask(component_mask)
        if centroid is None:
            return []

        x, y = centroid
        is_positive = bool(gt[y, x])
        return [{"x": x, "y": y, "positive": is_positive}]

    # ------------------------------------------------------------------
    #  Auxiliary convenience (used in trainer)
    # ------------------------------------------------------------------

    def get_random_gt_click(self, gt_mask: Any) -> List[ClickDict]:
        """
        Sample a single random positive click from the ground‑truth mask.

        This is a convenience wrapper around ``_random_point_from_mask``
        and is primarily called during training when a 10 % chance to
        ignore the error map is desired.

        Args:
            gt_mask: Ground‑truth mask, 2D array.

        Returns:
            A list containing either one ``{"x": x, "y": y, "positive": True}``
            dictionary, or an empty list if the mask has no positive pixels.
        """
        gt = self._to_numpy(gt_mask)
        pt = self._random_point_from_mask(gt)
        if pt is None:
            return []
        return [{"x": pt[0], "y": pt[1], "positive": True}]

