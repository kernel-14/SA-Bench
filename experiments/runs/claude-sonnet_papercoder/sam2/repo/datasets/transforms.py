## Code: datasets/transforms.py

```python
## datasets/transforms.py
"""Video-level data augmentations for SAM 2 training.

This module implements all augmentations described in Appendix D.2.2 of the
SAM 2 paper, applied consistently across all frames and masks in a video
sequence. Every transform is temporally consistent: random parameters are
sampled once per sequence and applied identically to every frame and mask.

Augmentations implemented:
    1. RandomHorizontalFlip — spatial, applied to frames + masks
    2. RandomAffine — spatial, applied to frames (bilinear) + masks (nearest)
    3. ColorJitter — color, applied to frames only
    4. RandomGrayscale — color, applied to frames only
    5. MosaicTransform — structural 2×2 tiling, applied to frames + masks

Config references (config.yaml):
    training.mosaic_prob: 0.10
    training.augmentation.horizontal_flip: true
    training.augmentation.random_affine: true
    training.augmentation.color_jitter: true
    training.augmentation.random_grayscale: true
    training.augmentation.mosaic_transform: true

Paper references:
    Appendix D.2.2: "We apply a series of data augmentations to the training
        videos, including random horizontal flips, random affine transforms,
        random color jittering, and random grayscale transforms."
    Appendix D.2.2: "With 10% probability, we tile the same training video
        into a 2×2 grid and select a masklet from one of the 4 quadrants as
        the target object to segment."
"""

import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Import VideoSample from the package-level __init__ to avoid circular imports.
# datasets/__init__.py defines VideoSample before importing submodules.
# ---------------------------------------------------------------------------
# We use a TYPE_CHECKING guard to avoid the circular import at runtime while
# still providing type hints for IDEs.
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import VideoSample


# ---------------------------------------------------------------------------
# Helper: import VideoSample at call time to avoid circular import
# ---------------------------------------------------------------------------

def _get_video_sample_class():
    """Lazily import VideoSample to avoid circular import at module load time."""
    from datasets import VideoSample as _VideoSample
    return _VideoSample


# ---------------------------------------------------------------------------
# 1. RandomHorizontalFlip
# ---------------------------------------------------------------------------


class RandomHorizontalFlip:
    """Randomly flip all frames and masks horizontally with probability p.

    Applies the same flip decision to every frame and every mask in the
    sequence (temporal consistency). Color information is unaffected.

    Paper reference: Appendix D.2.2 — "random horizontal flips"
    Config reference: training.augmentation.horizontal_flip: true

    Args:
        p: Probability of applying the horizontal flip. Defaults to 0.5.

    Example:
        flip = RandomHorizontalFlip(p=0.5)
        augmented = flip(sample)  # VideoSample with flipped frames and masks
    """

    def __init__(self, p: float = 0.5) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Probability p must be in [0, 1], got {p}.")
        self.p: float = p

    def __call__(self, sample: "VideoSample") -> "VideoSample":
        """Apply random horizontal flip to a VideoSample.

        Args:
            sample: Input VideoSample with frames [T, C, H, W] and
                masks [T, N, H, W].

        Returns:
            VideoSample with frames and masks horizontally flipped if the
            random draw succeeds, otherwise the original sample unchanged.
        """
        if random.random() >= self.p:
            return sample

        # Flip along the W dimension (dim=3 for [T, C, H, W])
        flipped_frames: Tensor = torch.flip(sample.frames, dims=[3])
        # Flip masks along W dimension (dim=3 for [T, N, H, W])
        flipped_masks: Tensor = torch.flip(sample.masks, dims=[3])

        VideoSample = _get_video_sample_class()
        return VideoSample(
            frames=flipped_frames,
            masks=flipped_masks,
            video_id=sample.video_id,
            frame_indices=sample.frame_indices,
            num_objects=sample.num_objects,
            is_occluded=sample.is_occluded,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


# ---------------------------------------------------------------------------
# 2. RandomAffine
# ---------------------------------------------------------------------------


class RandomAffine:
    """Apply a random affine transformation consistently to all frames and masks.

    Samples a single set of affine parameters (rotation, translation, scale,
    shear) once per sequence and applies the same transformation to every
    frame (bilinear interpolation) and every mask (nearest-neighbor
    interpolation to preserve binary values).

    Paper reference: Appendix D.2.2 — "random affine transforms"
    Config reference: training.augmentation.random_affine: true

    Args:
        degrees: Range of rotation in degrees. If a single float, the range
            is (-degrees, +degrees). Defaults to 15.0.
        translate: Maximum absolute fraction of total width/height for
            horizontal and vertical translations. Tuple (tx, ty) where each
            is in [0, 1]. Defaults to (0.1, 0.1) for ±10% translation.
        scale: Scaling factor range as (min_scale, max_scale). Defaults to
            (0.9, 1.1) for ±10% scale variation.
        shear: Range of shear in degrees. If a single float, the range is
            (-shear, +shear). Defaults to 5.0.
        fill_value_frames: Fill value for frames outside the transformed
            boundary. Defaults to 0.0 (black).
        fill_value_masks: Fill value for masks outside the transformed
            boundary. Defaults to 0.0 (background).

    Example:
        affine = RandomAffine(degrees=15.0, translate=(0.1, 0.1))
        augmented = affine(sample)
    """

    def __init__(
        self,
        degrees: float = 15.0,
        translate: Tuple[float, float] = (0.1, 0.1),
        scale: Tuple[float, float] = (0.9, 1.1),
        shear: float = 5.0,
        fill_value_frames: float = 0.0,
        fill_value_masks: float = 0.0,
    ) -> None:
        self.degrees: Tuple[float, float] = (-abs(degrees), abs(degrees))
        self.translate: Tuple[float, float] = translate
        self.scale: Tuple[float, float] = scale
        self.shear: Tuple[float, float] = (-abs(shear), abs(shear))
        self.fill_value_frames: float = fill_value_frames
        self.fill_value_masks: float = fill_value_masks

    def _get_params(self, height: int, width: int) -> Dict[str, Any]:
        """Sample random affine parameters for a single sequence.

        Args:
            height: Spatial height of the frames.
            width: Spatial width of the frames.

        Returns:
            Dict with keys: angle, translate_x, translate_y, scale, shear.
            translate_x and translate_y are in absolute pixels.
        """
        # Rotation angle in degrees
        angle: float = random.uniform(self.degrees[0], self.degrees[1])

        # Translation in absolute pixels
        max_dx: float = self.translate[0] * width
        max_dy: float = self.translate[1] * height
        translate_x: float = random.uniform(-max_dx, max_dx)
        translate_y: float = random.uniform(-max_dy, max_dy)

        # Scale factor
        scale: float = random.uniform(self.scale[0], self.scale[1])

        # Shear angle in degrees
        shear: float = random.uniform(self.shear[0], self.shear[1])

        return {
            "angle": angle,
            "translate_x": translate_x,
            "translate_y": translate_y,
            "scale": scale,
            "shear": shear,
        }

    def _apply_to_frame(
        self,
        frame: Tensor,
        params: Dict[str, Any],
    ) -> Tensor:
        """Apply affine transform to a single frame using bilinear interpolation.

        Args:
            frame: Single frame tensor of shape [C, H, W], float32.
            params: Affine parameters dict from _get_params().

        Returns:
            Transformed frame of shape [C, H, W], float32.
        """
        return TF.affine(
            frame,
            angle=params["angle"],
            translate=[int(params["translate_x"]), int(params["translate_y"])],
            scale=params["scale"],
            shear=params["shear"],
            interpolation=TF.InterpolationMode.BILINEAR,
            fill=self.fill_value_frames,
        )

    def _apply_to_mask(
        self,
        mask: Tensor,
        params: Dict[str, Any],
    ) -> Tensor:
        """Apply affine transform to a single mask using nearest-neighbor interpolation.

        Nearest-neighbor preserves binary values {0, 1} without introducing
        fractional values from bilinear blending.

        Args:
            mask: Single mask tensor of shape [H, W] or [N, H, W], float32 or bool.
            params: Affine parameters dict from _get_params().

        Returns:
            Transformed mask of same shape as input, float32 binary {0.0, 1.0}.
        """
        # TF.affine requires [C, H, W] format — add channel dim if needed
        squeeze: bool = False
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)  # [1, H, W]
            squeeze = True

        # Convert to float32 for TF.affine
        mask_float: Tensor = mask.float()

        transformed: Tensor = TF.affine(
            mask_float,
            angle=params["angle"],
            translate=[int(params["translate_x"]), int(params["translate_y"])],
            scale=params["scale"],
            shear=params["shear"],
            interpolation=TF.InterpolationMode.NEAREST,
            fill=self.fill_value_masks,
        )

        # Threshold to ensure binary values after any interpolation artifacts
        transformed = (transformed >= 0.5).float()

        if squeeze:
            transformed = transformed.squeeze(0)  # [H, W]

        return transformed

    def __call__(self, sample: "VideoSample") -> "VideoSample":
        """Apply random affine transform to all frames and masks in a VideoSample.

        Samples affine parameters once and applies them identically to all
        T frames and all T×N masks.

        Args:
            sample: Input VideoSample with frames [T, C, H, W] and
                masks [T, N, H, W].

        Returns:
            VideoSample with affine-transformed frames and masks.
        """
        T, C, H, W = sample.frames.shape
        N: int = sample.masks.shape[1]

        # Sample parameters once for the entire sequence
        params: Dict[str, Any] = self._get_params(H, W)

        # Apply to all frames: [T, C, H, W]
        transformed_frames: List[Tensor] = []
        for t in range(T):
            frame_t: Tensor = sample.frames[t]  # [C, H, W]
            transformed_frames.append(self._apply_to_frame(frame_t, params))
        frames_out: Tensor = torch.stack(transformed_frames, dim=0)  # [T, C, H, W]

        # Apply to all masks: [T, N, H, W]
        transformed_masks: List[Tensor] = []
        for t in range(T):
            masks_t_list: List[Tensor] = []
            for n in range(N):
                mask_tn: Tensor = sample.masks[t, n]  # [H, W]
                masks_t_list.append(self._apply_to_mask(mask_tn, params))
            # Stack N masks: [N, H, W]
            transformed_masks.append(torch.stack(masks_t_list, dim=0))
        masks_out: Tensor = torch.stack(transformed_masks, dim=0)  # [T, N, H, W]

        VideoSample = _get_video_sample_class()
        return VideoSample(
            frames=frames_out,
            masks=masks_out,
            video_id=sample.video_id,
            frame_indices=sample.frame_indices,
            num_objects=sample.num_objects,
            is_occluded=sample.is_occluded,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"degrees={self.degrees}, "
            f"translate={self.translate}, "
            f"scale={self.scale}, "
            f"shear={self.shear})"
        )


# ---------------------------------------------------------------------------
# 3. ColorJitter
# ---------------------------------------------------------------------------


class ColorJitter:
    """Apply random color jitter consistently to all frames in a sequence.

    Samples a single set of color jitter parameters (brightness, contrast,
    saturation, hue) once per sequence and applies the same color transform
    to every frame. Masks are NOT modified — they are spatial binary tensors.

    Paper reference: Appendix D.2.2 — "random color jittering"
    Config reference: training.augmentation.color_jitter: true

    Args:
        brightness: How much to jitter brightness. brightness_factor is
            chosen uniformly from [max(0, 1-brightness), 1+brightness].
            Defaults to 0.4.
        contrast: How much to jitter contrast. contrast_factor is chosen
            uniformly from [max(0, 1-contrast), 1+contrast]. Defaults to 0.4.
        saturation: How much to jitter saturation. saturation_factor is
            chosen uniformly from [max(0, 1-saturation), 1+saturation].
            Defaults to 0.4.
        hue: How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be in [0, 0.5]. Defaults to 0.1.

    Example:
        jitter = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        augmented = jitter(sample)
    """

    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1,
    ) -> None:
        self.brightness: float = brightness
        self.contrast: float = contrast
        self.saturation: float = saturation
        self.hue: float = min(abs(hue), 0.5)

    def _get_params(self) -> Dict[str, float]:
        """Sample random color jitter parameters for a single sequence.

        Returns:
            Dict with keys: brightness_factor, contrast_factor,
            saturation_factor, hue_factor.
        """
        # Brightness factor in [max(0, 1-b), 1+b]
        brightness_lo: float = max(0.0, 1.0 - self.brightness)
        brightness_hi: float = 1.0 + self.brightness
        brightness_factor: float = random.uniform(brightness_lo, brightness_hi)

        # Contrast factor in [max(0, 1-c), 1+c]
        contrast_lo: float = max(0.0, 1.0 - self.contrast)
        contrast_hi: float = 1.0 + self.contrast
        contrast_factor: float = random.uniform(contrast_lo, contrast_hi)

        # Saturation factor in [max(0, 1-s), 1+s]
        saturation_lo: float = max(0.0, 1.0 - self.saturation)
        saturation_hi: float = 1.0 + self.saturation
        saturation_factor: float = random.uniform(saturation_lo, saturation_hi)

        # Hue factor in [-hue, hue]
        hue_factor: float = random.uniform(-self.hue, self.hue)

        return {
            "brightness_factor": brightness_factor,
            "contrast_factor": contrast_factor,
            "saturation_factor": saturation_factor,
            "hue_factor": hue_factor,
        }

    def _apply_to_frame(
        self,
        frame: Tensor,
        params: Dict[str, float],
    ) -> Tensor:
        """Apply color jitter to a single frame.

        Applies brightness, contrast, saturation, and hue adjustments in a
        random order (sampled once per call) to avoid order-dependent artifacts.

        Args:
            frame: Single frame tensor of shape [C, H, W], float32 in [0, 1].
            params: Color jitter parameters dict from _get_params().

        Returns:
            Color-jittered frame of shape [C, H, W], float32 in [0, 1].
        """
        # Clamp to [0, 1] before applying color transforms
        frame = frame.clamp(0.0, 1.0)

        # Apply transforms in a random order
        fn_order: List[int] = list(range(4))
        random.shuffle(fn_order)

        for fn_idx in fn_order:
            if fn_idx == 0:
                frame = TF.adjust_brightness(frame, params["brightness_factor"])
            elif fn_idx == 1:
                frame = TF.adjust_contrast(frame, params["contrast_factor"])
            elif fn_idx == 2:
                frame = TF.adjust_saturation(frame, params["saturation_factor"])
            elif fn_idx == 3:
                frame = TF.adjust_hue(frame, params["hue_factor"])

        return frame.clamp(0.0, 1.0)

    def __call__(self, sample: "VideoSample") -> "VideoSample":
        """Apply random color jitter to all frames in a VideoSample.

        Samples color parameters once and applies them identically to all
        T frames. Masks are returned unchanged.

        Args:
            sample: Input VideoSample with frames [T, C, H, W] and
                masks [T, N, H, W].

        Returns:
            VideoSample with color-jittered frames and unchanged masks.
        """
        T: int = sample.frames.shape[0]

        # Sample parameters once for the entire sequence
        params: Dict[str, float] = self._get_params()

        # Apply to all frames
        jittered_frames: List[Tensor] = []
        for t in range(T):
            frame_t: Tensor = sample.frames[t]  # [C, H, W]
            jittered_frames.append(self._apply_to_frame(frame_t, params))
        frames_out: Tensor = torch.stack(jittered_frames, dim=0)  # [T, C, H, W]

        VideoSample = _get_video_sample_class()
        return VideoSample(
            frames=frames_out,
            masks=sample.masks,  # masks unchanged
            video_id=sample.video_id,
            frame_indices=sample.frame_indices,
            num_objects=sample.num_objects,
            is_occluded=sample.is_occluded,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}, "
            f"contrast={self.contrast}, "
            f"saturation={self.saturation}, "
            f"hue={self.hue})"
        )


# ---------------------------------------------------------------------------
# 4. RandomGrayscale
# ---------------------------------------------------------------------------


class RandomGrayscale:
    """Randomly convert all frames to grayscale with probability p.

    Applies the same grayscale decision to every frame in the sequence
    (temporal consistency). The output maintains 3 channels by replicating
    the grayscale channel. Masks are NOT modified.

    Paper reference: Appendix D.2.2 — "random grayscale transforms"
    Config reference: training.augmentation.random_grayscale: true

    Args:
        p: Probability of converting to grayscale. Defaults to 0.05
            (low probability to avoid degrading color-dependent features).

    Example:
        grayscale = RandomGrayscale(p=0.05)
        augmented = grayscale(sample)
    """

    def __init__(self, p: float = 0.05) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Probability p must be in [0, 1], got {p}.")
        self.p: float = p

    def __call__(self, sample: "VideoSample") -> "VideoSample":
        """Apply random grayscale conversion to all frames in a VideoSample.

        Args:
            sample: Input VideoSample with frames [T, C, H, W] and
                masks [T, N, H, W].

        Returns:
            VideoSample with grayscale frames (3 channels, replicated) if the
            random draw succeeds, otherwise the original sample unchanged.
        """
        if random.random() >= self.p:
            return sample

        T, C, H, W = sample.frames.shape

        # Convert each frame to grayscale and replicate to 3 channels
        gray_frames: List[Tensor] = []
        for t in range(T):
            frame_t: Tensor = sample.frames[t]  # [C, H, W]
            # TF.rgb_to_grayscale returns [1, H, W] for a [C, H, W] input
            gray: Tensor = TF.rgb_to_grayscale(frame_t, num_output_channels=1)
            # Replicate to 3 channels: [3, H, W]
            gray_3ch: Tensor = gray.expand(3, H, W)
            gray_frames.append(gray_3ch)

        frames_out: Tensor = torch.stack(gray_frames, dim=0)  # [T, 3, H, W]

        VideoSample = _get_video_sample_class()
        return VideoSample(
            frames=frames_out,
            masks=sample.masks,  # masks unchanged
            video_id=sample.video_id,
            frame_indices=sample.frame_indices,
            num_objects=sample.num_objects,
            is_occluded=sample.is_occluded,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


# ---------------------------------------------------------------------------
# 5. MosaicTransform
# ---------------------------------------------------------------------------


class MosaicTransform:
    """Tile the same video into a 2×2 grid with probability p.

    From Appendix D.2.2: "With 10% probability, we tile the same training
    video into a 2×2 grid and select a masklet from one of the 4 quadrants
    as the target object to segment. In this case, the model must focus on
    other cues like motion or temporal continuity to distinguish the target
    object from their identical-looking counterparts in other quadrants."

    Processing pipeline:
        1. Resize each frame to half resolution: [C, H, W] → [C, H//2, W//2]
        2. Tile 4 copies into a 2×2 grid: [C, H, W] (same spatial size)
        3. Resize each mask to half resolution using nearest-neighbor
        4. Tile 4 copies of each mask into a 2×2 grid
        5. Select one quadrant as the target; zero out masks in other quadrants
        6. Return updated VideoSample with tiled frames and target-only masks

    Config reference: training.mosaic_prob: 0.10

    Args:
        p: Probability of applying the mosaic transform. Defaults to 0.10
            (config.yaml: training.mosaic_prob: 0.10).

    Example:
        mosaic = MosaicTransform(p=0.10)
        augmented = mosaic(sample)
    """

    # Quadrant index → (row_start_frac, col_start_frac) in the tiled image
    # Each quadrant occupies [H//2, W//2] pixels
    QUADRANT_POSITIONS: List[Tuple[int, int]] = [
        (0, 0),  # 0: top-left
        (0, 1),  # 1: top-right
        (1, 0),  # 2: bottom-left
        (1, 1),  # 3: bottom-right
    ]

    def __init__(self, p: float = 0.10) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Probability p must be in [0, 1], got {p}.")
        self.p: float = p

    def _tile_frames(self, frames: Tensor) -> Tensor:
        """Tile T frames into a 2×2 grid at half resolution.

        Each frame is resized to half its original spatial dimensions, then
        four copies are arranged in a 2×2 grid to produce a frame of the
        same original spatial size.

        Args:
            frames: Input frames of shape [T, C, H, W], float32.

        Returns:
            Tiled frames of shape [T, C, H_out, W_out] where:
                H_out = (H // 2) * 2  (may be 1 pixel less than H if H is odd)
                W_out = (W // 2) * 2  (may be 1 pixel less than W if W is odd)
        """
        T, C, H, W = frames.shape
        h_half: int = H // 2
        w_half: int = W // 2

        # Resize all frames to half resolution: [T, C, h_half, w_half]
        # Use bilinear interpolation for frames
        frames_half: Tensor = F.interpolate(
            frames,
            size=(h_half, w_half),
            mode="bilinear",
            align_corners=False,
        )

        # Tile into 2×2 grid:
        # Top row: [top-left | top-right] → [T, C, h_half, w_half*2]
        # Bottom row: [bottom-left | bottom-right] → [T, C, h_half, w_half*2]
        # Full grid: [T, C, h_half*2, w_half*2]
        top_row: Tensor = torch.cat([frames_half, frames_half], dim=3)    # [T, C, h_half, w_half*2]
        bottom_row: Tensor = torch.cat([frames_half, frames_half], dim=3) # [T, C, h_half, w_half*2]
        tiled: Tensor = torch.cat([top_row, bottom_row], dim=2)           # [T, C, h_half*2, w_half*2]

        return tiled

    def _tile_masks(self, masks: Tensor) -> Tensor:
        """Tile T×N masks into a 2×2 grid at half resolution.

        Each mask is resized to half its original spatial dimensions using
        nearest-neighbor interpolation (to preserve binary values), then
        four copies are arranged in a 2×2 grid.

        Args:
            masks: Input masks of shape [T, N, H, W], float32 or bool.

        Returns:
            Tiled masks of shape [T, N, H_out, W_out] where:
                H_out = (H // 2) * 2
                W_out = (W // 2) * 2
        """
        T, N, H, W = masks.shape
        h_half: int = H // 2
        w_half: int = W // 2

        # Reshape to [T*N, 1, H, W] for F.interpolate (requires 4D input)
        masks_4d: Tensor = masks.float().view(T * N, 1, H, W)

        # Resize to half resolution using nearest-neighbor
        masks_half_4d: Tensor = F.interpolate(
            masks_4d,
            size=(h_half, w_half),
            mode="nearest",
        )
        # Threshold to ensure binary values after resize
        masks_half_4d = (masks_half_4d >= 0.5).float()

        # Reshape back to [T, N, h_half, w_half]
        masks_half: Tensor = masks_half_4d.view(T, N, h_half, w_half)

        # Tile into 2×2 grid
        top_row: Tensor = torch.cat([masks_half, masks_half], dim=3)    # [T, N, h_half, w_half*2]
        bottom_row: Tensor = torch.cat([masks_half, masks_half], dim=3) # [T, N, h_half, w_half*2]
        tiled: Tensor = torch.cat([top_row, bottom_row], dim=2)         # [T, N, h_half*2, w_half*2]

        return tiled

    def _select_quadrant_masks(
        self,
        tiled_masks: Tensor,
        quadrant: int,
        h_half: int,
        w_half: int,
    ) -> Tensor:
        """Zero out masks in all quadrants except the selected target quadrant.

        The model must learn to track the object in the selected quadrant
        while ignoring identical-looking copies in the other three quadrants.

        Args:
            tiled_masks: Tiled masks of shape [T, N, H_out, W_out].
            quadrant: Index of the selected target quadrant (0-3):
                0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right
            h_half: Half-height of the tiled image (height of each quadrant).
            w_half: Half-width of the tiled image (width of each quadrant).

        Returns:
            Masks of shape [T, N, H_out, W_out] with only the target quadrant
            containing non-zero values.
        """
        # Create a zero mask of the same shape
        target_masks: Tensor = torch.zeros_like(tiled_masks)

        # Determine the pixel coordinates of the selected quadrant
        row_frac, col_frac = self.QUADRANT_POSITIONS[quadrant]
        row_start: int = row_frac * h_half
        row_end: int = row_start + h_half
        col_start: int = col_frac * w_half
        col_end: int = col_start + w_half

        # Copy only the target quadrant's masks
        target_masks[:, :, row_start:row_end, col_start:col_end] = (
            tiled_masks[:, :, row_start:row_end, col_start:col_end]
        )

        return target_masks

    def __call__(self, sample: "VideoSample") -> "VideoSample":
        """Apply mosaic transform to a VideoSample with probability p.

        Args:
            sample: Input VideoSample with frames [T, C, H, W] and
                masks [T, N, H, W].

        Returns