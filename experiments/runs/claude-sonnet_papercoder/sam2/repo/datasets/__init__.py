## datasets/__init__.py
"""Public API for the SAM 2 datasets package.

This module defines the shared data contracts (VideoSample, PromptInput) and
re-exports all dataset classes and supporting utilities so that training,
evaluation, and other modules can import from a single clean namespace:

    from datasets import SAVDataset, VideoSample, PromptInput, PromptSampler

Shared data contracts are defined here (not in submodules) to prevent circular
imports: submodules import VideoSample and PromptInput from this package-level
__init__.py, which is resolved before the submodule imports execute.

Import order is carefully structured:
    1. Standard library and third-party imports
    2. Dataclass definitions (VideoSample, PromptInput)
    3. Dataset class imports from submodules
    4. Supporting utility imports

Config references (config.yaml):
    training.num_frames: 8                    → VideoSample.frames shape [T=8, C, H, W]
    training.max_masklets_per_sequence: 3     → VideoSample.num_objects <= 3
    training.occlusion_supervision.*          → VideoSample.is_occluded usage
    sav_dataset.*                             → SAVDataset configuration
    pretrain.data.dataset: "sa1b"             → SA1BDataset usage
    evaluation.image_segmentation.*           → ImageDataset usage

Paper references:
    Section 4: "we sample sequences of 8 frames and randomly select up to 2
        frames to prompt"
    Appendix D.2.2: "We restrict the maximum number of masklets for each
        sequence of 8 frames to 3 randomly chosen ones."
    Section 4: "it is possible for no valid object to exist on some frames
        (e.g. due to occlusion)" → VideoSample.is_occluded
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Shared data contracts — defined here to prevent circular imports
# ---------------------------------------------------------------------------


@dataclass
class VideoSample:
    """Data contract between all dataset classes and the trainer/evaluator.

    Represents a single training or evaluation sample consisting of T video
    frames with N annotated objects (masklets). This is the primary data
    structure passed from dataset __getitem__ to the training loop and
    evaluation pipeline.

    All tensor fields use consistent dimension ordering:
        T = number of sampled frames (8 for training, 16 for fine-tuning)
        N = number of objects/masklets in this sample (up to 3 for training)
        C = number of image channels (3 for RGB)
        H, W = spatial dimensions (1024×1024 for full training, 512×512 for ablations)

    Config references:
        training.num_frames: 8          → T dimension
        finetuning.num_frames: 16       → T dimension for fine-tuning
        training.max_masklets_per_sequence: 3  → N dimension upper bound
        model.input_resolution: 1024    → H, W dimensions

    Attributes:
        frames: Video frames tensor of shape [T, C, H, W], dtype float32.
            Pixel values normalized to [0, 1] or ImageNet-normalized depending
            on the encoder's preprocessing requirements.
        masks: Binary segmentation masks tensor of shape [T, N, H, W], dtype bool
            or float32. True/1.0 indicates foreground pixels for each object.
            A frame with no valid mask for an object (occluded) is represented
            as an all-zero mask for that object on that frame.
        video_id: String identifier for the source video. Used for logging,
            debugging, and result saving. Format varies by dataset:
            SA-V: video directory name, DAVIS: sequence name, etc.
        frame_indices: List of T integers indicating which frame indices were
            sampled from the full video. Length must equal T (frames.shape[0]).
            Used by the trainer to reconstruct temporal ordering and by the
            evaluator to map predictions back to the original video timeline.
        num_objects: Number of distinct objects/masklets in this sample.
            Equals N (masks.shape[1]). Stored explicitly for convenience.
            Upper bound: training.max_masklets_per_sequence = 3.
        is_occluded: Per-frame-per-object occlusion flags as a flat list of
            T * N booleans. Indexed as is_occluded[t * N + n] for frame t,
            object n. True means the object is not visible in that frame
            (e.g., fully occluded or out of frame). Used by the occlusion
            supervision head in training/losses.py.
            Config: training.occlusion_supervision.always_supervise_occlusion_head

    Example:
        sample = VideoSample(
            frames=torch.randn(8, 3, 1024, 1024),
            masks=torch.zeros(8, 2, 1024, 1024, dtype=torch.bool),
            video_id="sav_video_001",
            frame_indices=[0, 5, 10, 15, 20, 25, 30, 35],
            num_objects=2,
            is_occluded=[False] * 16,  # 8 frames * 2 objects
        )
    """

    frames: Tensor
    masks: Tensor
    video_id: str
    frame_indices: List[int]
    num_objects: int
    is_occluded: List[bool]

    def __post_init__(self) -> None:
        """Validate consistency of VideoSample fields after construction.

        Checks that:
        - frames and masks have matching T and H, W dimensions
        - frame_indices length matches T
        - is_occluded length matches T * N
        - num_objects matches masks.shape[1]

        Raises:
            ValueError: If any consistency check fails.
        """
        if self.frames.ndim != 4:
            raise ValueError(
                f"VideoSample.frames must be 4D [T, C, H, W], "
                f"got shape {self.frames.shape}."
            )
        if self.masks.ndim != 4:
            raise ValueError(
                f"VideoSample.masks must be 4D [T, N, H, W], "
                f"got shape {self.masks.shape}."
            )

        T_frames: int = self.frames.shape[0]
        T_masks: int = self.masks.shape[0]
        N: int = self.masks.shape[1]

        if T_frames != T_masks:
            raise ValueError(
                f"VideoSample.frames has T={T_frames} frames but "
                f"VideoSample.masks has T={T_masks} frames. They must match."
            )

        if len(self.frame_indices) != T_frames:
            raise ValueError(
                f"VideoSample.frame_indices has length {len(self.frame_indices)} "
                f"but VideoSample.frames has T={T_frames}. They must match."
            )

        if self.num_objects != N:
            raise ValueError(
                f"VideoSample.num_objects={self.num_objects} but "
                f"VideoSample.masks.shape[1]={N}. They must match."
            )

        expected_occ_len: int = T_frames * N
        if len(self.is_occluded) != expected_occ_len:
            raise ValueError(
                f"VideoSample.is_occluded has length {len(self.is_occluded)} "
                f"but expected T*N={T_frames}*{N}={expected_occ_len}."
            )

    def get_occlusion_mask(self) -> Tensor:
        """Return is_occluded as a boolean tensor of shape [T, N].

        Convenience method for the trainer and evaluator to access occlusion
        flags in tensor format without manual reshaping.

        Returns:
            Boolean tensor of shape [T, N] where True means the object is
            occluded (not visible) in that frame.
        """
        T: int = self.frames.shape[0]
        N: int = self.masks.shape[1]
        return torch.tensor(
            self.is_occluded,
            dtype=torch.bool,
        ).view(T, N)

    def get_frame_occlusion(self, frame_t: int, object_n: int) -> bool:
        """Get the occlusion flag for a specific frame and object.

        Args:
            frame_t: Frame index in [0, T).
            object_n: Object index in [0, N).

        Returns:
            True if the object is occluded in the specified frame.

        Raises:
            IndexError: If frame_t or object_n is out of range.
        """
        T: int = self.frames.shape[0]
        N: int = self.masks.shape[1]

        if frame_t < 0 or frame_t >= T:
            raise IndexError(
                f"frame_t={frame_t} is out of range [0, {T})."
            )
        if object_n < 0 or object_n >= N:
            raise IndexError(
                f"object_n={object_n} is out of range [0, {N})."
            )

        flat_idx: int = frame_t * N + object_n
        return self.is_occluded[flat_idx]


@dataclass
class PromptInput:
    """Input prompt container for a single video frame.

    Shared data contract between PromptSampler (training), all evaluators
    (evaluation), and SAM2Model.forward_video_frame (inference). All fields
    are optional — a PromptInput with all None fields represents an unprompted
    propagation frame.

    Coordinate convention: (x, y) pixel coordinates for points and boxes,
    consistent with PromptEncoder's expected input format. The x-axis is
    horizontal (column) and y-axis is vertical (row).

    Config references:
        training.prompt_probabilities.gt_mask: 0.50
        training.prompt_probabilities.positive_click: 0.25
        training.prompt_probabilities.bounding_box: 0.25
        evaluation.interactive.num_clicks_per_frame: 3

    Paper references:
        Section 4: "Initial prompts to the model can be the ground-truth mask
            with probability 0.5, a positive click sampled from the ground-truth
            mask with probability 0.25, or a bounding box input with probability
            0.25."
        Appendix F.1.2: "we simulate interactive video segmentation with
            N_click = 3 clicks per frame"

    Attributes:
        points: Optional click coordinates of shape [N, 2] in (x, y) pixel
            space. None if no click prompts for this frame. Values in
            [0, input_resolution] for both x and y.
        point_labels: Optional click labels of shape [N] with integer values:
            1 = positive click (foreground), 0 = negative click (background),
            -1 = padding token (used when boxes are also present).
            Must be provided when points is not None.
        boxes: Optional bounding box tensor. Shape [4] for a single box in
            (x1, y1, x2, y2) pixel coordinates, or [B, 4] for batched boxes.
            None if no box prompt for this frame.
        masks: Optional mask prompt tensor of shape [1, H, W] or [B, 1, H, W].
            Values can be binary {0, 1}, probabilities [0, 1], or raw logits.
            None if no mask prompt for this frame.
        frame_idx: Integer index of the video frame this prompt belongs to
            (0-based). Used by SAM2Model.propagate_video() to determine which
            frames are prompted vs. unprompted propagation frames.

    Example:
        # Single positive click at pixel (512, 256)
        prompt = PromptInput(
            points=torch.tensor([[512.0, 256.0]]),
            point_labels=torch.tensor([1]),
            frame_idx=0,
        )

        # Bounding box prompt
        prompt = PromptInput(
            boxes=torch.tensor([100.0, 200.0, 600.0, 800.0]),
            frame_idx=0,
        )

        # No prompt (propagation frame)
        prompt = PromptInput(frame_idx=5)
    """

    points: Optional[Tensor] = None
    point_labels: Optional[Tensor] = None
    boxes: Optional[Tensor] = None
    masks: Optional[Tensor] = None
    frame_idx: int = 0

    def __post_init__(self) -> None:
        """Validate consistency of PromptInput fields after construction.

        Checks that:
        - point_labels is provided when points is not None
        - points and point_labels have matching N dimension

        Raises:
            ValueError: If points is provided without point_labels, or if
                their shapes are inconsistent.
        """
        if self.points is not None and self.point_labels is None:
            raise ValueError(
                "PromptInput.point_labels must be provided when "
                "PromptInput.points is not None."
            )

        if self.points is not None and self.point_labels is not None:
            # Validate matching N dimension
            pts_n: int = self.points.shape[0] if self.points.ndim == 2 else self.points.shape[1]
            lbl_n: int = self.point_labels.shape[0] if self.point_labels.ndim == 1 else self.point_labels.shape[1]
            if pts_n != lbl_n:
                raise ValueError(
                    f"PromptInput.points has N={pts_n} clicks but "
                    f"PromptInput.point_labels has N={lbl_n}. They must match."
                )

    def is_empty(self) -> bool:
        """Check whether this prompt contains any actual prompt data.

        Returns:
            True if all prompt fields (points, boxes, masks) are None.
            A PromptInput with only frame_idx set is considered empty —
            it represents an unprompted propagation frame.
        """
        return (
            self.points is None
            and self.boxes is None
            and self.masks is None
        )

    def has_clicks(self) -> bool:
        """Check whether this prompt contains click (point) prompts.

        Returns:
            True if points is not None and contains at least one click.
        """
        return self.points is not None and self.points.shape[0] > 0

    def has_box(self) -> bool:
        """Check whether this prompt contains a bounding box prompt.

        Returns:
            True if boxes is not None.
        """
        return self.boxes is not None

    def has_mask(self) -> bool:
        """Check whether this prompt contains a mask prompt.

        Returns:
            True if masks is not None.
        """
        return self.masks is not None

    def num_clicks(self) -> int:
        """Return the number of click prompts.

        Returns:
            Integer count of clicks. 0 if points is None.
        """
        if self.points is None:
            return 0
        if self.points.ndim == 2:
            return int(self.points.shape[0])
        elif self.points.ndim == 3:
            return int(self.points.shape[1])
        return 0


# ---------------------------------------------------------------------------
# Dataset class imports
# ---------------------------------------------------------------------------
# These imports come AFTER the dataclass definitions above so that submodules
# can safely import VideoSample and PromptInput from this package without
# triggering circular import errors.

from datasets.sa1b_dataset import SA1BDataset
from datasets.sav_dataset import SAVDataset
from datasets.vos_dataset import VOSDataset
from datasets.image_dataset import ImageDataset

# ---------------------------------------------------------------------------
# Supporting utility imports
# ---------------------------------------------------------------------------

from datasets.prompt_sampler import PromptSampler
from datasets.transforms import VideoTransforms

# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__: List[str] = [
    # Shared data contracts — used by datasets, trainer, and evaluators
    "VideoSample",
    "PromptInput",
    # Dataset classes
    "SA1BDataset",
    "SAVDataset",
    "VOSDataset",
    "ImageDataset",
    # Supporting utilities
    "PromptSampler",
    "VideoTransforms",
]
