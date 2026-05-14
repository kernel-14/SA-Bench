## Code: datasets/sav_dataset.py

```python
## datasets/sav_dataset.py
"""SA-V dataset loader for SAM 2 full training and fine-tuning (Stage 2 & 3).

This module implements SAVDataset, a PyTorch Dataset that loads the SA-V
(Segment Anything Video) dataset for joint image/video training of SAM 2.

SA-V dataset statistics (config.yaml sav_dataset section):
    - 50.9K videos, 190.9K manual masklets, annotated at 6 FPS
    - Average video duration: 13.8 seconds (~83 frames at 6 FPS)
    - Average resolution: 1401×1037 pixels
    - SA-V val: 293 masklets, 155 videos
    - SA-V test: 278 masklets, 150 videos

Two training modes:
    - Standard training (num_frames=8): all manually annotated masklets
    - Fine-tuning (num_frames=16): top-50% most-edited masklets only

Config references (config.yaml):
    training.num_frames: 8
    training.temporal_reversal_prob: 0.50
    training.mosaic_prob: 0.10
    training.use_auto_masklets_for_training: false
    training.max_masklets_per_sequence: 3
    finetuning.num_frames: 16
    finetuning.most_edited_fraction: 0.50
    sav_dataset.annotation_fps: 6
    model.input_resolution: 1024

Paper references:
    Section 5.2: "SA-V dataset comprises 50.9K videos with 642.6K masklets."
    Appendix D.2.2: "we sample sequences of 8 frames and randomly select up
        to 2 frames to prompt"
    Appendix D.2.2: "We reverse the temporal order with a probability of 50%"
    Appendix D.2.2: "With 10% probability, we tile the same training video
        into a 2×2 grid"
    Appendix D.2.2: "we sort our masklets by number of edited frames and only
        consider the top 50% most edited masklets for training"
    Appendix D.2.2: "we only use those manually annotated masklets"
"""

import glob
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

import pycocotools.mask as coco_mask_utils

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SA-V dataset constants (from config.yaml)
# ---------------------------------------------------------------------------

# Default number of frames per training sequence (config: training.num_frames)
_DEFAULT_NUM_FRAMES: int = 8

# Default fine-tuning sequence length (config: finetuning.num_frames)
_FINETUNE_NUM_FRAMES: int = 16

# Probability of temporal reversal (config: training.temporal_reversal_prob)
_TEMPORAL_REVERSAL_PROB: float = 0.50

# Probability of mosaic transform (config: training.mosaic_prob)
_MOSAIC_PROB: float = 0.10

# Fraction of most-edited masklets to keep for fine-tuning
# (config: finetuning.most_edited_fraction)
_MOST_EDITED_FRACTION: float = 0.50

# SA-V annotation FPS (config: sav_dataset.annotation_fps)
_ANNOTATION_FPS: int = 6

# Default input resolution (config: model.input_resolution)
_DEFAULT_RESOLUTION: int = 1024

# ImageNet normalization constants (standard for MAE pre-trained Hiera)
_IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Quadrant positions for mosaic transform: (row_half_idx, col_half_idx)
# 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
_QUADRANT_POSITIONS: List[Tuple[int, int]] = [
    (0, 0),  # top-left
    (0, 1),  # top-right
    (1, 0),  # bottom-left
    (1, 1),  # bottom-right
]


# ---------------------------------------------------------------------------
# Lazy import helper to avoid circular imports
# ---------------------------------------------------------------------------


def _get_video_sample_class():
    """Lazily import VideoSample to avoid circular import at module load time.

    Returns:
        The VideoSample dataclass from datasets/__init__.py.
    """
    from datasets import VideoSample as _VideoSample
    return _VideoSample


# ---------------------------------------------------------------------------
# SAVDataset
# ---------------------------------------------------------------------------


class SAVDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for SA-V video and masklet loading for SAM 2 training.

    Loads SA-V videos and manually annotated masklets. Supports two training
    modes controlled by `num_frames`:
        - Standard training (num_frames=8): all manually annotated masklets
        - Fine-tuning (num_frames=16): top-50% most-edited masklets only

    Indexing is by masklet (not video): __len__ returns the number of masklets
    in the filtered index. Each __getitem__ call returns one VideoSample
    containing T frames and the masks for one masklet.

    Expected SA-V directory structure:
        root/
          sav_train/
            videos/
              <video_id>/
                00000.jpg
                00001.jpg
                ...
            annotations/
              <video_id>/
                <masklet_id>.json
          sav_val/
            ...
          sav_test/
            ...

    Each masklet annotation JSON contains:
        {
          "video_id": str,
          "masklet_id": str,
          "num_edited_frames": int,
          "is_manual": bool,
          "frames": [
            {
              "frame_idx": int,
              "is_occluded": bool,
              "segmentation": {"size": [H, W], "counts": str}  # COCO RLE
            },
            ...
          ]
        }

    Config references (config.yaml):
        training.num_frames: 8
        training.temporal_reversal_prob: 0.50
        training.mosaic_prob: 0.10
        training.use_auto_masklets_for_training: false
        finetuning.num_frames: 16
        finetuning.most_edited_fraction: 0.50
        sav_dataset.annotation_fps: 6
        model.input_resolution: 1024

    Args:
        root: Path to the SA-V dataset root directory containing split
            subdirectories (sav_train, sav_val, sav_test).
        split: Dataset split. One of "train", "val", "test".
        num_frames: Number of frames to sample per sequence. Use 8 for
            standard training (config: training.num_frames) or 16 for
            fine-tuning (config: finetuning.num_frames). Defaults to 8.
        transform: Optional callable that accepts a VideoSample and returns
            a transformed VideoSample. Applied after mosaic transform.
            If None, only basic resizing to input_resolution is applied.
        input_resolution: Target spatial resolution for frames and masks.
            Defaults to 1024 (config: model.input_resolution).
        temporal_reversal_prob: Probability of reversing temporal order.
            Defaults to 0.50 (config: training.temporal_reversal_prob).
        mosaic_prob: Probability of applying mosaic transform.
            Defaults to 0.10 (config: training.mosaic_prob).
        most_edited_fraction: Fraction of most-edited masklets to keep when
            num_frames == 16 (fine-tuning mode).
            Defaults to 0.50 (config: finetuning.most_edited_fraction).
        use_auto_masklets: If True, include automatically generated masklets.
            Defaults to False (config: training.use_auto_masklets_for_training).
        max_masklets: Optional cap on total masklets for debugging. None means
            use all available masklets.

    Example:
        dataset = SAVDataset(
            root="/data/sav",
            split="train",
            num_frames=8,
        )
        sample = dataset[0]
        # sample.frames: Tensor[8, 3, 1024, 1024]
        # sample.masks: Tensor[8, 1, 1024, 1024]
        # sample.num_objects: 1
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        num_frames: int = _DEFAULT_NUM_FRAMES,
        transform: Optional[Callable] = None,
        input_resolution: int = _DEFAULT_RESOLUTION,
        temporal_reversal_prob: float = _TEMPORAL_REVERSAL_PROB,
        mosaic_prob: float = _MOSAIC_PROB,
        most_edited_fraction: float = _MOST_EDITED_FRACTION,
        use_auto_masklets: bool = False,
        max_masklets: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Validate split
        valid_splits: List[str] = ["train", "val", "test"]
        if split not in valid_splits:
            raise ValueError(
                f"split must be one of {valid_splits}, got '{split}'."
            )

        self.root: str = root
        self.split: str = split
        self.num_frames: int = num_frames
        self.transform: Optional[Callable] = transform
        self.input_resolution: int = input_resolution
        self.temporal_reversal_prob: float = temporal_reversal_prob
        self.mosaic_prob: float = mosaic_prob
        self.most_edited_fraction: float = most_edited_fraction
        self.use_auto_masklets: bool = use_auto_masklets

        # Determine if fine-tuning mode (16-frame sequences)
        self._is_finetune_mode: bool = (num_frames == _FINETUNE_NUM_FRAMES)

        # Precompute normalization tensors for frame loading
        self._mean: Tensor = torch.tensor(
            _IMAGENET_MEAN, dtype=torch.float32
        ).view(3, 1, 1)
        self._std: Tensor = torch.tensor(
            _IMAGENET_STD, dtype=torch.float32
        ).view(3, 1, 1)

        # Build the split directory path
        # SA-V uses "sav_train", "sav_val", "sav_test" subdirectory names
        split_dir_map: Dict[str, str] = {
            "train": "sav_train",
            "val": "sav_val",
            "test": "sav_test",
        }
        self._split_dir: str = os.path.join(root, split_dir_map[split])
        self._videos_dir: str = os.path.join(self._split_dir, "videos")
        self._annotations_dir: str = os.path.join(self._split_dir, "annotations")

        # Build the masklet index: List[Dict] where each entry is one masklet
        self.masklet_index: List[Dict] = self._build_masklet_index(
            use_auto_masklets=use_auto_masklets,
            max_masklets=max_masklets,
        )

        # Apply fine-tuning filter: keep only top-50% most-edited masklets
        if self._is_finetune_mode and split == "train":
            self.masklet_index = self._apply_finetune_filter(
                self.masklet_index,
                fraction=most_edited_fraction,
            )
            logger.info(
                "SAVDataset (fine-tuning mode): kept %d masklets "
                "(top %.0f%% most-edited) from %s split.",
                len(self.masklet_index),
                most_edited_fraction * 100,
                split,
            )

        # Build video list (unique video IDs in the index)
        self.video_list: List[str] = sorted(
            list({entry["video_id"] for entry in self.masklet_index})
        )

        logger.info(
            "SAVDataset initialized: split=%s, num_frames=%d, "
            "num_masklets=%d, num_videos=%d, finetune_mode=%s",
            split,
            num_frames,
            len(self.masklet_index),
            len(self.video_list),
            self._is_finetune_mode,
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of masklets in the filtered index.

        Returns:
            Integer count of masklets. This is the unit of indexing —
            each __getitem__ call returns one masklet's VideoSample.
        """
        return len(self.masklet_index)

    def __getitem__(self, idx: int) -> Any:
        """Load and return a VideoSample for one masklet.

        Full processing pipeline:
            1. Look up masklet_index[idx]
            2. Sample T frame indices from the video
            3. Apply temporal reversal (50% probability, train only)
            4. Load T frames from disk
            5. Load masklet annotations for the T frames
            6. Build VideoSample
            7. Apply mosaic transform (10% probability, train only)
            8. Apply general transforms (flip, affine, color, resize)

        Args:
            idx: Integer index into self.masklet_index.

        Returns:
            VideoSample with:
                - frames: Tensor[T, 3, H, W] float32, ImageNet-normalized
                - masks: Tensor[T, 1, H, W] float32 binary {0, 1}
                - video_id: str
                - frame_indices: List[int] of length T
                - num_objects: 1 (single masklet per sample)
                - is_occluded: List[bool] of length T (T * 1 = T entries)

        Note:
            Returns a placeholder VideoSample on errors to allow DataLoader
            to continue without crashing.
        """
        VideoSample = _get_video_sample_class()

        entry: Dict = self.masklet_index[idx]
        video_id: str = entry["video_id"]
        masklet_id: str = entry["masklet_id"]
        video_length: int = entry["video_length"]

        # ------------------------------------------------------------------
        # Step 1: Sample frame indices
        # ------------------------------------------------------------------
        frame_indices: List[int] = self._sample_frame_indices(video_length)

        # ------------------------------------------------------------------
        # Step 2: Apply temporal reversal (train split only)
        # Config: training.temporal_reversal_prob: 0.50
        # Paper: "We reverse the temporal order with a probability of 50%"
        # ------------------------------------------------------------------
        if self.split == "train" and random.random() < self.temporal_reversal_prob:
            frame_indices = list(reversed(frame_indices))

        # ------------------------------------------------------------------
        # Step 3: Load frames from disk
        # ------------------------------------------------------------------
        try:
            frames: Tensor = self._load_video_frames(video_id, frame_indices)
        except Exception as exc:
            logger.warning(
                "SAVDataset: Failed to load frames for video %s: %s. "
                "Returning placeholder sample.",
                video_id,
                exc,
            )
            return self._make_placeholder_sample(video_id, frame_indices)

        # ------------------------------------------------------------------
        # Step 4: Load masklet annotations for the sampled frames
        # ------------------------------------------------------------------
        try:
            masks_raw, is_occluded = self._load_masklets(
                video_id, masklet_id, frame_indices
            )
        except Exception as exc:
            logger.warning(
                "SAVDataset: Failed to load masklets for video %s, "
                "masklet %s: %s. Returning placeholder sample.",
                video_id,
                masklet_id,
                exc,
            )
            return self._make_placeholder_sample(video_id, frame_indices)

        # masks_raw: Tensor[T, H_orig, W_orig] binary float32
        # Add object dimension: [T, 1, H_orig, W_orig]
        masks: Tensor = masks_raw.unsqueeze(1)

        # ------------------------------------------------------------------
        # Step 5: Build VideoSample
        # is_occluded has T entries (T frames * 1 object)
        # ------------------------------------------------------------------
        sample: Any = VideoSample(
            frames=frames,
            masks=masks,
            video_id=video_id,
            frame_indices=frame_indices,
            num_objects=1,
            is_occluded=is_occluded,
        )

        # ------------------------------------------------------------------
        # Step 6: Apply mosaic transform (train split only, 10% probability)
        # Config: training.mosaic_prob: 0.10
        # Paper: "With 10% probability, we tile the same training video into
        #         a 2×2 grid"
        # ------------------------------------------------------------------
        if self.split == "train" and random.random() < self.mosaic_prob:
            sample = self._apply_mosaic_transform(sample)

        # ------------------------------------------------------------------
        # Step 7: Apply general transforms (flip, affine, color, resize)
        # The transform callable handles resizing to input_resolution.
        # ------------------------------------------------------------------
        if self.transform is not None:
            try:
                sample = self.transform(sample)
            except Exception as exc:
                logger.warning(
                    "SAVDataset: Transform failed for video %s: %s. "
                    "Applying fallback resize only.",
                    video_id,
                    exc,
                )
                sample = self._apply_fallback_resize(sample)
        else:
            # No transform provided — apply basic resize to input_resolution
            sample = self._apply_fallback_resize(sample)

        return sample

    # ------------------------------------------------------------------
    # Core loading methods
    # ------------------------------------------------------------------

    def _load_video_frames(
        self,
        video_id: str,
        frame_indices: List[int],
    ) -> Tensor:
        """Load video frames from disk and return as a normalized tensor.

        Loads JPEG frames from the SA-V video directory, converts to float32,
        resizes to input_resolution, and applies ImageNet normalization.

        Args:
            video_id: String identifier for the video (directory name).
            frame_indices: List of T integer frame indices to load.
                Indices are 0-based and correspond to the annotation FPS (6 FPS).

        Returns:
            Tensor of shape [T, 3, H_orig, W_orig] float32, ImageNet-normalized.
            Spatial dimensions are at original video resolution — the transform
            callable handles resizing to input_resolution.

        Raises:
            FileNotFoundError: If the video directory does not exist.
            RuntimeError: If any frame file cannot be loaded.
        """
        video_dir: str = os.path.join(self._videos_dir, video_id)

        if not os.path.isdir(video_dir):
            raise FileNotFoundError(
                f"Video directory not found: {video_dir}. "
                f"Check that SA-V is correctly extracted at {self._videos_dir}."
            )

        # Discover available frame files in the video directory
        # SA-V frames are named as zero-padded integers: 00000.jpg, 00001.jpg, ...
        frame_files: List[str] = self._get_frame_files(video_dir)

        if len(frame_files) == 0:
            raise RuntimeError(
                f"No frame files found in video directory: {video_dir}."
            )

        loaded_frames: List[Tensor] = []

        for frame_idx in frame_indices:
            # Clamp frame_idx to valid range (handles short video padding)
            clamped_idx: int = min(frame_idx, len(frame_files) - 1)
            frame_path: str = frame_files[clamped_idx]

            frame_tensor: Tensor = self._load_single_frame(frame_path)
            loaded_frames.append(frame_tensor)

        # Stack: [T, 3, H, W]
        frames: Tensor = torch.stack(loaded_frames, dim=0)

        return frames

    def _load_single_frame(self, frame_path: str) -> Tensor:
        """Load a single JPEG frame and convert to normalized float32 tensor.

        Args:
            frame_path: Absolute path to the JPEG frame file.

        Returns:
            Tensor of shape [3, H, W] float32, ImageNet-normalized.

        Raises:
            RuntimeError: If the frame cannot be loaded.
        """
        try:
            img: Image.Image = Image.open(frame_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load frame from {frame_path}: {exc}"
            ) from exc

        # Convert PIL Image to float32 tensor [3, H, W] in [0, 1]
        img_np: np.ndarray = np.array(img, dtype=np.float32) / 255.0
        frame_tensor: Tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        # Apply ImageNet normalization
        frame_tensor = (frame_tensor - self._mean) / self._std

        return frame_tensor

    def _load_masklets(
        self,
        video_id: str,
        masklet_id: str,
        frame_indices: List[int],
    ) -> Tuple[Tensor, List[bool]]:
        """Load masklet annotations for the specified frames.

        Reads the masklet JSON annotation file, extracts masks for the
        requested frame indices, and decodes RLE encodings to binary arrays.

        Args:
            video_id: String identifier for the video.
            masklet_id: String identifier for the masklet within the video.
            frame_indices: List of T integer frame indices to load masks for.

        Returns:
            Tuple of:
                - masks: Tensor[T, H_orig, W_orig] float32 binary {0.0, 1.0}.
                  Zero mask for frames where the object is occluded or absent.
                - is_occluded: List[bool] of length T. True means the object
                  is not visible in that frame (occluded or out of frame).

        Raises:
            FileNotFoundError: If the annotation file does not exist.
            ValueError: If the annotation file is malformed.
        """
        annotation_path: str = os.path.join(
            self._annotations_dir, video_id, f"{masklet_id}.json"
        )

        if not os.path.isfile(annotation_path):
            raise FileNotFoundError(
                f"Masklet annotation file not found: {annotation_path}."
            )

        # Load annotation JSON
        try:
            with open(annotation_path, "r", encoding="utf-8") as f:
                annotation_data: Dict = json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"Failed to parse annotation file {annotation_path}: {exc}"
            ) from exc

        # Build a lookup from frame_idx → annotation entry
        # annotation_data["frames"] is a list of per-frame dicts
        frame_annotations: List[Dict] = annotation_data.get("frames", [])
        frame_ann_lookup: Dict[int, Dict] = {
            ann["frame_idx"]: ann
            for ann in frame_annotations
            if "frame_idx" in ann
        }

        # Determine image dimensions from the first available annotation
        # (needed to create zero masks for missing frames)
        img_height: int = 0
        img_width: int = 0
        for ann in frame_annotations:
            seg: Optional[Dict] = ann.get("segmentation")
            if seg is not None and "size" in seg:
                size: List[int] = seg["size"]
                if len(size) >= 2:
                    img_height = int(size[0])
                    img_width = int(size[1])
                    break

        # Fallback: try to get dimensions from video frames
        if img_height == 0 or img_width == 0:
            img_height, img_width = self._get_video_dimensions(video_id)

        # Extract masks and occlusion flags for each requested frame index
        masks_list: List[Tensor] = []
        is_occluded: List[bool] = []

        for frame_idx in frame_indices:
            # Clamp to valid annotation range
            ann_entry: Optional[Dict] = frame_ann_lookup.get(frame_idx)

            if ann_entry is None:
                # Frame not annotated — treat as occluded (zero mask)
                zero_mask: Tensor = torch.zeros(
                    img_height, img_width, dtype=torch.float32
                )
                masks_list.append(zero_mask)
                is_occluded.append(True)
                continue

            # Check explicit occlusion flag
            frame_is_occluded: bool = bool(ann_entry.get("is_occluded", False))

            # Decode RLE mask
            segmentation: Optional[Dict] = ann_entry.get("segmentation")
            if segmentation is None or frame_is_occluded:
                # No segmentation or explicitly occluded — zero mask
                zero_mask = torch.zeros(
                    img_height, img_width, dtype=torch.float32
                )
                masks_list.append(zero_mask)
                is_occluded.append(True)
                continue

            # Decode RLE to binary numpy array
            decoded_mask: Optional[np.ndarray] = self._decode_rle_mask(
                segmentation, img_height, img_width
            )

            if decoded_mask is None:
                # Decode failed — treat as occluded
                zero_mask = torch.zeros(
                    img_height, img_width, dtype=torch.float32
                )
                masks_list.append(zero_mask)
                is_occluded.append(True)
                continue

            # Convert to float32 tensor
            mask_tensor: Tensor = torch.from_numpy(
                decoded_mask.astype(np.float32)
            )

            # Check if mask is effectively empty (all zeros)
            mask_is_empty: bool = float(mask_tensor.sum().item()) == 0.0
            masks_list.append(mask_tensor)
            is_occluded.append(frame_is_occluded or mask_is_empty)

        # Stack: [T, H_orig, W_orig]
        masks: Tensor = torch.stack(masks_list, dim=0)

        return masks, is_occluded

    def _sample_frame_indices(self, video_length: int) -> List[int]:
        """Sample T frame indices from a video of the given length.

        Sampling strategy:
            - If video_length >= num_frames: sample a random contiguous window
              of length num_frames starting at a random position.
            - If video_length < num_frames: use all available frames and pad
              by repeating the last frame to reach num_frames.

        For val/test splits, always sample the first num_frames frames
        (deterministic for reproducibility).

        Args:
            video_length: Total number of frames in the video at annotation FPS.

        Returns:
            List of T integer frame indices (0-based). Length equals num_frames.
        """
        T: int = self.num_frames

        if video_length <= 0:
            # Degenerate case — return zeros
            return [0] * T

        if self.split != "train":
            # Deterministic sampling for val/test: use first T frames
            indices: List[int] = list(range(min(T, video_length)))
            # Pad with last frame if video is shorter than T
            while len(indices) < T:
                indices.append(indices[-1])
            return indices

        if video_length >= T:
            # Random contiguous window: start in [0, video_length - T]
            start_idx: int = random.randint(0, video_length - T)
            return list(range(start_idx, start_idx + T))
        else:
            # Video shorter than T: use all frames + repeat last frame
            indices = list(range(video_length))
            last_frame: int = video_length - 1
            while len(indices) < T:
                indices.append(last_frame)
            return indices

    def _apply_mosaic_transform(self, sample: Any) -> Any:
        """Apply the 2×2 mosaic transform to a VideoSample.

        Tiles the same video into a 2×2 grid at half resolution, then selects
        one quadrant as the target object. The model must distinguish the target
        from identical-looking copies in the other three quadrants.

        From Appendix D.2.2: "With 10% probability, we tile the same training
        video into a 2×2 grid and select a masklet from one of the 4 quadrants
        as the target object to segment. In this case, the model must focus on
        other cues like motion or temporal continuity to distinguish the target
        object from their identical-looking counterparts in other quadrants."

        Processing pipeline:
            1. Resize each frame to (H//2, W//2) — half resolution
            2. Tile 4 copies into a 2×2 grid → same spatial size (H, W)
            3. Resize each mask to (H//2, W//2) using nearest-neighbor
            4. Select target quadrant q ∈ {0, 1, 2, 3} randomly
            5. Place resized mask in quadrant q; zero out other quadrants
            6. Return updated VideoSample

        Args:
            sample: Input VideoSample with frames [T, C, H, W] and
                masks [T, N, H, W].

        Returns:
            VideoSample with mosaicked frames [T, C, H, W] and target-only
            masks [T, N, H, W]. Spatial dimensions are preserved.
        """
        VideoSample = _get_video_sample_class()

        frames: Tensor = sample.frames   # [T, C, H, W]
        masks: Tensor = sample.masks     # [T, N, H, W]

        T, C, H, W = frames.shape
        N: int = masks.shape[1]

        h_half: int = H // 2
        w_half: int = W // 2

        if h_half == 0 or w_half == 0:
            # Cannot apply mosaic to very small frames — return unchanged
            logger.debug(
                "_apply_mosaic_transform: frame too small (%d, %d) for mosaic. "
                "Returning unchanged sample.",
                H,
                W,
            )
            return sample

        # ------------------------------------------------------------------
        # Step 1: Resize frames to half resolution using bilinear interpolation
        # frames: [T, C, H, W] → [T, C, h_half, w_half]
        # ------------------------------------------------------------------
        frames_half: Tensor = F.interpolate(
            frames,
            size=(h_half, w_half),
            mode="bilinear",
            align_corners=False,
        )

        # ------------------------------------------------------------------
        # Step 2: Tile 4 copies into a 2×2 grid
        # Top row:    [TL | TR] → [T, C, h_half, w_half*2]
        # Bottom row: [BL | BR] → [T, C, h_half, w_half*2]
        # Full grid:  [T, C, h_half*2, w_half*2]
        # ------------------------------------------------------------------
        top_row: Tensor = torch.cat(
            [frames_half, frames_half], dim=3
        )  # [T, C, h_half, w_half*2]
        