```python
## datasets/vos_dataset.py
"""Unified VOS dataset loader for DAVIS 2017, MOSE, and YouTubeVOS.

This module implements VOSDataset, a PyTorch Dataset that provides a unified
interface for loading three standard Video Object Segmentation datasets used
in SAM 2's full training pipeline (Stage 2 and Stage 3).

Supported datasets:
    - DAVIS 2017: 60 train / 30 val sequences, dense annotations, palette PNG
    - MOSE: 2.1K videos, complex occlusions, palette PNG annotations
    - YouTubeVOS: 4.5K videos, sparse annotations (every 5th frame), meta.json

Data mixture (config.yaml training.data_mixture_with_oss):
    - DAVIS:       ~1.3% of training batches
    - MOSE:        ~9.4% of training batches
    - YouTubeVOS:  ~9.2% of training batches

Config references (config.yaml):
    model.input_resolution: 1024
    training.num_frames: 8
    finetuning.num_frames: 16
    training.use_auto_masklets_for_training: false
    training.data_mixture_with_oss.*

Paper references:
    Appendix D.2.2: "a mixture of open-source video datasets including DAVIS,
        MOSE, and YouTubeVOS"
    Appendix D.2.2: "we sample sequences of 8 frames"
    Appendix D.2.2: "We restrict the maximum number of masklets for each
        sequence of 8 frames to 3 randomly chosen ones."
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants from config.yaml
# ---------------------------------------------------------------------------

# Default input resolution (config: model.input_resolution)
_DEFAULT_RESOLUTION: int = 1024

# Default number of frames per training sequence (config: training.num_frames)
_DEFAULT_NUM_FRAMES: int = 8

# ImageNet normalization constants (standard for MAE pre-trained Hiera)
_IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Supported dataset names
_SUPPORTED_DATASETS: Set[str] = {"davis", "mose", "youtubevos"}

# DAVIS split file mapping
_DAVIS_SPLIT_FILES: Dict[str, str] = {
    "train": "train.txt",
    "val": "val.txt",
    "test": "test-dev.txt",
}

# MOSE split directory mapping
_MOSE_SPLIT_DIRS: Dict[str, str] = {
    "train": "train",
    "val": "valid",
    "test": "valid",
}

# YouTubeVOS split directory mapping
_YTBVOS_SPLIT_DIRS: Dict[str, str] = {
    "train": "train",
    "val": "valid",
    "test": "valid",
}


# ---------------------------------------------------------------------------
# Lazy import helper to avoid circular imports
# ---------------------------------------------------------------------------


def _get_video_sample_class() -> Any:
    """Lazily import VideoSample to avoid circular import at module load time.

    Returns:
        The VideoSample dataclass from datasets/__init__.py.
    """
    from datasets import VideoSample as _VideoSample
    return _VideoSample


# ---------------------------------------------------------------------------
# VOSDataset
# ---------------------------------------------------------------------------


class VOSDataset(torch.utils.data.Dataset):
    """Unified PyTorch Dataset for DAVIS 2017, MOSE, and YouTubeVOS.

    Provides a consistent VideoSample interface across all three VOS datasets,
    handling their different directory structures, annotation formats, and
    frame availability patterns.

    Indexing is at the sequence level: __len__ returns the number of video
    sequences in the split. Each __getitem__ call returns one VideoSample
    containing T frames and masks for all objects in that sequence.

    The Trainer handles masklet subsampling (up to 3 per sequence) internally.

    Args:
        root: Path to the dataset root directory. Expected structure varies
            by dataset_name (see module docstring for details).
        dataset_name: One of "davis", "mose", "youtubevos". Case-insensitive.
        split: Dataset split. One of "train", "val", "test".
        num_frames: Number of frames to sample per sequence. Use 8 for
            standard training (config: training.num_frames) or 16 for
            fine-tuning (config: finetuning.num_frames). Defaults to 8.
        transform: Optional callable that accepts a VideoSample and returns
            a transformed VideoSample. Applied after frame/mask loading.
            If None, only basic resizing to input_resolution is applied.
        input_resolution: Target spatial resolution for frames and masks.
            Defaults to 1024 (config: model.input_resolution).
        exclude_sequences: Optional list of sequence names to exclude from
            the dataset. Used to create the MOSE dev set (200 sequences
            excluded from training for ablation evaluation).

    Example:
        dataset = VOSDataset(
            root="/data/DAVIS",
            dataset_name="davis",
            split="train",
            num_frames=8,
        )
        sample = dataset[0]
        # sample.frames: Tensor[8, 3, 1024, 1024]
        # sample.masks: Tensor[8, N, 1024, 1024]
        # sample.num_objects: N (number of objects in this sequence)
    """

    def __init__(
        self,
        root: str,
        dataset_name: str = "davis",
        split: str = "train",
        num_frames: int = _DEFAULT_NUM_FRAMES,
        transform: Optional[Callable] = None,
        input_resolution: int = _DEFAULT_RESOLUTION,
        exclude_sequences: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        # Normalize and validate dataset_name
        dataset_name_lower: str = dataset_name.lower()
        if dataset_name_lower not in _SUPPORTED_DATASETS:
            raise ValueError(
                f"dataset_name must be one of {sorted(_SUPPORTED_DATASETS)}, "
                f"got '{dataset_name}'."
            )

        # Validate split
        valid_splits: List[str] = ["train", "val", "test"]
        if split not in valid_splits:
            raise ValueError(
                f"split must be one of {valid_splits}, got '{split}'."
            )

        self.root: str = root
        self.dataset_name: str = dataset_name_lower
        self.split: str = split
        self.num_frames: int = num_frames
        self.transform: Optional[Callable] = transform
        self.input_resolution: int = input_resolution
        self.exclude_sequences: Set[str] = set(exclude_sequences or [])

        # Precompute normalization tensors for frame loading
        self._mean: Tensor = torch.tensor(
            _IMAGENET_MEAN, dtype=torch.float32
        ).view(3, 1, 1)
        self._std: Tensor = torch.tensor(
            _IMAGENET_STD, dtype=torch.float32
        ).view(3, 1, 1)

        # YouTubeVOS meta.json cache: video_id -> {objects: {obj_id: {frames: [...]}}}
        self._ytbvos_meta: Dict[str, Any] = {}

        # Build the sequence list for this split
        self.sequences: List[str] = self._build_sequence_list()

        logger.info(
            "VOSDataset initialized: dataset=%s, split=%s, "
            "num_sequences=%d, num_frames=%d, resolution=%d",
            self.dataset_name,
            self.split,
            len(self.sequences),
            self.num_frames,
            self.input_resolution,
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of sequences in the split.

        Returns:
            Integer count of video sequences.
        """
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Any:
        """Load and return a VideoSample for one video sequence.

        Full processing pipeline:
            1. Look up sequence name from self.sequences[idx]
            2. Get list of annotated frame indices for this sequence
            3. Sample num_frames frame indices
            4. Load frames and masks via _load_sequence()
            5. Apply transform if provided

        Args:
            idx: Integer index into self.sequences.

        Returns:
            VideoSample with:
                - frames: Tensor[T, 3, H, W] float32, ImageNet-normalized
                - masks: Tensor[T, N, H, W] float32 binary {0, 1}
                - video_id: str (sequence name)
                - frame_indices: List[int] of length T
                - num_objects: int (N, number of distinct objects)
                - is_occluded: List[bool] of length T*N

        Note:
            Returns a placeholder VideoSample on errors to allow DataLoader
            to continue without crashing.
        """
        VideoSample = _get_video_sample_class()
        seq_name: str = self.sequences[idx]

        try:
            # Get annotated frame indices for this sequence
            annotated_frames: List[int] = self._get_annotated_frames(seq_name)

            if len(annotated_frames) == 0:
                logger.warning(
                    "VOSDataset: No annotated frames found for sequence %s. "
                    "Returning placeholder.",
                    seq_name,
                )
                return self._make_placeholder_sample(seq_name)

            # Sample frame indices
            frame_indices: List[int] = self._sample_frame_indices(
                annotated_frames, self.num_frames
            )

            # Load frames and masks
            sample: Any = self._load_sequence(seq_name, frame_indices)

            # Apply transform if provided
            if self.transform is not None:
                try:
                    sample = self.transform(sample)
                except Exception as exc:
                    logger.warning(
                        "VOSDataset: Transform failed for sequence %s: %s. "
                        "Applying fallback resize only.",
                        seq_name,
                        exc,
                    )
                    sample = self._apply_fallback_resize(sample)

            return sample

        except Exception as exc:
            logger.warning(
                "VOSDataset: Failed to load sequence %s (idx=%d): %s. "
                "Returning placeholder sample.",
                seq_name,
                idx,
                exc,
            )
            return self._make_placeholder_sample(seq_name)

    def _load_sequence(
        self,
        seq_name: str,
        frame_indices: List[int],
    ) -> Any:
        """Load frames and masks for a sequence and return a VideoSample.

        Dispatches to dataset-specific loading logic based on self.dataset_name.

        Args:
            seq_name: Sequence/video identifier string.
            frame_indices: List of T integer frame indices to load.

        Returns:
            VideoSample with frames [T, 3, H, W], masks [T, N, H, W],
            and associated metadata.

        Raises:
            RuntimeError: If frames or masks cannot be loaded.
        """
        if self.dataset_name == "davis":
            return self._load_davis_sequence(seq_name, frame_indices)
        elif self.dataset_name == "mose":
            return self._load_mose_sequence(seq_name, frame_indices)
        elif self.dataset_name == "youtubevos":
            return self._load_ytbvos_sequence(seq_name, frame_indices)
        else:
            raise ValueError(
                f"Unknown dataset_name: {self.dataset_name}. "
                f"Must be one of {sorted(_SUPPORTED_DATASETS)}."
            )

    # ------------------------------------------------------------------
    # Sequence list building
    # ------------------------------------------------------------------

    def _build_sequence_list(self) -> List[str]:
        """Build the list of sequence names for the current split.

        Dispatches to dataset-specific logic and applies exclusion filtering.

        Returns:
            Sorted list of sequence name strings, with excluded sequences
            removed.

        Raises:
            FileNotFoundError: If the split file or directory does not exist.
        """
        if self.dataset_name == "davis":
            sequences = self._build_davis_sequence_list()
        elif self.dataset_name == "mose":
            sequences = self._build_mose_sequence_list()
        elif self.dataset_name == "youtubevos":
            sequences = self._build_ytbvos_sequence_list()
        else:
            sequences = []

        # Apply exclusion filter (e.g., MOSE dev set)
        if self.exclude_sequences:
            original_count: int = len(sequences)
            sequences = [s for s in sequences if s not in self.exclude_sequences]
            excluded_count: int = original_count - len(sequences)
            if excluded_count > 0:
                logger.info(
                    "VOSDataset (%s): Excluded %d sequences from %s split.",
                    self.dataset_name,
                    excluded_count,
                    self.split,
                )

        return sorted(sequences)

    def _build_davis_sequence_list(self) -> List[str]:
        """Build sequence list for DAVIS 2017.

        Reads the split text file from ImageSets/2017/{split}.txt.
        Each line contains one sequence name.

        Returns:
            List of sequence name strings.

        Raises:
            FileNotFoundError: If the split file does not exist.
        """
        split_filename: str = _DAVIS_SPLIT_FILES.get(self.split, "train.txt")
        split_file: str = os.path.join(
            self.root, "ImageSets", "2017", split_filename
        )

        if not os.path.isfile(split_file):
            raise FileNotFoundError(
                f"DAVIS split file not found: {split_file}. "
                f"Expected structure: {self.root}/ImageSets/2017/{split_filename}"
            )

        sequences: List[str] = []
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                seq_name: str = line.strip()
                if seq_name:
                    sequences.append(seq_name)

        logger.debug(
            "DAVIS: Found %d sequences in %s split.",
            len(sequences),
            self.split,
        )
        return sequences

    def _build_mose_sequence_list(self) -> List[str]:
        """Build sequence list for MOSE.

        Scans the JPEGImages subdirectory for the current split.

        Returns:
            List of sequence name strings (subdirectory names).

        Raises:
            FileNotFoundError: If the split directory does not exist.
        """
        split_dir_name: str = _MOSE_SPLIT_DIRS.get(self.split, "train")
        images_dir: str = os.path.join(self.root, split_dir_name, "JPEGImages")

        if not os.path.isdir(images_dir):
            raise FileNotFoundError(
                f"MOSE JPEGImages directory not found: {images_dir}. "
                f"Expected structure: {self.root}/{split_dir_name}/JPEGImages/"
            )

        sequences: List[str] = [
            d for d in os.listdir(images_dir)
            if os.path.isdir(os.path.join(images_dir, d))
        ]

        logger.debug(
            "MOSE: Found %d sequences in %s split.",
            len(sequences),
            self.split,
        )
        return sequences

    def _build_ytbvos_sequence_list(self) -> List[str]:
        """Build sequence list for YouTubeVOS.

        Parses the meta.json file to get video IDs and caches the metadata
        for later use in frame/mask loading.

        Returns:
            List of video ID strings.

        Raises:
            FileNotFoundError: If meta.json does not exist.
            ValueError: If meta.json is malformed.
        """
        split_dir_name: str = _YTBVOS_SPLIT_DIRS.get(self.split, "train")
        meta_path: str = os.path.join(self.root, split_dir_name, "meta.json")

        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"YouTubeVOS meta.json not found: {meta_path}. "
                f"Expected structure: {self.root}/{split_dir_name}/meta.json"
            )

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_data: Dict = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(
                f"Failed to parse YouTubeVOS meta.json at {meta_path}: {exc}"
            ) from exc

        videos_meta: Dict = meta_data.get("videos", {})
        if not videos_meta:
            raise ValueError(
                f"YouTubeVOS meta.json at {meta_path} has no 'videos' key "
                "or it is empty."
            )

        # Cache the full metadata for use in _get_annotated_frames and _load_sequence
        self._ytbvos_meta = videos_meta

        video_ids: List[str] = list(videos_meta.keys())

        logger.debug(
            "YouTubeVOS: Found %d videos in %s split.",
            len(video_ids),
            self.split,
        )
        return video_ids

    # ------------------------------------------------------------------
    # Annotated frame discovery
    # ------------------------------------------------------------------

    def _get_annotated_frames(self, seq_name: str) -> List[int]:
        """Get the list of frame indices that have annotations for a sequence.

        For DAVIS and MOSE: all frames are annotated (dense annotations).
        For YouTubeVOS: only every 5th frame is annotated in training split.

        Args:
            seq_name: Sequence/video identifier string.

        Returns:
            Sorted list of integer frame indices with available annotations.
            Returns empty list if no annotations are found.
        """
        if self.dataset_name == "davis":
            return self._get_davis_annotated_frames(seq_name)
        elif self.dataset_name == "mose":
            return self._get_mose_annotated_frames(seq_name)
        elif self.dataset_name == "youtubevos":
            return self._get_ytbvos_annotated_frames(seq_name)
        return []

    def _get_davis_annotated_frames(self, seq_name: str) -> List[int]:
        """Get annotated frame indices for a DAVIS sequence.

        Lists all .png files in the Annotations/480p/{seq_name}/ directory
        and extracts frame numbers from filenames.

        Args:
            seq_name: DAVIS sequence name.

        Returns:
            Sorted list of integer frame indices.
        """
        ann_dir: str = os.path.join(
            self.root, "Annotations", "480p", seq_name
        )

        if not os.path.isdir(ann_dir):
            logger.warning(
                "DAVIS: Annotation directory not found: %s", ann_dir
            )
            return []

        frame_indices: List[int] = []
        for fname in os.listdir(ann_dir):
            if fname.endswith(".png"):
                try:
                    frame_idx: int = int(os.path.splitext(fname)[0])
                    frame_indices.append(frame_idx)
                except ValueError:
                    continue

        return sorted(frame_indices)

    def _get_mose_annotated_frames(self, seq_name: str) -> List[int]:
        """Get annotated frame indices for a MOSE sequence.

        Lists all .png files in the {split}/Annotations/{seq_name}/ directory.

        Args:
            seq_name: MOSE sequence name.

        Returns:
            Sorted list of integer frame indices.
        """
        split_dir_name: str = _MOSE_SPLIT_DIRS.get(self.split, "train")
        ann_dir: str = os.path.join(
            self.root, split_dir_name, "Annotations", seq_name
        )

        if not os.path.isdir(ann_dir):
            logger.warning(
                "MOSE: Annotation directory not found: %s", ann_dir
            )
            return []

        frame_indices: List[int] = []
        for fname in os.listdir(ann_dir):
            if fname.endswith(".png"):
                try:
                    frame_idx = int(os.path.splitext(fname)[0])
                    frame_indices.append(frame_idx)
                except ValueError:
                    continue

        return sorted(frame_indices)

    def _get_ytbvos_annotated_frames(self, video_id: str) -> List[int]:
        """Get annotated frame indices for a YouTubeVOS video.

        Uses the cached meta.json data to determine which frames have
        annotations. Returns the union of annotated frames across all objects.

        Args:
            video_id: YouTubeVOS video ID string.

        Returns:
            Sorted list of integer frame indices with annotations.
        """
        if not self._ytbvos_meta:
            # Meta not loaded yet — try to load it now
            self._build_ytbvos_sequence_list()

        video_meta: Dict = self._ytbvos_meta.get(video_id, {})
        objects_meta: Dict = video_meta.get("objects", {})

        if not objects_meta:
            # Fall back to scanning the annotation directory
            return self._scan_ytbvos_annotation_dir(video_id)

        # Collect all annotated frame IDs across all objects
        all_frame_ids: Set[str] = set()
        for obj_meta in objects_meta.values():
            frames_list: List[str] = obj_meta.get("frames", [])
            all_frame_ids.update(frames_list)

        # Convert frame ID strings to integers
        # YouTubeVOS uses zero-padded 5-digit strings: "00000", "00005", ...
        frame_indices: List[int] = []
        for frame_id in all_frame_ids:
            try:
                frame_indices.append(int(frame_id))
            except ValueError:
                continue

        return sorted(frame_indices)

    def _scan_ytbvos_annotation_dir(self, video_id: str) -> List[int]:
        """Fallback: scan YouTubeVOS annotation directory for frame indices.

        Args:
            video_id: YouTubeVOS video ID string.

        Returns:
            Sorted list of integer frame indices found in the annotation dir.
        """
        split_dir_name: str = _YTBVOS_SPLIT_DIRS.get(self.split, "train")
        ann_dir: str = os.path.join(
            self.root, split_dir_name, "Annotations", video_id
        )

        if not os.path.isdir(ann_dir):
            return []

        frame_indices: List[int] = []
        for fname in os.listdir(ann_dir):
            if fname.endswith(".png"):
                try:
                    frame_indices.append(int(os.path.splitext(fname)[0]))
                except ValueError:
                    continue

        return sorted(frame_indices)

    # ------------------------------------------------------------------
    # Frame index sampling
    # ------------------------------------------------------------------

    def _sample_frame_indices(
        self,
        annotated_frames: List[int],
        num_frames: int,
    ) -> List[int]:
        """Sample num_frames frame indices from the available annotated frames.

        Sampling strategy:
            - Training split: random contiguous window of num_frames frames
              from the annotated_frames list.
            - Val/test split: deterministic — use the first num_frames frames.
            - If fewer annotated frames than num_frames: use all available
              frames and pad by repeating the last frame.

        Args:
            annotated_frames: Sorted list of integer frame indices with
                available annotations.
            num_frames: Target number of frames to sample.

        Returns:
            List of num_frames integer frame indices. May contain duplicates
            if the video is shorter than num_frames (padding by repetition).
        """
        n_available: int = len(annotated_frames)

        if n_available == 0:
            return [0] * num_frames

        if n_available <= num_frames:
            # Use all available frames and pad with the last frame
            indices: List[int] = list(annotated_frames)
            last_frame: int = annotated_frames[-1]
            while len(indices) < num_frames:
                indices.append(last_frame)
            return indices

        # n_available > num_frames
        if self.split != "train":
            # Deterministic: use first num_frames annotated frames
            return list(annotated_frames[:num_frames])

        # Training: random contiguous window
        # Pick a random start position in [0, n_available - num_frames]
        max_start: int = n_available - num_frames
        start_pos: int = random.randint(0, max_start)
        return list(annotated_frames[start_pos: start_pos + num_frames])

    # ------------------------------------------------------------------
    # Dataset-specific sequence loading
    # ------------------------------------------------------------------

    def _load_davis_sequence(
        self,
        seq_name: str,
        frame_indices: List[int],
    ) -> Any:
        """Load frames and masks for a DAVIS 2017 sequence.

        DAVIS directory structure:
            JPEGImages/480p/{seq_name}/{frame_idx:05d}.jpg
            Annotations/480p/{seq_name}/{frame_idx:05d}.png

        Args:
            seq_name: DAVIS sequence name.
            frame_indices: List of T integer frame indices to load.

        Returns:
            VideoSample with frames [T, 3, H, W] and masks [T, N, H, W].

        Raises:
            RuntimeError: If frames or annotations cannot be loaded.
        """
        images_dir: str = os.path.join(self.root, "JPEGImages", "480p", seq_name)
        ann_dir: str = os.path.join(self.root, "Annotations", "480p", seq_name)

        if not os.path.isdir(images_dir):
            raise RuntimeError(
                f"DAVIS images directory not found: {images_dir}"
            )

        # Discover all available frame files for clamping
        available_frames: List[int] = self._get_davis_annotated_frames(seq_name)
        max_frame: int = max(available_frames) if available_frames else 0

        # Load frames
        frames_list: List[Tensor] = []
        for frame_idx in frame_indices:
            clamped_idx: int = min(frame_idx, max_frame)
            img_path: str = os.path.join(
                images_dir, f"{clamped_idx:05d}.jpg"
            )
            frame_tensor: Tensor = self._load_frame_from_path(img_path)
            frames_list.append(frame_tensor)

        frames: Tensor = torch.stack(frames_list, dim=0)  # [T, 3, H_orig, W_orig]

        # Load annotation PNGs and extract object masks
        ann_maps: List[np.ndarray] = []
        for frame_idx in frame_indices:
            clamped_idx = min(frame_idx, max_frame)
            ann_path: str = os.path.join(
                ann_dir, f"{clamped_idx:05d}.png"
            )
            ann_map: np.ndarray = self._load_annotation_png(ann_path)
            ann_maps.append(ann_map)

        # Build multi-object masks
        masks, is_occluded, num_objects = self._build_masks_from_annotations(
            ann_maps, frame_indices
        )

        # Resize frames and masks to input_resolution
        frames, masks = self._resize_frames_and_masks(frames, masks)

        VideoSample = _get_video_sample_class()
        return VideoSample(
            frames=frames,
            masks=masks,
            video_id=seq_name,
            frame_indices=frame_indices,
            num_objects=num_objects,
            is_occluded=is_occluded,
        )

    def _load_mose_sequence(
        self,
        seq_name: str,
        frame_indices: List[int],
    ) -> Any:
        """Load frames and masks for a MOSE sequence.

        MOSE directory structure:
            {split}/JPEGImages/{seq_name}/{frame_idx:05d}.jpg
            {split}/Annotations/{seq_name}/{frame_idx:05d}.png

        Args:
            seq_name: MOSE sequence name.
            frame_indices: List of T integer frame indices to load.

        Returns:
            VideoSample with frames [T, 3, H, W] and masks [T, N, H, W].

        Raises:
            RuntimeError: If frames or annotations cannot be loaded.
        """
        split_dir_name: str = _MOSE_SPLIT_DIRS.get(self.split, "train")
        images_dir: str = os.path.join(
            self.root, split_dir_name, "JPEGImages", seq_name
        )
        ann_dir: str = os.path.join(
            self.root, split_dir_name, "Annotations", seq_name
        )

        if not os.path.isdir(images_dir):
            raise RuntimeError(
                f"MOSE images directory not found: {images_dir}"
            )

        # Discover available frames for clamping
        available_frames: List[int] = self._get_mose_annotated_frames(seq_name)
        max_frame: int = max(available_frames) if available_frames else 0

        # Load frames
        frames_list: List[Tensor] = []
        for frame_idx in frame_indices:
            clamped_idx: int = min(frame_idx, max_frame)
            img_path: str = os.path.join(
                images_dir, f"{clamped_idx:05d}.jpg"
            )
            frame_tensor: Tensor = self._load_frame_from_path(img_path)
            frames_list.append(frame_tensor)

        frames: Tensor = torch.stack(frames_list, dim=0)  # [T, 3, H_orig, W_orig]

        # Load annotation PNGs
        ann_maps: List[np.ndarray] = []
        for frame_idx in frame_indices:
            clamped_idx = min(frame_idx, max_frame)
            ann_path: str = os.path.join(
                ann_dir, f"{clamped_idx:05d}.png"
            )
            ann_map: np.ndarray = self._load_annotation_png(ann_path)
            ann_maps.append(ann_map)

        # Build multi-object masks
        masks, is_occluded, num_objects = self._build_masks_from_annotations(
            ann_maps, frame_indices
        )

        # Resize to input_resolution
        frames, masks = self._resize_frames_and_masks(frames, masks)

        VideoSample = _get_video_sample_class()
        return VideoSample(
            frames=frames,
            masks=masks,
            video_id=seq_name,
            frame_indices