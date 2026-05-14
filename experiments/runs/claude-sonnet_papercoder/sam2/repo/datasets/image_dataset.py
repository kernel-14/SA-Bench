## Code: datasets/image_dataset.py

```python
## datasets/image_dataset.py
"""Generic image dataset loader for SAM 2 zero-shot image evaluation.

This module implements ImageDataset and MultiDatasetImageLoader for loading
the 37 zero-shot image evaluation benchmarks used in Section 6.3 and
Appendix F.4 of the SAM 2 paper.

The 37 datasets break down as:
    - SA-23: 23 datasets from the original SAM evaluation suite
    - 14 new video datasets evaluated as images (frames sampled from videos)

All datasets are loaded in a unified format compatible with ImageEvaluator:
    {
        "image": Tensor[C, H, W],          # float32, ImageNet-normalized
        "masks": List[Tensor[H, W]],       # binary masks, one per instance
        "image_id": str,                   # unique identifier
        "dataset_name": str,               # which of the 37 datasets
        "original_size": Tuple[int, int],  # (H, W) before any resizing
    }

Config references (config.yaml):
    evaluation.image_segmentation.num_sa23_datasets: 23
    evaluation.image_segmentation.num_new_video_datasets: 14
    evaluation.image_segmentation.total_datasets: 37
    pretrain.data.mask_area_filter: 0.90

Paper references:
    Section 6.3: "We evaluate SAM 2 on the Segment Anything task across
        37 zero-shot datasets."
    Appendix F.4.1: "For the interactive segmentation task, we evaluated
        SAM 2 on a comprehensive suite of 37 datasets."
    Appendix F.4.1: "In addition to these 23 datasets, we evaluated on
        frames sampled from 14 video datasets."
"""

import glob
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

import pycocotools.mask as coco_mask_utils

from utils.mask_utils import MaskUtils

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset name constants (Appendix F.4.1)
# ---------------------------------------------------------------------------

# 23 datasets from the original SAM evaluation suite (SA-23)
SA23_DATASETS: List[str] = [
    "LVIS",
    "ADE20K",
    "Hypersim",
    "Cityscapes",
    "BBBC038v1",
    "DOORS",
    "DRAM",
    "EgoHOS",
    "GTEA",
    "iShape",
    "NDD20",
    "NDISPark",
    "OVIS",
    "PPDLS",
    "Plittersdorf",
    "STREETS",
    "TimberSeg",
    "TrashCan",
    "VISOR",
    "WoodScape",
    "PIDRay",
    "ZeroWaste-f",
    "IBD",
]

# 14 new video datasets evaluated as images (Appendix F.4.1)
NEW_VIDEO_DATASETS: List[str] = [
    "LCT",
    "VOST",
    "LV-VIS",
    "FBMS",
    "VirtualKITTI2",
    "CFD",
    "VIPSeg",
    "DH_OCM",
    "EndoVis2018",
    "ESD",
    "UVO",
    "EgoExo4d",
    "LVOSv2",
    "HT1080WT",
]

# All 37 datasets combined
ALL_37_DATASETS: List[str] = SA23_DATASETS + NEW_VIDEO_DATASETS

# Set for O(1) membership checks
_ALL_37_DATASETS_SET: Set[str] = set(ALL_37_DATASETS)
_SA23_SET: Set[str] = set(SA23_DATASETS)
_NEW_VIDEO_SET: Set[str] = set(NEW_VIDEO_DATASETS)

# ---------------------------------------------------------------------------
# Image normalization constants (ImageNet, standard for MAE pre-trained Hiera)
# ---------------------------------------------------------------------------

_IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Mask area filter threshold (config: pretrain.data.mask_area_filter: 0.90)
_MASK_AREA_FILTER: float = 0.90

# Minimum mask area in pixels (skip empty masks)
_MIN_MASK_AREA: int = 1


# ---------------------------------------------------------------------------
# Default image transform
# ---------------------------------------------------------------------------


class ImageEvalTransform:
    """Default image transform for evaluation: ToTensor + ImageNet normalize.

    No augmentation is applied during evaluation — only normalization and
    tensor conversion. Resizing to the model's input resolution is handled
    by the evaluator, not the dataset.

    Args:
        normalize: If True, apply ImageNet normalization. Defaults to True.

    Example:
        transform = ImageEvalTransform()
        img_pil = Image.open("image.jpg").convert("RGB")
        img_tensor = transform(img_pil)  # Tensor[3, H, W]
    """

    def __init__(self, normalize: bool = True) -> None:
        self.normalize: bool = normalize
        self._mean: Tensor = torch.tensor(
            _IMAGENET_MEAN, dtype=torch.float32
        ).view(3, 1, 1)
        self._std: Tensor = torch.tensor(
            _IMAGENET_STD, dtype=torch.float32
        ).view(3, 1, 1)

    def __call__(self, image: Image.Image) -> Tensor:
        """Convert PIL Image to normalized float32 tensor.

        Args:
            image: PIL Image in RGB format.

        Returns:
            Tensor of shape [3, H, W], dtype float32, ImageNet-normalized.
        """
        # Convert to float32 tensor [3, H, W] in [0, 1]
        img_np: np.ndarray = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        img_tensor: Tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # [3, H, W]

        if self.normalize:
            img_tensor = (img_tensor - self._mean) / self._std

        return img_tensor

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(normalize={self.normalize})"


# ---------------------------------------------------------------------------
# ImageDataset
# ---------------------------------------------------------------------------


class ImageDataset(Dataset):
    """Generic image dataset loader for SAM 2 zero-shot image evaluation.

    Loads images and instance masks from any of the 37 zero-shot evaluation
    benchmarks in a unified format compatible with ImageEvaluator. Uses a
    registry/factory pattern to dispatch to dataset-specific loading logic
    based on the dataset_name parameter.

    Each sample returned by __getitem__ contains:
        - "image": Tensor[C, H, W] float32, ImageNet-normalized
        - "masks": List[Tensor[H, W]] binary float32, one per instance
        - "image_id": str unique identifier
        - "dataset_name": str name of the source dataset
        - "original_size": Tuple[int, int] (H, W) before any resizing

    Mask filtering (config: pretrain.data.mask_area_filter: 0.90):
        - Skip empty masks (area == 0)
        - Skip masks covering > 90% of image area
        - Skip crowd annotations (LVIS iscrowd=1)

    Args:
        root: Path to the dataset root directory. Structure varies by dataset.
        dataset_name: One of the 37 supported dataset names. Case-sensitive.
            See ALL_37_DATASETS for the complete list.
        split: Dataset split. One of "train", "val", "test", "all".
            Defaults to "val" (standard evaluation split).
        transform: Optional callable that accepts a PIL Image and returns a
            Tensor[C, H, W]. If None, ImageEvalTransform() is used.
        mask_area_filter: Maximum allowed normalized mask area (mask_area /
            image_area). Masks with area > this threshold are discarded.
            Defaults to 0.90 (config: pretrain.data.mask_area_filter).
        max_samples: Optional cap on the number of samples to index. Useful
            for debugging and ablations. None means use all available samples.

    Example:
        dataset = ImageDataset(
            root="/data/LVIS",
            dataset_name="LVIS",
            split="val",
        )
        sample = dataset[0]
        # sample["image"]: Tensor[3, H, W]
        # sample["masks"]: List[Tensor[H, W]]
        # sample["image_id"]: "000000001234"
        # sample["dataset_name"]: "LVIS"
        # sample["original_size"]: (480, 640)
    """

    def __init__(
        self,
        root: str,
        dataset_name: str = "LVIS",
        split: str = "val",
        transform: Optional[Callable] = None,
        mask_area_filter: float = _MASK_AREA_FILTER,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Validate dataset_name
        if dataset_name not in _ALL_37_DATASETS_SET:
            logger.warning(
                "ImageDataset: dataset_name '%s' is not in the standard list "
                "of 37 datasets. Proceeding with generic PNG mask loader.",
                dataset_name,
            )

        self.root: str = root
        self.dataset_name: str = dataset_name
        self.split: str = split
        self.mask_area_filter: float = mask_area_filter
        self.max_samples: Optional[int] = max_samples

        # Use default transform if none provided
        if transform is None:
            self.transform: Callable = ImageEvalTransform(normalize=True)
        else:
            self.transform = transform

        # Shared mask utilities for RLE decoding and area computation
        self._mask_utils: MaskUtils = MaskUtils()

        # Build the sample index: List[Dict] with image_path, mask_source, image_id
        self.samples: List[Dict[str, Any]] = self._build_sample_list()

        logger.info(
            "ImageDataset initialized: dataset=%s, split=%s, "
            "num_samples=%d, mask_area_filter=%.2f",
            dataset_name,
            split,
            len(self.samples),
            mask_area_filter,
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of samples in the dataset.

        Returns:
            Integer count of indexed samples.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load and return a unified sample dict for one image.

        Args:
            idx: Integer index into self.samples.

        Returns:
            Dict with keys:
                - "image": Tensor[C, H, W] float32, ImageNet-normalized
                - "masks": List[Tensor[H, W]] binary float32 {0.0, 1.0}
                - "image_id": str unique identifier
                - "dataset_name": str
                - "original_size": Tuple[int, int] (H, W)

        Note:
            Returns a sample with empty masks list on errors rather than
            raising exceptions, to allow evaluation to continue.
        """
        sample_meta: Dict[str, Any] = self.samples[idx]
        image_path: str = sample_meta["image_path"]
        mask_source: Any = sample_meta["mask_source"]
        image_id: str = sample_meta["image_id"]

        # ------------------------------------------------------------------
        # Load image
        # ------------------------------------------------------------------
        image_pil: Optional[Image.Image] = self._load_image_pil(image_path)
        if image_pil is None:
            return self._make_empty_sample(image_id)

        original_h: int = image_pil.height
        original_w: int = image_pil.width
        original_size: Tuple[int, int] = (original_h, original_w)

        # Apply transform: PIL Image -> Tensor[C, H, W]
        try:
            image_tensor: Tensor = self.transform(image_pil)
        except Exception as exc:
            logger.warning(
                "ImageDataset: Transform failed for %s: %s. "
                "Returning empty sample.",
                image_path,
                exc,
            )
            return self._make_empty_sample(image_id)

        # ------------------------------------------------------------------
        # Load masks
        # ------------------------------------------------------------------
        raw_masks: List[np.ndarray] = self._load_masks(
            mask_source, original_h, original_w
        )

        # ------------------------------------------------------------------
        # Filter masks by area
        # ------------------------------------------------------------------
        filtered_masks: List[Tensor] = self._filter_masks(
            raw_masks, original_h, original_w
        )

        return {
            "image": image_tensor,           # Tensor[C, H, W]
            "masks": filtered_masks,         # List[Tensor[H, W]]
            "image_id": image_id,            # str
            "dataset_name": self.dataset_name,  # str
            "original_size": original_size,  # Tuple[int, int]
        }

    # ------------------------------------------------------------------
    # Sample list building — dispatcher
    # ------------------------------------------------------------------

    def _build_sample_list(self) -> List[Dict[str, Any]]:
        """Build the sample index by dispatching to dataset-specific loaders.

        Returns:
            List of sample dicts, each containing:
                - "image_path": str absolute path to the image file
                - "mask_source": Any dataset-specific mask source
                  (path string, annotation dict, list of annotation dicts, etc.)
                - "image_id": str unique identifier for this sample

        Returns empty list if the dataset root does not exist or loading fails.
        """
        if not os.path.exists(self.root):
            logger.warning(
                "ImageDataset: Dataset root does not exist: %s. "
                "Returning empty dataset for '%s'.",
                self.root,
                self.dataset_name,
            )
            return []

        # Dispatch to dataset-specific loader
        loader_map: Dict[str, Callable] = {
            # COCO-style JSON annotation datasets
            "LVIS": self._load_lvis_samples,
            "OVIS": self._load_coco_style_samples,
            "ADE20K": self._load_ade20k_samples,
            "Cityscapes": self._load_cityscapes_samples,
            "BBBC038v1": self._load_png_mask_samples,
            "DOORS": self._load_png_mask_samples,
            "DRAM": self._load_png_mask_samples,
            "EgoHOS": self._load_png_mask_samples,
            "GTEA": self._load_png_mask_samples,
            "iShape": self._load_png_mask_samples,
            "NDD20": self._load_png_mask_samples,
            "NDISPark": self._load_coco_style_samples,
            "PPDLS": self._load_png_mask_samples,
            "Plittersdorf": self._load_png_mask_samples,
            "STREETS": self._load_png_mask_samples,
            "TimberSeg": self._load_png_mask_samples,
            "TrashCan": self._load_coco_style_samples,
            "VISOR": self._load_coco_style_samples,
            "WoodScape": self._load_coco_style_samples,
            "PIDRay": self._load_coco_style_samples,
            "ZeroWaste-f": self._load_coco_style_samples,
            "IBD": self._load_png_mask_samples,
            "Hypersim": self._load_hypersim_samples,
            # Video datasets evaluated as images
            "LCT": self._load_video_as_images_samples,
            "VOST": self._load_vost_samples,
            "LV-VIS": self._load_coco_style_samples,
            "FBMS": self._load_video_as_images_samples,
            "VirtualKITTI2": self._load_virtual_kitti_samples,
            "CFD": self._load_video_as_images_samples,
            "VIPSeg": self._load_vipseg_samples,
            "DH_OCM": self._load_video_as_images_samples,
            "EndoVis2018": self._load_png_mask_samples,
            "ESD": self._load_png_mask_samples,
            "UVO": self._load_coco_style_samples,
            "EgoExo4d": self._load_video_as_images_samples,
            "LVOSv2": self._load_lvos_samples,
            "HT1080WT": self._load_video_as_images_samples,
        }

        loader_fn: Optional[Callable] = loader_map.get(self.dataset_name)

        if loader_fn is None:
            # Unknown dataset — fall back to generic PNG mask loader
            logger.warning(
                "ImageDataset: No specific loader for dataset '%s'. "
                "Using generic PNG mask loader.",
                self.dataset_name,
            )
            loader_fn = self._load_png_mask_samples

        try:
            samples: List[Dict[str, Any]] = loader_fn()
        except Exception as exc:
            logger.warning(
                "ImageDataset: Failed to build sample list for '%s': %s. "
                "Returning empty dataset.",
                self.dataset_name,
                exc,
            )
            return []

        # Apply optional cap for debugging
        if self.max_samples is not None and len(samples) > self.max_samples:
            samples = samples[: self.max_samples]
            logger.info(
                "ImageDataset (%s): Capped to %d samples.",
                self.dataset_name,
                self.max_samples,
            )

        return samples

    # ------------------------------------------------------------------
    # Dataset-specific sample list builders
    # ------------------------------------------------------------------

    def _load_lvis_samples(self) -> List[Dict[str, Any]]:
        """Build sample list for LVIS dataset.

        LVIS uses COCO-style JSON annotations with crowd filtering.
        Expected structure:
            root/
              images/{split}2017/*.jpg
              annotations/lvis_v1_{split}.json

        Returns:
            List of sample dicts with COCO annotation lists as mask_source.
        """
        # Map split names to LVIS annotation file names
        split_map: Dict[str, str] = {
            "train": "lvis_v1_train.json",
            "val": "lvis_v1_val.json",
            "test": "lvis_v1_val.json",  # LVIS test uses val for zero-shot eval
            "all": "lvis_v1_val.json",
        }
        ann_filename: str = split_map.get(self.split, "lvis_v1_val.json")

        # Try multiple common annotation paths
        ann_candidates: List[str] = [
            os.path.join(self.root, "annotations", ann_filename),
            os.path.join(self.root, ann_filename),
        ]
        ann_path: Optional[str] = None
        for candidate in ann_candidates:
            if os.path.isfile(candidate):
                ann_path = candidate
                break

        if ann_path is None:
            logger.warning(
                "LVIS: Annotation file not found. Tried: %s",
                ann_candidates,
            )
            return []

        return self._parse_coco_json(
            ann_path=ann_path,
            images_root=self.root,
            skip_crowd=True,
        )

    def _load_coco_style_samples(self) -> List[Dict[str, Any]]:
        """Build sample list for generic COCO-style datasets.

        Searches for a JSON annotation file in common locations.
        Expected structure:
            root/
              images/ or JPEGImages/ or *.jpg
              annotations/*.json or *.json

        Returns:
            List of sample dicts with COCO annotation lists as mask_source.
        """
        # Search for JSON annotation files
        json_candidates: List[str] = []

        # Common annotation file patterns
        for pattern in [
            os.path.join(self.root, "annotations", f"*{self.split}*.json"),
            os.path.join(self.root, "annotations", "*.json"),
            os.path.join(self.root, f"*{self.split}*.json"),
            os.path.join(self.root, "*.json"),
        ]:
            matches: List[str] = glob.glob(pattern)
            json_candidates.extend(matches)

        if not json_candidates:
            logger.warning(
                "COCO-style loader: No JSON annotation file found in %s "
                "for dataset '%s'. Falling back to PNG mask loader.",
                self.root,
                self.dataset_name,
            )
            return self._load_png_mask_samples()

        # Use the first matching annotation file
        ann_path: str = sorted(json_candidates)[0]
        logger.debug(
            "COCO-style loader (%s): Using annotation file %s",
            self.dataset_name,
            ann_path,
        )

        return self._parse_coco_json(
            ann_path=ann_path,
            images_root=self.root,
            skip_crowd=True,
        )

    def _load_ade20k_samples(self) -> List[Dict[str, Any]]:
        """Build sample list for ADE20K dataset.

        ADE20K uses per-image segmentation PNG files where each pixel value
        encodes the object class. For instance segmentation, we use the
        instance segmentation files (*_atr.png).

        Expected structure:
            root/
              images/ADE/validation/*.jpg
              annotations/ADE/validation/*.png

        Returns:
            List of sample dicts with annotation PNG path as mask_source.
        """
        split_dir: str = "validation" if self.split in ("val", "test") else "training"

        images_dir: str = os.path.join(self.root, "images", "ADE", split_dir)
        ann_dir: str = os.path.join(self.root, "annotations", "ADE", split_dir)

        # Fall back to simpler structure
        if not os.path.isdir(images_dir):
            images_dir = os.path.join(self.root, split_dir)
            ann_dir = os.path.join(self.root, "annotations", split_dir)

        if not os.path.isdir(images_dir):
            logger.warning(
                "ADE20K: Images directory not found: %s", images_dir
            )
            return []

        samples: List[Dict[str, Any]] = []

        # Find all image files
        image_paths: List[str] = sorted(
            glob.glob(os.path.join(images_dir, "**", "*.jpg"), recursive=True)
            + glob.glob(os.path.join(images_dir, "**", "*.png"), recursive=True)
        )

        for img_path in image_paths:
            img_stem: str = Path(img_path).stem
            # ADE20K annotation: same name with .png extension
            ann_path: str = os.path.join(ann_dir, img_stem + ".png")

            if not os.path.isfile(ann_path):
                # Try recursive search
                ann_matches: List[str] = glob.glob(
                    os.path.join(ann_dir, "**", img_stem + ".png"),
                    recursive=True,
                )
                if ann_matches:
                    ann_path = ann_matches[0]
                else:
                    continue

            samples.append({
                "image_path": img_path,
                "mask_source": {"type": "ade20k_png", "path": ann_path},
                "image_id": img_stem,
            })

        logger.debug(
            "ADE20K: Found %d samples in %s split.", len(samples), self.split
        )
        return samples

    def _load_cityscapes_samples(self) -> List[Dict[str, Any]]:
        """Build sample list for Cityscapes dataset.

        Cityscapes uses JSON instance annotation files.
        Expected structure:
            root/
              leftImg8bit/{split}/*/*.png
              gtFine/{split}/*/*_gtFine_instanceIds.png

        Returns:
            List of sample dicts with instance PNG path as mask_source.
        """
        split_dir: str = self.split if self.split in ("train", "val", "test") else "val"

        images_dir: str = os.path.join(self.root, "leftImg8bit", split_dir)
        ann_dir: str = os.path.join(self.root, "gtFine", split_dir)

        if not os.path.isdir(images_dir):
            logger.warning(
                "Cityscapes: Images directory not found: %s", images_dir
            )
            return []

        samples: List[Dict[str, Any]] = []

        # Find all image files
        image_paths: List[str] = sorted(
            glob.glob(os.path.join(images_dir, "**", "*_leftImg8bit.png"), recursive=True)
        )

        for img_path in image_paths:
            # Derive instance annotation path
            # e.g., aachen_000000_000019_leftImg8bit.png
            #     -> aachen_000000_000019_gtFine_instanceIds.png
            img_name: str = Path(img_path).name
            city: str = Path(img_path).parent.name
            base_name: str = img_name.replace("_leftImg8bit.png", "")
            ann_name: str = base_name + "_gtFine_instanceIds.png"
            ann_path: str = os.path.join(ann_dir, city, ann_name)

            if not os.path.isfile(ann_path):
                continue

            samples.append({
                "image_path": img_path,
                "mask_source": {"type": "cityscapes_instance", "path": ann_path},
                "image_id": base_name,
            })

        logger.debug(
            "Cityscapes: Found %d samples in %s split.", len(samples), self.split
        )
        return samples

    def _load_hypersim_samples(self) -> List[Dict[str, Any]]:
        """Build sample list for Hypersim dataset.

        Hypersim is a photorealistic synthetic indoor scene dataset.
        Expected structure:
            root/
              ai_001_001/images/scene_cam_00_final_preview/*.tonemap.jpg
              ai_001_001/images/scene_cam_00_geometry_hdf5/*.semantic_instance.hdf5

        Returns:
            List of sample dicts with HDF5 path as mask_source.
        """
        samples: List[Dict[str, Any]] = []

        # Find all tonemap images
        image_paths: List[str] = sorted(
            glob.glob(
                os.path.join(self.root, "**", "*.tonemap.jpg"),
                recursive=True,
            )
        )

        for img_path in image_paths:
            # Derive HDF5 instance annotation path
            # Replace final_preview with geometry_hdf5 and .tonemap.jpg with .semantic_instance.hdf5
            img_dir: str = str(Path(img_path).parent)
            hdf5_dir: str = img_dir.replace(
                "scene_cam_00_final_preview", "scene_cam_00_geometry_hdf5"
            )
            img_stem: str = Path(img_path).stem.replace(".tonemap", "")
            hdf5_path: str = os.path.join(
                hdf5_dir, img_stem + ".semantic_instance.hdf5"
            )

            if not os.path.isfile(hdf5_path):
                continue

            image_id: str = Path(img_path).stem.replace(".tonemap", "")

            samples.append({
                "image_path": img_path,
                "mask_source": {"type": "hdf5_instance", "path": hdf5_path},
                "image_id": image_id,
            })

        logger.debug(
            "Hypersim: Found %d samples.", len(samples)
        )
        return samples

    def _load_png_mask_samples(self) -> List[Dict[str, Any]]:
        """Build sample list for datasets with PNG instance mask files.

        Generic loader for datasets where each image has a corresponding
        PNG file encoding instance IDs as pixel values. Searches for
        common directory structures.

        Expected structures (tried in order):
            root/images/*.jpg + root/annotations/*.png
            root/JPEGImages/*.jpg + root/Annotations/*.png
            root/images/*.jpg + root/masks/*.png
            root/*.jpg + root/*.png (flat structure)

        Returns:
            List of sample dicts with annotation PNG path as mask_source.
        """
        # Common directory structure patterns to try
        structure_candidates: List[Tuple[str, str]] = [
            (
                os.path.join(self.root, "images"),
                os.path.join(self.root, "annotations"),
            ),
            (
                os.path.join(self.root, "JPEGImages"),
                os.path.join(self.root, "Annotations"),
            ),
            (
                os.path.join(self.root, "images"),
                os.path.join(self.root, "masks"),
            ),
            (
                os.path.join(self.root, "imgs"),
                os.path.join(self.root, "masks"),
            ),
            (self.root, self.root),  # flat structure
        ]

        # Add split-specific subdirectories
        for split_name in [self.split, f"{self.split}2017", f"{self.split}2019"]:
            structure_candidates.extend([
                (
                    os.path.join(self.root, "images", split_name),
                    os.path.join(self.root, "annotations", split_name),
                ),
                (
                    os.path.join(self.root, split_name, "images"),
                    os.path.join(self.root, split_name, "annotations"),
                ),
            ])

        images_dir: Optional[str] = None
        ann_dir: Optional[str] = None

        for img_candidate, ann_candidate in structure_candidates:
            if os.path.isdir(img_candidate) and os.path.isdir(ann_candidate):
                # Check if there are actually image files
                test_images: List[str] = (
                    glob.glob(os.path.join(img_candidate, "*.jpg"))
                    + glob.glob(os.path.join(img_candidate, "*.png"))
                    + glob.glob(os.path.join(img_candidate, "*.jpeg"))
                )
                if test_images:
                    images_dir = img_candidate
                    ann_dir = ann_candidate
                    break

        if images_dir is None:
            logger.warning(
                "PNG mask loader (%s): Could not find images directory in %s.",
                self.dataset_name,
                self.root,
            )
            return []

        # Find