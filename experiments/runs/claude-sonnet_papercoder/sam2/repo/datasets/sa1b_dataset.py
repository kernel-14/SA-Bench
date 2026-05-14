```python
## datasets/sa1b_dataset.py
"""SA-1B dataset loader for SAM 2 pre-training (Stage 1).

This module implements SA1BDataset, a PyTorch Dataset that loads SA-1B images
and their corresponding COCO-RLE JSON mask annotations for use during SAM 2
pre-training on static images.

Data loading pipeline:
    1. Index all .jpg image paths at init time (lazy annotation loading)
    2. Load and parse JSON annotations per __getitem__ call
    3. Filter masks covering > 90% of image area (config: mask_area_filter: 0.90)
    4. Randomly sample up to 64 masks per image (config: max_masks_per_image: 64)
    5. Decode RLE masks for sampled annotations only (efficient path)
    6. Apply spatial transforms consistently to image and all masks
    7. Return dict with image tensor, stacked mask tensors, and metadata

Config references (config.yaml pretrain section):
    pretrain.data.max_masks_per_image: 64
    pretrain.data.mask_area_filter: 0.90
    pretrain.resolution: 1024
    pretrain.augmentation.horizontal_flip: true
    pretrain.augmentation.resize_to_square: 1024

Paper references:
    Appendix D.2.1: "we filter masks covering more than 90% of the image and
        restricted training to 64 randomly sampled masks per image."
    Appendix D.2.1: "we employ horizontal flip augmentation during training
        and resize the image to a square size of 1024×1024."
"""

import glob
import json
import logging
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

import pycocotools.mask as coco_mask_utils

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SA-1B dataset constants (from config.yaml pretrain section)
# ---------------------------------------------------------------------------

# Default maximum masks per image (config: pretrain.data.max_masks_per_image)
_DEFAULT_MAX_MASKS: int = 64

# Default area filter threshold (config: pretrain.data.mask_area_filter)
_DEFAULT_AREA_FILTER: float = 0.90

# Default input resolution (config: pretrain.resolution)
_DEFAULT_RESOLUTION: int = 1024

# ImageNet normalization constants (standard for MAE pre-trained Hiera)
_IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Default transform pipeline
# ---------------------------------------------------------------------------


class SA1BDefaultTransform:
    """Default image and mask transform pipeline for SA-1B pre-training.

    Applies the augmentations described in Appendix D.2.1:
        - Resize image to square (1024×1024) using bilinear interpolation
        - Resize masks to square (1024×1024) using nearest-neighbor
        - Random horizontal flip (applied consistently to image and all masks)
        - Normalize image with ImageNet mean/std

    This transform is used when no custom transform is provided to SA1BDataset.

    Config references:
        pretrain.resolution: 1024
        pretrain.augmentation.horizontal_flip: true
        pretrain.augmentation.resize_to_square: 1024

    Args:
        resolution: Target square resolution. Defaults to 1024.
        horizontal_flip_prob: Probability of horizontal flip. Defaults to 0.5.
        normalize: If True, apply ImageNet normalization. Defaults to True.

    Example:
        transform = SA1BDefaultTransform(resolution=1024)
        result = transform(image_pil, masks_np_list)
        # result["image"]: Tensor[3, 1024, 1024]
        # result["masks"]: Tensor[N, 1024, 1024]
    """

    def __init__(
        self,
        resolution: int = _DEFAULT_RESOLUTION,
        horizontal_flip_prob: float = 0.5,
        normalize: bool = True,
    ) -> None:
        self.resolution: int = resolution
        self.horizontal_flip_prob: float = horizontal_flip_prob
        self.normalize: bool = normalize

        # Precompute normalization tensors for efficiency
        self._mean: Tensor = torch.tensor(
            _IMAGENET_MEAN, dtype=torch.float32
        ).view(3, 1, 1)
        self._std: Tensor = torch.tensor(
            _IMAGENET_STD, dtype=torch.float32
        ).view(3, 1, 1)

    def __call__(
        self,
        image: Image.Image,
        masks: List[np.ndarray],
    ) -> Dict[str, Any]:
        """Apply transforms to image and masks.

        Args:
            image: PIL Image in RGB format.
            masks: List of binary numpy arrays, each of shape [H, W], dtype uint8.
                Values are 0 (background) or 1 (foreground).

        Returns:
            Dict with:
                - "image": Tensor[3, resolution, resolution], float32, normalized
                - "masks": Tensor[N, resolution, resolution], float32 binary {0, 1}
                    Shape [0, resolution, resolution] if masks is empty.
        """
        # ------------------------------------------------------------------
        # Step 1: Convert PIL image to float32 tensor [3, H, W] in [0, 1]
        # ------------------------------------------------------------------
        img_np: np.ndarray = np.array(image, dtype=np.float32) / 255.0
        if img_np.ndim == 2:
            # Grayscale → replicate to 3 channels
            img_np = np.stack([img_np, img_np, img_np], axis=-1)
        elif img_np.shape[2] == 4:
            # RGBA → drop alpha channel
            img_np = img_np[:, :, :3]

        # [H, W, 3] → [3, H, W]
        img_tensor: Tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        # ------------------------------------------------------------------
        # Step 2: Resize image to square resolution using bilinear interpolation
        # ------------------------------------------------------------------
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0),  # [1, 3, H, W]
            size=(self.resolution, self.resolution),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)  # [3, resolution, resolution]

        # ------------------------------------------------------------------
        # Step 3: Convert and resize masks
        # ------------------------------------------------------------------
        if len(masks) > 0:
            # Stack masks: [N, H_orig, W_orig]
            masks_np: np.ndarray = np.stack(masks, axis=0).astype(np.float32)
            masks_tensor: Tensor = torch.from_numpy(masks_np).unsqueeze(1)
            # [N, 1, H_orig, W_orig] → resize → [N, 1, resolution, resolution]
            masks_tensor = F.interpolate(
                masks_tensor,
                size=(self.resolution, self.resolution),
                mode="nearest",
            )
            # Threshold to ensure binary values: [N, resolution, resolution]
            masks_tensor = (masks_tensor.squeeze(1) >= 0.5).float()
        else:
            masks_tensor = torch.zeros(
                0, self.resolution, self.resolution, dtype=torch.float32
            )

        # ------------------------------------------------------------------
        # Step 4: Random horizontal flip (consistent across image and masks)
        # Config: pretrain.augmentation.horizontal_flip: true
        # ------------------------------------------------------------------
        if random.random() < self.horizontal_flip_prob:
            img_tensor = torch.flip(img_tensor, dims=[2])  # flip W dimension
            if masks_tensor.shape[0] > 0:
                masks_tensor = torch.flip(masks_tensor, dims=[2])  # flip W dimension

        # ------------------------------------------------------------------
        # Step 5: Normalize image with ImageNet mean/std
        # ------------------------------------------------------------------
        if self.normalize:
            img_tensor = (img_tensor - self._mean) / self._std

        return {
            "image": img_tensor,    # [3, resolution, resolution]
            "masks": masks_tensor,  # [N, resolution, resolution]
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"resolution={self.resolution}, "
            f"horizontal_flip_prob={self.horizontal_flip_prob}, "
            f"normalize={self.normalize})"
        )


# ---------------------------------------------------------------------------
# SA1BDataset
# ---------------------------------------------------------------------------


class SA1BDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for SA-1B image and mask loading for SAM 2 pre-training.

    Loads SA-1B images and their COCO-RLE JSON mask annotations. Implements
    the filtering and sampling strategy described in Appendix D.2.1:
        - Filter masks covering > 90% of image area
        - Randomly sample up to 64 masks per image
        - Apply horizontal flip and square resize augmentations

    SA-1B directory structure:
        root/
          sa_000000/
            sa_1.jpg
            sa_1.json
            sa_2.jpg
            sa_2.json
            ...
          sa_000001/
            ...

    Each JSON file contains:
        {
          "image": {"file_name": str, "height": int, "width": int},
          "annotations": [
            {
              "id": int,
              "segmentation": {"size": [H, W], "counts": str},  # COCO RLE
              "area": float,  # pre-computed pixel count
              "bbox": [x, y, w, h],
              "predicted_iou": float,
              "stability_score": float,
            },
            ...
          ]
        }

    Config references (config.yaml pretrain section):
        pretrain.data.max_masks_per_image: 64
        pretrain.data.mask_area_filter: 0.90
        pretrain.resolution: 1024

    Args:
        root: Path to the SA-1B dataset root directory containing shard
            subdirectories (sa_000000, sa_000001, etc.).
        transform: Optional callable that accepts (PIL.Image, List[np.ndarray])
            and returns a dict with "image" and "masks" tensors. If None,
            SA1BDefaultTransform(resolution=1024) is used.
        max_masks_per_image: Maximum number of masks to sample per image.
            Defaults to 64 (config: pretrain.data.max_masks_per_image).
        mask_area_filter: Maximum allowed normalized mask area (mask_area /
            image_area). Masks with area > this threshold are discarded.
            Defaults to 0.90 (config: pretrain.data.mask_area_filter).
        resolution: Target square resolution for image and mask resizing.
            Only used when transform is None (for SA1BDefaultTransform).
            Defaults to 1024 (config: pretrain.resolution).
        max_images: Optional cap on the number of images to index. Useful for
            ablations and debugging. None means use all available images.

    Example:
        dataset = SA1BDataset(
            root="/data/sa1b",
            max_masks_per_image=64,
            mask_area_filter=0.90,
        )
        sample = dataset[0]
        # sample["image"]: Tensor[3, 1024, 1024]
        # sample["masks"]: Tensor[N, 1024, 1024], N <= 64
        # sample["num_masks"]: int
        # sample["image_path"]: str
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        max_masks_per_image: int = _DEFAULT_MAX_MASKS,
        mask_area_filter: float = _DEFAULT_AREA_FILTER,
        resolution: int = _DEFAULT_RESOLUTION,
        max_images: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.root: str = root
        self.max_masks_per_image: int = max_masks_per_image
        self.mask_area_filter: float = mask_area_filter
        self.resolution: int = resolution

        # Use default transform if none provided
        if transform is None:
            self.transform: Callable = SA1BDefaultTransform(
                resolution=resolution,
                horizontal_flip_prob=0.5,
                normalize=True,
            )
        else:
            self.transform = transform

        # ------------------------------------------------------------------
        # Index all .jpg image paths at init time.
        # Annotation JSON files are loaded lazily in __getitem__ to avoid
        # loading all ~11M JSON files into memory upfront.
        # ------------------------------------------------------------------
        self.image_paths: List[str] = self._index_image_paths(root, max_images)

        logger.info(
            "SA1BDataset initialized: root=%s, num_images=%d, "
            "max_masks=%d, area_filter=%.2f, resolution=%d",
            root,
            len(self.image_paths),
            max_masks_per_image,
            mask_area_filter,
            resolution,
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of images in the dataset.

        Returns:
            Integer count of indexed image paths.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load and return a single image with its sampled masks.

        Implements the full data loading pipeline:
            1. Load image from disk as PIL Image
            2. Load and parse JSON annotations
            3. Filter masks by area threshold
            4. Randomly sample up to max_masks_per_image
            5. Decode RLE for sampled masks only
            6. Apply transform to image and masks
            7. Return structured dict

        Args:
            idx: Integer index into self.image_paths.

        Returns:
            Dict with keys:
                - "image": Tensor[3, H, W] float32, normalized
                - "masks": Tensor[N, H, W] float32 binary {0, 1}, N <= 64
                - "num_masks": int, actual number of valid masks
                - "image_path": str, path to the source image file
                - "valid_mask": Tensor[max_masks_per_image] bool, True for
                    real masks, False for padding slots (enables fixed-size batching)

        Note:
            Returns a sample with empty masks (num_masks=0) on errors rather
            than raising exceptions, to allow DataLoader to continue.
        """
        image_path: str = self.image_paths[idx]

        # ------------------------------------------------------------------
        # Step 1: Load image from disk
        # ------------------------------------------------------------------
        image: Optional[Image.Image] = self._load_image(image_path)
        if image is None:
            # Return a placeholder sample on image load failure
            return self._make_empty_sample(image_path)

        img_width: int = image.width
        img_height: int = image.height
        image_area: int = img_width * img_height

        # ------------------------------------------------------------------
        # Step 2: Load and parse JSON annotations
        # ------------------------------------------------------------------
        raw_annotations: List[Dict] = self._load_annotations(image_path)

        # ------------------------------------------------------------------
        # Step 3: Filter masks by area threshold
        # Config: pretrain.data.mask_area_filter: 0.90
        # ------------------------------------------------------------------
        filtered_annotations: List[Dict] = self._filter_masks(
            raw_annotations, image_area
        )

        # ------------------------------------------------------------------
        # Step 4: Randomly sample up to max_masks_per_image
        # Config: pretrain.data.max_masks_per_image: 64
        # ------------------------------------------------------------------
        num_available: int = len(filtered_annotations)
        if num_available > self.max_masks_per_image:
            sampled_annotations: List[Dict] = random.sample(
                filtered_annotations, self.max_masks_per_image
            )
        else:
            sampled_annotations = filtered_annotations

        # ------------------------------------------------------------------
        # Step 5: Decode RLE masks for sampled annotations only
        # Decoding only after sampling avoids decoding discarded masks.
        # ------------------------------------------------------------------
        decoded_masks: List[np.ndarray] = []
        for ann in sampled_annotations:
            mask_arr: Optional[np.ndarray] = self._decode_rle_mask(
                ann, img_height, img_width
            )
            if mask_arr is not None:
                decoded_masks.append(mask_arr)

        # ------------------------------------------------------------------
        # Step 6: Apply transform to image and decoded masks
        # ------------------------------------------------------------------
        try:
            transform_result: Dict[str, Any] = self.transform(image, decoded_masks)
        except Exception as exc:
            logger.warning(
                "SA1BDataset: transform failed for %s: %s. "
                "Returning empty sample.",
                image_path,
                exc,
            )
            return self._make_empty_sample(image_path)

        img_tensor: Tensor = transform_result["image"]    # [3, H, W]
        masks_tensor: Tensor = transform_result["masks"]  # [N, H, W]

        num_masks: int = int(masks_tensor.shape[0])

        # ------------------------------------------------------------------
        # Step 7: Pad masks to max_masks_per_image for fixed-size batching
        # Padding with zeros; valid_mask indicates real vs. padding slots.
        # ------------------------------------------------------------------
        padded_masks, valid_mask = self._pad_masks_to_fixed_size(
            masks_tensor, self.max_masks_per_image
        )

        return {
            "image": img_tensor,          # [3, H, W] float32
            "masks": padded_masks,        # [max_masks, H, W] float32
            "valid_mask": valid_mask,     # [max_masks] bool
            "num_masks": num_masks,       # int: actual valid masks
            "image_path": image_path,     # str: for debugging
        }

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    @staticmethod
    def _index_image_paths(
        root: str,
        max_images: Optional[int] = None,
    ) -> List[str]:
        """Walk the SA-1B directory tree and collect all .jpg image paths.

        SA-1B is organized into shard subdirectories (sa_000000, sa_000001, ...).
        Each shard contains .jpg images and corresponding .json annotation files.

        Args:
            root: Path to the SA-1B root directory.
            max_images: Optional cap on the number of images to index.
                None means index all available images.

        Returns:
            Sorted list of absolute paths to .jpg image files.
            Sorted for reproducibility across runs and machines.

        Raises:
            FileNotFoundError: If root directory does not exist.
        """
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"SA-1B root directory not found: {root}. "
                "Please download SA-1B from https://ai.meta.com/datasets/segment-anything/"
            )

        # Use glob to find all .jpg files recursively
        # Pattern: root/**/*.jpg (covers all shard subdirectories)
        pattern: str = os.path.join(root, "**", "*.jpg")
        all_paths: List[str] = sorted(glob.glob(pattern, recursive=True))

        if len(all_paths) == 0:
            # Try flat structure (some SA-1B distributions are not nested)
            pattern_flat: str = os.path.join(root, "*.jpg")
            all_paths = sorted(glob.glob(pattern_flat))

        if len(all_paths) == 0:
            logger.warning(
                "SA1BDataset: No .jpg files found in %s. "
                "Check that the SA-1B dataset is correctly downloaded and extracted.",
                root,
            )

        # Apply optional cap for ablations and debugging
        if max_images is not None and max_images < len(all_paths):
            all_paths = all_paths[:max_images]
            logger.info(
                "SA1BDataset: Capped to %d images (max_images=%d).",
                len(all_paths),
                max_images,
            )

        logger.info(
            "SA1BDataset: Indexed %d images from %s.",
            len(all_paths),
            root,
        )

        return all_paths

    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load an image from disk as a PIL Image in RGB format.

        Args:
            image_path: Absolute path to the .jpg image file.

        Returns:
            PIL Image in RGB mode, or None if loading fails.
        """
        try:
            img: Image.Image = Image.open(image_path).convert("RGB")
            return img
        except Exception as exc:
            logger.warning(
                "SA1BDataset: Failed to load image %s: %s",
                image_path,
                exc,
            )
            return None

    def _load_annotations(self, image_path: str) -> List[Dict]:
        """Load and parse the JSON annotation file for an image.

        Derives the JSON path by replacing the .jpg extension with .json.
        Returns an empty list if the JSON file does not exist or is malformed.

        Args:
            image_path: Absolute path to the .jpg image file.

        Returns:
            List of raw annotation dicts from data["annotations"].
            Each dict contains at minimum: "segmentation", "area", "id".
            Returns empty list on any error.
        """
        json_path: str = os.path.splitext(image_path)[0] + ".json"

        if not os.path.isfile(json_path):
            logger.debug(
                "SA1BDataset: JSON annotation file not found: %s. "
                "Returning empty annotations.",
                json_path,
            )
            return []

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data: Dict = json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            logger.warning(
                "SA1BDataset: Failed to parse JSON %s: %s. "
                "Returning empty annotations.",
                json_path,
                exc,
            )
            return []

        # Extract annotations list; handle both standard and variant formats
        annotations: List[Dict] = data.get("annotations", [])

        if not isinstance(annotations, list):
            logger.warning(
                "SA1BDataset: 'annotations' field in %s is not a list "
                "(got %s). Returning empty annotations.",
                json_path,
                type(annotations).__name__,
            )
            return []

        return annotations

    def _filter_masks(
        self,
        masks: List[Dict],
        image_area: int,
    ) -> List[Dict]:
        """Filter out masks that cover more than mask_area_filter of the image.

        From Appendix D.2.1: "we filter masks covering more than 90% of the image."

        Uses the pre-computed "area" field in each annotation (pixel count) to
        avoid decoding RLE masks just for filtering. This is the efficient path.

        Config reference: pretrain.data.mask_area_filter: 0.90

        Args:
            masks: List of raw annotation dicts, each containing "area" (float,
                pre-computed pixel count of the mask foreground).
            image_area: Total image area in pixels (height × width).

        Returns:
            Filtered list of annotation dicts where each mask's normalized area
            (mask_area / image_area) is <= mask_area_filter (0.90).
            Returns empty list if input is empty or image_area is 0.
        """
        if not masks or image_area <= 0:
            return []

        filtered: List[Dict] = []
        for ann in masks:
            # "area" is the pre-computed pixel count of the mask foreground
            mask_area: float = float(ann.get("area", 0.0))
            normalized_area: float = mask_area / image_area

            if normalized_area <= self.mask_area_filter:
                filtered.append(ann)
            else:
                logger.debug(
                    "SA1BDataset: Filtered mask with normalized area %.4f > %.2f.",
                    normalized_area,
                    self.mask_area_filter,
                )

        return filtered

    @staticmethod
    def _decode_rle_mask(
        annotation: Dict,
        height: int,
        width: int,
    ) -> Optional[np.ndarray]:
        """Decode a COCO RLE mask annotation to a binary numpy array.

        SA-1B uses COCO RLE format stored in annotation["segmentation"].
        The "counts" field can be either:
            - A string (uncompressed RLE): needs encoding to bytes
            - A bytes object (compressed RLE): used directly

        Args:
            annotation: Single annotation dict containing "segmentation" with
                keys "size" [H, W] and "counts" (str or bytes).
            height: Expected image height for validation.
            width: Expected image width for validation.

        Returns:
            Binary numpy array of shape [H, W], dtype uint8, with values
            0 (background) or 1 (foreground).
            Returns None if decoding fails.
        """
        segmentation: Optional[Dict] = annotation.get("segmentation")
        if segmentation is None:
            logger.debug(
                "SA1BDataset: Annotation missing 'segmentation' field. Skipping."
            )
            return None

        try:
            # Ensure counts is bytes for pycocotools compatibility
            rle: Dict = dict(segmentation)  # shallow copy to avoid mutation
            counts = rle.get("counts")

            if isinstance(counts, str):
                # Uncompressed RLE string → encode to bytes
                rle["counts"] = counts.encode("utf-8")
            elif isinstance(counts, list):
                # Polygon format (not expected in SA-1B, but handle gracefully)
                # Convert polygon to RLE via pycocotools
                size: List[int] = rle.get("size", [height, width])
                rle = coco_mask_utils.frPyObjects([counts], size[0], size[1])
                if isinstance(rle, list):
                    rle = coco_mask_utils.merge(rle)
            elif isinstance(counts, bytes):
                pass  # Already bytes — use as-is
            else:
                logger.debug(
                    "SA1BDataset: Unexpected 'counts' type %s. Skipping.",
                    type(counts).__name__,
                )
                return None

            # Decode RLE to binary mask: returns Fortran-contiguous uint8 array
            decoded: np.ndarray = coco_mask_utils.decode(rle)

            # Validate shape
            if decoded.shape[0] != height or decoded.shape[1] != width:
                logger.debug(
                    "SA1BDataset: Decoded mask shape (%d, %d) != image shape (%d, %d). "
                    "Skipping.",
                    decoded.shape[0],
                    decoded.shape[1],
                    height,
                    width,
                )
                return None

            # Convert to C-contiguous uint8 array
            return np.ascontiguousarray(decoded, dtype=np.uint8)

        except Exception as exc:
            logger.debug(
                "SA1BDataset: RLE decode failed: %s. Skipping mask.",
                exc,
            )
            return None

    def _pad_masks_to_fixed_size(
        self,
        masks: Tensor,
        target_size: int,
    ) -> Tuple[Tensor, Tensor]:
        """Pad masks tensor to a fixed number of slots for batched DataLoader.

        Pads with zero masks to reach target_size slots. Returns a validity
        boolean tensor indicating which slots contain real masks.

        This enables fixed-size batching without a custom collate_fn:
            - masks: [target_size, H, W] — real masks + zero padding
            - valid_mask: [target_size] bool — True for real, False for padding

        Args:
            masks: Mask tensor of shape [N, H, W] where N <= target_size.
            target_size: Target number of mask slots (max_masks_per_image = 64).

        Returns:
            Tuple of:
                - padded_masks: Tensor[target_size, H, W] float32
                - valid_mask: Tensor[target_size] bool

        Raises:
            ValueError: If N > target_size (should not happen after sampling).
        """
        N: int = masks.shape[0]

        if N > target_size:
            raise ValueError(
                f"Number of masks ({N}) exceeds target_size ({target_size}). "
                "This should not happen after sampling in __getitem__."
            )

        # Build validity tensor: True for real masks, False for padding
        valid_mask: Tensor = torch.zeros(target_size, dtype=torch.bool)
        valid_mask[:N] = True

        if N == target_size:
            # No padding needed
            return masks.float(), valid_mask

        if N == 0:
            # All padding — determine spatial size from resolution
            h: int = self.resolution
            w: int = self.resolution
            padded: Tensor = torch.zeros(target_size, h, w, dtype=torch.float32)
            return padded, valid_mask

        # Pad with zeros along the first dimension
        h = masks.shape[1]
        w = masks.shape[2]
        padding_slots: int = target_size - N
        padding: Tensor = torch.zeros(
            padding_slots, h, w, dtype=torch.float32
        )
        padded_masks: Tensor = torch.cat(
            [masks.float(), padding], dim=0
        )  # [target_size, H, W]

        return padded_masks, valid_mask

    def _make_empty_sample(self, image_path: str) -> Dict[str, Any]:
        """Create a placeholder sample for failed image loads.

        Returns a sample with a black image and all-zero masks. The trainer
        should handle samples with num_masks=0 gracefully (skip loss computation
        for mask outputs when no valid masks are present).

        Args:
            image_path: Path to the image that failed to load (for logging).

        Returns:
            Dict with the same structure as a normal __getitem__ return,
            but with a black image and empty masks.
        """
        h: int = self.resolution
        w: int = self.resolution

        # Black image (all zeros after normalization would be non-zero,
        # but we return raw zeros here for simplicity)
        empty_image: Tensor = torch.zeros(3, h, w, dtype=torch.float32)

        # Empty padded masks
        empty_masks: Tensor = torch.zeros(
            self.max_masks_per_image, h, w, dtype=torch.float32
        )
        empty_valid: Tensor = torch.zeros(
            self.max_masks_per_image, dtype=torch.bool
        )

        return {
            "image": empty_image,
            "masks": empty_masks,
            "valid_mask": empty_valid,
            "num_masks": 0,
            "image_path": image_path,
        }

    # ------------------------------------------------------------------
    # Utility / inspection methods
    # ------------------------------------------------------------------

    def get_image_path(self, idx: int) -> str:
        """Return the image path for a given dataset index.

        Args:
            idx: Integer index into self.image_paths.

        Returns:
            Absolute path to the .jpg image file.
        """
        return self.image_paths[idx]

    def get_json_path(self, idx: int) -> str