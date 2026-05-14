## datasets/manyshot_loader.py
"""Many-shot data loader for the PEFT Visual Recognition reproduction study.

This module provides the ManyShotLoader class for loading full training
datasets used in the many-shot evaluation (Section 5 of the paper). It
handles three datasets:

- CIFAR-100: 50K training images, 100 classes (torchvision)
- RESISC45: 25.2K training images, 45 classes (ImageFolder)
- Clevr-Distance: 70K training images, 6 depth classes (TensorFlow Datasets)

Dataset-specific augmentations follow Appendix A.1 of the paper:
- CIFAR-100: RandomHorizontalFlip only
- RESISC45: RandomHorizontalFlip + RandomVerticalFlip
- Clevr-Distance: No augmentation

All datasets use ImageNet normalization (Appendix A.2):
"All data are normalized by ImageNet mean and standard deviation."

Typical usage:
    loader = ManyShotLoader(
        dataset_name="cifar100",
        data_dir="./data",
        batch_size=64,
        image_size=224,
        num_workers=4,
    )
    train_loader, val_loader = loader.get_train_val_loaders(val_ratio=0.1)
    test_loader = loader.get_test_loader()
    n_cls = loader.num_classes()
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100, ImageFolder

from datasets.vtab_loader import IMAGENET_MEAN, IMAGENET_STD

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported dataset names
# ---------------------------------------------------------------------------
SUPPORTED_DATASETS: List[str] = ["cifar100", "resisc45", "clevr_distance"]

# ---------------------------------------------------------------------------
# Dataset configurations (from config.yaml: manyshot.datasets)
# ---------------------------------------------------------------------------
DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "cifar100": {
        "num_classes": 100,
        "train_size": 50000,
        "augmentation": {
            "random_horizontal_flip": True,
            "random_vertical_flip": False,
        },
        "source": "torchvision",
    },
    "resisc45": {
        "num_classes": 45,
        "train_size": 25200,
        "augmentation": {
            "random_horizontal_flip": True,
            "random_vertical_flip": True,
        },
        "source": "folder",
    },
    "clevr_distance": {
        "num_classes": 6,
        "train_size": 70000,
        "augmentation": {
            "random_horizontal_flip": False,
            "random_vertical_flip": False,
        },
        "source": "tensorflow_datasets",
    },
}

# ---------------------------------------------------------------------------
# Clevr-Distance depth bin edges (VTAB benchmark definition).
# 6 classes correspond to depth ranges in normalized Clevr units.
# Bins: [0,8), [8,9), [9,10), [10,11), [11,12), [12,inf)
# ---------------------------------------------------------------------------
CLEVR_DISTANCE_BIN_EDGES: List[float] = [0.0, 8.0, 9.0, 10.0, 11.0, 12.0]


# ---------------------------------------------------------------------------
# Inner helper: TransformDataset
# ---------------------------------------------------------------------------

class _TransformDataset(Dataset):
    """Lightweight Dataset wrapper that applies a transform to a Subset.

    This solves the train/val transform mismatch: both splits share the same
    underlying data (loaded once without transforms), but each gets its own
    transform applied in __getitem__.

    Attributes:
        subset: A torch.utils.data.Subset (or any Dataset) providing raw
            (PIL Image, label) pairs.
        transform: torchvision transform pipeline applied to each image.
    """

    def __init__(
        self,
        subset: Dataset,
        transform: transforms.Compose,
    ) -> None:
        """Initialises the wrapper.

        Args:
            subset: Underlying dataset returning (PIL Image, int label) pairs.
            transform: Transform pipeline to apply to each image.
        """
        self.subset: Dataset = subset
        self.transform: transforms.Compose = transform

    def __len__(self) -> int:
        """Returns the number of samples."""
        return len(self.subset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (transformed_image, label) at index idx.

        Args:
            idx: Integer index.

        Returns:
            Tuple of (image_tensor, label_tensor).
        """
        image: Any
        label: Any
        image, label = self.subset[idx]

        # image may be a PIL Image (from raw dataset) or already a tensor.
        if isinstance(image, torch.Tensor):
            # Already a tensor — apply only normalize-compatible transforms.
            image_tensor: torch.Tensor = image
        else:
            image_tensor = self.transform(image)

        label_tensor: torch.Tensor = (
            label
            if isinstance(label, torch.Tensor)
            else torch.tensor(label, dtype=torch.long)
        )
        return image_tensor, label_tensor


# ---------------------------------------------------------------------------
# Inner helper: _RawDataset (no-transform wrapper for torchvision datasets)
# ---------------------------------------------------------------------------

class _RawCIFAR100(Dataset):
    """CIFAR-100 wrapper that returns raw PIL Images without any transform.

    Used to enable separate train/val transforms via _TransformDataset.

    Attributes:
        dataset: Underlying torchvision CIFAR100 dataset with transform=None.
    """

    def __init__(self, root: str, train: bool, download: bool = True) -> None:
        """Loads CIFAR-100 without any transform.

        Args:
            root: Root directory for CIFAR-100 data.
            train: If True, loads training split; otherwise test split.
            download: If True, downloads the dataset if not present.
        """
        self.dataset: CIFAR100 = CIFAR100(
            root=root,
            train=train,
            download=download,
            transform=None,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        """Returns (PIL Image, int label) without any transform."""
        image: Image.Image
        label: int
        image, label = self.dataset[idx]
        # torchvision CIFAR100 with transform=None returns PIL Image directly.
        return image, label


class _RawImageFolder(Dataset):
    """ImageFolder wrapper that returns raw PIL Images without any transform.

    Used for RESISC45 to enable separate train/val transforms.

    Attributes:
        dataset: Underlying torchvision ImageFolder with transform=None.
    """

    def __init__(self, root: str) -> None:
        """Loads ImageFolder without any transform.

        Args:
            root: Root directory containing class subdirectories.

        Raises:
            FileNotFoundError: If root directory does not exist.
        """
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"RESISC45 directory not found: '{root}'. "
                "Please download RESISC45 and organize it as:\n"
                "  data_dir/resisc45/train/<class_name>/<image>.jpg\n"
                "  data_dir/resisc45/test/<class_name>/<image>.jpg\n"
                "Download from: https://onedrive.live.com/?authkey=..."
            )
        self.dataset: ImageFolder = ImageFolder(root=root, transform=None)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        """Returns (PIL Image, int label) without any transform."""
        image: Image.Image
        label: int
        image, label = self.dataset[idx]
        return image, label


# ---------------------------------------------------------------------------
# Inner class: ClevrDistanceDataset
# ---------------------------------------------------------------------------

class ClevrDistanceDataset(Dataset):
    """PyTorch Dataset for the Clevr-Distance task.

    Loads the CLEVR dataset from TensorFlow Datasets and extracts the
    distance-to-nearest-object label, discretized into 6 bins matching
    the VTAB benchmark definition.

    The 6 distance classes correspond to depth ranges:
        Class 0: depth in [0.0, 8.0)
        Class 1: depth in [8.0, 9.0)
        Class 2: depth in [9.0, 10.0)
        Class 3: depth in [10.0, 11.0)
        Class 4: depth in [11.0, 12.0)
        Class 5: depth in [12.0, inf)

    Attributes:
        samples: List of (PIL Image, int label) tuples loaded into memory.
        transform: Optional torchvision transform applied in __getitem__.
        num_classes: Number of distance classes (6).
    """

    NUM_CLASSES: int = 6

    def __init__(
        self,
        data_dir: str,
        split: str,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        """Loads and preprocesses the Clevr-Distance dataset.

        Iterates the TFDS 'clevr' dataset once, extracting images and
        computing distance labels. All samples are cached in memory as
        PIL Images to avoid repeated TFDS iteration.

        Args:
            data_dir: Root directory for TFDS data cache.
            split: Either 'train' or 'test'. Maps to TFDS 'train' and
                'validation' splits respectively (VTAB convention).
            transform: Optional torchvision transform applied in __getitem__.

        Raises:
            ImportError: If tensorflow or tensorflow_datasets are not installed.
            RuntimeError: If TFDS loading fails.
        """
        self.transform: Optional[transforms.Compose] = transform
        self.num_classes: int = self.NUM_CLASSES
        self.samples: List[Tuple[Image.Image, int]] = []

        # Map split name to TFDS split string.
        tfds_split: str = "train" if split == "train" else "validation"

        _logger.info(
            "Loading Clevr-Distance from TFDS (split='%s' -> tfds_split='%s')",
            split,
            tfds_split,
        )

        # ------------------------------------------------------------------
        # Import TensorFlow and disable GPU to avoid PyTorch conflict.
        # ------------------------------------------------------------------
        try:
            import tensorflow as tf  # type: ignore
            import tensorflow_datasets as tfds  # type: ignore

            tf.config.set_visible_devices([], "GPU")
        except ImportError as exc:
            raise ImportError(
                "tensorflow and tensorflow-datasets are required for Clevr-Distance. "
                "Install with: pip install tensorflow tensorflow-datasets"
            ) from exc

        # ------------------------------------------------------------------
        # Load TFDS dataset.
        # ------------------------------------------------------------------
        try:
            tf_dataset = tfds.load(
                "clevr",
                split=tfds_split,
                data_dir=data_dir,
                as_supervised=False,
                with_info=False,
                shuffle_files=False,
            )
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                f"Failed to load Clevr dataset from TFDS: {exc}\n"
                f"Ensure tensorflow-datasets is installed and data_dir='{data_dir}' "
                "is accessible."
            ) from exc

        # ------------------------------------------------------------------
        # Iterate and convert samples.
        # ------------------------------------------------------------------
        num_loaded: int = 0
        for sample in tf_dataset:
            # Convert image to PIL.
            img_array: np.ndarray = sample["image"].numpy()
            pil_image: Image.Image = Image.fromarray(img_array, mode="RGB")

            # Extract distance label.
            label: int = self._extract_distance_label(sample)

            self.samples.append((pil_image, label))
            num_loaded += 1

            if num_loaded % 10000 == 0:
                _logger.info("Clevr-Distance: loaded %d samples...", num_loaded)

        _logger.info(
            "Clevr-Distance loaded: %d samples (split='%s')",
            len(self.samples),
            split,
        )

    def _extract_distance_label(self, sample: Any) -> int:
        """Extracts and bins the distance-to-nearest-object label.

        Finds the minimum z-depth across all objects in the scene and
        discretizes it into one of 6 bins using CLEVR_DISTANCE_BIN_EDGES.

        Args:
            sample: A TFDS sample dict containing 'objects' with
                'pixel_coords' of shape (num_objects, 3).

        Returns:
            Integer class label in [0, 5].
        """
        try:
            # pixel_coords shape: (num_objects, 3), index 2 is z-depth.
            pixel_coords: np.ndarray = sample["objects"]["pixel_coords"].numpy()

            if pixel_coords.ndim == 2 and pixel_coords.shape[1] >= 3:
                # Find minimum depth across all objects.
                min_depth: float = float(np.min(pixel_coords[:, 2]))
            elif pixel_coords.ndim == 1 and len(pixel_coords) >= 3:
                # Single object.
                min_depth = float(pixel_coords[2])
            else:
                _logger.warning(
                    "Unexpected pixel_coords shape %s; defaulting to class 0.",
                    pixel_coords.shape,
                )
                return 0

        except (KeyError, AttributeError, Exception) as exc:  # pylint: disable=broad-except
            _logger.warning(
                "Failed to extract depth from sample: %s. Defaulting to class 0.", exc
            )
            return 0

        # Bin into 6 classes using np.digitize.
        # np.digitize returns index in [1, len(bins)] for values >= bins[0].
        # We subtract 1 to get 0-indexed classes and clamp to [0, NUM_CLASSES-1].
        bin_edges: np.ndarray = np.array(CLEVR_DISTANCE_BIN_EDGES)
        label: int = int(np.digitize(min_depth, bin_edges)) - 1
        label = max(0, min(label, self.NUM_CLASSES - 1))

        return label

    def __len__(self) -> int:
        """Returns the number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (transformed_image, label_tensor) at index idx.

        Args:
            idx: Integer index.

        Returns:
            Tuple of (image_tensor of shape (3, H, W), scalar label tensor).
        """
        pil_image: Image.Image
        label: int
        pil_image, label = self.samples[idx]

        image_tensor: torch.Tensor
        if self.transform is not None:
            image_tensor = self.transform(pil_image)
        else:
            image_tensor = transforms.ToTensor()(pil_image)

        label_tensor: torch.Tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor


# ---------------------------------------------------------------------------
# Public API: ManyShotLoader
# ---------------------------------------------------------------------------

class ManyShotLoader:
    """Data loader for many-shot evaluation datasets.

    Handles CIFAR-100, RESISC45, and Clevr-Distance with dataset-specific
    augmentations and a deterministic 90/10 train/val split. All datasets
    use ImageNet normalization and 224×224 input resolution for ViT-B/16.

    Paper Section 5: "We select one representative dataset from each group
    in VTAB: (1) CIFAR-100, (2) RESISC, and (3) Clevr-Distance."

    Attributes:
        dataset_name: One of 'cifar100', 'resisc45', 'clevr_distance'.
        data_dir: Root directory for dataset storage and TFDS cache.
        batch_size: Number of samples per DataLoader batch.
        image_size: Target spatial resolution (default 224).
        num_workers: Number of DataLoader worker processes.
        config: Dataset-specific configuration dict from DATASET_CONFIGS.
    """

    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        batch_size: int = 64,
        image_size: int = 224,
        num_workers: int = 4,
    ) -> None:
        """Initialises the loader and validates the dataset name.

        Does NOT load data — deferred to get_* methods.

        Args:
            dataset_name: One of 'cifar100', 'resisc45', 'clevr_distance'.
            data_dir: Root directory for dataset storage and TFDS cache.
            batch_size: Batch size for all DataLoaders. Default: 64
                (config.yaml: manyshot.training.batch_size).
            image_size: Target image resolution. Default: 224
                (config.yaml: image_size).
            num_workers: Number of DataLoader worker processes. Default: 4
                (config.yaml: num_workers).

        Raises:
            ValueError: If dataset_name is not in SUPPORTED_DATASETS.
        """
        if dataset_name not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unknown many-shot dataset: '{dataset_name}'. "
                f"Supported datasets: {SUPPORTED_DATASETS}"
            )

        self.dataset_name: str = dataset_name
        self.data_dir: str = data_dir
        self.batch_size: int = batch_size
        self.image_size: int = image_size
        self.num_workers: int = num_workers
        self.config: Dict[str, Any] = DATASET_CONFIGS[dataset_name]

        _logger.info(
            "ManyShotLoader initialised: dataset=%s, num_classes=%d, "
            "batch_size=%d, image_size=%d, source=%s",
            self.dataset_name,
            self.config["num_classes"],
            self.batch_size,
            self.image_size,
            self.config["source"],
        )

    # ------------------------------------------------------------------
    # Public DataLoader methods
    # ------------------------------------------------------------------

    def get_train_val_loaders(
        self, val_ratio: float = 0.1
    ) -> Tuple[DataLoader, DataLoader]:
        """Returns DataLoaders for the 90/10 train/val split.

        Loads the full training dataset without transforms, splits indices
        deterministically (seed=42), then wraps each split with the
        appropriate transform via _TransformDataset. The val split always
        uses base transforms (no augmentation) for unbiased evaluation.

        Paper Appendix A.2: "We perform 90/10 train-val split for CIFAR-100,
        RESISC and Clevr-Distance."

        Args:
            val_ratio: Fraction of training samples for validation.
                Default: 0.1 (config.yaml: manyshot.val_ratio).

        Returns:
            Tuple (train_loader, val_loader) where:
            - train_loader: DataLoader with training augmentation, shuffled.
            - val_loader: DataLoader with base transforms only, not shuffled.

        Raises:
            ValueError: If val_ratio results in 0 validation samples.
        """
        # ------------------------------------------------------------------
        # Load raw dataset (no transforms) for index-based splitting.
        # ------------------------------------------------------------------
        raw_dataset: Dataset = self._load_raw_dataset(split="train")

        total_size: int = len(raw_dataset)  # type: ignore[arg-type]
        val_size: int = int(total_size * val_ratio)
        train_size: int = total_size - val_size

        if val_size == 0:
            raise ValueError(
                f"val_ratio={val_ratio} results in 0 validation samples "
                f"for dataset '{self.dataset_name}' with {total_size} total samples."
            )

        _logger.info(
            "Splitting %d training samples: %d train / %d val "
            "(val_ratio=%.2f, seed=42)",
            total_size,
            train_size,
            val_size,
            val_ratio,
        )

        # Deterministic split — seed=42 from config.yaml.
        generator: torch.Generator = torch.Generator()
        generator.manual_seed(42)

        train_subset: Subset
        val_subset: Subset
        train_subset, val_subset = torch.utils.data.random_split(
            raw_dataset,
            [train_size, val_size],
            generator=generator,
        )

        # ------------------------------------------------------------------
        # Wrap each split with the appropriate transform.
        # Train split: augmentation + base transforms.
        # Val split: base transforms only (no augmentation).
        # ------------------------------------------------------------------
        train_transform: transforms.Compose = self._get_augmentation(
            self.dataset_name
        )
        base_transform: transforms.Compose = self._get_base_transforms()

        train_dataset: _TransformDataset = _TransformDataset(
            subset=train_subset,
            transform=train_transform,
        )
        val_dataset: _TransformDataset = _TransformDataset(
            subset=val_subset,
            transform=base_transform,
        )

        train_loader: DataLoader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        val_loader: DataLoader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        return train_loader, val_loader

    def get_test_loader(self) -> DataLoader:
        """Returns a DataLoader over the test split with base transforms only.

        No augmentation is applied to the test split.

        Returns:
            DataLoader over the test set, not shuffled, with base transforms
            (Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize).
        """
        base_transform: transforms.Compose = self._get_base_transforms()
        test_dataset: Dataset = self._load_dataset(
            split="test", transform=base_transform
        )

        test_loader: DataLoader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        _logger.info(
            "Test loader created: %d samples for dataset '%s'",
            len(test_dataset),  # type: ignore[arg-type]
            self.dataset_name,
        )

        return test_loader

    def num_classes(self) -> int:
        """Returns the number of output classes for this dataset.

        Returns:
            Integer number of classes: CIFAR-100→100, RESISC45→45,
            Clevr-Distance→6.
        """
        return self.config["num_classes"]

    # ------------------------------------------------------------------
    # Private transform builders
    # ------------------------------------------------------------------

    def _get_base_transforms(self) -> transforms.Compose:
        """Builds the standard ViT evaluation preprocessing pipeline.

        Pipeline: Resize(256) -> CenterCrop(image_size) -> ToTensor ->
        Normalize(IMAGENET_MEAN, IMAGENET_STD).

        The Resize(256) -> CenterCrop(224) is the standard ViT-B/16
        evaluation pipeline. For CIFAR-100 (32×32 images), Resize(256)
        upscales before cropping — this is intentional and consistent with
        how VTAB handles CIFAR-100.

        Returns:
            torchvision.transforms.Compose pipeline.
        """
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                ),
            ]
        )

    def _get_augmentation(self, dataset_name: str) -> transforms.Compose:
        """Builds the training-time transform pipeline with dataset-specific augmentation.

        Augmentation is prepended to the base transforms:
        - CIFAR-100: RandomHorizontalFlip -> base transforms.
          Paper: "We apply horizontal flipping for CIFAR100"
        - RESISC45: RandomHorizontalFlip + RandomVerticalFlip -> base transforms.
          Paper: "horizontal and vertical flipping for Resisc"
        - Clevr-Distance: base transforms only (no augmentation).
          Paper: "no augmentation for Clevr"

        Args:
            dataset_name: One of 'cifar100', 'resisc45', 'clevr_distance'.

        Returns:
            torchvision.transforms.Compose pipeline with augmentation.
        """
        aug_config: Dict[str, bool] = DATASET_CONFIGS[dataset_name]["augmentation"]
        h_flip: bool = aug_config["random_horizontal_flip"]
        v_flip: bool = aug_config["random_vertical_flip"]

        augmentation_transforms: List[Any] = []

        if h_flip:
            augmentation_transforms.append(transforms.RandomHorizontalFlip())
        if v_flip:
            augmentation_transforms.append(transforms.RandomVerticalFlip())

        # Append base transforms after augmentation.
        base_pipeline: List[Any] = [
            transforms.Resize(256),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]

        return transforms.Compose(augmentation_transforms + base_pipeline)

    # ------------------------------------------------------------------
    # Private dataset loading methods
    # ------------------------------------------------------------------

    def _load_raw_dataset(self, split: str) -> Dataset:
        """Loads the dataset without any transform for index-based splitting.

        Returns raw (PIL Image, int label) pairs. Used by
        get_train_val_loaders() to enable separate train/val transforms.

        Args:
            split: Either 'train' or 'test'.

        Returns:
            Dataset returning (PIL Image, int label) pairs.
        """
        source: str = self.config["source"]

        if source == "torchvision":
            return _RawCIFAR100(
                root=self.data_dir,
                train=(split == "train"),
                download=True,
            )
        elif source == "folder":
            folder_path: str = os.path.join(self.data_dir, "resisc45", split)
            return _RawImageFolder(root=folder_path)
        elif source == "tensorflow_datasets":
            # For Clevr-Distance, load with no transform (transform=None).
            return ClevrDistanceDataset(
                data_dir=self.data_dir,
                split=split,
                transform=None,
            )
        else:
            raise ValueError(
                f"Unknown data source '{source}' for dataset '{self.dataset_name}'."
            )

    def _load_dataset(self, split: str, transform: transforms.Compose) -> Dataset:
        """Loads the dataset with the given transform applied.

        Used by get_test_loader() to load the test split directly with
        base transforms.

        Args:
            split: Either 'train' or 'test'.
            transform: torchvision transform pipeline to apply.

        Returns:
            Dataset returning (image_tensor, label_tensor) pairs.
        """
        source: str = self.config["source"]

        if source == "torchvision":
            # CIFAR-100: use torchvision directly with transform.
            return CIFAR100(
                root=self.data_dir,
                train=(split == "train"),
                download=True,
                transform=transform,
            )
        elif source == "folder":
            # RESISC45: ImageFolder with transform.
            folder_path: str = os.path.join(self.data_dir, "resisc45", split)
            if not os.path.isdir(folder_path):
                raise FileNotFoundError(
                    f"RESISC45 directory not found: '{folder_path}'. "
                    "Please download RESISC45 and organize it as:\n"
                    "  data_dir/resisc45/train/<class_name>/<image>.jpg\n"
                    "  data_dir/resisc45/test/<class_name>/<image>.jpg"
                )
            return ImageFolder(root=folder_path, transform=transform)
        elif source == "tensorflow_datasets":
            # Clevr-Distance: load with transform.
            return ClevrDistanceDataset(
                data_dir=self.data_dir,
                split=split,
                transform=transform,
            )
        else:
            raise ValueError(
                f"Unknown data source '{source}' for dataset '{self.dataset_name}'."
            )
