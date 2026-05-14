## datasets/vtab_loader.py
"""VTAB-1K data loader for the PEFT Visual Recognition reproduction study.

This module provides the VTABLoader class and all associated constants for
loading all 19 VTAB-1K tasks as PyTorch DataLoaders. It bridges TensorFlow
Datasets (the original VTAB format) and PyTorch, handling the TFDS-to-PIL
conversion, deterministic 80/20 train/val splitting, and standard ViT
preprocessing (no augmentation, ImageNet normalization).

Module-level constants VTAB_TASK_NAMES, VTAB_GROUPS, IMAGENET_MEAN, and
IMAGENET_STD are imported by evaluation/metrics.py, datasets/manyshot_loader.py,
and datasets/imagenet_loader.py.

Typical usage:
    loader = VTABLoader(
        dataset_name="dtd",
        data_dir="./data/vtab",
        batch_size=64,
        image_size=224,
        num_workers=4,
    )
    train_loader, val_loader = loader.get_train_val_loaders(val_ratio=0.2)
    test_loader = loader.get_test_loader()
    full_train_loader = loader.get_full_train_loader()
    n_cls = loader.num_classes()
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared normalisation constants (config.yaml: normalization.mean / .std)
# Imported by datasets/manyshot_loader.py and datasets/imagenet_loader.py.
# ---------------------------------------------------------------------------
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# VTAB-1K fixed training set size (paper Section 3)
# ---------------------------------------------------------------------------
VTAB_TRAIN_SIZE: int = 1000

# ---------------------------------------------------------------------------
# Canonical task names (19 tasks, matching config.yaml vtab.task_groups keys)
# ---------------------------------------------------------------------------
VTAB_TASK_NAMES: List[str] = [
    # Natural group (7)
    "caltech101",
    "cifar100",
    "dtd",
    "flowers102",
    "pets",
    "sun397",
    "svhn",
    # Specialized group (4)
    "camelyon",
    "eurosat",
    "resisc45",
    "retinopathy",
    # Structured group (8)
    "clevr_count",
    "clevr_distance",
    "dmlab",
    "dsprites_loc",
    "dsprites_ori",
    "smallnorb_azimuth",
    "smallnorb_elevation",
    "kitti",
]

# ---------------------------------------------------------------------------
# Group membership (config.yaml: vtab.task_groups)
# Imported by evaluation/metrics.py for group-average computation.
# ---------------------------------------------------------------------------
VTAB_GROUPS: Dict[str, List[str]] = {
    "Natural": [
        "caltech101",
        "cifar100",
        "dtd",
        "flowers102",
        "pets",
        "sun397",
        "svhn",
    ],
    "Specialized": [
        "camelyon",
        "eurosat",
        "resisc45",
        "retinopathy",
    ],
    "Structured": [
        "clevr_count",
        "clevr_distance",
        "dmlab",
        "dsprites_loc",
        "dsprites_ori",
        "smallnorb_azimuth",
        "smallnorb_elevation",
        "kitti",
    ],
}

# ---------------------------------------------------------------------------
# Number of classes per task (config.yaml: vtab.num_classes)
# ---------------------------------------------------------------------------
VTAB_NUM_CLASSES: Dict[str, int] = {
    "caltech101": 102,
    "cifar100": 100,
    "dtd": 47,
    "flowers102": 102,
    "pets": 37,
    "sun397": 397,
    "svhn": 10,
    "camelyon": 2,
    "eurosat": 10,
    "resisc45": 45,
    "retinopathy": 5,
    "clevr_count": 8,
    "clevr_distance": 6,
    "dmlab": 6,
    "dsprites_loc": 16,
    "dsprites_ori": 16,
    "smallnorb_azimuth": 18,
    "smallnorb_elevation": 9,
    "kitti": 4,
}

# ---------------------------------------------------------------------------
# TensorFlow Datasets identifiers for each canonical task name.
# These must match the exact TFDS dataset/config strings.
# ---------------------------------------------------------------------------
VTAB_TFDS_NAMES: Dict[str, str] = {
    "caltech101": "caltech101",
    "cifar100": "cifar100",
    "dtd": "dtd",
    "flowers102": "oxford_flowers102",
    "pets": "oxford_iiit_pet",
    "sun397": "sun397",
    "svhn": "svhn_cropped",
    "camelyon": "patch_camelyon",
    "eurosat": "eurosat/rgb",
    "resisc45": "resisc45",
    "retinopathy": "diabetic_retinopathy_detection/btgraham-300",
    "clevr_count": "clevr/count_all",
    "clevr_distance": "clevr/closest_object_distance",
    "dmlab": "dmlab",
    "dsprites_loc": "dsprites/label_x_position",
    "dsprites_ori": "dsprites/label_orientation",
    "smallnorb_azimuth": "smallnorb/label_azimuth",
    "smallnorb_elevation": "smallnorb/label_elevation",
    "kitti": "kitti/closest_vehicle_distance",
}

# ---------------------------------------------------------------------------
# TFDS split strings for train and test per task.
# Default: train -> "train[:1000]", test -> "test".
# Exceptions are listed explicitly.
# ---------------------------------------------------------------------------
_DEFAULT_TRAIN_SPLIT: str = "train[:1000]"
_DEFAULT_TEST_SPLIT: str = "test"

VTAB_SPLIT_MAP: Dict[str, Dict[str, str]] = {
    # retinopathy has no "test" split in TFDS; use "validation" instead.
    "retinopathy": {"train": "train[:1000]", "test": "validation"},
    # sun397 uses a specific split structure in TFDS.
    "sun397": {"train": "train[:1000]", "test": "test"},
    # smallnorb uses "test" split.
    "smallnorb_azimuth": {"train": "train[:1000]", "test": "test"},
    "smallnorb_elevation": {"train": "train[:1000]", "test": "test"},
    # dsprites uses "train" for both (no separate test split in some versions).
    "dsprites_loc": {"train": "train[:1000]", "test": "train[90%:]"},
    "dsprites_ori": {"train": "train[:1000]", "test": "train[90%:]"},
    # dmlab uses "train" and "validation".
    "dmlab": {"train": "train[:1000]", "test": "validation"},
    # kitti uses "train" and "test".
    "kitti": {"train": "train[:1000]", "test": "test"},
    # camelyon uses "train" and "test".
    "camelyon": {"train": "train[:1000]", "test": "test"},
}

# ---------------------------------------------------------------------------
# Feature key overrides for datasets that do not use the standard
# 'image' / 'label' keys in TFDS.
# ---------------------------------------------------------------------------
VTAB_IMAGE_KEY: Dict[str, str] = {
    # All standard VTAB tasks use 'image'.
    # Override here if any task uses a different key.
}

VTAB_LABEL_KEY: Dict[str, str] = {
    # Most tasks use 'label'; dSprites and smallNORB use task-specific keys.
    "dsprites_loc": "label_x_position",
    "dsprites_ori": "label_orientation",
    "smallnorb_azimuth": "label_azimuth",
    "smallnorb_elevation": "label_elevation",
    # All others default to 'label'.
}

# ---------------------------------------------------------------------------
# Datasets that require manual download (licensing restrictions).
# ---------------------------------------------------------------------------
_MANUAL_DOWNLOAD_DATASETS: Dict[str, str] = {
    "retinopathy": (
        "diabetic_retinopathy_detection requires manual download. "
        "See: https://www.tensorflow.org/datasets/catalog/diabetic_retinopathy_detection"
    ),
    "resisc45": (
        "resisc45 requires manual download. "
        "See: https://www.tensorflow.org/datasets/catalog/resisc45"
    ),
}


# ---------------------------------------------------------------------------
# Internal helper: TorchVTABDataset
# ---------------------------------------------------------------------------

class TorchVTABDataset(Dataset):
    """PyTorch Dataset wrapping pre-loaded VTAB images and labels.

    Stores images as PIL Images so that torchvision transforms can be applied
    in the standard PyTorch way. Labels are returned as torch.long tensors.

    Attributes:
        images: List of PIL Images (RGB, converted from TFDS uint8 arrays).
        labels: List of integer class indices (0-indexed).
        transform: Optional torchvision transform applied in __getitem__.
    """

    def __init__(
        self,
        images: List[Image.Image],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        """Initialises the dataset with pre-loaded images and labels.

        Args:
            images: List of PIL Images. All images must already be in RGB mode.
            labels: List of integer class indices, one per image.
            transform: Optional torchvision transform pipeline applied to each
                image in __getitem__. If None, raw PIL Images are returned.
        """
        if len(images) != len(labels):
            raise ValueError(
                f"images and labels must have the same length, "
                f"got {len(images)} images and {len(labels)} labels."
            )
        self.images: List[Image.Image] = images
        self.labels: List[int] = labels
        self.transform: Optional[transforms.Compose] = transform

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the (image_tensor, label_tensor) pair at index idx.

        Args:
            idx: Integer index into the dataset.

        Returns:
            A tuple (image, label) where:
            - image is a float tensor of shape (3, H, W) after transform, or
              a PIL Image if no transform is set.
            - label is a scalar torch.long tensor.
        """
        image: Image.Image = self.images[idx]
        label: torch.Tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform is not None:
            image_tensor: torch.Tensor = self.transform(image)
            return image_tensor, label

        # No transform: return PIL Image (unusual but supported for debugging).
        return image, label  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public API: VTABLoader
# ---------------------------------------------------------------------------

class VTABLoader:
    """Data loader for all 19 VTAB-1K classification tasks.

    Handles TFDS-to-PyTorch conversion, deterministic 80/20 train/val
    splitting, and standard ViT preprocessing (no augmentation, ImageNet
    normalization). Designed to be instantiated once per task per experiment.

    Attributes:
        dataset_name: Canonical VTAB task name (one of VTAB_TASK_NAMES).
        data_dir: Root directory for TFDS data cache.
        batch_size: Number of samples per DataLoader batch.
        image_size: Target spatial resolution (default 224 for ViT-B/16).
        num_workers: Number of DataLoader worker processes.
        tfds_name: TFDS dataset identifier string.
        n_classes: Number of output classes for this task.
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

        Does NOT load data — deferred to get_* methods to avoid loading all
        19 datasets simultaneously during hyperparameter search.

        Args:
            dataset_name: Canonical VTAB task name. Must be one of
                VTAB_TASK_NAMES. Example: "dtd", "clevr_count".
            data_dir: Root directory for TFDS data cache. TFDS will download
                and cache datasets here on first run.
            batch_size: Batch size for all DataLoaders. Default: 64
                (config.yaml: vtab.training.batch_size).
            image_size: Target image resolution. Default: 224
                (config.yaml: image_size).
            num_workers: Number of DataLoader worker processes. Default: 4
                (config.yaml: num_workers).

        Raises:
            ValueError: If dataset_name is not in VTAB_TASK_NAMES.
        """
        if dataset_name not in VTAB_TASK_NAMES:
            raise ValueError(
                f"Unknown VTAB dataset: '{dataset_name}'. "
                f"Valid names are: {VTAB_TASK_NAMES}"
            )

        self.dataset_name: str = dataset_name
        self.data_dir: str = data_dir
        self.batch_size: int = batch_size
        self.image_size: int = image_size
        self.num_workers: int = num_workers
        self.tfds_name: str = VTAB_TFDS_NAMES[dataset_name]
        self.n_classes: int = VTAB_NUM_CLASSES[dataset_name]

        _logger.info(
            "VTABLoader initialised: dataset=%s, tfds_name=%s, "
            "num_classes=%d, batch_size=%d, image_size=%d",
            self.dataset_name,
            self.tfds_name,
            self.n_classes,
            self.batch_size,
            self.image_size,
        )

    # ------------------------------------------------------------------
    # Public DataLoader methods
    # ------------------------------------------------------------------

    def get_train_val_loaders(
        self, val_ratio: float = 0.2
    ) -> Tuple[DataLoader, DataLoader]:
        """Returns DataLoaders for the 80/20 hyperparameter tuning split.

        Loads the 1000 VTAB training samples, applies a deterministic split
        using seed=42 (config.yaml: seed), and returns separate DataLoaders
        for the 800-sample training subset and 200-sample validation subset.
        No data augmentation is applied to either split (paper Section 3).

        Args:
            val_ratio: Fraction of training samples to use for validation.
                Default: 0.2 (config.yaml: vtab.val_ratio), giving 800/200.

        Returns:
            A tuple (train_loader, val_loader) where:
            - train_loader: DataLoader over 800 training samples, shuffled.
            - val_loader: DataLoader over 200 validation samples, not shuffled.
        """
        # Load full 1000-sample training set with transforms applied.
        full_dataset: TorchVTABDataset = self._load_tfds_as_torch(
            split_type="train", transform=self._get_base_transform()
        )

        total_size: int = len(full_dataset)
        val_size: int = int(total_size * val_ratio)
        train_size: int = total_size - val_size

        _logger.info(
            "Splitting %d training samples: %d train / %d val (seed=42)",
            total_size,
            train_size,
            val_size,
        )

        # Deterministic split — seed=42 from config.yaml.
        generator: torch.Generator = torch.Generator()
        generator.manual_seed(42)

        train_subset: Subset
        val_subset: Subset
        train_subset, val_subset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=generator,
        )

        train_loader: DataLoader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        val_loader: DataLoader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        return train_loader, val_loader

    def get_test_loader(self) -> DataLoader:
        """Returns a DataLoader over the full VTAB test split.

        The paper reports: "The reported TOP-1 ACCURACY is obtained after
        training over the 1000 images and evaluating on the original test set."

        Returns:
            DataLoader over the full test set, not shuffled, with base
            transform (resize + crop + normalize, no augmentation).
        """
        test_dataset: TorchVTABDataset = self._load_tfds_as_torch(
            split_type="test", transform=self._get_base_transform()
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
            len(test_dataset),
            self.dataset_name,
        )

        return test_loader

    def get_full_train_loader(self) -> DataLoader:
        """Returns a DataLoader over all 1000 training samples.

        Used for final training after hyperparameter search is complete.
        The paper states: "The reported TOP-1 ACCURACY is obtained after
        training over the 1000 images."

        Returns:
            DataLoader over all 1000 training samples, shuffled, with base
            transform (no augmentation).
        """
        full_dataset: TorchVTABDataset = self._load_tfds_as_torch(
            split_type="train", transform=self._get_base_transform()
        )

        full_loader: DataLoader = DataLoader(
            full_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        _logger.info(
            "Full train loader created: %d samples for dataset '%s'",
            len(full_dataset),
            self.dataset_name,
        )

        return full_loader

    def num_classes(self) -> int:
        """Returns the number of output classes for this VTAB task.

        Returns:
            Integer number of classes from VTAB_NUM_CLASSES.
        """
        return self.n_classes

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_base_transform(self) -> transforms.Compose:
        """Builds the standard ViT preprocessing pipeline.

        No data augmentation is applied — consistent with the paper:
        "Consistent with the original VTAB-1k paper, most PEFT studies
        don't apply data augmentation... we don't apply data augmentation."

        The pipeline:
        1. Resize shorter edge to image_size (maintains aspect ratio).
        2. CenterCrop to image_size × image_size.
        3. ToTensor: converts PIL [0,255] uint8 to float [0,1].
        4. Normalize with ImageNet mean and std.

        Returns:
            torchvision.transforms.Compose pipeline.
        """
        return transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                ),
            ]
        )

    def _get_tfds_split_string(self, split_type: str) -> str:
        """Returns the TFDS split string for the given split type.

        Looks up task-specific overrides in VTAB_SPLIT_MAP, falling back to
        the default train/test split strings.

        Args:
            split_type: Either "train" or "test".

        Returns:
            TFDS split string, e.g. "train[:1000]" or "validation".
        """
        if self.dataset_name in VTAB_SPLIT_MAP:
            return VTAB_SPLIT_MAP[self.dataset_name][split_type]

        if split_type == "train":
            return _DEFAULT_TRAIN_SPLIT
        else:
            return _DEFAULT_TEST_SPLIT

    def _load_tfds_as_torch(
        self,
        split_type: str,
        transform: Optional[transforms.Compose] = None,
    ) -> TorchVTABDataset:
        """Loads a VTAB task split from TFDS and converts to a PyTorch Dataset.

        Handles the full TFDS-to-PIL-to-TorchVTABDataset pipeline:
        1. Disables TF GPU to prevent conflicts with PyTorch.
        2. Calls tfds.load with the appropriate split string.
        3. Iterates the TFDS dataset, converting each sample to PIL Image + int label.
        4. Ensures all images are RGB (handles grayscale datasets).
        5. Returns a TorchVTABDataset with the given transform.

        Args:
            split_type: Either "train" (loads up to 1000 samples) or "test"
                (loads the full test split).
            transform: Optional torchvision transform to attach to the dataset.

        Returns:
            TorchVTABDataset with PIL Images and integer labels.

        Raises:
            RuntimeError: If TFDS loading fails (e.g., dataset not downloaded).
            ValueError: If the loaded dataset has fewer samples than expected.
        """
        # ------------------------------------------------------------------
        # Step 1: Import TensorFlow and disable GPU to avoid PyTorch conflict.
        # ------------------------------------------------------------------
        try:
            import tensorflow as tf  # type: ignore
            import tensorflow_datasets as tfds  # type: ignore

            # Critical: prevent TF from claiming the GPU.
            tf.config.set_visible_devices([], "GPU")
        except ImportError as exc:
            raise ImportError(
                "tensorflow and tensorflow-datasets are required for VTAB-1K loading. "
                "Install with: pip install tensorflow tensorflow-datasets"
            ) from exc

        # ------------------------------------------------------------------
        # Step 2: Resolve TFDS split string.
        # ------------------------------------------------------------------
        split_string: str = self._get_tfds_split_string(split_type)
        _logger.info(
            "Loading TFDS dataset '%s' split '%s' (task: %s)",
            self.tfds_name,
            split_string,
            self.dataset_name,
        )

        # ------------------------------------------------------------------
        # Step 3: Load from TFDS.
        # ------------------------------------------------------------------
        try:
            tf_dataset = tfds.load(
                self.tfds_name,
                split=split_string,
                data_dir=self.data_dir,
                as_supervised=False,
                with_info=False,
                shuffle_files=False,  # Deterministic ordering.
            )
        except Exception as exc:  # pylint: disable=broad-except
            # Provide helpful error messages for common failure modes.
            manual_msg: str = _MANUAL_DOWNLOAD_DATASETS.get(self.dataset_name, "")
            if manual_msg:
                raise RuntimeError(
                    f"Failed to load VTAB dataset '{self.dataset_name}': {exc}\n"
                    f"This dataset requires manual download: {manual_msg}"
                ) from exc
            raise RuntimeError(
                f"Failed to load VTAB dataset '{self.dataset_name}' "
                f"(tfds_name='{self.tfds_name}', split='{split_string}'): {exc}\n"
                f"Ensure the dataset is downloaded to data_dir='{self.data_dir}'."
            ) from exc

        # ------------------------------------------------------------------
        # Step 4: Determine feature keys for this dataset.
        # ------------------------------------------------------------------
        image_key: str = VTAB_IMAGE_KEY.get(self.dataset_name, "image")
        label_key: str = VTAB_LABEL_KEY.get(self.dataset_name, "label")

        # ------------------------------------------------------------------
        # Step 5: Iterate TFDS dataset and convert to PIL Images + int labels.
        # ------------------------------------------------------------------
        images: List[Image.Image] = []
        labels: List[int] = []

        for sample in tf_dataset:
            # Extract image array (uint8, shape H×W×C or H×W).
            img_array: np.ndarray = sample[image_key].numpy()

            # Convert to PIL Image and ensure RGB (3 channels).
            if img_array.ndim == 2:
                # Grayscale (H, W) — e.g., dSprites.
                pil_image: Image.Image = Image.fromarray(img_array, mode="L").convert("RGB")
            elif img_array.ndim == 3 and img_array.shape[2] == 1:
                # Single-channel (H, W, 1) — e.g., smallNORB.
                pil_image = Image.fromarray(img_array[:, :, 0], mode="L").convert("RGB")
            elif img_array.ndim == 3 and img_array.shape[2] == 3:
                # Standard RGB (H, W, 3).
                pil_image = Image.fromarray(img_array, mode="RGB")
            elif img_array.ndim == 3 and img_array.shape[2] == 4:
                # RGBA — convert to RGB.
                pil_image = Image.fromarray(img_array, mode="RGBA").convert("RGB")
            else:
                # Fallback: let PIL infer the mode.
                pil_image = Image.fromarray(img_array).convert("RGB")

            images.append(pil_image)

            # Extract label as Python int.
            label_value = sample[label_key].numpy()
            labels.append(int(label_value))

        # ------------------------------------------------------------------
        # Step 6: Validate loaded data.
        # ------------------------------------------------------------------
        num_loaded: int = len(images)
        _logger.info(
            "Loaded %d samples from '%s' split '%s'",
            num_loaded,
            self.tfds_name,
            split_string,
        )

        if split_type == "train" and num_loaded < VTAB_TRAIN_SIZE:
            _logger.warning(
                "Expected at least %d training samples for VTAB task '%s', "
                "but only loaded %d. Results may not match the paper.",
                VTAB_TRAIN_SIZE,
                self.dataset_name,
                num_loaded,
            )

        # ------------------------------------------------------------------
        # Step 7: Validate label range.
        # ------------------------------------------------------------------
        if labels:
            min_label: int = min(labels)
            max_label: int = max(labels)
            expected_max: int = self.n_classes - 1

            if min_label < 0:
                _logger.warning(
                    "Negative labels found in dataset '%s' (min=%d). "
                    "Labels may need adjustment.",
                    self.dataset_name,
                    min_label,
                )
            elif min_label == 1 and max_label == self.n_classes:
                # 1-indexed labels — convert to 0-indexed.
                _logger.info(
                    "Dataset '%s' has 1-indexed labels [1, %d]; "
                    "converting to 0-indexed [0, %d].",
                    self.dataset_name,
                    self.n_classes,
                    expected_max,
                )
                labels = [lbl - 1 for lbl in labels]
            elif max_label > expected_max:
                _logger.warning(
                    "Dataset '%s': max label %d exceeds expected max %d "
                    "(num_classes=%d). Check VTAB_NUM_CLASSES.",
                    self.dataset_name,
                    max_label,
                    expected_max,
                    self.n_classes,
                )

        # ------------------------------------------------------------------
        # Step 8: Construct and return TorchVTABDataset.
        # ------------------------------------------------------------------
        return TorchVTABDataset(images=images, labels=labels, transform=transform)
