## datasets/imagenet_loader.py
"""ImageNet data loader for the PEFT Visual Recognition robustness experiment.

This module provides the ImageNetLoader class for loading ImageNet-1K and its
four distribution-shifted variants used in Section 7 of the paper:

    "How Robust are PEFT Methods to Distribution Shifts?"

The loader supports:
- 100-shot ImageNet-1K training set with strong augmentation (for CLIP fine-tuning)
- Standard ImageNet-1K validation set (target distribution evaluation)
- ImageNet-V2, ImageNet-R, ImageNet-S, ImageNet-A (distribution shift evaluation)

Paper Section 7: "We use 100-shot ImageNet-1K as our target distribution, with
each class containing 100 images. Following [96], we consider 4 natural
distribution shifts from ImageNet: ImageNet-V2, ImageNet-R, ImageNet-S,
ImageNet-A."

Paper Appendix A.1 (Robustness Setup): "we set a small learning rate as 3e-5
and weight decay as 5e-3. We use a strong data augmentation following [107]."

All datasets use ImageNet normalization (Appendix A.2).

Typical usage:
    loader = ImageNetLoader(
        data_dir="./data/imagenet_robustness",
        batch_size=64,
        num_shots=100,
        num_workers=4,
    )
    train_loader = loader.get_train_loader()
    imagenet_test_loader = loader.get_test_loader("imagenet")
    shift_loaders = loader.get_all_shift_loaders()
    n_cls = loader.num_classes()
"""

import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from datasets.vtab_loader import IMAGENET_MEAN, IMAGENET_STD

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapping from canonical split name to expected subdirectory under data_dir.
# All five directories are expected to follow the standard ImageFolder layout:
#   <split_dir>/<class_name>/<image_file>
# For ImageNet-1K, the val set lives under <split_dir>/val/.
# For shift datasets (V2, R, S, A), the root IS the ImageFolder root.
# ---------------------------------------------------------------------------
SHIFT_DATASET_DIRS: Dict[str, str] = {
    "imagenet": "imagenet",        # Standard ImageNet-1K (train + val)
    "imagenet_v2": "imagenet_v2",  # ImageNetV2 matched-frequency
    "imagenet_r": "imagenet_r",    # ImageNet-R (renditions, 200 classes)
    "imagenet_s": "imagenet_s",    # ImageNet-Sketch (1000 classes)
    "imagenet_a": "imagenet_a",    # ImageNet-A (adversarial, 200 classes)
}

# ---------------------------------------------------------------------------
# Valid split names for validation in get_test_loader().
# ---------------------------------------------------------------------------
VALID_SPLITS: set = set(SHIFT_DATASET_DIRS.keys())

# ---------------------------------------------------------------------------
# ImageNet-1K has 1000 classes.
# ---------------------------------------------------------------------------
IMAGENET_NUM_CLASSES: int = 1000

# ---------------------------------------------------------------------------
# Fixed random seed for reproducible n-shot sampling (config.yaml: seed: 42).
# ---------------------------------------------------------------------------
_SAMPLING_SEED: int = 42


# ---------------------------------------------------------------------------
# Private helper: _TransformSubset
# ---------------------------------------------------------------------------

class _TransformSubset(Dataset):
    """Thin wrapper that applies a transform to a torch.utils.data.Subset.

    This avoids mutating the parent dataset's transform attribute, which
    could cause unintended side effects if the parent dataset is reused
    (e.g., when building both train and val loaders from the same base).

    Attributes:
        subset: The underlying Subset (or any Dataset) providing raw
            (PIL Image, int label) pairs.
        transform: torchvision transform pipeline applied in __getitem__.
    """

    def __init__(
        self,
        subset: Dataset,
        transform: transforms.Compose,
    ) -> None:
        """Initialises the wrapper.

        Args:
            subset: Underlying dataset returning (PIL Image, int label) pairs.
                Typically a torch.utils.data.Subset wrapping an ImageFolder
                with transform=None.
            transform: Transform pipeline to apply to each image.
        """
        self.subset: Dataset = subset
        self.transform: transforms.Compose = transform

    def __len__(self) -> int:
        """Returns the number of samples in the wrapped subset."""
        return len(self.subset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (transformed_image_tensor, label_tensor) at index idx.

        Args:
            idx: Integer index into the subset.

        Returns:
            Tuple of:
            - image_tensor: Float tensor of shape (3, H, W) after transform.
            - label_tensor: Scalar torch.long tensor.
        """
        image, label = self.subset[idx]

        # Apply transform to PIL Image.
        image_tensor: torch.Tensor = self.transform(image)

        label_tensor: torch.Tensor = (
            label
            if isinstance(label, torch.Tensor)
            else torch.tensor(label, dtype=torch.long)
        )
        return image_tensor, label_tensor


# ---------------------------------------------------------------------------
# Public API: ImageNetLoader
# ---------------------------------------------------------------------------

class ImageNetLoader:
    """Data loader for ImageNet-1K and its distribution-shifted variants.

    Supports the robustness experiment (Section 7) of the paper:
    - 100-shot ImageNet-1K training with strong augmentation for CLIP fine-tuning.
    - Standard evaluation on ImageNet-1K val set (target distribution).
    - Evaluation on ImageNet-V2, R, S, A (distribution shift datasets).

    All datasets are loaded as torchvision ImageFolder instances, which
    require the standard directory layout:
        <data_dir>/<split_name>/<class_name>/<image_file>

    For ImageNet-1K specifically:
        <data_dir>/imagenet/train/<class_name>/<image_file>
        <data_dir>/imagenet/val/<class_name>/<image_file>

    For shift datasets (V2, R, S, A):
        <data_dir>/imagenet_v2/<class_name>/<image_file>
        <data_dir>/imagenet_r/<class_name>/<image_file>
        <data_dir>/imagenet_s/<class_name>/<image_file>
        <data_dir>/imagenet_a/<class_name>/<image_file>

    Note on ImageNet-R and ImageNet-A (200-class subsets):
        These datasets contain only 200 of the 1000 ImageNet classes.
        ImageFolder assigns class indices 0–199 (alphabetical order of the
        200 synset folders), NOT the original 0–999 ImageNet indices.
        Accuracy computation in evaluation/metrics.py must handle this
        remapping when comparing against a 1000-class model output.

    Attributes:
        data_dir: Root directory containing subdirectories for each split.
        batch_size: Number of samples per DataLoader batch.
        num_shots: Number of images per class for the training set.
        num_workers: Number of DataLoader worker processes.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_shots: int = 100,
        num_workers: int = 4,
    ) -> None:
        """Initialises the loader and validates the data directory.

        Does NOT load any dataset — loading is deferred to the getter methods
        to avoid loading all splits simultaneously.

        Args:
            data_dir: Root directory containing subdirectories for each split.
                Must exist at construction time. Example: "./data/imagenet_robustness".
            batch_size: Batch size for all DataLoaders. Default: 64
                (config.yaml: robustness.training.batch_size).
            num_shots: Number of images per class for the training set.
                Default: 100 (config.yaml: robustness.num_shots).
                Paper: "100-shot ImageNet-1K ... each class containing 100 images."
            num_workers: Number of DataLoader worker processes. Default: 4
                (config.yaml: num_workers).

        Raises:
            FileNotFoundError: If data_dir does not exist.
        """
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"ImageNet data directory not found: '{data_dir}'. "
                "Please ensure the directory exists and contains the expected "
                "subdirectories: imagenet/, imagenet_v2/, imagenet_r/, "
                "imagenet_s/, imagenet_a/."
            )

        self.data_dir: str = data_dir
        self.batch_size: int = batch_size
        self.num_shots: int = num_shots
        self.num_workers: int = num_workers

        _logger.info(
            "ImageNetLoader initialised: data_dir=%s, batch_size=%d, "
            "num_shots=%d, num_workers=%d",
            self.data_dir,
            self.batch_size,
            self.num_shots,
            self.num_workers,
        )

    # ------------------------------------------------------------------
    # Public DataLoader methods
    # ------------------------------------------------------------------

    def get_train_loader(self) -> DataLoader:
        """Returns a DataLoader for the 100-shot ImageNet-1K training set.

        Loads the full ImageNet-1K training set, samples exactly num_shots
        images per class using stratified sampling with a fixed seed (42),
        and applies the strong augmentation pipeline from config.yaml
        (robustness.strong_augmentation).

        Paper Section 7: "We use 100-shot ImageNet-1K as our target
        distribution, with each class containing 100 images."
        Paper Appendix A.1: "We use a strong data augmentation following [107]."

        Strong augmentation pipeline (config.yaml: robustness.strong_augmentation):
            RandomResizedCrop(224, scale=(0.08, 1.0))
            RandomHorizontalFlip()
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            RandomGrayscale(p=0.2)
            ToTensor()
            Normalize(IMAGENET_MEAN, IMAGENET_STD)

        Returns:
            DataLoader over the 100-shot training subset (num_shots × 1000
            samples), shuffled, with strong augmentation applied.

        Raises:
            FileNotFoundError: If the ImageNet training directory does not exist.
        """
        # ------------------------------------------------------------------
        # Step 1: Resolve the training directory path.
        # ------------------------------------------------------------------
        train_dir: str = os.path.join(
            self.data_dir, SHIFT_DATASET_DIRS["imagenet"], "train"
        )

        if not os.path.isdir(train_dir):
            raise FileNotFoundError(
                f"ImageNet-1K training directory not found: '{train_dir}'. "
                "Expected layout: <data_dir>/imagenet/train/<class_name>/<image>."
            )

        # ------------------------------------------------------------------
        # Step 2: Load the full training set WITHOUT transforms.
        # Transforms are applied after n-shot sampling via _TransformSubset.
        # Using transform=None here avoids applying transforms during the
        # sampling step, which only needs dataset.targets.
        # ------------------------------------------------------------------
        _logger.info(
            "Loading ImageNet-1K training set from '%s' (transform=None for sampling).",
            train_dir,
        )
        full_dataset: ImageFolder = ImageFolder(root=train_dir, transform=None)

        _logger.info(
            "ImageNet-1K training set loaded: %d total samples, %d classes.",
            len(full_dataset),
            len(full_dataset.classes),
        )

        # ------------------------------------------------------------------
        # Step 3: Sample num_shots images per class.
        # ------------------------------------------------------------------
        n_shot_subset: Subset = self._sample_n_shot(full_dataset, self.num_shots)

        _logger.info(
            "100-shot subset created: %d samples (%d shots × %d classes).",
            len(n_shot_subset),
            self.num_shots,
            len(full_dataset.classes),
        )

        # ------------------------------------------------------------------
        # Step 4: Build the strong augmentation transform.
        # Parameters from config.yaml: robustness.strong_augmentation.
        # ------------------------------------------------------------------
        strong_transform: transforms.Compose = self._get_strong_augmentation()

        # ------------------------------------------------------------------
        # Step 5: Wrap the subset with the strong transform.
        # ------------------------------------------------------------------
        train_dataset: _TransformSubset = _TransformSubset(
            subset=n_shot_subset,
            transform=strong_transform,
        )

        # ------------------------------------------------------------------
        # Step 6: Build and return the DataLoader.
        # ------------------------------------------------------------------
        train_loader: DataLoader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        _logger.info(
            "Train loader created: %d samples, batch_size=%d, shuffle=True.",
            len(train_dataset),
            self.batch_size,
        )

        return train_loader

    def get_test_loader(self, split: str = "imagenet") -> DataLoader:
        """Returns a DataLoader for a given evaluation split.

        Applies the standard inference transform pipeline (no augmentation):
            Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize

        This is the standard ImageNet evaluation pipeline used consistently
        across all five evaluation splits.

        Args:
            split: One of 'imagenet', 'imagenet_v2', 'imagenet_r',
                'imagenet_s', 'imagenet_a'. Default: 'imagenet'.
                - 'imagenet': Standard ImageNet-1K validation set.
                - 'imagenet_v2': ImageNetV2 matched-frequency test set.
                  Paper: "a new ImageNet test set collected with the original
                  labeling protocol."
                - 'imagenet_r': Renditions for 200 ImageNet classes.
                  Paper: "renditions for 200 ImageNet classes."
                - 'imagenet_s': Sketch images for 1K ImageNet classes.
                  Paper: "sketch images for 1K ImageNet classes."
                - 'imagenet_a': Natural adversarial examples for 200 classes.
                  Paper: "a test set of natural images misclassified by a
                  ImageNet pre-trained ResNet-50 for 200 ImageNet classes."

        Returns:
            DataLoader over the evaluation split, not shuffled, with standard
            inference transforms applied.

        Raises:
            ValueError: If split is not in VALID_SPLITS.
            FileNotFoundError: If the split directory does not exist.
        """
        # ------------------------------------------------------------------
        # Step 1: Validate split name.
        # ------------------------------------------------------------------
        if split not in VALID_SPLITS:
            raise ValueError(
                f"Unknown split: '{split}'. "
                f"Valid splits are: {sorted(VALID_SPLITS)}"
            )

        # ------------------------------------------------------------------
        # Step 2: Resolve the split directory path.
        # For ImageNet-1K, the val set lives under <split_dir>/val/.
        # For shift datasets, the root IS the ImageFolder root.
        # ------------------------------------------------------------------
        split_dir: str = os.path.join(self.data_dir, SHIFT_DATASET_DIRS[split])

        # Check for a 'val' subdirectory (standard ImageNet-1K layout).
        val_subdir: str = os.path.join(split_dir, "val")
        if os.path.isdir(val_subdir):
            split_dir = val_subdir
            _logger.debug(
                "Found 'val' subdirectory for split '%s'; using '%s'.",
                split,
                split_dir,
            )

        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Evaluation split directory not found: '{split_dir}'. "
                f"Expected layout for split='{split}':\n"
                f"  {os.path.join(self.data_dir, SHIFT_DATASET_DIRS[split])}/"
                f"<class_name>/<image_file>\n"
                "Please download the dataset and organize it in ImageFolder format."
            )

        # ------------------------------------------------------------------
        # Step 3: Build the standard inference transform.
        # ------------------------------------------------------------------
        inference_transform: transforms.Compose = self._get_inference_transform()

        # ------------------------------------------------------------------
        # Step 4: Load the dataset.
        # ------------------------------------------------------------------
        _logger.info(
            "Loading evaluation split '%s' from '%s'.",
            split,
            split_dir,
        )

        try:
            dataset: ImageFolder = ImageFolder(
                root=split_dir,
                transform=inference_transform,
            )
        except Exception as exc:  # pylint: disable=broad-except
            raise FileNotFoundError(
                f"Failed to load ImageFolder from '{split_dir}' for split='{split}': {exc}\n"
                "Ensure the directory contains class subdirectories with images."
            ) from exc

        _logger.info(
            "Evaluation split '%s' loaded: %d samples, %d classes.",
            split,
            len(dataset),
            len(dataset.classes),
        )

        # ------------------------------------------------------------------
        # Step 5: Build and return the DataLoader.
        # ------------------------------------------------------------------
        test_loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        _logger.info(
            "Test loader created for split '%s': %d samples, batch_size=%d, shuffle=False.",
            split,
            len(dataset),
            self.batch_size,
        )

        return test_loader

    def get_all_shift_loaders(self) -> Dict[str, DataLoader]:
        """Returns DataLoaders for all four distribution-shifted datasets.

        Excludes the target distribution ('imagenet') and returns only the
        four shift datasets. This dict is consumed directly by
        training/wise.py's sweep() method.

        Paper Section 7: "we consider 4 natural distribution shifts from
        ImageNet: ImageNet-V2, ImageNet-R, ImageNet-S, ImageNet-A."
        Config: robustness.shift_datasets (excludes 'imagenet').

        Returns:
            Dict mapping split name to DataLoader:
            {
                'imagenet_v2': DataLoader,
                'imagenet_r':  DataLoader,
                'imagenet_s':  DataLoader,
                'imagenet_a':  DataLoader,
            }
            Each DataLoader uses standard inference transforms (no augmentation),
            not shuffled.

        Note:
            If a shift dataset directory does not exist, a FileNotFoundError
            is raised by get_test_loader(). Callers should ensure all four
            shift dataset directories are present before calling this method.
        """
        shift_splits: List[str] = [
            "imagenet_v2",
            "imagenet_r",
            "imagenet_s",
            "imagenet_a",
        ]

        shift_loaders: Dict[str, DataLoader] = {}

        for split_name in shift_splits:
            _logger.info("Building shift loader for split '%s'.", split_name)
            try:
                shift_loaders[split_name] = self.get_test_loader(split=split_name)
            except FileNotFoundError as exc:
                _logger.warning(
                    "Shift dataset '%s' not found: %s. "
                    "Skipping this split. Downstream WiSE sweep will be incomplete.",
                    split_name,
                    exc,
                )
                # Do not re-raise — allow partial evaluation if some shift
                # datasets are unavailable. The caller can check which keys
                # are present in the returned dict.

        _logger.info(
            "Shift loaders built for %d / %d splits: %s",
            len(shift_loaders),
            len(shift_splits),
            list(shift_loaders.keys()),
        )

        return shift_loaders

    def num_classes(self) -> int:
        """Returns the number of ImageNet-1K classes.

        The robustness experiment always uses 1000 classes (ImageNet-1K).
        The CLIP zero-shot head is initialized with 1000-class text embeddings.

        Note: ImageNet-R and ImageNet-A contain only 200 of the 1000 classes.
        Their ImageFolder instances will have 200 classes (indices 0–199),
        not 1000. Accuracy computation must account for this remapping.

        Returns:
            1000 (the number of ImageNet-1K classes).
        """
        return IMAGENET_NUM_CLASSES

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------

    def _sample_n_shot(
        self,
        dataset: ImageFolder,
        n: int,
    ) -> Subset:
        """Stratified sampling: selects exactly n images per class.

        Uses a fixed random seed (42, from config.yaml: seed) for
        reproducibility. If a class has fewer than n images, all available
        images for that class are selected (no replacement).

        Paper Section 7: "100-shot ImageNet-1K ... each class containing
        100 images." → 100 shots × 1000 classes = 100,000 training samples.

        Args:
            dataset: An ImageFolder dataset with a .targets attribute
                (list of int class indices, one per sample) and a
                .class_to_idx attribute (dict mapping class name to index).
                The dataset should have transform=None so that sampling
                operates on the raw dataset structure.
            n: Number of images to sample per class. Default: self.num_shots.

        Returns:
            torch.utils.data.Subset containing the selected indices.
            The Subset preserves the parent dataset's transform (None at
            this point; the caller applies transforms via _TransformSubset).
        """
        # ------------------------------------------------------------------
        # Step 1: Group sample indices by class.
        # dataset.targets is a list of int class indices, one per sample.
        # ------------------------------------------------------------------
        class_indices: Dict[int, List[int]] = defaultdict(list)

        for sample_idx, target in enumerate(dataset.targets):
            class_indices[int(target)].append(sample_idx)

        num_classes_found: int = len(class_indices)
        _logger.info(
            "Stratified sampling: %d classes found, selecting %d shots per class.",
            num_classes_found,
            n,
        )

        # ------------------------------------------------------------------
        # Step 2: Sample n indices per class with a fixed seed.
        # Using numpy.random.RandomState (local) to avoid global seed side effects.
        # ------------------------------------------------------------------
        rng: np.random.RandomState = np.random.RandomState(seed=_SAMPLING_SEED)

        selected_indices: List[int] = []
        num_classes_with_fewer: int = 0

        for class_idx in sorted(class_indices.keys()):
            indices: List[int] = class_indices[class_idx]

            if len(indices) >= n:
                # Standard case: sample n without replacement.
                chosen: np.ndarray = rng.choice(indices, size=n, replace=False)
                selected_indices.extend(chosen.tolist())
            else:
                # Edge case: fewer than n samples — take all available.
                selected_indices.extend(indices)
                num_classes_with_fewer += 1
                _logger.debug(
                    "Class %d has only %d samples (< %d shots); taking all.",
                    class_idx,
                    len(indices),
                    n,
                )

        if num_classes_with_fewer > 0:
            _logger.warning(
                "%d class(es) had fewer than %d samples; all available "
                "samples were selected for those classes.",
                num_classes_with_fewer,
                n,
            )

        total_selected: int = len(selected_indices)
        _logger.info(
            "Stratified sampling complete: %d total samples selected "
            "(%d classes × up to %d shots).",
            total_selected,
            num_classes_found,
            n,
        )

        return Subset(dataset, selected_indices)

    def _get_strong_augmentation(self) -> transforms.Compose:
        """Builds the strong augmentation transform for CLIP fine-tuning.

        Parameters are taken directly from config.yaml:
        robustness.strong_augmentation. This pipeline follows [107]
        (Neural Prompt Search) as cited in the paper.

        Pipeline:
            RandomResizedCrop(224, scale=(0.08, 1.0))  — config: scale [0.08, 1.0]
            RandomHorizontalFlip()                      — config: random_horizontal_flip: true
            ColorJitter(0.4, 0.4, 0.4, 0.1)            — config: brightness/contrast/saturation/hue
            RandomGrayscale(p=0.2)                      — config: random_grayscale: 0.2
            ToTensor()
            Normalize(IMAGENET_MEAN, IMAGENET_STD)

        Returns:
            torchvision.transforms.Compose pipeline with strong augmentation.
        """
        return transforms.Compose(
            [
                # config.yaml: robustness.strong_augmentation.random_resized_crop
                # size: 224, scale: [0.08, 1.0]
                transforms.RandomResizedCrop(
                    size=224,
                    scale=(0.08, 1.0),
                ),
                # config.yaml: robustness.strong_augmentation.random_horizontal_flip: true
                transforms.RandomHorizontalFlip(),
                # config.yaml: robustness.strong_augmentation.color_jitter
                # brightness: 0.4, contrast: 0.4, saturation: 0.4, hue: 0.1
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1,
                ),
                # config.yaml: robustness.strong_augmentation.random_grayscale: 0.2
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                ),
            ]
        )

    def _get_inference_transform(self) -> transforms.Compose:
        """Builds the standard inference transform for evaluation splits.

        No augmentation is applied. This is the standard ImageNet evaluation
        pipeline used consistently across all five evaluation splits
        (imagenet, imagenet_v2, imagenet_r, imagenet_s, imagenet_a).

        Pipeline:
            Resize(256)                    — resize shorter edge to 256
            CenterCrop(224)                — center crop to 224×224
            ToTensor()                     — convert PIL [0,255] to float [0,1]
            Normalize(IMAGENET_MEAN, STD)  — ImageNet normalization

        Returns:
            torchvision.transforms.Compose pipeline for inference.
        """
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                ),
            ]
        )
