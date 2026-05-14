## dataset_manager.py

"""
Data loading and preprocessing for the PEFT reproducibility study.

Provides the DatasetManager class that encapsulates all data‑handling
logic for VTAB‑1K (low‑shot), many‑shot (CIFAR‑100, RESISC45, Clevr‑Distance),
and robustness (ImageNet‑1K + distribution shifts) experiments.
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# If available, import the CLIP model to get its standard transforms.
# We will hard‑code CLIP mean/std below because the paper uses CLIP ViT‑B/16.
# from open_clip import get_tokenizer, ...

try:
    from config import Config    # type: ignore
except ImportError:
    # In case of relative import issues, define a stub; but we assume it exists.
    class Config:
        pass


# ------------------------------------------------------------------------ #
#  CLIP normalisation constants (used only in robustness experiment)
#  Source: OpenAI CLIP ViT‑B/16 preprocessing.
# ------------------------------------------------------------------------ #
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

# Standard ImageNet‑21K normalisation used for VTAB and many‑shot backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class DatasetManager:
    """
    Manages dataset loading and preprocessing according to the experimental
    protocols defined in the paper and the supplied configuration.
    """

    # ------------------------------------------------------------------ #
    #  Task name list for VTAB (19 tasks grouped as in the paper)
    # ------------------------------------------------------------------ #
    # We need an explicit mapping because folder names might differ from
    # the paper’s display names.  We'll assume tasks are stored exactly as
    # listed here under `config.datasets.vtab_root`.
    VTAB_TASKS = [
        "caltech101", "cifar100", "dtd", "flowers102", "pets", "sun397",
        "svhn",                         # Natural (7 tasks)
        "camelyon", "eurosat", "resisc45", "retinopathy",  # Specialized (4 tasks)
        "clevr_count", "clevr_distance", "dmlab", "kitti",
        "dsprite_ori", "smallnorb_azim", "smallnorb_ele"   # Structured (8 tasks?)
        # Check exact count: Table 1 has 19 tasks. Above are 7+4+8=19.
        # "dSpr-Ori" -> dsprite_ori
    ]

    # Map from paper’s many‑shot dataset keys to folder names.
    MANY_SHOT_TASKS = {
        "cifar100": "cifar100",
        "resisc45": "resisc45",
        "clevr_distance": "clevr_distance"
    }

    # ------------------------------------------------------------------ #
    #  Initialisation
    # ------------------------------------------------------------------ #
    def __init__(self, config: Config) -> None:
        """
        Args:
            config: fully populated Config instance (from config.yaml).
        """
        self.config = config
        # Cache for 100‑shot ImageNet indices to avoid recomputation.
        self._imagenet_100shot_indices: Optional[List[int]] = None

    # ------------------------------------------------------------------ #
    #  Public API: VTAB low‑shot loading
    # ------------------------------------------------------------------ #
    def load_vtab(self, task: str, split: str = "full") -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Load VTAB‑1K data for a specific task.

        Args:
            task:   One of the 19 task names (exact name as in folder).
            split:  "train" (returns 800‑train/200‑val loaders),
                    "full" (returns full 1000 training loader),
                    "test" (returns test loader only).

        Returns:
            A tuple of DataLoaders.  For split=="full" or "test", the second
            element is None.  For "train", the first is the training loader,
            the second the validation loader.
        """
        # Validate task
        if task not in self.VTAB_TASKS:
            raise ValueError(f"Unknown VTAB task: '{task}'. Choose from {self.VTAB_TASKS}")

        # Retrieve file paths
        paths = self._get_vtab_paths()
        train_dir, test_dir = paths[task]

        # Common transform (no augmentation)
        vtab_transform = self._preprocess_vtab()

        # Helper to build a dataset with the transform
        def _make_dataset(path: str) -> Dataset:
            return datasets.ImageFolder(root=path, transform=vtab_transform)

        if split == "train":
            full_train_set = _make_dataset(train_dir)
            # 800/200 split with fixed seed
            val_len = int(len(full_train_set) * self.config.training["vtab"]["val_split_ratio"])
            train_len = len(full_train_set) - val_len
            gen = torch.Generator().manual_seed(self.config.misc["seed"])
            train_subset, val_subset = random_split(
                full_train_set, [train_len, val_len], generator=gen
            )
            train_loader = self._create_loader(
                train_subset,
                batch_size=self.config.training["vtab"]["batch_size"],
                shuffle=True
            )
            val_loader = self._create_loader(
                val_subset,
                batch_size=self.config.training["vtab"]["batch_size"],
                shuffle=False
            )
            return train_loader, val_loader
        elif split == "full":
            full_train_set = _make_dataset(train_dir)
            train_loader = self._create_loader(
                full_train_set,
                batch_size=self.config.training["vtab"]["batch_size"],
                shuffle=True
            )
            return train_loader, None
        elif split == "test":
            test_set = _make_dataset(test_dir)
            test_loader = self._create_loader(
                test_set,
                batch_size=self.config.training["vtab"]["batch_size"],
                shuffle=False
            )
            return test_loader, None
        else:
            raise ValueError(f"Unsupported split: '{split}'. Expected 'train', 'full', or 'test'.")

    # ------------------------------------------------------------------ #
    #  Public API: Many‑shot loading
    # ------------------------------------------------------------------ #
    def load_many_shot(self, dataset: str, split: str = "full") -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Load a many‑shot dataset for training/evaluation.

        Args:
            dataset: one of "cifar100", "resisc45", "clevr_distance".
            split:   "train" (returns 90/10 train‑val loaders),
                     "full" (returns full training loader),
                     "test" (returns test loader only).

        Returns:
            Tuple of DataLoaders; second is None for "full"/"test" splits.
        """
        if dataset not in self.MANY_SHOT_TASKS:
            raise ValueError(f"Unknown many‑shot dataset: '{dataset}'. Supported: {list(self.MANY_SHOT_TASKS.keys())}")

        # Determine augmentations based on dataset (as per paper)
        if dataset == "cifar100":
            horizontal_flip = True
            vertical_flip = False
        elif dataset == "resisc45":
            horizontal_flip = True
            vertical_flip = True
        else:  # clevr_distance
            horizontal_flip = False
            vertical_flip = False

        # Build transforms
        train_transform = self._preprocess_standard(
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip
        )
        # Validation/test transforms have no augmentation (flips) but keep resizing/normalization
        test_transform = self._preprocess_standard(
            horizontal_flip=False,
            vertical_flip=False
        )

        # Load the whole training and test sets
        data_root = self.config.datasets["many_shot"][dataset]
        if dataset == "cifar100":
            full_train_set = datasets.CIFAR100(
                root=data_root, train=True, download=False, transform=train_transform
            )
            test_set = datasets.CIFAR100(
                root=data_root, train=False, download=False, transform=test_transform
            )
        else:
            # RESISC45 and Clevr‑Distance are expected as ImageFolder directories.
            train_dir = os.path.join(data_root, "train")
            test_dir = os.path.join(data_root, "test")
            if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
                raise FileNotFoundError(
                    f"Expected train/ and test/ subdirectories in {data_root}"
                )
            full_train_set = datasets.ImageFolder(root=train_dir, transform=train_transform)
            test_set = datasets.ImageFolder(root=test_dir, transform=test_transform)

        # Handle the requested split
        batch_size = self.config.training["many_shot"]["batch_size"]
        seed = self.config.misc["seed"]

        if split == "train":
            # 90/10 split for validation
            total_len = len(full_train_set)
            val_len = max(1, int(0.1 * total_len))
            train_len = total_len - val_len
            gen = torch.Generator().manual_seed(seed)
            # Apply the appropriate transforms to the subsets:
            # For the training subset, use training transforms; for val, test transforms.
            # However, we loaded full_train_set with train_transform. For the val subset we need
            # to swap the transform. Simple approach: override the transform attribute.
            # But Subset objects refer to the original dataset, so we must be careful.
            # We'll create two separate datasets for train/val to avoid tangled transforms.
            # A cleaner way: use two ImageFolder instances over the same directory,
            # each with the correct transform, then apply random_split indices.
            # That is, we load the raw file list and apply transforms on the fly.
            # Simpler: we can just reload the dataset for the val with test_transform.
            # We'll implement by loading full_train_set again for val with test_transform.
            # However, the underlying files are the same, so splitting must use the same
            # index set. We'll use a helper that returns subsets with different transforms.
            # Alternatively, we can wrap the Subset with a custom transform. This is simpler.
            # We'll use the approach: load the dataset with train_transform, split, then
            # for the val subset we will wrap the Subset in a class that applies test_transform.
            # But that's cumbersome. Instead, we'll load the dataset twice: one for each transform,
            # but use the same indices. Since the dataset is small, this is acceptable.
            # For many-shot we assume the training set is large but double loading is okay.
            # For code clarity, we implement a helper _subset_with_transform.
            train_dataset = self._load_dataset_with_transform(
                dataset, data_root, train_transform, True
            )
            val_dataset = self._load_dataset_with_transform(
                dataset, data_root, test_transform, True
            )

            # Generate indices using the same seed on the original (non-transformed) order.
            # The order is deterministic (files are sorted). So we can split on the train part.
            # To ensure identical splits, we split on a `torch.randperm` with fixed seed.
            indices = torch.randperm(len(train_dataset), generator=gen).tolist()
            train_indices = indices[:train_len]
            val_indices = indices[train_len:]

            train_subset = Subset(train_dataset, train_indices)
            val_subset = Subset(val_dataset, val_indices)

            train_loader = self._create_loader(train_subset, batch_size, shuffle=True)
            val_loader = self._create_loader(val_subset, batch_size, shuffle=False)
            return train_loader, val_loader

        elif split == "full":
            full_dataset = self._load_dataset_with_transform(
                dataset, data_root, train_transform, True
            )
            train_loader = self._create_loader(full_dataset, batch_size, shuffle=True)
            return train_loader, None

        elif split == "test":
            test_dataset = self._load_dataset_with_transform(
                dataset, data_root, test_transform, False
            )
            test_loader = self._create_loader(test_dataset, batch_size, shuffle=False)
            return test_loader, None
        else:
            raise ValueError(f"Unsupported split: '{split}'. Expected 'train', 'full', or 'test'.")

    # ------------------------------------------------------------------ #
    #  Public API: Distribution‑shift (robustness) loading
    # ------------------------------------------------------------------ #
    def load_distribution_shift(self) -> Tuple[DataLoader, DataLoader, List[DataLoader]]:
        """
        Prepare data for the robustness experiment.

        Returns:
            train_loader:   100‑shot ImageNet‑1K training subset with strong augmentation.
            test_loader:    Full ImageNet‑1K validation set.
            shift_loaders:  List of four DataLoaders for ImageNet‑V2, ImageNet‑R,
                            ImageNet‑S (sketch), and ImageNet‑A.
        """
        # Paths
        imagenet_train_dir = self.config.datasets["robustness"]["imagenet_train"]
        imagenet_val_dir   = self.config.datasets["robustness"]["imagenet_val"]
        shift_dirs = [
            self.config.datasets["robustness"]["imagenet_v2"],
            self.config.datasets["robustness"]["imagenet_r"],
            self.config.datasets["robustness"]["imagenet_sketch"],
            self.config.datasets["robustness"]["imagenet_a"],
        ]

        # --- Transforms ---
        train_transform = self._get_clip_train_transform()
        test_transform  = self._get_clip_test_transform()

        # --- Training set (100‑shot) ---
        # Load the full ImageNet training set once to get class distribution
        base_train_set = datasets.ImageFolder(root=imagenet_train_dir)
        indices = self._get_imagenet_100shot_indices(base_train_set)

        # Create a subset with the training transforms applied
        # We need a new dataset with the same folder but with train_transform.
        # Because ImageFolder applies transforms when __getitem__ is called,
        # we can simply instantiate another ImageFolder with the same root.
        train_dataset_with_transform = datasets.ImageFolder(
            root=imagenet_train_dir, transform=train_transform
        )
        train_subset = Subset(train_dataset_with_transform, indices)

        batch_size = self.config.training["robustness"]["batch_size"]
        train_loader = self._create_loader(train_subset, batch_size=batch_size, shuffle=True)

        # --- Target test set (full ImageNet validation) ---
        val_dataset = datasets.ImageFolder(root=imagenet_val_dir, transform=test_transform)
        test_loader = self._create_loader(val_dataset, batch_size=batch_size, shuffle=False)

        # --- Distribution shift datasets ---
        shift_loaders = []
        for shift_dir in shift_dirs:
            shift_dataset = datasets.ImageFolder(root=shift_dir, transform=test_transform)
            shift_loader = self._create_loader(shift_dataset, batch_size=batch_size, shuffle=False)
            shift_loaders.append(shift_loader)

        return train_loader, test_loader, shift_loaders

    # ================================================================== #
    #  Private helper methods
    # ================================================================== #

    def _get_vtab_paths(self) -> Dict[str, Tuple[str, str]]:
        """
        Map VTAB task names to (train_directory, test_directory) paths.

        Assumes standard layout under config.datasets.vtab_root:
            vtab_root/<task>/train/   (class subdirectories)
            vtab_root/<task>/test/    (class subdirectories)
        """
        vtab_root = self.config.datasets["vtab_root"]
        paths = {}
        for task in self.VTAB_TASKS:
            task_dir = os.path.join(vtab_root, task)
            train_dir = os.path.join(task_dir, "train")
            test_dir = os.path.join(task_dir, "test")
            if not os.path.isdir(train_dir):
                raise FileNotFoundError(f"VTAB training directory not found: {train_dir}")
            if not os.path.isdir(test_dir):
                raise FileNotFoundError(f"VTAB test directory not found: {test_dir}")
            paths[task] = (train_dir, test_dir)
        return paths

    @staticmethod
    def _preprocess_vtab() -> transforms.Compose:
        """
        VTAB preprocessing: resize to 224, center crop, normalise with ImageNet‑21K stats.
        No data augmentation.
        """
        return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    @staticmethod
    def _preprocess_standard(horizontal_flip: bool = False,
                             vertical_flip: bool = False) -> transforms.Compose:
        """
        Standard preprocessing for many‑shot datasets (CIFAR‑100, RESISC, Clevr).
        Includes optional flips as specified in the paper.
        Resize to 224, center crop, normalise with ImageNet‑21K stats.
        """
        ops = []
        if horizontal_flip:
            ops.append(transforms.RandomHorizontalFlip())
        if vertical_flip:
            ops.append(transforms.RandomVerticalFlip())
        ops += [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        return transforms.Compose(ops)

    def _get_clip_train_transform(self) -> transforms.Compose:
        """
        Strong augmentation for CLIP fine‑tuning (robustness experiment).
        If config.strong_augmentation is True, uses RandomResizedCrop, RandAugment,
        and horizontal flip; otherwise falls back to simple resize+center crop.
        """
        strong = self.config.training["robustness"].get("strong_augmentation", True)
        if strong:
            return transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ])
        else:
            return self._get_clip_test_transform()

    @staticmethod
    def _get_clip_test_transform() -> transforms.Compose:
        """
        Test/evaluation transform for CLIP backbone: resize to 224 (bicubic),
        center crop, then normalise with CLIP statistics.
        """
        return transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])

    def _get_imagenet_100shot_indices(self, dataset: datasets.ImageFolder) -> List[int]:
        """
        From the full ImageNet training set, randomly select exactly 100 samples
        per class using the fixed seed from config.  Returns a flat list of indices.
        Caches the result for subsequent calls.
        """
        if self._imagenet_100shot_indices is not None:
            return self._imagenet_100shot_indices

        seed = self.config.misc["seed"]
        rng = np.random.RandomState(seed)

        # dataset.targets is a list of integer labels (length = total images)
        targets = np.array(dataset.targets)
        num_classes = 1000  # ImageNet‑1K has 1000 classes
        indices = []
        for cls in range(num_classes):
            cls_indices = np.where(targets == cls)[0]
            if len(cls_indices) < 100:
                raise RuntimeError(f"Class {cls} has only {len(cls_indices)} samples, need 100.")
            chosen = rng.choice(cls_indices, size=100, replace=False)
            indices.extend(chosen.tolist())
        # Sort indices to keep reproducible order
        indices.sort()
        self._imagenet_100shot_indices = indices
        return indices

    def _load_dataset_with_transform(self,
                                     dataset_name: str,
                                     data_root: str,
                                     transform: transforms.Compose,
                                     train: bool) -> Dataset:
        """
        Helper to load a raw dataset (CIFAR‑100 or ImageFolder) with a given
        transform.  Avoids code duplication in load_many_shot.
        """
        if dataset_name == "cifar100":
            return datasets.CIFAR100(
                root=data_root, train=train, transform=transform, download=False
            )
        else:
            # For RESISC45 and Clevr‑Distance, train=True/False indicates which sub‑folder to use.
            subfolder = "train" if train else "test"
            full_path = os.path.join(data_root, subfolder)
            if not os.path.isdir(full_path):
                raise FileNotFoundError(f"Expected directory {full_path}")
            return datasets.ImageFolder(root=full_path, transform=transform)

    def _create_loader(self,
                       dataset: Dataset,
                       batch_size: int,
                       shuffle: bool) -> DataLoader:
        """
        Create a DataLoader with standard parameters (num_workers, pin_memory).
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.misc["num_workers"],
            pin_memory=(self.config.misc["device"] == "cuda"),
        )
