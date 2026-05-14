"""
Dataset loading and preprocessing for all experiments.

Supports:
  - VTAB-1K (19 tasks via tensorflow-datasets)
  - Many-shot: CIFAR-100, RESISC45, Clevr-Distance
  - Robustness: ImageNet-1K (100-shot), ImageNet-V2/R/S/A
"""

from __future__ import annotations

import os
import math
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets
from PIL import Image

from config import (
    VTAB_ALL_TASKS,
    VTAB_NUM_CLASSES,
    VTAB_NATURAL,
    VTAB_SPECIALIZED,
    VTAB_STRUCTURED,
)

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# CLIP normalization statistics
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_vtab_transform(image_size: int = 224) -> transforms.Compose:
    """VTAB-1K: resize + center crop, no augmentation."""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_train_transform(
    image_size: int = 224,
    augmentations: Optional[List[str]] = None,
) -> transforms.Compose:
    """Training transform with optional augmentations."""
    aug_list = augmentations or []
    transform_list = [
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
    ]
    if "random_resized_crop" in aug_list:
        transform_list = [
            transforms.RandomResizedCrop(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        ]
    if "horizontal_flip" in aug_list:
        transform_list.append(transforms.RandomHorizontalFlip())
    if "vertical_flip" in aug_list:
        transform_list.append(transforms.RandomVerticalFlip())
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transforms.Compose(transform_list)


def get_clip_train_transform(image_size: int = 224) -> transforms.Compose:
    """Strong augmentation for CLIP robustness experiments (following CLIP paper)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.08, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def get_clip_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Evaluation transform for CLIP experiments."""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


# ---------------------------------------------------------------------------
# VTAB-1K Dataset
# ---------------------------------------------------------------------------

VTAB_TFDS_NAMES = {
    "caltech101": "caltech101",
    "cifar100": "cifar100",
    "dtd": "dtd",
    "flowers102": "oxford_flowers102",
    "pets": "oxford_iiit_pet",
    "svhn": "svhn_cropped",
    "sun397": "sun397",
    "camelyon": "patch_camelyon",
    "eurosat": "eurosat",
    "resisc45": "resisc45",
    "retinopathy": "diabetic_retinopathy_detection/btgraham-300",
    "clevr_count": "clevr",
    "clevr_distance": "clevr",
    "dmlab": "dmlab",
    "kitti": "kitti",
    "dsprites_loc": "dsprites",
    "dsprites_ori": "dsprites",
    "smallnorb_azimuth": "smallnorb",
    "smallnorb_elevation": "smallnorb",
}


class VTABDataset(Dataset):
    """
    VTAB-1K dataset wrapper using tensorflow-datasets.
    Loads 1000 training samples and the full test set.
    """

    def __init__(
        self,
        task_name: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        data_dir: str = "./data/vtab",
        num_train_samples: int = 1000,
        seed: int = 42,
    ):
        self.task_name = task_name
        self.split = split
        self.transform = transform
        self.data_dir = data_dir
        self.num_train_samples = num_train_samples
        self.seed = seed

        self.images, self.labels = self._load_data()

    def _load_data(self) -> Tuple[List, List]:
        """Load data using tensorflow-datasets."""
        try:
            import tensorflow_datasets as tfds
            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')
        except ImportError:
            raise ImportError(
                "tensorflow-datasets is required for VTAB-1K. "
                "Install with: pip install tensorflow-datasets tensorflow"
            )

        tfds_name = VTAB_TFDS_NAMES[self.task_name]

        if self.split in ("train", "val"):
            ds_split = "train"
        else:
            ds_split = "test"

        ds = tfds.load(
            tfds_name,
            split=ds_split,
            data_dir=self.data_dir,
            as_supervised=False,
        )

        images, labels = [], []
        for example in tfds.as_numpy(ds):
            img = self._extract_image(example)
            label = self._extract_label(example)
            images.append(img)
            labels.append(label)

        if self.split in ("train", "val"):
            # Use first 1000 samples (consistent with VTAB-1K protocol)
            rng = np.random.RandomState(self.seed)
            indices = rng.permutation(len(images))[:self.num_train_samples]
            images = [images[i] for i in indices]
            labels = [labels[i] for i in indices]

            # 80/20 train/val split
            n_train = int(0.8 * len(images))
            if self.split == "train":
                images = images[:n_train]
                labels = labels[:n_train]
            else:
                images = images[n_train:]
                labels = labels[n_train:]

        return images, labels

    def _extract_image(self, example: dict) -> np.ndarray:
        """Extract image array from tfds example."""
        if "image" in example:
            return example["image"]
        elif "img" in example:
            return example["img"]
        else:
            # Try first array key
            for k, v in example.items():
                if isinstance(v, np.ndarray) and v.ndim == 3:
                    return v
        raise KeyError(f"Cannot find image in example keys: {list(example.keys())}")

    def _extract_label(self, example: dict) -> int:
        """Extract label from tfds example, handling task-specific label fields."""
        label_keys = {
            "clevr_count": "count",
            "clevr_distance": "closest_object_distance",
            "dsprites_loc": "label_x_position",
            "dsprites_ori": "label_orientation",
            "smallnorb_azimuth": "label_azimuth",
            "smallnorb_elevation": "label_elevation",
            "kitti": "label",
            "dmlab": "label",
        }
        key = label_keys.get(self.task_name, "label")
        label = example[key]
        if hasattr(label, "item"):
            return int(label.item())
        return int(label)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_array = self.images[idx]
        label = self.labels[idx]

        # Convert to PIL Image
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[-1] == 1:
            img_array = np.concatenate([img_array] * 3, axis=-1)

        img = Image.fromarray(img_array)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def get_vtab_dataloaders(
    task_name: str,
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 224,
    data_dir: str = "./data/vtab",
    num_train_samples: int = 1000,
) -> Dict[str, DataLoader]:
    """Create train, val, and test dataloaders for a VTAB-1K task."""
    transform = get_vtab_transform(image_size)

    train_dataset = VTABDataset(
        task_name, split="train", transform=transform,
        data_dir=data_dir, num_train_samples=num_train_samples,
    )
    val_dataset = VTABDataset(
        task_name, split="val", transform=transform,
        data_dir=data_dir, num_train_samples=num_train_samples,
    )
    test_dataset = VTABDataset(
        task_name, split="test", transform=transform,
        data_dir=data_dir, num_train_samples=num_train_samples,
    )

    return {
        "train": DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        ),
        "val": DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
        "test": DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
    }


# ---------------------------------------------------------------------------
# Many-shot datasets
# ---------------------------------------------------------------------------

class RESISC45Dataset(Dataset):
    """
    RESISC45 remote sensing scene classification dataset.
    Expects directory structure: root/class_name/image.jpg
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        train_val_split: float = 0.9,
        seed: int = 42,
    ):
        self.root = Path(root)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        all_samples = []
        for cls in self.classes:
            cls_dir = self.root / cls
            for img_path in sorted(cls_dir.glob("*.jpg")):
                all_samples.append((str(img_path), self.class_to_idx[cls]))

        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(all_samples))
        n_train = int(train_val_split * len(all_samples))

        if split == "train":
            self.samples = [all_samples[i] for i in indices[:n_train]]
        else:
            self.samples = [all_samples[i] for i in indices[n_train:]]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_manyshot_dataloaders(
    dataset_name: str,
    data_root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 224,
    augmentations: Optional[List[str]] = None,
    train_val_split: float = 0.9,
) -> Dict[str, DataLoader]:
    """Create train and val dataloaders for many-shot datasets."""
    train_transform = get_train_transform(image_size, augmentations)
    eval_transform = get_vtab_transform(image_size)

    if dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=data_root, train=True, download=True, transform=train_transform
        )
        val_dataset = datasets.CIFAR100(
            root=data_root, train=False, download=True, transform=eval_transform
        )
        # 90/10 split of training data
        n_total = len(train_dataset)
        n_train = int(train_val_split * n_total)
        indices = list(range(n_total))
        random.seed(42)
        random.shuffle(indices)
        train_subset = Subset(train_dataset, indices[:n_train])
        val_subset = Subset(
            datasets.CIFAR100(root=data_root, train=True, download=True, transform=eval_transform),
            indices[n_train:]
        )
        return {
            "train": DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=True),
            "val": DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True),
            "test": DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True),
        }

    elif dataset_name == "resisc45":
        train_dataset = RESISC45Dataset(
            data_root, split="train", transform=train_transform,
            train_val_split=train_val_split,
        )
        val_dataset = RESISC45Dataset(
            data_root, split="val", transform=eval_transform,
            train_val_split=train_val_split,
        )
        return {
            "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=True),
            "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True),
            "test": DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True),
        }

    elif dataset_name == "clevr_distance":
        # Load via VTAB interface (full training set)
        transform = get_vtab_transform(image_size)
        train_ds = VTABDataset("clevr_distance", split="train", transform=transform,
                               data_dir=data_root, num_train_samples=70000)
        test_ds = VTABDataset("clevr_distance", split="test", transform=transform,
                              data_dir=data_root)
        return {
            "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=True),
            "val": DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True),
            "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True),
        }

    else:
        raise ValueError(f"Unknown many-shot dataset: {dataset_name}")


# ---------------------------------------------------------------------------
# ImageNet and distribution shift datasets
# ---------------------------------------------------------------------------

def get_imagenet_few_shot_dataset(
    imagenet_root: str,
    num_shots: int = 100,
    transform: Optional[Callable] = None,
    seed: int = 42,
) -> Dataset:
    """
    Sample num_shots images per class from ImageNet training set.
    """
    full_dataset = datasets.ImageFolder(
        root=os.path.join(imagenet_root, "train"),
        transform=transform,
    )
    class_indices: Dict[int, List[int]] = {}
    for idx, (_, label) in enumerate(full_dataset.samples):
        class_indices.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    selected_indices = []
    for label, indices in sorted(class_indices.items()):
        sampled = rng.sample(indices, min(num_shots, len(indices)))
        selected_indices.extend(sampled)

    return Subset(full_dataset, selected_indices)


def get_imagenet_dataloaders(
    imagenet_root: str,
    num_shots: int = 100,
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 224,
    use_clip_norm: bool = True,
    use_strong_augmentation: bool = True,
) -> Dict[str, DataLoader]:
    """Create dataloaders for ImageNet few-shot + distribution shift evaluation."""
    if use_clip_norm:
        train_transform = (
            get_clip_train_transform(image_size)
            if use_strong_augmentation
            else get_clip_eval_transform(image_size)
        )
        eval_transform = get_clip_eval_transform(image_size)
    else:
        train_transform = get_train_transform(image_size, ["horizontal_flip"])
        eval_transform = get_vtab_transform(image_size)

    train_dataset = get_imagenet_few_shot_dataset(
        imagenet_root, num_shots=num_shots, transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(imagenet_root, "val"),
        transform=eval_transform,
    )

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True),
    }


def get_distribution_shift_dataloaders(
    data_roots: Dict[str, str],
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 224,
    use_clip_norm: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for ImageNet distribution shift datasets.

    Args:
        data_roots: dict mapping dataset name to root directory
          Keys: "imagenet_v2", "imagenet_r", "imagenet_s", "imagenet_a"
    """
    eval_transform = (
        get_clip_eval_transform(image_size)
        if use_clip_norm
        else get_vtab_transform(image_size)
    )

    loaders = {}
    for name, root in data_roots.items():
        if not os.path.exists(root):
            continue
        dataset = datasets.ImageFolder(root=root, transform=eval_transform)
        loaders[name] = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
    return loaders
