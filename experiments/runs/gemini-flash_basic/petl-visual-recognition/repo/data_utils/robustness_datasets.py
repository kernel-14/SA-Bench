import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
from collections import defaultdict

from .transformations import get_transforms

class ImageNet1KSubset(Subset):
    """
    A subset of ImageNet-1K for 100-shot experiments.
    """
    def __init__(self, dataset, indices, num_classes):
        super().__init__(dataset, indices)
        self.num_classes = num_classes

def get_robustness_datasets(image_size=224, batch_size=64, num_workers=4, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_transform = get_transforms(image_size, is_train=True)
    eval_transform = get_transforms(image_size, is_train=False)

    robustness_tasks = {
        "imagenet_1k_100shot": {
            "dataset_class": datasets.FakeData, # Use FakeData for simplicity in static reproduction
            "num_classes": 1000,
            "root": "./data/imagenet_1k",
            "fake_data_args": {"size": 100 * 1000, "image_size": (3, image_size, image_size), "num_classes": 1000},
        },
        "imagenet_v2": {
            "dataset_class": datasets.FakeData, # Simulate ImageNet-V2
            "num_classes": 1000,
            "root": "./data/imagenet_v2",
            "fake_data_args": {"size": 10000, "image_size": (3, image_size, image_size), "num_classes": 1000},
        },
        "imagenet_r": {
            "dataset_class": datasets.FakeData, # Simulate ImageNet-R (200 classes)
            "num_classes": 200,
            "root": "./data/imagenet_r",
            "fake_data_args": {"size": 5000, "image_size": (3, image_size, image_size), "num_classes": 200},
        },
        "imagenet_s": {
            "dataset_class": datasets.FakeData, # Simulate ImageNet-S
            "num_classes": 1000,
            "root": "./data/imagenet_s",
            "fake_data_args": {"size": 10000, "image_size": (3, image_size, image_size), "num_classes": 1000},
        },
        "imagenet_a": {
            "dataset_class": datasets.FakeData, # Simulate ImageNet-A (200 classes)
            "num_classes": 200,
            "root": "./data/imagenet_a",
            "fake_data_args": {"size": 5000, "image_size": (3, image_size, image_size), "num_classes": 200},
        },
    }

    loaders = defaultdict(dict)

    for task_name, task_info in robustness_tasks.items():
        dataset_class = task_info["dataset_class"]
        num_classes = task_info["num_classes"]
        root = task_info["root"]
        fake_data_args = task_info["fake_data_args"]

        if task_name == "imagenet_1k_100shot":
            # Simulate ImageNet-1K 100-shot training
            # We need to ensure we have 100 samples per class for 1000 classes.
            # For FakeData, we can't easily control per-class sampling, so we'll just create a large enough dataset
            # and then take a subset to represent the 100-shot scenario.
            full_dataset = dataset_class(transform=train_transform, **fake_data_args)

            # Simulate 100-shot by taking a random subset of 100 * num_classes images
            if len(full_dataset) > 100 * num_classes:
                all_indices = list(range(len(full_dataset)))
                np.random.shuffle(all_indices)
                hundred_shot_indices = all_indices[:(100 * num_classes)]
            else:
                hundred_shot_indices = list(range(len(full_dataset)))

            train_dataset = ImageNet1KSubset(full_dataset, hundred_shot_indices, num_classes)
            # For evaluation, we'll use a separate FakeData instance to simulate a test set.
            test_dataset = dataset_class(transform=eval_transform, **fake_data_args)
        else:
            # For other ImageNet variants (V2, R, S, A), they are purely for evaluation (distribution shifts)
            train_dataset = None # No training for these datasets
            test_dataset = dataset_class(transform=eval_transform, **fake_data_args)

        if train_dataset:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
            )
            loaders[task_name]["train"] = train_loader

        if test_dataset:
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
            )
            loaders[task_name]["test"] = test_loader
        
        loaders[task_name]["num_classes"] = num_classes

    return loaders

if __name__ == '__main__':
    robustness_loaders = get_robustness_datasets(num_workers=0)
    for task_name, data_info in robustness_loaders.items():
        print(f"
--- {task_name} ---")
        print(f"Num classes: {data_info['num_classes']}")
        if "train" in data_info:
            print(f"Train batches: {len(data_info['train'])}")
        if "test" in data_info:
            print(f"Test batches: {len(data_info['test'])}")

