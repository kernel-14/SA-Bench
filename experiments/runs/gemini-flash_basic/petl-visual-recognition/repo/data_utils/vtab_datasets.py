import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
from collections import defaultdict

from .transformations import get_transforms

class VTAB1KSubset(Subset):
    """
    A subset of a dataset for VTAB-1K experiments, handling the 1000-sample training limit.
    """
    def __init__(self, dataset, indices, num_classes):
        super().__init__(dataset, indices)
        self.num_classes = num_classes

def get_vtab_datasets(image_size=224, batch_size=64, num_workers=4, seed=42):
    # For this reproduction, we simulate VTAB-1K behavior by using readily available datasets
    # and applying the 1000-shot logic. Actual VTAB-1K requires specific dataset handling.
    # We'll use CIFAR-100 as a stand-in for a 'Natural' task, and FakeData for others.

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_transform = get_transforms(image_size, is_train=True)
    eval_transform = get_transforms(image_size, is_train=False)

    vtab_tasks = {
        "cifar100": { # Representative for Natural group
            "dataset_class": datasets.CIFAR100,
            "num_classes": 100,
            "root": "./data/cifar100",
        },
        "caltech101": { # Representative for Natural group (using FakeData for simplicity)
            "dataset_class": datasets.FakeData,
            "num_classes": 101,
            "root": "./data/caltech101",
            "fake_data_args": {"size": 10000, "image_size": (3, image_size, image_size), "num_classes": 101},
        },
        "eurosat": { # Representative for Specialized group (using FakeData)
            "dataset_class": datasets.FakeData,
            "num_classes": 10,
            "root": "./data/eurosat",
            "fake_data_args": {"size": 10000, "image_size": (3, image_size, image_size), "num_classes": 10},
        },
        "dmlab": { # Representative for Structured group (using FakeData)
            "dataset_class": datasets.FakeData,
            "num_classes": 6,
            "root": "./data/dmlab",
            "fake_data_args": {"size": 10000, "image_size": (3, image_size, image_size), "num_classes": 6},
        },
    }

    loaders = defaultdict(dict)

    for task_name, task_info in vtab_tasks.items():
        dataset_class = task_info["dataset_class"]
        num_classes = task_info["num_classes"]
        root = task_info["root"]

        # Load full training and test datasets
        if task_name == "cifar100":
            full_train_dataset = dataset_class(root=root, train=True, download=True, transform=train_transform)
            test_dataset = dataset_class(root=root, train=False, download=True, transform=eval_transform)
        elif task_name in ["caltech101", "eurosat", "dmlab"]:
            # For FakeData, train=True/False doesn't matter much, create separate instances
            fake_data_args = task_info["fake_data_args"]
            full_train_dataset = dataset_class(transform=train_transform, **fake_data_args)
            test_dataset = dataset_class(transform=eval_transform, **fake_data_args) # Use a fresh set for test
        else:
            raise ValueError(f"Unsupported dataset for VTAB-1K simulation: {task_name}")

        # Simulate 1000-shot training
        if len(full_train_dataset) > 1000:
            # Randomly sample 1000 indices from the full training dataset
            all_indices = list(range(len(full_train_dataset)))
            np.random.shuffle(all_indices)
            thousand_shot_indices = all_indices[:1000]
        else:
            thousand_shot_indices = list(range(len(full_train_dataset)))

        # Split 1000-shot into 800 for training and 200 for validation
        train_indices, val_indices = train_test_split(
            thousand_shot_indices, train_size=800, test_size=200, random_state=seed
        )

        train_dataset = VTAB1KSubset(full_train_dataset, train_indices, num_classes)
        val_dataset = VTAB1KSubset(full_train_dataset, val_indices, num_classes)
        # Test dataset remains the original test set, not a subset of 1000

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        loaders[task_name] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
            "num_classes": num_classes,
        }
    return loaders

if __name__ == '__main__':
    vtab_loaders = get_vtab_datasets(num_workers=0) # Set num_workers=0 for easier debugging on some systems
    for task_name, data_info in vtab_loaders.items():
        print(f"
--- {task_name} ---")
        print(f"Num classes: {data_info['num_classes']}")
        print(f"Train batches: {len(data_info['train'])}")
        print(f"Val batches: {len(data_info['val'])}")
        print(f"Test batches: {len(data_info['test'])}")

        # Example: iterate through a batch
        # for batch_idx, (images, labels) in enumerate(data_info['train']):
        #     print(f"Train batch {batch_idx}: images shape {images.shape}, labels shape {labels.shape}")
        #     break

