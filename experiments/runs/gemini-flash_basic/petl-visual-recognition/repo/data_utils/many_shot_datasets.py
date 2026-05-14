import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from collections import defaultdict

from .transformations import get_transforms

def get_many_shot_datasets(image_size=224, batch_size=64, num_workers=4, seed=42):
    torch.manual_seed(seed)

    train_transform = get_transforms(image_size, is_train=True)
    eval_transform = get_transforms(image_size, is_train=False)

    many_shot_tasks = {
        "cifar100": {
            "dataset_class": datasets.CIFAR100,
            "num_classes": 100,
            "root": "./data/cifar100",
        },
        "resisc": { # Using FakeData as a placeholder
            "dataset_class": datasets.FakeData,
            "num_classes": 45,
            "root": "./data/resisc",
            "fake_data_args": {"size": 50000, "image_size": (3, image_size, image_size), "num_classes": 45},
        },
        "clevr_distance": { # Using FakeData as a placeholder
            "dataset_class": datasets.FakeData,
            "num_classes": 6,
            "root": "./data/clevr_distance",
            "fake_data_args": {"size": 70000, "image_size": (3, image_size, image_size), "num_classes": 6},
        },
    }

    loaders = defaultdict(dict)

    for task_name, task_info in many_shot_tasks.items():
        dataset_class = task_info["dataset_class"]
        num_classes = task_info["num_classes"]
        root = task_info["root"]

        if task_name == "cifar100":
            train_dataset = dataset_class(root=root, train=True, download=True, transform=train_transform)
            test_dataset = dataset_class(root=root, train=False, download=True, transform=eval_transform)
        elif task_name in ["resisc", "clevr_distance"]:
            fake_data_args = task_info["fake_data_args"]
            train_dataset = dataset_class(transform=train_transform, **fake_data_args)
            test_dataset = dataset_class(transform=eval_transform, **fake_data_args) # Use a fresh set for test
        else:
            raise ValueError(f"Unsupported dataset for many-shot simulation: {task_name}")

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        loaders[task_name] = {
            "train": train_loader,
            "test": test_loader,
            "num_classes": num_classes,
        }
    return loaders

if __name__ == '__main__':
    many_shot_loaders = get_many_shot_datasets(num_workers=0)
    for task_name, data_info in many_shot_loaders.items():
        print(f"
--- {task_name} ---")
        print(f"Num classes: {data_info['num_classes']}")
        print(f"Train batches: {len(data_info['train'])}")
        print(f"Test batches: {len(data_info['test'])}")

