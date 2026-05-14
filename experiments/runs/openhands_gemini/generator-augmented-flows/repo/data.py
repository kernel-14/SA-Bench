
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

class CustomDataset:
    def __init__(self, config):
        self.config = config
        self.transform = self._get_transform()

    def _get_transform(self):
        # Preprocessing: resize, center crop, scale to [-1, 1]
        img_size = self.config.image_resolution
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Scales to [-1, 1]
        ])

    def get_dataloader(self, train=True):
        dataset_name = self.config.dataset.lower()
        data_dir = self.config.data_dir
        os.makedirs(data_dir, exist_ok=True)

        if dataset_name == "cifar10":
            dataset = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=self.transform)
        elif dataset_name == "imagenet32":
            # ImageNet is large, so we'll use a placeholder or assume it's pre-downloaded
            # For reproduction, actual ImageNet loading might require manual setup or a custom loader
            # This is a placeholder for ImageNet loading
            raise NotImplementedError("ImageNet dataset loading not fully implemented. Please ensure it's pre-downloaded or use a custom loader.")
        elif dataset_name == "celeba":
            dataset = datasets.CelebA(root=data_dir, split='train' if train else 'test', download=True, transform=self.transform)
        elif dataset_name == "lsun_church":
            # LSUN has specific categories, 'church_outdoor_train' is common for LSUN Church
            # This is a placeholder for LSUN loading
            raise NotImplementedError("LSUN Church dataset loading not fully implemented. Please ensure it's pre-downloaded or use a custom loader.")
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=train,
            num_workers=os.cpu_count() // 2, # Use half of CPU cores for data loading
            pin_memory=True
        )
        return dataloader

# Example usage (for testing purposes)
if __name__ == "__main__":
    from config import Config

    # Test CIFAR-10
    config_cifar10 = Config(dataset="cifar10")
    cifar10_dataset = CustomDataset(config_cifar10)
    cifar10_dataloader = cifar10_dataset.get_dataloader(train=True)
    print(f"CIFAR-10 DataLoader: {len(cifar10_dataloader.dataset)} samples, {len(cifar10_dataloader)} batches")
    for batch in cifar10_dataloader:
        images, labels = batch
        print(f"CIFAR-10 Batch shape: {images.shape}, Labels shape: {labels.shape}")
        break

    # Test CelebA
    config_celeba = Config(dataset="celeba")
    celeba_dataset = CustomDataset(config_celeba)
    celeba_dataloader = celeba_dataset.get_dataloader(train=True)
    print(f"CelebA DataLoader: {len(celeba_dataloader.dataset)} samples, {len(celeba_dataloader)} batches")
    for batch in celeba_dataloader:
        images, labels = batch
        print(f"CelebA Batch shape: {images.shape}, Labels shape: {labels.shape}")
        break

