
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

class ImageNetDataset:
    def __init__(self, root, image_size, batch_size, num_workers, train=True):
        self.root = root
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = train

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def get_dataloader(self):
        # ImageNet dataset structure: root/train/class_name/xxx.JPEG and root/val/class_name/xxx.JPEG
        if self.train:
            dataset_path = os.path.join(self.root, 'train')
        else:
            dataset_path = os.path.join(self.root, 'val') # Using validation set for evaluation

        dataset = datasets.ImageFolder(root=dataset_path, transform=self.transform)
        
        # The paper mentions 1.2M training images and 50k validation images.
        # torchvision.datasets.ImageNet can directly load if the structure is ImageNet-standard.
        # For simplicity, using ImageFolder which assumes a structured folder.
        # If ImageNet class needs specific handling, this part might need adjustment.

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.train,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return dataloader

# Example usage (for testing data.py)
if __name__ == '__main__':
    from nfig.config import get_config
    config = get_config()

    # Create a dummy dataset directory for testing if it doesn't exist
    dummy_data_path = './dummy_imagenet_data'
    os.makedirs(os.path.join(dummy_data_path, 'train', 'class1'), exist_ok=True)
    os.makedirs(os.path.join(dummy_data_path, 'val', 'class1'), exist_ok=True)
    # Create dummy image files if not exist
    # from PIL import Image
    # dummy_img = Image.new('RGB', (config.image_size, config.image_size), color = 'red')
    # dummy_img.save(os.path.join(dummy_data_path, 'train', 'class1', 'dummy_train.png'))
    # dummy_img.save(os.path.join(dummy_data_path, 'val', 'class1', 'dummy_val.png'))

    train_dataset_loader = ImageNetDataset(
        root=dummy_data_path, # Replace with actual ImageNet path
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train=True
    ).get_dataloader()

    val_dataset_loader = ImageNetDataset(
        root=dummy_data_path, # Replace with actual ImageNet path
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train=False
    ).get_dataloader()

    print(f"Number of training batches: {len(train_dataset_loader)}")
    print(f"Number of validation batches: {len(val_dataset_loader)}")

    for batch_idx, (images, labels) in enumerate(train_dataset_loader):
        print(f"Batch {batch_idx}: images shape {images.shape}, labels shape {labels.shape}")
        if batch_idx >= 0: # Just check first batch
            break
