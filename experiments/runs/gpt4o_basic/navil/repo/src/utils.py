# Utility functions for NaViL

import torch
from torch.utils.data import DataLoader, Dataset

class MultimodalDataset(Dataset):
    def __init__(self, image_paths, text_data):
        self.image_paths = image_paths
        self.text_data = text_data

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Placeholder for image and text token loading
        image = torch.randn(3, 224, 224)  # Random image tensor
        text = torch.randint(0, 1000, (50,))  # Random text IDs
        return image, text

def create_dataloader(image_paths, text_data, batch_size):
    dataset = MultimodalDataset(image_paths, text_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

