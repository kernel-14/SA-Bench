import torch
from torch.utils.data import Dataset
import os
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        # Example: Load data from .npy files
        data_files = [f for f in os.listdir(self.data_path) if f.endswith('.npy')]
        data = [np.load(os.path.join(self.data_path, f)) for f in data_files]
        return np.concatenate(data, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        # Example: Return data and a dummy sigma value
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(0.1, dtype=torch.float32)

def get_dataset(dataset_name, dataset_path):
    if dataset_name == "custom":
        return CustomDataset(dataset_path)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")