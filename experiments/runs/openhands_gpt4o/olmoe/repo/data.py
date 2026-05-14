# data.py

import torch
from torch.utils.data import Dataset

class OLMoEDataset(Dataset):
    def __init__(self, data_path: str):
        # Load and preprocess data
        self.data = self.load_data(data_path)

    def load_data(self, data_path: str):
        # Placeholder for loading data
        # Replace with actual data loading logic
        return [(torch.randint(0, 100, (128,)), torch.randint(0, 100, (128,))) for _ in range(1000)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]