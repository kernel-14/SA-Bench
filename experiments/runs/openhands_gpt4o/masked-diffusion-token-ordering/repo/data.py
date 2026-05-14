# data.py

import torch
from torch.utils.data import Dataset

class MaskedDataset(Dataset):
    def __init__(self, data_path: str):
        self.data = self.load_data(data_path)

    def load_data(self, data_path: str):
        # Placeholder for loading data from a file or other source
        # Replace this with actual data loading logic
        return torch.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_t, t, target = self.data[idx]
        return x_t, t, target