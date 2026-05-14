import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

class PDEDataLoader(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        with h5py.File(data_path, 'r') as f:
            self.data = f['data'][:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return torch.tensor(sample, dtype=torch.float32)

def get_dataloader(data_path, batch_size, shuffle=True):
    dataset = PDEDataLoader(data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)