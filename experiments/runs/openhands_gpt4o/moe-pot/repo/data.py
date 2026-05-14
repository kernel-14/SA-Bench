import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class PDEDataLoader(Dataset):
    def __init__(self, data_path: str, batch_size: int, shuffle: bool = True):
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = self._load_data()

    def _load_data(self):
        # Assuming data is stored as .npz files with 'inputs' and 'targets' arrays
        data_files = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path) if f.endswith('.npz')]
        inputs, targets = [], []
        for file in data_files:
            with np.load(file) as data:
                inputs.append(data['inputs'])
                targets.append(data['targets'])
        return list(zip(np.concatenate(inputs, axis=0), np.concatenate(targets, axis=0)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs, targets = self.data[idx]
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

    def get_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, shuffle=self.shuffle)