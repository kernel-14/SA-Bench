import os
import torch
from torch.utils.data import Dataset
import numpy as np

class SokobanDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        for file_name in os.listdir(self.data_path):
            file_path = os.path.join(self.data_path, file_name)
            if file_name.endswith('.npz'):
                with np.load(file_path) as data:
                    self.data.append(data['board'])
                    self.labels.append(data['action'])

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board = torch.tensor(self.data[idx], dtype=torch.float32)
        action = torch.tensor(self.labels[idx], dtype=torch.long)
        return board, action