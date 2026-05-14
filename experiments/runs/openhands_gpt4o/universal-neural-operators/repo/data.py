# data.py

import torch
from torch.utils.data import Dataset
import numpy as np
import os

class PDEBenchDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.inputs, self.targets = self.load_data()

    def load_data(self):
        inputs = []
        targets = []
        for file_name in os.listdir(self.data_path):
            if file_name.endswith('.npz'):
                data = np.load(os.path.join(self.data_path, file_name))
                inputs.append(data['inputs'])
                targets.append(data['targets'])
        inputs = np.concatenate(inputs, axis=0)
        targets = np.concatenate(targets, axis=0)
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]