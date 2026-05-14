import os
import torch
from torch.utils.data import Dataset

class OpenWebTextDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = self._load_data()

    def _load_data(self):
        data = []
        for file_name in os.listdir(self.dataset_path):
            file_path = os.path.join(self.dataset_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                data.extend(f.readlines())
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        tokens = line.strip().split()
        input_ids = [int(token) for token in tokens[:-1]]
        target_ids = [int(token) for token in tokens[1:]]
        return torch.tensor(input_ids), torch.tensor(target_ids)