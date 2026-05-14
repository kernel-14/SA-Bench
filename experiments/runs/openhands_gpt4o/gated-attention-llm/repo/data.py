import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path):
        # Load preprocessed data from file
        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Each item is a tuple (input, target)
        return self.data[idx]