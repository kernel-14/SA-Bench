import numpy as np
import torch
from torch.utils.data import Dataset

class PDEDataLoader(Dataset):
    def __init__(self, dataset_paths, preprocess_fn=None):
        # Paths to PDE datasets
        self.dataset_paths = dataset_paths
        self.preprocess_fn = preprocess_fn
        self.data = []
        self.load_datasets()

    def load_datasets(self):
        # Load and concatenate multiple PDE datasets
        for path in self.dataset_paths:
            loaded_data = np.load(path)
            if self.preprocess_fn:
                loaded_data = self.preprocess_fn(loaded_data)
            self.data.append(loaded_data)
        self.data = np.concatenate(self.data, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return discretized PDE sample at index idx
        return torch.tensor(self.data[idx], dtype=torch.float32)

# Example utility for mesh preprocessing
def preprocess_example(input_data):
    # Placeholder function for domain-specific preprocessing (e.g., padding, masking)
    return input_data / np.max(input_data)   # Normalization

