import torch
from torch.utils.data import Dataset
import pywt
import numpy as np

class WaveletDataset(Dataset):
    def __init__(self, data_path: str):
        # Load data from the given path
        self.data = np.load(data_path)
        self.inputs = self.data['inputs']
        self.targets = self.data['targets']

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        target_sample = self.targets[idx]

        # Apply wavelet transform
        coeffs = pywt.wavedec(input_sample, wavelet='bior2.4', mode='periodization')
        input_transformed = np.concatenate([np.ravel(c) for c in coeffs])

        return torch.tensor(input_transformed, dtype=torch.float32), torch.tensor(target_sample, dtype=torch.float32)