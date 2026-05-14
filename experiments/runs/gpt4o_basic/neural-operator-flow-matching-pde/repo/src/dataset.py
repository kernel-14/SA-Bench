import os
import torch
from torch.utils.data import Dataset, DataLoader
import h5py

class PDEDataset(Dataset):
    def __init__(self, directory, split='train'):
        super(PDEDataset, self).__init__()
        self.data_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.h5')]
        self.split = split

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_path = self.data_files[idx]
        with h5py.File(file_path, 'r') as f:
            data = f[self.split][:]
        return torch.tensor(data)

def load_data(directory, batch_size, split='train'):
    dataset = PDEDataset(directory, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'))
