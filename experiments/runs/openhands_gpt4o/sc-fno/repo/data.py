import torch
from torch.utils.data import Dataset, DataLoader

class DifferentialEquationDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Example usage
# data = torch.randn(1000, 3)  # 1000 samples, 3 features (initial condition, spatial, temporal)
# targets = torch.randn(1000, 1)  # 1000 samples, 1 target
# dataset = DifferentialEquationDataset(data, targets)
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True)