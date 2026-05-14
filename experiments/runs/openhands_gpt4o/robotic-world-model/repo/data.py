import torch
from torch.utils.data import Dataset

class RoboticDataset(Dataset):
    def __init__(self, data_path, seq_len, input_dim, output_dim):
        # Load data from the given path
        # For now, we use dummy data
        self.data = torch.randn(10000, seq_len, input_dim)
        self.labels = torch.randn(10000, output_dim)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Example usage
if __name__ == "__main__":
    dataset = RoboticDataset("/path/to/data", seq_len=10, input_dim=128, output_dim=64)
    print(f"Dataset size: {len(dataset)}")
    sample_data, sample_label = dataset[0]
    print(f"Sample data shape: {sample_data.shape}, Sample label shape: {sample_label.shape}")