import torch
from torch.utils.data import Dataset

class ExampleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_dataset(dataset_name):
    if dataset_name == "example":
        # Example dataset
        train_data = torch.randn(1000, 10)
        train_labels = torch.randint(0, 2, (1000,))
        val_data = torch.randn(200, 10)
        val_labels = torch.randint(0, 2, (200,))

        train_dataset = ExampleDataset(train_data, train_labels)
        val_dataset = ExampleDataset(val_data, val_labels)

        return train_dataset, val_dataset
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")