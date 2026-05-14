import torch
from torch.utils.data import Dataset
from typing import List, Tuple

class TextDataset(Dataset):
    def __init__(self, data_path: str):
        self.prompts, self.responses, self.labels = self.load_data(data_path)

    def load_data(self, data_path: str) -> Tuple[List[str], List[str], List[int]]:
        prompts, responses, labels = [], [], []
        with open(data_path, 'r') as f:
            for line in f:
                prompt, response, label = line.strip().split('\t')
                prompts.append(prompt)
                responses.append(response)
                labels.append(int(label))
        return prompts, responses, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.prompts[idx], self.responses[idx], torch.tensor(self.labels[idx], dtype=torch.float32)

# Example usage
if __name__ == "__main__":
    dataset = TextDataset(data_path="data/sample_data.txt")
    print(f"Dataset size: {len(dataset)}")
    for i in range(3):
        print(dataset[i])