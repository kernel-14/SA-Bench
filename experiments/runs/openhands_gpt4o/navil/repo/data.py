import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class MultimodalDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('RGB')
        text = item['text']
        label = item['label']

        # Preprocess image
        image = self.preprocess_image(image)

        # Preprocess text
        text = self.preprocess_text(text)

        return image, text, label

    def preprocess_image(self, image):
        # Example preprocessing: resize and convert to tensor
        image = image.resize((224, 224))
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        return image

    def preprocess_text(self, text):
        # Example preprocessing: tokenize text
        return torch.tensor([ord(c) for c in text])

# Example usage
if __name__ == "__main__":
    dataset = MultimodalDataset('path_to_data.jsonl')
    print(len(dataset))
    image, text, label = dataset[0]
    print(image.shape, text, label)