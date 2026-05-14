import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class MathDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(
            item['question'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        labels = self.tokenizer(
            item['answer'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels['input_ids'].squeeze(0)
        }

def load_dataset(dataset_name):
    # Placeholder for dataset loading logic
    # Replace with actual dataset loading code
    if dataset_name == "MATH":
        train_data = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is 3*3?", "answer": "9"}
        ]
        val_data = [
            {"question": "What is 5-2?", "answer": "3"}
        ]
    else:
        raise ValueError("Unsupported dataset")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    train_dataset = MathDataset(train_data, tokenizer, max_length=128)
    val_dataset = MathDataset(val_data, tokenizer, max_length=128)

    return train_dataset, val_dataset