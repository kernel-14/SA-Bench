
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Optional

class CustomTokenizedDataset(Dataset):
    """
    A placeholder for a custom tokenized dataset.
    In a real scenario, this would load pre-tokenized data
    or tokenize raw text on the fly.
    """
    def __init__(self, tokenizer: AutoTokenizer, seq_len: int, num_samples: int = 10000):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.num_samples = num_samples
        # Dummy data for demonstration. In a real scenario, you'd load your dataset here.
        # This creates 'num_samples' sequences of 'seq_len' tokens.
        # The actual tokens would be loaded from the 3.5T token dataset mentioned in the paper.
        self.dummy_data = [torch.randint(0, tokenizer.vocab_size, (seq_len,)) for _ in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # In a real scenario, you would retrieve tokenized input_ids and attention_mask
        # from your loaded dataset.
        input_ids = self.dummy_data[idx]
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

def get_dataloader(
    tokenizer: AutoTokenizer,
    seq_len: int,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
    num_samples: int = 10000 # For dummy data
) -> DataLoader:
    """
    Creates a DataLoader for the custom tokenized dataset.
    """
    dataset = CustomTokenizedDataset(tokenizer, seq_len, num_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

def load_tokenizer(model_name_or_path: str = "gpt2") -> AutoTokenizer:
    """
    Loads a tokenizer. The paper doesn't specify the base model, so using gpt2 as a common example.
    """
    return AutoTokenizer.from_pretrained(model_name_or_path)

if __name__ == "__main__":
    # Example usage
    tokenizer = load_tokenizer()
    seq_len = 4096
    batch_size = 2
    dataloader = get_dataloader(tokenizer, seq_len, batch_size, num_samples=100)

    print(f"Number of batches: {len(dataloader)}")
    for i, batch in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Attention Mask shape: {batch['attention_mask'].shape}")
        if i == 0:
            break
