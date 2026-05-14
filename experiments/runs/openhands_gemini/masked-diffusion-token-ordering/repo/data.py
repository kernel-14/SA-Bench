
import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

from config import Config

class BaseDataset(Dataset):
    def __init__(self, data: list, max_sequence_length: int, vocab_size: int, mask_token_id: int):
        self.data = data
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        raise NotImplementedError

    def _pad_sequence(self, sequence: list):
        if len(sequence) < self.max_sequence_length:
            # Pad with mask_token_id (or a dedicated padding token if different)
            padded_sequence = sequence + [self.mask_token_id] * (self.max_sequence_length - len(sequence))
        else:
            padded_sequence = sequence[:self.max_sequence_length]
        return torch.tensor(padded_sequence, dtype=torch.long)

class TextDataset(BaseDataset):
    """
    A placeholder Text Dataset. In a real scenario, this would load and tokenize
    datasets like Slimpajama.
    """
    def __init__(self, num_samples: int, max_sequence_length: int, vocab_size: int, mask_token_id: int):
        super().__init__([], max_sequence_length, vocab_size, mask_token_id)
        print(f"Generating {num_samples} synthetic text samples...")
        self.data = self._generate_synthetic_data(num_samples)
        print("Synthetic text data generation complete.")

    def _generate_synthetic_data(self, num_samples: int):
        synthetic_data = []
        for _ in range(num_samples):
            # Generate a random sequence of tokens
            length = random.randint(self.max_sequence_length // 2, self.max_sequence_length)
            sequence = [random.randint(1, self.vocab_size - 1) for _ in range(length)] # Avoid mask_token_id for content
            synthetic_data.append(sequence)
        return synthetic_data

    def __getitem__(self, idx: int):
        raw_sequence = self.data[idx]
        input_ids = self._pad_sequence(raw_sequence)
        
        # For training, the target is the original sequence
        labels = input_ids.clone() 
        return {"input_ids": input_ids, "labels": labels}

class SudokuDataset(BaseDataset):
    """
    A placeholder Sudoku Dataset.
    Generates synthetic Sudoku puzzles as sequences.
    A Sudoku puzzle can be represented as a sequence of 81 tokens (9x9 grid).
    0 can represent an empty cell (mask token).
    """
    def __init__(self, num_samples: int, mask_token_id: int = 0):
        # Sudoku grid is 9x9, so sequence length is 81. Vocab is 0-9.
        super().__init__([], 81, 10, mask_token_id)
        print(f"Generating {num_samples} synthetic Sudoku samples...")
        self.data = self._generate_synthetic_data(num_samples)
        print("Synthetic Sudoku data generation complete.")

    def _generate_synthetic_data(self, num_samples: int):
        synthetic_data = []
        for _ in range(num_samples):
            # A very simple synthetic Sudoku: mostly filled, some masked
            puzzle = [random.randint(1, 9) for _ in range(81)]
            # Mask some positions
            num_masked = random.randint(10, 50)
            masked_indices = random.sample(range(81), num_masked)
            for idx in masked_indices:
                puzzle[idx] = self.mask_token_id # 0 represents masked/empty cell
            synthetic_data.append(puzzle)
        return synthetic_data

    def __getitem__(self, idx: int):
        puzzle_sequence = self.data[idx]
        input_ids = torch.tensor(puzzle_sequence, dtype=torch.long)
        
        # In a Sudoku context, labels would be the full solution,
        # but for MDM training, the labels are the original tokens for masked positions.
        # Here, we assume input_ids is already x_0 with some masked positions (0s)
        # and the labels are the original values that *should* be there.
        # For this synthetic data, we can't easily get the *true* solution.
        # We'll just use the input_ids as labels and rely on the masking in the training loop.
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


def get_dataloader(dataset_name: str, batch_size: int, shuffle: bool, **kwargs):
    if dataset_name == "Slimpajama":
        dataset = TextDataset(
            num_samples=kwargs.get("num_samples", 10000),
            max_sequence_length=Config.max_sequence_length,
            vocab_size=Config.vocab_size,
            mask_token_id=Config.mask_token_id
        )
    elif dataset_name == "Sudoku":
        dataset = SudokuDataset(
            num_samples=kwargs.get("num_samples", 10000),
            mask_token_id=Config.mask_token_id
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

