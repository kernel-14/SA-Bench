import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from typing import List, Dict, Any, Optional

# Assuming config.py is in the same directory or accessible via PYTHONPATH
try:
    from config import Config
except ImportError:
    # Fallback for isolated testing or if config.py is not yet available.
    # In a full project setup, config.py would always be present.
    print("Warning: Could not import Config from config.py. Using a dummy Config class for DataLoader testing.")
    class Config:
        """Dummy Config class for isolated DataLoader testing."""
        def __init__(self):
            self.model = type('ModelConfig', (object,), {
                'max_seq_len': 4096
            })()
            self.training = type('TrainingConfig', (object,), {
                'global_batch_size': 16
            })()
            self.data = type('DataConfig', (object,), {
                'tokenizer_path': 'gpt2', # A common tokenizer for testing purposes
                'train_data_paths': ['dummy_train.jsonl'],
                'eval_data_paths': ['dummy_eval.txt']
            })()
            self.long_context_extension = type('LCEConfig', (object,), {
                'enabled': False,
                'extended_seq_len': 8192
            })()

            # Create dummy data files if they don't exist for testing purposes
            if not os.path.exists('dummy_train.jsonl'):
                with open('dummy_train.jsonl', 'w', encoding='utf-8') as f:
                    f.write(json.dumps({'text': 'This is a dummy training sentence.'}) + '\n')
                    f.write(json.dumps({'text': 'Another dummy sentence for training.'}) + '\n')
            if not os.path.exists('dummy_eval.txt'):
                with open('dummy_eval.txt', 'w', encoding='utf-8') as f:
                    f.write('This is a dummy evaluation sentence.\n')
                    f.write('A second line for dummy evaluation.\n')


class DataLoader:
    """
    Handles loading, tokenizing, and batching of training and evaluation data.
    Supports standard and long-context specific data loading for a Gated Attention LLM.
    """

    def __init__(self, config: Config):
        """
        Initializes the DataLoader instance.

        Args:
            config: The global configuration object, containing paths, model, and training settings.
        """
        self.config: Config = config

        # Initialize tokenizer from the specified path
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(self.config.data.tokenizer_path)

        # Ensure tokenizer has a pad_token, which is crucial for padding sequences.
        # For many decoder-only models, EOS token is often used as pad_token.
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"Tokenizer's pad_token was None, setting it to eos_token: '{self.tokenizer.eos_token}'")
            else:
                # Fallback if neither pad_token nor eos_token is available.
                # Adding a new special token might alter vocabulary size or require model re-embedding.
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                print("Warning: No pad_token or eos_token found. Added '[PAD]' as pad_token.")
        
        # Store configuration parameters for easier access
        self.max_seq_len: int = self.config.model.max_seq_len
        self.global_batch_size: int = self.config.training.global_batch_size
        self.train_data_paths: List[str] = self.config.data.train_data_paths
        self.eval_data_paths: List[str] = self.config.data.eval_data_paths
        
        # Number of workers for data loading. Set to 0 for simplicity/compatibility
        # with distributed training frameworks like Accelerate/FSDP to avoid deadlocks.
        # Can be configured via config.yaml if needed for performance.
        self._num_workers: int = 0 

    def _load_and_tokenize(self, file_paths: List[str], max_seq_len: int) -> Dataset:
        """
        A private helper method to load raw text data, tokenize it according to the
        specified sequence length, and convert it into a `torch.utils.data.Dataset`.

        Assumes files are either plain text (one sample per line) or JSONL (each line
        is a JSON object with a 'text' key).

        Args:
            file_paths: A list of strings, where each string is a path to a data file.
            max_seq_len: An integer specifying the maximum sequence length for tokenization.

        Returns:
            A `torch.utils.data.TensorDataset` containing tokenized `input_ids`,
            `attention_mask`, and `labels`.
        
        Raises:
            ValueError: If no text data is successfully loaded from the provided paths.
        """
        all_texts: List[str] = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"Warning: Data file not found: '{file_path}'. Skipping.")
                continue

            try:
                # Determine file type based on extension, otherwise assume plain text
                if file_path.endswith('.jsonl'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            data = json.loads(line)
                            if 'text' in data:
                                all_texts.append(data['text'])
                            else:
                                print(f"Warning: JSONL line in '{file_path}' missing 'text' key: {line.strip()}")
                else: # Default to plain text, one sample per line
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            stripped_line = line.strip()
                            if stripped_line: # Only add non-empty lines
                                all_texts.append(stripped_line)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from '{file_path}': {e}. Treating as plain text.")
                # Attempt to read as plain text if JSON fails
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        stripped_line = line.strip()
                        if stripped_line:
                            all_texts.append(stripped_line)
            except Exception as e:
                print(f"Error loading data from '{file_path}': {e}")
        
        if not all_texts:
            raise ValueError(f"No text data found in the provided file paths: {file_paths}. "
                             "Please ensure files exist and contain valid data.")

        # Tokenize all collected texts
        # `truncation=True` cuts sequences longer than `max_length`.
        # `padding="max_length"` pads sequences shorter than `max_length` to the full length.
        tokenized_data = self.tokenizer(
            all_texts,
            max_length=max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt", # Return PyTorch tensors
        )

        input_ids: torch.Tensor = tokenized_data["input_ids"]
        attention_mask: torch.Tensor = tokenized_data["attention_mask"]

        # Prepare labels for causal language modeling.
        # Labels are simply the input_ids shifted one position to the left.
        labels: torch.Tensor = input_ids.clone()
        # Set padding tokens in labels to -100 so that `nn.CrossEntropyLoss` ignores them.
        labels[labels == self.tokenizer.pad_token_id] = -100

        return TensorDataset(input_ids, attention_mask, labels)

    def get_train_dataloader(self) -> DataLoader:
        """
        Creates and returns a `torch.utils.data.DataLoader` for the standard training phase.

        Returns:
            A configured `torch.utils.data.DataLoader` for training.
        """
        print(f"Preparing training dataloader for sequence length: {self.max_seq_len}")
        train_dataset: Dataset = self._load_and_tokenize(self.train_data_paths, self.max_seq_len)
        
        return DataLoader(
            train_dataset,
            batch_size=self.global_batch_size,
            shuffle=True,          # Randomize data order for training
            drop_last=True,        # Ensures all batches have the same size, important for distributed training
            num_workers=self._num_workers,
            pin_memory=True        # Speeds up data transfer to GPU
        )

    def get_eval_dataloader(self) -> DataLoader:
        """
        Creates and returns a `torch.utils.data.DataLoader` for evaluating the model on
        standard evaluation benchmarks.

        Returns:
            A configured `torch.utils.data.DataLoader` for evaluation.
        """
        print(f"Preparing evaluation dataloader for sequence length: {self.max_seq_len}")
        eval_dataset: Dataset = self._load_and_tokenize(self.eval_data_paths, self.max_seq_len)
        
        return DataLoader(
            eval_dataset,
            batch_size=self.global_batch_size, # Use global_batch_size for evaluation
            shuffle=False,         # Maintain consistent order for evaluation
            drop_last=False,       # Process all samples, even if the last batch is smaller
            num_workers=self._num_workers,
            pin_memory=True
        )

    def get_long_context_dataloader(self) -> DataLoader:
        """
        Creates and returns a `torch.utils.data.DataLoader` specifically for the
        long-context continued training phase, utilizing a larger sequence length.

        Returns:
            A configured `torch.utils.data.DataLoader` for long-context training.
        
        Raises:
            RuntimeError: If long-context extension is not enabled in the configuration.
        """
        if not self.config.long_context_extension.enabled:
            raise RuntimeError(
                "Long-context extension is not enabled in the configuration. "
                "Cannot create long-context dataloader without it being enabled."
            )
        
        extended_seq_len: int = self.config.long_context_extension.extended_seq_len
        print(f"Preparing long-context dataloader for extended sequence length: {extended_seq_len}")
        
        # The paper implies using the same training data source but re-tokenized
        long_context_dataset: Dataset = self._load_and_tokenize(self.train_data_paths, extended_seq_len)
        
        return DataLoader(
            long_context_dataset,
            batch_size=self.config.training.global_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self._num_workers,
            pin_memory=True
        )
