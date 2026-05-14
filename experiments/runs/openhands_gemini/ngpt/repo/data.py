
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

class OpenWebTextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer_name='llama-2', block_size=1024):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # Resize model embeddings if using this tokenizer with a model
            # model.resize_token_embeddings(len(tokenizer))

        self.block_size = block_size
        self.dataset = load_dataset("openwebtext", split="train")

        # Process the dataset once to tokenize and concatenate
        self.tokenized_data = self.dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=self.dataset.column_names,
            desc="Tokenizing and concatenating text",
            batch_size=1000, # Process in batches for efficiency
        )

        # Concatenate all texts and split into fixed-size blocks
        # This part requires careful handling of concatenated sequences
        self.all_tokens = []
        for i in range(0, len(self.tokenized_data), 1000): # Process in chunks
            chunk = self.tokenized_data[i:i+1000]
            for tokens in chunk['input_ids']:
                self.all_tokens.extend(tokens)

        # Create blocks
        self.examples = []
        for i in range(0, len(self.all_tokens) - block_size + 1, block_size):
            self.examples.append(self.all_tokens[i : i + block_size])

    def _tokenize_function(self, examples):
        return self.tokenizer(examples["text"], return_attention_mask=False)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        chunk = self.examples[idx]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        return input_ids, labels

def get_dataloader(tokenizer_name='llama-2', block_size=1024, batch_size=64, shuffle=True, num_workers=0):
    dataset = OpenWebTextDataset(tokenizer_name=tokenizer_name, block_size=block_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True, # For faster data transfer to GPU
    )
    return dataloader

