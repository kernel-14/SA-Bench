# dataset_loader.py
import os
from typing import Tuple, List, Dict
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

class DatasetLoader:
    def __init__(self, config: Dict):
        """Initialize DatasetLoader with configuration.

        Args:
            config (Dict): A dictionary loaded from 'config.yaml'.
        """
        self.config = config
        # Validate and load tokenizer from the Hugging Face library
        tokenizer_name = self.config.get("models", {}).get("tokenizer", "gemma-2b")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # Maximum token lengths for prompts and responses
        self.max_prompt_length = self.config.get("models", {}).get("max_prompt_length", 512)
        self.max_response_length = self.config.get("models", {}).get("max_response_length", 512)

    def load_data(self) -> Tuple[Dataset, Dict]:
        """Load dataset from configuration file paths and split into train/validation.

        Returns:
            Tuple[Dataset, Dict]: PyTorch Dataset containing tokenized data, 
            and metadata dictionary containing dataset stats.
        """
        dataset_paths = self.config.get("training", {}).get("dataset_paths", {})
        split_ratios = self.config.get("training", {}).get("split_ratios", {})

        task_type = self.config.get("training", {}).get("task_type", "tl_dr")
        dataset_path = dataset_paths.get(task_type, None)
        split_ratio = split_ratios.get(task_type, [0.8, 0.2])

        if not dataset_path or not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist!")

        raw_dataset = load_dataset("json", data_files=dataset_path)["train"]
        train_size = int(len(raw_dataset) * split_ratio[0])
        val_size = len(raw_dataset) - train_size

        train_data = raw_dataset.select(range(train_size))
        val_data = raw_dataset.select(range(train_size, len(raw_dataset)))

        metadata = {
            "task_type": task_type,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "split_ratio": split_ratio
        }

        return train_data, metadata

    def tokenize(self, sequences: List[str]) -> List[int]:
        """Tokenize a sequence of strings using the Hugging Face tokenizer.

        Args:
            sequences (List[str]): A list of text strings.

        Returns:
            List[int]: List of token IDs for each string, truncated/padded based on max token length.
        """
        tokenized = self.tokenizer(
            sequences,
            padding="max_length",
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="pt"
        )
        return tokenized["input_ids"]

    def format_supervised_finetuning(self, raw_data: List[Dict]) -> Dataset:
        """Format raw data into SFT-compatible tokenized input-output pairs.

        Args:
            raw_data (List[Dict]): Raw dataset containing prompts and responses.

        Returns:
            Dataset: PyTorch Dataset containing tokenized input-output pairs.
        """
        formatted_data = []
        for entry in raw_data:
            prompt = entry.get("prompt", "")
            response = entry.get("response", "")

            tokenized_prompt = self.tokenize([prompt])[0]
            tokenized_response = self.tokenize([response])[0]

            formatted_data.append({
                "input_ids": tokenized_prompt,
                "output_ids": tokenized_response
            })

        return formatted_data

    def format_reward_model(self, raw_data: List[Dict]) -> Dataset:
        """Format raw data into RM-compatible preferences dataset.

        Args:
            raw_data (List[Dict]): Raw dataset containing preference-labeled data.

        Returns:
            Dataset: PyTorch Dataset formatted for ranking-based reward modeling.
        """
        formatted_data = []
        for entry in raw_data:
            prompt = entry.get("prompt", "")
            chosen_response = entry.get("chosen_response", "")
            rejected_response = entry.get("rejected_response", "")

            tokenized_prompt = self.tokenize([prompt])[0]
            tokenized_chosen = self.tokenize([chosen_response])[0]
            tokenized_rejected = self.tokenize([rejected_response])[0]

            formatted_data.append({
                "prompt_ids": tokenized_prompt,
                "chosen_token_ids": tokenized_chosen,
                "rejected_token_ids": tokenized_rejected
            })

        return formatted_data

    def format_rlhf(self, raw_data: List[Dict]) -> Dataset:
        """Format raw data into RLHF-compatible sequences with macro actions.

        Args:
            raw_data (List[Dict]): Raw dataset containing response sequences.

        Returns:
            Dataset: PyTorch Dataset formatted for RLHF training.
        """
        formatted_data = []
        for entry in raw_data:
            prompt = entry.get("prompt", "")
            generated_response = entry.get("generated_response", "")

            tokenized_prompt = self.tokenize([prompt])[0]
            tokenized_response = self.tokenize([generated_response])[0]

            # Additional processing for macro actions will occur in Termination class
            # For simplicity, assume straight tokenization here:
            formatted_data.append({
                "input_ids": tokenized_prompt,
                "output_ids": tokenized_response
            })

        return formatted_data

    def batchify(self, data: List[Dict], batch_size: int) -> DataLoader:
        """Create a PyTorch DataLoader for batch processing tokenized dataset.

        Args:
            data (List[Dict]): List of tokenized data dictionaries.
            batch_size (int): Number of samples per batch.

        Returns:
            DataLoader: PyTorch DataLoader for batch iteration.
        """
        tensor_data = [
            {
                "input_ids": torch.tensor(item["input_ids"]),
                "output_ids": torch.tensor(item["output_ids"])
            }
            for item in data
        ]
        return DataLoader(tensor_data, batch_size=batch_size, shuffle=True)


# Example usage
if __name__ == "__main__":
    # Load configuration from config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset_loader = DatasetLoader(config)
    train_dataset, metadata = dataset_loader.load_data()

    print(f"Loaded {metadata['train_size']} training samples and {metadata['val_size']} validation samples.")
    print(metadata)
