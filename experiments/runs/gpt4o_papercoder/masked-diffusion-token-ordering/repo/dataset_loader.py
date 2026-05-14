## dataset_loader.py
import numpy as np
import torch
from datasets import load_dataset
import os
from typing import Dict, Tuple

class DatasetLoader:
    def __init__(self, config: dict):
        """
        Initializes DatasetLoader with configuration parameters.

        Args:
            config (dict): Dictionary containing configuration parameters.
        """
        self.config = config
        # Load paths and settings from configuration
        self.dataset_paths = {
            "slim_pajama": self.config["dataset"].get("slim_pajama_path", "path/to/slim_pajama"),
            "sudoku_train": self.config["dataset"].get("sudoku_train_path", "path/to/sudoku_train"),
            "sudoku_test": self.config["dataset"].get("sudoku_test_path", "path/to/sudoku_test"),
            "zebra_train": self.config["dataset"].get("zebra_train_path", "path/to/zebra_train"),
            "zebra_test": self.config["dataset"].get("zebra_test_path", "path/to/zebra_test")
        }
        self.sequence_length = self.config["dataset"].get("sequence_length", 2048)
        self.masking_schedule = {
            "alpha_start": self.config["dataset"]["masking_schedule"].get("alpha_start", 1.0),
            "alpha_end": self.config["dataset"]["masking_schedule"].get("alpha_end", 0.0)
        }

    def load_data(self) -> Dict[str, torch.utils.data.Dataset]:
        """
        Loads datasets from specified paths.

        Returns:
            dict: Dictionary containing Torch datasets.
        """
        datasets = {}
        
        # SlimPajama dataset
        if os.path.exists(self.dataset_paths["slim_pajama"]):
            text_data = load_dataset("text", data_files=self.dataset_paths["slim_pajama"])
            datasets["slim_pajama"] = self._process_text_data(text_data)

        # Sudoku training and testing datasets
        if os.path.exists(self.dataset_paths["sudoku_train"]):
            sudoku_train_data = np.loadtxt(self.dataset_paths["sudoku_train"], delimiter=",", dtype=str)
            sudoku_test_data = np.loadtxt(self.dataset_paths["sudoku_test"], delimiter=",", dtype=str)
            datasets["sudoku_train"] = self._process_puzzle_data(sudoku_train_data, format="sudoku")
            datasets["sudoku_test"] = self._process_puzzle_data(sudoku_test_data, format="sudoku")

        # Zebra training and testing datasets
        if os.path.exists(self.dataset_paths["zebra_train"]):
            zebra_train_data = np.load(self.dataset_paths["zebra_train"], allow_pickle=True)
            zebra_test_data = np.load(self.dataset_paths["zebra_test"], allow_pickle=True)
            datasets["zebra_train"] = self._process_puzzle_data(zebra_train_data, format="zebra")
            datasets["zebra_test"] = self._process_puzzle_data(zebra_test_data, format="zebra")

        return datasets

    def preprocess_data(self, dataset: Dict[str, torch.utils.data.Dataset]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Preprocesses datasets by applying uniform masking and formatting for Masked Diffusion training.

        Args:
            dataset (Dict[str, torch.utils.data.Dataset]): Raw loaded datasets.

        Returns:
            Dict[str, Dict[str, torch.Tensor]]: Preprocessed datasets with masking applied.
        """
        processed_datasets = {}

        for name, raw_dataset in dataset.items():
            processed_datasets[name] = {
                "input_sequences": raw_dataset["input_sequences"],
                "masked_sequences": [],
                "metadata": []
            }

            for sequence in raw_dataset["input_sequences"]:
                masked_sequence, mask_metadata = self._apply_masking(sequence)
                processed_datasets[name]["masked_sequences"].append(masked_sequence)
                processed_datasets[name]["metadata"].append(mask_metadata)

        # Convert lists to Tensors
        for name in processed_datasets.keys():
            processed_datasets[name]["masked_sequences"] = torch.stack(processed_datasets[name]["masked_sequences"])
            processed_datasets[name]["metadata"] = torch.stack(processed_datasets[name]["metadata"])

        return processed_datasets

    def _process_text_data(self, dataset: Dict) -> Dict[str, torch.Tensor]:
        """
        Processes SlimPajama text data by tokenizing, deduplicating, and batching.

        Args:
            dataset (Dict): Text dataset loaded from HuggingFace.

        Returns:
            Dict[str, torch.Tensor]: Tokenized and batched text data.
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")  # Example tokenizer
        tokenized_sequences = []

        # Tokenize and preprocess sequences
        for example in dataset["train"]:  # Assuming usage of 'train' split
            tokens = tokenizer(example["text"], truncation=True, max_length=self.sequence_length)["input_ids"]
            tokenized_sequences.append(torch.tensor(tokens))

        return {"input_sequences": tokenized_sequences}

    def _process_puzzle_data(self, raw_data: np.ndarray, format: str) -> Dict[str, torch.Tensor]:
        """
        Processes Sudoku/Zebra puzzle data into sequences suited for Masked Diffusion training.

        Args:
            raw_data (np.ndarray): Raw data from .csv or .json files.
            format (str): Format of puzzles ('sudoku' or 'zebra').

        Returns:
            Dict[str, torch.Tensor]: Preprocessed puzzle data.
        """
        processed_sequences = []

        # Reshape and format puzzles into sequences
        if format == "sudoku":
            for row in raw_data:
                sequence = np.array([int(cell) if cell.isdigit() else 0 for cell in row]).flatten()
                processed_sequences.append(torch.tensor(sequence[:self.sequence_length]))
        elif format == "zebra":
            for puzzle in raw_data:
                sequence = np.array(puzzle).flatten()  # Assume Zebra puzzles are already tokenized
                processed_sequences.append(torch.tensor(sequence[:self.sequence_length]))

        return {"input_sequences": processed_sequences}

    def _apply_masking(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies uniform token masking based on the masking schedule.

        Args:
            sequence (torch.Tensor): Input token sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Masked sequence and metadata about mask positions.
        """
        alpha_start = self.masking_schedule["alpha_start"]
        alpha_end = self.masking_schedule["alpha_end"]

        # Linear masking schedule for probabilities
        mask_prob = np.linspace(alpha_end, alpha_start, sequence.shape[0])
        mask_positions = torch.rand(sequence.shape).lt(mask_prob)
        
        masked_sequence = sequence.clone()
        masked_sequence[mask_positions] = 0  # Replace with mask token (e.g., `0`)
        
        return masked_sequence, mask_positions
