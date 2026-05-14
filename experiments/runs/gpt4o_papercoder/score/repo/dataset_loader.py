# dataset_loader.py

import os
import torch
from typing import Tuple, Dict
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer
from torch.utils.data import Dataset as TorchDataset
from utils import load_config, set_seed


class DatasetLoader:
    """
    Handles data loading and preprocessing for SCoRe methodology on MATH, HumanEval, and MBPP datasets.
    Includes tokenization, instruction-guided prompt creation, and data formatting.
    """

    def __init__(self, config: Dict):
        """
        Initializes the DatasetLoader with configuration parameters.

        Args:
            config: Dictionary containing dataset paths, tokenizer details, task-specific prompts, and preprocessing settings.
        """
        self.config = config
        self.seed = config['logging']['seed']
        self.math_data_path = config['datasets']['math']
        self.humaneval_data_path = config['datasets']['humaneval']
        self.models_config = config['models']
        set_seed(self.seed)  # Ensures reproducible data splits

        # Initialize tokenizer based on pre-trained model for language understanding
        self.tokenizer = AutoTokenizer.from_pretrained(self.models_config['reasoning_model'], use_fast=True)

    def load_data(self) -> Tuple[Dataset, Dataset]:
        """
        Loads the training and testing datasets from the specified paths in the configuration.

        Returns:
            A tuple containing the training and test datasets as Hugging Face Dataset objects.
        """
        # Load MATH dataset
        math_train = load_dataset('json', data_files=self.math_data_path['train_path'])['train']
        math_test = load_dataset('json', data_files=self.math_data_path['test_path'])['train']

        # Validate number of records align with expectations
        if len(math_train) < self.math_data_path['num_train_problems']:
            raise ValueError("Insufficient training dataset size in MATH. Check config.")
        if len(math_test) < self.math_data_path['num_test_problems']:
            raise ValueError("Insufficient testing dataset size in MATH. Check config.")

        return (math_train, math_test)

    def preprocess_data(self, dataset: Dataset) -> TorchDataset:
        """
        Preprocesses the dataset by applying tokenization, instruction-guided prompting, 
        and formatting for multi-turn RL training.

        Args:
            dataset: Hugging Face Dataset object loaded via load_data.

        Returns:
            PyTorch Dataset object containing tokenized sequences and labels.
        """

        def process_math(example):
            """
            Processes a single example in the MATH dataset to generate zero-shot prompts and tokenized sequences.
            
            Args:
                example: Individual example containing problem text and solution.

            Returns:
                A dictionary with tokenized input and output sequences.
            """
            # Problem and solution
            problem = example["Problem"]
            solution = example["Solution"]

            # Create prompt for the first response
            first_turn_prompt = f"You are a math expert. Solve the following problem:\n{problem}\nAnswer:"
            first_turn_encoded = self.tokenizer(
                first_turn_prompt, 
                truncation=True, 
                padding="max_length", 
                max_length=self.tokenizer.model_max_length
            )

            # Add self-correction instructions for the second response
            second_turn_prompt = (
                f"{first_turn_prompt} Please check and correct any mistakes in your initial answer if there are any.\nSolution:"
            )
            second_turn_encoded = self.tokenizer(
                second_turn_prompt, 
                truncation=True, 
                padding="max_length", 
                max_length=self.tokenizer.model_max_length
            )

            # Ground truth solution tokenization
            solution_encoded = self.tokenizer(
                solution, 
                truncation=True, 
                padding="max_length", 
                max_length=self.tokenizer.model_max_length
            )

            return {
                "input_ids_1": first_turn_encoded.input_ids,
                "attention_mask_1": first_turn_encoded.attention_mask,
                "input_ids_2": second_turn_encoded.input_ids,
                "attention_mask_2": second_turn_encoded.attention_mask,
                "labels": solution_encoded.input_ids
            }

        # Apply the processing to the full dataset
        processed_dataset = dataset.map(process_math, batched=False)

        # Convert processed dataset into PyTorch Dataset
        class TorchDatasetWrapper(TorchDataset):
            def __init__(self, hf_dataset):
                self.dataset = hf_dataset

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                item = self.dataset[idx]
                # Convert necessary fields to tensors
                return {
                    "input_ids_1": torch.tensor(item["input_ids_1"], dtype=torch.long),
                    "attention_mask_1": torch.tensor(item["attention_mask_1"], dtype=torch.long),
                    "input_ids_2": torch.tensor(item["input_ids_2"], dtype=torch.long),
                    "attention_mask_2": torch.tensor(item["attention_mask_2"], dtype=torch.long),
                    "labels": torch.tensor(item["labels"], dtype=torch.long)
                }

        return TorchDatasetWrapper(processed_dataset)
