"""
dataset_loader.py: A module for loading and preprocessing datasets required for 
pretraining and fine-tuning stages in NaViL experiments.

Dependencies:
- PyTorch: For dataset and tensor manipulation.
- Configuration: Loads dataset paths and preprocessing configurations from config.py.
- Utils: Provides helper functions like multiscale packing.

Design:
- Implements `DatasetLoader` class, adhering to predefined architecture.
- Supports high-level functions for pretraining and fine-tuning dataset loading.
"""

import os
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
from typing import List, Dict, Any
import utils

class DatasetLoader:
    """
    Handles loading and preprocessing datasets for NaViL experiments.
    
    Attributes:
        config (dict): Configuration dictionary fetched from config.yaml.
        pretraining_datasets (List[str]): Pretraining datasets specified in the configuration.
        finetuning_datasets (List[str]): Finetuning datasets specified in the configuration.
        batch_sizes (Dict[str, int]): Batch size settings for pretraining and fine-tuning stages.
        preprocessing_config (Dict[Any, Any]): Preprocessing settings for multiscale packing.
    """

    def __init__(self, config: dict):
        """
        Initializes DatasetLoader by setting up dataset paths and preprocessing configurations.

        Args:
            config (dict): Configuration dictionary loaded from config.yaml.
        
        Raises:
            KeyError: If mandatory configuration sections are missing for datasets.
        """
        self.config = config
        self.pretraining_datasets = self.config.get("data", {}).get("pretraining_datasets", [])
        self.finetuning_datasets = self.config.get("data", {}).get("finetuning_datasets", [])
        
        if not self.pretraining_datasets or not self.finetuning_datasets:
            raise KeyError("Dataset configurations missing in 'data' section of config.yaml.")
        
        self.batch_sizes = {
            "pretraining": self.config.get("training", {}).get("pretraining", {}).get("batch_size", 7000),
            "finetuning": self.config.get("training", {}).get("finetuning", {}).get("batch_size", 2500)
        }
        
        self.preprocessing_config = {
            "tau": self.config.get("model", {}).get("visual_encoder", {}).get("patch_size", 16),
            "area_threshold": 500,
            "scales": [1.0, 0.7, 0.5]  # Default scales for multiscale packing.
        }

    def load_pretraining_data(self) -> Dataset:
        """
        Load noisy image-text datasets for pretraining.

        Returns:
            torch.utils.data.Dataset: Preprocessed Dataset object ready for pretraining.
        
        Notes:
            - Applies synthetic captions using pre-trained MLLMs if necessary.
            - Preprocesses images using multiscale packing and patch embedding.
        """
        dataset_list = []

        for dataset_name in self.pretraining_datasets:
            data_path = os.path.join("datasets/pretraining", dataset_name)  # Placeholder path
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Pretraining dataset not found at {data_path}.")
            
            print(f"Loading pretraining dataset: {dataset_name}")
            raw_data = self._load_raw_data(data_path)
            preprocessed_data = self._preprocess_multimodal_data(
                raw_data,
                alignment_required=True
            )
            dataset_list.append(preprocessed_data)

        # Combine multiple datasets into a unified PyTorch Dataset
        combined_dataset = TensorDataset(*zip(*dataset_list))
        return combined_dataset

    def load_finetuning_data(self) -> Dataset:
        """
        Load high-quality multimodal datasets for fine-tuning.

        Returns:
            torch.utils.data.Dataset: Preprocessed Dataset object ready for fine-tuning.

        Notes:
            - Ensures inclusion of diverse modalities like charts, documents, and dialogs.
            - Applies multiscale packing and other preprocessing methods.
        """
        dataset_list = []
        
        for dataset_name in self.finetuning_datasets:
            data_path = os.path.join("datasets/finetuning", dataset_name)  # Placeholder path
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Fine-tuning dataset not found at {data_path}.")
            
            print(f"Loading fine-tuning dataset: {dataset_name}")
            raw_data = self._load_raw_data(data_path)
            preprocessed_data = self._preprocess_multimodal_data(
                raw_data,
                alignment_required=True
            )
            dataset_list.append(preprocessed_data)

        # Combine multiple datasets into a unified PyTorch Dataset
        combined_dataset = TensorDataset(*zip(*dataset_list))
        return combined_dataset

    def _load_raw_data(self, data_path: str) -> List[Dict[str, Any]]:
        """
        Loads raw data from the specified path. Supports image-text pair format.

        Args:
            data_path (str): Path to the data directory/file.

        Returns:
            List[Dict[str, Any]]: List of raw data samples.
        
        Notes:
            - Assumes data is stored in a CSV format for simplicity.
            - Each row contains an image path and a text caption.
        """
        raw_data = []
        with open(data_path, "r") as file:
            for line in file:
                image_path, caption = line.strip().split(",")
                raw_data.append({"image": torch.tensor(torch.load(image_path)), "text": caption})
        return raw_data

    def _preprocess_multimodal_data(self, raw_data: List[Dict[str, Any]], alignment_required: bool = False) -> List[Dict[str, torch.Tensor]]:
        """
        Preprocess multimodal data by applying visual multiscale packing, embedding, and alignment.

        Args:
            raw_data (List[Dict[str, Any]]): List of raw data samples containing image and text.
            alignment_required (bool): Whether visual-text alignment is required.

        Returns:
            List[Dict[str, torch.Tensor]]: Preprocessed multimodal samples ready for training.
        """
        preprocessed_data = []
        
        for sample in raw_data:
            image = sample["image"]
            text = sample["text"]

            # Apply visual multiscale packing
            visual_tokens = utils.process_visual_multiscale(
                image, 
                scales=self.preprocessing_config["scales"], 
                tau=self.preprocessing_config["tau"], 
                area_threshold=self.preprocessing_config["area_threshold"]
            )

            # Align visual tokens to LLM dimensions
            if alignment_required:
                visual_tokens = utils.align_visual_to_llm(visual_tokens, self.config["model"]["llm"]["width"])

            # Prepare text tokens (assuming text tokenization utility available)
            text_tokens = torch.tensor([ord(t) for t in text])  # Placeholder token generation.

            preprocessed_data.append({"visual_tokens": visual_tokens, "text_tokens": text_tokens})

        return preprocessed_data

if __name__ == "__main__":
    try:
        # Example usage with config loaded dynamically
        from config import Config
        
        config = Config("config.yaml").get_config()
        dataset_loader = DatasetLoader(config)

        pretraining_dataset = dataset_loader.load_pretraining_data()
        print(f"Pretraining Dataset Loaded: {pretraining_dataset}")

        finetuning_dataset = dataset_loader.load_finetuning_data()
        print(f"Fine-tuning Dataset Loaded: {finetuning_dataset}")
    except Exception as e:
        print(f"Error: {e}")
