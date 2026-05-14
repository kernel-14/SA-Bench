# dataset_loader.py

import os
import json
import numpy as np
from typing import List, Dict
from configparser import ConfigParser


class DatasetLoader:
    """
    DatasetLoader handles the loading and preprocessing of the Boxoban dataset
    for Sokoban experiments. It ensures the symbolic inputs are correctly formatted
    as (8x8x7) tensors and splits the data into training and validation subsets.
    """

    def __init__(self, dataset_path: str = "./data/boxoban", config: Dict = None):
        """
        Initializes the DatasetLoader with dataset path and configuration settings.

        Args:
            dataset_path (str): Path to the Boxoban dataset directory.
            config (dict): Configuration dictionary loaded from config.yaml.
        """
        self.dataset_path = dataset_path
        self.training_levels = config.get("dataset", {}).get(
            "training_levels", 900000
        )
        self.validation_levels = config.get("dataset", {}).get(
            "validation_levels", 1000
        )
        self.input_format = config.get("dataset", {}).get(
            "format", "8x8x7"
        )  # Ensure format is standard
        self.grid_size = config["environment"]["grid_size"]  # Typically 8
        self._validate_dataset_path()

        # Internal placeholders for preloaded splits
        self.training_data = []
        self.validation_data = []

    def _validate_dataset_path(self):
        """Ensures the dataset path exists and is accessible."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Dataset path {self.dataset_path} does not exist. "
                f"Please check your configuration."
            )

    def load_training_data(self) -> List[Dict]:
        """
        Loads and preprocesses the training subset of Boxoban levels.

        Returns:
            List[Dict]: A list of symbolic training episodes formatted (8x8x7).
        """
        training_files = self._get_data_files(split="train")
        self.training_data = self._load_and_validate_data(
            training_files, max_levels=self.training_levels
        )
        return self.training_data

    def load_validation_data(self) -> List[Dict]:
        """
        Loads and preprocesses the validation subset of Boxoban levels.

        Returns:
            List[Dict]: A list of symbolic validation episodes formatted (8x8x7).
        """
        validation_files = self._get_data_files(split="validation")
        self.validation_data = self._load_and_validate_data(
            validation_files, max_levels=self.validation_levels
        )
        return self.validation_data

    def _get_data_files(self, split: str) -> List[str]:
        """
        Retrieves a list of files corresponding to the dataset split.

        Args:
            split (str): Either 'train' or 'validation'.

        Returns:
            List[str]: List of file paths.
        """
        split_path = os.path.join(self.dataset_path, split)
        if not os.path.exists(split_path):
            raise FileNotFoundError(
                f"The {split} dataset path {split_path} does not exist."
            )

        # Collect all JSON files in the directory
        all_files = [
            os.path.join(split_path, f)
            for f in os.listdir(split_path)
            if f.endswith(".json")
        ]
        if not all_files:
            raise FileNotFoundError(
                f"No {split} files (.json) were found in {split_path}."
            )
        return all_files

    def _load_and_validate_data(
        self, file_list: List[str], max_levels: int
    ) -> List[Dict]:
        """
        Loads levels from a list of files and validates their format.

        Args:
            file_list (List[str]): List of file paths containing the levels.
            max_levels (int): Maximum number of levels to load.

        Returns:
            List[Dict]: A list of validated symbolic episodes.
        """
        loaded_data = []
        for file_path in file_list:
            with open(file_path, "r") as f:
                levels = json.load(f)

            for level in levels:
                if len(loaded_data) >= max_levels:
                    break

                # Validate and process the level
                if self._validate_level_format(level):
                    symbolic_level = self._process_to_symbolic(level)
                    loaded_data.append(symbolic_level)

            if len(loaded_data) >= max_levels:
                break

        return loaded_data

    def _validate_level_format(self, level: Dict) -> bool:
        """
        Validates if a level conforms to the symbolic (8x8x7) format.

        Args:
            level (Dict): The raw level data.

        Returns:
            bool: True if valid, False otherwise.
        """
        # Check grid size
        if (
            "board" not in level
            or len(level["board"]) != self.grid_size
            or len(level["board"][0]) != self.grid_size
        ):
            return False

        # Ensure one-hot encoding in the symbolic format
        for row in level["board"]:
            for cell in row:
                if not isinstance(cell, list) or len(cell) != 7:
                    return False
                if np.sum(cell) != 1:  # Exactly one active state in one-hot
                    return False

        return True

    def _process_to_symbolic(self, level: Dict) -> Dict:
        """
        Processes level data into a symbolic representation.

        Args:
            level (Dict): The raw level data.

        Returns:
            Dict: The symbolic level suitable for training (8x8x7 structure).
        """
        board_array = np.array(level["board"], dtype=np.float32)
        return {"symbolic_board": board_array, "level_metadata": level.get("metadata")}


# Sample usage
if __name__ == "__main__":
    # Loading config from config.yaml (parsed as dict)
    import yaml

    config_path = "./config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset_loader = DatasetLoader(dataset_path="./data/boxoban", config=config)
    training_data = dataset_loader.load_training_data()
    validation_data = dataset_loader.load_validation_data()

    print(f"Loaded {len(training_data)} training episodes.")
    print(f"Loaded {len(validation_data)} validation episodes.")
