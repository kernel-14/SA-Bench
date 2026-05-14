# config.py

import yaml
from typing import Any, Dict, Union


class Config:
    """Config Loader for Centralized Project Configuration"""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """
        Initializes the Config object by loading values from a YAML file.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Loads and parses the YAML configuration file.

        Returns:
            dict: Dictionary containing all configuration parameters from the YAML file.
        """
        try:
            with open(self.config_path, "r") as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing the YAML configuration file: {e}")

    def get_training_config(self) -> Dict[str, Union[int, float, str]]:
        """
        Returns the training configuration.

        Returns:
            dict: Training-related hyperparameters and settings.
        """
        return self.config.get("training", {})

    def get_dataset_config(self) -> Dict[str, Any]:
        """
        Returns the dataset configuration.

        Returns:
            dict: Dataset paths and preprocessing parameters for images and videos.
        """
        return self.config.get("dataset", {})

    def get_model_config(self) -> Dict[str, Any]:
        """
        Returns the model configuration.

        Returns:
            dict: Model architecture details for VAE and Flow Matching Models.
        """
        return self.config.get("model", {})

    def get_evaluation_config(self) -> Dict[str, Any]:
        """
        Returns the evaluation configuration.

        Returns:
            dict: Parameters and paths for benchmarks and metrics.
        """
        return self.config.get("evaluation", {})

    def get_resource_config(self) -> Dict[str, Union[int, str]]:
        """
        Returns the hardware resource configuration.

        Returns:
            dict: GPU allocations and estimated training time per stage.
        """
        return self.config.get("resources", {})


# Example Usage:
# config = Config("config.yaml")
# training_config = config.get_training_config()
# dataset_config = config.get_dataset_config()
# model_config = config.get_model_config()
# evaluation_config = config.get_evaluation_config()
# resource_config = config.get_resource_config()
