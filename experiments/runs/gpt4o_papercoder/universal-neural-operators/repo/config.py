"""config.py

Module to handle configuration management for the project. Provides functionality to load, validate, and save experiment configurations from a YAML file.

Imports:
    yaml: For parsing and serializing YAML files.
    os: For checking file and path validity.
    typing: For type annotations.
"""

# Required imports
import yaml
import os
from typing import Dict, Any


class Config:
    """A class to handle experiment configurations with utilities for loading, saving, and validating configurations.

    Attributes:
        config (dict): The loaded configuration dictionary.
    """

    def __init__(self, file_path: str):
        """Initialize the Config object by loading and validating the configuration.

        Args:
            file_path (str): The path to the configuration YAML file.

        Raises:
            FileNotFoundError: If the YAML file doesn't exist.
            ValueError: If the configuration is invalid.
        """
        self.file_path = file_path
        self.config = self.get_config(file_path)
        self.validate_config(self.config)

    def get_config(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from a YAML file.

        Args:
            file_path (str): Path to the configuration file.

        Returns:
            dict: A dictionary containing the loaded configuration.

        Raises:
            FileNotFoundError: If the configuration file is missing.
            yaml.YAMLError: If the file cannot be parsed properly.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The configuration file '{file_path}' does not exist. Please provide a valid path.")

        with open(file_path, "r") as file:
            try:
                config = yaml.safe_load(file)
                return config
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing the configuration file '{file_path}': {e}")

    def save_config(self, config: Dict[str, Any], file_path: str) -> None:
        """Save the configuration dictionary to a YAML file.

        Args:
            config (dict): Configuration dictionary to be saved.
            file_path (str): Path to save the YAML file.

        Raises:
            ValueError: If the input configuration is not a dictionary.
            IOError: If there's an issue writing to the file.
        """
        if not isinstance(config, dict):
            raise ValueError(f"Invalid configuration format. Expected a dictionary but got {type(config)}.")

        try:
            with open(file_path, "w") as file:
                yaml.dump(config, file, default_flow_style=False)
        except IOError as e:
            raise IOError(f"Error saving the configuration file to '{file_path}': {e}")

    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate the configuration dictionary for completeness and correctness.

        Args:
            config (dict): Configuration dictionary to validate.

        Raises:
            ValueError: If any required section or parameter is missing or has invalid values.
        """
        required_sections = ["training", "fine_tuning", "model", "datasets", "evaluation"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section '{section}' in configuration.")

        # Validation for 'training' section
        self._validate_training_config(config.get("training"))

        # Validation for 'fine_tuning' section
        self._validate_fine_tuning_config(config.get("fine_tuning"))

        # Validation for 'model' section
        self._validate_model_config(config.get("model"))

        # Validation for 'datasets' section
        self._validate_datasets_config(config.get("datasets"))

        # Validation for 'evaluation' section
        self._validate_evaluation_config(config.get("evaluation"))

    def _validate_training_config(self, training_config: Dict[str, Any]) -> None:
        """Validate the training section of the configuration.

        Args:
            training_config (dict): Training-related configuration.

        Raises:
            ValueError: If required parameters are missing or invalid.
        """
        if not isinstance(training_config, dict):
            raise ValueError("The 'training' section must be a dictionary.")
        
        if not (isinstance(training_config.get("learning_rate"), (float, int)) and training_config["learning_rate"] > 0):
            raise ValueError("Invalid 'learning_rate' in 'training'. It must be a positive number.")
        
        if not (isinstance(training_config.get("batch_size"), int) and training_config["batch_size"] > 0):
            raise ValueError("Invalid 'batch_size' in 'training'. It must be a positive integer.")
        
        if not (isinstance(training_config.get("epochs"), int) and training_config["epochs"] > 0):
            raise ValueError("Invalid 'epochs' in 'training'. It must be a positive integer.")

    def _validate_fine_tuning_config(self, fine_tuning_config: Dict[str, Any]) -> None:
        """Validate the fine-tuning section of the configuration.

        Args:
            fine_tuning_config (dict): Fine-tuning-related configuration.

        Raises:
            ValueError: If required parameters are missing or invalid.
        """
        if not isinstance(fine_tuning_config, dict):
            raise ValueError("The 'fine_tuning' section must be a dictionary.")
        
        if not (isinstance(fine_tuning_config.get("learning_rate"), (float, int)) and fine_tuning_config["learning_rate"] > 0):
            raise ValueError("Invalid 'learning_rate' in 'fine_tuning'. It must be a positive number.")
        
        if not (isinstance(fine_tuning_config.get("batch_size"), int) and fine_tuning_config["batch_size"] > 0):
            raise ValueError("Invalid 'batch_size' in 'fine_tuning'. It must be a positive integer.")
        
        if not (isinstance(fine_tuning_config.get("epochs"), int) and fine_tuning_config["epochs"] > 0):
            raise ValueError("Invalid 'epochs' in 'fine_tuning'. It must be a positive integer.")

    def _validate_model_config(self, model_config: Dict[str, Any]) -> None:
        """Validate the model section of the configuration.

        Args:
            model_config (dict): Model-related configuration.

        Raises:
            ValueError: If required parameters are missing or invalid.
        """
        if not isinstance(model_config, dict):
            raise ValueError("The 'model' section must be a dictionary.")
        
        if model_config.get("type") not in ["Fourier Neural Operator", "Mamba-SSM", "Perceiver IO"]:
            raise ValueError("Invalid 'type' in 'model'. Supported types are 'Fourier Neural Operator', 'Mamba-SSM', and 'Perceiver IO'.")
        
        if not (isinstance(model_config.get("hidden_modes"), int) and model_config["hidden_modes"] > 0):
            raise ValueError("Invalid 'hidden_modes' in 'model'. It must be a positive integer.")
        
        if not (isinstance(model_config.get("layers"), int) and model_config["layers"] > 0):
            raise ValueError("Invalid 'layers' in 'model'. It must be a positive integer.")

    def _validate_datasets_config(self, datasets_config: Dict[str, Any]) -> None:
        """Validate the datasets section of the configuration.

        Args:
            datasets_config (dict): Datasets-related configuration.

        Raises:
            ValueError: If required parameters are missing or invalid.
        """
        if not isinstance(datasets_config, dict):
            raise ValueError("The 'datasets' section must be a dictionary.")
        
        required_paths = ["pretraining_data_path", "fine_tuning_data_path", "test_data_path"]
        for path in required_paths:
            if path not in datasets_config:
                raise ValueError(f"Missing '{path}' in 'datasets'.")
            if datasets_config.get("synthetic_generation") is False and not os.path.exists(datasets_config[path]):
                raise ValueError(f"The dataset path '{datasets_config[path]}' does not exist.")

    def _validate_evaluation_config(self, evaluation_config: Dict[str, Any]) -> None:
        """Validate the evaluation section of the configuration.

        Args:
            evaluation_config (dict): Evaluation-related configuration.

        Raises:
            ValueError: If required parameters are missing or invalid.
        """
        if not isinstance(evaluation_config, dict):
            raise ValueError("The 'evaluation' section must be a dictionary.")
        
        if "metrics" not in evaluation_config or not isinstance(evaluation_config["metrics"], str):
            raise ValueError("Invalid 'metrics' in 'evaluation'. It must be a string.")

    def get_section(self, section_name: str) -> Dict[str, Any]:
        """Retrieve a specific section of the configuration.

        Args:
            section_name (str): The name of the section to retrieve.

        Returns:
            dict: The specified section of the configuration.

        Raises:
            ValueError: If the section is not found in the configuration.
        """
        if section_name not in self.config:
            raise ValueError(f"Section '{section_name}' not found in configuration.")
        return self.config[section_name]

    def update_config(self, section_name: str, key: str, value: Any) -> None:
        """Update a specific value in the configuration.

        Args:
            section_name (str): The section to update.
            key (str): The key within the section to update.
            value (Any): The new value to set.

        Raises:
            ValueError: If the section or key does not exist in the configuration.
        """
        if section_name not in self.config:
            raise ValueError(f"Section '{section_name}' not found in configuration.")
        if key not in self.config[section_name]:
            raise ValueError(f"Key '{key}' not found in section '{section_name}'.")
        self.config[section_name][key] = value
