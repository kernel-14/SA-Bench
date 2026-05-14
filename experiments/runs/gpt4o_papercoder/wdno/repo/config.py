# config.py
"""
Config module for managing experiment configurations.
This module allows loading, validation, and programmatic access to a centrally-defined YAML configuration file.
"""

import os
from typing import Any, Dict, Optional
import yaml

class Config:
    """
    Handles configuration loading, validation, and nested key-based access.

    Attributes:
        file_path (str): Path to the configuration YAML file.
        config_data (dict): Parsed YAML configuration stored as a nested dictionary.
    """

    def __init__(self, file_path: str = "config.yaml") -> None:
        """
        Initializes the Config instance by loading and validating the configuration file.

        Args:
            file_path (str): Path to the configuration YAML file. Defaults to "config.yaml".

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            RuntimeError: If the YAML file cannot be parsed or validated.
        """
        self.file_path = file_path
        self.config_data = self._load_config(file_path)
        self._validate_config()

    def _load_config(self, file_path: str) -> Dict[str, Any]:
        """
        Loads the configuration YAML file into a nested dictionary.

        Args:
            file_path (str): Path to the YAML configuration file.

        Returns:
            dict: The configuration as a nested dictionary.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            yaml.YAMLError: If the YAML file is malformed.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found at `{file_path}`.")
        
        try:
            with open(file_path, "r") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing YAML configuration: {e}")
        
        return config

    def _validate_config(self) -> None:
        """
        Validates essential keys in the configuration dictionary.

        Raises:
            KeyError: If any required top-level keys are missing.
        """
        required_keys = ["training", "wavelet", "data", "evaluation", "hardware"]
        for key in required_keys:
            if key not in self.config_data:
                raise KeyError(f"Missing required configuration section: `{key}`.")

        # Validate nested keys for critical sections
        required_nested_keys = {
            "training": ["learning_rate", "batch_size", "epochs", "scheduler"],
            "wavelet": ["basis_1d", "mode_1d", "basis_2d", "mode_2d"],
            "data": ["burgers_equation", "navier_stokes", "fluid_2D"],
        }
        
        for section, keys in required_nested_keys.items():
            if section not in self.config_data:
                raise KeyError(f"Missing configuration section: `{section}`.")
            for sub_key in keys:
                if sub_key not in self.config_data[section]:
                    raise KeyError(f"Missing key `{sub_key}` in `{section}` configuration.")

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Retrieves a value from the configuration dictionary using dot-separated keys.

        Args:
            key (str): Dot-separated key to access nested values in the configuration.
            default (Optional[Any]): Default value to return if the key is not found.

        Returns:
            Any: The value from the configuration dictionary if key exists, else the default.
        """
        keys = key.split(".")
        value = self.config_data

        try:
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            if default is not None:
                return default
            raise KeyError(f"Configuration key `{key}` not found.")

    def override_config(self, overrides: Dict[str, Any]) -> None:
        """
        Dynamically updates the configuration values at runtime.

        Args:
            overrides (dict): Dictionary of configuration overrides.
        """
        for key, value in overrides.items():
            keys = key.split(".")
            target = self.config_data
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            target[keys[-1]] = value

    def __repr__(self) -> str:
        """
        String representation of the Config object.

        Returns:
            str: A string summarizing the configuration data.
        """
        return f"Config(file_path={self.file_path})"


# Usage Example for integration:
# config = Config("config.yaml")
# training_config = config.get("training")
# data_resolutions = config.get("data.burgers_equation.resolution", {})
# config.override_config({"training.batch_size": 32})
