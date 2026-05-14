## utils/config_manager.py

import os
import yaml
from typing import Any, Dict


class ConfigManager:
    """
    Manages loading and accessing configuration parameters from a YAML file.
    Supports nested parameter retrieval using dot notation.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initializes the ConfigManager by loading the configuration from the
        specified YAML file.

        Args:
            config_path (str): The file path to the YAML configuration file.

        Raises:
            FileNotFoundError: If the specified config file does not exist.
            yaml.YAMLError: If there is an error parsing the YAML file.
            Exception: For other potential errors during file operations.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

        self._config: Dict[str, Any] = {}
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config is None:
                    # Handle empty YAML file case
                    self._config = {}
                elif not isinstance(loaded_config, dict):
                    # Ensure the root of the YAML is a dictionary
                    raise TypeError(f"Root of YAML file must be a dictionary, got {type(loaded_config).__name__}")
                else:
                    self._config = loaded_config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML configuration file '{config_path}': {e}") from e
        except Exception as e:
            raise Exception(f"An unexpected error occurred while loading '{config_path}': {e}") from e

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the entire loaded configuration dictionary.

        Returns:
            Dict[str, Any]: The complete configuration dictionary.
        """
        return self._config

    def get_param(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a specific configuration parameter, supporting nested keys
        using dot notation (e.g., 'model.backbone.type').

        Args:
            key (str): The dot-separated string representing the path to the
                       desired parameter.
            default (Any, optional): The value to return if the specified key
                                     path does not exist. Defaults to None.

        Returns:
            Any: The value of the requested parameter, or the default value
                 if not found.
        """
        key_parts = key.split('.')
        current_value = self._config

        for sub_key in key_parts:
            if isinstance(current_value, dict) and sub_key in current_value:
                current_value = current_value[sub_key]
            else:
                return default  # Key path not found
        return current_value

