import yaml
import os
import logging
from typing import Any, Dict

# Set up logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a console handler if one doesn't exist to prevent duplicate handlers
if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class Config:
    """
    Manages experiment configurations loaded from a YAML file.
    Provides structured access to parameters and supports nested keys.
    """

    def __init__(self, config_path: str):
        """
        Initializes the Config manager by loading the specified YAML file.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config_path: str = config_path
        self._config_data: Dict[str, Any] = {}
        self.load_config()

    def load_config(self) -> None:
        """
        Loads the configuration data from the YAML file specified during initialization.

        Raises:
            FileNotFoundError: If the config file does not exist.
            yaml.YAMLError: If the config file is not a valid YAML format.
            Exception: For other unexpected errors during file loading.
        """
        if not os.path.exists(self.config_path):
            logger.error(f"Configuration file not found at: {self.config_path}")
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f)
            if self._config_data is None:  # Handle empty YAML file gracefully
                self._config_data = {}
            logger.info(f"Configuration loaded successfully from: {self.config_path}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration file {self.config_path}: {e}")
            raise e
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading config file {self.config_path}: {e}")
            raise e

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value, supporting nested keys.

        Example:
            config.get("p2vae_model.base_channels")
            config.get("dataset.root_dir", "./default_data")

        Args:
            key (str): The configuration key, potentially with '.' for nested access.
            default (Any, optional): The default value to return if the key is not found.
                                     Defaults to None.

        Returns:
            Any: The value associated with the key, or the default value if not found.
        """
        parts = key.split('.')
        current_data = self._config_data

        for part in parts:
            if isinstance(current_data, dict) and part in current_data:
                current_data = current_data[part]
            else:
                # Key or part of the path not found, or an intermediate element is not a dictionary
                return default
        return current_data

    def save_config(self, output_path: str) -> None:
        """
        Saves the current configuration to a new YAML file.

        Args:
            output_path (str): The path where the configuration should be saved.
        """
        try:
            # Ensure the directory for the output path exists
            output_dir = os.path.dirname(output_path)
            if output_dir: # Only create if output_path includes a directory
                os.makedirs(output_dir, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(self._config_data, f, indent=2)
            logger.info(f"Configuration saved successfully to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to {output_path}: {e}")
            raise e
