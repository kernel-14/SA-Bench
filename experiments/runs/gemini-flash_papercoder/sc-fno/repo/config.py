import yaml
from typing import Any, Dict, Optional


class Config:
    """
    Manages experiment configurations loaded from a YAML file.

    This class provides structured access to all hyperparameters, paths, and
    other settings required for an experiment. It supports loading from and
    saving to YAML files, and offers a convenient way to access nested
    configuration values.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes a new Config instance.

        Args:
            params: An optional dictionary containing the configuration parameters.
                    If None, an empty dictionary is used.
        """
        self.params: Dict[str, Any] = params if params is not None else {}

    @staticmethod
    def from_yaml(filepath: str) -> 'Config':
        """
        Loads configuration from a YAML file and creates a Config instance.

        Args:
            filepath: The path to the YAML configuration file.

        Returns:
            A Config instance populated with the parameters from the YAML file.

        Raises:
            FileNotFoundError: If the specified YAML file does not exist.
        """
        try:
            with open(filepath, 'r') as f:
                raw_params = yaml.safe_load(f)
            return Config(raw_params)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at: {filepath}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {filepath}: {e}")

    def to_yaml(self, filepath: str) -> None:
        """
        Saves the current configuration to a YAML file.

        Args:
            filepath: The path where the YAML configuration file should be saved.
        """
        with open(filepath, 'w') as f:
            yaml.safe_dump(self.params, f, sort_keys=False)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value, supporting dot-separated paths for nested access.

        Example: config.get("training.epochs")

        Args:
            key: The dot-separated string path to the configuration value.
            default: The default value to return if the key is not found.

        Returns:
            The configuration value associated with the key, or the default value
            if the key does not exist.
        """
        keys = key.split('.')
        current_level: Any = self.params

        for sub_key in keys:
            if isinstance(current_level, dict) and sub_key in current_level:
                current_level = current_level[sub_key]
            else:
                return default
        return current_level

    def __str__(self) -> str:
        """
        Returns a string representation of the Config object.
        """
        return f"Config(params={self.params})"

    def __repr__(self) -> str:
        """
        Returns a developer-friendly string representation of the Config object.
        """
        return self.__str__()
