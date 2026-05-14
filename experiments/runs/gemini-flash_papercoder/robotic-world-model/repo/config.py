## config.py
import yaml
from typing import Any, Dict

class Config:
    """
    A class to load and manage configuration parameters from a YAML file.
    Provides dot-notation access to parameters and a safe `get` method with
    default value support.
    """

    def __init__(self, config_path: str):
        """
        Initializes the Config object by loading parameters from a YAML file.

        Args:
            config_path: The path to the config.yaml file.
        
        Raises:
            FileNotFoundError: If the specified config_path does not exist.
            yaml.YAMLError: If there is an error parsing the YAML file.
        """
        self._data: Dict[str, Any] = {}
        try:
            with open(config_path, 'r') as f:
                self._data = yaml.safe_load(f)
            if not isinstance(self._data, dict):
                raise yaml.YAMLError("YAML file must contain a dictionary at its root.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML configuration file: {e}")

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """
        Internal class method to create a Config instance directly from a dictionary.
        Used for nested configuration objects.

        Args:
            data: A dictionary containing configuration parameters.

        Returns:
            A new Config instance initialized with the given dictionary.
        """
        instance = cls.__new__(cls)  # Create a new instance without calling __init__
        instance._data = data
        return instance

    def __getattr__(self, name: str) -> Any:
        """
        Enables dot-notation access to configuration parameters (e.g., config.rwm_model.learning_rate).

        Args:
            name: The name of the attribute being accessed.

        Returns:
            The value of the configuration parameter.

        Raises:
            AttributeError: If the configuration parameter is not found.
        """
        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return Config._from_dict(value)
            return value
        raise AttributeError(f"Configuration parameter '{name}' not found.")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Provides a safe way to retrieve configuration values, supporting dot-separated
        nested keys and offering a default value if the key is not found.

        Args:
            key: The dot-separated path to the configuration parameter
                 (e.g., "rwm_model.training.learning_rate").
            default: The value to return if the key is not found. Defaults to None.

        Returns:
            The value corresponding to the key, or the default value if not found.
        """
        parts = key.split('.')
        current = self._data

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def __repr__(self) -> str:
        """
        Returns a string representation of the Config object.
        """
        return f"Config({self._data})"

    def __str__(self) -> str:
        """
        Returns a string representation of the Config object, showing its data.
        """
        return yaml.dump(self._data, indent=2)

