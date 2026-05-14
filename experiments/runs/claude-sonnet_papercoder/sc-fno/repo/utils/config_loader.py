## utils/config_loader.py
"""Configuration loader for SC-FNO experiments.

Provides a simple YAML-based configuration loader with dot-notation access
for nested keys. All hyperparameters, paths, and flags are sourced exclusively
from config files — no hardcoded values anywhere in the codebase.
"""

import os
from typing import Any

import yaml


class ConfigLoader:
    """Loads YAML configuration files and provides dot-notation key access.

    This is the foundational utility that every other module depends on.
    It reads config.yaml (or per-equation YAML files) and exposes their
    contents through a clean, predictable interface.

    Attributes:
        config_path: Path to the YAML file passed at construction.
        cfg: The fully loaded configuration dictionary.

    Example:
        >>> loader = ConfigLoader('config.yaml')
        >>> width = loader.get('model.width')          # returns 20
        >>> batch = loader.get('ode1.training.batch_size')  # returns 16
        >>> lr = loader.get('training.lr', 0.001)      # returns 0.001
        >>> cfg = loader.cfg                            # full dict
    """

    def __init__(self, config_path: str) -> None:
        """Initializes the ConfigLoader by reading the YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Raises:
            FileNotFoundError: If the file does not exist at config_path.
            yaml.YAMLError: If the YAML file is malformed.
        """
        self.config_path: str = config_path
        self.cfg: dict = {}
        self.cfg = self._parse_yaml(config_path)

    def _parse_yaml(self, path: str) -> dict:
        """Reads and parses a YAML file into a dictionary.

        Args:
            path: Path to the YAML file.

        Returns:
            Parsed configuration as a dict. Returns {} for empty files.

        Raises:
            FileNotFoundError: If the file does not exist.
            yaml.YAMLError: If the YAML content is malformed.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Configuration file not found: '{path}'. "
                f"Please ensure the file exists at the specified path."
            )

        with open(path, "r", encoding="utf-8") as f:
            # safe_load prevents execution of arbitrary Python objects
            # embedded in YAML — all config values are scalars, lists, dicts.
            parsed = yaml.safe_load(f)

        # Empty YAML files parse to None; normalize to empty dict.
        if parsed is None:
            return {}

        return parsed

    def load(self) -> dict:
        """Re-reads the YAML file from disk and refreshes self.cfg.

        Useful when the config file has been edited on disk without
        constructing a new ConfigLoader instance (e.g., in tests or
        interactive sessions).

        Returns:
            The freshly loaded configuration dictionary.

        Raises:
            FileNotFoundError: If the file no longer exists.
            yaml.YAMLError: If the YAML content is malformed.
        """
        self.cfg = self._parse_yaml(self.config_path)
        return self.cfg

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a value from cfg using dot-notation for nested keys.

        Traverses the nested configuration dictionary one segment at a time.
        Never raises KeyError — returns default when any segment is missing
        or when an intermediate node is not a dict.

        Args:
            key: Dot-separated key path, e.g. 'model.width' or
                 'ode1.training.batch_size' or 'training.loss_weights.c1'.
            default: Value to return when the key path is not found.
                     Defaults to None.

        Returns:
            The value at the specified key path, or default if not found.

        Examples:
            >>> loader.get('model.width')                    # 20
            >>> loader.get('training.loss_weights.c1')       # 1.0
            >>> loader.get('pde1.discretization.Sx')         # 20
            >>> loader.get('nonexistent.key', 42)            # 42
            >>> loader.get('training.lr')                    # 0.001
        """
        segments = key.split(".")
        current: Any = self.cfg

        for segment in segments:
            # If the current node is not a dict, we cannot descend further.
            if not isinstance(current, dict):
                return default
            # If the segment is missing from the current dict, return default.
            if segment not in current:
                return default
            current = current[segment]

        return current
