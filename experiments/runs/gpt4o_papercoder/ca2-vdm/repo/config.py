"""config.py: Responsible for managing and parsing configuration settings from the config.yaml file."""

import yaml
from typing import Any, Dict

class Config:
    """Centralized configuration manager for the system."""
    
    _instance = None  # Singleton instance.

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, file_path: str = "config.yaml") -> None:
        """
        Initialize the Config class by loading and validating the configuration file.

        Args:
            file_path (str): Path to the config.yaml file. Defaults to 'config.yaml'.
        """
        self.file_path = file_path
        self.config_dict: Dict[str, Any] = {}
        self.load_config()
        self.validate()

    def load_config(self) -> None:
        """
        Load configuration from the specified YAML file.

        Raises:
            FileNotFoundError: If the YAML file is not found at the specified path.
            ValueError: If the YAML file cannot be parsed.
        """
        try:
            with open(self.file_path, "r") as file:
                self.config_dict = yaml.safe_load(file)
                if not isinstance(self.config_dict, dict):
                    raise ValueError("Invalid configuration format: Expected a dictionary at the top level.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.file_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file {self.file_path}: {e}")

    def get(self, section: str, default: Any = None) -> Any:
        """
        Retrieve a specific configuration value using dot-separated keys.

        Args:
            section (str): Dot-separated string representing the configuration key path.
                           For example, "training.learning_rate".
            default (Any): Default value to return if the key is missing.

        Returns:
            Any: The configuration value or the default if the key is not found.
        """
        keys = section.split(".")
        value = self.config_dict
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def validate(self) -> None:
        """
        Validate the structure and required keys of the configuration file.

        Raises:
            ValueError: If critical keys or sections are missing or malformed.
        """
        required_sections = [
            "training", "vae", "datasets", "model", "evaluation",
        ]

        # Check for required sections.
        for section in required_sections:
            if section not in self.config_dict:
                raise ValueError(f"Missing required section: '{section}' in the configuration file.")

        # Validate `training` section.
        training_keys = [
            "learning_rate", "optimizer", "batch_size_t2v_stage1", "batch_size_t2v_stage2",
            "batch_size_video_pred", "epochs_t2v_stage1", "epochs_t2v_stage2",
            "epochs_video_pred", "scheduler", "noise_timesteps"
        ]
        self._validate_section("training", training_keys)

        # Validate `vae` section.
        vae_keys = ["pretrained", "resolution", "downsample_factor"]
        self._validate_section("vae", vae_keys)

        # Validate `datasets`.
        dataset_keys = ["intern_vid", "sky_timelapse", "ucf101"]
        self._validate_section("datasets", dataset_keys)
        self._validate_subkeys("datasets.intern_vid", ["resolution", "splits"])
        self._validate_subkeys("datasets.sky_timelapse", ["resolution", "splits"])
        self._validate_subkeys("datasets.ucf101", ["resolution", "split"])

        # Validate `model` section.
        model_keys = [
            "base", "prefix_length_max_t2v", "chunk_length_t2v",
            "prefix_length_max_video_pred", "chunk_length_video_pred",
            "temporal_attention", "spatial_attention", "positional_embeddings", "kv_cache"
        ]
        self._validate_section("model", model_keys)
        self._validate_subkeys("model.positional_embeddings", ["sinusoidal", "cyclic_tpe"])
        self._validate_subkeys("model.kv_cache", ["enabled", "max_length", "reset_strategy"])

        # Validate `evaluation` section.
        evaluation_keys = ["metrics"]
        self._validate_section("evaluation", evaluation_keys)
        self._validate_subkeys("evaluation.fvd", ["pretrained_i3d", "chunk_size"])

    def _validate_section(self, section: str, keys: list) -> None:
        """
        Validate that a section contains the required keys.

        Args:
            section (str): Section name in the configuration (e.g., "training").
            keys (list): List of required keys for the section.

        Raises:
            ValueError: If any of the required keys are missing.
        """
        if section not in self.config_dict:
            raise ValueError(f"Missing section: '{section}' in the configuration file.")
        
        for key in keys:
            if key not in self.config_dict[section]:
                raise ValueError(f"Missing key '{key}' under section '{section}' in the configuration file.")

    def _validate_subkeys(self, section: str, subkeys: list) -> None:
        """
        Validate that a specific section contains the required subkeys.

        Args:
            section (str): Dot-separated path to the section (e.g., "datasets.intern_vid").
            subkeys (list): List of required subkeys for the section.

        Raises:
            ValueError: If any of the required subkeys are missing.
        """
        section_value = self.get(section, default=None)
        if not section_value:
            raise ValueError(f"Missing or malformed section '{section}' in the configuration file.")

        for subkey in subkeys:
            if subkey not in section_value:
                raise ValueError(f"Missing subkey '{subkey}' under section '{section}'.")

# Instantiate the singleton instance for global access.
config_instance = Config()

# Function to get a singleton instance of Config.
def get_config() -> Config:
    """
    Retrieve the singleton instance of the Config class.

    Returns:
        Config: The configuration instance.
    """
    return config_instance
