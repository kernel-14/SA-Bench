import yaml
from pathlib import Path
from typing import Any, Dict, List, Union

class Config:
    """
    A centralized configuration manager that loads, accesses, modifies, and saves
    configuration parameters from a YAML file. It supports nested keys using dot notation.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initializes the Config manager by loading parameters from a YAML file.

        Args:
            config_path (str): The path to the YAML configuration file.

        Raises:
            FileNotFoundError: If the specified config_path does not exist.
            yaml.YAMLError: If there is an error parsing the YAML file.
        """
        self.config_path: str = config_path
        self.config_dict: Dict[str, Any] = self._load_config(config_path)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Loads the YAML configuration file.

        Args:
            config_path (str): The path to the YAML configuration file.

        Returns:
            Dict[str, Any]: The loaded configuration as a dictionary.

        Raises:
            FileNotFoundError: If the specified config_path does not exist.
            yaml.YAMLError: If there is an error parsing the YAML file.
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred while loading config {config_path}: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value, supporting dot notation for nested keys.

        Args:
            key (str): The configuration key (e.g., "data.dataset_type", "model.num_layers").
            default (Any, optional): The default value to return if the key is not found.
                                     Defaults to None.

        Returns:
            Any: The value associated with the key, or the default value if the key is not found.

        Raises:
            KeyError: If the key is not found and no default value is provided.
            TypeError: If an intermediate part of the key path is not a dictionary.
        """
        keys: List[str] = key.split('.')
        current_dict: Dict[str, Any] = self.config_dict

        try:
            for i, k in enumerate(keys):
                if not isinstance(current_dict, dict):
                    raise TypeError(f"Attempted to access key '{k}' on a non-dictionary "
                                    f"value at path '{'.'.join(keys[:i])}'")
                if k not in current_dict:
                    if default is not None:
                        return default
                    raise KeyError(f"Configuration key '{key}' not found. "
                                   f"Missing part: '{k}' at path '{'.'.join(keys[:i])}'")
                current_dict = current_dict[k]
            return current_dict
        except (KeyError, TypeError) as e:
            if default is not None:
                return default
            raise e

    def set(self, key: str, value: Any) -> None:
        """
        Sets a configuration value, supporting dot notation for nested keys.
        Creates intermediate dictionaries if they do not exist.

        Args:
            key (str): The configuration key (e.g., "data.batch_size", "model.hidden_dim").
            value (Any): The value to set for the given key.

        Raises:
            TypeError: If an intermediate part of the key path is not a dictionary and cannot be
                       overwritten to become one.
        """
        keys: List[str] = key.split('.')
        current_dict: Dict[str, Any] = self.config_dict

        for i, k in enumerate(keys):
            if i == len(keys) - 1:
                # Last key, set the value
                current_dict[k] = value
            else:
                # Intermediate key, ensure it's a dictionary
                if not isinstance(current_dict, dict):
                    raise TypeError(f"Cannot set key '{key}'. Intermediate path "
                                    f"'{'.'.join(keys[:i])}' is not a dictionary.")
                if k not in current_dict or not isinstance(current_dict[k], dict):
                    current_dict[k] = {}
                current_dict = current_dict[k]

    def save(self, output_path: str) -> None:
        """
        Saves the current configuration dictionary to a YAML file.

        Args:
            output_path (str): The file path where the configuration should be saved.

        Raises:
            IOError: If there's an issue writing to the file.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.safe_dump(self.config_dict, f, sort_keys=False)
        except IOError as e:
            raise IOError(f"Error saving configuration to {output_path}: {e}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred while saving config to {output_path}: {e}")

    def __repr__(self) -> str:
        """
        Returns a string representation of the Config object.
        """
        return f"Config(path='{self.config_path}', data=\n{yaml.dump(self.config_dict, sort_keys=False)})"

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the Config object.
        """
        return self.__repr__()

# Example Usage (for testing purposes, remove in final integration if not needed)
if __name__ == '__main__':
    # Create a dummy config.yaml for testing
    dummy_config_content = """
general:
  experiment_name: "test_exp"
  seed: 123
  device: "cpu"
data:
  dataset_type: "test_data"
  max_sequence_length: 128
model:
  num_layers: 6
  common:
    hidden_dim: 256
training:
  optimizer:
    type: "Adam"
    lr: 0.001
"""
    dummy_config_path = "temp_config.yaml"
    with open(dummy_config_path, "w", encoding='utf-8') as f:
        f.write(dummy_config_content)

    print("--- Testing Config Load and Get ---")
    try:
        config = Config(dummy_config_path)
        print(config)

        print(f"\nGeneral experiment name: {config.get('general.experiment_name')}")
        print(f"Data max sequence length: {config.get('data.max_sequence_length')}")
        print(f"Model num layers: {config.get('model.num_layers')}")
        print(f"Model common hidden dim: {config.get('model.common.hidden_dim')}")
        print(f"Training optimizer type: {config.get('training.optimizer.type')}")

        print(f"Non-existent key with default: {config.get('non_existent.key', 'DEFAULT_VALUE')}")
        
        try:
            config.get('non_existent.key')
        except KeyError as e:
            print(f"Caught expected error for non-existent key without default: {e}")

        # Test invalid path access
        try:
            config.get('model.num_layers.sub_key')
        except TypeError as e:
            print(f"Caught expected error for invalid path access: {e}")

    except Exception as e:
        print(f"Error during config loading/getting: {e}")

    print("\n--- Testing Config Set ---")
    try:
        config.set('general.device', 'cuda')
        print(f"Set general.device to: {config.get('general.device')}")

        config.set('new_section.param1', True)
        print(f"Set new_section.param1 to: {config.get('new_section.param1')}")
        
        config.set('model.common.attention_heads', 8)
        print(f"Set model.common.attention_heads to: {config.get('model.common.attention_heads')}")

        print("\nConfig after setting new values:")
        print(config)

    except Exception as e:
        print(f"Error during config setting: {e}")

    print("\n--- Testing Config Save ---")
    try:
        output_dir_path = Path("temp_output")
        output_dir_path.mkdir(exist_ok=True)
        output_path = output_dir_path / "saved_config.yaml"
        config.save(str(output_path))
        print(f"Config saved to {output_path}")

        # Verify saved config
        loaded_saved_config = Config(str(output_path))
        print("\nLoaded saved config for verification:")
        print(loaded_saved_config)
        assert loaded_saved_config.get('general.device') == 'cuda'
        assert loaded_saved_config.get('new_section.param1') is True
        print("Verification successful!")

    except Exception as e:
        print(f"Error during config saving: {e}")
    finally:
        # Clean up dummy files
        Path(dummy_config_path).unlink(missing_ok=True)
        if 'output_dir_path' in locals() and output_dir_path.exists():
            for f in output_dir_path.iterdir():
                f.unlink(missing_ok=True)
            output_dir_path.rmdir()
        print("\n--- Cleaned up dummy files ---")
