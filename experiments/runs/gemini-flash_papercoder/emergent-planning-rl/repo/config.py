import os
import yaml
from typing import Any, Dict

class Config:
    """
    Manages all hyperparameters, file paths, and experimental settings from a YAML file.
    Provides structured access to settings using dot-notation and allows programmatic overrides.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initializes the Config object by loading settings from a YAML file.

        Args:
            config_path (str): The path to the YAML configuration file.

        Raises:
            FileNotFoundError: If the specified config_path does not exist.
            yaml.YAMLError: If the YAML file has an invalid format.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

        with open(config_path, 'r') as f:
            try:
                self._data: Dict[str, Any] = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")

        if self._data is None:
            self._data = {} # Ensure _data is always a dict

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value using dot-notation for nested access.

        Args:
            key (str): The configuration key, e.g., "environment.sokoban.grid_size".
            default (Any, optional): The default value to return if the key is not found.
                                     If None and key is not found, a KeyError is raised.
                                     Defaults to None.

        Returns:
            Any: The configuration value associated with the key, or the default value.

        Raises:
            KeyError: If the key is not found and no default value is provided.
        """
        keys = key.split('.')
        current_data = self._data

        for i, k in enumerate(keys):
            if isinstance(current_data, dict) and k in current_data:
                current_data = current_data[k]
            else:
                if default is not None:
                    return default
                raise KeyError(f"Configuration key '{key}' not found. Missing part: '{k}' (at level {i+1})")
        return current_data

    def set(self, key: str, value: Any) -> None:
        """
        Sets a configuration value. Supports dot-notation for nested keys.
        Intermediate dictionaries are created if they do not exist.

        Args:
            key (str): The configuration key, e.g., "agent.type".
            value (Any): The value to set.
        """
        keys = key.split('.')
        current_data = self._data

        for k in keys[:-1]:
            if not isinstance(current_data, dict):
                # If current_data is not a dict, it cannot be traversed further
                # This handles cases like trying to set "a.b.c" where "a.b" was previously
                # set to a non-dict value like an int. In such cases, we overwrite.
                # To align with Google style, log a warning or raise an error in real usage.
                # For this task, we assume we want to overwrite.
                # For now, we will reset current_data to a new dict at this key.
                # More robust error handling for unexpected overwrites might be needed.
                current_data = {} 

            if k not in current_data or not isinstance(current_data[k], dict):
                current_data[k] = {}
            current_data = current_data[k]

        if not isinstance(current_data, dict):
             # Handle case where the parent of the final key is not a dictionary.
             # This means the path up to keys[:-1] resulted in a non-dict value.
             # Overwrite the parent path with a new dictionary.
            last_key_parent_path = ".".join(keys[:-1])
            self.set(last_key_parent_path, {keys[-1]: value})
            return

        current_data[keys[-1]] = value

    def save(self, output_path: str) -> None:
        """
        Saves the current state of the configuration to a YAML file.

        Args:
            output_path (str): The file path where the configuration should be saved.
        """
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_path, 'w') as f:
            yaml.dump(self._data, f, default_flow_style=False, sort_keys=False)

# Example usage (for testing, will not be part of the final config.py but useful for validation)
if __name__ == '__main__':
    # Create a dummy config.yaml for testing
    dummy_config_content = """
    # Global Configuration
    experiment_name: "test_exp"
    seed: 123

    # Environment Configuration
    environment:
      name: "Sokoban"
      sokoban:
        grid_size: [8, 8]
        reward_structure:
          step_penalty: -0.01

    # Agent Configuration
    agent:
      type: "DRCAgent"
      drc_agent:
        D: 3
    """
    with open("temp_config.yaml", "w") as f:
        f.write(dummy_config_content)

    print("--- Testing Config class ---")

    # Test initialization and get
    try:
        config = Config("temp_config.yaml")
        print(f"Loaded experiment_name: {config.get('experiment_name')}")
        print(f"Loaded seed: {config.get('seed')}")
        print(f"Loaded grid_size: {config.get('environment.sokoban.grid_size')}")
        print(f"Loaded step_penalty: {config.get('environment.sokoban.reward_structure.step_penalty')}")
        print(f"Default value for non-existent key: {config.get('non_existent.key', 'default_val')}")
        
        try:
            config.get("non_existent.key")
        except KeyError as e:
            print(f"Caught expected KeyError: {e}")

    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error during initialization: {e}")

    # Test set and save
    print("\n--- Testing set and save ---")
    config.set("agent.type", "ResNetAgent")
    config.set("new_section.param", 100)
    config.set("environment.mini_pacman.grid_size", [13, 13])
    config.set("environment.sokoban.reward_structure.box_on_target", 1.0) # Overwriting existing
    config.set("agent.drc_agent.N", 5) # Overwriting existing
    
    print(f"Updated agent type: {config.get('agent.type')}")
    print(f"New section param: {config.get('new_section.param')}")
    print(f"Mini PacMan grid size: {config.get('environment.mini_pacman.grid_size')}")
    print(f"Updated box_on_target: {config.get('environment.sokoban.reward_structure.box_on_target')}")
    print(f"Updated DRC N: {config.get('agent.drc_agent.N')}")

    config.save("temp_config_output.yaml")
    print("Configuration saved to temp_config_output.yaml")

    # Verify saved content
    with open("temp_config_output.yaml", "r") as f:
        saved_data = yaml.safe_load(f)
        print("\nContent of saved config:")
        print(yaml.dump(saved_data, default_flow_style=False, sort_keys=False))

    # Clean up dummy files
    os.remove("temp_config.yaml")
    os.remove("temp_config_output.yaml")
    print("\nCleaned up temporary files.")
