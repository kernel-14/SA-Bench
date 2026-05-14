"""
config.py: Defines the Config class for loading, validating, and accessing configuration settings from a YAML file.

Dependencies:
- yaml: For parsing YAML configuration files.
- os: For handling file paths.
"""
import yaml
import os

class Config:
    """
    Config class for handling configurations from a YAML file.
    
    Attributes:
        config_path (str): Path to the YAML configuration file.
        config (dict): Dictionary containing parsed configuration data.
    """

    def __init__(self, file_path: str = "config.yaml"):
        """
        Initializes the configuration loader by reading a YAML file.

        Args:
            file_path (str): Path to the YAML configuration file (default: "config.yaml").
        """
        self.config_path = file_path
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """
        Loads and parses the YAML configuration file into a dictionary.

        Returns:
            dict: Dictionary containing the parsed YAML configuration.
        
        Raises:
            FileNotFoundError: If the configuration file is not found.
            yaml.YAMLError: If the YAML file contains invalid syntax.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        
        try:
            with open(self.config_path, "r") as file:
                config_data = yaml.safe_load(file)
                self._validate_config(config_data)
                return config_data
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML syntax in {self.config_path}: {str(e)}")

    def _validate_config(self, config_data: dict):
        """
        Validates the structure and mandatory keys in the YAML configuration.

        Args:
            config_data (dict): Parsed YAML configuration dictionary.

        Raises:
            KeyError: If mandatory configuration sections or keys are missing.
        """
        mandatory_sections = ["training", "model", "data", "logging", "hardware"]
        for section in mandatory_sections:
            if section not in config_data:
                raise KeyError(f"Mandatory section '{section}' missing in configuration file")

    def get_config(self) -> dict:
        """
        Retrieves the entire configuration dictionary.

        Returns:
            dict: Complete configuration settings.
        """
        return self.config

    def get_training_config(self) -> dict:
        """
        Retrieves the training-related configurations.

        Returns:
            dict: Training settings from the configuration.
        """
        return self.config.get("training", {})

    def get_model_config(self) -> dict:
        """
        Retrieves the model-related configurations.

        Returns:
            dict: Model architecture settings from the configuration.
        """
        return self.config.get("model", {})

    def get_data_config(self) -> dict:
        """
        Retrieves the dataset-related configurations.

        Returns:
            dict: Dataset settings from the configuration.
        """
        return self.config.get("data", {})

    def get_logging_config(self) -> dict:
        """
        Retrieves the logging-related configurations.

        Returns:
            dict: Logging settings from the configuration.
        """
        return self.config.get("logging", {})

    def get_hardware_config(self) -> dict:
        """
        Retrieves the hardware-related configurations.

        Returns:
            dict: Hardware settings from the configuration.
        """
        return self.config.get("hardware", {})

if __name__ == "__main__":
    # Example usage
    try:
        config = Config("config.yaml")
        print("Full Config:", config.get_config())
        print("Training Config:", config.get_training_config())
        print("Model Config:", config.get_model_config())
        print("Data Config:", config.get_data_config())
        print("Logging Config:", config.get_logging_config())
        print("Hardware Config:", config.get_hardware_config())
    except Exception as e:
        print(f"Error: {e}")
