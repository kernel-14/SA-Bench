# config.py
import yaml
from typing import Any, Dict


class Config:
    """Handles loading and validation of a YAML configuration file for system-wide use."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the Config class with the path to the YAML file.

        Args:
            config_path (str): Path to the YAML configuration file. Defaults to 'config.yaml'.
        """
        self.config_path = config_path
        self.config = {}

    def load_config(self) -> Dict[str, Any]:
        """
        Load and validate configuration from the YAML file.

        Returns:
            Dict[str, Any]: A dictionary containing the loaded configuration.

        Raises:
            FileNotFoundError: If the configuration file is not found.
            ValueError: If the YAML file is malformed or validation fails.
        """
        # Load the YAML configuration file
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}") from e
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {self.config_path}") from e

        # Validate and merge defaults
        self._validate_config()
        self._merge_defaults()

        return self.config

    def _validate_config(self):
        """
        Validate the presence of essential configuration sections and structure.

        Raises:
            ValueError: If mandatory fields are missing or invalid.
        """
        # Check for required top-level sections
        required_sections = ["training", "model", "sampling", "dataset", "evaluation", "output"]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section in config: {section}")

        # Validate 'training' fields
        training = self.config["training"]
        if not isinstance(training.get("learning_rate"), (float, int)) or training["learning_rate"] <= 0:
            raise ValueError("Invalid learning_rate in training section")
        if not isinstance(training.get("batch_size"), int) or training["batch_size"] <= 0:
            raise ValueError("Invalid batch_size in training section")
        if not isinstance(training.get("epochs"), int) or training["epochs"] <= 0:
            raise ValueError("Invalid epochs in training section")

        # Validate 'model' fields
        model = self.config["model"]
        if not isinstance(model.get("score_function", {}).get("pretrained"), bool):
            raise ValueError("Invalid 'pretrained' setting in model.score_function")

        # Validate 'sampling' fields
        sampling = self.config["sampling"]
        if not isinstance(sampling.get("T"), int) or sampling["T"] <= 0:
            raise ValueError("Invalid T in sampling section")
        if not isinstance(sampling.get("K"), int) or sampling["K"] <= 0:
            raise ValueError("Invalid K in sampling section")
        if not isinstance(sampling.get("Lipschitz_constant"), (float, int)) or sampling["Lipschitz_constant"] <= 0:
            raise ValueError("Invalid Lipschitz_constant in sampling section")
        step_schedule = sampling.get("step_schedule", {})
        if not isinstance(step_schedule.get("randomized"), bool):
            raise ValueError("Invalid 'randomized' setting in sampling.step_schedule")
        if not isinstance(step_schedule.get("c0"), (float, int)):
            raise ValueError("Invalid 'c0' in sampling.step_schedule")
        if not isinstance(step_schedule.get("c1"), (float, int)):
            raise ValueError("Invalid 'c1' in sampling.step_schedule")

        # Validate 'dataset' fields
        dataset = self.config["dataset"]
        if not isinstance(dataset.get("type"), str):
            raise ValueError("Invalid dataset type in dataset section")
        if not isinstance(dataset.get("dimensions"), int) or dataset["dimensions"] <= 0:
            raise ValueError("Invalid dimensions in dataset section")
        if not isinstance(dataset.get("training_samples"), int) or dataset["training_samples"] <= 0:
            raise ValueError("Invalid training_samples in dataset section")
        if not isinstance(dataset.get("testing_samples"), int) or dataset["testing_samples"] <= 0:
            raise ValueError("Invalid testing_samples in dataset section")

        # Validate 'evaluation' fields
        evaluation = self.config["evaluation"]
        if not isinstance(evaluation.get("metrics"), list) or not evaluation["metrics"]:
            raise ValueError("Invalid or missing metrics in evaluation section")

        # Validate 'output' fields
        output = self.config["output"]
        if not isinstance(output.get("directory"), str):
            raise ValueError("Invalid directory in output section")
        if not isinstance(output.get("save_checkpoints"), bool):
            raise ValueError("Invalid save_checkpoints in output section")
        if not isinstance(output.get("save_plots"), bool):
            raise ValueError("Invalid save_plots in output section")

    def _merge_defaults(self):
        """
        Merge default values for optional fields into the configuration.
        """
        # Defaults for training
        training_defaults = {
            "gradient_clip": 1.0,  # Default gradient clipping value
        }
        self.config["training"].update({k: v for k, v in training_defaults.items() if k not in self.config["training"]})

        # Defaults for model
        model_defaults = {
            "layers": 3,
            "hidden_units": 128,
        }
        self.config["model"].update({k: v for k, v in model_defaults.items() if k not in self.config["model"]})

        # Defaults for sampling
        sampling_defaults = {
            "step_schedule": {
                "randomized": True,
                "c0": 0.001,
                "c1": 0.01,
            }
        }
        for key, default_value in sampling_defaults.items():
            if key not in self.config["sampling"]:
                self.config["sampling"][key] = default_value

        # Defaults for output
        output_defaults = {
            "save_checkpoints": True,
            "save_plots": True,
        }
        self.config["output"].update({k: v for k, v in output_defaults.items() if k not in self.config["output"]})
