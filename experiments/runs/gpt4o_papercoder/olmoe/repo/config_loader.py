## config_loader.py
import yaml
from typing import Dict, Any


class ConfigLoader:
    """
    Class for loading and validating configurations from a YAML file.
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        """
        Initialize ConfigLoader with the provided YAML file path.
        :param config_path: Path to the configuration file.
        """
        self.config_path = config_path
        self.config_dict = None

    def load_config(self) -> Dict[str, Any]:
        """
        Load and parse configuration settings from a YAML file.
        :return: Parsed configuration as a Python dictionary.
        :raises ValueError: If any errors occur during loading or validation.
        """
        try:
            # Load YAML file
            with open(self.config_path, "r") as file:
                self.config_dict = yaml.safe_load(file)

            # Validate the configuration structure and values
            self._validate_config()

            return self.config_dict

        except FileNotFoundError:
            raise ValueError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")

    def _validate_config(self) -> None:
        """
        Validate the loaded YAML configuration dictionary.
        :raises ValueError: If validation fails.
        """
        # Mandatory top-level sections
        mandatory_sections = ["training", "model", "data", "evaluation"]
        for section in mandatory_sections:
            if section not in self.config_dict:
                raise ValueError(f"Missing mandatory top-level section '{section}' in configuration.")

        # Validate individual sections
        self._validate_training_config(self.config_dict["training"])
        self._validate_model_config(self.config_dict["model"])
        self._validate_data_config(self.config_dict["data"])
        self._validate_evaluation_config(self.config_dict["evaluation"])

    def _validate_training_config(self, training_config: Dict[str, Any]) -> None:
        """
        Validate the training section.
        :param training_config: Dictionary containing training settings.
        :raises ValueError: If validation checks fail.
        """
        # Ensure pretraining and adaptation keys exist
        if "pretraining" not in training_config or "adaptation" not in training_config:
            raise ValueError("Missing 'pretraining' or 'adaptation' sections in 'training' configuration.")

        # Validate pretraining settings
        pretraining_keys = ["learning_rate", "epochs", "batch_size", "sequence_length", "warmup_steps",
                            "peak_lr", "min_lr", "global_max_grad_norm", "optimizer", "optimizer_betas",
                            "weight_decay", "epsilon"]
        self._check_keys(training_config["pretraining"], pretraining_keys, "training:pretraining")

        # Validate adaptation settings
        adaptation_keys = ["sft", "dpo"]
        self._check_keys(training_config, adaptation_keys, "training")
        self._validate_adaptation(training_config["adaptation"])

    def _validate_adaptation(self, adaptation_config: Dict[str, Any]) -> None:
        """
        Validate adaptation settings (SFT and DPO configurations).
        :param adaptation_config: Dictionary containing adaptation settings.
        :raises ValueError: If validation checks fail.
        """
        for adaptation_type in ["sft", "dpo"]:
            if adaptation_type not in adaptation_config:
                raise ValueError(f"Missing '{adaptation_type}' section in 'adaptation' configuration.")

            if adaptation_type == "sft":
                expected_keys = ["learning_rate", "batch_size", "gradient_accumulation_steps", "epochs", "optimizer",
                                 "precision"]
            elif adaptation_type == "dpo":
                expected_keys = ["learning_rate", "batch_size", "epochs", "optimizer", "beta"]

            self._check_keys(adaptation_config[adaptation_type], expected_keys, f"adaptation:{adaptation_type}")

    def _validate_model_config(self, model_config: Dict[str, Any]) -> None:
        """
        Validate the model section of the configuration.
        :param model_config: Dictionary containing the model settings.
        :raises ValueError: If validation checks fail.
        """
        # Required model keys
        model_keys = ["architecture", "hidden_size", "num_layers", "ffn_type", "num_experts", "active_experts",
                      "embedding_size", "ff_dim_per_expert", "vocab_size", "attention_heads",
                      "max_sequence_length", "rotary_positional_embeddings", "norm_type", "router_z_loss_weight",
                      "load_balancing_loss_weight"]

        self._check_keys(model_config, model_keys, "model")

        # Specific validations
        if model_config["architecture"] != "decoder_only":
            raise ValueError("Invalid model architecture. Expected 'decoder_only'.")
        if model_config["ffn_type"] != "mixture_of_experts":
            raise ValueError("Invalid FFN type. Expected 'mixture_of_experts'.")

    def _validate_data_config(self, data_config: Dict[str, Any]) -> None:
        """
        Validate the data section of the configuration.
        :param data_config: Dictionary containing the data settings.
        :raises ValueError: If validation checks fail.
        """
        # Pretraining dataset keys
        if "pretraining_dataset" not in data_config:
            raise ValueError("Missing 'pretraining_dataset' section in 'data' configuration.")
        pretraining_keys = ["sources", "filters"]
        self._check_keys(data_config["pretraining_dataset"], pretraining_keys, "data:pretraining_dataset")

        # Adaptation dataset keys
        if "adaptation_dataset" not in data_config:
            raise ValueError("Missing 'adaptation_dataset' section in 'data' configuration.")
        adaptation_keys = ["sft_sources", "dpo_sources", "max_token_length"]
        self._check_keys(data_config["adaptation_dataset"], adaptation_keys, "data:adaptation_dataset")

    def _validate_evaluation_config(self, evaluation_config: Dict[str, Any]) -> None:
        """
        Validate the evaluation section of the configuration.
        :param evaluation_config: Dictionary containing the evaluation settings.
        :raises ValueError: If validation checks fail.
        """
        required_keys = ["metrics", "benchmarks"]
        self._check_keys(evaluation_config, required_keys, "evaluation")

        if not isinstance(evaluation_config["metrics"], list) or not isinstance(evaluation_config["benchmarks"], list):
            raise ValueError("'metrics' and 'benchmarks' should both be lists in 'evaluation' configuration.")

    def _check_keys(self, config_section: Dict[str, Any], required_keys: list, section_name: str) -> None:
        """
        Validate that required keys are present in the config section.
        :param config_section: Section of the configuration to check.
        :param required_keys: List of required keys.
        :param section_name: Name of the section for error reporting.
        :raises ValueError: If keys are missing.
        """
        for key in required_keys:
            if key not in config_section:
                raise ValueError(f"Missing key '{key}' in '{section_name}' configuration.")
