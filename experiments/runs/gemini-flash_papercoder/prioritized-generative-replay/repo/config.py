import argparse
import yaml
from ml_collections import config_dict
import torch
import os
from typing import Any, Optional, Dict

class Config:
    """
    Manages loading, parsing, and storing all hyperparameters and experiment settings
    from a YAML file or command-line arguments. It ensures a single source of truth
    for all configurable parameters.
    """

    def __init__(self, config_path: str = "config.yaml", cmd_args: Optional[argparse.Namespace] = None):
        """
        Initializes the Config object by loading from a YAML file and merging command-line arguments.

        Args:
            config_path (str): Path to the YAML configuration file.
            cmd_args (argparse.Namespace, optional): Command-line arguments parsed by argparse.
                                                     Values from cmd_args will override YAML settings.
        """
        self._config = config_dict.ConfigDict()

        # 1. Load configuration from the YAML file
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        with open(config_path, 'r') as f:
            yaml_content: Dict[str, Any] = yaml.safe_load(f)
            # Convert loaded dict to ConfigDict for nested attribute access
            self._config.update(config_dict.ConfigDict(yaml_content))

        # 2. Merge command-line arguments, overriding YAML values
        if cmd_args:
            self._apply_cmd_args(cmd_args)

        # 3. Perform post-processing and type conversions
        self._post_process_config()

    def _apply_cmd_args(self, cmd_args: argparse.Namespace):
        """
        Applies command-line arguments to override specific configuration values.
        This method maps flat argparse arguments to their corresponding nested config paths.
        It should cover the key parameters specified in config.yaml that are likely to be overridden
        from the command line for convenience.
        """
        # Define a mapping from flat command-line argument names to nested config paths
        override_map: Dict[str, str] = {
            'experiment_name': 'experiment.name',
            'seed': 'experiment.seed',
            'device': 'experiment.device',
            'env_name': 'environment.name',
            'env_suite': 'environment.suite',
            'pixel_based': 'environment.pixel_based',
            'rl_algo': 'rl_agent.algorithm',
            'relevance_type': 'relevance_function.type',
            'utd_ratio': 'rl_agent.utd_ratio',
            'batch_size': 'rl_agent.batch_size',
            'synthetic_data_ratio': 'pgr_loop.synthetic_data_ratio',
            'inner_loop_freq': 'pgr_loop.inner_loop_freq_env_steps',
            'total_env_steps': 'environment.total_env_steps',
            'q_hidden_layers': 'rl_agent.q_hidden_layers',
            'q_hidden_units': 'rl_agent.q_hidden_units',
            'policy_hidden_layers': 'rl_agent.policy_hidden_layers',
            'policy_hidden_units': 'rl_agent.policy_hidden_units',
            'guidance_scale': 'generative_model.guidance_scale'
        }

        for arg_name, config_path in override_map.items():
            arg_value: Any = getattr(cmd_args, arg_name, None)
            if arg_value is not None:
                parts = config_path.split('.')
                current_config_dict: config_dict.ConfigDict = self._config
                for i, part in enumerate(parts):
                    if i == len(parts) - 1:  # Last part is the key to set
                        current_config_dict[part] = arg_value
                    else:  # Navigate deeper
                        if part not in current_config_dict or not isinstance(current_config_dict[part], config_dict.ConfigDict):
                            current_config_dict[part] = config_dict.ConfigDict()
                        current_config_dict = current_config_dict[part]

    def _post_process_config(self):
        """
        Performs type conversions and sets dynamic defaults based on other configuration values.
        This ensures that certain values are in the correct format (e.g., torch.device)
        or have sensible defaults derived from other settings.
        """
        # Convert device string to torch.device object
        device_str: str = self.get_hyperparam('experiment.device')
        if device_str == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA not available, switching device to 'cpu'.")
            self._config.experiment.device = "cpu"
        self._config.experiment.device = torch.device(self._config.experiment.device)

        # Ensure pixel_based flag is boolean
        pixel_based_val: Any = self.get_hyperparam('environment.pixel_based')
        if isinstance(pixel_based_val, str):
            self._config.environment.pixel_based = pixel_based_val.lower() == 'true'
        elif not isinstance(pixel_based_val, bool):
            raise TypeError(f"Expected 'environment.pixel_based' to be boolean or string, got {type(pixel_based_val)}")

        # Adjust visual_encoder_output_dim based on pixel_based status
        if not self._config.environment.pixel_based:
            # If not pixel-based, visual encoder output dimension is not relevant.
            # Set to None or ensure it's not accessed by state-based models.
            self._config.environment.visual_encoder_output_dim = None

        # Special handling for 'Finger-Turn-Hard' total_env_steps as per paper
        if self.get_hyperparam('environment.name') == 'Finger-Turn-Hard':
            if self.get_hyperparam('environment.total_env_steps') == 100000: # Only if default is used
                self._config.environment.total_env_steps = 300000 # 300K timesteps for this task

        # Special handling for DMLab environments (10M steps)
        if self.get_hyperparam('environment.suite') == 'DMLab':
            if self.get_hyperparam('environment.total_env_steps') == 100000: # Only if default is used
                self._config.environment.total_env_steps = 10000000 # 10M timesteps for DMLab

        # Handle D_syn_capacity scaling for UTD=40 exp
        if self.get_hyperparam('rl_agent.utd_ratio') == 40:
            if self.get_hyperparam('replay_buffers.d_syn_capacity') == 1000000: # Only if default is used
                self._config.replay_buffers.d_syn_capacity = 2000000

        # Set default for guidance_scale if NOT_SPECIFIED
        try:
            if self.get_hyperparam('generative_model.guidance_scale') == "NOT_SPECIFIED":
                self._config.generative_model.guidance_scale = 2.0 # Reasonable default, can be tuned
        except KeyError: # If the key itself is missing
            self._config.generative_model.guidance_scale = 2.0


    def get_hyperparam(self, key: str) -> Any:
        """
        Retrieves a hyperparameter value using a dot-separated key string.

        Args:
            key (str): A dot-separated string representing the path to the hyperparameter
                       (e.g., "environment.name", "rl_agent.learning_rate.actor").

        Returns:
            Any: The value of the requested hyperparameter.

        Raises:
            KeyError: If the key is not found or if the value is 'NOT_SPECIFIED'.
        """
        parts = key.split('.')
        current_level: Any = self._config
        for part in parts:
            if not isinstance(current_level, (config_dict.ConfigDict, dict)):
                raise KeyError(f"Invalid path in config for '{key}': '{part}' is not a dictionary-like object at path '{'.'.join(parts[:parts.index(part)])}'.")
            if part not in current_level:
                raise KeyError(f"Hyperparameter '{key}' not found in configuration. Missing part: '{part}'.")
            current_level = current_level[part]

        if isinstance(current_level, str) and current_level.strip().upper() == "NOT_SPECIFIED":
            raise KeyError(f"Hyperparameter '{key}' is marked as 'NOT_SPECIFIED' in the configuration. "
                           "Please provide a concrete value in config.yaml or override via command-line arguments.")
        return current_level

    def __str__(self) -> str:
        """Returns a string representation of the configuration."""
        return self._config.pretty_text

    def to_dict(self) -> Dict[str, Any]:
        """Converts the ConfigDict to a standard Python dictionary."""
        return self._config.to_dict()

    def save_config(self, output_path: str):
        """Saves the current configuration to a YAML file."""
        # Ensure that torch.device objects are converted back to string for YAML serialization
        config_to_save = self.to_dict()
        if isinstance(config_to_save['experiment']['device'], torch.device):
            config_to_save['experiment']['device'] = str(config_to_save['experiment']['device'])

        with open(output_path, 'w') as f:
            yaml.dump(config_to_save, f, default_flow_style=False)

