import argparse
import os
import sys
import yaml
from typing import Any, Dict, List, Optional, Union


class Config:
    """
    Manages configuration for the WDNO project, loading from YAML,
    applying defaults, problem-specific overrides, and CLI arguments.
    """

    def __init__(self, config_path: str = 'config.yaml', cli_args: Optional[argparse.Namespace] = None):
        """
        Initializes the Config object.

        Args:
            config_path: Path to the main YAML configuration file.
            cli_args: An argparse.Namespace object containing command-line arguments.
                      If None, arguments are parsed from sys.argv.
        """
        self._raw_config: Dict[str, Any] = self._load_yaml(config_path)
        self._cli_overrides: Dict[str, Any] = self._parse_cli_args(cli_args)
        self._data: Dict[str, Any] = {}  # Stores the final resolved configuration

        self._resolve_final_config()
        self._validate_config()

    def _load_yaml(self, config_path: str) -> Dict[str, Any]:
        """
        Loads the YAML configuration file.

        Args:
            config_path: Path to the YAML file.

        Returns:
            A dictionary containing the YAML content.

        Raises:
            FileNotFoundError: If the config file does not exist.
            yaml.YAMLError: If there is an error parsing the YAML file.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")

    def _parse_cli_args(self, cli_args: Optional[argparse.Namespace]) -> Dict[str, Any]:
        """
        Parses command-line arguments that can override configuration settings.

        Args:
            cli_args: An argparse.Namespace object. If None, sys.argv is parsed.

        Returns:
            A dictionary of parsed command-line arguments, excluding None values.
        """
        parser = argparse.ArgumentParser(add_help=False)  # Add help later if needed
        parser.add_argument('--problem_name', type=str, default=None,
                            help="Name of the PDE problem to run (e.g., '1d_burgers').")
        parser.add_argument('--mode', type=str, default=None,
                            help="Operation mode: 'train' or 'evaluate'.")
        parser.add_argument('--device', type=str, default=None,
                            help="Device to use for computation (e.g., 'cuda', 'cpu').")
        parser.add_argument('--save_path', type=str, default=None,
                            help="Path to save checkpoints and logs.")
        parser.add_argument('--seed', type=int, default=None,
                            help="Random seed for reproducibility.")
        parser.add_argument('--learning_rate', type=float, default=None,
                            help="Learning rate for the optimizer.")
        parser.add_argument('--train_batch_size', type=int, default=None,
                            help="Batch size for training.")
        parser.add_argument('--training_steps', type=int, default=None,
                            help="Total number of training steps.")
        parser.add_argument('--ddim_steps', type=int, default=None,
                            help="Number of DDIM sampling iterations.")
        parser.add_argument('--ddim_eta', type=float, default=None,
                            help="Eta parameter for DDIM sampling.")
        parser.add_argument('--guidance_lambda', type=float, default=None,
                            help="Weight for control guidance (lambda).")
        parser.add_argument('--checkpoint_path', type=str, default=None,
                            help="Path to a checkpoint to load for evaluation or resuming training.")


        if cli_args is None:
            # Parse only known arguments to avoid errors with other CLI tools
            known_args, _ = parser.parse_known_args(sys.argv[1:])
            parsed_args = vars(known_args)
        else:
            parsed_args = vars(cli_args)

        return {k: v for k, v in parsed_args.items() if v is not None}

    @staticmethod
    def _deep_merge(source: Dict[str, Any], destination: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merges two dictionaries. Keys in source overwrite keys in destination.
        If a value is a list, the source list replaces the destination list.

        Args:
            source: The dictionary to merge from.
            destination: The dictionary to merge into.

        Returns:
            The merged dictionary (destination is modified in-place).
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in destination and isinstance(destination[key], dict):
                destination[key] = Config._deep_merge(value, destination[key])
            elif isinstance(value, list) and key in destination and isinstance(destination[key], list):
                # For lists, replace completely rather than extending, as typically done in config overrides
                destination[key] = value
            else:
                destination[key] = value
        return destination

    def _apply_and_flatten_section(self, source_dict: Dict[str, Any],
                                   destination_dict: Dict[str, Any],
                                   key_map: Dict[str, str]):
        """
        Applies key-value pairs from source_dict to destination_dict, renaming keys
        according to key_map. Only applies if the key exists in source_dict.

        Args:
            source_dict: Dictionary to extract values from.
            destination_dict: Dictionary to place values into.
            key_map: A mapping from source_dict keys to destination_dict keys.
        """
        for source_key, dest_key in key_map.items():
            if source_key in source_dict:
                destination_dict[dest_key] = source_dict[source_key]

    def _resolve_final_config(self):
        """
        Resolves the final configuration by applying global defaults, then problem-specific
        overrides, and finally command-line arguments. Flattens relevant sections.
        """
        # Phase 1: Global and Default Settings
        global_config = self._raw_config.get('global', {})
        self._data = self._deep_merge(global_config, self._data)

        # Apply default UNet architecture parameters
        default_unet_arch = self._raw_config.get('default_unet_architecture', {})
        unet_key_map = {
            'initial_dimension': 'unet_initial_dim',
            'num_down_up_layers': 'unet_num_down_up_layers',
            'convolution_kernel_size': 'unet_conv_kernel_size',
            'dimension_multipliers': 'unet_dimension_multipliers',
            'resnet_block_groups': 'unet_resnet_block_groups',
            'attention_hidden_dimension': 'unet_attention_hidden_dimension',
            'attention_heads': 'unet_attention_heads',
            'time_embedding_dimension': 'unet_time_embedding_dimension'
        }
        self._apply_and_flatten_section(default_unet_arch, self._data, unet_key_map)

        # Apply default training parameters
        default_training = self._raw_config.get('default_training', {})
        training_key_map = {
            'batch_size': 'train_batch_size',
            'optimizer': 'optimizer',
            'learning_rate': 'learning_rate',
            'training_steps': 'training_steps',
            'learning_rate_scheduler': 'learning_rate_scheduler'
        }
        self._apply_and_flatten_section(default_training, self._data, training_key_map)

        # Apply default diffusion parameters
        default_diffusion = self._raw_config.get('default_diffusion', {})
        diffusion_key_map = {
            'ddpm_timesteps': 'ddpm_timesteps',
            'ddim_sampling_iterations': 'ddim_steps',
            'ddim_eta': 'ddim_eta',
            'guidance_scheduler': 'guidance_schedule'
        }
        self._apply_and_flatten_section(default_diffusion, self._data, diffusion_key_map)

        # Phase 2: Problem-Specific Settings
        problem_name = self._cli_overrides.get('problem_name') or self._data.get('problem_name')
        if problem_name is None:
            raise ValueError("No PDE problem name specified. Use 'problem_name' in global config or CLI.")
        self._data['problem_name'] = problem_name # Ensure problem_name is in final config

        problem_config = self._raw_config.get('problems', {}).get(problem_name)
        if problem_config is None:
            raise ValueError(f"Configuration for problem '{problem_name}' not found in config.yaml.")

        # Merge top-level problem-specific attributes
        top_level_problem_keys = ['problem_type', 'wavelet_type', 'wavelet_mode', 'wavelet_data_dim']
        for key in top_level_problem_keys:
            if key in problem_config:
                self._data[key] = problem_config[key]
        # Special handling for wavelet_data_dim to become data_dim
        if 'wavelet_data_dim' in self._data:
            self._data['data_dim'] = self._data.pop('wavelet_data_dim')

        # Problem-specific UNet architecture overrides
        problem_unet_arch = problem_config.get('unet_architecture', {})
        self._apply_and_flatten_section(problem_unet_arch, self._data, unet_key_map)
        # Handle specific 2D/3D conv kernels if lists are passed
        if 'convolution_kernel_size' in problem_unet_arch:
            self._data['unet_conv_kernel_size'] = problem_unet_arch['convolution_kernel_size']
        if 'convolution_padding' in problem_unet_arch:
            self._data['unet_conv_padding'] = problem_unet_arch['convolution_padding']
        if 'convolution_stride' in problem_unet_arch:
            self._data['unet_conv_stride'] = problem_unet_arch['convolution_stride']
        else: # Default 1D conv padding/stride if not specified in problem config
            self._data['unet_conv_padding'] = (self._data['unet_conv_kernel_size'] // 2 if isinstance(self._data['unet_conv_kernel_size'], int) else [k // 2 for k in self._data['unet_conv_kernel_size']]) # Assuming symmetric padding
            self._data['unet_conv_stride'] = (1 if isinstance(self._data['unet_conv_kernel_size'], int) else [1 for _ in self._data['unet_conv_kernel_size']])


        # Problem-specific inference overrides
        problem_inference = problem_config.get('inference', {})
        if 'ddim_sampling_iterations' in problem_inference:
            self._data['ddim_steps'] = problem_inference['ddim_sampling_iterations']
        if 'ddim_eta' in problem_inference:
            self._data['ddim_eta'] = problem_inference['ddim_eta']

        # Extract guidance_lambda from control_task if enabled
        control_task_config = problem_config.get('control_task', {})
        if control_task_config.get('enabled', False) and 'guidance_lambda' in control_task_config:
            self._data['guidance_lambda'] = control_task_config['guidance_lambda']
        else:
            self._data['guidance_lambda'] = 0.0 # Default to no guidance if not enabled/specified

        # Calculate multi_res_levels
        super_resolution_task_config = problem_config.get('super_resolution_task', {})
        if super_resolution_task_config.get('enabled', False) and 'sr_target_resolutions' in super_resolution_task_config:
            self._data['multi_res_levels'] = len(super_resolution_task_config['sr_target_resolutions'])
        else:
            self._data['multi_res_levels'] = 0

        # Store complex, nested problem-specific sections directly
        nested_sections = ['data_generation', 'simulation_task', 'control_task', 'super_resolution_task']
        for section_name in nested_sections:
            if section_name in problem_config:
                self._data[section_name] = problem_config[section_name]
            else:
                self._data[section_name] = {} # Ensure section exists even if empty

        # Ensure default data_path is set if not already from global config
        if 'data_path' not in self._data:
            self._data['data_path'] = "./data" # Default data directory

        # Phase 3: CLI Overrides (highest precedence)
        cli_to_data_key_map = {
            'problem_name': 'problem_name',
            'mode': 'mode', # Not explicitly in design data structure, but useful
            'device': 'device',
            'save_path': 'save_path',
            'seed': 'seed',
            'learning_rate': 'learning_rate',
            'train_batch_size': 'train_batch_size',
            'training_steps': 'training_steps',
            'ddim_steps': 'ddim_steps',
            'ddim_eta': 'ddim_eta',
            'guidance_lambda': 'guidance_lambda',
            'checkpoint_path': 'checkpoint_path' # Not explicitly in design data structure, but useful
        }
        self._apply_and_flatten_section(self._cli_overrides, self._data, cli_to_data_key_map)

    def _validate_config(self):
        """
        Validates that all essential configuration parameters are present and have correct types.

        Raises:
            ValueError: If a required parameter is missing or has an incorrect type.
        """
        required_params = [
            ('problem_type', str), ('device', str), ('wavelet_type', str), ('wavelet_mode', str),
            ('data_dim', int), ('unet_initial_dim', int), ('unet_num_down_up_layers', int),
            ('unet_conv_kernel_size', (int, list)), ('unet_dimension_multipliers', list),
            ('unet_resnet_block_groups', int), ('unet_attention_hidden_dimension', int),
            ('unet_attention_heads', int), ('unet_time_embedding_dimension', int),
            ('train_batch_size', int), ('learning_rate', float), ('training_steps', int),
            ('learning_rate_scheduler', str), ('ddpm_timesteps', int), ('ddim_steps', int),
            ('ddim_eta', float), ('guidance_schedule', str), ('save_path', str), ('seed', int),
            ('log_interval_steps', int), ('eval_interval_steps', int)
        ]

        for param, expected_type in required_params:
            if param not in self._data:
                raise ValueError(f"Missing required configuration parameter: '{param}'")
            if not isinstance(self._data[param], expected_type):
                # Special handling for convolution_kernel_size which can be int or list
                if not (isinstance(expected_type, tuple) and any(isinstance(self._data[param], t) for t in expected_type)):
                    raise ValueError(f"Parameter '{param}' has incorrect type. Expected {expected_type}, got {type(self._data[param])}")

        # Ensure that nested sections are dictionaries
        for section in ['data_generation', 'simulation_task', 'control_task', 'super_resolution_task']:
            if section not in self._data or not isinstance(self._data[section], dict):
                self._data[section] = {} # Default to empty dict if not present or wrong type

    def __getattr__(self, name: str) -> Any:
        """Allows attribute-style access to configuration parameters."""
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        """Allows dictionary-style access to configuration parameters."""
        if key in self._data:
            return self._data[key]
        raise KeyError(f"'{type(self).__name__}' object has no key '{key}'")

    def save(self, path: str):
        """
        Saves the resolved configuration to a YAML file.

        Args:
            path: The file path to save the configuration to.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self._data, f, default_flow_style=False)

    def load(self, path: str):
        """
        Loads a configuration from a YAML file, overwriting the current state.
        This assumes the loaded file contains a fully resolved configuration.

        Args:
            path: The file path to load the configuration from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found at: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            self._data = yaml.safe_load(f)


if __name__ == "__main__":
    # Example usage and testing of the Config class
    # You would typically run this from main.py, passing cli_args
    # For standalone test, we simulate cli_args

    # Create a dummy config.yaml if it doesn't exist for testing
    dummy_config_content = """
## config.yaml
global:
  seed: 42
  device: "cuda"
  save_path: "./checkpoints"
  log_interval_steps: 1000
  eval_interval_steps: 10000

default_unet_architecture:
  initial_dimension: 128
  num_down_up_layers: 4
  convolution_kernel_size: 3
  dimension_multipliers: [1, 2, 4, 8]
  resnet_block_groups: 8
  attention_hidden_dimension: 32
  attention_heads: 4
  time_embedding_dimension: 256

default_training:
  batch_size: 16
  optimizer: "Adam"
  learning_rate: 0.0001
  training_steps: 190000
  learning_rate_scheduler: "cosine_annealing"

default_diffusion:
  ddpm_timesteps: 1000
  ddim_sampling_iterations: 50
  ddim_eta: 1.0
  guidance_scheduler: "cosine"

problems:
  1d_burgers:
    problem_type: "1d_burgers"
    wavelet_data_dim: 2
    wavelet_type: "bior2.4"
    wavelet_mode: "periodization"
    data_generation:
      num_train_samples: 40000
    simulation_task:
      enabled: true
    control_task:
      enabled: true
      objective_alpha: 1.0
      guidance_lambda: 120000
    super_resolution_task:
      enabled: true
      train_resolution: [80, 120]
      sr_target_resolutions: [[160, 240], [320, 480], [640, 960]]

  2d_fluid:
    problem_type: "2d_fluid"
    wavelet_data_dim: 3
    wavelet_type: "bior1.3"
    wavelet_mode: "zero"
    unet_architecture:
      convolution_kernel_size: [1, 4, 4]
      convolution_padding: [1, 2, 2]
      convolution_stride: [0, 1, 1]
    control_task:
      enabled: true
      guidance_lambda: 10000
    super_resolution_task:
      enabled: true
      sr_target_resolutions: [[32, 128, 128]]
"""
    if not os.path.exists("config.yaml"):
        with open("config.yaml", "w", encoding='utf-8') as f:
            f.write(dummy_config_content)
        print("Created dummy config.yaml for testing.")

    # Simulate CLI arguments
    class MockCliArgs:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    print("\n--- Test 1: Load 1D Burgers' with default CLI ---")
    mock_cli_args_1 = MockCliArgs(problem_name='1d_burgers', device='cpu', training_steps=50000)
    config1 = Config(cli_args=mock_cli_args_1)
    print(f"Problem Type: {config1.problem_type}")
    print(f"Device: {config1.device}")
    print(f"Learning Rate: {config1.learning_rate}")
    print(f"Training Steps: {config1.training_steps}")
    print(f"UNet Initial Dim: {config1.unet_initial_dim}")
    print(f"Wavelet Type: {config1.wavelet_type}")
    print(f"Data Dim: {config1.data_dim}")
    print(f"Guidance Lambda (Control Task): {config1.guidance_lambda}")
    print(f"Multi-res Levels: {config1.multi_res_levels}")
    print(f"SR Target Resolutions: {config1.super_resolution_task['sr_target_resolutions']}")
    print(f"Control task enabled: {config1.control_task['enabled']}")
    print(f"Sim task enabled: {config1.simulation_task['enabled']}")
    print(f"UNet Conv Kernel Size: {config1.unet_conv_kernel_size}")
    print(f"UNet Conv Padding: {config1.unet_conv_padding}")
    print(f"UNet Conv Stride: {config1.unet_conv_stride}")
    assert config1.problem_type == "1d_burgers"
    assert config1.device == "cpu"
    assert config1.training_steps == 50000
    assert config1.unet_conv_kernel_size == 3 # Default from default_unet_architecture
    assert config1.unet_conv_padding == 1
    assert config1.unet_conv_stride == 1


    print("\n--- Test 2: Load 2D Fluid with CLI override for lambda ---")
    mock_cli_args_2 = MockCliArgs(problem_name='2d_fluid', guidance_lambda=15000.0)
    config2 = Config(cli_args=mock_cli_args_2)
    print(f"Problem Type: {config2.problem_type}")
    print(f"Wavelet Type: {config2.wavelet_type}")
    print(f"Data Dim: {config2.data_dim}")
    print(f"Guidance Lambda (Control Task): {config2.guidance_lambda}") # Should be 15000.0 due to CLI
    print(f"UNet Conv Kernel Size: {config2.unet_conv_kernel_size}") # Should be [1,4,4] from problem config
    print(f"UNet Conv Padding: {config2.unet_conv_padding}")
    print(f"UNet Conv Stride: {config2.unet_conv_stride}")
    assert config2.problem_type == "2d_fluid"
    assert config2.guidance_lambda == 15000.0
    assert config2.unet_conv_kernel_size == [1, 4, 4]
    assert config2.unet_conv_padding == [1, 2, 2]
    assert config2.unet_conv_stride == [0, 1, 1]


    print("\n--- Test 3: Save and Load resolved config ---")
    output_config_path = "./temp_resolved_config.yaml"
    config1.save(output_config_path)
    print(f"Saved resolved config to {output_config_path}")

    loaded_config = Config(config_path=output_config_path) # Load directly from resolved
    print(f"Loaded config problem type: {loaded_config.problem_type}")
    print(f"Loaded config training steps: {loaded_config.training_steps}")
    assert loaded_config.problem_type == config1.problem_type
    assert loaded_config.training_steps == config1.training_steps

    os.remove(output_config_path)
    os.remove("config.yaml") # Clean up dummy config
    print("Cleaned up temporary files.")

    print("\nAll Config tests passed!")
