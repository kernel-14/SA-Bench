import argparse
import os
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import yaml


class _AttributeDict(dict):
    """
    A dictionary that allows attribute-style access and recursive conversion
    of nested dictionaries into _AttributeDicts. This base class is used
    to provide dot-notation access to configuration parameters.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = _AttributeDict(v)
            elif isinstance(v, list):
                # Recursively convert dicts within lists
                self[k] = [(_AttributeDict(x) if isinstance(x, dict) else x) for x in v]

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            # Raise AttributeError for non-existent keys to match standard object behavior
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        # Allow setting new attributes or updating existing ones via dot notation
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def to_dict(self) -> Dict[str, Any]:
        """Converts the _AttributeDict object back into a standard dictionary,
        recursively converting nested _AttributeDicts."""
        result = {}
        for k, v in self.items():
            if isinstance(v, _AttributeDict):
                result[k] = v.to_dict()
            elif isinstance(v, list):
                result[k] = [(x.to_dict() if isinstance(x, _AttributeDict) else x) for x in v]
            else:
                result[k] = v
        return result


class Config(_AttributeDict):
    """
    Configuration class for MoE-POT experiments.
    Loads settings from a YAML file (config.yaml) and allows overrides from
    command-line arguments. Provides attribute-style access to nested configurations.
    """

    _MODEL_SIZE_CONFIGS: Dict[str, Dict[str, int]] = {
        "Tiny": {"attention_dim": 512, "mlp_dim": 512, "num_layers": 4, "num_heads": 4},
        "Small": {"attention_dim": 1024, "mlp_dim": 1024, "num_layers": 6, "num_heads": 8},
        "Medium": {"attention_dim": 1024, "mlp_dim": 2048, "num_layers": 8, "num_heads": 8},
    }

    def __init__(self, config_path: str = "config.yaml", cmd_args: Optional[argparse.Namespace] = None):
        """
        Initializes the Config object by loading from a YAML file and applying
        command-line overrides.

        Args:
            config_path: Path to the YAML configuration file. Defaults to "config.yaml".
            cmd_args: An argparse.Namespace object containing command-line arguments.
                      Arguments should be handled such that 'model.attention_dim'
                      in the Namespace maps to config.model.attention_dim.
        
        Raises:
            FileNotFoundError: If the specified config_path does not exist.
            ValueError: If an unknown model size is specified or if spatial resolution
                        is not divisible by patch size.
        """
        self.config_path = config_path

        # 1. Load from YAML
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg_dict = yaml.safe_load(f)

        # Initialize the base _AttributeDict with loaded configuration,
        # which handles recursive conversion of nested dicts.
        super().__init__(cfg_dict)

        # 2. Apply command-line overrides
        if cmd_args:
            self._apply_cmd_args(cmd_args)

        # 3. Update model specific parameters based on 'model.size'
        # This must happen AFTER CLI args are applied, in case model.size is overridden.
        self.update_for_model_size()

        # 4. Initialize dynamic channel placeholders
        # These will be set by set_dynamic_channels later by the data module.
        # Default to 0, which implies "not yet determined".
        if not hasattr(self.model, 'input_channels'):
            self.model.input_channels = 0
        if not hasattr(self.model, 'output_channels'):
            self.model.output_channels = 0

        # 5. Initialize current experiment specific data/training info placeholders
        # These will be populated by update_for_experiment_type.
        if not hasattr(self.data, 'current_data_info'):
            self.data.current_data_info = _AttributeDict()
        if not hasattr(self.training, 'current_epochs'):
            self.training.current_epochs = 0
        if not hasattr(self.training, 'current_warmup_epochs'):
            self.training.current_warmup_epochs = 0

        # 6. Post-initialization validation
        self._post_init_validation()

    def _apply_cmd_args(self, cmd_args: argparse.Namespace):
        """
        Applies command-line arguments to override configuration values.
        Handles flattened arguments (e.g., 'model.attention_dim') by splitting on '.'
        and traversing the nested _AttributeDict structure.

        Args:
            cmd_args: An argparse.Namespace object containing command-line arguments.
        """
        for arg_key, arg_value in vars(cmd_args).items():
            if arg_value is not None:  # Only override if the argument was explicitly set
                keys = arg_key.split('.')
                current_level = self
                for i, key in enumerate(keys):
                    if i == len(keys) - 1:
                        # Reached the final key, set the value.
                        # For global arguments like 'experiment_name', `keys` will have one element.
                        if len(keys) == 1 and not isinstance(current_level, _AttributeDict):
                            # This handles top-level attributes like `experiment_name` directly on the Config object
                            setattr(self, key, arg_value)
                        else:
                            # This handles nested attributes
                            if not isinstance(current_level, _AttributeDict):
                                raise TypeError(
                                    f"Attempted to set attribute '{key}' on a non-dictionary like object. "
                                    f"Path: {'.'.join(keys[:i])}, Type: {type(current_level)}"
                                )
                            current_level[key] = arg_value
                    else:
                        # Navigate deeper, creating _AttributeDict if necessary.
                        # This handles cases where a sub-dictionary might not exist in the YAML but is
                        # specified in the command line (e.g., --new_section.param value).
                        if key not in current_level or not isinstance(current_level[key], _AttributeDict):
                            current_level[key] = _AttributeDict()
                        current_level = current_level[key]

    def update_for_model_size(self):
        """
        Updates model-specific parameters (attention_dim, mlp_dim, num_layers, num_heads)
        based on the 'model.size' attribute from predefined configurations.
        """
        model_size_key = self.model.size
        if model_size_key not in self._MODEL_SIZE_CONFIGS:
            raise ValueError(f"Unknown model size: '{model_size_key}'. "
                             f"Available sizes are: {list(self._MODEL_SIZE_CONFIGS.keys())}")

        model_params = self._MODEL_SIZE_CONFIGS[model_size_key]
        self.model.attention_dim = model_params["attention_dim"]
        self.model.mlp_dim = model_params["mlp_dim"]
        self.model.num_layers = model_params["num_layers"]
        self.model.num_heads = model_params["num_heads"]

    def set_dynamic_channels(self, input_channels: int, output_channels: int):
        """
        Sets the dynamically determined input and output channel counts.
        This is typically called after data preprocessing determines the maximum channels.

        Args:
            input_channels: The number of input channels for the model.
            output_channels: The number of output channels for the model.
        """
        self.model.input_channels = input_channels
        self.model.output_channels = output_channels

    def update_for_experiment_type(self, experiment_type: str, dataset_name: Optional[str] = None):
        """
        Adjusts training epochs, warmup epochs, and data information based on the
        specified experiment type (pretrain, finetune, downstream).

        Args:
            experiment_type: Type of experiment ('pretrain', 'finetune', 'downstream').
            dataset_name: Optional; for fine-tuning or downstream, specifies a single dataset.
                          If None, all datasets for that type are considered (e.g., pretrain).
        
        Raises:
            ValueError: If an unknown experiment type is provided, or if `dataset_name` is
                        missing for 'finetune'/'downstream' types, or if the specified
                        dataset is not found in the respective data info.
        """
        if experiment_type == 'pretrain':
            self.training.current_epochs = self.training.pretrain_epochs
            self.training.current_warmup_epochs = self.training.pretrain_warmup_epochs
            # Make a deep copy to avoid modifying the original data structure if current_data_info is modified
            self.data.current_data_info = _AttributeDict(self.data.pretrain_data_info.to_dict())
        elif experiment_type == 'finetune':
            self.training.current_epochs = self.training.finetune_epochs
            self.training.current_warmup_epochs = self.training.finetune_warmup_epochs
            if dataset_name is None:
                raise ValueError("dataset_name must be provided for 'finetune' experiment type.")
            if dataset_name not in self.data.finetune_data_info:
                 raise ValueError(f"Dataset '{dataset_name}' not found in finetune_data_info.")
            # current_data_info for finetune is a dict with only the specified dataset
            self.data.current_data_info = _AttributeDict({dataset_name: self.data.finetune_data_info[dataset_name].to_dict()})
        elif experiment_type == 'downstream':
            self.training.current_epochs = self.training.downstream_epochs
            self.training.current_warmup_epochs = self.training.downstream_warmup_epochs
            if dataset_name is None:
                raise ValueError("dataset_name must be provided for 'downstream' experiment type.")
            if dataset_name not in self.data.downstream_data_info:
                 raise ValueError(f"Dataset '{dataset_name}' not found in downstream_data_info.")
            # current_data_info for downstream is a dict with only the specified dataset
            self.data.current_data_info = _AttributeDict({dataset_name: self.data.downstream_data_info[dataset_name].to_dict()})
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")

    def _post_init_validation(self):
        """
        Performs basic validation checks after initialization.
        """
        if self.model.input_spatial_resolution % self.model.patch_size != 0:
            raise ValueError(
                f"Input spatial resolution ({self.model.input_spatial_resolution}) "
                f"must be divisible by patch size ({self.model.patch_size})."
            )

    def __repr__(self):
        """String representation of the Config object."""
        return (f"Config(path='{self.config_path}', "
                f"experiment_name='{self.experiment_name}', "
                f"model_size='{self.model.size}', "
                f"attention_dim={self.model.attention_dim}, "
                f"num_layers={self.model.num_layers})")


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments, allowing overrides of config.yaml values.
    Uses a simple flat structure for args, e.g., --model.attention_dim VALUE,
    which will map to `model.attention_dim` in the parsed Namespace.

    Returns:
        An argparse.Namespace object containing parsed arguments.
    """
    parser = argparse.ArgumentParser(description="MoE-POT Configuration Parser")

    # Arguments for global settings (top-level in config.yaml)
    parser.add_argument("--config_path", type=str, default="config.yaml",
                        help="Path to the main YAML configuration file.")
    parser.add_argument("--experiment_name", type=str, dest="experiment_name",
                        help="Name of the current experiment.")
    parser.add_argument("--seed", type=int, dest="seed",
                        help="Random seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, dest="output_dir",
                        help="Base directory for experiment outputs.")
    parser.add_argument("--checkpoint_dir", type=str, dest="checkpoint_dir",
                        help="Subdirectory for checkpoints relative to output_dir.")
    parser.add_argument("--log_dir", type=str, dest="log_dir",
                        help="Subdirectory for logs relative to output_dir.")

    # Model Configuration overrides (nested under 'model')
    parser.add_argument("--model.size", type=str, choices=list(Config._MODEL_SIZE_CONFIGS.keys()), dest="model.size",
                        help="Model size (Tiny, Small, Medium).")
    parser.add_argument("--model.attention_dim", type=int, dest="model.attention_dim",
                        help="Attention dimension (d_z).")
    parser.add_argument("--model.mlp_dim", type=int, dest="model.mlp_dim",
                        help="MLP dimension for expert networks.")
    parser.add_argument("--model.num_layers", type=int, dest="model.num_layers",
                        help="Number of transformer blocks (N).")
    parser.add_argument("--model.num_heads", type=int, dest="model.num_heads",
                        help="Number of heads in Fourier layer (h).")
    parser.add_argument("--model.num_routed_experts", type=int, dest="model.num_routed_experts",
                        help="Number of routed experts (N_r).")
    parser.add_argument("--model.num_shared_experts", type=int, dest="model.num_shared_experts",
                        help="Number of shared experts (N_s).")
    parser.add_argument("--model.top_k", type=int, dest="model.top_k",
                        help="Number of top experts to select (K).")
    parser.add_argument("--model.patch_size", type=int, dest="model.patch_size",
                        help="Patch size (P).")
    parser.add_argument("--model.input_spatial_resolution", type=int, dest="model.input_spatial_resolution",
                        help="Standardized spatial resolution (H).")
    parser.add_argument("--model.T_in", type=int, dest="model.T_in",
                        help="Number of previous frames to input for prediction.")
    parser.add_argument("--model.activation", type=str, dest="model.activation",
                        help="Activation function to use (e.g., GELU, ReLU).")
    parser.add_argument("--model.router_cnn_layers", type=int, dest="model.router_cnn_layers",
                        help="Number of CNN layers in router gating network.")
    parser.add_argument("--model.router_cnn_kernel_size", type=int, dest="model.router_cnn_kernel_size",
                        help="Kernel size for CNN layers in router gating network.")
    parser.add_argument("--model.expert_cnn_layers", type=int, dest="model.expert_cnn_layers",
                        help="Number of CNN layers in expert networks.")
    parser.add_argument("--model.expert_cnn_kernel_size", type=int, dest="model.expert_cnn_kernel_size",
                        help="Kernel size for CNN layers in expert networks.")

    # Data Configuration overrides (nested under 'data')
    parser.add_argument("--data.data_root", type=str, dest="data.data_root",
                        help="Root directory for raw PDE datasets.")
    parser.add_argument("--data.interpolation_method", type=str, dest="data.interpolation_method",
                        help="Interpolation method for spatial resizing.")
    parser.add_argument("--data.padding_value", type=float, dest="data.padding_value",
                        help="Constant value for channel padding.")
    # Overriding specific dataset properties (e.g., train_samples, weight) is complex
    # via simple CLI args for all datasets and is better managed via config.yaml.

    # Training Configuration overrides (nested under 'training')
    parser.add_argument("--training.optimizer", type=str, dest="training.optimizer",
                        help="Optimizer name (e.g., Adam).")
    parser.add_argument("--training.learning_rate", type=float, dest="training.learning_rate",
                        help="Initial learning rate.")
    parser.add_argument("--training.weight_decay", type=float, dest="training.weight_decay",
                        help="Weight decay for optimizer.")
    parser.add_argument("--training.beta1", type=float, dest="training.beta1",
                        help="Beta1 for Adam optimizer.")
    parser.add_argument("--training.beta2", type=float, dest="training.beta2",
                        help="Beta2 for Adam optimizer.")
    parser.add_argument("--training.batch_size", type=int, dest="training.batch_size",
                        help="Total batch size.")
    parser.add_argument("--training.noise_epsilon", type=float, dest="training.noise_epsilon",
                        help="Epsilon for noise injection during pre-training.")
    parser.add_argument("--training.load_balance_weight", type=float, dest="training.load_balance_weight",
                        help="Weight for the load balancing loss.")
    parser.add_argument("--training.pretrain_epochs", type=int, dest="training.pretrain_epochs",
                        help="Number of epochs for pre-training.")
    parser.add_argument("--training.pretrain_warmup_epochs", type=int, dest="training.pretrain_warmup_epochs",
                        help="Number of warmup epochs for pre-training.")
    parser.add_argument("--training.finetune_epochs", type=int, dest="training.finetune_epochs",
                        help="Number of epochs for fine-tuning.")
    parser.add_argument("--training.finetune_warmup_epochs", type=int, dest="training.finetune_warmup_epochs",
                        help="Number of warmup epochs for fine-tuning.")
    parser.add_argument("--training.downstream_epochs", type=int, dest="training.downstream_epochs",
                        help="Number of epochs for downstream tasks.")
    parser.add_argument("--training.downstream_warmup_epochs", type=int, dest="training.downstream_warmup_epochs",
                        help="Number of warmup epochs for downstream tasks.")
    parser.add_argument("--training.log_interval", type=int, dest="training.log_interval",
                        help="Log every N batches.")
    parser.add_argument("--training.eval_interval", type=int, dest="training.eval_interval",
                        help="Evaluate every N epochs.")
    parser.add_argument("--training.save_interval", type=int, dest="training.save_interval",
                        help="Save checkpoint every N epochs.")

    # Distributed Training Configuration overrides (nested under 'distributed')
    parser.add_argument("--distributed.world_size", type=int, dest="distributed.world_size",
                        help="Number of GPUs/processes for distributed training.")
    parser.add_argument("--distributed.backend", type=str, dest="distributed.backend",
                        help="Distributed training backend (e.g., nccl, gloo).")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Example Usage (for testing during development)
    # Create a dummy config.yaml for testing
    dummy_config_content = """
# Global Configuration
experiment_name: moe-pot-reproduction
seed: 42

# Model Configuration
model:
  size: Small
  attention_dim: 1024
  mlp_dim: 1024
  num_layers: 6
  num_heads: 8
  num_routed_experts: 16
  num_shared_experts: 2
  top_k: 4
  patch_size: 8
  input_spatial_resolution: 128
  input_channels: 0
  output_channels: 0
  T_in: 10
  activation: 'GELU'
  router_cnn_layers: 2
  router_cnn_kernel_size: 3
  expert_cnn_layers: 2
  expert_cnn_kernel_size: 1

# Data Configuration
data:
  data_root: './data/raw'
  interpolation_method: 'bicubic'
  padding_value: 1.0
  pretrain_data_info:
    FNO_1e-5: {train_samples: 1000, test_samples: 200, weight: 1.0}
    CNS_0.1_0.01: {train_samples: 9000, test_samples: 200, weight: 1.0}
    SWE: {train_samples: 900, test_samples: 60, weight: 1.0}
  finetune_data_info:
    FNO_1e-5: {train_samples: 1000, test_samples: 200}
  downstream_data_info:
    NS_1e-4: {train_samples: 2000, test_samples: 200}

# Training Configuration
training:
  optimizer: 'Adam'
  learning_rate: 0.001
  weight_decay: 0.000001
  beta1: 0.9
  beta2: 0.9
  batch_size: 20
  noise_epsilon: 0.01
  load_balance_weight: 0.1
  pretrain_epochs: 1000
  pretrain_warmup_epochs: 200
  finetune_epochs: 200
  finetune_warmup_epochs: 40
  downstream_epochs: 500
  downstream_warmup_epochs: 100
  log_interval: 10
  eval_interval: 50
  save_interval: 50

# Distributed Training Configuration
distributed:
  world_size: 8
  backend: 'nccl'

# Output and Logging
output_dir: './experiments'
checkpoint_dir: 'checkpoints'
log_dir: 'logs'
"""
    temp_config_path = "temp_config.yaml"
    with open(temp_config_path, "w", encoding='utf-8') as f:
        f.write(dummy_config_content)

    print("--- Test 1: Basic Load ---")
    cfg = Config(config_path=temp_config_path)
    print(f"Config path: {cfg.config_path}")
    print(f"Experiment Name: {cfg.experiment_name}")
    print(f"Model size: {cfg.model.size}")
    print(f"Attention dim: {cfg.model.attention_dim}")
    print(f"Pretrain epochs: {cfg.training.pretrain_epochs}")
    print(f"Data root: {cfg.data.data_root}")
    print(f"Initial input channels: {cfg.model.input_channels}")
    print(f"Config as dict: {cfg.to_dict()['model']['size']}")
    print("-" * 30)

    print("--- Test 2: Update Model Size ---")
    cfg.model.size = "Tiny"
    cfg.update_for_model_size()
    print(f"New model size: {cfg.model.size}")
    print(f"New attention dim: {cfg.model.attention_dim}")
    print(f"New num layers: {cfg.model.num_layers}")
    print("-" * 30)

    print("--- Test 3: Set Dynamic Channels ---")
    cfg.set_dynamic_channels(input_channels=5, output_channels=5)
    print(f"Dynamic input channels: {cfg.model.input_channels}")
    print(f"Dynamic output channels: {cfg.model.output_channels}")
    print("-" * 30)

    print("--- Test 4: Update for Experiment Type (Pretrain) ---")
    cfg.update_for_experiment_type("pretrain")
    print(f"Current epochs (pretrain): {cfg.training.current_epochs}")
    print(f"Current warmup epochs (pretrain): {cfg.training.current_warmup_epochs}")
    print(f"Current data info (pretrain keys): {list(cfg.data.current_data_info.keys())}")
    print("-" * 30)

    print("--- Test 5: Update for Experiment Type (Finetune) ---")
    cfg.update_for_experiment_type("finetune", dataset_name="FNO_1e-5")
    print(f"Current epochs (finetune): {cfg.training.current_epochs}")
    print(f"Current warmup epochs (finetune): {cfg.training.current_warmup_epochs}")
    print(f"Current data info (finetune FNO_1e-5): {cfg.data.current_data_info['FNO_1e-5']}")
    print("-" * 30)

    print("--- Test 6: Command-line Overrides (simulated) ---")
    # Simulate cmd args as they would be parsed by argparse
    pseudo_cmd_args = argparse.Namespace(
        config_path=temp_config_path, # Not overridden as it's passed directly to Config()
        experiment_name="my_custom_exp",
        seed=123,
        output_dir="/tmp/output",
        checkpoint_dir="my_checkpoints",
        log_dir="my_logs",
        # Model overrides
        **{f"model.{k}": None for k in Config._MODEL_SIZE_CONFIGS['Small'].keys()}, # Set defaults to None
        **{f"training.{k}": None for k in ['optimizer', 'learning_rate', 'weight_decay', 'beta1', 'beta2', 'batch_size', 'noise_epsilon', 'load_balance_weight', 'pretrain_epochs', 'pretrain_warmup_epochs', 'finetune_epochs', 'finetune_warmup_epochs', 'downstream_epochs', 'downstream_warmup_epochs', 'log_interval', 'eval_interval', 'save_interval']}, # Set defaults to None
        **{f"data.{k}": None for k in ['data_root', 'interpolation_method', 'padding_value']}, # Set defaults to None
        **{f"distributed.{k}": None for k in ['world_size', 'backend']}, # Set defaults to None
        model_size="Medium", # Override model size
        model_num_routed_experts=32, # Override this
        model_num_shared_experts=3, # Override this
        training_learning_rate=0.0001, # Override learning rate
        training_batch_size=32, # Override batch size
        data_data_root="/mnt/pde_data", # Override data root
        distributed_world_size=4 # Override world size
    )

    cfg_cli = Config(config_path=temp_config_path, cmd_args=pseudo_cmd_args)
    print(f"CLI Experiment Name: {cfg_cli.experiment_name}")
    print(f"CLI Seed: {cfg_cli.seed}")
    print(f"CLI Output Dir: {cfg_cli.output_dir}")
    print(f"CLI Checkpoint Dir: {cfg_cli.checkpoint_dir}")
    print(f"CLI Log Dir: {cfg_cli.log_dir}")
    print(f"CLI Model Size: {cfg_cli.model.size}") # Should be Medium
    print(f"CLI Model Attention Dim: {cfg_cli.model.attention_dim}") # Should be updated by Medium size
    print(f"CLI Num Routed Experts: {cfg_cli.model.num_routed_experts}") # Should be 32
    print(f"CLI Num Shared Experts: {cfg_cli.model.num_shared_experts}") # Should be 3
    print(f"CLI Learning Rate: {cfg_cli.training.learning_rate}") # Should be 0.0001
    print(f"CLI Batch Size: {cfg_cli.training.batch_size}") # Should be 32
    print(f"CLI Data Root: {cfg_cli.data.data_root}") # Should be /mnt/pde_data
    print(f"CLI Distributed World Size: {cfg_cli.distributed.world_size}") # Should be 4
    print("-" * 30)
    
    # Test validation
    try:
        cfg.model.input_spatial_resolution = 127
        cfg._post_init_validation()
    except ValueError as e:
        print(f"Validation error caught: {e}")

    # Clean up dummy config file
    os.remove(temp_config_path)
