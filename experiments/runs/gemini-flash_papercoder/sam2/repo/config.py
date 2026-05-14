import argparse
import os
import yaml
import torch
from typing import Any, Dict, List, Optional, Union
from collections.abc import Mapping


class _ConfigNamespace(object):
    """
    A helper class to allow dot-notation access to nested dictionary values.
    """

    def __init__(self, data: Union[Dict, Any]):
        if isinstance(data, Mapping):
            self.__dict__['_data'] = {
                k: _ConfigNamespace(v) for k, v in data.items()
            }
        else:
            self.__dict__['_data'] = data

    def __getattr__(self, name: str) -> Any:
        if isinstance(self.__dict__['_data'], Mapping):
            if name in self.__dict__['_data']:
                return self.__dict__['_data'][name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        # Allow dict-like access for convenience
        if isinstance(self.__dict__['_data'], Mapping):
            if key in self.__dict__['_data']:
                return self.__dict__['_data'][key]
        raise KeyError(f"'{self.__class__.__name__}' object has no key '{key}'")

    def __setattr__(self, name: str, value: Any):
        if isinstance(self.__dict__['_data'], Mapping):
            self.__dict__['_data'][name] = _ConfigNamespace(value)
        else:
            raise TypeError("Cannot set attribute on non-mapping data.")

    def __repr__(self) -> str:
        return repr(self.__dict__['_data'])

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Safely retrieves a nested value using a dot-separated key path.
        """
        keys = key_path.split('.')
        current_data = self.__dict__['_data']
        for key in keys:
            if isinstance(current_data, Mapping) and key in current_data:
                current_data = current_data[key]
                if isinstance(current_data, _ConfigNamespace):
                    current_data = current_data.__dict__['_data']
            else:
                return default
        if isinstance(current_data, _ConfigNamespace):
            return current_data.__dict__['_data']
        return current_data


def _set_nested_config(config_dict: Dict, key_path: str, value: Any) -> None:
    """Helper to set a value in a nested dictionary given a dot-separated key path."""
    keys = key_path.split('.')
    current = config_dict
    for i, key in enumerate(keys):
        if key not in current:
            if i == len(keys) - 1:
                try: # Attempt type conversion for common types
                    if isinstance(value, str):
                        if value.lower() == 'true': value = True
                        elif value.lower() == 'false': value = False
                        elif value.isdigit(): value = int(value)
                        elif value.replace('.', '', 1).isdigit(): value = float(value)
                except ValueError:
                    pass # Keep as string if conversion fails
                current[key] = value
                return
            else:
                current[key] = {}
        if not isinstance(current[key], Dict):
            if i == len(keys) - 1:
                try: # Attempt type conversion for common types
                    if isinstance(value, str):
                        if value.lower() == 'true': value = True
                        elif value.lower() == 'false': value = False
                        elif value.isdigit(): value = int(value)
                        elif value.replace('.', '', 1).isdigit(): value = float(value)
                except ValueError:
                    pass # Keep as string if conversion fails
                current[key] = value
                return
            else:
                raise TypeError(f"Cannot set nested key '{key_path}': '{key}' is not a dictionary.")
        current = current[key]
    current = value # Should not be reached for non-final keys


class Config(object):
    """
    Manages experiment configurations, loading from YAML, applying CLI overrides,
    and validating settings.
    """

    def __init__(self, config_path: str, cli_args: Optional[argparse.Namespace] = None):
        """
        Initializes the configuration object.

        Args:
            config_path (str): Path to the YAML configuration file.
            cli_args (Optional[argparse.Namespace]): Parsed command-line arguments.
        """
        self._config_path: str = config_path
        self._raw_data: Dict = self._load_yaml(config_path)

        if cli_args:
            self._apply_cli_overrides(cli_args)

        self._data: _ConfigNamespace = _ConfigNamespace(self._raw_data)
        self.validate()

    def _load_yaml(self, config_path: str) -> Dict:
        """
        Loads configuration from a YAML file.

        Args:
            config_path (str): Path to the YAML file.

        Returns:
            Dict: Parsed YAML content.

        Raises:
            FileNotFoundError: If the config file does not exist.
            yaml.YAMLError: If there's an error parsing the YAML file.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error parsing YAML configuration file: {e}")

    def _apply_cli_overrides(self, cli_args: argparse.Namespace) -> None:
        """
        Applies command-line arguments as overrides to the loaded configuration.
        This method anticipates common CLI arguments for direct overrides and
        a generic `--set` argument for arbitrary nested configurations.

        Args:
            cli_args (argparse.Namespace): Parsed command-line arguments.
        """
        override_map: Dict[str, str] = {
            'learning_rate': 'training.full_train.learning_rate',
            'batch_size': 'training.full_train.video_batch_size', # Assuming video batch size for simplicity if generic
            'device': 'system.device',
            'model_image_encoder_type': 'model.image_encoder.type',
            'seq_len': 'training.full_train.seq_len',
            'num_gpus': 'system.num_gpus',
            # Add other direct CLI argument mappings here if main.py defines them
        }

        # Apply direct overrides
        for cli_attr, config_path in override_map.items():
            if hasattr(cli_args, cli_attr) and getattr(cli_args, cli_attr) is not None:
                _set_nested_config(self._raw_data, config_path, getattr(cli_args, cli_attr))

        # Apply generic --set overrides (if 'set_config' attribute exists)
        if hasattr(cli_args, 'set_config') and cli_args.set_config:
            for item in cli_args.set_config:
                if '=' in item:
                    key_path, value_str = item.split('=', 1)
                    _set_nested_config(self._raw_data, key_path.strip(), value_str.strip())
                else:
                    print(f"Warning: Ignoring malformed --set argument: {item}. Expected 'key.path=value'.")

    def validate(self) -> None:
        """
        Validates the entire configuration for completeness, correctness, and consistency.

        Raises:
            ValueError: If any configuration parameter is invalid or inconsistent.
        """
        print("Validating configuration...")

        # --- Path Validation ---
        data_dir = self.get('paths.data_dir')
        if not data_dir or not os.path.isdir(data_dir):
            raise ValueError(f"Invalid or non-existent base data directory: {data_dir}")

        for sub_dir_key in ['sa1b_dir', 'sav_dir', 'vos_dir']:
            sub_dir_name = self.get(f'paths.{sub_dir_key}')
            if sub_dir_name:
                full_path = os.path.join(data_dir, sub_dir_name)
                if not os.path.exists(full_path):
                    print(f"Warning: Data directory '{full_path}' for '{sub_dir_key}' does not exist. This might be fine if not using this dataset.")
        
        # Output directories: ensure parent exists, don't create here
        for out_dir_key in ['checkpoint_dir', 'log_dir']:
            path = self.get(f'paths.{out_dir_key}')
            if not path:
                raise ValueError(f"Path for '{out_dir_key}' cannot be empty.")

        # --- Model Configuration Validation ---
        model_cfg = self.get('model')
        if not model_cfg:
            raise ValueError("Model configuration is missing.")

        img_encoder_type = self.get('model.image_encoder.type')
        supported_hiera_types = ["Hiera-T", "Hiera-S", "Hiera-B+", "Hiera-L"]
        if img_encoder_type not in supported_hiera_types:
            raise ValueError(f"Invalid image_encoder.type: {img_encoder_type}. Must be one of {supported_hiera_types}")
        
        # Validate global_attn_blocks consistency with image_encoder.type
        expected_global_attn_blocks = {
            "Hiera-T": [5, 7, 9],
            "Hiera-S": [7, 10, 13],
            "Hiera-B+": [12, 16, 20],
            "Hiera-L": [23, 33, 43],
        }
        actual_global_attn_blocks = self.get('model.image_encoder.global_attn_blocks')
        if actual_global_attn_blocks != expected_global_attn_blocks[img_encoder_type]:
            print(f"Warning: model.image_encoder.global_attn_blocks ({actual_global_attn_blocks}) does not match expected for {img_encoder_type} ({expected_global_attn_blocks[img_encoder_type]}).")

        mem_attn_layers = self.get('model.memory_attention.num_layers')
        if not isinstance(mem_attn_layers, int) or mem_attn_layers <= 0:
            raise ValueError(f"model.memory_attention.num_layers must be a positive integer, got {mem_attn_layers}")
        
        for mb_key in ['max_recent_frames', 'max_prompted_frames', 'memory_feature_dim', 'object_pointer_dim']:
            val = self.get(f'model.memory_bank.{mb_key}')
            if not isinstance(val, int) or val <= 0:
                raise ValueError(f"model.memory_bank.{mb_key} must be a positive integer, got {val}")
        
        # --- Training Configuration Validation ---
        train_cfg = self.get('training')
        if not train_cfg:
            raise ValueError("Training configuration is missing.")

        # Pre-training
        if self.get('training.pretrain.enabled'):
            pretrain_cfg = self.get('training.pretrain')
            if not isinstance(pretrain_cfg.steps, int) or pretrain_cfg.steps <= 0:
                raise ValueError(f"pretrain.steps must be positive integer, got {pretrain_cfg.steps}")
            if not isinstance(pretrain_cfg.resolution, int) or pretrain_cfg.resolution <= 0:
                raise ValueError(f"pretrain.resolution must be positive integer, got {pretrain_cfg.resolution}")
            if not isinstance(pretrain_cfg.batch_size, int) or pretrain_cfg.batch_size <= 0:
                raise ValueError(f"pretrain.batch_size must be positive integer, got {pretrain_cfg.batch_size}")
            if not isinstance(pretrain_cfg.learning_rate, (int, float)) or pretrain_cfg.learning_rate <= 0:
                raise ValueError(f"pretrain.learning_rate must be positive, got {pretrain_cfg.learning_rate}")
            
            supported_precisions = ["bfloat16", "float32", "float16"]
            if pretrain_cfg.precision not in supported_precisions:
                raise ValueError(f"pretrain.precision must be one of {supported_precisions}, got {pretrain_cfg.precision}")
            if pretrain_cfg.optimizer != "AdamW": # Only AdamW specified
                raise ValueError(f"pretrain.optimizer must be 'AdamW', got {pretrain_cfg.optimizer}")
            
            # Loss weights
            for loss_key in ['mask_focal_weight', 'mask_dice_weight', 'iou_l1_weight']:
                val = self.get(f'training.pretrain.losses.{loss_key}')
                if not isinstance(val, (int, float)) or val < 0:
                    raise ValueError(f"Pretrain loss weight {loss_key} must be non-negative, got {val}")

        # Full Training
        if self.get('training.full_train.enabled'):
            full_train_cfg = self.get('training.full_train')
            sampling_weights = full_train_cfg.dataset_sampling_weights
            if not isinstance(sampling_weights, Dict) or not sampling_weights:
                raise ValueError("full_train.dataset_sampling_weights must be a non-empty dictionary.")
            total_weight = sum(sampling_weights.values())
            if not (0.99 <= total_weight <= 1.01): # Allow small float discrepancies
                print(f"Warning: full_train.dataset_sampling_weights sum to {total_weight}, not 1.0.")

            if not isinstance(full_train_cfg.seq_len, int) or full_train_cfg.seq_len <= 0:
                raise ValueError(f"full_train.seq_len must be a positive integer, got {full_train_cfg.seq_len}")
            
            if not isinstance(full_train_cfg.max_prompted_frames_per_seq, int) or \
               not (0 <= full_train_cfg.max_prompted_frames_per_seq <= full_train_cfg.seq_len):
                raise ValueError(f"full_train.max_prompted_frames_per_seq must be between 0 and seq_len ({full_train_cfg.seq_len}), got {full_train_cfg.max_prompted_frames_per_seq}")
            
            for bs_key in ['sa1b_batch_size', 'video_batch_size']:
                val = self.get(f'training.full_train.{bs_key}')
                if not isinstance(val, int) or val <= 0:
                    raise ValueError(f"full_train.{bs_key} must be a positive integer, got {val}")

            # Probabilities validation
            for prob_key in ['corrective_click_sampling_gt_prob', 'mosaic_transform_prob', 'reverse_temporal_order_prob']:
                val = self.get(f'training.full_train.{prob_key}')
                if not isinstance(val, (int, float)) or not (0.0 <= val <= 1.0):
                    raise ValueError(f"full_train.{prob_key} must be a float between 0.0 and 1.0, got {val}")
            
            initial_prompt_strategy = self.get('training.full_train.initial_prompt_strategy')
            if not initial_prompt_strategy or not isinstance(initial_prompt_strategy, Dict):
                raise ValueError("full_train.initial_prompt_strategy must be a dictionary.")
            total_initial_prompt_prob = sum(initial_prompt_strategy.values())
            if not (0.99 <= total_initial_prompt_prob <= 1.01):
                print(f"Warning: full_train.initial_prompt_strategy probabilities sum to {total_initial_prompt_prob}, not 1.0.")

            # Loss weights for full training
            for loss_key in ['mask_focal_weight', 'mask_dice_weight', 'iou_l1_weight', 'occlusion_ce_weight']:
                val = self.get(f'training.full_train.losses.{loss_key}')
                if not isinstance(val, (int, float)) or val < 0:
                    raise ValueError(f"Full train loss weight {loss_key} must be non-negative, got {val}")

        # Fine-tuning
        if self.get('training.finetune.enabled'):
            finetune_cfg = self.get('training.finetune')
            if not isinstance(finetune_cfg.seq_len, int) or finetune_cfg.seq_len <= 0:
                raise ValueError(f"finetune.seq_len must be a positive integer, got {finetune_cfg.seq_len}")
            if not isinstance(finetune_cfg.iterations, int) or finetune_cfg.iterations <= 0:
                raise ValueError(f"finetune.iterations must be a positive integer, got {finetune_cfg.iterations}")
            if not isinstance(finetune_cfg.learning_rate_multiplier, (int, float)) or finetune_cfg.learning_rate_multiplier <= 0:
                raise ValueError(f"finetune.learning_rate_multiplier must be positive, got {finetune_cfg.learning_rate_multiplier}")

        # --- System Configuration Validation ---
        system_cfg = self.get('system')
        if not system_cfg:
            raise ValueError("System configuration is missing.")
        
        device = system_cfg.device
        if device not in ["cuda", "cpu"]:
            raise ValueError(f"system.device must be 'cuda' or 'cpu', got {device}")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is selected as device but is not available.")
        
        num_gpus = system_cfg.num_gpus
        if not isinstance(num_gpus, int) or num_gpus < 0:
            raise ValueError(f"system.num_gpus must be a non-negative integer, got {num_gpus}")
        if num_gpus > 0 and device == "cpu":
            raise ValueError("system.num_gpus > 0 requires system.device to be 'cuda'.")
        if num_gpus > torch.cuda.device_count():
            print(f"Warning: system.num_gpus ({num_gpus}) is greater than available CUDA devices ({torch.cuda.device_count()}). Will use available devices.")
            # Set to available to prevent issues if not using distributed setup.
            # In distributed setting, rank will limit visible devices.
            self._raw_data['system']['num_gpus'] = torch.cuda.device_count() if torch.cuda.is_available() else 0


        # --- Evaluation Configuration Validation ---
        eval_cfg = self.get('evaluation')
        if not eval_cfg:
            raise ValueError("Evaluation configuration is missing.")
        
        interactive_cfg = eval_cfg.interactive
        for key in ['num_clicks', 'max_interacted_frames']:
            val = interactive_cfg.get(key)
            if not isinstance(val, int) or val <= 0:
                raise ValueError(f"evaluation.interactive.{key} must be a positive integer, got {val}")
        for key in ['T_loc', 'T_click', 'T_exam_300frame']:
            val = interactive_cfg.get(key)
            if not isinstance(val, (int, float)) or val <= 0:
                raise ValueError(f"evaluation.interactive.{key} must be a positive number, got {val}")
        
        for key in ['prompt_types']: # for semi-supervised and image_segmentation
            ss_val = eval_cfg.semisupervised_vos.get(key)
            img_val = eval_cfg.image_segmentation.get(key)
            if not isinstance(ss_val, list) or not ss_val:
                raise ValueError(f"evaluation.semisupervised_vos.{key} must be a non-empty list.")
            if not isinstance(img_val, list) or not img_val:
                raise ValueError(f"evaluation.image_segmentation.{key} must be a non-empty list.")

        print("Configuration validated successfully.")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Public method to safely retrieve a nested configuration value.

        Args:
            key_path (str): Dot-separated path to the configuration key (e.g., "model.image_encoder.type").
            default (Any, optional): Default value to return if the key path is not found. Defaults to None.

        Returns:
            Any: The value associated with the key path, or the default value if not found.
        """
        return self._data.get(key_path, default)

    def __getattr__(self, name: str) -> Any:
        """
        Enables direct attribute access for top-level configuration sections.
        """
        if hasattr(self._data, name):
            return getattr(self._data, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __repr__(self) -> str:
        """
        Provides a string representation of the configuration.
        """
        return f"Config(path='{self._config_path}', data={self._data})"

