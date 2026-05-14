import yaml
from easydict import EasyDict
from typing import Any, Dict, List, Optional
import os
import json # For __str__ representation

class Config:
    """
    Manages all configuration parameters for the NaViL project, loading them from a YAML file.
    It provides structured access to model architecture, training, data, and evaluation settings.
    """

    def __init__(self):
        """
        Initializes the Config object with an empty EasyDict to store configuration data.
        """
        self._config_data: EasyDict = EasyDict()

    def load_config(self, path: str, model_variant_name: str) -> None:
        """
        Loads the configuration from a YAML file, selecting a specific model variant.

        This method first loads common settings, then special tokens, then the specific
        model variant's settings, and finally evaluation settings. Model variant settings
        will take precedence for any conflicting keys with common settings (though the
        current YAML structure avoids this at the top level).

        Args:
            path: Path to the YAML configuration file (e.g., "config.yaml").
            model_variant_name: The name of the model variant to load (e.g., "navil_2b", "navil_9b").

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If the YAML format is invalid, the model variant is not found,
                        or if any critical configuration parameter fails validation.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found at: {path}")

        with open(path, 'r') as f:
            full_config = yaml.safe_load(f)

        if not isinstance(full_config, dict):
            raise ValueError("Invalid YAML configuration format. Expected a dictionary.")

        # Load common parameters
        if 'common' in full_config:
            self._config_data.update(EasyDict(full_config['common']))
        else:
            print("Warning: 'common' section not found in config.yaml. Proceeding without common settings.")

        # Load special tokens
        if 'special_tokens' in full_config:
            self._config_data.update(EasyDict(full_config['special_tokens']))
        else:
            print("Warning: 'special_tokens' section not found in config.yaml. Proceeding without special tokens.")

        # Load model variant specific parameters
        if 'model_variants' not in full_config or model_variant_name not in full_config['model_variants']:
            raise ValueError(
                f"Model variant '{model_variant_name}' not found in 'model_variants' section of the config file. "
                f"Available variants: {list(full_config.get('model_variants', {}).keys())}"
            )
        
        model_variant_config = EasyDict(full_config['model_variants'][model_variant_name])
        self._config_data.update(model_variant_config)

        # Apply specific overrides/clarifications from the paper for NaViL-9B
        # This handles the contradiction where Table 8 for NaViL-9B S1.1 shows VMP enabled,
        # but Appendix A text states it's disabled for acceleration.
        if model_variant_name == "navil_9b":
            if 'training_stages' in self._config_data and 'stage_1_1' in self._config_data.training_stages:
                self._config_data.training_stages.stage_1_1.visual_multi_scale_packing = False
                print(f"Info: For {model_variant_name}, visual_multi_scale_packing for stage_1_1 "
                      "has been explicitly set to False based on Appendix A clarification.")
            # Note: global_batch_size for navil_9b's stage_2 is `null` in config.yaml.
            # EasyDict handles this as None, which will be processed by the trainer.

        # Load evaluation settings
        if 'evaluation' in full_config:
            self._config_data.update(EasyDict(full_config['evaluation']))
        else:
            print("Warning: 'evaluation' section not found in config.yaml. Proceeding without evaluation settings.")
        
        # Perform validation after all configuration is loaded
        self._validate_config()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value using a dot-separated key (e.g., "model_architecture.visual_encoder.depth").

        Args:
            key: The dot-separated key path to the configuration value.
            default: The default value to return if the key is not found.

        Returns:
            The configuration value or the default value if the key path does not exist.
        """
        keys = key.split('.')
        current_config = self._config_data
        for k in keys:
            if isinstance(current_config, dict) and k in current_config:
                current_config = current_config[k]
            else:
                return default
        return current_config

    def _validate_config(self) -> None:
        """
        Performs sanity checks on the loaded configuration to ensure critical parameters
        are present, have expected types, and fall within reasonable bounds.

        Raises:
            ValueError: If any critical configuration parameter is invalid or missing.
        """
        # --- Common Parameters Validation ---
        llm_max_seq_len = self.get("llm_max_sequence_length")
        if not isinstance(llm_max_seq_len, int) or llm_max_seq_len <= 0:
            raise ValueError("Config validation error: 'llm_max_sequence_length' must be a positive integer.")
        
        optimizer_name = self.get("optimizer.name")
        if not isinstance(optimizer_name, str) or not optimizer_name:
            raise ValueError("Config validation error: 'optimizer.name' is missing or invalid.")
        
        valid_precisions = ["bfloat16", "float32", "float16"]
        numerical_precision = self.get("numerical_precision")
        if numerical_precision not in valid_precisions:
            raise ValueError(f"Config validation error: 'numerical_precision' must be one of {valid_precisions}, got '{numerical_precision}'.")

        # --- Special Tokens Validation ---
        special_token_keys = ["begin_of_image", "end_of_image", "end_of_line", "end_of_scale"]
        for token_key in special_token_keys:
            token_value = self.get(f"special_tokens.{token_key}")
            if not isinstance(token_value, str) or not token_value:
                raise ValueError(f"Config validation error: Special token '{token_key}' is missing or invalid.")

        # --- Model Architecture Validation ---
        model_arch = self.get("model_architecture")
        if not model_arch:
            raise ValueError("Config validation error: 'model_architecture' section is missing.")

        # Visual encoder checks
        ve_config = self.get("model_architecture.visual_encoder")
        if not ve_config:
            raise ValueError("Config validation error: 'model_architecture.visual_encoder' section is missing.")
        if not isinstance(ve_config.depth, int) or ve_config.depth <= 0:
            raise ValueError("Config validation error: Visual encoder 'depth' must be a positive integer.")
        if not isinstance(ve_config.width, int) or ve_config.width <= 0:
            raise ValueError("Config validation error: Visual encoder 'width' must be a positive integer.")
        if not isinstance(ve_config.mlp_width, int) or ve_config.mlp_width <= 0:
            raise ValueError("Config validation error: Visual encoder 'mlp_width' must be a positive integer.")
        if not isinstance(ve_config.num_attention_heads, int) or ve_config.num_attention_heads <= 0:
            raise ValueError("Config validation error: Visual encoder 'num_attention_heads' must be a positive integer.")
        if not isinstance(ve_config.patch_embedding_stride, int) or ve_config.patch_embedding_stride <= 0:
            raise ValueError("Config validation error: Visual encoder 'patch_embedding_stride' must be a positive integer.")
        
        # LLM MoE checks
        llm_moe_config = self.get("model_architecture.llm_moe")
        if not llm_moe_config:
            raise ValueError("Config validation error: 'model_architecture.llm_moe' section is missing.")
        if not isinstance(llm_moe_config.num_experts, int) or llm_moe_config.num_experts <= 0:
            raise ValueError("Config validation error: LLM MoE 'num_experts' must be a positive integer.")
        if not isinstance(llm_moe_config.depth, int) or llm_moe_config.depth <= 0:
            raise ValueError("Config validation error: LLM MoE 'depth' must be a positive integer.")
        if not isinstance(llm_moe_config.width, int) or llm_moe_config.width <= 0:
            raise ValueError("Config validation error: LLM MoE 'width' must be a positive integer.")
        if not isinstance(llm_moe_config.mlp_width, int) or llm_moe_config.mlp_width <= 0:
            raise ValueError("Config validation error: LLM MoE 'mlp_width' must be a positive integer.")
        if not isinstance(llm_moe_config.num_attention_heads, int) or llm_moe_config.num_attention_heads <= 0:
            raise ValueError("Config validation error: LLM MoE 'num_attention_heads' must be a positive integer.")

        # --- Training Stages Validation ---
        training_stages = self.get("training_stages")
        if not training_stages:
            raise ValueError("Config validation error: 'training_stages' section is missing.")
        
        for stage_name, stage_config in training_stages.items():
            if not isinstance(stage_config, EasyDict):
                raise ValueError(f"Config validation error: Training stage '{stage_name}' configuration is not a dictionary.")
            
            if not isinstance(stage_config.training_steps, int) or stage_config.training_steps <= 0:
                raise ValueError(f"Config validation error: Training stage '{stage_name}' 'training_steps' must be a positive integer.")
            
            # global_batch_size can be None for unspecified values (e.g., NaViL-9B S2)
            if stage_config.global_batch_size is not None and (not isinstance(stage_config.global_batch_size, int) or stage_config.global_batch_size <= 0):
                raise ValueError(f"Config validation error: Training stage '{stage_name}' 'global_batch_size' must be a positive integer or None.")
            
            if not isinstance(stage_config.peak_learning_rate, (float, int)) or stage_config.peak_learning_rate <= 0:
                raise ValueError(f"Config validation error: Training stage '{stage_name}' 'peak_learning_rate' must be a positive number.")
            
            if not isinstance(stage_config.lr_schedule, str) or not stage_config.lr_schedule:
                raise ValueError(f"Config validation error: Training stage '{stage_name}' 'lr_schedule' is missing or invalid.")
            
            if not isinstance(stage_config.visual_multi_scale_packing, bool):
                raise ValueError(f"Config validation error: Training stage '{stage_name}' 'visual_multi_scale_packing' must be a boolean.")

        # --- Data Paths Validation ---
        data_paths = self.get("data_paths")
        if not data_paths:
            print("Warning: 'data_paths' section is missing. Data loading might fail if not configured elsewhere.")
        else:
            for stage_name, paths in data_paths.items():
                if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
                    print(f"Warning: Data paths for stage '{stage_name}' are not a list of strings or are empty.")

        # --- Evaluation Settings Validation ---
        eval_config = self.get("evaluation")
        if not eval_config:
            print("Warning: 'evaluation' section is missing. Evaluation might not run as expected.")
        else:
            if not isinstance(eval_config.benchmarks, list):
                raise ValueError("Config validation error: 'evaluation.benchmarks' must be a list.")
            if not isinstance(eval_config.output_dir, str) or not eval_config.output_dir:
                raise ValueError("Config validation error: 'evaluation.output_dir' is missing or invalid.")
            if not isinstance(eval_config.vmp_area_threshold, (int, float)) or eval_config.vmp_area_threshold <= 0:
                raise ValueError("Config validation error: 'evaluation.vmp_area_threshold' must be a positive number.")

    def __str__(self) -> str:
        """
        Returns a formatted string representation of the configuration data.
        """
        return json.dumps(self._config_data, indent=2)

    def __repr__(self) -> str:
        """
        Returns a developer-friendly string representation of the Config object.
        """
        return f"Config({repr(self._config_data)})"
