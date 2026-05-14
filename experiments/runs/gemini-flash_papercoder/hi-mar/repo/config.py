import yaml
from typing import Dict, Any, Tuple

class Config:
    """
    Manages loading and accessing configuration parameters from a YAML file.
    Provides static methods to retrieve specific sections of the configuration.
    """

    _config_data: Dict[str, Any] = {}
    _loaded: bool = False

    @staticmethod
    def load_config(path: str) -> Dict[str, Any]:
        """
        Loads the configuration from the specified YAML file.

        Args:
            path: The file path to the config.yaml.

        Returns:
            A dictionary containing the loaded configuration.

        Raises:
            FileNotFoundError: If the config file does not exist.
            yaml.YAMLError: If there is an error parsing the YAML file.
        """
        if Config._loaded:
            print(f"Warning: Config already loaded. Reloading from {path}.")
        try:
            with open(path, 'r') as f:
                Config._config_data = yaml.safe_load(f)
            Config._loaded = True
            return Config._config_data
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at: {path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML configuration file: {e}")

    @staticmethod
    def _ensure_config_loaded():
        """Ensures that the configuration has been loaded."""
        if not Config._loaded:
            raise RuntimeError(
                "Configuration not loaded. Call Config.load_config(path) first."
            )

    @staticmethod
    def get_global_config() -> Dict[str, Any]:
        """
        Retrieves general application settings.

        Returns:
            A dictionary with global configuration parameters.
        """
        Config._ensure_config_loaded()
        return {
            "device": Config._config_data.get("device", "cpu"),
            "output_dir": Config._config_data.get("output_dir", "./output"),
            "project_name": Config._config_data.get("project_name", "himar_project"),
            "seed": Config._config_data.get("seed", 42),
        }

    @staticmethod
    def get_tokenizer_config() -> Dict[str, Any]:
        """
        Retrieves configuration for the VAE tokenizer.

        Returns:
            A dictionary with VAE tokenizer parameters.
        """
        Config._ensure_config_loaded()
        tokenizer_cfg = Config._config_data.get("tokenizer", {})
        return {
            "vae_path": tokenizer_cfg.get("vae_path", "path/to/pretrained/mar_kl16_vae.pt"),
            "latent_channels": tokenizer_cfg.get("latent_channels", 4),
            "high_res_image_size": tokenizer_cfg.get("high_res_image_size", 256),
            "low_res_image_size": tokenizer_cfg.get("low_res_image_size", 128),
        }

    @staticmethod
    def get_clip_text_encoder_config() -> Dict[str, Any]:
        """
        Retrieves configuration for the CLIP text encoder.

        Returns:
            A dictionary with CLIP text encoder parameters.
        """
        Config._ensure_config_loaded()
        clip_cfg = Config._config_data.get("clip_text_encoder", {})
        return {
            "model_name": clip_cfg.get("model_name", "openai/clip-vit-large-patch14"),
        }

    @staticmethod
    def get_model_config(model_size: str) -> Dict[str, Any]:
        """
        Retrieves a consolidated dictionary of hyperparameters for building the HiMARModel.

        Args:
            model_size: The variant of the Hi-MAR model (e.g., "Hi-MAR-B", "Hi-MAR-L", "Hi-MAR-H").

        Returns:
            A dictionary containing all parameters needed to instantiate the HiMARModel.

        Raises:
            ValueError: If the specified model_size is not found in the configuration.
        """
        Config._ensure_config_loaded()
        model_cfg = Config._config_data.get("model_config", {})
        variants_details = model_cfg.get("variants_details", {})
        common_transformer_params = model_cfg.get("common_transformer_params", {})

        if model_size not in variants_details:
            raise ValueError(
                f"Model size '{model_size}' not found in model_config.variants_details. "
                f"Available sizes: {list(variants_details.keys())}"
            )

        variant_cfg = variants_details[model_size]
        tokenizer_cfg = Config.get_tokenizer_config()
        global_cfg = Config.get_global_config()

        # Consolidate all model-related parameters
        config = {
            # Global/Common parameters
            "device": global_cfg["device"],
            "num_scales": model_cfg.get("num_scales", 2),
            "tokenizer_latent_channels": tokenizer_cfg["latent_channels"],
            "low_res_image_size": tokenizer_cfg["low_res_image_size"],
            "high_res_image_size": tokenizer_cfg["high_res_image_size"],

            # Hi-MAR Transformer common parameters
            "num_attention_heads": common_transformer_params.get("num_attention_heads", 12),
            "dropout_rate": common_transformer_params.get("dropout_rate", 0.1),
            "ffn_multiplier": common_transformer_params.get("ffn_multiplier", 4),
            "activation_fn": common_transformer_params.get("activation_fn", "GELU"),

            # Variant-specific parameters
            "himar_transformer_layers": variant_cfg["himar_transformer_layers"],
            "himar_hidden_size": variant_cfg["himar_hidden_size"],
            "diff_head1_layers": variant_cfg["diff_head1_layers"],
            "diff_head1_hidden_size": variant_cfg["diff_head1_hidden_size"],
            "diff_head2_layers": variant_cfg["diff_head2_layers"],
            "diff_head2_hidden_size": variant_cfg["diff_head2_hidden_size"],
            # Inferred parameters based on paper's description
            # Both diffusion heads receive conditional tokens from the Hi-MAR Transformer,
            # which has himar_hidden_size as its output dimension.
            "diff_head1_input_dim": variant_cfg["himar_hidden_size"],
            "diff_head2_input_dim": variant_cfg["himar_hidden_size"],
        }
        return config

    @staticmethod
    def get_training_config(task_type: str) -> Dict[str, Any]:
        """
        Retrieves training-specific parameters for a given task (e.g., "imagenet", "mscoco").

        Args:
            task_type: The type of training task ("imagenet" or "mscoco").

        Returns:
            A dictionary with training configuration parameters.

        Raises:
            ValueError: If the specified task_type is not found in the configuration.
        """
        Config._ensure_config_loaded()
        training_cfg = Config._config_data.get("training", {})
        optimizer_cfg = training_cfg.get("optimizer", {})

        if task_type not in training_cfg:
            raise ValueError(
                f"Training task type '{task_type}' not found in training configuration. "
                f"Available types: {list(training_cfg.keys())}"
            )

        task_specific_cfg = training_cfg[task_type]

        config = {
            "optimizer": {
                "name": optimizer_cfg.get("name", "AdamW"),
                "betas": tuple(optimizer_cfg.get("betas", [0.9, 0.95])),
                "eps": optimizer_cfg.get("eps", 1e-8),
                "weight_decay": task_specific_cfg.get("weight_decay", 0.0), # Task-specific weight decay
            },
            "ema_momentum": training_cfg.get("ema_momentum", 0.9999),
            "gradient_accumulation_steps": training_cfg.get("gradient_accumulation_steps", 1),

            "enabled": task_specific_cfg.get("enabled", False),
            "dataset_name": task_specific_cfg.get("dataset_name", ""),
            "data_path": task_specific_cfg.get("data_path", "path/to/dataset"),
            "learning_rate": task_specific_cfg.get("learning_rate", 1e-4),
            "epochs": task_specific_cfg.get("epochs", 1),
            "warmup_epochs": task_specific_cfg.get("warmup_epochs"), # Can be None if warmup_steps is used
            "warmup_steps": task_specific_cfg.get("warmup_steps"), # Can be None if warmup_epochs is used
            "batch_size": task_specific_cfg.get("batch_size", 1),
            "phase1_masking_strategy": task_specific_cfg.get("phase1_masking_strategy", "random_uniform"),
            "phase1_masking_params": tuple(task_specific_cfg.get("phase1_masking_params", (0.0, 1.0))),
            "phase2_masking_strategy": task_specific_cfg.get("phase2_masking_strategy", "cosine"),
            "phase2_masking_params": tuple(task_specific_cfg.get("phase2_masking_params", (0.0, 1.0))),
            "conditional_type": task_specific_cfg.get("conditional_type", "class"),
            "log_every_n_steps": task_specific_cfg.get("log_every_n_steps", 100),
            "validate_every_n_epochs": task_specific_cfg.get("validate_every_n_epochs", 1),
        }
        return config

    @staticmethod
    def get_generation_config() -> Dict[str, Any]:
        """
        Retrieves configuration for the image generation process (inference).

        Returns:
            A dictionary with generation parameters.
        """
        Config._ensure_config_loaded()
        gen_cfg = Config._config_data.get("generation", {})
        inference_steps_cfg = gen_cfg.get("inference_steps", {})
        return {
            "noise_scheduler_type": gen_cfg.get("noise_scheduler_type", "cosine"),
            "inference_steps": {
                "phase1": inference_steps_cfg.get("phase1", 32),
                "phase2": inference_steps_cfg.get("phase2", 4),
            },
            "guidance_scale": gen_cfg.get("guidance_scale", 7.0),
            "phase1_cfg_on_for_eval": gen_cfg.get("phase1_cfg_on_for_eval", True),
            "num_generated_samples_per_run": gen_cfg.get("num_generated_samples_per_run", 16),
            "save_generated_images": gen_cfg.get("save_generated_images", True),
        }

    @staticmethod
    def get_evaluation_config(task_type: str) -> Dict[str, Any]:
        """
        Retrieves evaluation settings for a specific task (e.g., "imagenet", "mscoco").

        Args:
            task_type: The type of evaluation task ("imagenet" or "mscoco").

        Returns:
            A dictionary with evaluation configuration parameters.

        Raises:
            ValueError: If the specified task_type is not found in the configuration.
        """
        Config._ensure_config_loaded()
        eval_cfg = Config._config_data.get("evaluation", {})

        if task_type not in eval_cfg:
            raise ValueError(
                f"Evaluation task type '{task_type}' not found in evaluation configuration. "
                f"Available types: {list(eval_cfg.keys())}"
            )

        task_specific_cfg = eval_cfg[task_type]

        return {
            "enabled": task_specific_cfg.get("enabled", False),
            "num_samples": task_specific_cfg.get("num_samples", 1000),
            "metrics": task_specific_cfg.get("metrics", []),
            "real_features_path": task_specific_cfg.get("real_features_path", "path/to/real_features.npz"),
            "eval_with_cfg": task_specific_cfg.get("eval_with_cfg", True),
            "eval_without_cfg": task_specific_cfg.get("eval_without_cfg", False),
            "evaluation_prompts_file": task_specific_cfg.get("evaluation_prompts_file"), # Optional, e.g., for MS-COCO
        }

