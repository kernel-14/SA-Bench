import os
from omegaconf import OmegaConf, DictConfig

# Define a type alias for better readability
Config = DictConfig

def _apply_stage_overrides(
    stage_config: DictConfig,
    model_id: str,
    task_name: str,
    stage_name: str
) -> DictConfig:
    """
    Applies model-specific and task-specific overrides to a given stage's configuration.

    Args:
        stage_config: The base configuration for a stage (e.g., sft_config, rm_config).
        model_id: Identifier for the base model (e.g., "gemma_2b").
        task_name: Name of the task (e.g., "tldr_summarization").
        stage_name: The name of the stage (e.g., "sft", "rm", "ppo").

    Returns:
        A new DictConfig with overrides applied.
    """
    # Start with the default configuration for the stage
    resolved_config = stage_config.default.copy()

    # Apply model-specific overrides
    if model_id in stage_config.model_overrides:
        OmegaConf.merge(resolved_config, stage_config.model_overrides[model_id])

    # Apply task-specific overrides (most specific first, then general)
    task_key_specific = f"{task_name}_{model_id}"
    if task_key_specific in stage_config.task_overrides:
        OmegaConf.merge(resolved_config, stage_config.task_overrides[task_key_specific])
    elif task_name in stage_config.task_overrides:
        OmegaConf.merge(resolved_config, stage_config.task_overrides[task_name])

    return resolved_config


def load_config(yaml_path: str, model_id: str, task_name: str) -> DictConfig:
    """
    Loads the base configuration from a YAML file and applies model and task-specific overrides.

    Args:
        yaml_path: Path to the configuration YAML file (e.g., "config.yaml").
        model_id: Identifier for the base model (e.g., "gemma_2b", "codegemma_7b").
        task_name: Name of the task (e.g., "tldr_summarization", "apps_code_gen").

    Returns:
        A DictConfig object containing the full resolved configuration.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found at: {yaml_path}")

    base_cfg = OmegaConf.load(yaml_path)

    # Resolve SFT configuration
    base_cfg.sft_config = _apply_stage_overrides(base_cfg.sft_config, model_id, task_name, "sft")

    # Resolve RM configuration
    base_cfg.rm_config = _apply_stage_overrides(base_cfg.rm_config, model_id, task_name, "rm")
    # Special handling for APPS dataset if RM stage is to be skipped
    if task_name == "apps_code_gen" and "apps" in base_cfg.rm_config.task_overrides:
        if base_cfg.rm_config.task_overrides.apps.get("skip_training", False):
            base_cfg.rm_config.skip_training = True
    else:
        base_cfg.rm_config.skip_training = False


    # Resolve PPO configuration
    base_cfg.ppo_config = _apply_stage_overrides(base_cfg.ppo_config, model_id, task_name, "ppo")

    # Store the resolved model_id and task_name in the config for easy access
    base_cfg.resolved_model_id = model_id
    base_cfg.resolved_task_name = task_name

    return base_cfg


# Example usage (for testing or direct script execution)
if __name__ == "__main__":
    # Simulate command line arguments for model_id and task_name
    test_model_id = "gemma_7b"
    test_task_name = "tldr_summarization" # Or "webgpt", "apps_code_gen", etc.
    config_file = "config.yaml"

    try:
        cfg = load_config(config_file, test_model_id, test_task_name)
        print("--- Global Config ---")
        print(OmegaConf.to_yaml(cfg.global))
        print("\n--- Model Configs ---")
        # Print the relevant model config entry
        print(OmegaConf.to_yaml(cfg.model_configs[test_model_id]))
        print("\n--- Data Configs ---")
        print(OmegaConf.to_yaml(cfg.data_configs))
        print(f"\n--- SFT Config for {test_model_id} on {test_task_name} ---")
        print(OmegaConf.to_yaml(cfg.sft_config))
        print(f"\n--- RM Config for {test_model_id} on {test_task_name} ---")
        print(OmegaConf.to_yaml(cfg.rm_config))
        print(f"\n--- PPO Config for {test_model_id} on {test_task_name} ---")
        print(OmegaConf.to_yaml(cfg.ppo_config))
        print(f"\n--- Macro Action Config ---")
        print(OmegaConf.to_yaml(cfg.macro_action_config))
        print(f"\n--- Evaluation Config ---")
        print(OmegaConf.to_yaml(cfg.evaluation_config))

        # Test specific overrides
        print("\n--- Testing specific overrides ---")
        cfg_webgpt_gemma_7b = load_config(config_file, "gemma_7b", "webgpt_comparison")
        print(f"\nSFT epochs for Gemma-7B on WebGPT: {cfg_webgpt_gemma_7b.sft_config.epochs}")
        print(f"RM epochs for Gemma-7B on WebGPT: {cfg_webgpt_gemma_7b.rm_config.epochs}")
        print(f"PPO KL Coeff for Gemma-7B on WebGPT: {cfg_webgpt_gemma_7b.ppo_config.kl_coefficient}")
        print(f"PPO KL Coeff for Gemma-7B on TLDR: {cfg.ppo_config.kl_coefficient}") # Should be 0.01

        cfg_apps_codegemma_2b = load_config(config_file, "codegemma_2b", "apps_code_gen")
        print(f"\nRM skip_training for CodeGemma-2B on APPS: {cfg_apps_codegemma_2b.rm_config.skip_training}")
        print(f"SFT batch_size for CodeGemma-2B on APPS: {cfg_apps_codegemma_2b.sft_config.batch_size}")
        print(f"PPO warmup_steps for CodeGemma-2B on APPS: {cfg_apps_codegemma_2b.ppo_config.warmup_steps}")


    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")

