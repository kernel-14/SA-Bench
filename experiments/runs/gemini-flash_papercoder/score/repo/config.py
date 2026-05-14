import yaml
from typing import Any, Dict, Optional


class Config:
    """
    Configuration class to load and manage all experiment hyperparameters,
    file paths, and model configurations from a YAML file.
    """

    # Base model configuration
    base_model_name: Optional[str]
    task_type: Optional[str]

    # Paths
    math_train_path: Optional[str]
    math_eval_path: Optional[str]
    mbpp_train_path: Optional[str]
    human_eval_path: Optional[str]
    mbpp_r_path: Optional[str]
    checkpoint_dir: Optional[str]
    log_dir: Optional[str]

    # Training Hyperparameters - Stage I (MATH)
    training_stage1_math: Optional[Dict[str, Any]]

    # Training Hyperparameters - Stage II (MATH)
    training_stage2_math: Optional[Dict[str, Any]]

    # Training Hyperparameters - Stage I (MBPP)
    training_stage1_mbpp: Optional[Dict[str, Any]]

    # Training Hyperparameters - Stage II (MBPP)
    training_stage2_mbpp: Optional[Dict[str, Any]]

    # Evaluation settings
    evaluation: Optional[Dict[str, Any]]

    # PEFT (LoRA) settings (optional)
    use_peft: Optional[bool]
    peft_method: Optional[str]
    peft_lora_r: Optional[int]
    peft_lora_alpha: Optional[float]
    peft_lora_dropout: Optional[float]

    # Prompts
    prompts: Optional[Dict[str, str]]

    # Other settings
    seed: Optional[int]

    def __init__(self) -> None:
        """
        Initializes the Config class with all attributes set to None.
        These will be populated by loading from a YAML file.
        """
        self.base_model_name = None
        self.task_type = None

        self.math_train_path = None
        self.math_eval_path = None
        self.mbpp_train_path = None
        self.human_eval_path = None
        self.mbpp_r_path = None
        self.checkpoint_dir = None
        self.log_dir = None

        self.training_stage1_math = None
        self.training_stage2_math = None
        self.training_stage1_mbpp = None
        self.training_stage2_mbpp = None

        self.evaluation = None

        self.use_peft = None
        self.peft_method = None
        self.peft_lora_r = None
        self.peft_lora_alpha = None
        self.peft_lora_dropout = None

        self.prompts = None

        self.seed = None

    def load_from_yaml(self, path: str) -> None:
        """
        Loads configuration data from a specified YAML file into the Config instance.

        Args:
            path: The file path to the config.yaml file.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                config_data_dict = yaml.safe_load(f)

            if not isinstance(config_data_dict, dict):
                raise TypeError("YAML file content is not a dictionary.")

            for key, value in config_data_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    print(f"Warning: Unknown configuration key '{key}' found in YAML file.")

        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at: {path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration file: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while loading config: {e}")

    def get_stage_hyperparameters(self, stage: int) -> Dict[str, Any]:
        """
        Retrieves the appropriate hyperparameters for a given stage and task type.

        Args:
            stage: The training stage (1 or 2).

        Returns:
            A dictionary of hyperparameters for the specified stage and task.

        Raises:
            ValueError: If an invalid stage or task_type is requested or
                        if hyperparameters are not defined for the current task_type.
        """
        if self.task_type not in ["math", "code"]:
            raise ValueError(f"Unsupported task_type: {self.task_type}. Must be 'math' or 'code'.")

        if self.task_type == "math":
            if stage == 1:
                if self.training_stage1_math is None:
                    raise ValueError("MATH Stage 1 hyperparameters not loaded.")
                return self.training_stage1_math
            elif stage == 2:
                if self.training_stage2_math is None:
                    raise ValueError("MATH Stage 2 hyperparameters not loaded.")
                return self.training_stage2_math
            else:
                raise ValueError(f"Invalid stage for MATH task: {stage}. Must be 1 or 2.")
        elif self.task_type == "code":
            if stage == 1:
                if self.training_stage1_mbpp is None:
                    raise ValueError("MBPP Stage 1 hyperparameters not loaded.")
                return self.training_stage1_mbpp
            elif stage == 2:
                if self.training_stage2_mbpp is None:
                    raise ValueError("MBPP Stage 2 hyperparameters not loaded.")
                return self.training_stage2_mbpp
            else:
                raise ValueError(f"Invalid stage for MBPP task: {stage}. Must be 1 or 2.")
        # This part should ideally not be reached due to the initial task_type check
        raise ValueError("An unexpected error occurred in get_stage_hyperparameters.")


if __name__ == "__main__":
    # Example usage and validation
    config_path = "config.yaml"  # Assuming config.yaml is in the same directory
    config = Config()
    try:
        config.load_from_yaml(config_path)
        print("Configuration loaded successfully!")

        print(f"Base Model Name: {config.base_model_name}")
        print(f"Task Type: {config.task_type}")
        print(f"MATH Train Path: {config.math_train_path}")
        print(f"Use PEFT: {config.use_peft}")
        print(f"Seed: {config.seed}")
        print(f"MATH First Turn Prompt: {config.prompts['math_first_turn']}")

        # Accessing nested structures
        if config.training_stage1_math:
            print(f"\nStage 1 MATH Learning Rate: {config.training_stage1_math['learning_rate']}")
            print(f"Stage 1 MATH Batch Size: {config.training_stage1_math['batch_size']}")

        # Test get_stage_hyperparameters
        math_stage1_hps = config.get_stage_hyperparameters(1)
        print(f"\nHyperparameters for MATH Stage 1: {math_stage1_hps}")

        # Temporarily change task_type for testing MBPP (if not 'math' in config.yaml)
        original_task_type = config.task_type
        if original_task_type == 'math':
            print("\nSwitching task_type to 'code' for MBPP hyperparameter test...")
            config.task_type = 'code'
            mbpp_stage2_hps = config.get_stage_hyperparameters(2)
            print(f"Hyperparameters for MBPP Stage 2: {mbpp_stage2_hps}")
            config.task_type = original_task_type # Restore original

        # Test error handling
        # config.load_from_yaml("non_existent_config.yaml")
        # config.task_type = "invalid"
        # config.get_stage_hyperparameters(1)

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}")

