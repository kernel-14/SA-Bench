import argparse
import os
from typing import List, Optional
import yaml

from peft import LoraConfig # From peft library

# Define the path to the config file
CONFIG_PATH = "config.yaml"

class Config:
    """
    Configuration class for the LoRA-SB experiment.
    Loads settings from config.yaml and allows overriding with command-line arguments.
    """

    def __init__(self):
        """
        Initializes configuration attributes with default values and loads settings
        from the config.yaml file.
        """
        # Global experiment settings
        self.model_name: str = "mistralai/Mistral-7B-v0.1"
        self.task_name: str = "MetaMathQA"
        self.output_dir: str = os.path.abspath("./outputs")
        self.random_seed: int = 42
        self.num_runs: int = 3

        # LoRA-SB specific configuration
        self.rank: int = 64
        # lora_alpha is primarily for compatibility with peft.LoraConfig's alpha/r scaling.
        # For LoRA-SB, the actual scaling 's' is fixed to 1.0.
        # We set lora_alpha = rank here to ensure alpha/r = 1 for any PEFT functions
        # that might implicitly rely on it, even though LoRASBLinear uses self.lora_sb_s directly.
        self.lora_alpha: int = 64 # Default to self.rank, will be updated to match actual rank
        self.lora_sb_dropout: float = 0.0 # Dropout specifically for LoRA-SB (B R A) layers
        self.init_sample_ratio: float = 0.001
        self.target_modules: List[str] = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj" # Common target modules for Llama/Mistral-like models
        ]
        self.lora_sb_s: float = 1.0 # LoRA-SB's fixed scaling factor 's', as per paper Section 2.6

        # Training specific configuration
        self.optimizer: str = "AdamW"
        self.learning_rate: float = 1.0e-4
        self.batch_size: int = 1
        self.max_seq_len: int = 512
        self.grad_acc_steps: int = 32
        self.epochs: int = 1
        self.dropout: float = 0.0 # General dropout for the base model (e.g., from training section)
        self.lr_scheduler_type: str = "cosine"
        self.warmup_ratio: float = 0.02

        # Evaluation specific configuration
        self.evaluation_batch_size: int = 8

        self._load_from_yaml(CONFIG_PATH)

    def _load_from_yaml(self, path: str) -> None:
        """
        Loads configuration settings from a YAML file.
        """
        if not os.path.exists(path):
            print(f"Warning: Config file not found at {path}. Using default settings.")
            return

        with open(path, 'r') as f:
            yaml_config = yaml.safe_load(f)

        # Experiment settings
        self.model_name = yaml_config.get("experiment", {}).get("model_name", self.model_name)
        self.task_name = yaml_config.get("experiment", {}).get("task_name", self.task_name)
        # Use os.path.abspath to ensure output_dir is an absolute path
        self.output_dir = os.path.abspath(yaml_config.get("experiment", {}).get("output_dir", self.output_dir))
        self.random_seed = yaml_config.get("experiment", {}).get("random_seed", self.random_seed)
        self.num_runs = yaml_config.get("experiment", {}).get("num_runs", self.num_runs)

        # LoRA-SB configuration
        self.rank = yaml_config.get("lora_sb", {}).get("rank", self.rank)
        self.lora_alpha = self.rank # Dynamically set lora_alpha to match rank for PEFT compatibility
        self.lora_sb_dropout = yaml_config.get("lora_sb", {}).get("lora_dropout", self.lora_sb_dropout)
        self.init_sample_ratio = yaml_config.get("lora_sb", {}).get("init_sample_ratio", self.init_sample_ratio)
        self.target_modules = yaml_config.get("lora_sb", {}).get("target_modules", self.target_modules)
        self.lora_sb_s = yaml_config.get("lora_sb", {}).get("s", self.lora_sb_s)

        # Training configuration
        self.optimizer = yaml_config.get("training", {}).get("optimizer", self.optimizer)
        self.learning_rate = yaml_config.get("training", {}).get("learning_rate", self.learning_rate)
        self.batch_size = yaml_config.get("training", {}).get("batch_size", self.batch_size)
        self.max_seq_len = yaml_config.get("training", {}).get("max_seq_len", self.max_seq_len)
        self.grad_acc_steps = yaml_config.get("training", {}).get("grad_acc_steps", self.grad_acc_steps)
        self.epochs = yaml_config.get("training", {}).get("epochs", self.epochs)
        self.dropout = yaml_config.get("training", {}).get("dropout", self.dropout)
        self.lr_scheduler_type = yaml_config.get("training", {}).get("lr_scheduler_type", self.lr_scheduler_type)
        self.warmup_ratio = yaml_config.get("training", {}).get("warmup_ratio", self.warmup_ratio)

        # Evaluation configuration
        self.evaluation_batch_size = yaml_config.get("evaluation", {}).get("batch_size", self.evaluation_batch_size)

    def load_from_args(self, args: argparse.Namespace) -> None:
        """
        Overrides configuration attributes with values provided via command-line arguments.
        Only non-None arguments will override existing values.
        """
        for arg_name, arg_value in vars(args).items():
            # Check if the attribute exists in Config and if the arg_value is not None.
            # We explicitly prevent 'lora_alpha' from being directly overridden as it's
            # intended to track 'rank'.
            if hasattr(self, arg_name) and arg_value is not None and arg_name != "lora_alpha":
                setattr(self, arg_name, arg_value)

        # After potentially updating 'rank' from args, ensure 'lora_alpha' matches 'rank'.
        self.lora_alpha = self.rank


class LoRASBConfig(LoraConfig):
    """
    Configuration for LoRA-SB adaptation, inheriting from peft.LoraConfig.
    Includes the fixed scaling factor 's' unique to LoRA-SB.
    """
    def __init__(
        self,
        r: int,
        target_modules: List[str],
        lora_dropout: float,
        s: float = 1.0,
        **kwargs,
    ):
        """
        Initializes LoRA-SB specific parameters.

        Args:
            r (`int`):
                LoRA attention dimension (rank).
            target_modules (`List[str]`):
                The names of the modules in the base model to apply LoRA-SB to.
            lora_dropout (`float`):
                The dropout probability for the LoRA-SB (B R A) layers.
            s (`float`, defaults to 1.0):
                The scaling factor for the LoRA-SB update (W = W_0 + s B R A).
                This is fixed to 1.0 for LoRA-SB as per the paper, Section 2.6.
            kwargs:
                Any additional keyword arguments to pass to the base LoraConfig.
        """
        # For LoRA-SB, the paper states s=1 for optimal gradient approximation.
        # The base LoraConfig expects `lora_alpha`. Standard LoRA scaling is `lora_alpha / r`.
        # By setting `lora_alpha = r`, the effective `alpha/r` would be 1 if used by PEFT
        # internals, which aligns with LoRA-SB's fixed `s=1`. This maintains compatibility.
        super().__init__(
            r=r,
            lora_alpha=r,  # Set alpha = r for consistency with peft (alpha/r effectively equals 1)
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none", # As per common LoRA practice, bias is generally not adapted
            **kwargs,
        )
        self.s = s

