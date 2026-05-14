import math
from dataclasses import dataclass, field
from typing import Tuple

# Importing MMField from dataclasses_json.config is specific to that library's
# serialization mechanism, often used for more advanced field handling.
# For basic YAML/JSON serialization with 'dataclasses_json',
# a simple `@dataclass_json` decorator (or `config` with `mm_field=None`) is often enough.
# Let's use @dataclass_json directly for simplicity and standard usage.
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Config:
    """
    Configuration dataclass for the NGPT model and training.

    This class holds all hyperparameters, training configurations, dataset paths,
    and initialization settings, ensuring reproducibility and ease of management.
    Values are sourced directly from the config.yaml and paper specifications.
    """

    # --- Model Configuration ---
    model_type: str = "ngpt"  # Options: "ngpt", "gpt" (for baseline comparison)
    n_layers: int = 24  # Number of Transformer layers. 24 for 0.5B, 36 for 1.0B models.
    d_model: int = 1024  # Dimensionality of the model. 1024 for 0.5B, 1280 for 1.0B models.
    n_heads: int = 16  # Number of attention heads. 16 for 0.5B, 20 for 1.0B models.
    vocab_size: int = 32000  # LLaMA-2 tokenizer vocabulary size.
    rope_base: float = 10000.0  # Base for Rotary Position Embeddings.
    block_size: int = 4096  # Context length in tokens. Common values: 1k, 4k, 8k.
    init_std_dev: float = 0.03125  # Initial standard deviation for matrix parameters.
    # For nGPT: 1 / sqrt(d_model) = 1 / sqrt(1024) = 0.03125.
    # For GPT baseline: 0.02.

    # Derived parameters (calculated in __post_init__)
    d_k: int = field(init=False)  # Dimensionality of query/key vectors per head
    d_mlp: int = field(init=False)  # Dimensionality of MLP intermediate layer

    # --- Optimizer Configuration ---
    optimizer_name: str = "adam"  # For nGPT: "adam". For GPT baseline: "adamw".
    learning_rate: float = 0.0005  # Initial learning rate. Tuned per model/context_length.
    weight_decay: float = 0.0  # For nGPT: 0.0. For GPT baseline: 0.1.
    learning_rate_schedule: str = "cosine_annealing"  # Learning rate schedule.
    warmup_steps: int = 0  # For nGPT: 0. For GPT baseline: 2000.
    final_learning_rate: float = 0.0  # Learning rate at the end of cosine annealing.

    # --- Training Configuration ---
    global_batch_size: int = 512  # Total batch size across all GPUs.
    gradient_accumulation_steps: int = 1  # Steps to accumulate gradients.
    max_train_steps: int = 100000  # Total training steps.
    eval_interval: int = 1000  # Evaluate validation loss every N steps.
    log_interval: int = 10  # Log training progress every N steps.
    precision: str = "bfloat16"  # Training precision ("float32", "bfloat16").

    # --- NGPT-Specific Scaling Parameters (for ScaledLearnableParameter) ---
    # These define s_init and s_scale for ScaledLearnableParameter instances.
    # Based on Section 2.6 and 2.5 of the paper.

    # Alpha (eigen learning rates for Attention and MLP)
    s_alpha_init: float = 0.05
    s_alpha_scale_factor: float = 0.03125  # 1 / sqrt(d_model)

    # s_qk (Q/K scaling in attention)
    s_qk_init: float = 1.0
    s_qk_scale_factor: float = 0.03125  # 1 / sqrt(d_model)

    # s_u (MLP intermediate scaling, u-branch)
    s_u_init: float = 1.0
    s_u_scale_factor: float = 1.0

    # s_nu (MLP intermediate scaling, nu-branch)
    s_nu_init: float = 1.0
    s_nu_scale_factor: float = 1.0

    # s_z (Logits scaling)
    s_z_init: float = 1.0
    s_z_scale_factor: float = 0.03125  # 1 / sqrt(d_model)

    # --- Data Configuration ---
    dataset_name: str = "openwebtext"  # Dataset to use.
    tokenizer_name: str = "NousResearch/Llama-2-7b-hf"  # HuggingFace tokenizer ID.

    # --- System Configuration ---
    num_gpus: int = 64  # Number of GPUs for distributed training.
    master_port: int = 29500  # Port for distributed training communication.
    seed: int = 42  # Random seed for reproducibility.

    def __post_init__(self):
        """
        Post-initialization to derive dependent configuration parameters.
        """
        # Calculate d_k: Dimensionality of query and key vectors for each attention head.
        # Paper Section 2.3.1: "d_k is typically set to d_model / n_heads."
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        self.d_k = self.d_model // self.n_heads

        # Calculate d_mlp: Dimensionality of the intermediate layer in the MLP block.
        # Paper Section 2.1, Table 2: "d_MLP = 4d_model"
        self.d_mlp = 4 * self.d_model

        # Conditional adjustments based on model_type (for baseline GPT comparison)
        if self.model_type == "gpt":
            # Baseline GPT optimizer settings (Table 3)
            self.optimizer_name = "adamw"
            self.weight_decay = 0.1
            self.warmup_steps = 2000
            # Baseline GPT initialization (Section A.6)
            self.init_std_dev = 0.02
        elif self.model_type == "ngpt":
            # nGPT optimizer settings (Table 3)
            self.optimizer_name = "adam"
            self.weight_decay = 0.0
            self.warmup_steps = 0
            # nGPT initialization (Section A.6)
            # The config.yaml has a pre-calculated value for 1/sqrt(d_model),
            # but we can re-calculate it to ensure consistency with d_model.
            self.init_std_dev = 1.0 / math.sqrt(self.d_model)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}. Must be 'ngpt' or 'gpt'.")

        # Re-calculate scale factors based on actual d_model for ngpt_specific_config
        # This ensures the factors are always consistent with the chosen d_model
        # even if d_model changes.
        if self.model_type == "ngpt":
            self.s_alpha_scale_factor = 1.0 / math.sqrt(self.d_model)
            self.s_qk_scale_factor = 1.0 / math.sqrt(self.d_model)
            self.s_z_scale_factor = 1.0 / math.sqrt(self.d_model)

