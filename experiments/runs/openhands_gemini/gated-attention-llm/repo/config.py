
import torch

class ModelConfig:
    """
    Configuration for the Transformer model.
    Based on 1.7B dense model unless specified for MoE.
    """
    model_type: str = "dense" # "dense" or "moe"

    # Model dimensions
    d_model: int = 4096 # The paper implies d_model = n_heads * d_k = 32 * 128 = 4096
    n_layers: int = 28 # For 1.7B model, 28 or 48 layers (Table 2)
    n_heads: int = 32 # Corresponds to 'q' in paper (query heads)
    kv_heads: int = 4 # Corresponds to 'k' in paper (key-value heads) for GQA
    d_k: int = 128 # Head dimension (d_model // n_heads if d_model is 4096, d_k=128 means d_model=4096 for 32 heads)
    d_ff: int = d_model * 4 # Placeholder for FFN inner dimension, typically 4*d_model, but paper mentions reducing FFN width with gating.

    # MoE specific (if model_type is "moe")
    num_experts: int = 128
    top_k_experts: int = 8
    moe_loss_coeff: float = 0.1 # Placeholder for Z-loss/load balancing

    # Attention and Gating
    use_gated_attention: bool = True
    gating_position: str = "G1" # G1 (SDPA output), G2 (Value output), G3 (Key output), G4 (Query output), G5 (Final output)
    gating_granularity: str = "elementwise" # "elementwise" or "headwise"
    head_specific_gating: bool = True # True for head-specific, False for head-shared
    gating_type: str = "multiplicative" # "multiplicative" or "additive"
    gating_activation: str = "sigmoid" # "sigmoid" or "SiLU" or "identity" (for additive without non-linearity)

    # General Transformer settings
    norm_type: str = "RMSNorm" # Often used in modern LLMs
    dropout: float = 0.1 # Placeholder, not explicitly mentioned but common

class TrainingConfig:
    """
    Configuration for the training process.
    """
    # Tokenizer
    model_name_or_path: str = "gpt2" # Using gpt2 as a common example, not specified in paper
    pad_token: str = "[PAD]" # Custom pad token if not available in tokenizer

    # Dataset and Data loading
    dataset_name: str = "custom_3_5T_tokens" # Represents the 3.5T token dataset
    context_length: int = 4096
    batch_size: int = 2048 # Global batch size, can be 1024, 2048, 4096 (Table 2)

    # Optimizer
    optimizer_type: str = "AdamW"
    learning_rate: float = 4.5e-3 # Max LR, can be 2e-3, 4e-3, 4.5e-3, 5.3e-3, 8e-3
    lr_warmup_steps: int = 1000
    lr_decay_strategy: str = "cosine"
    weight_decay: float = 0.01 # Placeholder, common for AdamW

    # Training steps
    total_train_steps: int = 100000 # 100k for MoE models on 400B tokens, longer for 3.5T
    eval_interval_steps: int = 1000
    save_interval_steps: int = 5000

    # Mixed precision training
    use_bf16: bool = True

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class EvaluationConfig:
    """
    Configuration for evaluation metrics and benchmarks.
    """
    metrics: list = ["PPL", "Hellaswag", "MMLU", "GSM8k", "HumanEval", "C-eval", "CMMLU", "RULER"]
    # Other evaluation parameters like few-shot settings, etc.



class MoEModelConfig(ModelConfig):
    model_type: str = "moe"
    n_layers: int = 28 # Assuming MoE also uses 28 layers as a base
    num_experts: int = 128
    top_k_experts: int = 8
    moe_loss_coeff: float = 0.01 # Adjusted Z-loss coeff, paper mentions it exists but not specific value. Default to a small value.

class DenseModelConfig(ModelConfig):
    model_type: str = "dense"
    n_layers: int = 28 # Default for 1.7B model
    
class Dense48LayerModelConfig(ModelConfig):
    model_type: str = "dense"
    n_layers: int = 48

class MoETrainingConfig(TrainingConfig):
    learning_rate: float = 2e-3
    batch_size: int = 1024
    total_train_steps: int = 100000 # 100k steps on 400B tokens, implies fixed step count

class DenseTrainingConfig400B(TrainingConfig):
    learning_rate: float = 4e-3
    batch_size: int = 1024

class DenseTrainingConfig3_5T(TrainingConfig):
    learning_rate: float = 4.5e-3
    batch_size: int = 2048
    total_train_steps: int = 350000 # Rough estimation for 3.5T tokens with similar bsz/steps scaling

class Dense48LayerTrainingConfig400B(TrainingConfig):
    n_layers: int = 48 # Explicitly setting for 48 layer
    learning_rate: float = 4e-3
    batch_size: int = 1024

class Dense48LayerTrainingConfigHighLR(TrainingConfig):
    n_layers: int = 48 # Explicitly setting for 48 layer
    learning_rate: float = 8e-3 # High LR for stability test
    batch_size: int = 1024

class Dense48LayerTrainingConfig1T(TrainingConfig):
    n_layers: int = 48
    learning_rate: float = 5.3e-3
    batch_size: int = 4096
    total_train_steps: int = 250000 # Rough estimation for 1T tokens

class Dense48LayerTrainingConfig1THighLR(TrainingConfig):
    n_layers: int = 48
    learning_rate: float = 8e-3
    batch_size: int = 4096
    total_train_steps: int = 250000 # Rough estimation for 1T tokens


# Default configuration to use
current_model_config = DenseModelConfig()
current_training_config = DenseTrainingConfig400B()
current_evaluation_config = EvaluationConfig()

# Example: To use MoE configuration
# current_model_config = MoEModelConfig()
# current_training_config = MoETrainingConfig()

# Example: To use 1.7B dense model trained on 3.5T tokens
# current_model_config = DenseModelConfig()
# current_training_config = DenseTrainingConfig3_5T()

