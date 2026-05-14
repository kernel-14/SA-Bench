
import torch

class Config:
    # General
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Model
    model_name: str = "mistralai/Mistral-7B-v0.1" # Example default, will be overridden by task-specific configs
    
    # LoRA-SB Specific
    rank: int = 32 # Default rank, will be overridden by task-specific configs
    scaling_factor: float = 1.0 # s is set to 1 for LoRA-SB
    init_data_percentage: float = 0.001 # 0.1% of the dataset
    num_initialization_samples: int = 50 # Heuristic from paper Table 5 for 50k dataset

    # Training Hyperparameters
    optimizer: str = "AdamW"
    learning_rate: float = 1e-4 # Default, will be overridden
    batch_size: int = 1 # Default, will be overridden
    max_seq_len: int = 512 # Default, will be overridden
    gradient_accumulation_steps: int = 32 # Default, will be overridden
    epochs: int = 1 # Default, will be overridden
    dropout: float = 0.0
    lr_scheduler_type: str = "cosine" # Default, will be overridden
    warmup_ratio: float = 0.02

    # Dataset Configuration (examples, actual will be dynamically loaded)
    # MetaMathQA
    metamath_dataset: str = "tau/metamath"
    metamath_train_split: str = "train"
    metamath_eval_datasets: list = ["gsm8k", "math"]

    # COMMONSENSE170K (will need to combine multiple datasets)
    commonsense_datasets: list = [
        "hellaswag", "arc_easy", "piqa", "siqa", "winogrande", "arc_challenge", "openbookqa", "boolq"
    ]
    commonsense_train_split: str = "train" # Or similar, needs verification
    commonsense_eval_split: str = "validation" # Or similar, needs verification

    # GLUE
    glue_task_names: list = ["cola", "rte", "mrpc", "stsb", "qnli", "sst2"]
    glue_train_split: str = "train"
    glue_validation_split: str = "validation"
    glue_test_split: str = "test"

    # Target modules for LoRA application
    # Paper states: key, value, query, attention output, and all fully connected weight matrices for LLMs
    # For RoBERTa: only self-attention layers
    target_modules_llm: list = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    target_modules_roberta: list = ["query", "key", "value", "output.dense"] # Assuming these map to attention layers

    # Task-specific configurations (will be dynamically set or selected)
    TASK_CONFIGS = {
        "arithmetic": {
            "model_name": "mistralai/Mistral-7B-v0.1", # Also "Gemma-2 9B"
            "ranks": [32, 64, 96],
            "learning_rate": 1e-4,
            "batch_size": 1,
            "max_seq_len": 512,
            "gradient_accumulation_steps": 32,
            "epochs": 1,
            "dropout": 0.0,
            "lr_scheduler_type": "cosine",
            "target_modules": target_modules_llm,
        },
        "commonsense_reasoning": {
            "model_name": "meta-llama/Llama-3-8B", # Paper says Llama-3.2 3B, assuming 8B as it's common
            "ranks": [32, 64, 96],
            "learning_rate": 2e-3,
            "batch_size": 6,
            "max_seq_len": 256,
            "gradient_accumulation_steps": 24,
            "epochs": 2,
            "dropout": 0.05,
            "lr_scheduler_type": "linear",
            "target_modules": target_modules_llm,
        },
        "nlu": { # RoBERTa-large on GLUE
            "model_name": "roberta-large",
            "ranks": [8, 16, 24],
            "learning_rate": 1e-3,
            "batch_size": 30, # Max batch size on table 9
            "max_seq_len": 128, # Max seq len on table 9
            "gradient_accumulation_steps": 1, # Not specified, assuming 1
            "epochs": 30,
            "dropout": 0.0, # Not specified for RoBERTa, assuming 0 as per other LLMs
            "lr_scheduler_type": "linear",
            "warmup_ratio": 0.06,
            "target_modules": target_modules_roberta,
            "glue_specific_configs": {
                "cola": {"max_seq_len": 30, "epochs": 30},
                "rte": {"max_seq_len": 30, "epochs": 30},
                "mrpc": {"max_seq_len": 128, "epochs": 30},
                "stsb": {"max_seq_len": 128, "epochs": 30},
                "qnli": {"max_seq_len": 128, "epochs": 30},
                "sst2": {"max_seq_len": 128, "epochs": 30},
            }
        }
    }
