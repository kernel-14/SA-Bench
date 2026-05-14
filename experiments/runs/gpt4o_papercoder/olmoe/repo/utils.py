## utils.py
import random
import torch
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Any


def set_random_seed(seed: int = 42) -> None:
    """
    Set the random seed to ensure reproducibility of experiments across all components.
    
    Args:
        seed (int): The global seed value used for deterministic operation.
    """
    # Set Python and NumPy random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    
    # Ensure deterministic behavior for CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> Optimizer:
    """
    Generate an optimizer for training the model based on configuration.
    
    Args:
        model (torch.nn.Module): The PyTorch model whose parameters are being optimized.
        config (Dict[str, Any]): Configuration dictionary containing optimizer settings.
    
    Returns:
        Optimizer: The optimizer initialized with model parameters and config settings.
    """
    # Extract optimizer settings from configuration
    learning_rate = config["training"]["pretraining"]["learning_rate"]
    weight_decay = config["training"]["pretraining"]["weight_decay"]
    epsilon = config["training"]["pretraining"]["epsilon"]
    beta1 = config["training"]["pretraining"]["optimizer_betas"]["beta1"]
    beta2 = config["training"]["pretraining"]["optimizer_betas"]["beta2"]
    
    # Separate parameters into groups for potential custom optimizations
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    # Initialize the optimizer (AdamW as per configuration)
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=epsilon
    )

    return optimizer


def generate_scheduler(optimizer: Optimizer, config: Dict[str, Any], total_training_steps: int) -> _LRScheduler:
    """
    Generate a learning rate scheduler to manage the learning rate over training epochs.
    
    Args:
        optimizer (Optimizer): Optimizer instance for the model.
        config (Dict[str, Any]): Configuration dictionary containing scheduler settings.
        total_training_steps (int): Total number of training steps planned.

    Returns:
        _LRScheduler: Learning rate scheduler configured as per training setup.
    """
    warmup_steps = config["training"]["pretraining"]["warmup_steps"]
    peak_lr = config["training"]["pretraining"]["peak_lr"]
    min_lr = config["training"]["pretraining"]["min_lr"]
    
    # Define a custom lambda function for linear warmup and decay
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            float((total_training_steps - current_step)) / float(max(1, total_training_steps - warmup_steps)), 
            min_lr / peak_lr
        )
    
    # Create the LambdaLR scheduler using the lambda function
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def log_metrics(metrics: Dict[str, Any], step: int, log_to_wandb: bool = True, project_name: str = "OLMoE") -> None:
    """
    Log evaluation metrics to Wandb or locally.

    Args:
        metrics (Dict[str, Any]): Dictionary containing evaluation metrics and their values.
        step (int): Step or epoch number associated with the logging.
        log_to_wandb (bool): Flag to enable logging to Wandb.
        project_name (str): Name of the Wandb project where results will be logged.
    """
    try:
        if log_to_wandb:
            import wandb
            
            # Initialize or use the existing project
            wandb.init(project=project_name, reinit=True)
            
            # Log metrics
            wandb.log({**metrics, "step": step})
        else:
            # Log metrics locally (e.g., save to a JSON or CSV file)
            print(f"[Local Log] Step {step}: {metrics}")
    except ModuleNotFoundError:
        print("Wandb module not found! Fallback to local logging.")
        print(f"[Local Log] Step {step}: {metrics}")


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate the structure and contents of a configuration dictionary.

    Args:
        config (Dict[str, Any]): Configuration dictionary loaded from 'config.yaml'.

    Raises:
        ValueError: If required keys or values are missing or invalid in the configuration.
    """
    required_top_level_keys = ["training", "model", "data", "evaluation"]
    
    # Check presence of top-level keys
    for key in required_top_level_keys:
        if key not in config:
            raise ValueError(f"Missing required top-level key: {key}")
            
    # Validate training keys
    training_keys = ["pretraining", "adaptation"]
    for key in training_keys:
        if key not in config["training"]:
            raise ValueError(f"Missing 'training:{key}' configuration")
    
    # Validate pretraining learning rate-related settings
    pretraining_keys = ["learning_rate", "warmup_steps", "peak_lr", "min_lr", "global_max_grad_norm"]
    for key in pretraining_keys:
        if key not in config["training"]["pretraining"]:
            raise ValueError(f"Missing 'training:pretraining:{key}' configuration")
        
    # Validate adaptation settings
    adaptation_keys = ["sft", "dpo"]
    for key in adaptation_keys:
        if key not in config["training"]["adaptation"]:
            raise ValueError(f"Missing 'training:adaptation:{key}' configuration")
