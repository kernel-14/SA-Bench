## utils.py
import os
import torch
import json
import logging
from pathlib import Path
from typing import Dict
from model import ConsistencyModel  # Importing ConsistencyModel for checkpoints
import yaml


def load_config(config_path: str) -> Dict:
    """
    Loads and parses the configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict: Parsed configuration as a dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Validate key sections and add defaults if missing
    default_config = {
        "training": {
            "learning_rate": 0.00008,
            "batch_size": 128,
            "epochs": 1,
            "noise_schedule": {
                "sigma_0": 0.002,
                "sigma_t": 80,
                "rho": 7
            }
        },
        "dataset": {
            "name": "CIFAR-10",
            "resolution": 32,
            "preprocessing": {
                "resize": True,
                "normalize": True,
                "scaling": [-1, 1]
            }
        },
        "evaluation": {
            "metrics": ["FID", "KID", "IS"],
            "metrics_samples": 50000
        },
        "logging": {
            "save_checkpoints": True,
            "checkpoint_path": "checkpoints/",
            "log_train_loss": True,
            "log_frequency_steps": 100
        },
        "model": {
            "architecture": "NCSN++",
            "channels": 128,
            "blocks_per_resolution": 3,
            "embedding_type": "positional",
            "attention_resolutions": [],
            "dropout": 0.0
        },
        "shared_settings": {
            "mu": 0.5
        },
        "hardware": {
            "use_gpu": True,
            "gpu_memory": "40GB",
            "distributed_training": False
        }
    }

    # Update default config with user-provided values
    def recursive_update(default: Dict, custom: Dict):
        for key, value in custom.items():
            if isinstance(value, dict) and key in default:
                recursive_update(default[key], value)
            else:
                default[key] = value

    recursive_update(default_config, config)
    return default_config


def save_checkpoint(model: ConsistencyModel, path: str) -> None:
    """
    Saves the model's state dictionary to the specified checkpoint path.

    Args:
        model (ConsistencyModel): Consistency model instance to save.
        path (str): Path where the checkpoint will be written.
    """
    directory = Path(path).parent
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": {
            "architecture": model.__class__.__name__,
            "channels": model.channels,
            "dropout": model.dropout
        }
    }
    
    torch.save(checkpoint, path)
    logging.info(f"Checkpoint saved at: {path}")


def load_checkpoint(path: str) -> ConsistencyModel:
    """
    Loads the model's state dictionary from the specified checkpoint path.

    Args:
        path (str): Path to the checkpoint file.

    Returns:
        ConsistencyModel: Model instance with restored weights.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at path: {path}")
    
    checkpoint = torch.load(path, map_location=torch.device('cpu'))

    # Instantiate a ConsistencyModel with checkpoint metadata
    metadata = checkpoint.get("metadata", {})
    model = ConsistencyModel({
        "model": {
            "channels": metadata.get("channels", 128),
            "dropout": metadata.get("dropout", 0.0)
        }
    })

    model.load_state_dict(checkpoint["model_state_dict"])
    logging.info(f"Checkpoint loaded from: {path}")

    return model


def setup_logging(log_type: str, frequency_steps: int) -> None:
    """
    Sets up the logging system for training and evaluation processes.

    Args:
        log_type (str): Type of logs to capture (e.g., 'training', 'evaluation').
        frequency_steps (int): Frequency of logging steps.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{log_type}_log.txt", mode='w')
        ]
    )
    logging.info(f"Logging initialized for {log_type}. Frequency of logs: {frequency_steps} steps.")


def validate_environment() -> None:
    """
    Validates the execution environment for hardware and software dependencies.
    Ensures GPU availability and memory requirements.
    """
    if torch.cuda.is_available():
        device_properties = torch.cuda.get_device_properties(0)
        logging.info(f"Using GPU: {device_properties.name} with {device_properties.total_memory / (1024 ** 3):.2f} GB of memory")
    else:
        logging.warning("CUDA is not available. Falling back to CPU, which may slow down experimentation.")
    
    # Validate PyTorch version compatibility
    if torch.__version__ < "1.11.0":
        logging.warning("PyTorch version is < 1.11.0. Consider upgrading for compatibility.")
