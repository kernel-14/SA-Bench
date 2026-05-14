import argparse
import os
import random
import yaml
import time
from datetime import timedelta # Used for dist.init_process_group timeout
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.backends.cudnn

# Import custom modules
from config import Config
from data import DataModule
from model import NGPTModel
from trainer import NGPTTrainer
from evaluation import NGPTEvaluator


def load_and_process_config(args: argparse.Namespace, rank: int) -> Config:
    """
    Loads configuration from a YAML file, applies dynamic calculations based on
    model_type and other parameters, and overrides with command-line arguments.

    Args:
        args: Command-line arguments.
        rank: The rank of the current process (0 for non-distributed or main process).

    Returns:
        An instance of the Config dataclass with all parameters set.
    """
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file not found at {args.config_path}")

    with open(args.config_path, 'r') as f:
        yaml_data = yaml.safe_load(f)

    # Instantiate Config from the loaded dictionary
    config = Config.from_dict(yaml_data)

    # The Config's __post_init__ method handles dynamic calculations and
    # model_type-specific adjustments upon initialization.
    # If model_type is overridden by CLI, re-call __post_init__ to apply changes.
    original_model_type = config.model_config.model_type
    if args.model_type is not None and original_model_type != args.model_type:
        config.model_config.model_type = args.model_type
        # Re-run post_init to update dependent fields like init_std_dev, optimizer settings
        config.__post_init__() 

    # Apply command-line overrides for specific parameters
    if args.learning_rate is not None:
        config.optimizer_config.learning_rate = args.learning_rate
    if args.block_size is not None:
        config.model_config.block_size = args.block_size
        # Block size change might affect derived parameters in Config, e.g., if
        # other scaling factors or parameters were tied to it. In this case,
        # it mostly affects data processing.
    if args.global_batch_size is not None:
        config.training_config.global_batch_size = args.global_batch_size
    if args.max_train_steps is not None:
        config.training_config.max_train_steps = args.max_train_steps
    
    # Final check of dynamic parameters in case other CLI args changed base values
    config.__post_init__()

    if rank == 0:
        print("--- Configuration Loaded and Processed ---")
        # Use to_json_string for a pretty-printed representation of the dataclass_json object
        print(config.to_json_string(indent=2)) 
        print("------------------------------------------")

    return config


def main():
    parser = argparse.ArgumentParser(description="Train NGPT model.")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to the YAML configuration file.")
    parser.add_argument("--run_id", type=str, default=f"ngpt_run_{int(time.time())}",
                        help="Unique identifier for the current training run.")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank for distributed training (set by torchrun or launch).")
    
    # Optional command-line overrides for common configuration parameters
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Override learning rate from config.")
    parser.add_argument("--block_size", type=int, default=None,
                        help="Override context length (block_size) from config.")
    parser.add_argument("--global_batch_size", type=int, default=None,
                        help="Override global batch size from config.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Override max training steps from config.")
    parser.add_argument("--model_type", type=str, default=None,
                        choices=["ngpt", "gpt"],
                        help="Override model type (ngpt/gpt) from config. This will re-trigger associated parameter adjustments.")

    args = parser.parse_args()

    # Determine distributed environment context
    is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0")) if is_distributed else 0
    world_size = int(os.environ.get("WORLD_SIZE", "1")) if is_distributed else 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if is_distributed else 0

    # If running in a distributed environment, set the CUDA device for the current process.
    # The `NGPTTrainer` will handle `dist.init_process_group`.
    if is_distributed:
        torch.cuda.set_device(local_rank)
        # Device will be determined internally by trainer.

    # Load and process the full configuration based on CLI args and YAML file
    config = load_and_process_config(args, rank)

    # Set random seeds for reproducibility (different seed per rank to avoid identical data access patterns)
    seed = config.system_config.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Enable cuDNN benchmark for performance if input sizes don't change
    torch.backends.cudnn.benchmark = True

    if rank == 0:
        print(f"Starting training run with ID: {args.run_id}")

    # Instantiate core components
    data_module = DataModule(config)
    
    # In a distributed setting, `prepare_data` might involve downloading or
    # initial processing. It's often run once by rank 0 and others wait,
    # or it's idempotent and run by all. HuggingFace `datasets` handles caching.
    data_module.prepare_data() 
    if is_distributed:
        dist.barrier() # Ensure all data is prepared before tokenization
    
    # `setup` involves tokenization and chunking. `datasets` parallelizes and caches.
    data_module.setup()
    if is_distributed:
        dist.barrier() # Ensure all datasets are set up before model construction/training

    model = NGPTModel(config)
    
    # Instantiate evaluator and trainer
    evaluator = NGPTEvaluator(config, model, data_module)
    trainer = NGPTTrainer(config, model, data_module, evaluator)

    # Start the training process
    try:
        trainer.train()
    except KeyboardInterrupt:
        if rank == 0:
            print(f"Training run {args.run_id} interrupted by user.")
    finally:
        if rank == 0:
            print(f"\n--- Training Finished for run {args.run_id}. Performing Final Evaluation ---")
            final_metrics = evaluator.evaluate_downstream_tasks()
            print(f"Final Evaluation Results for run {args.run_id}: {final_metrics}")
        
        # Cleanup distributed environment if it was initialized
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
