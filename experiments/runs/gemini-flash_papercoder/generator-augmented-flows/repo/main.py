# main.py

import os
import random
import numpy as np
import torch
import torch.optim as optim
import yaml # For loading initial config.yaml

# Required third-party optimizer if using Lion
from lion_pytorch import Lion # As per Required packages

# Project-specific imports
from config import Config
from data_utils import DataUtils
from utils import Utils # Not directly used in main, but good to have if needed for specific utils in main
from model import ConsistencyModel, EMA, SongUNet # Import SongUNet for parameter passing clarity
from trainer import Trainer
from evaluator import Evaluator

# Optional: For distributed training
try:
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
except ImportError:
    Accelerator = None
    DistributedDataParallelKwargs = None
    print("Warning: 'accelerate' library not found. Distributed training will be disabled.")


def main():
    """
    Main function to orchestrate the training and evaluation pipeline for
    Generator-Augmented Flows consistency models.
    """
    # 1. Load Configuration
    config_path = "config.yaml"
    config = Config(config_path)

    # 2. Setup Device and Accelerator (if distributed)
    accelerator = None
    if config.DISTRIBUTED:
        if Accelerator is None:
            raise ImportError(
                "Cannot use distributed training without 'accelerate' library. "
                "Please install it using 'pip install accelerate'."
            )
        # find_unused_parameters=True is often necessary for consistency models
        # due to stop_gradient operations creating unused parameters in DDP.
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        config.DEVICE = accelerator.device # Use accelerator's device
        print(f"Distributed training initialized on device: {config.DEVICE}")
    else:
        config.DEVICE = torch.device(config.DEVICE_STR if torch.cuda.is_available() else "cpu")
        print(f"Running on single device: {config.DEVICE}")
    
    # Update config.DEVICE_STR to reflect actual device being used
    config.DEVICE_STR = str(config.DEVICE)

    # 3. Set Random Seeds for reproducibility
    seed_value = config.SEED
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    print(f"Random seed set to {seed_value}")

    # 4. Data Preparation: Load DataLoaders and Calculate Data Variance
    print(f"Loading dataset: {config.DATASET_NAME} at resolution {config.RESOLUTION}")
    train_dataloader = DataUtils.get_dataloader(
        dataset_name=config.DATASET_NAME,
        resolution=config.RESOLUTION,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True
    )
    test_dataloader = DataUtils.get_dataloader(
        dataset_name=config.DATASET_NAME,
        resolution=config.RESOLUTION,
        batch_size=config.BATCH_SIZE, # Use the same batch size for test
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False # Do not shuffle test data
    )

    # Calculate empirical variance of the training data
    # Only calculate on the main process to avoid redundant computation and potential data issues
    if accelerator is None or accelerator.is_main_process:
        config.SIGMA_D_SQ = DataUtils.calculate_sigma_d_squared(train_dataloader, config.DEVICE)
    if accelerator:
        # Broadcast sigma_d_sq from main process to all other processes
        # Create a tensor on the main process with the value, then broadcast
        if accelerator.is_main_process:
            sigma_d_sq_tensor = torch.tensor(config.SIGMA_D_SQ, device=config.DEVICE)
        else:
            sigma_d_sq_tensor = torch.tensor(0.0, device=config.DEVICE) # Placeholder
        
        sigma_d_sq_tensor = accelerator.broadcast(sigma_d_sq_tensor)
        config.SIGMA_D_SQ = sigma_d_sq_tensor.item()
        accelerator.wait_for_everyone() # Ensure all processes have the correct value

    # Save the final resolved config (after dataset overrides and sigma_d_sq calculation)
    if accelerator is None or accelerator.is_main_process:
        config.save_config(os.path.join(config.CHECKPOINT_DIR, "final_config.yaml"))
    if accelerator:
        accelerator.wait_for_everyone()


    # 5. Initialize Consistency Model (SongUNet backbone)
    print("Initializing Consistency Model...")
    unet_params = {
        "img_resolution": config.RESOLUTION,
        "in_channels": config.IMG_CHANNELS,
        "out_channels": config.IMG_CHANNELS, # Output is typically same channels as input
        "model_channels": config.MODEL_CHANNELS,
        "num_blocks": config.NUM_BLOCKS,
        "channel_mult": config.CHANNEL_MULT,
        "attn_resolutions": config.ATTN_RESOLUTIONS,
        "dropout_rate": config.DROPOUT_RATE,
        "embedding_type": config.EMBEDDING_TYPE
    }

    model = ConsistencyModel(
        unet_params=unet_params,
        sigma_0=config.SIGMA_0,
        sigma_d_sq=config.SIGMA_D_SQ,
        data_dim=config.DATA_DIM,
        device=config.DEVICE # Pass device, model will move to it internally
    )
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters.")
    # For distributed setup, model is moved to device by accelerator later.
    # For non-distributed, it's moved in ConsistencyModel's init or Trainer's init.


    # 6. Initialize EMA Model
    ema_model = EMA(model=model, decay=config.EMA_DECAY)
    print("EMA model initialized.")


    # 7. Initialize Optimizer
    if config.OPTIMIZER.lower() == "lion":
        optimizer = Lion(model.parameters(), lr=config.LEARNING_RATE)
    elif config.OPTIMIZER.lower() == "adam": # Add Adam as a fallback/alternative
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    else:
        raise ValueError(f"Unsupported optimizer: {config.OPTIMIZER}")
    print(f"Optimizer '{config.OPTIMIZER}' initialized with learning rate {config.LEARNING_RATE}.")


    # 8. Initialize Trainer and Evaluator
    evaluator = Evaluator(
        model=model, # Pass the base model, it's not used for generation in Evaluator directly
        ema_model=ema_model,
        test_dataloader=test_dataloader,
        config=config,
        accelerator=accelerator
    )
    
    trainer = Trainer(
        model=model,
        ema_model=ema_model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        sigma_d_sq=config.SIGMA_D_SQ,
        config=config,
        evaluator=evaluator # Pass the evaluator instance to the trainer
    )
    print("Trainer and Evaluator initialized.")


    # 9. Start Training
    print("Starting training...")
    trainer.train()
    print("Training finished.")


    # 10. Final Actions (Post-Training) - Performed only by main process
    if accelerator is None or accelerator.is_main_process:
        print("\n--- Final Evaluation ---")
        final_metrics = evaluator.evaluate(config.TRAINING_STEPS, use_ema_for_generation=True)
        for metric_name, (mean_val, std_val) in final_metrics.items():
            print(f"  Final {metric_name}: {mean_val:.4f} +/- {std_val:.4f}")
        print("--------------------------")

        print("Saving final models...")
        final_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "model_final_after_eval.pth")
        
        # Ensure models are unwrapped if DDP was used before saving
        model_to_save = accelerator.unwrap_model(model) if accelerator else model
        ema_model_to_save = accelerator.unwrap_model(ema_model.get_model()) if accelerator else ema_model.get_model()

        torch.save({
            'global_step': config.TRAINING_STEPS,
            'model_state_dict': model_to_save.state_dict(),
            'ema_model_state_dict': ema_model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config.save_config(None) # Pass None to just get the dict for saving
        }, final_checkpoint_path)
        print(f"Final model states saved to {final_checkpoint_path}")


if __name__ == "__main__":
    main()

