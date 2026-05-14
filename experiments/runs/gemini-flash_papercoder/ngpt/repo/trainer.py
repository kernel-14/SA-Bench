import os
import time
from datetime import timedelta
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

# Local imports
from config import Config
from data import DataModule
from evaluation import NGPTEvaluator
from model import NGPTModel


class NGPTTrainer:
    """
    Orchestrates the training process for NGPTModel. Manages the optimizer,
    learning rate scheduler, mixed-precision training, distributed training setup,
    and the custom post-optimizer normalization step.
    """

    def __init__(self, config: Config, model: NGPTModel, data_module: DataModule, evaluator: NGPTEvaluator):
        """
        Initializes the NGPTTrainer.

        Args:
            config: An instance of the Config dataclass.
            model: The NGPTModel instance to be trained.
            data_module: The DataModule instance for data loading.
            evaluator: The NGPTEvaluator instance for model evaluation.
        """
        self.config = config
        self.model = model
        self.data_module = data_module
        self.evaluator = evaluator

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.scaler: Optional[GradScaler] = None

        # Determine device and set up distributed training if applicable
        self.rank: int = 0
        self.world_size: int = 1
        self.device: torch.device = torch.device("cpu") # Default to CPU

        if self.config.system_config.num_gpus > 1:
            self.setup_distributed()
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # Move model to device (before DDP wrap for DDP to initialize properly)
        self.model.to(self.device)

        # Initialize GradScaler for mixed precision
        if self.config.training_config.precision == "bfloat16":
            self.scaler = GradScaler()
            print(f"Using mixed precision training with {self.config.training_config.precision}.")

        # Wrap model with DDP if distributed
        if self.config.system_config.num_gpus > 1:
            # find_unused_parameters=False is an optimization.
            # It assumes all parameters will receive gradients. If some paths
            # are conditionally taken resulting in unused parameters, this must be True.
            # For Transformer models, usually all parameters are used.
            self.model = DDP(self.model, device_ids=[self.device], find_unused_parameters=False)
            if self.rank == 0:
                print(f"Model wrapped with DistributedDataParallel (DDP) on rank {self.rank}.")

        self.init_optimizer_scheduler()
        if self.rank == 0:
            print("Trainer initialized.")

    def setup_distributed(self) -> None:
        """
        Configures the environment for PyTorch Distributed Data Parallel (DDP) training.
        This method retrieves environment variables set by the launcher (e.g., torchrun).
        """
        if not dist.is_initialized():
            self.rank = int(os.environ.get("RANK", "0"))
            self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
            master_addr = os.environ.get("MASTER_ADDR", "localhost")
            master_port = os.environ.get("MASTER_PORT", str(self.config.system_config.master_port))

            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port

            print(f"Rank {self.rank}/{self.world_size} initializing process group...")
            dist.init_process_group(
                backend="nccl",
                rank=self.rank,
                world_size=self.world_size,
                init_method=f"env://",
                timeout=timedelta(minutes=30)
            )
            print(f"Process group initialized for rank {self.rank}.")

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        self.device = torch.device(f"cuda:{local_rank}")
        dist.barrier() # Synchronize all processes

    def init_optimizer_scheduler(self) -> None:
        """
        Initializes the optimizer and learning rate scheduler based on config settings.
        Handles different optimizer types and learning rate schedules for NGPT vs. GPT.
        """
        # Get model parameters (handle DDP wrapper)
        model_params = self.model.module.parameters() if isinstance(self.model, DDP) else self.model.parameters()

        # Optimizer Initialization
        lr = self.config.optimizer_config.learning_rate
        weight_decay = self.config.optimizer_config.weight_decay
        optimizer_name = self.config.optimizer_config.optimizer_name

        if optimizer_name == "adam":
            self.optimizer = Adam(model_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            self.optimizer = AdamW(model_params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Learning Rate Scheduler Initialization
        T_max = self.config.training_config.max_train_steps
        eta_min = self.config.optimizer_config.final_learning_rate
        warmup_steps = self.config.optimizer_config.warmup_steps

        if self.config.model_config.model_type == "ngpt":
            # NGPT: Cosine Annealing without warmup (paper Section 2.6, Table 3)
            self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)
        elif self.config.model_config.model_type == "gpt":
            # GPT Baseline: Linear Warmup followed by Cosine Annealing (paper Table 3)
            if warmup_steps > 0:
                warmup_scheduler = LinearLR(
                    self.optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
                )
                cosine_scheduler = CosineAnnealingLR(
                    self.optimizer, T_max=T_max - warmup_steps, eta_min=eta_min
                )
                self.lr_scheduler = SequentialLR(
                    self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
                )
            else:
                self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)
        else:
            raise ValueError(f"Unknown model_type: {self.config.model_config.model_type}")

        if self.rank == 0:
            print(f"Optimizer: {type(self.optimizer).__name__}, LR Scheduler: {type(self.lr_scheduler).__name__}")

    def train(self) -> None:
        """
        Executes the main training loop, orchestrating training steps, validation,
        logging, checkpointing, and the critical NGPT normalization.
        """
        self.model.train()  # Set model to training mode

        # Prepare data loader and iterator
        train_dataloader = self.data_module.train_dataloader()
        train_dataloader_iter = iter(train_dataloader)

        # Initialize progress bar on rank 0
        pbar = tqdm(range(self.config.training_config.max_train_steps), desc="Training",
                    disable=self.rank != 0)

        for current_step in pbar:
            log_loss = 0.0
            for _ in range(self.config.training_config.gradient_accumulation_steps):
                try:
                    batch = next(train_dataloader_iter)
                except StopIteration:
                    # Dataset exhausted, re-initialize iterator
                    if self.rank == 0:
                        print("Train dataloader exhausted, re-initializing.")
                    train_dataloader_iter = iter(train_dataloader)
                    batch = next(train_dataloader_iter) # Fetch next batch from fresh iterator

                loss = self.train_step(batch)
                log_loss += loss.item() # Keep track of the unscaled loss for logging

            # Perform optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # --- NGPT Specific Post-Optimizer Normalization (Crucial Step) ---
            # Paper Section 2.6, Step 2: "After each training step ..., normalize matrices..."
            # The model's post_optimizer_norm handles iterating through all relevant parameters.
            # If using DDP, call on the unwrapped model module.
            if isinstance(self.model, DDP):
                self.model.module.post_optimizer_norm()
            else:
                self.model.post_optimizer_norm()

            self.optimizer.zero_grad()
            self.lr_scheduler.step()

            # Logging
            if (current_step % self.config.training_config.log_interval == 0) and (self.rank == 0):
                current_lr = self.lr_scheduler.get_last_lr()[0]
                pbar.set_postfix(
                    loss=f"{log_loss:.4f}",
                    lr=f"{current_lr:.6f}"
                )
                print(f"Step {current_step}: Loss {log_loss:.4f}, LR {current_lr:.6f}")

            # Evaluation
            if (current_step % self.config.training_config.eval_interval == 0) and (current_step > 0):
                val_loss = self.validation_step()
                if self.rank == 0:
                    print(f"--- Validation at Step {current_step}: Loss {val_loss:.4f} ---")
                self.model.train() # Set model back to training mode after validation

            # Checkpointing
            # Not specified in config, adding a default for saving intermediate states
            checkpoint_interval = self.config.training_config.max_train_steps // 5 # Save 5 times during training
            if (current_step % checkpoint_interval == 0 and current_step > 0) or \
               (current_step == self.config.training_config.max_train_steps - 1):
                self.save_checkpoint(current_step)

        if self.rank == 0:
            print("\nTraining complete.")
            pbar.close()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Performs a single forward and backward pass for a given batch.

        Args:
            batch: A dictionary containing 'input_ids' and 'labels' tensors.

        Returns:
            The computed loss for the batch (unscaled by gradient_accumulation_steps).
        """
        input_ids = batch['input_ids'].to(self.device)
        targets = batch['labels'].to(self.device)

        # Mixed precision context
        with autocast(enabled=self.scaler is not None,
                      dtype=torch.bfloat16 if self.config.training_config.precision == "bfloat16" else torch.float32):
            loss, _ = self.model(input_ids, targets)

        # Scale the loss for gradient accumulation (gradient is effectively accumulated over steps)
        loss = loss / self.config.training_config.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss # Return the loss tensor (will be .item() for logging)

    def validation_step(self) -> float:
        """
        Orchestrates the validation process by calling the evaluator.

        Returns:
            The global average validation loss.
        """
        return self.evaluator.evaluate_validation_loss()

    def save_checkpoint(self, step: int) -> None:
        """
        Saves the current state of the model, optimizer, and scheduler.
        Only executed on rank 0 in a distributed setting.

        Args:
            step: The current training step number.
        """
        if self.rank == 0:
            print(f"Saving checkpoint at step {step}...")
            # Unwrapped model for state_dict if DDP is used
            model_to_save = self.model.module if isinstance(self.model, DDP) else self.model

            checkpoint_dict = {
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                "step": step,
                "config": self.config.to_json_string(), # Save config for exact reproducibility
            }
            if self.scaler is not None:
                checkpoint_dict["scaler_state_dict"] = self.scaler.state_dict()

            checkpoint_dir = "checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step:06d}.pt")
            torch.save(checkpoint_dict, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

