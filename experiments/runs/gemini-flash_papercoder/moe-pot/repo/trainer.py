# trainer.py
import os
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW # Using AdamW as a common practice, paper states Adam
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from config import Config
from data.datamodule import PDEDataModule
from evaluator import Evaluator
from model.moepot import MoEPOT
from utils import (calculate_l2re, cleanup_distributed_training,
                   get_lr_scheduler, inject_noise, load_checkpoint,
                   save_checkpoint, set_seed, setup_distributed_training)


class Trainer:
    """
    Manages the training loop for MoE-POT models, including pre-training, fine-tuning,
    loss calculation, optimization, and evaluation.
    """

    def __init__(self,
                 model: MoEPOT,
                 datamodule: PDEDataModule,
                 config: Config,
                 rank: int,
                 world_size: int,
                 is_pretraining: bool = True):
        """
        Initializes the Trainer.

        Args:
            model: The MoEPOT model instance.
            datamodule: The PDEDataModule instance providing data loaders.
            config: The global configuration object.
            rank: The current process rank in distributed training.
            world_size: The total number of processes participating in distributed training.
            is_pretraining: A boolean flag indicating if the current stage is pre-training.
        """
        self.model = model
        self.datamodule = datamodule
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_pretraining = is_pretraining

        self.device = torch.device(f'cuda:{self.rank}')

        # 1. Model Device Placement and DDP Wrapper
        self.model.to(self.device)
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.rank])
        
        # 2. Optimizer Initialization
        optimizer_name = self.config.training.optimizer
        if optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=(self.config.training.beta1, self.config.training.beta2)
            )
        elif optimizer_name == 'AdamW': # Added for completeness, paper states Adam
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=(self.config.training.beta1, self.config.training.beta2)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # 3. Learning Rate Scheduler Initialization
        self.total_epochs = self.config.training.current_epochs
        self.warmup_epochs = self.config.training.current_warmup_epochs
        self.lr_scheduler = get_lr_scheduler(
            self.optimizer, self.config, self.total_epochs, self.warmup_epochs
        )

        # 4. Evaluator Initialization
        self.evaluator = Evaluator(self.model, self.config, self.rank, self.world_size)

        # 5. Logging and Checkpointing Setup
        self.best_val_metric = float('inf') # Lower L2RE is better
        self.start_epoch = 0

        # Construct checkpoint directory specific to the experiment
        self.experiment_dir = os.path.join(self.config.output_dir, self.config.experiment_name)
        self.checkpoint_dir = os.path.join(self.experiment_dir, self.config.checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.log_dir = os.path.join(self.experiment_dir, self.config.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # If pre-training, check for latest checkpoint to resume
        if self.is_pretraining and self.rank == 0:
            latest_checkpoint = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
            if os.path.exists(latest_checkpoint):
                self.start_epoch, self.best_val_metric = load_checkpoint(self.model, self.optimizer, latest_checkpoint)
                if self.world_size > 1: # Broadcast best_val_metric to all ranks
                    torch.distributed.broadcast(torch.tensor(self.best_val_metric, device=self.device), src=0)
                    torch.distributed.broadcast(torch.tensor(self.start_epoch, device=self.device), src=0)


        # 6. Fine-tuning Specifics
        if not self.is_pretraining:
            # For fine-tuning, freeze router parameters
            # Call on the unwrapped model first if DDP, then DDP will reflect the state
            if self.world_size > 1:
                self.model.module.freeze_router()
            else:
                self.model.freeze_router()


    def train(self) -> str:
        """
        Executes the main training loop.

        Returns:
            The file path to the best saved model checkpoint.
        """
        if self.rank == 0:
            print(f"Starting {'Pre-training' if self.is_pretraining else 'Fine-tuning/Downstream'} "
                  f"for {self.total_epochs} epochs. Warmup for {self.warmup_epochs} epochs.")

        train_loader = self.datamodule.train_dataloader(self.rank, self.world_size)
        val_loader = self.datamodule.val_dataloader(self.rank, self.world_size)
        val_dataloaders = {'validation': val_loader} # For Evaluator API

        best_model_path: str = os.path.join(self.checkpoint_dir, "model_best.pt")

        for epoch in range(self.start_epoch, self.total_epochs):
            # Set epoch for DistributedSampler to ensure proper shuffling
            if self.world_size > 1:
                if isinstance(train_loader.sampler, DistributedSampler):
                    train_loader.sampler.set_epoch(epoch)
                elif hasattr(train_loader.sampler, 'set_epoch'): # Custom samplers might have this
                     train_loader.sampler.set_epoch(epoch)
                
            self.model.train()
            total_train_loss = 0.0
            
            # Using tqdm for progress bar on rank 0
            if self.rank == 0:
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.total_epochs} (Train)")
            else:
                pbar = train_loader

            for batch_idx, batch in enumerate(pbar):
                u_seq = batch['u_seq'].to(self.device) # (B, T_in, C, H, W)
                u_target = batch['u_target'].to(self.device) # (B, C, H, W)

                # Noise Injection (Pre-training only)
                u_seq_input = u_seq
                if self.is_pretraining:
                    # Noise is applied to the entire input sequence u^<t
                    u_seq_input = inject_noise(u_seq, self.config.training.noise_epsilon)

                self.optimizer.zero_grad()
                
                # Forward Pass
                predicted_frame, router_weights_per_layer = self.model(u_seq_input)

                # Loss Calculation
                loss = self._compute_loss(predicted_frame, u_target, router_weights_per_layer)
                
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step() # Step scheduler after each batch for OneCycleLR

                total_train_loss += loss.item()

                if self.rank == 0 and (batch_idx + 1) % self.config.training.log_interval == 0:
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    pbar.set_postfix({'loss': loss.item(), 'lr': current_lr})

            avg_train_loss = total_train_loss / len(train_loader)
            if self.rank == 0:
                print(f"Epoch {epoch+1}/{self.total_epochs} - Avg Train Loss: {avg_train_loss:.4f}")

            # Validation Phase
            if (epoch + 1) % self.config.training.eval_interval == 0 or (epoch == self.total_epochs - 1):
                if self.rank == 0:
                    print(f"--- Running Validation for Epoch {epoch+1} ---")
                
                # The paper's main table (Table 1) evaluates L2RE with a 10-step rollout if T_in is 10.
                # It says "predict the solution x_pred for the next 10 steps".
                # For validation, we use T_in as rollout steps to match this.
                val_metrics = self.evaluator.evaluate_model(val_dataloaders, rollout_steps=self.config.model.T_in)
                current_val_l2re = val_metrics.get('validation_l2re', float('inf'))

                if self.rank == 0:
                    print(f"Epoch {epoch+1} Validation L2RE: {current_val_l2re:.4f}")
                    if current_val_l2re < self.best_val_metric:
                        self.best_val_metric = current_val_l2re
                        save_checkpoint(self.model, self.optimizer, epoch + 1, 
                                        os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt"),
                                        self.best_val_metric, is_best=True)
                        print(f"*** New best model saved at epoch {epoch+1} with L2RE: {self.best_val_metric:.4f} ***")

            # Regular Checkpoint Saving (only on rank 0)
            if self.rank == 0 and (epoch + 1) % self.config.training.save_interval == 0 and (epoch != self.total_epochs - 1):
                save_checkpoint(self.model, self.optimizer, epoch + 1, 
                                os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt"),
                                current_val_l2re, is_best=False)
                
            # Sync all processes before moving to next epoch or finishing
            if self.world_size > 1:
                torch.distributed.barrier()

        if self.rank == 0:
            print(f"Training complete. Best validation L2RE: {self.best_val_metric:.4f}")
        
        return best_model_path

    def _compute_loss(self,
                      pred_frame: torch.Tensor,
                      target_frame: torch.Tensor,
                      router_weights_per_layer: List[torch.Tensor]) -> torch.Tensor:
        """
        Calculates the combined prediction loss (MSE) and load balancing loss.

        Args:
            pred_frame: The predicted next frame. Shape (batch_size, C_out, H_patched, W_patched).
            target_frame: The ground truth next frame. Shape (batch_size, C_out, H_patched, W_patched).
            router_weights_per_layer: A list of softmax probabilities from the
                                      router of each MoE layer.
                                      Each element has shape (batch_size, num_routed_experts).

        Returns:
            The total loss (prediction loss + load balancing loss).
        """
        # 1. Prediction Loss (L2 squared error / MSE)
        # The paper uses ||.||_2^2 which implies sum of squared differences.
        # F.mse_loss with reduction='mean' computes (sum(x_i - y_i)^2) / N, where N is num elements.
        # This is equivalent to L2 squared error averaged over elements.
        pred_loss = F.mse_loss(pred_frame, target_frame, reduction='mean')

        # 2. Load Balancing Loss (if applicable)
        balance_loss = torch.tensor(0.0, device=self.device)
        if self.is_pretraining:
            for router_weights_for_layer in router_weights_per_layer:
                # router_weights_for_layer: (batch_size, num_routed_experts)
                
                # Importance of each expert over the batch
                importance_per_expert = torch.sum(router_weights_for_layer, dim=0) # (num_routed_experts,)
                
                mean_importance = torch.mean(importance_per_expert)
                std_importance = torch.std(importance_per_expert)

                # Coefficient of Variation (CV): std / mean
                # Add a small epsilon to avoid division by zero if mean_importance is 0
                cv = std_importance / (mean_importance + 1e-6)

                balance_loss += self.config.training.load_balance_weight * (cv ** 2)

        total_loss = pred_loss + balance_loss
        return total_loss

