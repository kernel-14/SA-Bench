## trainers/base_trainer.py
import abc
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from models.flow_matching_unet import FlowMatchingUNet
from models.reward_model import RewardModel
from data.dataset import TextPromptDataset
from diffusion.sde_solver import SDESolver
from diffusion.noise_schedule import NoiseSchedule


class BaseTrainer(ABC):
  """
  Abstract base class for all fine-tuning trainers.

  This class provides common functionalities for managing the training process,
  including model and device setup, optimizer management, basic training loop
  structure, gradient handling (mixed precision and clipping), and checkpointing.
  Concrete subclasses must implement the `_compute_loss` method with their
  specific loss calculation logic.
  """

  def __init__(
      self,
      config: Config,
      flow_model: FlowMatchingUNet,
      reward_model: RewardModel,
      dataset: TextPromptDataset,
      sde_solver: SDESolver,
      noise_schedule: NoiseSchedule,
      optimizer: torch.optim.Optimizer,
  ):
    """
    Initializes the BaseTrainer.

    Args:
        config: The global configuration object.
        flow_model: The FlowMatchingUNet instance representing v_finetune,
                    whose parameters are being optimized.
        reward_model: The RewardModel instance.
        dataset: The TextPromptDataset for training prompts.
        sde_solver: The SDESolver instance for forward trajectory simulation.
        noise_schedule: The NoiseSchedule utility.
        optimizer: The instantiated torch.optim.Optimizer for flow_model.
    """
    if not isinstance(config, Config):
      raise TypeError("config must be an instance of Config.")
    if not isinstance(flow_model, FlowMatchingUNet):
      raise TypeError("flow_model must be an instance of FlowMatchingUNet.")
    if not isinstance(reward_model, RewardModel):
      raise TypeError("reward_model must be an instance of RewardModel.")
    if not isinstance(dataset, TextPromptDataset):
      raise TypeError("dataset must be an instance of TextPromptDataset.")
    if not isinstance(sde_solver, SDESolver):
      raise TypeError("sde_solver must be an instance of SDESolver.")
    if not isinstance(noise_schedule, NoiseSchedule):
      raise TypeError("noise_schedule must be an instance of NoiseSchedule.")
    if not isinstance(optimizer, torch.optim.Optimizer):
      raise TypeError("optimizer must be an instance of torch.optim.Optimizer.")

    self.config: Config = config
    self.flow_model: FlowMatchingUNet = flow_model
    self.reward_model: RewardModel = reward_model
    self.dataset: TextPromptDataset = dataset
    self.sde_solver: SDESolver = sde_solver
    self.noise_schedule: NoiseSchedule = noise_schedule
    self.optimizer: torch.optim.Optimizer = optimizer

    self.device: str = config.general.device

    # Ensure models are on the correct device (they should be already from main.py,
    # but this is a safeguard)
    self.flow_model.to(self.device)
    self.reward_model.to(self.device)

    # Setup DataLoader
    self.dataloader: DataLoader = DataLoader(
        self.dataset,
        batch_size=self.config.fine_tuning.batch_size,
        shuffle=True,
        # Using 0 workers for simplicity; can be increased (e.g., os.cpu_count())
        # for better performance on systems with many CPU cores and fast I/O.
        num_workers=0,
        pin_memory=(self.device == "cuda"),
    )
    # Initialize data iterator to fetch batches
    self.data_iterator = iter(self.dataloader)

    # Setup Mixed Precision (AMP)
    self.grad_scaler: Optional[torch.cuda.amp.GradScaler] = None
    if self.config.fine_tuning.precision == "bfloat16":
      self.grad_scaler = torch.cuda.amp.GradScaler()
      print("Initialized GradScaler for bfloat16 mixed precision training.")

    # Create checkpoint directory
    self.checkpoint_dir: str = os.path.join(
        self.config.general.output_dir, self.config.general.run_name, "checkpoints"
    )
    os.makedirs(self.checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory created at: {self.checkpoint_dir}")

  @abstractmethod
  def _compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
    """
    Abstract method to compute the specific loss for a given fine-tuning method.
    Subclasses must implement this method.

    Args:
        batch: A dictionary containing a batch of data from the DataLoader.
               Expected to include at least 'text_embeddings' and 'prompts'.

    Returns:
        A scalar torch.Tensor representing the loss value for the current batch.
    """
    pass

  def _save_checkpoint(self, iteration: int) -> None:
    """
    Saves the state dictionary of the flow model to a checkpoint file.

    Args:
        iteration: The current training iteration number, used for naming the checkpoint.
    """
    checkpoint_name = f"flow_model_iter_{iteration:06d}.pt"
    checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
    torch.save(self.flow_model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint at iteration {iteration} to {checkpoint_path}")

  def train(self) -> FlowMatchingUNet:
    """
    Executes the main fine-tuning loop for the flow model.

    Manages batch processing, optimizer steps, gradient scaling (if using AMP),
    gradient clipping, and periodic checkpointing.

    Returns:
        The fine-tuned FlowMatchingUNet model.
    """
    print(f"Starting fine-tuning with method: {self.config.fine_tuning.method}")
    print(f"Total iterations: {self.config.fine_tuning.num_fine_tune_iterations}")

    self.flow_model.train()  # Set the trainable model to training mode
    self.reward_model.eval()  # Reward model is typically in evaluation mode during fine-tuning

    pbar = tqdm(
        range(self.config.fine_tuning.num_fine_tune_iterations),
        desc=f"Fine-tuning ({self.config.fine_tuning.method})",
    )

    for iteration in pbar:
      try:
        batch = next(self.data_iterator)
      except StopIteration:
        # Re-initialize the data iterator if it runs out of batches
        self.data_iterator = iter(self.dataloader)
        batch = next(self.data_iterator)

      # Move relevant batch data to the correct device.
      # `prompts` is a list of strings and does not need to be moved to device.
      if "text_embeddings" in batch and isinstance(batch["text_embeddings"], torch.Tensor):
        batch["text_embeddings"] = batch["text_embeddings"].to(self.device)
      # Other potential tensor data in batch should also be moved if applicable
      # For now, we assume only text_embeddings are primary tensor inputs from dataset.

      self.optimizer.zero_grad()

      with torch.cuda.amp.autocast(enabled=(self.grad_scaler is not None)):
        loss = self._compute_loss(batch)

      if self.grad_scaler is not None:
        # Scale loss and perform backward pass for mixed precision
        self.grad_scaler.scale(loss).backward()
        # Unscale gradients before clipping, as clipping should be on original scale
        self.grad_scaler.unscale_(self.optimizer)
      else:
        # Standard backward pass for full precision
        loss.backward()

      # Apply gradient norm clipping to prevent exploding gradients
      torch.nn.utils.clip_grad_norm_(
          self.flow_model.parameters(),
          self.config.fine_tuning.optimizer.gradient_norm_clip,
      )

      if self.grad_scaler is not None:
        # Update optimizer's parameters and update the scaler
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
      else:
        # Standard optimizer step
        self.optimizer.step()

      # Update the progress bar with the current loss
      pbar.set_postfix(loss=f"{loss.item():.4f}")

      # Checkpointing: Save model weights periodically and at the very end
      if (iteration + 1) % self.config.evaluation.eval_frequency == 0 or (
          iteration + 1 == self.config.fine_tuning.num_fine_tune_iterations
      ):
        self._save_checkpoint(iteration + 1)

    print("Fine-tuning completed.")
    return self.flow_model

