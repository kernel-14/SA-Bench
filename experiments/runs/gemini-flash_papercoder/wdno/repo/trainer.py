import torch
import torch.optim
import torch.optim.lr_scheduler
from tqdm import tqdm
from typing import Iterator, Optional, Union, Dict, Any

# Local imports
from config import Config
from wdno_models import BaseResolutionModel, SuperResolutionModel
from data_module import MultiResolutionDataset, SingleResolutionDataset
from utils import save_checkpoint, load_checkpoint, get_device, find_latest_checkpoint


class DiffusionTrainer:
    """
    Manages the training loop for both Base-Resolution Model (BRM) and
    Super-Resolution Model (SRM). Handles optimization, learning rate scheduling,
    logging, and checkpointing.
    """

    def __init__(self,
                 config: Config,
                 model: Union[BaseResolutionModel, SuperResolutionModel],
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
        """
        Initializes the DiffusionTrainer.

        Args:
            config: The global configuration object.
            model: An instance of either BaseResolutionModel or SuperResolutionModel.
            optimizer: The optimizer for the model.
            lr_scheduler: Optional learning rate scheduler for the optimizer.
        """
        self.config: Config = config
        self.model: Union[BaseResolutionModel, SuperResolutionModel] = model
        self.optimizer: torch.optim.Optimizer = optimizer
        self.lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = lr_scheduler
        self.device: torch.device = get_device(self.config.device)

        # Move model to the specified device
        self.model.to(self.device)

        self.log_interval_steps: int = self.config.log_interval_steps
        self.eval_interval_steps: int = self.config.eval_interval_steps
        self.start_step: int = 0

    def _get_data_iterator(self, dataloader: torch.utils.data.DataLoader) -> Iterator[Any]:
        """
        Creates an infinite iterator from a DataLoader.

        Args:
            dataloader: The DataLoader to create an iterator from.

        Returns:
            An infinite iterator over the DataLoader's batches.
        """
        while True:
            for batch in dataloader:
                yield batch

    def train(self,
              dataloader: torch.utils.data.DataLoader,
              total_steps: int,
              model_type_str: str):
        """
        Executes the main training process for the model.

        Args:
            dataloader: A DataLoader providing batches of training data.
                        Can be SingleResolutionDataset for BRM or MultiResolutionDataset for SRM.
            total_steps: Total number of training steps to perform.
            model_type_str: Identifier for the model being trained (e.g., "BRM", "SRM").
        """
        self.model.train()  # Set model to training mode
        data_iterator: Iterator[Any] = self._get_data_iterator(dataloader)

        # Try to load a checkpoint to resume training
        try:
            self.start_step = self.load_checkpoint(model_type_str, step=None) # Attempt to load latest
            tqdm.write(f"Resuming {model_type_str} training from step {self.start_step}")
        except FileNotFoundError:
            tqdm.write(f"No existing checkpoint found for {model_type_str}. Starting training from scratch.")
        except Exception as e:
            tqdm.write(f"Error loading checkpoint for {model_type_str}: {e}. Starting training from scratch.")

        pbar = tqdm(range(self.start_step + 1, total_steps + 1), initial=self.start_step, total=total_steps, desc=f"Training {model_type_str}")

        for step in pbar:
            batch: Dict[str, Any] = next(data_iterator)

            # Move batch data to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)
                elif isinstance(value, list) and all(isinstance(item, torch.Tensor) for item in value):
                    batch[key] = [item.to(self.device) for item in value]
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, torch.Tensor):
                            batch[key][sub_key] = sub_value.to(self.device)


            self.optimizer.zero_grad()

            loss: torch.Tensor
            if isinstance(self.model, BaseResolutionModel):
                # For BRM, batch contains 'x_0_wavelets' and 'conditions_wavelets'
                x_0_wavelets = batch['x_0_wavelets']
                conditions_wavelets = batch['conditions_wavelets']
                loss = self.model.forward_diffusion_step(x_0_wavelets, conditions_wavelets)
            elif isinstance(self.model, SuperResolutionModel):
                # For SRM, batch contains 'high_res_x_0_wavelets', 'low_res_x_wavelets_upsampled', 'high_res_conditions_wavelets'
                high_res_x_0_wavelets = batch['high_res_x_0_wavelets']
                low_res_x_wavelets = batch['low_res_x_wavelets_upsampled'] # This is W_l
                high_res_conditions_wavelets = batch['high_res_conditions_wavelets'] # This is W_a_h
                loss = self.model.forward_diffusion_step(high_res_x_0_wavelets, low_res_x_wavelets, high_res_conditions_wavelets)
            else:
                raise TypeError(f"Unsupported model type for training: {type(self.model)}")

            loss.backward()
            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Logging
            if step % self.log_interval_steps == 0 or step == total_steps:
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.6f}")
                tqdm.write(f"Step {step}/{total_steps} - Loss: {loss.item():.6f} - LR: {current_lr:.8f}")

            # Checkpointing
            if step % self.eval_interval_steps == 0 or step == total_steps:
                self.save_checkpoint(step, model_type_str)

        tqdm.write(f"{model_type_str} training completed after {total_steps} steps.")

    def save_checkpoint(self, step: int, model_type_str: str):
        """
        Saves the current training state to a checkpoint file.

        Args:
            step: The current training step.
            model_type_str: A string indicating the type of model (e.g., "BRM", "SRM").
        """
        checkpoint_filename = f"{model_type_str}_step_{step:07d}.pt"
        filepath = os.path.join(self.config.save_path, checkpoint_filename)
        save_checkpoint(step, self.model, self.optimizer, self.lr_scheduler, filepath)

    def load_checkpoint(self, model_type_str: str, step: Optional[int] = None) -> int:
        """
        Loads a previously saved training state from a checkpoint file.

        Args:
            model_type_str: A string indicating the type of model (e.g., "BRM", "SRM").
            step: An optional integer specifying a particular step checkpoint to load.
                  If None, the latest available checkpoint for model_type_str will be loaded.

        Returns:
            The training step from which the checkpoint was loaded.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        filepath: str
        if step is None:
            # Find the latest checkpoint
            latest_checkpoint_path = find_latest_checkpoint(self.config.save_path, model_type_str)
            if latest_checkpoint_path is None:
                raise FileNotFoundError(f"No latest checkpoint found for model type '{model_type_str}' in '{self.config.save_path}'")
            filepath = latest_checkpoint_path
        else:
            checkpoint_filename = f"{model_type_str}_step_{step:07d}.pt"
            filepath = os.path.join(self.config.save_path, checkpoint_filename)

        loaded_step = load_checkpoint(self.model, self.optimizer, self.lr_scheduler, filepath, self.device)
        return loaded_step

