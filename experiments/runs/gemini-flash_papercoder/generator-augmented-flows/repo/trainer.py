# trainer.py

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs # Import Accelerator
import os
from typing import Tuple, Optional, Iterator, Any

# Project-specific imports
from config import Config
from utils import Utils
from model import ConsistencyModel, EMA
from evaluator import Evaluator # Assuming Evaluator exists and has the evaluate method

class Trainer:
    """
    Orchestrates the training process for consistency models with generator-augmented flows.
    Handles the training loop, loss calculation (Algorithm 1), optimizer steps, EMA updates,
    and periodic evaluation.
    """

    def __init__(
        self,
        model: ConsistencyModel,
        ema_model: EMA,
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        sigma_d_sq: float,
        config: Config,
        evaluator: Evaluator
    ):
        """
        Initializes the Trainer instance.

        Args:
            model: The primary ConsistencyModel instance to be trained.
            ema_model: An EMA wrapper around the model used for stable predictions (target network).
            optimizer: The torch.optim.Optimizer used for updating model's weights.
            train_dataloader: A torch.utils.data.DataLoader providing training data batches.
            sigma_d_sq: The empirically calculated variance of the data distribution.
            config: The global Config object containing all hyperparameters.
            evaluator: An Evaluator instance to perform periodic evaluations.
        """
        self.config = config
        self.device = config.DEVICE
        self.global_step = 0

        self.sigma_d_sq = sigma_d_sq
        self.evaluator = evaluator
        self.train_dataloader_iter: Optional[Iterator[Any]] = None

        # Accelerator setup for distributed/mixed precision training
        if self.config.DISTRIBUTED:
            # DDP is handled by accelerate.
            # DDP kwargs to allow gradient checkpointing or other features if needed
            # find_unused_parameters=True is often necessary for consistency models
            # due to stop_gradient operations creating unused parameters in DDP.
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
            self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
            print(f"Distributed training enabled. Accelerator device: {self.accelerator.device}")
            self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
                model, optimizer, train_dataloader
            )
            # The EMA model needs to be manually moved to the accelerator's device as it's not trained directly
            self.ema_model.ema_model = self.accelerator.prepare(self.ema_model.ema_model)
            self.device = self.accelerator.device # Update device to accelerator's device
        else:
            self.accelerator = None
            self.model = model.to(self.device)
            self.ema_model.ema_model = ema_model.ema_model.to(self.device) # Move EMA model to device
            self.optimizer = optimizer
            self.train_dataloader = train_dataloader
            print(f"Running on single device: {self.device}")

        self.ema_model = ema_model # Keep the EMA wrapper


    def _sample_timesteps(
        self,
        batch_size: int,
        sigmas: torch.Tensor,
        timestep_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Selects timestep indices `i` and `i+1` for each sample in a batch,
        along with their corresponding sigma values.

        Args:
            batch_size (int): The current batch size.
            sigmas (torch.Tensor): A 1D tensor of sorted noise levels (sigma_0, ..., sigma_N).
                                   Its length is N+1.
            timestep_probs (torch.Tensor): A 1D tensor of probabilities for sampling each
                                           timestep index, shape (N,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - idx_i (torch.Tensor): Sampled timestep indices, shape (batch_size,).
                - sigma_ti (torch.Tensor): Noise levels corresponding to idx_i, shape (batch_size,).
                - sigma_ti_plus_1 (torch.Tensor): Noise levels corresponding to idx_i + 1,
                                                 shape (batch_size,).
        """
        # Sample timestep index `i` from {0, ..., N-1}
        # torch.multinomial expects probabilities for sampling elements.
        # timestep_probs has shape (N,) where N is current_N.
        idx_i = torch.multinomial(
            timestep_probs,
            num_samples=batch_size,
            replacement=True
        ).to(self.device)

        # The sigma schedule `sigmas` has N+1 elements (sigma_0 to sigma_N).
        # sigma_ti corresponds to sigmas[idx_i].
        # sigma_ti_plus_1 corresponds to sigmas[idx_i + 1].
        # This is safe because idx_i is sampled from [0, N-1], so idx_i+1 is max N.
        sigma_ti = sigmas[idx_i]
        sigma_ti_plus_1 = sigmas[idx_i + 1]

        return idx_i, sigma_ti, sigma_ti_plus_1

    def _calculate_loss(self, batch: torch.Tensor, current_step: int) -> torch.Tensor:
        """
        Implements the core logic of Algorithm 1 to compute the joint learning loss
        L_GC-mu(theta) for a given batch.

        Args:
            batch (torch.Tensor): A batch of real data samples (x_star).
            current_step (int): The current global training step.

        Returns:
            torch.Tensor: The scalar loss for the current batch.
        """
        x_star = batch.to(self.device)
        batch_size = x_star.shape[0]

        # Sample noise vector z
        z = torch.randn_like(x_star).to(self.device)

        # Progressive timestep scheduling
        current_N = Utils.get_progressive_N(
            current_step=current_step,
            total_steps=self.config.TRAINING_STEPS,
            s0=self.config.S0,
            s1=self.config.S1
        )

        # Generate Karras sigmas and loss weights for the current N
        sigmas = Utils.get_karras_sigmas(
            sigma_min=self.config.SIGMA_0,
            sigma_max=self.config.SIGMA_T,
            rho=self.config.RHO,
            N=current_N,
            device=self.device
        )
        
        # Loss weights lambda(sigma_i)
        loss_weights_lambda = Utils.get_loss_weights(sigmas)
        
        # Timestep sampling probabilities. Note: probs are for N-1 intervals.
        # sigmas[:-1] provides sigma_0 to sigma_{N-1} which are the starting points of intervals.
        timestep_probs = Utils.get_timestep_probabilities(
            sigmas=sigmas, # Pass full sigmas, Utils will use sigmas[:-1] and sigmas[1:]
            p_mean=self.config.P_MEAN,
            p_std=self.config.P_STD,
            device=self.device
        )

        # Sample timestep indices and corresponding sigma values
        idx_i, sigma_ti, sigma_ti_plus_1 = self._sample_timesteps(
            batch_size, sigmas, timestep_probs
        )

        # Reshape sigma_tensors for broadcasting: (batch_size,) -> (batch_size, 1, 1, 1)
        sigma_ti_reshaped = sigma_ti.view(-1, 1, 1, 1)
        sigma_ti_plus_1_reshaped = sigma_ti_plus_1.view(-1, 1, 1, 1)

        # 1. Construct IC intermediate point x_ti
        x_ti = x_star + sigma_ti_reshaped * z

        # 2. Predict endpoint hat_x_ti from EMA model (with stop_gradient)
        # Ensure EMA model is in eval mode for consistent behavior and no grad.
        # Unwrap model if DDP is used for direct access to its state_dict/forward method.
        ema_model_inst = self.ema_model.get_model()
        if self.accelerator:
            ema_model_inst = self.accelerator.unwrap_model(ema_model_inst)
        
        ema_model_inst.eval() 
        with torch.no_grad():
            hat_x_ti = ema_model_inst(x_ti, sigma_ti) # sigma_ti is passed as (batch_size,)
        
        # 3. Joint Learning Mixing
        # m ~ binomial(mu, size=batch_size)
        m_mask = (torch.rand(batch_size, device=self.device) < self.config.MU).float()
        m_mask_reshaped = m_mask.view(-1, 1, 1, 1)
        
        # hat_x_ti_mixed = m * hat_x_ti + (1 - m) * x_star
        hat_x_ti_mixed = m_mask_reshaped * hat_x_ti + (1 - m_mask_reshaped) * x_star

        # 4. Construct GC trajectory points
        tilde_x_ti = hat_x_ti_mixed + sigma_ti_reshaped * z
        tilde_x_ti_plus_1 = hat_x_ti_mixed + sigma_ti_plus_1_reshaped * z

        # 5. Calculate model predictions for loss
        # The training model should be in train mode for its forward pass
        self.model.train() 

        # The first term (f_theta(tilde_x_ti, sigma_ti)) has stop_gradient
        # `pred_tilde_x_0_i_sg` will act as the target, detached from graph.
        with torch.no_grad():
            pred_tilde_x_0_i_sg = self.model(tilde_x_ti, sigma_ti)
        
        pred_tilde_x_0_i_plus_1 = self.model(tilde_x_ti_plus_1, sigma_ti_plus_1)

        # 6. Compute Consistency Loss (MSE with lambda weighting)
        mse_loss_raw = F.mse_loss(pred_tilde_x_0_i_sg, pred_tilde_x_0_i_plus_1, reduction='none')
        
        # Apply the loss weighting lambda(sigma_i)
        # loss_weights_lambda[idx_i] gives the weight for each sample in the batch
        # Reshape for broadcasting over image dimensions (C, H, W)
        weighted_loss = loss_weights_lambda[idx_i].view(-1, 1, 1, 1) * mse_loss_raw
        
        # Average over batch and spatial dimensions to get a scalar loss
        loss = weighted_loss.mean()

        return loss

    def train(self) -> None:
        """
        Executes the main training loop.
        """
        # Ensure models are on the correct device and in appropriate modes
        # Accelerator.prepare() already handles moving model/optimizer/dataloader to device
        # For non-accelerator, models are moved in __init__
        self.model.train()
        # EMA model used for prediction, so always eval mode; handled in _calculate_loss context
        
        # Initialize progress bar
        progress_bar = tqdm(
            range(self.global_step, self.config.TRAINING_STEPS),
            initial=self.global_step,
            total=self.config.TRAINING_STEPS,
            desc="Training"
        )

        for step in progress_bar:
            self.global_step = step

            # Fetch batch from dataloader, re-initialize iterator if exhausted
            try:
                if self.train_dataloader_iter is None:
                    self.train_dataloader_iter = iter(self.train_dataloader)
                batch, _ = next(self.train_dataloader_iter)
            except StopIteration:
                self.train_dataloader_iter = iter(self.train_dataloader)
                batch, _ = next(self.train_dataloader_iter)

            # Calculate loss
            if self.accelerator:
                # Use accelerator.accumulate to handle gradient accumulation if configured
                with self.accelerator.accumulate(self.model):
                    loss = self._calculate_loss(batch, self.global_step)
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            else:
                loss = self._calculate_loss(batch, self.global_step)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Update EMA model weights
            # Pass the unwrapped model to EMA update if DDP is used
            model_for_ema_update = self.accelerator.unwrap_model(self.model) if self.accelerator else self.model
            self.ema_model.update(model_for_ema_update)

            # Logging
            if (self.global_step + 1) % self.config.LOG_INTERVAL_STEPS == 0:
                if self.accelerator is None or self.accelerator.is_main_process:
                    progress_bar.set_postfix(loss=loss.item())

            # Evaluation
            if (self.global_step + 1) % self.config.EVAL_INTERVAL_STEPS == 0:
                # Only the main process performs evaluation to avoid redundant calculations
                if self.accelerator is None or self.accelerator.is_main_process:
                    print(f"\n--- Evaluating at step {self.global_step + 1} ---")
                    # Evaluator internally handles setting models to eval mode
                    metrics = self.evaluator.evaluate(self.global_step + 1, use_ema_for_generation=True)

                    for metric_name, (mean_val, std_val) in metrics.items():
                        print(f"  {metric_name}: {mean_val:.4f} +/- {std_val:.4f}")
                    print(f"------------------------------------")
                # Ensure all processes sync before continuing training, especially after eval
                if self.accelerator:
                    self.accelerator.wait_for_everyone()

            # Checkpointing
            if (self.global_step + 1) % self.config.SAVE_INTERVAL_STEPS == 0:
                if self.accelerator is None or self.accelerator.is_main_process:
                    checkpoint_path = os.path.join(
                        self.config.CHECKPOINT_DIR,
                        f"model_step_{self.global_step + 1}.pth"
                    )
                    # Unwrap model if DDP is used before saving state dict
                    model_to_save = self.accelerator.unwrap_model(self.model) if self.accelerator else self.model
                    ema_model_to_save = self.accelerator.unwrap_model(self.ema_model.get_model()) if self.accelerator else self.ema_model.get_model()

                    torch.save({
                        'global_step': self.global_step + 1,
                        'model_state_dict': model_to_save.state_dict(),
                        'ema_model_state_dict': ema_model_to_save.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'config': self.config # Save the entire config object for reproducibility
                    }, checkpoint_path)
                    print(f"Checkpoint saved to {checkpoint_path}")
                if self.accelerator:
                    self.accelerator.wait_for_everyone() # Wait for all processes to finish saving/sync

        # Final save after training completes
        if self.accelerator is None or self.accelerator.is_main_process:
            final_checkpoint_path = os.path.join(self.config.CHECKPOINT_DIR, "model_final.pth")
            model_to_save = self.accelerator.unwrap_model(self.model) if self.accelerator else self.model
            ema_model_to_save = self.accelerator.unwrap_model(self.ema_model.get_model()) if self.accelerator else self.ema_model.get_model()
            torch.save({
                'global_step': self.config.TRAINING_STEPS,
                'model_state_dict': model_to_save.state_dict(),
                'ema_model_state_dict': ema_model_to_save.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config
            }, final_checkpoint_path)
            print(f"Final model saved to {final_checkpoint_path}")

