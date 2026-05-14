## trainers/adjoint_matching_trainer.py
import os
import random
from typing import Dict, Any, List, Tuple

import torch
from tqdm import tqdm

from config import Config
from models.flow_matching_unet import FlowMatchingUNet
from models.reward_model import RewardModel
from data.dataset import TextPromptDataset
from diffusion.sde_solver import SDESolver
from diffusion.adjoint_solver import LeanAdjointSolver
from diffusion.noise_schedule import NoiseSchedule
from trainers.base_trainer import BaseTrainer
from utils.helpers import stop_gradient, compute_vector_jacobian_product


class AdjointMatchingTrainer(BaseTrainer):
  """
  Trainer for the Adjoint Matching fine-tuning algorithm.

  This class orchestrates the fine-tuning process by performing:
  1. Forward SDE simulation to generate trajectories (X_t).
  2. Backward ODE solution to compute lean adjoint states (a_tilde_t).
  3. Calculation of the Adjoint Matching loss with LCT clipping.
  4. Optimization of the flow model parameters.
  """

  def __init__(
      self,
      config: Config,
      flow_model: FlowMatchingUNet,  # v_finetune
      base_flow_model: FlowMatchingUNet,  # v_base (frozen)
      reward_model: RewardModel,
      dataset: TextPromptDataset,
      sde_solver: SDESolver,
      lean_adjoint_solver: LeanAdjointSolver,
      noise_schedule: NoiseSchedule,
      optimizer: torch.optim.Optimizer,
  ):
    """
    Initializes the AdjointMatchingTrainer.

    Args:
        config: The global configuration object.
        flow_model: The FlowMatchingUNet instance representing v_finetune,
                    whose parameters are being optimized.
        base_flow_model: The pre-trained FlowMatchingUNet instance (v_base),
                         which remains frozen.
        reward_model: The RewardModel instance.
        dataset: The TextPromptDataset for training prompts.
        sde_solver: The SDESolver instance for forward trajectory simulation.
        lean_adjoint_solver: The LeanAdjointSolver instance for backward
                             adjoint calculation.
        noise_schedule: The NoiseSchedule utility.
        optimizer: The instantiated torch.optim.Optimizer for flow_model.
    """
    super().__init__(
        config,
        flow_model,
        reward_model,
        dataset,
        sde_solver,
        noise_schedule,
        optimizer,
    )

    self.base_flow_model: FlowMatchingUNet = base_flow_model
    self.lean_adjoint_solver: LeanAdjointSolver = lean_adjoint_solver
    self.noise_schedule: NoiseSchedule = noise_schedule

    # Ensure base_flow_model is on correct device and frozen
    self.base_flow_model.to(self.device)
    self.base_flow_model.eval()
    for param in self.base_flow_model.parameters():
      param.requires_grad = False

    # Calculate LCT value based on the fine_tuning config
    self.lct_value: float = self.config.fine_tuning.lct_value

    # Set random seed for timestep selection
    random.seed(self.config.general.seed)

    print(f"Adjoint Matching LCT value: {self.lct_value:.2f}")

  def _forward_pass(
      self, x_0: torch.Tensor, text_embeddings: torch.Tensor
  ) -> List[torch.Tensor]:
    """
    Simulates the controlled SDE forward in time using Euler-Maruyama
    discretization and the memoryless noise schedule.
    All intermediate X_t states are detached from the graph.

    Args:
        x_0: The initial latent state (noise N(0,I)).
             Shape: (batch_size, channels, H, W).
        text_embeddings: The conditional text embeddings.
                         Shape: (batch_size, seq_len, embed_dim).

    Returns:
        A list of detached torch.Tensor, representing the trajectory of X_t states
        from X_0 to X_K (K+1 elements).
    """
    x_t_current = x_0.to(self.device)
    x_trajectory: List[torch.Tensor] = [stop_gradient(x_t_current)]

    # Timesteps for the forward SDE integration: [0, h, 2h, ..., (K-1)h]
    timesteps_float = self.noise_schedule.get_timesteps_float(
        self.config.fine_tuning.num_timesteps
    )

    for t_float in timesteps_float:
      t_tensor = torch.tensor(
          [t_float], device=self.device, dtype=x_0.dtype
      ).repeat(x_0.shape[0])

      # Get velocity prediction from the fine-tuning model (v_finetune)
      v_finetune_pred = self.flow_model.get_velocity(
          x_t_current, t_tensor, text_embeddings
      )

      # Calculate drift for the forward pass of the controlled SDE (Algorithm 1, Eq. 40)
      # drift = 2 * v_theta_finetune(X_t, t) - kappa_t * X_t
      kappa_t = self.noise_schedule.get_kappa_t(t_tensor).view(-1, 1, 1, 1)
      drift = 2.0 * v_finetune_pred - kappa_t * x_t_current

      # Get the memoryless diffusion coefficient for fine-tuning
      sigma_t_current = self.noise_schedule.get_memoryless_sigma_t(t_tensor).view(
          -1, 1, 1, 1
      )

      # Generate Gaussian noise
      epsilon_noise = torch.randn_like(x_t_current).to(self.device)

      # Euler-Maruyama update for X_{t+h}
      x_t_next = x_t_current + self.config.fine_tuning.h_timestep * drift + torch.sqrt(
          torch.tensor(
              self.config.fine_tuning.h_timestep,
              device=self.device,
              dtype=x_0.dtype,
          )
      ) * sigma_t_current * epsilon_noise

      x_trajectory.append(stop_gradient(x_t_next))
      x_t_current = x_t_next

    return x_trajectory

  def _backward_adjoint_pass(
      self,
      x_trajectory: List[torch.Tensor],
      text_embeddings: torch.Tensor,
      prompts: List[str],
  ) -> List[torch.Tensor]:
    """
    Computes the lean adjoint trajectory (a_tilde_t) by solving the lean adjoint
    ODE backward in time using Euler discretization.
    All intermediate a_tilde_t states are detached from the graph.

    Args:
        x_trajectory: The list of detached X_t states from the forward SDE simulation.
                      Length K+1 (X_0 to X_K).
        text_embeddings: Conditional text embeddings.
                         Shape: (batch_size, sequence_length, hidden_size).
        prompts: The list of original text prompts corresponding to the batch.
                 Required for the RewardModel in terminal condition.

    Returns:
        A list of detached torch.Tensor, representing the a_tilde trajectory
        from a_tilde_0 to a_tilde_K, in forward time order. Length K+1.
    """
    a_tilde_trajectory_reversed: List[torch.Tensor] = []

    # Get timesteps for forward SDE path, needed to match X_trajectory indexing
    timesteps_float_forward = self.noise_schedule.get_timesteps_float(
        self.config.fine_tuning.num_timesteps
    )
    # The last timestep in the forward integration is (K-1)*h. The final state is X_K (at t=1.0).
    # X_trajectory has K+1 elements from X_0 to X_K.
    # The lean adjoint terminal condition is at t=1.0.

    # 1. Compute the terminal condition a_tilde_K (at t=1.0)
    a_tilde_K = self.lean_adjoint_solver.compute_a_tilde_K(
        x_trajectory=x_trajectory,
        text_embeddings=text_embeddings,
        timesteps_float=timesteps_float_forward,
        prompts=prompts,
    )
    a_tilde_trajectory_reversed.append(stop_gradient(a_tilde_K))

    # 2. Iterate backward from k = K-1 down to 0 (corresponding to times (K-1)h down to 0)
    for k_idx in range(self.config.fine_tuning.num_timesteps - 1, -1, -1):
      t_current_float = timesteps_float_forward[
          k_idx
      ]  # This is t = k*h in the forward direction
      x_t_current = x_trajectory[k_idx]  # X_k from the forward pass
      a_tilde_next = a_tilde_trajectory_reversed[-1]  # a_tilde_{k+1} (at time (k+1)*h)

      t_current_tensor = torch.tensor(
          [t_current_float], device=self.device, dtype=x_t_current.dtype
      ).repeat(x_t_current.shape[0])

      # Define the drift term for the base model's process whose Jacobian is needed
      # This is `b(X_t, t)` in the general lean adjoint ODE (Eq. 38), which for
      # Flow Matching with memoryless noise is `2 * v_base(X_t, t) - kappa_t * X_t` (Algorithm 1, Eq. 41).
      kappa_t = self.noise_schedule.get_kappa_t(t_current_tensor).view(-1, 1, 1, 1)

      # The input to VJP must have requires_grad=True. X_t_current is detached.
      # Create a clone with requires_grad=True for the VJP calculation.
      x_t_differentiable = x_t_current.clone().detach().requires_grad_(True)

      # Function to compute `2 * v_base(x, t, text_embeddings) - κ_t * x`
      def base_drift_for_jacobian(x: torch.Tensor) -> torch.Tensor:
        return 2.0 * self.base_flow_model.get_velocity(
            x, t_current_tensor, text_embeddings
        ) - kappa_t * x

      # Compute the Vector-Jacobian Product (VJP)
      # This computes `(a_tilde_next)^T * ∇_x(base_drift_for_jacobian(X_t_differentiable))`
      vjp_result = compute_vector_jacobian_product(
          output=base_drift_for_jacobian(x_t_differentiable),
          input_tensor=x_t_differentiable,
          vector=a_tilde_next,
      )

      # Ensure vjp_result has the same dtype as a_tilde_next for addition
      vjp_result = vjp_result.to(a_tilde_next.dtype)

      # Lean adjoint ODE update (Euler step backwards in time)
      # a_tilde_{t-h} = a_tilde_t + h * (a_tilde_t^T * ∇_x b(X_t, t))
      a_tilde_curr = a_tilde_next + self.config.fine_tuning.h_timestep * vjp_result

      a_tilde_trajectory_reversed.append(stop_gradient(a_tilde_curr))

    # Reverse the trajectory to be in forward time order (a_tilde_0 to a_tilde_K)
    a_tilde_trajectory = list(reversed(a_tilde_trajectory_reversed))

    return a_tilde_trajectory

  def _get_gradient_eval_timesteps(self) -> List[float]:
    """
    Selects a subset of timesteps for loss computation based on Appendix G.2.
    It samples 10 timesteps uniformly from the first 72.5% of steps and
    always includes the last 10 timesteps.

    Returns:
        A list of float time values (t = k*h) for which the loss should be evaluated.
    """
    num_timesteps = self.config.fine_tuning.num_timesteps  # K
    h_timestep = self.config.fine_tuning.h_timestep

    # Indices for all timesteps, from 0 to K-1 (inclusive)
    all_k_indices = list(range(num_timesteps))

    # Identify indices for the last 10 timesteps (0.75 to 0.975)
    # K=40 -> 0.75*40=30. So indices 30 to 39.
    last_10_k_indices = list(range(int(0.75 * num_timesteps), num_timesteps))

    # Identify indices for the remaining pool (0 to 0.725*K - 1)
    # K=40 -> 0.725*40=29. So indices 0 to 29.
    remaining_pool_k_indices = list(range(int(0.725 * num_timesteps)))

    # Randomly sample 10 unique indices from the remaining pool
    num_random_samples = min(
        10, len(remaining_pool_k_indices)
    )  # Ensure we don't sample more than available
    sampled_k_indices = random.sample(
        remaining_pool_k_indices, num_random_samples
    )

    # Combine and get unique indices, then sort for consistency
    selected_k_indices = sorted(list(set(last_10_k_indices + sampled_k_indices)))

    # Convert selected indices back to float time values (k * h)
    selected_timesteps_float = [float(k * h_timestep) for k in selected_k_indices]

    return selected_timesteps_float

  def _compute_loss(
      self,
      x_trajectory: List[torch.Tensor],
      a_tilde_trajectory: List[torch.Tensor],
      text_embeddings: torch.Tensor,
      timesteps_to_evaluate: List[float],
  ) -> torch.Tensor:
    """
    Calculates the Adjoint Matching objective with Loss Clipping Threshold (LCT).

    L_Adj-Match(θ) = sum_{t in κ} min(LCT, || (2/σ(t)) * (v_finetune(X_t, t) - v_base(X_t, t)) + σ(t) * a_tilde_t ||^2 )

    Args:
        x_trajectory: The list of detached X_t states from the forward SDE simulation.
                      Length K+1 (X_0 to X_K).
        a_tilde_trajectory: The list of detached a_tilde_t states from the
                            backward adjoint calculation. Length K+1 (a_tilde_0 to a_tilde_K).
        text_embeddings: Conditional text embeddings.
                         Shape: (batch_size, sequence_length, hidden_size).
        timesteps_to_evaluate: A list of float time values (t=k*h) selected for loss computation.

    Returns:
        A scalar torch.Tensor representing the total Adjoint Matching loss for the batch.
    """
    total_loss = torch.tensor(0.0, device=self.device, dtype=text_embeddings.dtype)

    for t_float in timesteps_to_evaluate:
      k_idx = int(t_float / self.config.fine_tuning.h_timestep)

      x_t_batch = x_trajectory[k_idx]
      a_tilde_t_batch = a_tilde_trajectory[k_idx]

      t_tensor = torch.tensor(
          [t_float], device=self.device, dtype=x_t_batch.dtype
      ).repeat(x_t_batch.shape[0])

      # Get memoryless sigma_t for the current time
      sigma_t_batch = self.noise_schedule.get_memoryless_sigma_t(t_tensor).view(
          -1, 1, 1, 1
      )
      
      # For numerical stability: avoid division by extremely small sigma_t
      # The get_memoryless_sigma_t already has self.h offset.
      # So, sigma_t_batch should not be zero. Add a small epsilon as safeguard.
      sigma_t_batch_safe = sigma_t_batch + 1e-8


      # Predict velocity from the fine-tuned model (gradients flow here)
      v_finetune_pred = self.flow_model.get_velocity(
          x_t_batch, t_tensor, text_embeddings
      )

      # Predict velocity from the base model (no gradients flow to base_flow_model)
      # Its parameters are frozen, so it's already detached implicitly.
      v_base_pred = self.base_flow_model.get_velocity(
          x_t_batch, t_tensor, text_embeddings
      )

      # Calculate the core term of the Adjoint Matching loss (Algorithm 1, Eq. 42)
      term_t = (2.0 / sigma_t_batch_safe) * (v_finetune_pred - v_base_pred) + sigma_t_batch * a_tilde_t_batch

      # Compute squared L2 norm and average over batch
      current_loss_term = torch.mean(term_t.square())

      # Apply LCT clipping (Appendix G.3)
      clipped_loss_term = torch.min(current_loss_term, torch.tensor(self.lct_value, device=self.device, dtype=current_loss_term.dtype))

      total_loss += clipped_loss_term

    return total_loss

  def _run_single_iteration(self, batch: Dict[str, Any]) -> torch.Tensor:
    """
    Executes a single fine-tuning iteration for Adjoint Matching.

    This method is called by the `BaseTrainer.train` loop.

    Args:
        batch: A dictionary containing a batch of data from the DataLoader,
               including 'prompts' and 'text_embeddings'.

    Returns:
        A scalar torch.Tensor representing the loss value for the current batch.
    """
    prompts: List[str] = batch["prompt"]
    text_embeddings: torch.Tensor = batch["text_embeddings"].to(self.device)

    # 1. Initialize x_0 with random noise
    latent_channels = self.flow_model.unet.config.in_channels
    latent_size = self.flow_model.unet.config.sample_size
    x_0 = torch.randn(
        text_embeddings.shape[0],
        latent_channels,
        latent_size,
        latent_size,
        device=self.device,
        dtype=text_embeddings.dtype,
    )

    # 2. Simulate Forward SDE Trajectory (X_t)
    x_trajectory = self._forward_pass(x_0, text_embeddings)

    # 3. Compute Lean Adjoint State Trajectory (a_tilde_t)
    a_tilde_trajectory = self._backward_adjoint_pass(
        x_trajectory, text_embeddings, prompts
    )

    # 4. Select Timesteps for Loss Evaluation
    timesteps_to_evaluate = self._get_gradient_eval_timesteps()

    # 5. Compute Adjoint Matching Loss
    loss = self._compute_loss(
        x_trajectory, a_tilde_trajectory, text_embeddings, timesteps_to_evaluate
    )

    return loss

  # Overriding the BaseTrainer's train method to ensure _run_single_iteration is called
  # This makes AdjointMatchingTrainer self-contained as per the design flow
  def train(self) -> FlowMatchingUNet:
    """
    Executes the main fine-tuning loop for the flow model using Adjoint Matching.
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

      # Move text embeddings to the correct device.
      if "text_embeddings" in batch and isinstance(batch["text_embeddings"], torch.Tensor):
        batch["text_embeddings"] = batch["text_embeddings"].to(self.device)

      self.optimizer.zero_grad()

      with torch.cuda.amp.autocast(enabled=(self.grad_scaler is not None)):
        loss = self._run_single_iteration(batch) # Call the AdjointMatching specific iteration logic

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

