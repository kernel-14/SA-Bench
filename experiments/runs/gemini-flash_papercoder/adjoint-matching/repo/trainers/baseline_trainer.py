## trainers/baseline_trainer.py
import os
import random
from typing import Dict, Any, List, Tuple, Callable, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from models.flow_matching_unet import FlowMatchingUNet
from models.reward_model import RewardModel
from data.dataset import TextPromptDataset
from diffusion.sde_solver import SDESolver, zero_sigma_fn
from diffusion.noise_schedule import NoiseSchedule
from trainers.base_trainer import BaseTrainer
from utils.helpers import stop_gradient, get_text_embeddings


class BaselineTrainer(BaseTrainer):
  """
  Trainer for various baseline fine-tuning algorithms (ReFL, DPO, Continuous Adjoint, Discrete Adjoint).
  This class dispatches to specific loss computation methods based on the configured baseline type.
  """

  def __init__(
      self,
      config: Config,
      flow_model: FlowMatchingUNet,  # v_finetune
      base_flow_model: FlowMatchingUNet,  # v_base (frozen)
      reward_model: RewardModel,
      dataset: TextPromptDataset,
      sde_solver: SDESolver,
      noise_schedule: NoiseSchedule,
      optimizer: torch.optim.Optimizer,
      baseline_type: str,
      vae_decoder: Any,  # VAE decoder to convert latents to pixel space for RewardModel
      text_encoder: Any,  # CLIP text encoder for DPO sampling conditional embeddings
      tokenizer: Any,  # CLIP tokenizer for DPO sampling conditional embeddings
  ):
    """
    Initializes the BaselineTrainer.

    Args:
        config: The global configuration object.
        flow_model: The FlowMatchingUNet instance representing v_finetune,
                    whose parameters are being optimized.
        base_flow_model: The pre-trained FlowMatchingUNet instance (v_base),
                         which remains frozen.
        reward_model: The RewardModel instance.
        dataset: The TextPromptDataset for training prompts.
        sde_solver: The SDESolver instance for forward trajectory simulation.
        noise_schedule: The NoiseSchedule utility.
        optimizer: The instantiated torch.optim.Optimizer for flow_model.
        baseline_type: A string indicating which baseline method to use
                       ("ReFL", "DPO", "ContinuousAdjoint", "DiscreteAdjoint").
        vae_decoder: A VAE decoder instance (e.g., from diffusers.AutoencoderKL) to
                     decode latents to pixel space for the reward model.
        text_encoder: An instance of a pre-loaded CLIPTextModel for text conditioning.
        tokenizer: An instantiated tokenizer corresponding to the text_encoder.
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

    if baseline_type not in [
        "ReFL",
        "DPO",
        "ContinuousAdjoint",
        "DiscreteAdjoint",
    ]:
      raise ValueError(f"Unknown baseline type: {baseline_type}")

    self.baseline_type: str = baseline_type
    self.base_flow_model: FlowMatchingUNet = base_flow_model
    self.vae_decoder: Any = vae_decoder
    self.text_encoder: Any = text_encoder
    self.tokenizer: Any = tokenizer

    # Ensure base_flow_model and vae_decoder are on correct device and frozen
    self.base_flow_model.to(self.device)
    self.base_flow_model.eval()
    for param in self.base_flow_model.parameters():
      param.requires_grad = False

    self.vae_decoder.to(self.device)
    self.vae_decoder.eval()
    for param in self.vae_decoder.parameters():
      param.requires_grad = False

    # The VAE scale factor for Stable Diffusion models (Appendix G.1 of Adjoint Matching)
    # The value 0.18215 is common for SD's AutoencoderKL.
    self.vae_scale_factor: float = 0.18215

    # Apply baseline-specific optimizer overrides
    baseline_config = self.config.baselines.get_baseline_specific_overrides(
        self.baseline_type
    )
    if baseline_config and baseline_config.learning_rate is not None:
      for param_group in self.optimizer.param_groups:
        param_group["lr"] = baseline_config.learning_rate
      print(
          f"Adjusted learning rate for {self.baseline_type} to"
          f" {baseline_config.learning_rate}"
      )

    # Store effective fine-tuning sigma type for the forward pass simulation
    self.fine_tuning_sigma_type = (
        baseline_config.fine_tuning_sigma_type
        if baseline_config and baseline_config.fine_tuning_sigma_type is not None
        else self.config.fine_tuning.fine_tuning_sigma_type
    )

    # Calculate LCT value for Adjoint baselines if applicable
    if self.baseline_type in ["ContinuousAdjoint", "DiscreteAdjoint"]:
      self.lct_value: float = self.config.fine_tuning.lct_value
      print(f"{self.baseline_type} LCT value: {self.lct_value:.2f}")

    # DPO specific temperature if defined
    self.dpo_lambda_temperature: float = (
        self.config.fine_tuning.lambda_reward
    ) # Using lambda_reward as the beta for DPO as per plan


  def _decode_latents_to_pixels(self, latents: torch.Tensor) -> torch.Tensor:
    """Decodes latent representations to pixel space using the VAE decoder."""
    # Scale and decode
    decoded_images_pixel_space = (
        self.vae_decoder.decode(latents / self.vae_scale_factor).sample()
    )
    # Scale decoded image to [0, 1] range and clamp
    decoded_images_pixel_space = (decoded_images_pixel_space / 2 + 0.5).clamp(0, 1)
    return decoded_images_pixel_space

  def _compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
    """
    Dispatches to the specific loss computation method for the chosen baseline type.

    Args:
        batch: A dictionary containing a batch of data from the DataLoader.
               Expected to include at least 'prompt' and 'text_embeddings'.

    Returns:
        A scalar torch.Tensor representing the loss value for the current batch.
    """
    prompts: List[str] = batch["prompt"]
    text_embeddings: torch.Tensor = batch["text_embeddings"].to(self.device)

    # Initialize x_0 with random noise for the current batch
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

    if self.baseline_type == "ReFL":
      return self._compute_refl_loss(x_0, text_embeddings, prompts)
    elif self.baseline_type == "DPO":
      return self._compute_dpo_loss(x_0, text_embeddings, prompts)
    elif self.baseline_type == "ContinuousAdjoint":
      return self._compute_cont_adjoint_loss(x_0, text_embeddings, prompts)
    elif self.baseline_type == "DiscreteAdjoint":
      return self._compute_disc_adjoint_loss(x_0, text_embeddings, prompts)
    else:
      # Should not happen due to initial check in __init__
      raise ValueError(f"Unknown baseline type: {self.baseline_type}")

  def _get_flow_matching_denoiser_map(
      self,
      x: torch.Tensor,
      t: torch.Tensor,
      text_embeddings: torch.Tensor,
      velocity_model: FlowMatchingUNet,  # Could be self.flow_model or self.base_flow_model
  ) -> torch.Tensor:
    """
    Computes the Flow Matching denoiser map hat_X_1(x, t) as per Equation 229.
    hat_X_1(x, t) = (v(x,t) - (dot_beta_t/beta_t) * x) / (dot_alpha_t - (dot_beta_t/beta_t) * alpha_t)
    """
    # Ensure t is expanded to match batch size if it's a scalar
    if t.shape[0] != x.shape[0]:
      if t.numel() == 1:
        t_expanded = t.repeat(x.shape[0])
      else:
        raise ValueError(
            f"Batch size of t ({t.shape[0]}) must match x ({x.shape[0]})"
            " or be a scalar."
        )
    else:
      t_expanded = t

    v_pred = velocity_model.get_velocity(x, t_expanded, text_embeddings)

    alpha_t = self.noise_schedule.get_alpha_t(t_expanded).view(-1, 1, 1, 1)
    beta_t = self.noise_schedule.get_beta_t(t_expanded).view(-1, 1, 1, 1)
    dot_alpha_t = self.noise_schedule.get_dot_alpha_t(t_expanded).view(-1, 1, 1, 1)
    dot_beta_t = self.noise_schedule.get_dot_beta_t(t_expanded).view(-1, 1, 1, 1)

    # Numerical stability for beta_t and denominator (D_kh)
    beta_t_safe = torch.max(beta_t, torch.tensor(1e-8, device=self.device))
    D_kh = dot_alpha_t - (dot_beta_t / beta_t_safe) * alpha_t
    D_kh_safe = torch.max(D_kh, torch.tensor(1e-8, device=self.device))

    numerator = v_pred - (dot_beta_t / beta_t_safe) * x
    hat_X_1 = numerator / D_kh_safe
    return hat_X_1

  def _compute_refl_loss(
      self, x_0: torch.Tensor, text_embeddings: torch.Tensor, prompts: List[str]
  ) -> torch.Tensor:
    """
    Computes the Reward Feedback Learning (ReFL) loss for Flow Matching models
    (Appendix F.1).
    """
    # 1. Select a random timestep index k
    num_timesteps = self.config.fine_tuning.num_timesteps
    k_idx = random.randint(0, num_timesteps - 1)
    t_float = float(k_idx * self.config.fine_tuning.h_timestep)
    t_tensor = torch.tensor([t_float], device=self.device, dtype=x_0.dtype).repeat(
        x_0.shape[0]
    )

    # 2. Simulate forward trajectory to X_t using the fine-tuning sigma schedule
    sigma_fn = (
        self.noise_schedule.get_memoryless_sigma_t
        if self.fine_tuning_sigma_type == "memoryless"
        else zero_sigma_fn
    )

    # sde_solver.simulate_forward returns X_0, ..., X_K. We need X_k
    timesteps_tensor = self.noise_schedule.get_timesteps_tensor(
        num_timesteps, self.device, x_0.dtype
    )

    # We pass the full timesteps tensor but will only use X_trajectory[k_idx]
    # The default simulate_forward in sde_solver.py does *not* detach.
    # We must detach X_t_sampled as ReFL treats it as a fixed observation.
    full_x_trajectory = self.sde_solver.simulate_forward(
        x_0=x_0,
        text_embeddings=text_embeddings,
        timesteps=timesteps_tensor,
        sigma_fn=sigma_fn,
        h_val=self.config.fine_tuning.h_timestep,
    )
    x_t_sampled = full_x_trajectory[k_idx].detach()  # X_k

    # 3. Compute hat_X_1 (denoiser map) from the fine-tuned model (Eq. 229)
    hat_X_1_finetune = self._get_flow_matching_denoiser_map(
        x=x_t_sampled,
        t=t_tensor,
        text_embeddings=text_embeddings,
        velocity_model=self.flow_model,
    )

    # 4. Decode latents and compute reward
    pixel_images = self._decode_latents_to_pixels(hat_X_1_finetune)
    rewards = self.reward_model.predict(pixel_images, prompts)

    # 5. Calculate loss: negative mean reward
    loss = -self.config.fine_tuning.lambda_reward * rewards.mean()
    return loss

  def _compute_dpo_loss(
      self, x_0: torch.Tensor, text_embeddings: torch.Tensor, prompts: List[str]
  ) -> torch.Tensor:
    """
    Computes the Direct Preference Optimization (DPO) loss adapted for
    Flow Matching models (Appendix F.2).
    """
    # Get unconditional text embeddings for sampling (if CFG is enabled in SDESolver)
    # The SDESolver for trainer should have cfg_weight = 0 for fine-tuning.
    # We generate dummy unconditional embeddings for consistency in SDESolver.sample signature.
    dummy_uncond_embeddings = get_text_embeddings(
        [""],
        self.text_encoder,
        self.tokenizer,
        self.device,
        self.config.model.text_encoder.max_length,
    )

    num_timesteps = self.config.fine_tuning.num_timesteps
    timesteps_tensor_eval = self.noise_schedule.get_timesteps_tensor(
        self.config.evaluation.num_inference_timesteps,
        self.device,
        x_0.dtype,
    )
    h_val_eval = 1.0 / self.config.evaluation.num_inference_timesteps

    # 1. Generate x_1_a and x_1_b using the current fine-tuned model (flow_model)
    # These samples are considered the 'data' for DPO.
    dpo_sampling_sigma_type = (
        self.config.baselines.dpo.evaluation_sampling_sigma_types[0]
        if self.config.baselines.dpo and self.config.baselines.dpo.evaluation_sampling_sigma_types
        else self.config.fine_tuning.evaluation_sampling_sigma_types[0]
    )
    dpo_sigma_fn = (
        self.noise_schedule.get_memoryless_sigma_t
        if dpo_sampling_sigma_type == "memoryless"
        else zero_sigma_fn
    )

    x_1_a = self.sde_solver.sample(
        num_samples=x_0.shape[0],
        text_prompts=prompts,
        unconditional_text_embeddings=dummy_uncond_embeddings.repeat(
            x_0.shape[0], 1, 1
        ),  # Must match batch size if CFG is used
        timesteps=timesteps_tensor_eval,
        sigma_fn=dpo_sigma_fn,
        h_val=h_val_eval,
        text_encoder_tokenizer=(self.text_encoder, self.tokenizer),
        text_encoder_max_length=self.config.model.text_encoder.max_length,
    )
    x_1_b = self.sde_solver.sample(
        num_samples=x_0.shape[0],
        text_prompts=prompts,
        unconditional_text_embeddings=dummy_uncond_embeddings.repeat(
            x_0.shape[0], 1, 1
        ),
        timesteps=timesteps_tensor_eval,
        sigma_fn=dpo_sigma_fn,
        h_val=h_val_eval,
        text_encoder_tokenizer=(self.text_encoder, self.tokenizer),
        text_encoder_max_length=self.config.model.text_encoder.max_length,
    )

    # Detach x_1_a and x_1_b for reward calculation (as they are fixed data)
    x_1_a = x_1_a.detach()
    x_1_b = x_1_b.detach()

    # 2. Sample a random timestep k for the forward process (noise addition)
    k_idx = random.randint(0, num_timesteps - 1)
    kh_float = float(k_idx * self.config.fine_tuning.h_timestep)
    kh_tensor = torch.tensor(
        [kh_float], device=self.device, dtype=x_0.dtype
    ).repeat(x_0.shape[0])

    # 3. Generate x_kh_a and x_kh_b using the forward process (Eq. 2: X_t = beta_t*X_0 + alpha_t*X_1)
    # The X_0 term is random noise.
    alpha_kh = self.noise_schedule.get_alpha_t(kh_tensor).view(-1, 1, 1, 1)
    beta_kh = self.noise_schedule.get_beta_t(kh_tensor).view(-1, 1, 1, 1)

    noise_for_x_kh_a = torch.randn_like(x_1_a).to(self.device)
    noise_for_x_kh_b = torch.randn_like(x_1_b).to(self.device)

    # These are the noisy latents for which the denoiser map (v_theta, v_ref) will try to predict x_1
    x_kh_a = stop_gradient(beta_kh * noise_for_x_kh_a + alpha_kh * x_1_a)
    x_kh_b = stop_gradient(beta_kh * noise_for_x_kh_b + alpha_kh * x_1_b)

    # 4. Compute Rewards for x_1_a and x_1_b (pixel space for reward model)
    r_a = self.reward_model.predict(self._decode_latents_to_pixels(x_1_a), prompts)
    r_b = self.reward_model.predict(self._decode_latents_to_pixels(x_1_b), prompts)

    # 5. Compute denoiser map output for fine-tuned and base models
    hat_x_1_theta_a = self._get_flow_matching_denoiser_map(
        x_kh_a, kh_tensor, text_embeddings, self.flow_model
    )
    hat_x_1_ref_a = self._get_flow_matching_denoiser_map(
        x_kh_a, kh_tensor, text_embeddings, self.base_flow_model
    )
    hat_x_1_theta_b = self._get_flow_matching_denoiser_map(
        x_kh_b, kh_tensor, text_embeddings, self.flow_model
    )
    hat_x_1_ref_b = self._get_flow_matching_denoiser_map(
        x_kh_b, kh_tensor, text_embeddings, self.base_flow_model
    )

    # 6. Compute delta terms (squared L2 norm between predicted x_1 and actual x_1)
    delta_theta_a = F.mse_loss(hat_x_1_theta_a, x_1_a, reduction="none").mean(
        dim=[1, 2, 3]
    )
    delta_ref_a = F.mse_loss(hat_x_1_ref_a, x_1_a, reduction="none").mean(
        dim=[1, 2, 3]
    )
    delta_theta_b = F.mse_loss(hat_x_1_theta_b, x_1_b, reduction="none").mean(
        dim=[1, 2, 3]
    )
    delta_ref_b = F.mse_loss(hat_x_1_ref_b, x_1_b, reduction="none").mean(
        dim=[1, 2, 3]
    )

    # Using lambda_reward as the DPO beta temperature from config
    beta_dpo_loss = self.dpo_lambda_temperature

    # Following a common DPO loss structure for reward models based on likelihood ratios.
    # log_pi_theta(x|y) - log_pi_ref(x|y) is proportional to -0.5 * (||hat_x1_theta - x1||^2 - ||hat_x1_ref - x1||^2)
    log_pi_ratio_theta_ref_a = -0.5 * (delta_theta_a - delta_ref_a)
    log_pi_ratio_theta_ref_b = -0.5 * (delta_theta_b - delta_ref_b)

    loss = -F.logsigmoid(
        beta_dpo_loss * (r_a - r_b)
        - (log_pi_ratio_theta_ref_a - log_pi_ratio_theta_ref_b)
    ).mean()

    return loss

  def _compute_cont_adjoint_loss(
      self, x_0: torch.Tensor, text_embeddings: torch.Tensor, prompts: List[str]
  ) -> torch.Tensor:
    """
    Computes the objective function for the Continuous Adjoint method (Section 5.1.1, Eq. 28).
    This involves simulating the SDE forward and computing running and terminal costs.
    Gradients are implicitly handled by PyTorch's autodiff graph.
    """
    # 1. Determine sigma_fn for the forward SDE simulation
    sigma_fn = (
        self.noise_schedule.get_memoryless_sigma_t
        if self.fine_tuning_sigma_type == "memoryless"
        else zero_sigma_fn
    )

    # 2. Simulate Forward SDE (full trajectory)
    # Crucially, this trajectory must retain its computational graph for backpropagation.
    # SDESolver.simulate_forward will not detach by default.
    num_timesteps = self.config.fine_tuning.num_timesteps
    timesteps_tensor = self.noise_schedule.get_timesteps_tensor(
        num_timesteps, self.device, x_0.dtype
    )
    h_val = self.config.fine_tuning.h_timestep

    # full_x_trajectory includes X_0, X_1, ..., X_K. Total K+1 states.
    # The simulation is from t=0 to t=1-h, producing X_h, X_2h, ..., X_1.
    # So X_trajectory[k] is X_{k*h}.
    full_x_trajectory = self.sde_solver.simulate_forward(
        x_0=x_0,
        text_embeddings=text_embeddings,
        timesteps=timesteps_tensor,
        sigma_fn=sigma_fn,
        h_val=h_val,
    )

    running_cost_sum = torch.tensor(0.0, device=self.device, dtype=x_0.dtype)

    # We need to compute u(X_t,t) at each timestep t_k = k*h
    for k_idx in range(num_timesteps):  # k from 0 to K-1 (for times t=0 to 1-h)
      t_float = float(k_idx * h_val)
      t_tensor = torch.tensor(
          [t_float], device=self.device, dtype=x_0.dtype
      ).repeat(x_0.shape[0])
      x_t_current = full_x_trajectory[k_idx]  # X_{k*h}

      # Compute v_finetune and v_base
      v_finetune_pred = self.flow_model.get_velocity(
          x_t_current, t_tensor, text_embeddings
      )
      v_base_pred = self.base_flow_model.get_velocity(
          x_t_current, t_tensor, text_embeddings
      )

      # Parameters for u(X_t,t) calculation
      eta_t = self.noise_schedule.get_eta_t(t_tensor).view(-1, 1, 1, 1)
      eta_t_safe = torch.max(eta_t, torch.tensor(1e-8, device=self.device))

      # As derived in the Adjoint Matching notes, the term (1/2)||u||^2 simplifies to:
      # (1 / eta_t) * ||v_finetune - v_base||^2 (for Flow Matching with memoryless noise schedule for u)
      u_norm_sq_term_per_batch_element = (1.0 / eta_t_safe) * F.mse_loss(
          v_finetune_pred, v_base_pred, reduction="none"
      ).mean(dim=[1, 2, 3])
      running_cost_sum += u_norm_sq_term_per_batch_element.mean()  # Mean over batch

    # 3. Compute Terminal Cost (g(X_1))
    X_K_final = full_x_trajectory[-1]  # X_K from the simulation

    # Decode latents to pixel space for reward model
    pixel_images_final = self._decode_latents_to_pixels(X_K_final)
    final_rewards = self.reward_model.predict(pixel_images_final, prompts)

    # g(X_1) = -λ * RewardModel(X_1)
    terminal_cost = -self.config.fine_tuning.lambda_reward * final_rewards.mean()

    # 4. Total Loss (running_cost is averaged, so multiply by h_val to approximate integral)
    # Objective (12): min E[∫ (1/2 ||u||^2 + f) dt + g(X_1)] where f=0.
    loss = running_cost_sum * h_val + terminal_cost

    return loss

  def _compute_disc_adjoint_loss(
      self, x_0: torch.Tensor, text_embeddings: torch.Tensor, prompts: List[str]
  ) -> torch.Tensor:
    """
    Computes the objective function for the Discrete Adjoint method (Section 5.1.1).
    This is functionally identical to Continuous Adjoint in terms of loss value
    computation, as PyTorch's autodiff handles the "differentiate-then-discretize"
    (continuous) or "discretize-then-differentiate" (discrete) depending on its internal
    graph representation and optimization. For this implementation, the loss calculation
    is the same.
    """
    # The loss computation logic is identical to Continuous Adjoint.
    # The difference lies in how gradients are handled (e.g., memory management,
    # explicit adjoint ODE solver vs. implicit autodiff) which is typically
    # configured through the auto-differentiation backend (e.g. torch.autograd.grad)
    # or memory optimization techniques like gradient checkpointing.
    # For a direct PyTorch implementation of the loss function itself,
    # the forward pass to compute the loss value is the same.
    return self._compute_cont_adjoint_loss(x_0, text_embeddings, prompts)

