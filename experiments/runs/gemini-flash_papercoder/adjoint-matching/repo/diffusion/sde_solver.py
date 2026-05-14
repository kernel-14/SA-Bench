## diffusion/sde_solver.py
import torch
from typing import List, Any, Callable, Tuple

from models.flow_matching_unet import FlowMatchingUNet
from diffusion.noise_schedule import NoiseSchedule
from utils.helpers import get_text_embeddings


class SDESolver:
  """
  Solves Stochastic Differential Equations (SDEs) and Ordinary Differential Equations (ODEs)
  using Euler-Maruyama discretization. It handles both forward trajectory generation
  for fine-tuning and sampling for evaluation, including Classifier-Free Guidance.
  """

  def __init__(
      self,
      generative_model: FlowMatchingUNet,
      noise_schedule: NoiseSchedule,
      cfg_weight: float = 0.0,
      device: str = "cuda",
  ):
    """
    Initializes the SDESolver.

    Args:
        generative_model: The model predicting the velocity field v(x,t,cond).
                          During fine-tuning, this is v_finetune.
                          During evaluation, it's the fine-tuned model.
        noise_schedule: An instance providing time-dependent coefficients.
        cfg_weight: The guidance scale 'w' for Classifier-Free Guidance. Defaults to 0.0.
        device: The computational device ('cuda' or 'cpu').
    """
    if not isinstance(generative_model, FlowMatchingUNet):
      raise TypeError(
          "generative_model must be an instance of FlowMatchingUNet."
      )
    if not isinstance(noise_schedule, NoiseSchedule):
      raise TypeError("noise_schedule must be an instance of NoiseSchedule.")
    if not isinstance(cfg_weight, (float, int)) or cfg_weight < 0:
      raise ValueError("cfg_weight must be a non-negative float.")
    if not isinstance(device, str):
      raise TypeError("device must be a string.")

    self.generative_model: FlowMatchingUNet = generative_model.to(device)
    self.noise_schedule: NoiseSchedule = noise_schedule
    self.cfg_weight: float = cfg_weight
    self.device: str = device

  def _get_effective_drift_for_finetuning(
      self,
      x_t: torch.Tensor,
      t: torch.Tensor,
      v_pred: torch.Tensor,
  ) -> torch.Tensor:
    """
    Computes the drift term for the Euler-Maruyama discretization used in
    the forward simulation for fine-tuning. This corresponds to the drift
    of the controlled SDE under the memoryless noise schedule, as given in
    Algorithm 1, Equation 40 and Section 4.3 (Memoryless Flow Matching).

    Drift = 2 * v_finetune(X_t, t) - kappa_t * X_t

    Args:
        x_t: The current latent state. Shape: (batch_size, channels, H, W).
        t: The current time point. Shape: (batch_size,) or scalar.
        v_pred: The velocity predicted by self.generative_model (v_finetune).
                Shape: (batch_size, channels, H, W).

    Returns:
        The drift tensor.
    """
    # Ensure t is expanded to match batch size if it's a scalar
    if t.shape[0] != x_t.shape[0]:
      if t.numel() == 1:
        t_expanded = t.repeat(x_t.shape[0])
      else:
        raise ValueError(
            f"Batch size of t ({t.shape[0]}) must match x_t ({x_t.shape[0]})"
            " or be a scalar."
        )
    else:
      t_expanded = t

    kappa_t = self.noise_schedule.get_kappa_t(t_expanded).view(-1, 1, 1, 1)

    # From Algorithm 1, Eq. 40 (rearranged):
    # X_{t+h} = X_t + h * (2 * v_theta_finetune(X_t, t) - (dot_alpha_t/alpha_t) * X_t) + ...
    # So, drift = 2 * v_finetune(X_t, t) - kappa_t * X_t
    drift = 2.0 * v_pred - kappa_t * x_t
    return drift

  def _get_effective_drift_for_sampling(
      self,
      x_t: torch.Tensor,
      t: torch.Tensor,
      v_pred: torch.Tensor,
      sigma_t: torch.Tensor,
  ) -> torch.Tensor:
    """
    Computes the effective drift term for sampling from the fine-tuned model.
    This implements the drift b(x,t) from the unified SDE (Equations 10-11) by
    substituting s(x,t) with its expression in terms of v(x,t) (Equation 107).

    Drift = (1 + sigma_t^2 / (2 * eta_t)) * v_pred - (sigma_t^2 / (2 * eta_t)) * kappa_t * x_t
    This is derived from Eq. (4) which is an SDE for Flow Matching with arbitrary sigma(t).

    Args:
        x_t: The current latent state. Shape: (batch_size, channels, H, W).
        t: The current time point. Shape: (batch_size,) or scalar.
        v_pred: The (CFG-applied) velocity prediction from the fine-tuned model.
                Shape: (batch_size, channels, H, W).
        sigma_t: The diffusion coefficient currently used for sampling (can be 0
                 for ODE, or sqrt(2*eta_t) for memoryless).
                 Shape: (batch_size,) or scalar.

    Returns:
        The drift tensor.
    """
    # Ensure t and sigma_t are expanded to match batch size if they are scalars
    if t.shape[0] != x_t.shape[0]:
      if t.numel() == 1:
        t_expanded = t.repeat(x_t.shape[0])
      else:
        raise ValueError(
            f"Batch size of t ({t.shape[0]}) must match x_t ({x_t.shape[0]})"
            " or be a scalar."
        )
    else:
      t_expanded = t

    if sigma_t.shape[0] != x_t.shape[0]:
      if sigma_t.numel() == 1:
        sigma_t_expanded = sigma_t.repeat(x_t.shape[0])
      else:
        raise ValueError(
            f"Batch size of sigma_t ({sigma_t.shape[0]}) must match x_t"
            f" ({x_t.shape[0]}) or be a scalar."
        )
    else:
      sigma_t_expanded = sigma_t

    kappa_t = self.noise_schedule.get_kappa_t(t_expanded).view(-1, 1, 1, 1)
    eta_t = self.noise_schedule.get_eta_t(t_expanded).view(-1, 1, 1, 1)
    sigma_t_sq = sigma_t_expanded.view(-1, 1, 1, 1) ** 2

    # Numerical stability for eta_t. While get_eta_t incorporates 'h',
    # a minimal epsilon ensures safety in extreme floating point cases.
    eta_t_safe = torch.max(eta_t, torch.tensor(1e-8, device=self.device))

    term_factor = sigma_t_sq / (2.0 * eta_t_safe)

    # Derived from Eq. (4) with v(x,t) substituted:
    # dX_t = ( v(X_t, t) + (sigma(t)^2 / (2 * eta_t)) * (v(X_t, t) - kappa_t * X_t) ) dt + sigma(t) dB_t
    # drift = v_pred * (1 + term_factor) - term_factor * kappa_t * x_t
    drift = (1.0 + term_factor) * v_pred - term_factor * kappa_t * x_t
    return drift

  def simulate_forward(
      self,
      x_0: torch.Tensor,
      text_embeddings: torch.Tensor,
      timesteps: torch.Tensor,
      sigma_fn: Callable[[torch.Tensor], torch.Tensor],
      h_val: float,
  ) -> List[torch.Tensor]:
    """
    Generates a sequence of latent states X_t by numerically integrating
    the controlled SDE forward in time, starting from initial noise X_0.
    This trajectory is used by the AdjointMatchingTrainer to compute the loss.

    Args:
        x_0: The initial latent state (noise N(0,I)).
             Shape: (batch_size, channels, H, W).
        text_embeddings: The conditional text embeddings.
                         Shape: (batch_size, seq_len, embed_dim).
        timesteps: A 1D tensor of discrete time points (e.g., [0, h, 2h, ..., (K-1)h]).
                   These are the 't' values for the SDE.
        sigma_fn: A function sigma(t) to get the diffusion coefficient for time t.
                  For fine-tuning, this will typically be noise_schedule.get_memoryless_sigma_t.
        h_val: The fixed time step size (e.g., 1/K_finetune).

    Returns:
        A list of torch.Tensor, representing the trajectory of X_t states.
    """
    if not isinstance(x_0, torch.Tensor) or not isinstance(
        text_embeddings, torch.Tensor
    ) or not isinstance(timesteps, torch.Tensor) or not isinstance(
        sigma_fn, Callable
    ) or not isinstance(h_val, (float, int)):
      raise TypeError("Invalid input types for simulate_forward.")

    x_t_current = x_0.to(self.device)
    trajectory: List[torch.Tensor] = [x_t_current]

    for t_idx, t_current in enumerate(timesteps):
      t_current = t_current.to(self.device) # Ensure time is on device

      # Get velocity prediction from the fine-tuning model (v_finetune)
      v_pred = self.generative_model(x_t_current, t_current, text_embeddings)

      # Calculate drift for fine-tuning forward pass (Eq. 40 from Algorithm 1)
      drift = self._get_effective_drift_for_finetuning(
          x_t_current, t_current, v_pred
      )

      # Get the diffusion coefficient for the current timestep
      sigma_t_current = sigma_fn(t_current).view(
          -1, 1, 1, 1
      )  # Expand dims for element-wise op

      # Generate Gaussian noise
      epsilon_noise = torch.randn_like(x_t_current).to(self.device)

      # Euler-Maruyama update
      x_t_next = x_t_current + h_val * drift + torch.sqrt(
          torch.tensor(h_val, device=self.device, dtype=x_0.dtype)
      ) * sigma_t_current * epsilon_noise
      
      # For numerical stability if needed, though Flow Matching generally
      # doesn't require clipping intermediate states like diffusion models do.
      # x_t_next = torch.clamp(x_t_next, -10.0, 10.0) # Example clipping

      trajectory.append(x_t_next)
      x_t_current = x_t_next

    return trajectory

  def sample(
      self,
      num_samples: int,
      text_prompts: List[str],
      unconditional_text_embeddings: torch.Tensor,
      timesteps: torch.Tensor,
      sigma_fn: Callable[[torch.Tensor], torch.Tensor],
      h_val: float,
      text_encoder_tokenizer: Tuple[Any, Any], # (text_encoder, tokenizer)
      text_encoder_max_length: int
  ) -> torch.Tensor:
    """
    Generates images/latents by numerically integrating the SDE/ODE forward in time.
    This method is used for evaluation and supports Classifier-Free Guidance.

    Args:
        num_samples: The number of samples to generate.
        text_prompts: A list of strings, one for each sample, to condition generation.
        unconditional_text_embeddings: Pre-computed unconditional text embeddings
                                       (e.g., from an empty string). Shape: (1, seq_len, embed_dim).
        timesteps: A 1D tensor of discrete time points for inference.
        sigma_fn: A function sigma(t) to get the diffusion coefficient for sampling.
                  (e.g., noise_schedule.get_memoryless_sigma_t or zero_sigma_fn).
        h_val: The fixed time step size for inference.
        text_encoder_tokenizer: A tuple (text_encoder, tokenizer) required for
                                generating conditional text embeddings.
        text_encoder_max_length: Max length for text tokenization.

    Returns:
        A torch.Tensor of the final latent states (generated samples).
        Shape: (num_samples, channels, H, W).
    """
    if not isinstance(num_samples, int) or num_samples <= 0:
      raise ValueError("num_samples must be a positive integer.")
    if not isinstance(text_prompts, list) or not all(
        isinstance(p, str) for p in text_prompts
    ):
      raise TypeError("text_prompts must be a list of strings.")
    if not isinstance(unconditional_text_embeddings, torch.Tensor) or unconditional_text_embeddings.ndim != 3:
      raise TypeError("unconditional_text_embeddings must be a 3D tensor.")
    if not isinstance(timesteps, torch.Tensor) or timesteps.ndim != 1:
      raise TypeError("timesteps must be a 1D tensor.")
    if not isinstance(sigma_fn, Callable) or not isinstance(h_val, (float, int)):
      raise TypeError("Invalid input types for sample.")
    if not isinstance(text_encoder_tokenizer, Tuple) or len(text_encoder_tokenizer) != 2:
      raise TypeError("text_encoder_tokenizer must be a tuple of (encoder, tokenizer).")
    if not isinstance(text_encoder_max_length, int) or text_encoder_max_length <= 0:
      raise ValueError("text_encoder_max_length must be a positive integer.")

    text_encoder, tokenizer = text_encoder_tokenizer

    # Generate conditional text embeddings
    conditional_text_embeddings = get_text_embeddings(
        prompts=text_prompts,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        device=self.device,
        max_length=text_encoder_max_length
    )

    # Initialize x_t_current with random noise
    # Assumes latent space dimensions are consistent with the generative model's output
    latent_channels = self.generative_model.unet.config.in_channels
    latent_size = self.generative_model.unet.config.sample_size
    x_t_current = torch.randn(
        num_samples,
        latent_channels,
        latent_size,
        latent_size,
        device=self.device,
        dtype=conditional_text_embeddings.dtype # Use the same dtype as embeddings for consistency
    )

    with torch.no_grad(): # Sampling should not compute gradients
      for t_idx, t_current in enumerate(timesteps):
        t_current = t_current.to(self.device) # Ensure time is on device

        # Get conditional velocity
        v_cond = self.generative_model(x_t_current, t_current, conditional_text_embeddings)

        v_pred = v_cond
        # Apply Classifier-Free Guidance if cfg_weight > 0
        if self.cfg_weight > 0.0:
          # Expand unconditional embeddings to match batch size
          uncond_embeddings_expanded = unconditional_text_embeddings.repeat(num_samples, 1, 1)
          v_uncond = self.generative_model(x_t_current, t_current, uncond_embeddings_expanded)
          v_pred = self.cfg_velocity(v_cond, v_uncond)

        # Get the diffusion coefficient for the current timestep
        sigma_t_current = sigma_fn(t_current).view(-1, 1, 1, 1) # Expand dims for element-wise op

        # Calculate drift for sampling (Eq. 4 from paper)
        drift = self._get_effective_drift_for_sampling(x_t_current, t_current, v_pred, sigma_t_current)

        # Generate Gaussian noise. If sigma_t is 0 (ODE sampling), noise term becomes 0.
        epsilon_noise = torch.randn_like(x_t_current).to(self.device)

        # Euler-Maruyama update
        x_t_next = x_t_current + h_val * drift + torch.sqrt(
            torch.tensor(h_val, device=self.device, dtype=x_t_current.dtype)
        ) * sigma_t_current * epsilon_noise
        
        x_t_current = x_t_next

    return x_t_current

  def cfg_velocity(
      self,
      v_cond: torch.Tensor,
      v_uncond: torch.Tensor,
  ) -> torch.Tensor:
    """
    Computes the Classifier-Free Guidance (CFG) adjusted velocity.

    Formula: (1 + w) * v_cond - w * v_uncond (Ho and Salimans, 2022)

    Args:
        v_cond: The conditional velocity field prediction.
        v_uncond: The unconditional velocity field prediction.

    Returns:
        The CFG-adjusted velocity.
    """
    if self.cfg_weight == 0.0:
      return v_cond
    return (1.0 + self.cfg_weight) * v_cond - self.cfg_weight * v_uncond


def zero_sigma_fn(t: torch.Tensor) -> torch.Tensor:
  """
  A utility function to return a tensor of zeros, used as sigma_fn
  when sampling with sigma(t)=0 (ODE sampling).

  Args:
      t: A torch.Tensor representing time values.

  Returns:
      A torch.Tensor of zeros, matching the shape of t.
  """
  return torch.zeros_like(t)

