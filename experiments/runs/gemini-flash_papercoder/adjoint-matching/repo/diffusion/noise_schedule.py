## diffusion/noise_schedule.py
import torch
from typing import Union


class NoiseSchedule:
  """
  Manages the time-dependent coefficients and noise schedules for Flow Matching models.

  Based on the unified SDE formulation and memoryless noise schedule
  described in the "Adjoint Matching" paper.
  """

  def __init__(self, num_timesteps: int):
    """
    Initializes the NoiseSchedule with the given number of discrete timesteps.

    Args:
        num_timesteps: The number of discrete timesteps (K) for the SDE integration.
    """
    if not isinstance(num_timesteps, int) or num_timesteps <= 0:
      raise ValueError("num_timesteps must be a positive integer.")
    self.num_timesteps: int = num_timesteps
    self.h: float = 1.0 / self.num_timesteps  # Timestep size (delta t)

  def get_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
    """
    Returns the alpha_t coefficient for the reference flow.

    For Flow Matching, alpha_t is typically t.

    Args:
        t: A torch.Tensor representing time values (0 to 1).

    Returns:
        A torch.Tensor of alpha_t values, matching the shape of t.
    """
    return t

  def get_beta_t(self, t: torch.Tensor) -> torch.Tensor:
    """
    Returns the beta_t coefficient for the reference flow.

    For Flow Matching, beta_t is typically 1 - t.

    Args:
        t: A torch.Tensor representing time values (0 to 1).

    Returns:
        A torch.Tensor of beta_t values, matching the shape of t.
    """
    return 1.0 - t

  def get_dot_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
    """
    Returns the time derivative of alpha_t (d(alpha_t)/dt).

    For Flow Matching with alpha_t = t, d(alpha_t)/dt = 1.

    Args:
        t: A torch.Tensor representing time values (0 to 1).

    Returns:
        A torch.Tensor of d(alpha_t)/dt values, matching the shape of t.
    """
    return torch.ones_like(t)

  def get_dot_beta_t(self, t: torch.Tensor) -> torch.Tensor:
    """
    Returns the time derivative of beta_t (d(beta_t)/dt).

    For Flow Matching with beta_t = 1 - t, d(beta_t)/dt = -1.

    Args:
        t: A torch.Tensor representing time values (0 to 1).

    Returns:
        A torch.Tensor of d(beta_t)/dt values, matching the shape of t.
    """
    return -torch.ones_like(t)

  def get_kappa_t(self, t: torch.Tensor) -> torch.Tensor:
    """
    Returns the kappa_t coefficient for the unified SDE drift.

    Defined as kappa_t = d(alpha_t)/dt / alpha_t.
    With alpha_t = t, this is 1/t.
    A numerical stability adjustment is applied as per Appendix G.1.

    Args:
        t: A torch.Tensor representing time values (0 to 1).

    Returns:
        A torch.Tensor of kappa_t values, matching the shape of t.
    """
    # Numerical stability: t=0 would cause division by zero.
    # The paper uses h as an offset in sigma(t) denominator,
    # we apply similar logic here for kappa_t for consistency.
    # The paper effectively maps continuous-time t to discrete k*h.
    # If t is exactly 0, replace with h to avoid division by zero.
    # For t in (0, 1], (t + self.h) might not be strictly necessary as t itself is not zero.
    # However, to avoid 1/epsilon for very small t, we can use a small offset.
    # The key point from G.1 is sigma(t) = sqrt(2 * (1-t+h)/(t+h)),
    # which implies an effective eta_t = (1-t+h)/(t+h).
    # Kappa_t = 1/t. The stability adjustment for sigma implies t -> t+h.
    # Let's align directly with the stable sigma(t) form derivation.
    return 1.0 / (t + self.h)

  def get_eta_t(self, t: torch.Tensor) -> torch.Tensor:
    """
    Returns the eta_t coefficient for the unified SDE drift.

    Defined as eta_t = beta_t * (d(alpha_t)/dt / alpha_t * beta_t - d(beta_t)/dt).
    With alpha_t = t and beta_t = 1-t, this simplifies to (1-t)/t.
    A numerical stability adjustment is applied consistent with the paper's stable sigma(t).

    Args:
        t: A torch.Tensor representing time values (0 to 1).

    Returns:
        A torch.Tensor of eta_t values, matching the shape of t.
    """
    # From paper, for Flow Matching: eta_t = (1-t)/t
    # The stable sigma(t) in Appendix G.1 is sqrt(2 * (1 - t + h) / (t + h)).
    # This implies the effective eta_t for the memoryless schedule is (1 - t + h) / (t + h).
    # We use this adjusted form for get_eta_t to ensure consistency with sigma(t) calculation.
    return (1.0 - t + self.h) / (t + self.h)

  def get_memoryless_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
    """
    Returns the diffusion coefficient sigma_t for the memoryless noise schedule.

    This schedule is crucial for fine-tuning as specified in Section 4.3 and Theorem 1.
    The stable form from Appendix G.1 is implemented: sigma(t) = sqrt(2 * (1 - t + h) / (t + h)).

    Args:
        t: A torch.Tensor representing time values (0 to 1).

    Returns:
        A torch.Tensor of memoryless sigma_t values, matching the shape of t.
    """
    # The paper's derived formula is sigma(t) = sqrt(2 * eta_t).
    # However, for practical stability, Appendix G.1 provides:
    # sigma(t) = sqrt(2 * (1 - t + h) / (t + h))
    # where h is the timestep size.
    # This implicitly adjusts the theoretical eta_t to (1 - t + h) / (t + h).
    numerator = 2.0 * (1.0 - t + self.h)
    denominator = t + self.h
    return torch.sqrt(numerator / denominator)

