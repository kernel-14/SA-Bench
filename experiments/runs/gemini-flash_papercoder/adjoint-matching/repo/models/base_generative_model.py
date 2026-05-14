## models/base_generative_model.py
import abc
from abc import ABC, abstractmethod
from typing import Callable, Tuple

import torch

from diffusion.noise_schedule import NoiseSchedule


class BaseGenerativeModel(ABC):
  """
  Abstract base class for generative models within the Adjoint Matching framework.

  This class defines a common interface for operations essential to the
  Stochastic Optimal Control (SOC) formulation, such as predicting velocity,
  computing the score function, and calculating the SDE drift term.
  Concrete implementations (e.g., FlowMatchingUNet) will inherit from this class.
  """

  def __init__(self, noise_schedule: NoiseSchedule):
    """
    Initializes the BaseGenerativeModel with a noise schedule.

    Args:
        noise_schedule: An instance of NoiseSchedule providing time-dependent
                        coefficients for SDE calculations.
    """
    if not isinstance(noise_schedule, NoiseSchedule):
      raise TypeError("noise_schedule must be an instance of NoiseSchedule.")
    self.noise_schedule = noise_schedule

  @abstractmethod
  def forward(
      self,
      x: torch.Tensor,
      t: torch.Tensor,
      text_embeddings: torch.Tensor,
  ) -> torch.Tensor:
    """
    Abstract method for the model's forward pass.

    For a Flow Matching model, this should return the predicted velocity v(x, t, cond).
    For a Diffusion model, it might return the predicted noise epsilon(x, t, cond),
    which then needs conversion to velocity/score.

    Args:
        x: The current state (e.g., latent image) at time t.
        t: The current time tensor.
        text_embeddings: Conditional information (e.g., CLIP text embeddings).

    Returns:
        The model's primary prediction output (e.g., velocity).
    """
    pass

  def get_velocity(
      self,
      x: torch.Tensor,
      t: torch.Tensor,
      text_embeddings: torch.Tensor,
  ) -> torch.Tensor:
    """
    Returns the velocity vector field v(x, t, cond) for the given state, time, and conditioning.

    For Flow Matching models, this directly calls the forward method.
    Subclasses for other generative model types (e.g., diffusion) might override this
    to convert their raw output (e.g., noise prediction) into a velocity field.

    Args:
        x: The current state (e.g., latent image) at time t.
        t: The current time tensor.
        text_embeddings: Conditional information (e.g., CLIP text embeddings).

    Returns:
        The velocity vector field v.
    """
    # Default implementation assumes the model directly predicts velocity (Flow Matching).
    return self.forward(x, t, text_embeddings)

  def get_score(
      self,
      x: torch.Tensor,
      t: torch.Tensor,
      text_embeddings: torch.Tensor,
  ) -> torch.Tensor:
    """
    Computes the score function s(x, t) based on the model's predicted velocity.

    Uses the relationship derived for Flow Matching models (Equation 107)
    which links velocity v(x,t) to the score function s(x,t).
    s(x,t) = (1 / eta_t) * (v(x,t) - kappa_t * x)

    Args:
        x: The current state (e.g., latent image) at time t.
        t: The current time tensor.
        text_embeddings: Conditional information (e.g., CLIP text embeddings).

    Returns:
        The score function s(x, t).
    """
    kappa_t = self.noise_schedule.get_kappa_t(t)
    eta_t = self.noise_schedule.get_eta_t(t)

    # Adding a small epsilon for numerical stability in case eta_t becomes very close to zero.
    # The NoiseSchedule's get_eta_t already includes a self.h offset to prevent division by zero
    # at t=0 and stabilize at t=1, so eta_t should generally be non-zero.
    # However, for extremely small eta_t values, this epsilon acts as a safeguard.
    eta_t_stabilized = eta_t + 1e-6 # small epsilon for stability

    velocity = self.get_velocity(x, t, text_embeddings)

    # Equation 107: s(x,t) = (1 / eta_t) * (v(x,t) - kappa_t * x)
    score = (1.0 / eta_t_stabilized) * (velocity - kappa_t * x)
    return score

  def get_drift(
      self,
      x: torch.Tensor,
      t: torch.Tensor,
      text_embeddings: torch.Tensor,
      sigma_t: torch.Tensor,
      s_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
  ) -> torch.Tensor:
    """
    Computes the drift term b(x, t) of the unified SDE (Equation 10-11).

    b(x, t) = kappa_t * x + (sigma_t^2 / 2 + eta_t) * s(x, t)

    Args:
        x: The current state (e.g., latent image) at time t.
        t: The current time tensor.
        text_embeddings: Conditional information (e.g., CLIP text embeddings).
        sigma_t: The diffusion coefficient sigma(t) at time t.
        s_fn: A callable function that computes the score s(x, t, text_embeddings).
              This allows using either the model's own get_score or an external one
              (e.g., from a base model when fine-tuning).

    Returns:
        The drift term b(x, t).
    """
    kappa_t = self.noise_schedule.get_kappa_t(t)
    eta_t = self.noise_schedule.get_eta_t(t)

    score = s_fn(x, t, text_embeddings)

    # Equation 11: b(x, t) = kappa_t * x + (sigma(t)^2 / 2 + eta_t) * s(x, t)
    drift = kappa_t * x + (sigma_t**2 / 2.0 + eta_t) * score
    return drift

