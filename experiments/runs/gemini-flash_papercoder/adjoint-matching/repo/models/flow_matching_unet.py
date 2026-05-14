## models/flow_matching_unet.py
import torch
from typing import Any, Optional, Callable

from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel # Although not directly used in forward, it clarifies the type of text_encoder

from models.base_generative_model import BaseGenerativeModel
from diffusion.noise_schedule import NoiseSchedule


class FlowMatchingUNet(BaseGenerativeModel):
  """
  A Flow Matching model implemented using a U-Net architecture, inheriting from
  BaseGenerativeModel. This class wraps a diffusers UNet2DConditionModel
  and provides methods to predict velocity, compute score functions, and drift terms
  as required by the Adjoint Matching framework.
  """

  def __init__(
      self,
      unet_config: dict,
      text_encoder: CLIPTextModel,  # Expecting CLIPTextModel for conditioning
      noise_schedule: NoiseSchedule,
      pretrained_path: Optional[str] = None,
      device: str = "cuda",
  ):
    """
    Initializes the FlowMatchingUNet model.

    Args:
        unet_config: A dictionary containing configuration parameters for
                     diffusers.UNet2DConditionModel.
        text_encoder: An instance of a pre-loaded CLIPTextModel for text conditioning.
                      This model is not trained as part of FlowMatchingUNet.
        noise_schedule: An instance of NoiseSchedule to handle time-dependent coefficients.
        pretrained_path: Optional path to load pre-trained weights for the U-Net.
        device: The device to load the model onto ('cuda' or 'cpu').
    """
    super().__init__(noise_schedule)

    # Initialize the UNet model from diffusers
    # The out_channels should be equal to in_channels for velocity prediction
    self.unet = UNet2DConditionModel.from_config(unet_config).to(device)

    # Store the text encoder (it should be in eval mode and not require gradients)
    self.text_encoder = text_encoder
    self.text_encoder.eval()
    for param in self.text_encoder.parameters():
      param.requires_grad = False

    # Load pretrained weights if path is provided
    if pretrained_path:
      self.load_pretrained(pretrained_path)

    self.device = device

  def load_pretrained(self, path: str) -> None:
    """
    Loads pretrained state dictionary into the U-Net model.

    Args:
        path: The file path to the pretrained model's state dictionary.
    """
    if not isinstance(path, str):
      raise TypeError("Pretrained path must be a string.")
    
    # Load to CPU first if CUDA is not available or explicitly specified for loading
    map_location = torch.device(self.device if torch.cuda.is_available() and self.device == "cuda" else "cpu")
    state_dict = torch.load(path, map_location=map_location)
    
    self.unet.load_state_dict(state_dict)
    print(f"Loaded pretrained U-Net weights from {path}")

  def forward(
      self,
      x: torch.Tensor,
      t: torch.Tensor,
      text_embeddings: torch.Tensor,
  ) -> torch.Tensor:
    """
    Performs the forward pass of the Flow Matching U-Net to predict the velocity field.

    Args:
        x: The current state (latent image) at time t. Shape: (batch_size, channels, H, W).
        t: The current time tensor. Shape: (batch_size,). Expected to be in [0, 1].
           The diffusers UNet2DConditionModel typically expects timesteps internally
           to be scaled or mapped to its learned timestep embeddings. Assuming
           `t` is compatible or handled internally by the `diffusers` model.
        text_embeddings: Pre-computed conditional information (CLIP text embeddings).
                         Shape: (batch_size, sequence_length, hidden_size).

    Returns:
        The predicted velocity vector field v(x, t, cond).
        Shape: (batch_size, channels, H, W), same as x.
    """
    if not isinstance(x, torch.Tensor) or not isinstance(t, torch.Tensor) or not isinstance(text_embeddings, torch.Tensor):
      raise TypeError("All inputs must be torch.Tensor.")
    if x.ndim != 4:
      raise ValueError(f"Expected x to be 4-dimensional (B, C, H, W), got {x.ndim}")
    if t.ndim != 1:
      raise ValueError(f"Expected t to be 1-dimensional (B,), got {t.ndim}")
    if text_embeddings.ndim != 3:
      raise ValueError(f"Expected text_embeddings to be 3-dimensional (B, L, H), got {text_embeddings.ndim}")

    # Ensure t is broadcastable to the batch size if it's a scalar
    if t.shape[0] != x.shape[0]:
      if t.numel() == 1:
        t = t.repeat(x.shape[0])
      else:
        raise ValueError(f"Batch size of t ({t.shape[0]}) must match x ({x.shape[0]}) or be a scalar.")

    # The diffusers UNet2DConditionModel's forward method returns a tuple/object
    # where the predicted sample is in the 'sample' attribute.
    # For Flow Matching, this output directly represents the velocity.
    predicted_velocity = self.unet(
        sample=x,
        timestep=t, # Assumes the UNet is compatible with [0,1] timesteps or handles scaling internally
        encoder_hidden_states=text_embeddings,
        return_dict=False # To directly get the tensor output, rather than an object
    )[0] # The output is (sample_output, None) when return_dict=False

    return predicted_velocity

  def get_velocity(
      self,
      x: torch.Tensor,
      t: torch.Tensor,
      text_embeddings: torch.Tensor,
  ) -> torch.Tensor:
    """
    Retrieves the velocity vector field by calling the forward pass of the U-Net.

    This method directly implements the abstract method from BaseGenerativeModel.

    Args:
        x: Current state.
        t: Current time.
        text_embeddings: Conditional text embeddings.

    Returns:
        The predicted velocity field.
    """
    return self.forward(x, t, text_embeddings)

  def get_score(
      self,
      x: torch.Tensor,
      t: torch.Tensor,
      text_embeddings: torch.Tensor,
  ) -> torch.Tensor:
    """
    Computes the score function s(x, t) from the predicted velocity.

    Uses the relationship derived for Flow Matching models (Equation 107 in Appendix B.4):
    s(x,t) = (1 / η_t) * (v(x,t) - κ_t * x)

    Args:
        x: Current state.
        t: Current time.
        text_embeddings: Conditional text embeddings.

    Returns:
        The score function s(x, t, cond).
    """
    # Ensure t is broadcastable to the batch size
    if t.shape[0] != x.shape[0]:
        if t.numel() == 1:
            t_expanded = t.repeat(x.shape[0])
        else:
            raise ValueError(f"Batch size of t ({t.shape[0]}) must match x ({x.shape[0]}) or be a scalar.")
    else:
        t_expanded = t


    v_pred = self.get_velocity(x, t, text_embeddings)
    kappa_t = self.noise_schedule.get_kappa_t(t_expanded).view(-1, 1, 1, 1) # Expand dims for element-wise op
    eta_t = self.noise_schedule.get_eta_t(t_expanded).view(-1, 1, 1, 1) # Expand dims for element-wise op

    # Add a small epsilon for numerical stability. The noise_schedule.get_eta_t
    # already has a stabilization factor `h`, but an additional small epsilon
    # can protect against extremely small floating point values or edge cases.
    eta_t_stabilized = eta_t + 1e-8

    score = (1.0 / eta_t_stabilized) * (v_pred - kappa_t * x)
    return score

  def get_drift(
      self,
      x: torch.Tensor,
      t: torch.Tensor,
      text_embeddings: torch.Tensor, # Added as per updated BaseGenerativeModel signature for s_fn
      sigma_t: torch.Tensor,
      s_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
  ) -> torch.Tensor:
    """
    Computes the drift term b(x, t) of the unified SDE (Equation 10-11).

    b(x, t) = κ_t * x + (σ(t)^2 / 2 + η_t) * s(x, t)

    Args:
        x: Current state.
        t: Current time.
        text_embeddings: Conditional text embeddings.
        sigma_t: The diffusion coefficient σ(t) at time t.
        s_fn: A callable function that computes the score s(x, t, text_embeddings).

    Returns:
        The drift term b(x, t).
    """
    # Ensure t is broadcastable to the batch size
    if t.shape[0] != x.shape[0]:
        if t.numel() == 1:
            t_expanded = t.repeat(x.shape[0])
        else:
            raise ValueError(f"Batch size of t ({t.shape[0]}) must match x ({x.shape[0]}) or be a scalar.")
    else:
        t_expanded = t

    # Ensure sigma_t is broadcastable
    # sigma_t can be a scalar or a tensor with shape (batch_size,)
    if sigma_t.shape[0] != x.shape[0]:
        if sigma_t.numel() == 1:
            sigma_t_expanded = sigma_t.repeat(x.shape[0])
        else:
            raise ValueError(f"Batch size of sigma_t ({sigma_t.shape[0]}) must match x ({x.shape[0]}) or be a scalar.")
    else:
        sigma_t_expanded = sigma_t

    # Expand sigma_t_expanded to match x's dimensions for element-wise operations
    sigma_t_expanded_view = sigma_t_expanded.view(-1, 1, 1, 1)

    kappa_t = self.noise_schedule.get_kappa_t(t_expanded).view(-1, 1, 1, 1)
    eta_t = self.noise_schedule.get_eta_t(t_expanded).view(-1, 1, 1, 1)

    score = s_fn(x, t, text_embeddings)

    drift = kappa_t * x + (sigma_t_expanded_view**2 / 2.0 + eta_t) * score
    return drift

