import torch
from typing import List, Any, Callable

from config import Config
from models.flow_matching_unet import FlowMatchingUNet
from models.reward_model import RewardModel
from diffusion.noise_schedule import NoiseSchedule
from utils.helpers import stop_gradient, compute_vector_jacobian_product


class LeanAdjointSolver:
  """
  Solves the lean adjoint ODE backward in time to compute the lean adjoint
  trajectory (a_tilde_t). This is a core component of the Adjoint Matching algorithm.

  The lean adjoint ODE is defined in the paper's Section 5.2, Equations 38-39,
  and its discrete form for Flow Matching is in Algorithm 1, Equation 41.
  """

  def __init__(
      self,
      config: Config,
      base_model: FlowMatchingUNet,
      reward_model: RewardModel,
      noise_schedule: NoiseSchedule,
      vae_decoder: Any,  # Expected to be a VAE decoder (e.g., from diffusers.AutoencoderKL)
  ):
    """
    Initializes the LeanAdjointSolver.

    Args:
        config: An instance of Config containing global settings and hyperparameters.
        base_model: The pre-trained base FlowMatchingUNet model (v_base). Used for
                    evaluating the drift term in the adjoint ODE.
        reward_model: The RewardModel used to compute rewards and their gradients
                      for the terminal condition of the adjoint.
        noise_schedule: The NoiseSchedule instance providing time-dependent coefficients.
        vae_decoder: A VAE decoder instance (e.g., from diffusers.AutoencoderKL) to
                     decode latents to pixel space for the reward model.
    """
    if not isinstance(config, Config):
      raise TypeError("config must be an instance of Config.")
    if not isinstance(base_model, FlowMatchingUNet):
      raise TypeError("base_model must be an instance of FlowMatchingUNet.")
    if not isinstance(reward_model, RewardModel):
      raise TypeError("reward_model must be an instance of RewardModel.")
    if not isinstance(noise_schedule, NoiseSchedule):
      raise TypeError("noise_schedule must be an instance of NoiseSchedule.")
    # vae_decoder type check is `Any` because it can be various VAE implementations.

    self.config: Config = config
    self.base_model: FlowMatchingUNet = base_model
    self.reward_model: RewardModel = reward_model
    self.noise_schedule: NoiseSchedule = noise_schedule
    self.vae_decoder: Any = vae_decoder

    self.lambda_reward: float = config.fine_tuning.lambda_reward
    self.h: float = config.fine_tuning.h_timestep
    self.device: str = config.general.device

    # Ensure base_model and vae_decoder are in evaluation mode and on the correct device
    self.base_model.eval()
    self.vae_decoder.eval()
    self.base_model.to(self.device)
    self.vae_decoder.to(self.device)
    # Ensure no gradients are computed for base_model and vae_decoder parameters
    for param in self.base_model.parameters():
      param.requires_grad = False
    for param in self.vae_decoder.parameters():
      param.requires_grad = False

    # The VAE scale factor for Stable Diffusion models
    self.vae_scale_factor: float = 0.18215 # Common value for Stable Diffusion VAEs

  def compute_hat_X_1(
      self,
      X_prev: torch.Tensor,
      t_prev_float: float,
      text_embeddings: torch.Tensor,
  ) -> torch.Tensor:
    """
    Computes hat_X_1, the noiseless final state X_1, by performing a single
    noiseless Euler integration step from X_{1-h} using the v_base model.
    This is used for the terminal condition of the adjoint. (Appendix G.1).

    Args:
        X_prev: The latent state X_t at time t_prev (typically X_{1-h}).
                Shape: (batch_size, channels, H, W).
        t_prev_float: The continuous time value t_prev (typically 1-h).
        text_embeddings: The conditional text embeddings.
                         Shape: (batch_size, sequence_length, hidden_size).

    Returns:
        A torch.Tensor representing hat_X_1, detached from the graph.
        Shape: (batch_size, channels, H, W).
    """
    if not isinstance(X_prev, torch.Tensor) or not isinstance(
        t_prev_float, float
    ) or not isinstance(text_embeddings, torch.Tensor):
      raise TypeError("Invalid input types for compute_hat_X_1.")

    t_tensor = torch.tensor(
        [t_prev_float], device=self.device, dtype=X_prev.dtype
    ).repeat(X_prev.shape[0])

    # Get the base model's velocity prediction at X_prev and t_prev
    # Ensure base_model is used here, not the fine-tuned model
    v_base_at_t_prev = self.base_model.get_velocity(
        X_prev, t_tensor, text_embeddings
    )

    # Compute hat_X_1: X_prev + h * v_base(X_prev, t_prev)
    hat_X_1 = X_prev + self.h * v_base_at_t_prev

    return stop_gradient(hat_X_1)

  def compute_a_tilde_K(
      self,
      X_trajectory: List[torch.Tensor],
      text_embeddings: torch.Tensor,
      timesteps_float: List[float],
      prompts: List[str], # Prompts are needed for ImageReward
  ) -> torch.Tensor:
    """
    Computes the terminal condition a_tilde_K for the lean adjoint ODE.
    a_tilde_K = -λ * ∇_x RewardModel(hat_X_1). (Algorithm 1, Eq. 41).

    Args:
        X_trajectory: The list of X_t states from the forward SDE simulation,
                      expected to be stop_graded.
        text_embeddings: Conditional text embeddings.
        timesteps_float: The discrete time points used for the forward pass.
        prompts: The list of original text prompts corresponding to the batch.
                 Required for the RewardModel.

    Returns:
        A torch.Tensor representing a_tilde_K, detached from the graph.
        Shape: (batch_size, channels, H, W).
    """
    if not isinstance(X_trajectory, list) or not all(
        isinstance(x, torch.Tensor) for x in X_trajectory
    ):
      raise TypeError("X_trajectory must be a list of torch.Tensor.")
    if not isinstance(text_embeddings, torch.Tensor):
      raise TypeError("text_embeddings must be a torch.Tensor.")
    if not isinstance(timesteps_float, list) or not all(
        isinstance(t, float) for t in timesteps_float
    ):
      raise TypeError("timesteps_float must be a list of floats.")
    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
      raise TypeError("prompts must be a list of strings.")

    # Get the last state (X_{1-h}) and its corresponding time from the trajectory
    X_K_prev_h = X_trajectory[-1]
    t_prev_h = timesteps_float[-1]

    # Compute hat_X_1: the noiseless final state
    hat_X_1 = self.compute_hat_X_1(X_K_prev_h, t_prev_h, text_embeddings)

    # Ensure hat_X_1 requires gradients for the reward model prediction
    hat_X_1_differentiable = hat_X_1.clone().detach().requires_grad_(True)

    # Decode latents to pixel space for the reward model
    # VAE decoding often involves scaling the latents. For SD1.5, it's 1/0.18215.
    decoded_images_pixel_space = (
        self.vae_decoder.decode(hat_X_1_differentiable / self.vae_scale_factor).sample()
    )
    # Scale decoded image to [0, 1] range and clamp
    decoded_images_pixel_space = (decoded_images_pixel_space / 2 + 0.5).clamp(0, 1)

    # Predict the reward for hat_X_1. Assuming reward_model.predict takes
    # a batch of images and a list of prompts.
    # The design states predict(image: torch.Tensor, prompt: str). This implies
    # it expects one image and one string. However, for batch processing and
    # ImageReward library, it is usually list of images and list of prompts.
    # We will assume a flexible implementation in RewardModel.predict that can
    # handle either a single image/prompt or a batch. For this context, we pass
    # the batch of decoded images and the list of prompts directly.
    # This implies the RewardModel.predict implementation in models/reward_model.py
    # would be adapted to accept batch inputs.
    rewards = self.reward_model.predict(decoded_images_pixel_space, prompts)

    # Compute the gradient of the rewards with respect to hat_X_1_differentiable
    # The gradient is computed from the sum of rewards for numerical stability and batch processing
    grad_reward_sum = torch.autograd.grad(
        outputs=rewards.sum(),  # Sum to get a scalar for backward
        inputs=hat_X_1_differentiable,
        retain_graph=False,
        create_graph=torch.is_grad_enabled(),
    )[0]

    # Calculate a_tilde_K = -λ * ∇_x RewardModel(hat_X_1)
    a_tilde_K = -self.lambda_reward * grad_reward_sum

    return stop_gradient(a_tilde_K)

  def solve(
      self,
      X_trajectory: List[torch.Tensor],
      text_embeddings: torch.Tensor,
      prompts: List[str], # Prompts are needed for ImageReward in compute_a_tilde_K
      timesteps_float: List[float],
  ) -> List[torch.Tensor]:
    """
    Solves the lean adjoint ODE backward in time to get the full a_tilde trajectory.

    Args:
        X_trajectory: A list of torch.Tensor representing the X_t states
                      from the forward SDE simulation. These should be stop_graded.
                      Length is K+1 (from X_0 to X_K).
        text_embeddings: Conditional text embeddings for the entire batch.
                         Shape: (batch_size, sequence_length, hidden_size).
        prompts: The list of original text prompts corresponding to the batch.
                 Required for the RewardModel.
        timesteps_float: A list of floats representing the discrete time points
                         (e.g., [0.0, h, 2h, ..., 1.0]). Length K+1.

    Returns:
        A list of torch.Tensor, representing the a_tilde trajectory in forward time order.
        Length is K+1.
    """
    if not isinstance(X_trajectory, list) or not all(
        isinstance(x, torch.Tensor) for x in X_trajectory
    ):
      raise TypeError("X_trajectory must be a list of torch.Tensor.")
    if not isinstance(text_embeddings, torch.Tensor):
      raise TypeError("text_embeddings must be a torch.Tensor.")
    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
      raise TypeError("prompts must be a list of strings.")
    if not isinstance(timesteps_float, list) or not all(
        isinstance(t, float) for t in timesteps_float
    ):
      raise TypeError("timesteps_float must be a list of floats.")
    if len(X_trajectory) != self.config.fine_tuning.num_timesteps + 1:
      raise ValueError(f"X_trajectory length {len(X_trajectory)} does not match"
                       f" expected {self.config.fine_tuning.num_timesteps + 1}.")
    if len(timesteps_float) != self.config.fine_tuning.num_timesteps + 1:
      raise ValueError(f"timesteps_float length {len(timesteps_float)} does not match"
                       f" expected {self.config.fine_tuning.num_timesteps + 1}.")


    # Initialize a_tilde_trajectory_reversed for storing results in backward order
    a_tilde_trajectory_reversed: List[torch.Tensor] = []

    # 1. Compute the terminal condition a_tilde_K (at t=1.0)
    # We use X_trajectory[-1] as X_K, and timesteps_float[-1] as t=1.0.
    a_tilde_K = self.compute_a_tilde_K(X_trajectory, text_embeddings, timesteps_float, prompts)
    a_tilde_trajectory_reversed.append(a_tilde_K)

    # 2. Iterate backward from K-1 down to 0
    # X_trajectory has K+1 elements (X_0 to X_K)
    # timesteps_float has K+1 elements (0.0 to 1.0)
    # Loop from k = K-1 down to 0
    for k_idx in range(self.config.fine_tuning.num_timesteps - 1, -1, -1):
      t_current_float = timesteps_float[k_idx]
      X_t_current = X_trajectory[k_idx]  # X_k from the forward pass
      a_tilde_next = a_tilde_trajectory_reversed[-1]  # a_tilde_{k+1}

      # Convert current time to a tensor for model/noise schedule calls
      t_current_tensor = torch.tensor(
          [t_current_float], device=self.device, dtype=X_t_current.dtype
      ).repeat(X_t_current.shape[0])

      # The drift term for the base model's process in the lean adjoint ODE (Algorithm 1, Eq. 41)
      # drift_func = 2 * v_base(X_t, t) - (dot_alpha_t/alpha_t) * X_t
      # This is the `b(X_t, t)` in the general lean adjoint ODE (Eq. 38) when using the
      # specific fine-tuning setup with memoryless noise for Flow Matching.
      # The gradient is wrt X_t_differentiable
      def base_drift_term_for_jacobian(x: torch.Tensor) -> torch.Tensor:
        """
        Helper function to define the drift term `2 * v_base(X_t, t) - κ_t * X_t`
        whose Jacobian is required for the lean adjoint update.
        """
        kappa_t = self.noise_schedule.get_kappa_t(t_current_tensor).view(
            -1, 1, 1, 1
        )
        return 2.0 * self.base_model.get_velocity(
            x, t_current_tensor, text_embeddings
        ) - kappa_t * x

      # Crucially, X_t_current needs to have requires_grad=True for VJP calculation
      # We clone and detach the original X_t from the forward pass, and then enable gradients
      X_t_differentiable = X_t_current.clone().detach().requires_grad_(True)

      # Compute the VJP: (a_tilde_next)^T * ∇_x(base_drift_term_for_jacobian(X_t))
      vjp_result = compute_vector_jacobian_product(
          output=base_drift_term_for_jacobian(X_t_differentiable),
          input_tensor=X_t_differentiable,
          vector=a_tilde_next,
      )
      
      # Ensure vjp_result has the same dtype as a_tilde_next for addition
      vjp_result = vjp_result.to(a_tilde_next.dtype)

      # Lean adjoint ODE update: a_tilde_{t-h} = a_tilde_t + h * (a_tilde_t^T * ∇_x b(X_t, t))
      # In this backward loop, a_tilde_next is a_tilde_t, and a_tilde_curr is a_tilde_{t-h}
      a_tilde_curr = a_tilde_next + self.h * vjp_result

      # Detach from the graph to prevent unwanted gradient flow and manage memory
      a_tilde_trajectory_reversed.append(stop_gradient(a_tilde_curr))

    # Reverse the trajectory to match the forward time order (from X_0 to X_K)
    a_tilde_trajectory = list(reversed(a_tilde_trajectory_reversed))

    return a_tilde_trajectory

