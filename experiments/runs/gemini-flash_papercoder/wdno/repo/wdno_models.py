```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Union, Tuple, Any

# Local imports
from config import Config
from diffusion_model import WaveletDiffusionUNet, NoiseScheduler
from wavelet_utils import WaveletTransformManager
from pde_solvers import PdeSolver  # Assuming PdeSolver is the base class for type hinting
from utils import normalize_data, denormalize_data, get_device # Assuming get_device is in utils


class BaseResolutionModel(nn.Module):
    """
    Base Resolution Model (BRM) for WDNO. Handles the diffusion process
    in the wavelet domain for either simulation or control tasks.
    It wraps a WaveletDiffusionUNet for noise prediction and uses a NoiseScheduler
    for the diffusion steps. For control tasks, it incorporates objective-based guidance.
    """

    def __init__(self,
                 unet: WaveletDiffusionUNet,
                 noise_scheduler: NoiseScheduler,
                 config: Config,
                 wavelet_manager: WaveletTransformManager,
                 problem_mode: str,
                 pde_solver: Optional[PdeSolver] = None):
        """
        Initializes the BaseResolutionModel.

        Args:
            unet: The WaveletDiffusionUNet used for noise prediction.
            noise_scheduler: The NoiseScheduler managing the diffusion process.
            config: The global configuration object.
            wavelet_manager: Manager for wavelet transforms.
            problem_mode: 'simulation' or 'control'.
            pde_solver: An instance of PdeSolver, required for control tasks to evaluate objectives.
        """
        super().__init__()
        self.unet = unet
        self.noise_scheduler = noise_scheduler
        self.config = config
        self.wavelet_manager = wavelet_manager
        self.problem_mode = problem_mode
        self.pde_solver = pde_solver
        self.device = get_device(self.config.device)

        self.ddpm_timesteps = self.config.ddpm_timesteps

        # Classifier-free guidance dropout probability during training
        # Default value as per common practice if not specified in config.
        self.unet_conditional_dropout_probability = getattr(config, 'unet_conditional_dropout_probability', 0.1)

        if self.problem_mode == 'control':
            if self.pde_solver is None:
                raise ValueError("PdeSolver instance is required for control tasks when problem_mode is 'control'.")
            self.guidance_lambda = self.config.guidance_lambda
            print(f"Control mode enabled. Guidance lambda: {self.guidance_lambda}")

    def forward_diffusion_step(self,
                               x_0_wavelets: torch.Tensor,
                               conditions_wavelets: torch.Tensor) -> torch.Tensor:
        """
        Computes the training loss for the UNet using the DDPM objective.

        Args:
            x_0_wavelets: The clean ground-truth data, as a flattened/concatenated
                          tensor of all wavelet coefficients (e.g., W_u for simulation,
                          W_f for control). Shape (B, C_total, D1, D2, ...)
            conditions_wavelets: The conditioning data, as a flattened/concatenated
                                 tensor of wavelet coefficients (e.g., W_a).
                                 Shape (B, C_cond_total, D1, D2, ...)

        Returns:
            The computed MSE loss (scalar tensor).
        """
        batch_size = x_0_wavelets.shape[0]
        device = x_0_wavelets.device

        # Randomly sample a timestep
        t = torch.randint(0, self.ddpm_timesteps, (batch_size,), device=device).long()

        # Generate Gaussian noise
        noise = torch.randn_like(x_0_wavelets)

        # Add noise to x_0_wavelets to get x_t_wavelets
        x_t_wavelets = self.noise_scheduler.add_noise(x_0_wavelets, noise, t)

        # Classifier-Free Guidance Training: Randomly drop out conditioning
        # A mask of 0s means unconditional prediction, 1s means conditional
        # The mask should be boolean for the UNet to interpret it as a conditional flag.
        unet_cond_mask = (torch.rand(batch_size, device=device) > self.unet_conditional_dropout_probability).bool()
        # Reshape for broadcasting with spatial dimensions if necessary by UNet
        # Current design of WaveletDiffusionUNet expects (B,) for mask if it's bool, or (B,1,1,...) if float
        # Let's stick to (B,) boolean and UNet handles it.

        # Predict noise using the UNet
        predicted_noise = self.unet(x_t_wavelets, t, conditions_wavelets, unet_cond_mask=unet_cond_mask)

        # Calculate MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    def sample(self,
               initial_noise: torch.Tensor,
               conditions_wavelets: torch.Tensor, # W_a (for simulation) or W_u0_uT (for control)
               guidance_scale: float = 1.0,
               ddim_steps: Optional[int] = None,
               ddim_eta: Optional[float] = None,
               control_objective_fn: Optional[Callable] = None, # For control guidance
               condition_info_for_control: Optional[Dict[str, Any]] = None # For control guidance, contains u0_w, u_target_w, stats
              ) -> torch.Tensor:
        """
        Generates a sample (simulated trajectory or control sequence) from initial_noise
        using the DDIM reverse process, incorporating classifier-free guidance and
        optional control objective guidance.

        Args:
            initial_noise: The starting Gaussian noise for the DDIM process.
                           Shape (B, C_total, D1, D2, ...)
            conditions_wavelets: Flattened wavelet coefficients of conditioning data.
                                 Shape (B, C_cond_total, D1, D2, ...)
            guidance_scale: Weight for classifier-free guidance (omega in paper).
            ddim_steps: Number of sampling steps. If None, uses config default.
            ddim_eta: Parameter for DDIM. If None, uses config default.
            control_objective_fn: The callable objective function I for control tasks.
                                  Only used if problem_mode is 'control'.
            condition_info_for_control: A dictionary containing necessary info (like u0_wavelets_flat,
                                        u_target_wavelets_flat, and their normalization stats, original shapes)
                                        for calculate_control_guidance_gradient if in control mode.

        Returns:
            The final denoised wavelet coefficients (x_0_wavelets).
        """
        ddim_steps = ddim_steps if ddim_steps is not None else self.config.ddim_steps
        ddim_eta = ddim_eta if ddim_eta is not None else self.config.ddim_eta
        device = initial_noise.device

        current_sample = initial_noise

        # Generate timesteps sequence for DDIM sampling
        times = torch.linspace(self.ddpm_timesteps - 1, 0, ddim_steps + 1, dtype=torch.long, device=device)
        timesteps = list(times.int().cpu().numpy())[:-1] # Exclude t=0 from prediction steps
        time_next = [-1] + list(timesteps[:-1]) # t_next corresponding to each timestep

        # Loop through DDIM sampling steps
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((initial_noise.shape[0],), t, device=device, dtype=torch.long)

            # --- Classifier-Free Guidance ---
            # Predict noise with and without conditioning for classifier-free guidance
            # For inference, we explicitly perform both and combine.
            
            # Unconditional prediction: pass with unet_cond_mask=False
            unconditional_mask_tensor = torch.full((initial_noise.shape[0],), False, device=device).bool()
            predicted_noise_unconditional = self.unet(current_sample, t_tensor, conditions_wavelets, unet_cond_mask=unconditional_mask_tensor)

            # Conditional prediction: pass with unet_cond_mask=True
            conditional_mask_tensor = torch.full((initial_noise.shape[0],), True, device=device).bool()
            predicted_noise_conditional = self.unet(current_sample, t_tensor, conditions_wavelets, unet_cond_mask=conditional_mask_tensor)

            # Combine predictions
            epsilon_combined = predicted_noise_unconditional + guidance_scale * (predicted_noise_conditional - predicted_noise_unconditional)

            # --- Control Guidance (if applicable) ---
            if self.problem_mode == 'control' and self.guidance_lambda > 0 and control_objective_fn is not None and condition_info_for_control is not None:
                # Need to calculate gradient of objective wrt estimated x_0 (control sequence)
                # Note: t is used as the current k in the paper's Algorithm 1 pseudo-code.
                control_gradient = self.calculate_control_guidance_gradient(
                    x_t_wavelets=current_sample,
                    condition_info_for_control=condition_info_for_control,
                    t_ddpm_step=t,
                    control_objective_fn=control_objective_fn,
                    current_pde_solver=self.pde_solver
                )
                # Add guidance term to the predicted noise
                # Equation 4 in paper: -eta * (epsilon_theta + lambda * grad_I)
                # The DDIM step effectively subtracts this combined noise.
                epsilon_combined = epsilon_combined + self.guidance_lambda * control_gradient

            # Perform one DDIM step to update the sample
            current_sample = self.noise_scheduler.step(
                model_output=epsilon_combined,
                timestep=t_tensor,
                sample=current_sample,
                eta=ddim_eta,
                variance_schedule_timestep_next=time_next[i]
            )

        return current_sample

    def calculate_control_guidance_gradient(self,
                                            x_t_wavelets: torch.Tensor, # W_f[0,T]^(k)
                                            condition_info_for_control: Dict[str, Any],
                                            t_ddpm_step: int, # current timestep k
                                            control_objective_fn: Callable,
                                            current_pde_solver: PdeSolver
                                           ) -> torch.Tensor:
        