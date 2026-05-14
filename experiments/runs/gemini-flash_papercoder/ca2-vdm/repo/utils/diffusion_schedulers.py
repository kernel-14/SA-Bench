import torch
import numpy as np
from typing import Tuple, List

class DiffusionScheduler:
    """
    Manages the diffusion process (noise addition) and denoising steps.
    Implements the core mathematical operations for DDPM and Improved DDPM.
    """

    def __init__(self, num_train_timesteps: int, beta_schedule: str, beta_start: float, beta_end: float):
        """
        Initializes the scheduler with diffusion parameters.

        Args:
            num_train_timesteps (int): The total number of diffusion timesteps (T).
            beta_schedule (str): The type of beta schedule ("linear").
            beta_start (float): The starting value for beta.
            beta_end (float): The ending value for beta.
        """
        self.num_train_timesteps = num_train_timesteps

        if beta_schedule == "linear":
            # Linear interpolation in sqrt space for betas, then square for actual betas.
            # This is a common practice to produce a linear schedule for beta that results
            # in better performance, often used in variants of DDPM.
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32)**2
        else:
            raise ValueError(f"Unsupported beta_schedule: {beta_schedule}. Only 'linear' is supported.")

        self.betas = betas
        
        # Calculations for forward diffusion process (noise addition)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0) # alpha_bar_t
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]], dim=0) # alpha_bar_{t-1}

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for denoising process (sampling)
        # Constants used in predicting x0 from xt and epsilon
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        # This one is used for clarity, though not directly in the standard mean formula
        # self.sqrt_recip_alphas_minus_one = torch.sqrt(1.0 / self.alphas - 1) 

        # Posterior variance (from DDPM/Improved DDPM, often referred to as fixed_small)
        # var = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # Clip the variance to prevent numerical issues, ensuring it's always positive.
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)

        # Ensure all precomputed tensors are on CPU to save GPU memory when not in use.
        # They will be moved to the appropriate device when needed during computations.
        self.betas = self.betas.cpu()
        self.alphas = self.alphas.cpu()
        self.alphas_cumprod = self.alphas_cumprod.cpu()
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.cpu()
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.cpu()
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.cpu()
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.cpu()
        self.posterior_variance = self.posterior_variance.cpu()

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Applies noise to original samples for a given set of timesteps,
        simulating the forward diffusion process.

        Args:
            original_samples (torch.Tensor): The clean latent representation (x_0).
                                             Shape: (batch_size, channels, frames, height, width).
            noise (torch.Tensor): The noise sampled from a standard normal distribution (epsilon).
                                  Shape: (batch_size, channels, frames, height, width).
            timesteps (torch.Tensor): A batch of integer timesteps (t). Shape: (batch_size,).

        Returns:
            torch.Tensor: The noisy latent samples (x_t).
        """
        # Ensure constants are on the same device as input samples
        current_device = original_samples.device
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].to(current_device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].to(current_device)

        # Reshape for broadcasting: (batch_size, 1, 1, 1, 1) for 5D video latents
        # This assumes original_samples is 5D (batch, channels, frames, height, width)
        while len(sqrt_alphas_cumprod_t.shape) < len(original_samples.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        noisy_samples = (
            sqrt_alphas_cumprod_t * original_samples + 
            sqrt_one_minus_alphas_cumprod_t * noise
        )
        return noisy_samples

    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs one denoising step from x_t to x_{t-1} using the predicted noise from the model.
        This implements the "Improved DDPM" denoising step with fixed posterior variance.

        Args:
            model_output (torch.Tensor): The predicted noise (epsilon_theta(x_t, t)) from the Ca2-VDM model.
                                         Shape: (batch_size, channels, frames, height, width).
            timestep (int): The current integer timestep (t).
            sample (torch.Tensor): The noisy latent sample at the current timestep (x_t).
                                   Shape: (batch_size, channels, frames, height, width).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - pred_prev_sample (torch.Tensor): The denoised sample for x_{t-1}.
                - pred_original_sample (torch.Tensor): The predicted x_0.
        """
        current_device = sample.device

        # Retrieve relevant scheduler constants for the given timestep
        # Ensure these are on the correct device for computation
        alpha_t = self.alphas[timestep].to(current_device)
        alphas_cumprod_t = self.alphas_cumprod[timestep].to(current_device)
        beta_t = self.betas[timestep].to(current_device)
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[timestep].to(current_device)
        posterior_variance_t = self.posterior_variance[timestep].to(current_device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timestep].to(current_device)

        # 1. Predict x_0 (original sample)
        # x_0_pred = (x_t - sqrt(1 - alpha_bar_t) * epsilon_theta) / sqrt(alpha_bar_t)
        pred_original_sample = (sample - sqrt_one_minus_alphas_cumprod_t * model_output) / torch.sqrt(alphas_cumprod_t)

        # 2. Compute mean for x_{t-1}
        # mean = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * epsilon_theta)
        mean = sqrt_recip_alpha_t * (sample - beta_t * model_output / sqrt_one_minus_alphas_cumprod_t)

        # 3. Sample x_{t-1}
        if timestep == 0:
            # At the last step, no noise is added
            pred_prev_sample = mean
        else:
            noise = torch.randn_like(sample, device=current_device)
            # x_{t-1} = mean + sqrt(posterior_variance_t) * noise
            pred_prev_sample = mean + torch.sqrt(posterior_variance_t) * noise

        return pred_prev_sample, pred_original_sample

    def get_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Generates a batch of random timesteps for training.

        Args:
            batch_size (int): The number of timesteps to generate.
            device (torch.device): The target device for the generated timesteps.

        Returns:
            torch.Tensor: A tensor of random integer timesteps, shape (batch_size,).
        """
        # Timesteps are 0-indexed, so we sample up to num_train_timesteps - 1
        timesteps = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device, dtype=torch.long)
        return timesteps

    def get_ddim_timesteps(self, num_inference_steps: int) -> List[int]:
        """
        Generates a sequence of timesteps for inference (Improved DDPM style),
        typically a subset of num_train_timesteps for faster sampling.
        The timesteps are generated such that they span the entire diffusion process
        and are in descending order, as required for denoising.

        Args:
            num_inference_steps (int): The desired number of denoising steps for inference.

        Returns:
            List[int]: A list of integer timesteps in descending order.
        """
        # Calculate step size to select timesteps from the total diffusion steps
        # This mimics the behavior of Diffusers' DDIM/DPMSolver schedulers.
        step_ratio = self.num_train_timesteps // num_inference_steps
        
        # Generate timesteps that are roughly evenly spaced
        # Example: if T=1000, num_inference_steps=100, step_ratio=10
        # timesteps_raw = [0, 10, 20, ..., 990]
        timesteps_raw = np.arange(0, self.num_train_timesteps, step_ratio)
        
        # Adjust timesteps to be closer to the end of each interval.
        # Example: [9, 19, 29, ..., 999] for a full span.
        # However, to match the diffusers common way for simplified sequence,
        # we can just take the last timestep (T-1) and then linearly space
        # down to 0, and make sure 0 is included.
        # More robust: use linspace then convert to integers.
        ddim_timesteps = np.linspace(0, self.num_train_timesteps - 1, num_inference_steps, dtype=np.int64)
        
        # Convert to a list and reverse for denoising (from T to 0)
        ddim_timesteps_list = sorted(list(ddim_timesteps), reverse=True)
        
        # Ensure timestep 0 is explicitly included if not already (important for final denoise step)
        if 0 not in ddim_timesteps_list:
            ddim_timesteps_list.append(0)
        
        return ddim_timesteps_list
