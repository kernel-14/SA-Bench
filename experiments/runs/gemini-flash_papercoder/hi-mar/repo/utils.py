import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Callable, List, Optional
from copy import deepcopy

# --- Masking Utilities ---

def _mask_tokens(
    tokens: torch.Tensor,
    strategy_name: str,
    masking_params: Tuple[float, ...],
    mask_token_embedding: torch.Tensor,
    return_mask: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies masking to a batch of tokens according to a specified strategy.

    Args:
        tokens: A tensor of shape (B, N, D_latent) representing the visual tokens.
        strategy_name: The masking strategy to use ("random_uniform", "cosine", "beta_distribution").
        masking_params: Parameters specific to the strategy (e.g., [min_ratio, max_ratio] or [alpha, beta]).
        mask_token_embedding: A tensor of shape (D_latent) representing the learned embedding
                              to replace masked tokens.
        return_mask: If True, returns the boolean mask indicating masked positions.

    Returns:
        A tuple (masked_tokens, is_masked_bool).
        masked_tokens: Tensor of shape (B, N, D_latent) with selected tokens replaced.
        is_masked_bool: Boolean tensor of shape (B, N) indicating masked positions.
    """
    batch_size, seq_len, embed_dim = tokens.shape
    device = tokens.device

    # Determine masking ratio 'r'
    if strategy_name == "random_uniform":
        min_ratio, max_ratio = masking_params
        r = (torch.rand(1, device=device) * (max_ratio - min_ratio) + min_ratio).item()
    elif strategy_name == "cosine":
        # For training, MaskGIT's 'cosine' strategy samples a random ratio for *which*
        # tokens to mask. The cosine scheduling is more for inference steps.
        # So we interpret this as a uniform random ratio.
        r = torch.rand(1, device=device).item()
    elif strategy_name == "beta_distribution":
        alpha, beta = masking_params
        m = torch.distributions.Beta(torch.tensor([alpha], device=device), torch.tensor([beta], device=device)).sample()
        r = m.item()
    else:
        raise ValueError(f"Unknown masking strategy: {strategy_name}")

    num_masked = int(r * seq_len)
    if num_masked == 0:
        return tokens.clone(), torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    is_masked_bool = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    for i in range(batch_size):
        # Randomly select indices to mask
        masked_indices = torch.randperm(seq_len, device=device)[:num_masked]
        is_masked_bool[i, masked_indices] = True

    # Expand mask_token_embedding to match batch and sequence dimensions for torch.where
    # The mask_token_embedding is (D_latent)
    # Unsqueeze to (1, 1, D_latent) then expand to (B, N, D_latent) for broadcasting with `where`
    expanded_mask_token = mask_token_embedding.view(1, 1, embed_dim).expand(batch_size, seq_len, embed_dim)

    # Use torch.where to replace tokens where is_masked_bool is True
    # is_masked_bool needs to be unsqueezed to (B, N, 1) to broadcast correctly
    masked_tokens = torch.where(is_masked_bool.unsqueeze(-1), expanded_mask_token, tokens)

    return masked_tokens, is_masked_bool


# --- Diffusion Noise Schedule Utilities ---

def get_noise_scheduler(
    schedule_type: str,
    num_train_timesteps: int = 1000,
    ddpm_beta_start: float = 0.0001,
    ddpm_beta_end: float = 0.02
) -> Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns a callable function that provides noise schedule parameters for given timesteps.

    Args:
        schedule_type: Type of schedule ("cosine", "linear").
        num_train_timesteps: Total number of timesteps in the schedule.
        ddpm_beta_start: Beta start value for linear schedule.
        ddpm_beta_end: Beta end value for linear schedule.

    Returns:
        A callable function (timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        that returns (sqrt_alpha_prod_t, sqrt_one_minus_alpha_prod_t) for the given timesteps.
    """
    if schedule_type == "cosine":
        # Following 'Improved Denoising Diffusion Probabilistic Models' (Nichol & Dhariwal, 2021)
        s = 0.008
        t = torch.linspace(0, num_train_timesteps, num_train_timesteps + 1, dtype=torch.float64)
        f_t = torch.cos(((t / num_train_timesteps) + s) / (1 + s) * math.pi / 2)**2
        alphas_cumprod = f_t / f_t[0]

        # Ensure betas are within a reasonable range for stability
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.999) # Clip betas

        alphas = 1 - betas
        # Recalculate alphas_cumprod with clipped betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

    elif schedule_type == "linear":
        betas = torch.linspace(
            ddpm_beta_start, ddpm_beta_end, num_train_timesteps, dtype=torch.float64
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
    else:
        raise ValueError(f"Unknown noise schedule type: {schedule_type}")

    sqrt_alpha_prod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alpha_prod = torch.sqrt(1.0 - alphas_cumprod)

    def noise_scheduler_fn(timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves noise schedule parameters for specified timesteps.
        Timesteps are expected to be 0-indexed.
        """
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long)
        
        # Clamp timesteps to ensure they are within valid range [0, num_train_timesteps-1]
        timesteps = torch.clamp(timesteps, 0, num_train_timesteps - 1)

        _sqrt_alpha_prod_t = sqrt_alpha_prod[timesteps].to(timesteps.device)
        _sqrt_one_minus_alpha_prod_t = sqrt_one_minus_alpha_prod[timesteps].to(timesteps.device)
        return _sqrt_alpha_prod_t, _sqrt_one_minus_alpha_prod_t

    return noise_scheduler_fn

def _add_noise_to_latents(
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies noise to original latent tokens according to a specified noise schedule.

    Args:
        latents: Clean latent tokens of shape (B, N, D_latent).
        timesteps: Timesteps tensor of shape (B,) or scalar.
        noise_scheduler: A callable function from get_noise_scheduler.

    Returns:
        A tuple (noisy_latents, epsilon).
        noisy_latents: Noisy latent tokens of shape (B, N, D_latent).
        epsilon: The sampled noise of shape (B, N, D_latent).
    """
    epsilon = torch.randn_like(latents)
    sqrt_alpha_prod_t, sqrt_one_minus_alpha_prod_t = noise_scheduler(timesteps)

    # Reshape for broadcasting: (B,) -> (B, 1, 1) if latents is (B, N, D_latent)
    sqrt_alpha_prod_t = sqrt_alpha_prod_t.view(-1, 1, 1)
    sqrt_one_minus_alpha_prod_t = sqrt_one_minus_alpha_prod_t.view(-1, 1, 1)

    noisy_latents = sqrt_alpha_prod_t * latents + sqrt_one_minus_alpha_prod_t * epsilon
    return noisy_latents, epsilon

def _predict_original_from_noise(
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    predicted_noise: torch.Tensor,
    noise_scheduler: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
) -> torch.Tensor:
    """
    Estimates the original clean latent tokens from noisy latents and predicted noise.

    Args:
        noisy_latents: Noisy latent tokens of shape (B, N, D_latent).
        timesteps: Timesteps tensor of shape (B,) or scalar.
        predicted_noise: Model's predicted noise of shape (B, N, D_latent).
        noise_scheduler: A callable function from get_noise_scheduler.

    Returns:
        Estimated clean latent tokens of shape (B, N, D_latent).
    """
    sqrt_alpha_prod_t, sqrt_one_minus_alpha_prod_t = noise_scheduler(timesteps)

    # Reshape for broadcasting
    sqrt_alpha_prod_t = sqrt_alpha_prod_t.view(-1, 1, 1)
    sqrt_one_minus_alpha_prod_t = sqrt_one_minus_alpha_prod_t.view(-1, 1, 1)

    pred_original_sample = (noisy_latents - sqrt_one_minus_alpha_prod_t * predicted_noise) / sqrt_alpha_prod_t
    return pred_original_sample

# --- Adaptive Normalization Layers ---

class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN) module.
    Modulates Layer Normalization with scale and shift parameters derived from a context vector.
    Used in MLPDiffusionHead and DiffusionTransformerHead.
    """
    def __init__(self, hidden_size: int, context_embedding_dim: int):
        """
        Initializes the AdaLN module.

        Args:
            hidden_size: The dimension of the input features.
            context_embedding_dim: The dimension of the context vector.
        """
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.proj_params = nn.Linear(context_embedding_dim, 2 * hidden_size) # For scale and shift

        # Initialize projection weights to zero for identity transformation initially
        nn.init.zeros_(self.proj_params.weight)
        nn.init.zeros_(self.proj_params.bias)

    def forward(self, x: torch.Tensor, context_vector: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for AdaLN.

        Args:
            x: Input tensor to be normalized and modulated (B, ..., hidden_size).
            context_vector: Context vector (e.g., timestep embedding) (B, context_embedding_dim).

        Returns:
            Output tensor after AdaLN (B, ..., hidden_size).
        """
        # Apply standard Layer Normalization without learnable affine parameters
        norm_x = self.norm(x)

        # Project context vector to get scale and shift parameters
        scale_shift = self.proj_params(context_vector)
        # Split into scale and shift
        scale, shift = scale_shift.chunk(2, dim=-1)

        # Reshape scale and shift for broadcasting
        # Assuming x has shape (B, N, D) or (B, N_tokens, D)
        # scale, shift should become (B, 1, D) for broadcasting
        num_spatial_dims = len(x.shape) - len(scale.shape)
        scale = scale.view(scale.shape[0], *(1,) * num_spatial_dims, scale.shape[-1])
        shift = shift.view(shift.shape[0], *(1,) * num_spatial_dims, shift.shape[-1])
        
        # Apply modulation: (1 + scale) * x_normalized + shift
        # (1 + scale) ensures it starts near identity for residual connections
        output = (1 + scale) * norm_x + shift
        return output

class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN-Zero) module with zero-initialization for residual scaling.
    This version produces six parameters (alpha1, beta1, gamma1, alpha2, beta2, gamma2)
    to modulate two sub-blocks (e.g., Attention and FFN) and their residual connections.
    Used in HiMARTransformerBlock.
    """
    def __init__(self, hidden_size: int, context_embedding_dim: int):
        """
        Initializes the AdaLNZero module.

        Args:
            hidden_size: The dimension of the input features to the transformer block.
            context_embedding_dim: The dimension of the context vector (e.g., scale embedding).
        """
        super().__init__()
        # Projects context_vector into 6 parameters: alpha1, beta1, gamma1 for attention block
        # and alpha2, beta2, gamma2 for FFN block.
        # Each parameter is of size hidden_size.
        self.proj_params = nn.Linear(context_embedding_dim, 6 * hidden_size)

        # Initialize projection weights and biases to zero, so gamma1 and gamma2 are initially 0.
        # This makes the residual connections effectively identity at initialization (AdaLN-Zero).
        nn.init.zeros_(self.proj_params.weight)
        nn.init.zeros_(self.proj_params.bias)

    def forward(self, context_vector: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for AdaLNZero.

        Args:
            context_vector: Context vector (B, context_embedding_dim).

        Returns:
            A tuple of six tensors: (alpha1, beta1, gamma1, alpha2, beta2, gamma2),
            each of shape (B, hidden_size).
        """
        # Project context vector to get all parameters
        params = self.proj_params(context_vector)
        
        # Split params into 6 chunks
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = params.chunk(6, dim=-1)

        # Apply (1 + alpha) as per DiT/paper's formula, which starts alpha at 0.
        # The paper's formula is `alpha * LN(x) + beta` which implies `(1+alpha)`
        # is baked into the interpretation of alpha, or 'alpha' here represents
        # the deviation from 1. Given DiT context, `1 + alpha` is standard.
        alpha1 = 1 + alpha1
        alpha2 = 1 + alpha2
        
        return alpha1, beta1, gamma1, alpha2, beta2, gamma2


# --- EMA Model ---

class EMAModel(nn.Module):
    """
    Maintains an Exponential Moving Average (EMA) of a PyTorch model's weights.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """
        Initializes the EMAModel.

        Args:
            model: The nn.Module whose weights are to be averaged.
            decay: The EMA momentum value (0.9999 for MS-COCO from config).
        """
        super().__init__()
        self.decay = decay
        self.ema_model = deepcopy(model)
        self.ema_model.requires_grad_(False) # EMA model does not need gradients
        self.ema_model.eval() # Set to eval mode, dropout/batchnorm are off

        # Ensure all parameters are on the same device as the original model
        self.ema_model.to(next(model.parameters()).device)

    def update(self, model: nn.Module) -> None:
        """
        Updates the EMA model's weights.

        Args:
            model: The current training model.
        """
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.copy_(self.decay * ema_param + (1. - self.decay) * model_param)
            
            # Also update buffers (e.g., BatchNorm running_mean, running_var)
            for ema_buffer, model_buffer in zip(self.ema_model.buffers(), model.buffers()):
                ema_buffer.copy_(model_buffer) # Buffers are usually directly copied, not averaged

    def swap_parameters(self, model: nn.Module) -> None:
        """
        Temporarily swaps the parameters of the original model with the EMA model.
        Useful for evaluating with EMA weights without changing the training model.

        Args:
            model: The current training model. Its parameters will be swapped with EMA.
        """
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                # Swap data, using a temporary buffer
                temp = ema_param.data.clone()
                ema_param.data.copy_(model_param.data)
                model_param.data.copy_(temp)

            # Swap buffers too
            for ema_buffer, model_buffer in zip(self.ema_model.buffers(), model.buffers()):
                temp = ema_buffer.data.clone()
                ema_buffer.data.copy_(model_buffer.data)
                model_buffer.data.copy_(temp)

