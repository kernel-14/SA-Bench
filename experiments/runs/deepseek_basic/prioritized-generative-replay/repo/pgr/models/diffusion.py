"""
Conditional diffusion model for PGR.

Architecture mirrors SYNTHER: a residual MLP denoising diffusion model.
For pixel-based tasks, generates in the latent space of the policy's CNN encoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# ---------- Sinusoidal Position Embeddings ----------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


# ---------- Residual Block ----------
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = F.silu(self.linear1(x))
        x = self.norm2(x)
        x = self.linear2(x)
        return F.silu(x + residual)


# ---------- Conditional Diffusion Model (MLP backbone) ----------
class ConditionalDiffusionModel(nn.Module):
    """
    Conditional diffusion model with MLP backbone.
    
    Takes as input a noised transition (s, a, s', r) and a condition value c,
    along with the diffusion timestep n, and predicts the noise epsilon.
    
    For state-based tasks: input = concat([s, a, s', r])
    For pixel-based tasks: operates in latent space (features from CNN encoder)
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        num_residual_blocks: int = 2,
        cond_dim: int = 1,           # scalar relevance value
        time_emb_dim: int = 256,
        use_latent: bool = False,
        latent_dim: int = 50,        # dimension of CNN latent for pixel-based
        dropout: float = 0.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_latent = use_latent
        
        # Input dimension: s + a + s' + r
        if use_latent:
            # For pixel-based: latent_s + a + latent_s' + r
            self.input_dim = 2 * latent_dim + action_dim + 1
        else:
            self.input_dim = 2 * state_dim + action_dim + 1
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, hidden_dim),
        )
        
        # Condition embedding
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )
        
        # Input projection
        self.input_proj = nn.Linear(self.input_dim, hidden_dim)
        
        # Main MLP with residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, self.input_dim)
        
        # Activation
        self.act = nn.SiLU()

    def forward(
        self, 
        x: torch.Tensor, 
        timestep: torch.Tensor, 
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: noised transition, shape (B, input_dim)
            timestep: diffusion timestep, shape (B,) or scalar
            condition: relevance value, shape (B, cond_dim) or None for unconditional
        Returns:
            predicted noise epsilon, shape (B, input_dim)
        """
        # Time embedding
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        t_emb = self.time_mlp(timestep)
        
        # Project input
        h = self.input_proj(x)
        
        # Add time embedding
        h = h + t_emb
        
        # Add condition embedding (if provided)
        if condition is not None:
            c_emb = self.cond_mlp(condition.float())
            h = h + c_emb
        
        # Residual blocks
        for block in self.blocks:
            h = block(h)
        
        # Output
        h = self.norm(h)
        out = self.output_proj(h)
        
        return out


# ---------- Diffusion Process (DDPM) ----------
class DiffusionProcess:
    """
    Denoising Diffusion Probabilistic Model (DDPM) utilities.
    Handles the forward (noising) and reverse (denoising) processes.
    
    Uses the cosine schedule similar to improved DDPM.
    """
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cuda",
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Linear schedule
        betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        
    def forward_diffusion(
        self, 
        x0: torch.Tensor, 
        t: torch.Tensor, 
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        xt = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise
        return xt, noise

    @torch.no_grad()
    def sample(
        self,
        model: ConditionalDiffusionModel,
        batch_size: int,
        input_dim: int,
        condition: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Reverse diffusion sampling with optional classifier-free guidance.
        
        At each step: epsilon_pred = (1 + w) * epsilon_cond - w * epsilon_uncond
        where w = guidance_scale - 1
        """
        model.eval()
        x = torch.randn(batch_size, input_dim, device=self.device)
        
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            # Conditional prediction
            eps_cond = model(x, t_batch, condition)
            
            if guidance_scale != 1.0 and condition is not None:
                # Unconditional prediction
                eps_uncond = model(x, t_batch, None)
                # CFG: eps = eps_uncond + w * (eps_cond - eps_uncond)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = eps_cond
            
            # DDPM reverse step
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            # x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_cumprod_t) * eps) + sqrt(beta_t) * z
            x = (1.0 / torch.sqrt(alpha)) * (
                x - (1.0 - alpha) / torch.sqrt(1.0 - alpha_cumprod) * eps
            )
            
            if t > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta) * noise
        
        model.train()
        return x


# ---------- Loss Function ----------
def diffusion_loss(
    model: ConditionalDiffusionModel,
    diffusion: DiffusionProcess,
    x0: torch.Tensor,
    condition: Optional[torch.Tensor] = None,
    p_uncond: float = 0.25,
) -> torch.Tensor:
    """
    Compute diffusion training loss with optional CFG conditioning dropout.
    
    Args:
        model: conditional diffusion model
        diffusion: diffusion process
        x0: clean transitions, shape (B, input_dim)
        condition: relevance values, shape (B, cond_dim)
        p_uncond: probability of dropping condition for CFG training
    
    Returns:
        scalar loss
    """
    batch_size = x0.shape[0]
    device = x0.device
    
    # Sample random timesteps
    t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
    
    # Forward diffusion
    noise = torch.randn_like(x0)
    xt, target_noise = diffusion.forward_diffusion(x0, t, noise)
    
    # CFG conditioning dropout
    if condition is not None and p_uncond > 0:
        drop_mask = torch.rand(batch_size, device=device) < p_uncond
        condition_input = condition.clone()
        condition_input[drop_mask] = 0.0  # null condition (zeros)
    else:
        condition_input = condition
    
    # Predict noise
    pred_noise = model(xt, t, condition_input)
    
    # MSE loss
    loss = F.mse_loss(pred_noise, target_noise)
    
    return loss
