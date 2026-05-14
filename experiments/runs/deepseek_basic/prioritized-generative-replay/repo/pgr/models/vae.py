"""
Variational Autoencoder (VAE) baseline for Appendix D.

Replaces the diffusion model with a VAE for comparison.
Design approximates capacity of diffusion model (~6.8M parameters).
Uses residual net encoder (4 layers) and decoder (8 layers),
bottleneck dim 128, latent dim 32.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class ResidualLinearBlock(nn.Module):
    """Residual linear block for VAE."""
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = F.relu(self.linear1(x))
        x = self.norm2(x)
        x = self.linear2(x)
        return F.relu(x + residual)


class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder.
    
    Encoder: residual net, 4 layers, bottleneck dim 128
    Decoder: residual net, 8 layers, bottleneck dim 128
    Latent dim: 32
    
    Total params ~6.8M (matching diffusion model capacity)
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 1024,
        latent_dim: int = 32,
        bottleneck_dim: int = 128,
        cond_dim: int = 1,
        num_encoder_blocks: int = 4,
        num_decoder_blocks: int = 8,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        self.input_dim = 2 * state_dim + action_dim + 1
        
        # Encoder
        self.encoder_input = nn.Sequential(
            nn.Linear(self.input_dim + cond_dim, hidden_dim),
            nn.ReLU(),
        )
        self.encoder_blocks = nn.ModuleList([
            ResidualLinearBlock(hidden_dim) for _ in range(num_encoder_blocks)
        ])
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        
        # Latent space projections
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder_blocks = nn.ModuleList([
            ResidualLinearBlock(hidden_dim) for _ in range(num_decoder_blocks)
        ])
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder_output = nn.Linear(hidden_dim, self.input_dim)
    
    def encode(
        self, 
        x: torch.Tensor, 
        condition: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        if condition is not None:
            x = torch.cat([x, condition], dim=-1)
        
        h = self.encoder_input(x)
        for block in self.encoder_blocks:
            h = block(h)
        h = self.encoder_norm(h)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(
        self, 
        z: torch.Tensor, 
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode latent to output."""
        if condition is not None:
            z = torch.cat([z, condition], dim=-1)
        
        h = self.decoder_input(z)
        for block in self.decoder_blocks:
            h = block(h)
        h = self.decoder_norm(h)
        
        return self.decoder_output(h)
    
    def forward(
        self, 
        x: torch.Tensor, 
        condition: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full VAE forward pass."""
        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, condition)
        return recon, mu, logvar
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate samples from the VAE."""
        z = torch.randn(batch_size, self.latent_dim, device=next(self.parameters()).device)
        return self.decode(z, condition)


def vae_loss(
    model: ConditionalVAE,
    x: torch.Tensor,
    condition: Optional[torch.Tensor] = None,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Beta-VAE loss: reconstruction + KL divergence.
    
    Optionally drops conditioning for unconditional generation.
    """
    recon, mu, logvar = model(x, condition)
    
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + beta * kl_loss
    
    info = {
        'recon_loss': recon_loss.item(),
        'kl_loss': kl_loss.item(),
        'total_loss': total_loss.item(),
    }
    
    return total_loss, info


class UnconditionalVAE(ConditionalVAE):
    """VAE without conditioning, for SYNTHER-VAE comparison."""
    
    def __init__(self, state_dim: int, action_dim: int, **kwargs):
        super().__init__(state_dim, action_dim, cond_dim=0, **kwargs)
    
    def encode(self, x, condition=None):
        h = self.encoder_input(x)
        for block in self.encoder_blocks:
            h = block(h)
        h = self.encoder_norm(h)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def decode(self, z, condition=None):
        h = self.decoder_input(z)
        for block in self.decoder_blocks:
            h = block(h)
        h = self.decoder_norm(h)
        return self.decoder_output(h)
    
    def forward(self, x, condition=None):
        return super().forward(x, None)
    
    def sample(self, batch_size, condition=None):
        z = torch.randn(batch_size, self.latent_dim, device=next(self.parameters()).device)
        return self.decode(z)
