"""
Diffusion heads for Hi-MAR.

Phase 1: MLP-based diffusion head (same as MAR).
Phase 2: Diffusion Transformer head (new in Hi-MAR) that uses self-attention
         to model inter-token dependencies.

Both heads implement a diffusion denoising process:
    L(z_i, x_i) = E_{ε,t} [||ε - ε_θ(x_i^t | t, z_i)||^2]

The difference:
- MLP-based: treats each token independently, conditioned on its conditional token z_i
- Transformer-based: processes all tokens together with self-attention, conditioned
  on all conditional tokens Z
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half_dim, device=device).float() / half_dim
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class AdaLN(nn.Module):
    """
    Adaptive Layer Norm for diffusion head conditioning.
    Used in both MLP-based and Transformer-based diffusion heads.
    """
    def __init__(self, hidden_size, condition_dim):
        super().__init__()
        self.linear = nn.Linear(condition_dim, 4 * hidden_size)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, condition):
        params = self.linear(condition)
        alpha1, beta1, alpha2, beta2 = params.chunk(4, dim=-1)
        return alpha1, beta1, alpha2, beta2


class MLPDiffusionHeadBlock(nn.Module):
    """
    MLP-based diffusion head block with adaLN conditioning.
    
    As in Figure 2(d):
    - adaLN
    - LayerNorm
    - Feed-forward (MLP)
    
    Used in the first phase of Hi-MAR.
    """
    def __init__(self, hidden_size, condition_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.adaLN = AdaLN(hidden_size, condition_dim)

    def forward(self, x, condition):
        """
        Args:
            x: (B, N, C) noisy tokens at current diffusion step
            condition: (B, N, C) or (B, C) conditional tokens
        Returns:
            (B, N, C) denoised tokens
        """
        alpha1, beta1, alpha2, beta2 = self.adaLN(condition)
        
        # If condition has same shape as x, use per-token conditioning
        if condition.dim() == 3 and condition.shape[1] == x.shape[1]:
            normed = self.norm1(x)
            normed = alpha1 * normed + beta1
            ffn_out = self.ffn(normed)
            x = x + ffn_out
            
            normed = self.norm2(x)
            normed = alpha2 * normed + beta2
            ffn_out = self.ffn(normed)
            x = x + ffn_out
        else:
            # Global conditioning
            alpha1 = alpha1.unsqueeze(1)
            beta1 = beta1.unsqueeze(1)
            alpha2 = alpha2.unsqueeze(1)
            beta2 = beta2.unsqueeze(1)
            
            normed = self.norm1(x)
            normed = alpha1 * normed + beta1
            ffn_out = self.ffn(normed)
            x = x + ffn_out
            
            normed = self.norm2(x)
            normed = alpha2 * normed + beta2
            ffn_out = self.ffn(normed)
            x = x + ffn_out
        
        return x


class MLPDiffusionHead(nn.Module):
    """
    MLP-based diffusion head for Phase 1.
    
    This is the same as MAR's diffusion head. It predicts noise for each
    token independently using an MLP conditioned on the conditional tokens
    from the Transformer backbone.
    
    Architecture:
    - Timestep embedding
    - Multiple MLPDiffusionHeadBlocks
    - Final linear projection to predict noise
    """
    def __init__(
        self,
        num_layers=6,
        hidden_size=1024,
        latent_dim=16,  # VAE latent channel dim
        condition_dim=None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.condition_dim = condition_dim if condition_dim is not None else hidden_size
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            TimestepEmbedding(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        # Input projection from latent_dim to hidden_size
        self.input_proj = nn.Linear(latent_dim, hidden_size)
        
        # Condition projection (if condition dim != hidden_size)
        if condition_dim is not None and condition_dim != hidden_size:
            self.cond_proj = nn.Linear(condition_dim, hidden_size)
        else:
            self.cond_proj = nn.Identity()
        
        # Diffusion head blocks (Figure 2d)
        self.blocks = nn.ModuleList([
            MLPDiffusionHeadBlock(hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, latent_dim)
        
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Zero out output projection
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x_t, t, cond):
        """
        Args:
            x_t: (B, N, latent_dim) noisy tokens at timestep t
            t: (B,) diffusion timestep
            cond: (B, N, C_cond) conditional tokens from Transformer
        Returns:
            predicted noise: (B, N, latent_dim)
        """
        B, N, _ = x_t.shape
        
        # Embed timestep
        t_emb = self.time_embed(t)  # (B, hidden_size)
        
        # Project input tokens
        h = self.input_proj(x_t)  # (B, N, hidden_size)
        
        # Project condition
        cond_proj = self.cond_proj(cond)  # (B, N, hidden_size)
        
        # Combine condition and timestep
        t_emb = t_emb.unsqueeze(1).expand(-1, N, -1)  # (B, N, hidden_size)
        condition = cond_proj + t_emb  # (B, N, hidden_size)
        
        # Pass through blocks
        for block in self.blocks:
            h = block(h, condition)
        
        # Output noise prediction
        noise_pred = self.output_proj(h)  # (B, N, latent_dim)
        
        return noise_pred


class DiffusionTransformerBlock(nn.Module):
    """
    Diffusion Transformer head block with adaLN and self-attention.
    
    As shown in Figure 2(e):
    - adaLN
    - LayerNorm
    - Self-attention
    - Feed-forward
    
    Unlike MLP-based head, this uses self-attention to model inter-token
    dependencies during the denoising process.
    """
    def __init__(self, hidden_size, num_heads, condition_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        
        self.adaLN = AdaLN(hidden_size, condition_dim)

    def forward(self, x, condition):
        """
        Args:
            x: (B, N, C) token sequence
            condition: (B, C) global condition vector
        Returns:
            (B, N, C) updated tokens
        """
        alpha1, beta1, alpha2, beta2 = self.adaLN(condition)
        
        # Reshape for broadcasting
        alpha1 = alpha1.unsqueeze(1)
        beta1 = beta1.unsqueeze(1)
        alpha2 = alpha2.unsqueeze(1)
        beta2 = beta2.unsqueeze(1)
        
        # First sub-block: Self-attention
        normed = self.norm1(x)
        normed = alpha1 * normed + beta1
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        
        # Second sub-block: FFN
        normed = self.norm2(x)
        normed = alpha2 * normed + beta2
        ffn_out = self.ffn(normed)
        x = x + ffn_out
        
        return x


class DiffusionTransformerHead(nn.Module):
    """
    Diffusion Transformer head for Phase 2.
    
    Unlike the MLP-based head, this head:
    1. Takes ALL tokens (masked + unmasked conditional tokens) as input
    2. Uses self-attention to model inter-token dependencies
    3. Conditions on the sum of timestep embedding and conditional tokens
    
    Architecture (Figure 2e):
    - Timestep embedding
    - Multiple DiffusionTransformerBlocks with self-attention
    - Final linear projection
    
    As noted in the paper, this is only used in the second phase, and
    with much fewer diffusion steps (e.g., 4 steps) due to being heavier.
    """
    def __init__(
        self,
        num_layers=6,
        hidden_size=512,
        num_heads=8,
        latent_dim=16,
        condition_dim=None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.condition_dim = condition_dim if condition_dim is not None else hidden_size
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            TimestepEmbedding(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        # Input projection
        self.input_proj = nn.Linear(latent_dim, hidden_size)
        
        # Condition projection
        if condition_dim is not None and condition_dim != hidden_size:
            self.cond_proj = nn.Linear(condition_dim, hidden_size)
        else:
            self.cond_proj = nn.Identity()
        
        # Diffusion Transformer blocks (Figure 2e)
        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(hidden_size, num_heads, hidden_size)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, latent_dim)
        
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Zero out output projection
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x_t, t, cond, mask_pos=None):
        """
        Args:
            x_t: (B, N, latent_dim) noisy tokens (all tokens, masked + unmasked)
            t: (B,) diffusion timestep
            cond: (B, N, C_cond) conditional tokens from Transformer (for ALL positions)
            mask_pos: (B, N) boolean mask indicating which positions are masked
        Returns:
            predicted noise: (B, N, latent_dim)
        """
        B, N, _ = x_t.shape
        
        # Embed timestep
        t_emb = self.time_embed(t)  # (B, hidden_size)
        
        # Project input tokens
        h = self.input_proj(x_t)  # (B, N, hidden_size)
        
        # Project condition
        cond_proj = self.cond_proj(cond)  # (B, N, hidden_size)
        
        # Global condition: sum of timestep and mean conditional tokens
        # As described in Section 3.3: c = sum(time_embed + conditional_tokens)
        cond_global = cond_proj.mean(dim=1) + t_emb  # (B, hidden_size)
        
        # Pass through Diffusion Transformer blocks
        for block in self.blocks:
            h = block(h, cond_global)
        
        # Output noise prediction
        noise_pred = self.output_proj(h)  # (B, N, latent_dim)
        
        return noise_pred
