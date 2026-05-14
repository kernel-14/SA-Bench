"""
Hi-MAR Transformer backbone with Scale-Aware Transformer blocks.

The Hi-MAR Transformer uses adaLN-Zero operations to inject scale-specific
information, allowing the same backbone to handle two resolution scales
(low-resolution and high-resolution) with explicit scale awareness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal position/scale embedding."""
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half_dim, device=device).float() / half_dim
        )
        args = x.float().unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Norm Zero: learns scale, shift, and gate parameters
    conditioned on some input (e.g., scale embedding or timestep embedding).
    
    In Hi-MAR, this is used both for scale-aware Transformer blocks
    and for diffusion head conditioning.
    """
    def __init__(self, hidden_size, condition_dim):
        super().__init__()
        self.linear = nn.Linear(condition_dim, 6 * hidden_size)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, condition):
        params = self.linear(condition)
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = params.chunk(6, dim=-1)
        return alpha1, beta1, gamma1, alpha2, beta2, gamma2


class ScaleAwareTransformerBlock(nn.Module):
    """
    Scale-aware Transformer block with adaLN-Zero for scale conditioning.
    
    As described in Section 3.2 and Figure 2(c):
    
    Given input z^i:
        v_tilde = a * v + b
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = split(v_tilde)
        z_a = z^i + gamma1 * Attention(alpha1 * LN(z^i) + beta1)
        z^{i+1} = z_a + gamma2 * FFN(alpha2 * LN(z_a) + beta2)
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, condition_dim=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        
        # Self-attention
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        
        # Feed-forward network
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )
        
        # adaLN-Zero: projects condition to 6 params
        if condition_dim is not None:
            self.adaLN = AdaLNZero(hidden_size, condition_dim)
        else:
            self.adaLN = None

    def forward(self, x, condition=None):
        if self.adaLN is not None and condition is not None:
            alpha1, beta1, gamma1, alpha2, beta2, gamma2 = self.adaLN(condition)
            alpha1 = alpha1.unsqueeze(1)
            beta1 = beta1.unsqueeze(1)
            gamma1 = gamma1.unsqueeze(1)
            alpha2 = alpha2.unsqueeze(1)
            beta2 = beta2.unsqueeze(1)
            gamma2 = gamma2.unsqueeze(1)
        else:
            alpha1 = beta1 = gamma1 = alpha2 = beta2 = gamma2 = 1.0
        
        # First sub-block: Attention
        normed = self.norm1(x)
        if self.adaLN is not None and condition is not None:
            normed = alpha1 * normed + beta1
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + gamma1 * attn_out
        
        # Second sub-block: FFN
        normed = self.norm2(x)
        if self.adaLN is not None and condition is not None:
            normed = alpha2 * normed + beta2
        ffn_out = self.ffn(normed)
        x = x + gamma2 * ffn_out
        
        return x


class HiMARTransformer(nn.Module):
    """
    Hi-MAR Transformer backbone.
    
    This is a scale-aware Transformer that processes both low-resolution
    and high-resolution token sequences. Scale information is injected 
    via sinusoidal embeddings through adaLN-Zero operations in each block.
    """
    def __init__(
        self,
        num_layers=24,
        hidden_size=768,
        num_heads=12,
        mlp_ratio=4.0,
        max_seq_len=1024,
        vocab_size=None,
        input_dim=16,  # VAE latent dimension
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input projection (from input_dim to hidden_size)
        self.input_proj = nn.Linear(input_dim, hidden_size)
        
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, hidden_size) * 0.02)
        
        # Scale embedding: sinusoidal embedding + MLP
        self.scale_embed_dim = hidden_size
        self.scale_sinusoidal = SinusoidalEmbedding(hidden_size)
        self.scale_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        # Class embedding (for class-conditional generation)
        self.num_classes = vocab_size
        if vocab_size is not None:
            self.class_embed = nn.Embedding(vocab_size, hidden_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            ScaleAwareTransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                condition_dim=hidden_size,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        
        # Final adaLN for output
        self.final_adaLN = AdaLNZero(hidden_size, hidden_size)
        
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.elementwise_affine:
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def get_scale_vector(self, scale_idx):
        scale_tensor = torch.tensor([scale_idx], device=self.pos_embed.device)
        scale_embed = self.scale_sinusoidal(scale_tensor)
        scale_vec = self.scale_mlp(scale_embed)
        return scale_vec

    def forward(self, x, scale_idx=0, class_idx=None, context_embeds=None):
        """
        Args:
            x: (B, N, input_dim) input token sequence (latent tokens)
            scale_idx: int, 0 for low-resolution, 1 for high-resolution
            class_idx: (B,) class indices
            context_embeds: (B, N_ctx, C) optional context embeddings
        Returns:
            conditional_tokens: (B, N+N_ctx+1, hidden_size) output
        """
        B, N, C_in = x.shape
        
        # Project input to hidden_size
        x = self.input_proj(x)
        
        # Add positional embeddings
        x = x + self.pos_embed[:, :N, :]
        
        # Add class embeddings if provided
        if class_idx is not None and self.num_classes is not None:
            class_emb = self.class_embed(class_idx).unsqueeze(1)
            x = torch.cat([class_emb, x], dim=1)
        
        # Add context embeddings if provided
        if context_embeds is not None:
            x = torch.cat([context_embeds, x], dim=1)
        
        # Get scale vector
        scale_vec = self.get_scale_vector(scale_idx)
        scale_vec = scale_vec.expand(B, -1)
        
        # Pass through scale-aware Transformer blocks
        for block in self.blocks:
            x = block(x, condition=scale_vec)
        
        # Final layer norm with adaLN
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = self.final_adaLN(scale_vec)
        x = gamma1.unsqueeze(1) * self.final_norm(alpha1.unsqueeze(1) * x + beta1.unsqueeze(1))
        
        return x
