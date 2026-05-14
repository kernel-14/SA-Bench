import torch
import torch.nn as nn
from .layers import (
    SelfAttention,
    MLPBlock,
    ScaleEmbedding,
    TimestepEmbedding,
    AdaLN,
)


class ScaleAwareTransformerBlock(nn.Module):
    """
    Scale-Aware Transformer Block (Figure 2c).
    Input: z^i, scale vector v.
    v → linear(a·v + b) → split → α1,β1,γ1,α2,β2,γ2.
    z^i → LN(z^i) → α1·LN(z^i) + β1 → Attention → * γ1 → residual add → z_a.
    z_a → LN(z_a) → α2·LN(z_a) + β2 → FFN → * γ2 → residual add → z^{i+1}.
    """
    def __init__(self, dim: int, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(dim, elementwise_affine=False)
        # Projects scale vector to 6 modulation parameters per token
        self.scale_mod = nn.Linear(dim, dim * 6)
        # Zero-initialize for identity-like behavior at start
        nn.init.constant_(self.scale_mod.weight, 0)
        nn.init.constant_(self.scale_mod.bias, 0)

        self.attn = SelfAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.mlp = MLPBlock(dim=dim, expansion_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor, scale_emb: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        params = self.scale_mod(scale_emb)
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = params.chunk(6, dim=-1)
        # Broadcast to match token dimension
        while alpha1.dim() < x.dim():
            alpha1 = alpha1.unsqueeze(1)
            beta1 = beta1.unsqueeze(1)
            gamma1 = gamma1.unsqueeze(1)
            alpha2 = alpha2.unsqueeze(1)
            beta2 = beta2.unsqueeze(1)
            gamma2 = gamma2.unsqueeze(1)

        x_ln1 = self.ln1(x)
        modulated = alpha1 * x_ln1 + beta1
        x_a = x + gamma1 * self.attn(modulated, attn_mask=attn_mask)

        x_ln2 = self.ln2(x_a)
        modulated2 = alpha2 * x_ln2 + beta2
        x_out = x_a + gamma2 * self.mlp(modulated2)
        return x_out


class HiMARTransformer(nn.Module):
    """
    Hierarchical Masked Autoregressive Transformer backbone.
    Bidirectional self-attention. Shared across both phases; scale index provides phase-awareness.
    Context tokens (class/text) are prepended externally to the token sequence.
    """
    def __init__(
        self,
        dim: int = 768,
        depth: int = 24,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # Scale embedding: sinusoidal embed → MLP → scale vector v (per-phase)
        self.scale_embed = ScaleEmbedding(embedding_dim=256, hidden_dim=dim, output_dim=dim)

        self.blocks = nn.ModuleList([
            ScaleAwareTransformerBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        scale_idx: int,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Generate scale-aware modulation vector for this phase
        scale_emb = self.scale_embed(scale_idx, x.device)  # [dim]
        for block in self.blocks:
            x = block(x, scale_emb, attn_mask=attn_mask)
        x = self.final_norm(x)
        return x


class MLPDiffusionHeadBlock(nn.Module):
    """MLP-based Diffusion head block (Figure 2d). Uses AdaLN with timestep+conditional embedding."""
    def __init__(self, dim: int, cond_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.adaln = AdaLN(dim=dim, cond_dim=cond_dim)
        self.mlp = MLPBlock(dim=dim, expansion_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.adaln(x, cond)
        return self.mlp(x)


class MLPDiffusionHead(nn.Module):
    """
    MLP-based Diffusion Head (Figure 2d).
    Used in phase 1. Each masked token is processed independently with shared weights.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 1024,
        depth: int = 6,
        cond_out_dim: int = 768,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_out_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.t_embed = TimestepEmbedding(embedding_dim=256, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            MLPDiffusionHeadBlock(dim=hidden_dim, cond_dim=hidden_dim)
            for _ in range(depth)
        ])
        self.out_proj = nn.Linear(hidden_dim, in_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_embed(t)
        cond_emb = self.cond_proj(cond)
        c = t_emb + cond_emb
        h = self.in_proj(x_t)
        for block in self.blocks:
            h = block(h, c)
        return self.out_proj(h)


class DiffTransformerBlock(nn.Module):
    """
    Diffusion Transformer Head block (Figure 2e).
    c = timestep + conditional tokens → split → α1,β1,γ1,α2,β2,γ2.
    y_a = y^i + γ1 · Attention(α1 · LN(y^i) + β1)
    y^{i+1} = y_a + γ2 · FFN(α2 · LN(y_a) + β2)
    """
    def __init__(self, dim: int, cond_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Linear(cond_dim, dim * 6)
        nn.init.constant_(self.modulation.weight, 0)
        nn.init.constant_(self.modulation.bias, 0)

        self.attn = SelfAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.mlp = MLPBlock(dim=dim, expansion_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        params = self.modulation(cond)  # [B, N, dim*6] or [B, 1, dim*6]
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = params.chunk(6, dim=-1)
        while alpha1.dim() < x.dim():
            alpha1 = alpha1.unsqueeze(1)
            beta1 = beta1.unsqueeze(1)
            gamma1 = gamma1.unsqueeze(1)
            alpha2 = alpha2.unsqueeze(1)
            beta2 = beta2.unsqueeze(1)
            gamma2 = gamma2.unsqueeze(1)

        x_ln1 = self.ln1(x)
        x_a = x + gamma1 * self.attn(alpha1 * x_ln1 + beta1, attn_mask=attn_mask)

        x_ln2 = self.ln2(x_a)
        x_out = x_a + gamma2 * self.mlp(alpha2 * x_ln2 + beta2)
        return x_out


class DiffusionTransformerHead(nn.Module):
    """
    Diffusion Transformer Head (Figure 2e).
    Used in phase 2. Takes all tokens (masked + unmasked) with self-attention.
    Context vector c = timestep_embedding + conditional_tokens (summed).
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        cond_out_dim: int = 768,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.t_embed = TimestepEmbedding(embedding_dim=256, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_out_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            DiffTransformerBlock(dim=hidden_dim, cond_dim=hidden_dim, num_heads=num_heads)
            for _ in range(depth)
        ])
        self.out_proj = nn.Linear(hidden_dim, in_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, D = x_t.shape
        t_emb = self.t_embed(t).unsqueeze(1)  # [B, 1, hidden_dim]
        cond_emb = self.cond_proj(cond)       # [B, N, hidden_dim]
        c = t_emb + cond_emb                  # [B, N, hidden_dim]
        h = self.in_proj(x_t)
        for block in self.blocks:
            h = block(h, c, attn_mask=attn_mask)
        return self.out_proj(h)
