import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = False, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # B, num_heads, N, head_dim

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization with Zero initialization (DiT style).
    Used in Hi-MAR Transformer blocks.
    Outputs the six parameters (alpha1, beta1, gamma1, alpha2, beta2, gamma2) directly.
    """
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, 6 * dim),
            nn.SiLU() # Changed from original paper's MLP for scale vector to SiLU as seen in DiT implementations
        )
        # Initialize output to zeros
        nn.init.constant_(self.mlp[-1].weight, 0)
        nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, cond: torch.Tensor):
        # cond: (B, cond_dim)
        # Returns: (B, 1, dim) x 6 for broadcasting
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = self.mlp(cond).chunk(6, dim=1)
        alpha1 = alpha1.unsqueeze(1)
        beta1 = beta1.unsqueeze(1)
        gamma1 = gamma1.unsqueeze(1)
        alpha2 = alpha2.unsqueeze(1)
        beta2 = beta2.unsqueeze(1)
        gamma2 = gamma2.unsqueeze(1)
        return alpha1, beta1, gamma1, alpha2, beta2, gamma2

class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization.
    Used in Diffusion Transformer Head and MLP-based Diffusion Head.
    Outputs the six parameters (alpha1, beta1, gamma1, alpha2, beta2, gamma2) directly.
    """
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, 6 * dim)
        )

    def forward(self, cond: torch.Tensor):
        # cond: (B, cond_dim)
        # Returns: (B, 1, dim) x 6 for broadcasting
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = self.mlp(cond).chunk(6, dim=1)
        alpha1 = alpha1.unsqueeze(1)
        beta1 = beta1.unsqueeze(1)
        gamma1 = gamma1.unsqueeze(1)
        alpha2 = alpha2.unsqueeze(1)
        beta2 = beta2.unsqueeze(1)
        gamma2 = gamma2.unsqueeze(1)
        return alpha1, beta1, gamma1, alpha2, beta2, gamma2


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm, implementing Equation (3) from the paper.
    Flexible to use AdaLNZero or AdaLN.
    """
    def __init__(self, dim: int, num_heads: int, cond_dim: int, ada_type: str = "adaln_zero"):
        super().__init__()
        # LN, Attention, FFN
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = FeedForward(dim, int(4 * dim)) # Hidden dim usually 4*dim

        if ada_type == "adaln_zero":
            self.adaln = AdaLNZero(dim, cond_dim)
        elif ada_type == "adaln":
            self.adaln = AdaLN(dim, cond_dim)
        else:
            raise ValueError(f"Unknown adaln type: {ada_type}")

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Get alpha, beta, gamma parameters from AdaLN
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = self.adaln(cond)

        # Self-attention block: z_a = z^i + gamma_1 * Attention(alpha_1 * LN(z^i) + beta_1)
        norm_x_msa = self.norm1(x)
        attn_input = alpha1 * norm_x_msa + beta1
        attn_output = self.attn(attn_input)
        x = x + gamma1 * attn_output

        # Feed-forward block: z^{i+1} = z_a + gamma_2 * FFN(alpha_2 * LN(z_a) + beta_2)
        norm_x_mlp = self.norm2(x)
        ffn_input = alpha2 * norm_x_mlp + beta2
        ffn_output = self.ffn(ffn_input)
        x = x + gamma2 * ffn_output
        return x

# Positional encoding for sequence length (N)
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = grid_w = grid_size
    grid = torch.meshgrid([torch.arange(grid_h, dtype=torch.float32), torch.arange(grid_w, dtype=torch.float32)])
    grid = torch.stack(grid, dim=0) # 2, grid_h, grid_w

    grid = grid.reshape([2, 1, grid_h, grid_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = torch.cat([torch.zeros([1, embed_dim]), pos_embed], dim=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1) # (H*W, D)
    return emb

def get_sincos_pos_embed_from_grid(embed_dim, grid):
    emb = torch.empty(grid.shape[0], embed_dim)
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    grid = grid.reshape(-1, 1)  # (M, 1)
    pos = grid * omega  # (M, D/2)
    emb[:, 0::2] = torch.sin(pos)
    emb[:, 1::2] = torch.cos(pos)
    return emb