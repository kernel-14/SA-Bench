
import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional

class RMSNorm(nn.Module):
    """
    RMSNorm as described in the paper, used for layer normalization.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class PatchEmbedding(nn.Module):
    """
    Patch Embedding Layer: Converts input images into sequences of visual tokens.
    """
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int, stride: int = 16):
        super().__init__()
        # Ensure img_size is divisible by patch_size for simplicity in this base implementation
        # The paper mentions padding images to be multiples of 32 for NaViL-2B (Section 5.1)
        # However, the input images are first padded to ensure its length and width are multiples of 32.
        # This implies that the image size used here would be after padding, and directly compatible.
        
        # In a more complete implementation, we'd handle dynamic padding or ensure input is pre-padded.
        # For now, assume input img_size is already compatible.

        # The paper says "The stride of Patch Embedding layer is set to 16." (Section 5.1)
        # This implies a Conv2D with kernel_size=patch_size and stride=stride
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x) # (B, embed_dim, H', W')
        x = rearrange(x, 'b c h w -> b (h w) c') # (B, N_patches, embed_dim)
        return x

class MHA_MMoE(nn.Module):
    """
    Modality-Specific Multi-Head Attention (MHA-MMoE) expert.
    Uses different projection layers (qkvo) for visual and linguistic modalities.
    """
    def __init__(self, embed_dim: int, num_heads: int, is_bidirectional: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Modality-specific projection matrices for Q, K, V, O
        self.q_proj_visual = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj_visual = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj_visual = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj_visual = nn.Linear(embed_dim, embed_dim, bias=False)

        self.q_proj_linguistic = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj_linguistic = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj_linguistic = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj_linguistic = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.is_bidirectional = is_bidirectional

    def forward(self, x: torch.Tensor, modality_indicator: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                rope_pos_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, L, E) - Batch, Sequence Length, Embedding Dimension
        # modality_indicator: (B, L) - 0 for linguistic, 1 for visual

        batch_size, seq_len, embed_dim = x.shape

        q_visual = self.q_proj_visual(x)
        k_visual = self.k_proj_visual(x)
        v_visual = self.v_proj_visual(x)

        q_linguistic = self.q_proj_linguistic(x)
        k_linguistic = self.k_proj_linguistic(x)
        v_linguistic = self.v_proj_linguistic(x)

        # Apply modality-specific projections
        q = torch.where(modality_indicator.unsqueeze(-1).bool(), q_visual, q_linguistic)
        k = torch.where(modality_indicator.unsqueeze(-1).bool(), k_visual, k_linguistic)
        v = torch.where(modality_indicator.unsqueeze(-1).bool(), v_visual, v_linguistic)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b l (h d) -> b h l d', h = self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h = self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h = self.num_heads)

        # Scaled Dot-Product Attention
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if rope_pos_bias is not None:
             # Apply RoPE bias. Assuming rope_pos_bias is already shaped for attention_weights
             # The paper mentions 2D-RoPE for visual encoder, 1D-RoPE for LLM.
             # This would typically be applied to Q and K *before* matmul.
             # For simplicity, assuming rope_pos_bias is a pre-calculated additive bias to attention weights.
             # A full RoPE implementation would modify Q/K directly.
             attn_weights = attn_weights + rope_pos_bias

        if attn_mask is not None:
            if self.is_bidirectional: # Mask out padding or future tokens for causal
                attn_weights = attn_weights.masked_fill(attn_mask.unsqueeze(1).unsqueeze(1).bool(), float('-inf'))
            else: # Causal attention for LLM (1D-RoPE, Section 5.1)
                # Create a causal mask for LLM, assuming attn_mask only handles padding if needed.
                # Standard causal mask:
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
                attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
                if attn_mask is not None: # Apply additional mask if provided (e.g. padding mask)
                    attn_weights = attn_weights.masked_fill(attn_mask.unsqueeze(1).unsqueeze(1).bool(), float('-inf'))


        attn_weights = torch.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v) # (B, H, L, D_head)

        output = rearrange(output, 'b h l d -> b l (h d)') # (B, L, E)

        # Apply modality-specific output projection
        o_visual = self.o_proj_visual(output)
        o_linguistic = self.o_proj_linguistic(output)
        output = torch.where(modality_indicator.unsqueeze(-1).bool(), o_visual, o_linguistic)

        return output

class FFN_MMoE(nn.Module):
    """
    Modality-Specific Feed-Forward Network (FFN-MMoE) expert.
    """
    def __init__(self, embed_dim: int, mlp_dim: int):
        super().__init__()
        self.gate_proj_visual = nn.Linear(embed_dim, mlp_dim, bias=False)
        self.up_proj_visual = nn.Linear(embed_dim, mlp_dim, bias=False)
        self.down_proj_visual = nn.Linear(mlp_dim, embed_dim, bias=False)

        self.gate_proj_linguistic = nn.Linear(embed_dim, mlp_dim, bias=False)
        self.up_proj_linguistic = nn.Linear(embed_dim, mlp_dim, bias=False)
        self.down_proj_linguistic = nn.Linear(mlp_dim, embed_dim, bias=False)

        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor, modality_indicator: torch.Tensor) -> torch.Tensor:
        # x: (B, L, E)
        # modality_indicator: (B, L) - 0 for linguistic, 1 for visual

        gate_visual = self.silu(self.gate_proj_visual(x))
        up_visual = self.up_proj_visual(x)
        output_visual = self.down_proj_visual(gate_visual * up_visual)

        gate_linguistic = self.silu(self.gate_proj_linguistic(x))
        up_linguistic = self.up_proj_linguistic(x)
        output_linguistic = self.down_proj_linguistic(gate_linguistic * up_linguistic)

        output = torch.where(modality_indicator.unsqueeze(-1).bool(), output_visual, output_linguistic)
        return output

