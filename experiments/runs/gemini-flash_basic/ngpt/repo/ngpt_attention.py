import torch
import torch.nn as nn
from normalization import normalize

class RotaryPositionalEmbedding(nn.Module):
    """
    Placeholder for Rotary Positional Embeddings (RoPE).
    The paper mentions using RoPE but does not modify its application in nGPT.
    Actual implementation details are abstracted for this reproduction.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In a real implementation, this would apply RoPE to the input tensor.
        # For this reproduction, we assume it's applied correctly if needed.
        return x

class NGPTAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, s_qk_init: float = 1.0, s_qk_scale: float = None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.W_q = nn.Parameter(torch.rand(d_model, d_model))
        self.W_k = nn.Parameter(torch.rand(d_model, d_model))
        self.W_v = nn.Parameter(torch.rand(d_model, d_model))
        self.W_o = nn.Parameter(torch.rand(d_model, d_model))

        if s_qk_scale is None:
            # Corrected: s_qk_scale should be related to d_k, not d_model (Section 2.3.2, 2.6.4)
            s_qk_scale = 1.0 / (self.d_k**0.5) 

        # s_qk is a trainable scaling factor for query and key vectors (Section 2.3.2, 2.6.4)
        # It's a vector of d_k elements per head.
        self.s_qk_unscaled = nn.Parameter(torch.full((n_heads, self.d_k), s_qk_init))
        self.s_qk_scale_factor = s_qk_scale
        self.s_qk = self.s_qk_unscaled * (s_qk_init / s_qk_scale) # Effective s_qk as per Section 2.5

        self.rope = RotaryPositionalEmbedding(self.d_k) # Apply RoPE to query and key vectors

    def forward(self, h: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, d_model = h.size()

        # 1. Normalize weight matrices (Section 2.6, point 2)
        self.W_q.data = normalize(self.W_q.data, dim=-1)
        self.W_k.data = normalize(self.W_k.data, dim=-1)
        self.W_v.data = normalize(self.W_v.data, dim=-1)
        self.W_o.data = normalize(self.W_o.data, dim=-1)

        # Linear projections (Section 2.3.1, Equation 12)
        q = torch.matmul(h, self.W_q).view(batch_size, seq_len, self.n_heads, self.d_k)
        k = torch.matmul(h, self.W_k).view(batch_size, seq_len, self.n_heads, self.d_k)
        v = torch.matmul(h, self.W_v).view(batch_size, seq_len, self.n_heads, self.d_k)

        # Apply RoPE (abstracted implementation)
        q = self.rope(q)
        k = self.rope(k)

        # Normalize q and k, then apply s_qk (Section 2.3.2, Equations 15, 16)
        q = normalize(q, dim=-1) * self.s_qk
        k = normalize(k, dim=-1) * self.s_qk

        # Transpose for attention calculation
        q = q.transpose(1, 2) # (batch_size, n_heads, seq_len, d_k)
        k = k.transpose(1, 2) # (batch_size, n_heads, seq_len, d_k)
        v = v.transpose(1, 2) # (batch_size, n_heads, seq_len, d_k)

        # Compute attention scores (Section 2.3.1, Equation 13)
        # Adjusted softmax scaling factor to sqrt(d_k) (Section 2.3.2, 2.6.4)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.d_k**0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)

        # Concatenate heads and final linear projection (Section 2.3.1, Equation 14)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        h_A = torch.matmul(attention_output, self.W_o)

        return h_A

