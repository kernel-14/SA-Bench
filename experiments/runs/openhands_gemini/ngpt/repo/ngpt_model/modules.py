
import torch
import torch.nn as nn
import torch.nn.functional as F
from ngpt_model.layers import Norm, RMSNorm, RotaryPositionEmbedding
import math

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention block.
    Can be configured for both baseline Transformer and Normalized Transformer (nGPT).
    """
    def __init__(self, d_model: int, n_heads: int, d_k: int, dropout: float = 0.1, is_ngpt: bool = False, rope_base: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.is_ngpt = is_ngpt

        # Linear projections for query, key, value
        self.Wq = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.Wk = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.Wv = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.Wo = nn.Linear(n_heads * d_k, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.rope = RotaryPositionEmbedding(d_k, base=rope_base)

        if self.is_ngpt:
            # Trainable scaling factor for QK dot product
            self.s_qk = nn.Parameter(torch.ones(n_heads, d_k))
            self.s_qk.data.fill_(1.0) # Initialized to 1, scaled by 1/sqrt(d_model) effectively
            self.s_qk_scale = 1.0 / math.sqrt(d_model) # From paper, s_qk_scale = 1/sqrt(d_model)

            # Normalization of weights
            self.normalize_weights()

    def normalize_weights(self):
        """Normalize weights along their embedding dimension after each training step."""
        with torch.no_grad():
            self.Wq.weight.data = Norm(self.Wq.weight.data)
            self.Wk.weight.data = Norm(self.Wk.weight.data)
            self.Wv.weight.data = Norm(self.Wv.weight.data)
            self.Wo.weight.data = Norm(self.Wo.weight.data)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = h.size()

        if self.is_ngpt:
            # In nGPT, weights are normalized. No RMSNorm before QKV projections.
            q = self.Wq(h).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            k = self.Wk(h).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            v = self.Wv(h).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        else:
            # Baseline Transformer applies RMSNorm before QKV projections.
            q = self.Wq(h).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            k = self.Wk(h).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            v = self.Wv(h).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Apply RoPE
        q = self.rope(q, seq_len=seq_len)
        k = self.rope(k, seq_len=seq_len)

        if self.is_ngpt:
            # Normalize q and k, then scale by s_qk
            q = Norm(q) * (self.s_qk * self.s_qk_scale)
            k = Norm(k) * (self.s_qk * self.s_qk_scale)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        if self.is_ngpt:
            # Change softmax scaling factor from 1/sqrt(d_k) to sqrt(d_k)
            attn_scores = attn_scores * math.sqrt(self.d_k)
        else:
            attn_scores = attn_scores / math.sqrt(self.d_k)

        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.Wo(output)
        output = self.proj_dropout(output)

        return output

class MLP(nn.Module):
    """
    Multi-layer perceptron block with SwiGLU activation.
    Can be configured for both baseline Transformer and Normalized Transformer (nGPT).
    """
    def __init__(self, d_model: int, d_mlp: int, dropout: float = 0.1, is_ngpt: bool = False):
        super().__init__()
        self.is_ngpt = is_ngpt

        self.Wu = nn.Linear(d_model, d_mlp, bias=False)
        self.Wv = nn.Linear(d_model, d_mlp, bias=False)
        self.Wo_mlp = nn.Linear(d_mlp, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        if self.is_ngpt:
            # Trainable scaling factors for u and v
            self.s_u = nn.Parameter(torch.ones(d_mlp))
            self.s_v = nn.Parameter(torch.ones(d_mlp))
            self.s_u.data.fill_(1.0) # Initialized to 1
            self.s_v.data.fill_(1.0) # Initialized to 1
            self.sqrt_d_model = math.sqrt(d_model)

            # Normalization of weights
            self.normalize_weights()

    def normalize_weights(self):
        """Normalize weights along their embedding dimension after each training step."""
        with torch.no_grad():
            self.Wu.weight.data = Norm(self.Wu.weight.data)
            self.Wv.weight.data = Norm(self.Wv.weight.data)
            self.Wo_mlp.weight.data = Norm(self.Wo_mlp.weight.data)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.is_ngpt:
            u = self.Wu(h)
            v = self.Wv(h)
            u = u * self.s_u
            v = v * self.s_v * self.sqrt_d_model
        else:
            u = self.Wu(h)
            v = self.Wv(h)

        swiglu = u * F.silu(v)
        output = self.Wo_mlp(swiglu)
        output = self.dropout(output)
        return output

