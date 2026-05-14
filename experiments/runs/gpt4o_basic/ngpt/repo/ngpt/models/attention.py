import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalizedAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(NormalizedAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Projection matrices
        self.W_q = nn.Parameter(torch.randn(d_model, d_model))
        self.W_k = nn.Parameter(torch.randn(d_model, d_model))
        self.W_v = nn.Parameter(torch.randn(d_model, d_model))
        self.W_o = nn.Parameter(torch.randn(d_model, d_model))

        # Scaling factors
        self.s_qk = nn.Parameter(torch.ones(self.d_k))

    def forward(self, h):
        # Normalize projection matrices
        W_q_norm = F.normalize(self.W_q, p=2, dim=0)
        W_k_norm = F.normalize(self.W_k, p=2, dim=0)
        W_v_norm = F.normalize(self.W_v, p=2, dim=0)
        W_o_norm = F.normalize(self.W_o, p=2, dim=0)

        # Compute queries, keys, values
        q = torch.matmul(h, W_q_norm)
        k = torch.matmul(h, W_k_norm)
        v = torch.matmul(h, W_v_norm)

        # Normalize q and k
        q = F.normalize(q, p=2, dim=-1) * self.s_qk
        k = F.normalize(k, p=2, dim=-1) * self.s_qk

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / self.d_k**0.5
        attn_weights = F.softmax(scores, dim=-1)

        # Compute weighted sum of values
        attn_output = torch.matmul(attn_weights, v)

        # Final projection
        output = torch.matmul(attn_output, W_o_norm)
        return output
