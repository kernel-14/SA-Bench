import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedSelfAttention(nn.Module):
    """
    Implements the Gated Self-Attention mechanism as described in the paper
    "Gated Attention for Large Language Models" at position G1 (after SDPA).

    This module introduces a head-specific, element-wise sigmoid gate
    to modulate the output of the Scaled Dot-Product Attention.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)

        # Gating mechanism (G1: after SDPA output)
        # The gating scores are derived from the query (X in Y' = Y * sigma(X W_theta)).
        # It's head-specific and element-wise, meaning W_theta projects to the same
        # dimensionality as the multi-head SDPA output.
        # The input to the gating projection is the original query 'query'.
        self.gating_W_theta = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len_q, embed_dim)
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len_k, embed_dim)
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len_v, embed_dim)
            attn_mask (torch.Tensor, optional): Optional mask for attention scores.
                Shape (seq_len_q, seq_len_k) or (batch_size, num_heads, seq_len_q, seq_len_k).
                A 0 value means masked.
            key_padding_mask (torch.Tensor, optional): Optional mask for key padding.
                Shape (batch_size, seq_len_k). A 0 value means masked.

        Returns:
            tuple:
                - output (torch.Tensor): Output tensor of shape (batch_size, seq_len_q, embed_dim)
                - attn_weights (torch.Tensor): Attention weights of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size, seq_len_q, _ = query.size()
        seq_len_k = key.size(1)

        # 1. QKV Linear Projections
        # (batch, seq_len, embed_dim) -> (batch, num_heads, seq_len, head_dim)
        q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Scaled Dot-Product Attention (SDPA)
        # (batch, num_heads, seq_len_q, head_dim) @ (batch, num_heads, head_dim, seq_len_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5) # (batch, num_heads, seq_len_q, seq_len_k)

        # Apply attention mask
        if attn_mask is not None:
            # Ensure mask is broadcastable (e.g., (1, 1, seq_len_q, seq_len_k) or (batch, num_heads, seq_len_q, seq_len_k))
            if attn_mask.dim() == 2: # (seq_len_q, seq_len_k)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3: # (batch, seq_len_q, seq_len_k) for causal masks from Transformer Decoder
                 attn_mask = attn_mask.unsqueeze(1) # -> (batch, 1, seq_len_q, seq_len_k)

            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        # Apply key padding mask
        if key_padding_mask is not None:
            # key_padding_mask: (batch_size, seq_len_k)
            # Expand to (batch_size, 1, 1, seq_len_k) to broadcast over heads and query length
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # (batch, num_heads, seq_len_q, seq_len_k) @ (batch, num_heads, seq_len_k, head_dim)
        sdpa_output = torch.matmul(attn_weights, v) # (batch, num_heads, seq_len_q, head_dim)

        # 3. Apply Gating Mechanism (G1)
        # Input for gating scores is the original query tensor
        gating_scores_input = self.gating_W_theta(query) # (batch_size, seq_len_q, embed_dim)
        # Reshape to (batch_size, num_heads, seq_len_q, head_dim) to match sdpa_output for element-wise multiplication
        gating_scores_input = gating_scores_input.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        gating_scores = torch.sigmoid(gating_scores_input) # Apply sigmoid activation

        # Element-wise multiplication of SDPA output with gating scores
        gated_sdpa_output = sdpa_output * gating_scores

        # 4. Concatenate heads and Final Output Layer
        # (batch, num_heads, seq_len_q, head_dim) -> (batch, seq_len_q, num_heads, head_dim) -> (batch, seq_len_q, embed_dim)
        concat_output = gated_sdpa_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embed_dim)
        output = self.out_proj(concat_output) # (batch_size, seq_len_q, embed_dim)

        return output, attn_weights
