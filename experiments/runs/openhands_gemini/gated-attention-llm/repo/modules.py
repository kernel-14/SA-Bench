
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from layers import RMSNorm, GatedMechanism # Assuming RMSNorm and GatedMechanism are in layers.py
from config import ModelConfig

class GatedMultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.kv_heads = config.kv_heads
        self.d_k = config.d_k
        self.head_dim = self.d_k
        self.use_gated_attention = config.use_gated_attention
        self.gating_position = config.gating_position
        self.head_specific_gating = config.head_specific_gating
        self.gating_granularity = config.gating_granularity
        self.gating_type = config.gating_type
        self.gating_activation = config.gating_activation

        # QKV projections
        self.wq = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.d_model, self.kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.d_model, self.kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

        # Grouped Query Attention
        self.n_group = self.n_heads // self.kv_heads

        # Gating mechanisms
        self.gating_G1, self.gating_G2, self.gating_G3, self.gating_G4, self.gating_G5 = \
            None, None, None, None, None

        if self.use_gated_attention:
            # G1: After SDPA output
            if self.gating_position == "G1":
                self.gating_G1 = GatedMechanism(
                    output_dim=self.n_heads * self.head_dim,
                    gating_dim=self.d_model, # SDPA output is derived from query in this context (Equ. 8), so input for gate is X (original input to attention)
                    gating_granularity=self.gating_granularity,
                    head_specific=self.head_specific_gating,
                    gating_type=self.gating_type,
                    activation_function=self.gating_activation,
                    num_heads=self.n_heads,
                    head_dim=self.head_dim if (self.gating_granularity == "elementwise" and not self.head_specific) else None
                )
            # G2: After Value projection (W_V)
            elif self.gating_position == "G2":
                self.gating_G2 = GatedMechanism(
                    output_dim=self.kv_heads * self.head_dim,
                    gating_dim=self.d_model, # Input for gate is X
                    gating_granularity=self.gating_granularity,
                    head_specific=self.head_specific_gating,
                    gating_type=self.gating_type,
                    activation_function=self.gating_activation,
                    num_heads=self.kv_heads, # Value projection output has kv_heads
                    head_dim=self.head_dim if (self.gating_granularity == "elementwise" and not self.head_specific) else None
                )
            # G3: After Key projection (W_K)
            elif self.gating_position == "G3":
                self.gating_G3 = GatedMechanism(
                    output_dim=self.kv_heads * self.head_dim,
                    gating_dim=self.d_model,
                    gating_granularity=self.gating_granularity,
                    head_specific=self.head_specific_gating,
                    gating_type=self.gating_type,
                    activation_function=self.gating_activation,
                    num_heads=self.kv_heads,
                    head_dim=self.head_dim if (self.gating_granularity == "elementwise" and not self.head_specific) else None
                )
            # G4: After Query projection (W_Q)
            elif self.gating_position == "G4":
                self.gating_G4 = GatedMechanism(
                    output_dim=self.n_heads * self.head_dim,
                    gating_dim=self.d_model,
                    gating_granularity=self.gating_granularity,
                    head_specific=self.head_specific_gating,
                    gating_type=self.gating_type,
                    activation_function=self.gating_activation,
                    num_heads=self.n_heads,
                    head_dim=self.head_dim if (self.gating_granularity == "elementwise" and not self.head_specific) else None
                )
            # G5: After final dense output layer (W_O)
            elif self.gating_position == "G5":
                self.gating_G5 = GatedMechanism(
                    output_dim=self.d_model,
                    gating_dim=self.d_model,
                    gating_granularity=self.gating_granularity,
                    head_specific=False, # G5 applies after all heads are combined, so not head-specific
                    gating_type=self.gating_type,
                    activation_function=self.gating_activation,
                    num_heads=None, # Not directly applicable for head_specific, but for elementwise head_shared, it would be a single logical unit.
                    head_dim=self.d_model if (self.gating_granularity == "elementwise" and not self.head_specific) else None
                )

    def _shape(self, tensor: torch.Tensor, seq_len: int, num_heads: int) -> torch.Tensor:
        return tensor.view(tensor.shape[0], seq_len, num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        B, S, D = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Apply gating G4 (Query)
        if self.use_gated_attention and self.gating_position == "G4":
            q = self.gating_G4(q, x) # Gate input is X (original input)

        # Apply gating G3 (Key)
        if self.use_gated_attention and self.gating_position == "G3":
            k = self.gating_G3(k, x) # Gate input is X (original input)

        # Apply gating G2 (Value)
        if self.use_gated_attention and self.gating_position == "G2":
            v = self.gating_G2(v, x) # Gate input is X (original input)

        q = self._shape(q, S, self.n_heads) # (B, H, S, d_k)
        k = self._shape(k, S, self.kv_heads) # (B, kv_H, S, d_k)
        v = self._shape(v, S, self.kv_heads) # (B, kv_H, S, d_k)

        # Grouped Query Attention (GQA)
        # Repeat k and v heads to match number of query heads
        if self.n_group > 1:
            k = k.repeat_interleave(self.n_group, dim=1) # (B, H, S, d_k)
            v = v.repeat_interleave(self.n_group, dim=1) # (B, H, S, d_k)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(2, 3)) / (self.head_dim ** 0.5) # (B, H, S, S)

        if attention_mask is not None:
            # attention_mask (B, 1, 1, S) or (B, 1, S, S) for causal masking
            scores = scores + attention_mask

        attention_weights = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(attention_weights, v) # (B, H, S, d_k)

        # Transpose to get (B, S, H, d_k) and concatenate heads
        output = output.transpose(1, 2).contiguous().view(B, S, self.n_heads * self.head_dim)

        # Apply gating G1 (SDPA output)
        if self.use_gated_attention and self.gating_position == "G1":
            # For G1, the paper states X is "derived from the hidden states corresponding to the current query"
            # In the GatedMechanism, we pass 'x' (the original input to the attention block) as 'gate_input'
            # because the gate for G1 is based on the input features before QKV projections
            # (as suggested by Fig 1. G1 arrow coming from input X before QKV) and Eq 8.
            output = self.gating_G1(output, x)

        output = self.wo(output)

        # Apply gating G5 (Final output layer)
        if self.use_gated_attention and self.gating_position == "G5":
            output = self.gating_G5(output, x)

        return output

class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        # Paper mentions FFN width reduction when using gating to maintain parameter size.
        # This implementation uses the d_ff from config, assuming it's already adjusted if needed.

        self.w1 = nn.Linear(self.d_model, self.d_ff, bias=False)
        self.w2 = nn.Linear(self.d_ff, self.d_model, bias=False)
        self.act = nn.SiLU() # SwiGLU variant is common, but paper mentions SiLU as activation for gating, and standard FFNs usually have GELU or SiLU

    def forward(self, x: torch.Tensor):
        return self.w2(self.act(self.w1(x)))

