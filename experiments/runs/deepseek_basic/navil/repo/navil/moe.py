"""
NaViL Modality-Specific Mixture-of-Experts (MoE) Module.

Implements the MHA-MMoE (Multi-Head Attention Modality Mixture of Experts) 
and FFN-MMoE (Feed-Forward Network Modality Mixture of Experts) as described 
in Section 3.2.2.

Key formulas (Eq 3-5):

For each token x_{i,m} with modality m ∈ {visual, linguistic}:

    MHA-MMoE(x_{i,m}) = softmax(Q K^T / √d) V W_O^m
    with Q_{i,m} = x_{i,m} W_Q^m, K_{i,m} = x_{i,m} W_K^m, V_{i,m} = x_{i,m} W_V^m

    FFN-MMoE(x_{i,m}) = (SiLU(x_{i,m} W_gate^m) ⊙ x_{i,m} W_up^m) W_down^m

The key insight: instead of using separate FFN experts only (which can lead to 
feature scale mismatch), NaViL uses modality-specific projection matrices for 
BOTH attention (qkvo) and FFN (gate, up, down). This ensures:
1. Modality-specific processing while maintaining unified global attention
2. Consistent feature scales across modalities
3. No increase in activated parameters (only 1 expert active per token)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class CausalAttention_MMoE(nn.Module):
    """
    Multi-Head Attention with Modality Mixture of Experts (MHA-MMoE).
    
    Uses modality-specific projection matrices W_Q^m, W_K^m, W_V^m, W_O^m
    for visual and linguistic modalities. The attention computation itself
    is unified (global) across modalities.
    """
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size
        
        # Modality-specific projections: separate weights for visual and text
        self.q_proj_visual = nn.Linear(hidden_size, hidden_size)
        self.k_proj_visual = nn.Linear(hidden_size, hidden_size)
        self.v_proj_visual = nn.Linear(hidden_size, hidden_size)
        self.o_proj_visual = nn.Linear(hidden_size, hidden_size)
        
        self.q_proj_text = nn.Linear(hidden_size, hidden_size)
        self.k_proj_text = nn.Linear(hidden_size, hidden_size)
        self.v_proj_text = nn.Linear(hidden_size, hidden_size)
        self.o_proj_text = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        modality_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, N, C) all token embeddings
            modality_mask: (B, N) boolean mask, True=visual, False=text
            attention_mask: causal attention mask
            rope_cos, rope_sin: 1D-RoPE position encodings
            
        Returns:
            (B, N, C) attention output
        """
        B, N, C = hidden_states.shape
        
        # Compute Q, K, V with modality-specific projections
        # Initialize output tensors
        q = torch.zeros(B, N, self.num_heads, self.head_dim, 
                        device=hidden_states.device, dtype=hidden_states.dtype)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        
        # Visual tokens use visual projections
        vis_mask = modality_mask.unsqueeze(-1)  # (B, N, 1)
        txt_mask = ~modality_mask.unsqueeze(-1)
        
        # For visual tokens
        vis_tokens = hidden_states * vis_mask
        q += self.q_proj_visual(vis_tokens).view(B, N, self.num_heads, self.head_dim) * vis_mask.unsqueeze(-1)
        k += self.k_proj_visual(vis_tokens).view(B, N, self.num_heads, self.head_dim) * vis_mask.unsqueeze(-1)
        v += self.v_proj_visual(vis_tokens).view(B, N, self.num_heads, self.head_dim) * vis_mask.unsqueeze(-1)
        
        # For text tokens
        txt_tokens = hidden_states * txt_mask
        q += self.q_proj_text(txt_tokens).view(B, N, self.num_heads, self.head_dim) * txt_mask.unsqueeze(-1)
        k += self.k_proj_text(txt_tokens).view(B, N, self.num_heads, self.head_dim) * txt_mask.unsqueeze(-1)
        v += self.v_proj_text(txt_tokens).view(B, N, self.num_heads, self.head_dim) * txt_mask.unsqueeze(-1)
        
        # Apply RoPE
        if rope_cos is not None and rope_sin is not None:
            q, k = self._apply_rope(q, k, rope_cos, rope_sin)
        
        # Transpose for attention: (B, num_heads, N, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)
        
        # Modality-specific output projection
        o_vis = self.o_proj_visual(attn_output * vis_mask)
        o_txt = self.o_proj_text(attn_output * txt_mask)
        
        return o_vis + o_txt
    
    @staticmethod
    def _apply_rope(q, k, cos, sin):
        """Apply rotary position embeddings."""
        q_dim = q.shape[-1]
        k_dim = k.shape[-1]
        
        cos_q = cos[..., :q_dim // 2].unsqueeze(1)
        sin_q = sin[..., :q_dim // 2].unsqueeze(1)
        cos_k = cos[..., :k_dim // 2].unsqueeze(1)
        sin_k = sin[..., :k_dim // 2].unsqueeze(1)
        
        q_rot = q.reshape(*q.shape[:-1], -1, 2)
        q_rot = torch.stack([-q_rot[..., 1], q_rot[..., 0]], dim=-1).reshape(q.shape)
        q = q * cos_q + q_rot * sin_q
        
        k_rot = k.reshape(*k.shape[:-1], -1, 2)
        k_rot = torch.stack([-k_rot[..., 1], k_rot[..., 0]], dim=-1).reshape(k.shape)
        k = k * cos_k + k_rot * sin_k
        
        return q, k


class FFN_MMoE(nn.Module):
    """
    Feed-Forward Network with Modality Mixture of Experts (FFN-MMoE).
    
    Uses modality-specific projection matrices W_gate^m, W_up^m, W_down^m.
    The formula (Eq 5):
        FFN-MMoE(x_{i,m}) = (SiLU(x W_gate^m) ⊙ x W_up^m) W_down^m
    """
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Visual modality FFN experts
        self.gate_proj_visual = nn.Linear(hidden_size, intermediate_size)
        self.up_proj_visual = nn.Linear(hidden_size, intermediate_size)
        self.down_proj_visual = nn.Linear(intermediate_size, hidden_size)
        
        # Text modality FFN experts
        self.gate_proj_text = nn.Linear(hidden_size, intermediate_size)
        self.up_proj_text = nn.Linear(hidden_size, intermediate_size)
        self.down_proj_text = nn.Linear(intermediate_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, N, C) 
            modality_mask: (B, N) boolean, True=visual
        Returns:
            (B, N, C)
        """
        vis_mask = modality_mask.unsqueeze(-1)
        txt_mask = (~modality_mask).unsqueeze(-1)
        
        # Process visual tokens through visual experts
        vis_tokens = hidden_states * vis_mask
        gate_vis = F.silu(self.gate_proj_visual(vis_tokens))
        up_vis = self.up_proj_visual(vis_tokens)
        vis_output = self.down_proj_visual(gate_vis * up_vis)
        
        # Process text tokens through text experts  
        txt_tokens = hidden_states * txt_mask
        gate_txt = F.silu(self.gate_proj_text(txt_tokens))
        up_txt = self.up_proj_text(txt_tokens)
        txt_output = self.down_proj_text(gate_txt * up_txt)
        
        output = (vis_output * vis_mask) + (txt_output * txt_mask)
        return self.dropout(output)


class ModalityMoELayer(nn.Module):
    """
    Single MoE-extended LLM transformer layer.
    
    Combines MHA-MMoE and FFN-MMoE as described in Eq (2-3):
        x' = x + MHA-MMoE(RMSNorm(x))
        x = x' + FFN-MMoE(RMSNorm(x'))
        
    Only one expert is activated per token (visual or linguistic), 
    maintaining consistent inference costs compared to vanilla LLM.
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int, 
        intermediate_size: int, 
        dropout: float = 0.0,
        use_moe: bool = True,
    ):
        super().__init__()
        self.use_moe = use_moe
        
        self.norm1 = RMSNorm(hidden_size)
        if use_moe:
            self.attn = CausalAttention_MMoE(hidden_size, num_heads, dropout)
        else:
            # Fallback to standard causal attention
            self.attn = StandardCausalAttention(hidden_size, num_heads, dropout)
            
        self.norm2 = RMSNorm(hidden_size)
        if use_moe:
            self.ffn = FFN_MMoE(hidden_size, intermediate_size, dropout)
        else:
            self.ffn = StandardFFN(hidden_size, intermediate_size, dropout)
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm attention
        residual = hidden_states
        normed = self.norm1(hidden_states)
        
        if self.use_moe and modality_mask is not None:
            attn_out = self.attn(normed, modality_mask, attention_mask, rope_cos, rope_sin)
        else:
            attn_out = self.attn(normed, attention_mask, rope_cos, rope_sin)
            
        hidden_states = residual + attn_out
        
        # Pre-norm FFN
        residual = hidden_states
        normed = self.norm2(hidden_states)
        
        if self.use_moe and modality_mask is not None:
            ffn_out = self.ffn(normed, modality_mask)
        else:
            ffn_out = self.ffn(normed)
            
        hidden_states = residual + ffn_out
        
        return hidden_states


class StandardCausalAttention(nn.Module):
    """Standard causal multi-head attention (no MoE)."""
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, N, self.num_heads, self.head_dim)
        
        if rope_cos is not None and rope_sin is not None:
            q, k = CausalAttention_MMoE._apply_rope(q, k, rope_cos, rope_sin)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)
        
        return self.o_proj(attn_output)


class StandardFFN(nn.Module):
    """Standard SwiGLU Feed-Forward Network (no MoE)."""
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))
