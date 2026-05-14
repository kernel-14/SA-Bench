import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def get_causal_mask(L: int, device: torch.device) -> torch.Tensor:
    """Create lower triangular causal attention mask.
    
    M[i, j] = -inf if i < j else 0
    """
    mask = torch.triu(torch.ones(L, L, device=device) * float("-inf"), diagonal=1)
    return mask


class CausalTemporalAttention(nn.Module):
    """Causal temporal attention as described in Eq. (3).
    
    Each frame only attends to its preceding frames.
    Supports KV-cache for efficient autoregressive inference.
    """

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        self.q_proj = nn.Linear(hidden_size, self.inner_dim)
        self.k_proj = nn.Linear(hidden_size, self.inner_dim)
        self.v_proj = nn.Linear(hidden_size, self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        timestep_embed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states: (B, L, HW, C) where L is time, HW is spatial flattened, C is hidden_size
            causal_mask: (L, L) or (l, P_k + l) for cache case
            kv_cache: (K_cache, V_cache) from clean prefix, shape (B, P_k, HW, inner_dim)
            timestep_embed: added to hidden_states before projection (optional)
        
        Returns:
            output: (B, L, HW, C)
            new_kv: (K, V) for cache writing or None
        """
        B, L, HW, C = hidden_states.shape
        inner_dim = self.inner_dim

        if timestep_embed is not None:
            hidden_states = hidden_states + timestep_embed

        Q = self.q_proj(hidden_states)  # (B, L, HW, inner_dim)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)

        # Reshape for multi-head attention: (B, L, HW, heads, head_dim) -> (B, heads, L*HW, head_dim)
        Q = Q.view(B, L, HW, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4).reshape(B, self.num_heads, L * HW, self.head_dim)
        K_new = K.view(B, L, HW, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4).reshape(B, self.num_heads, L * HW, self.head_dim)
        V_new = V.view(B, L, HW, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4).reshape(B, self.num_heads, L * HW, self.head_dim)

        if kv_cache is not None:
            K_cache, V_cache = kv_cache  # (B, P_k, HW, inner_dim)
            Bc, Pk, HWC, ID = K_cache.shape
            K_cache = K_cache.view(Bc, Pk, HW, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4).reshape(Bc, self.num_heads, Pk * HW, self.head_dim)
            V_cache = V_cache.view(Bc, Pk, HW, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4).reshape(Bc, self.num_heads, Pk * HW, self.head_dim)
            # Concatenate: K = [K_cache, K_new], V = [V_cache, V_new]
            K_full = torch.cat([K_cache, K_new], dim=2)  # (B, heads, (P_k + L)*HW, head_dim)
            V_full = torch.cat([V_cache, V_new], dim=2)
        else:
            K_full = K_new
            V_full = V_new

        # Causal mask
        if causal_mask is not None:
            # causal_mask expects shape (L, total_L) but we need to handle HW
            # Build per-grid mask: causal_mask of shape (L, total_L) expanded to (L*HW, total_L*HW)
            total_L = K_full.shape[2] // HW
            mask = causal_mask[:L, :total_L]  # (L, total_L)
            # Expand: each spatial grid attends to same frames
            mask = mask.unsqueeze(2).unsqueeze(3)  # (L, total_L, 1, 1)
            mask = mask.expand(L, total_L, HW, HW).reshape(L * HW, total_L * HW)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L*HW, total_L*HW)
        else:
            mask = None

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(Q, K_full.transpose(-2, -1)) * scale

        if mask is not None:
            attn_weights = attn_weights + mask.to(attn_weights.dtype)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V_full)

        # Reshape back: (B, heads, L*HW, head_dim) -> (B, L, HW, inner_dim)
        attn_output = attn_output.view(B, self.num_heads, L, HW, self.head_dim).permute(0, 2, 3, 1, 4).reshape(B, L, HW, inner_dim)
        output = self.out_proj(attn_output)

        # Return new KV (without cache concatenation) for cache writing
        K_clean = K_new.view(B, self.num_heads, L, HW, self.head_dim).permute(0, 2, 3, 1, 4).reshape(B, L, HW, inner_dim)
        V_clean = V_new.view(B, self.num_heads, L, HW, self.head_dim).permute(0, 2, 3, 1, 4).reshape(B, L, HW, inner_dim)

        return output, (K_clean, V_clean)


class PrefixEnhancedSpatialAttention(nn.Module):
    """Prefix-enhanced spatial attention as described in Eq. (4).
    
    For frames i >= P (denoising target), the key and value are enhanced by 
    concatenating P' clean prefix frames along the spatial dimension.
    For frames i < P (clean prefix), self-repeat is used.
    """

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, prefix_len: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.prefix_len = prefix_len

        self.q_proj = nn.Linear(hidden_size, self.inner_dim)
        self.k_proj = nn.Linear(hidden_size, self.inner_dim)
        self.v_proj = nn.Linear(hidden_size, self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        P: int,
        spatial_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states: (B, L, HW, C) where L is total frames, HW is spatial flattened
            P: number of clean prefix frames (first P frames)
            spatial_kv_cache: cached K, V from clean prefix for inference (B, P', HW, inner_dim)
        
        Returns:
            output: (B, L, HW, C)
        """
        B, L, HW, C = hidden_states.shape
        P_prime = self.prefix_len

        # Prepare keys and values for each frame
        K_all = self.k_proj(hidden_states)  # (B, L, HW, inner_dim)
        V_all = self.v_proj(hidden_states)
        Q_all = self.q_proj(hidden_states)

        outputs = []
        new_kv_cache = None

        for i in range(L):
            Q_i = Q_all[:, i:i+1]  # (B, 1, HW, inner_dim)

            if i >= P:
                # Denoising target: enhance with clean prefix
                if spatial_kv_cache is not None:
                    # Inference: use cached spatial K, V from clean prefix
                    K_cache, V_cache = spatial_kv_cache  # (B, P', HW, inner_dim)
                else:
                    # Training: use clean prefix frames P-P' to P-1
                    start_idx = max(0, P - P_prime)
                    end_idx = P
                    prefix_len_actual = end_idx - start_idx
                    # Pad if needed
                    K_cache = K_all[:, start_idx:end_idx]  # (B, prefix_len_actual, HW, inner_dim)
                    V_cache = V_all[:, start_idx:end_idx]
                    if prefix_len_actual < P_prime:
                        # Pad with self-repeat of the last available frame
                        pad_len = P_prime - prefix_len_actual
                        pad_K = K_all[:, end_idx-1:end_idx].repeat(1, pad_len, 1, 1)
                        pad_V = V_all[:, end_idx-1:end_idx].repeat(1, pad_len, 1, 1)
                        K_cache = torch.cat([pad_K, K_cache], dim=1)
                        V_cache = torch.cat([pad_V, V_cache], dim=1)

                # Concatenate: [prefix K; current frame K] along spatial dim
                K_i = K_all[:, i:i+1]  # (B, 1, HW, inner_dim)
                V_i = V_all[:, i:i+1]
                K_enhanced = torch.cat([K_cache, K_i], dim=2)  # (B, P', HW*(P'+1)? no)
                V_enhanced = torch.cat([V_cache, V_i], dim=2)
                # Actually paper concatenates along spatial dimension, so K_enhanced is:
                # [h_0^{P-P'}; ...; h_0^{P-1}; h_t^i] => shape (B, P'+1, HW, inner_dim) but stacked as (B, 1, (P'+1)*HW, inner_dim)
            else:
                # Clean prefix: self-repeat P' times
                K_enhanced = K_all[:, i:i+1].repeat(1, P_prime + 1, 1, 1)  # (B, P'+1, HW, inner_dim)
                V_enhanced = V_all[:, i:i+1].repeat(1, P_prime + 1, 1, 1)

            # Flatten spatial dim for attention
            B_curr = K_enhanced.shape[0]
            K_flat = K_enhanced.view(B_curr, (P_prime + 1) * HW, self.inner_dim)  # (B, (P'+1)*HW, inner_dim)
            V_flat = V_enhanced.view(B_curr, (P_prime + 1) * HW, self.inner_dim)
            Q_flat = Q_i.view(B_curr, HW, self.inner_dim)  # (B, HW, inner_dim)

            # Multi-head reshape
            K_mh = K_flat.view(B_curr, (P_prime + 1) * HW, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, heads, (P'+1)*HW, head_dim)
            V_mh = V_flat.view(B_curr, (P_prime + 1) * HW, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            Q_mh = Q_flat.view(B_curr, HW, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, heads, HW, head_dim)

            scale = 1.0 / math.sqrt(self.head_dim)
            attn_i = torch.matmul(Q_mh, K_mh.transpose(-2, -1)) * scale
            attn_i = F.softmax(attn_i, dim=-1)
            out_i = torch.matmul(attn_i, V_mh)  # (B, heads, HW, head_dim)

            out_i = out_i.permute(0, 2, 1, 3).reshape(B_curr, 1, HW, self.inner_dim)
            outputs.append(out_i)

        output = torch.cat(outputs, dim=1)  # (B, L, HW, inner_dim)
        output = self.out_proj(output)

        return output


class VisualTextCrossAttention(nn.Module):
    """Cross attention between video features and text embeddings."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, cross_attn_dim: int = 4096):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        self.q_proj = nn.Linear(hidden_size, self.inner_dim)
        self.k_proj = nn.Linear(cross_attn_dim, self.inner_dim)
        self.v_proj = nn.Linear(cross_attn_dim, self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, L*HW, C) video features
            encoder_hidden_states: (B, T_text, cross_attn_dim) text features
        """
        B, LHW, C = hidden_states.shape
        Q = self.q_proj(hidden_states).view(B, LHW, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(encoder_hidden_states).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(encoder_hidden_states).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        out = out.permute(0, 2, 1, 3).reshape(B, LHW, self.inner_dim)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Standard feed-forward network with GELU activation."""

    def __init__(self, hidden_size: int, ff_mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * ff_mult),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size * ff_mult, hidden_size),
        )

    def forward(self, x):
        return self.net(x)


class AdaLayerNorm(nn.Module):
    """Adaptive layer normalization modulated by timestep embedding."""

    def __init__(self, hidden_size: int, adaLN_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(adaLN_dim, 6 * hidden_size),
        )

    def forward(self, x: torch.Tensor, adaLN_emb: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(adaLN_emb).chunk(6, dim=-1)
        # Norm and modulate
        x_norm = self.norm(x)
        return x_norm, (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
