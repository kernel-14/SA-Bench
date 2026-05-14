## model/attention.py
"""
Attention modules for Ca2‑VDM.

Classes:
    CausalTemporalAttention  -- causal temporal attention with KV‑cache
    PrefixEnhancedSpatialAttention -- spatial self‑attention enhanced by a short
                                      prefix of clean frames
    CrossAttention           -- cross‑attention to text embeddings (T5)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalTemporalAttention(nn.Module):
    """
    Causal temporal attention with optional KV‑cache for autoregressive inference.

    During training the full sequence is processed with a lower‑triangular mask.
    During inference the keys/values of previously generated (clean) frames are
    provided as a cache and concatenated to the current noisy frames' K,V.
    When ``write_cache=True`` the layer returns the clean K,V of the current
    frames for storage, without modifying the attention logic.
    """

    def __init__(self, dim: int, num_heads: int, config) -> None:
        """
        Args:
            dim: Hidden dimension (must be divisible by num_heads).
            num_heads: Number of attention heads.
            config: Global configuration (not actively used, kept for uniformity).
        """
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Joint QKV projection
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cache_k: Optional[torch.Tensor] = None,
        cache_v: Optional[torch.Tensor] = None,
        write_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape ``(B, L, N, D)`` where
                B=batch, L=number of frames in this step,
                N=number of spatial tokens, D=hidden dim.
            cache_k: Optional tensor of shape ``(B*N, H, P_k, d_h)`` containing
                     cached clean keys from previous autoregressive steps.
            cache_v: Same shape as cache_k for cached clean values.
            write_cache: If True, return the clean K,V of the current frames
                         (before any cache concatenation) for storage.

        Returns:
            output: Attended tensor of shape ``(B, L, N, D)``.
            new_cache_k: If ``write_cache`` is True, tensor of shape
                         ``(B*N, H, L, d_h)``; otherwise None.
            new_cache_v: Same as new_cache_k.
        """
        B, L, N, D = x.shape
        H = self.num_heads
        d_h = self.head_dim

        # 1) Permute to treat spatial dimension as batch: (B*N, L, D)
        x_t = x.permute(0, 2, 1, 3).reshape(B * N, L, D)

        # 2) Compute Q, K, V jointly and split -> each (B*N, L, D)
        qkv = self.qkv_proj(x_t)                    # (B*N, L, 3*D)
        Q, K, V = qkv.chunk(3, dim=-1)

        # 3) Reshape to multi‑head: (B*N, H, L, d_h)
        Q = Q.view(B * N, L, H, d_h).transpose(1, 2)
        K = K.view(B * N, L, H, d_h).transpose(1, 2)
        V = V.view(B * N, L, H, d_h).transpose(1, 2)

        # 4) If writing cache, save the current clean K,V before any concatenation
        if write_cache:
            new_cache_k = K.detach()   # (B*N, H, L, d_h)
            new_cache_v = V.detach()
        else:
            new_cache_k = None
            new_cache_v = None

        # 5) Handle cached prefix frames
        if cache_k is not None and cache_v is not None:
            # cache has shape (B*N, H, P_k, d_h)
            P_k = cache_k.size(2)
            K = torch.cat([cache_k, K], dim=2)   # (B*N, H, P_k+L, d_h)
            V = torch.cat([cache_v, V], dim=2)
            L_total = P_k + L
        else:
            P_k = 0
            L_total = L

        # 6) Build causal mask: shape (1, 1, L, L_total)
        #    For query index i (0-based) allow attention to:
        #      - all cached frames (indices 0..P_k-1)
        #      - current frames j where j <= i
        #    This is equivalent to: mask[:, P_k:] = triu(-inf, diag=1)
        mask = torch.zeros(L, L_total, device=x.device, dtype=x.dtype)
        mask[:, P_k:] = torch.triu(
            torch.full((L, L), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        # mask entries are 0 (allowed) or -inf, add to attention scores

        # 7) Scaled dot‑product attention
        #    Q: (B*N, H, L, d_h), K,V: (B*N, H, L_total, d_h)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B*N, H, L, L_total)
        attn_weights = attn_weights + mask.unsqueeze(0).unsqueeze(0)      # broadcast over B*N, H
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V)                        # (B*N, H, L, d_h)

        # 8) Merge heads: (B*N, L, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B * N, L, D)
        attn_output = self.out_proj(attn_output)

        # 9) Reshape back to (B, L, N, D)
        output = attn_output.view(B, N, L, D).permute(0, 2, 1, 3)

        return output, new_cache_k, new_cache_v


# ---------------------------------------------------------------------------
# PrefixEnhancedSpatialAttention
# ---------------------------------------------------------------------------

class PrefixEnhancedSpatialAttention(nn.Module):
    """
    Spatial self‑attention enhanced by a short prefix of clean frames.

    The enhancement follows Eq. (4) of the paper:
      - For a clean frame (i < P): keys/values are repeated ``prefix_length``
        times spatially.
      - For a noisy frame (i >= P): its own keys/values are concatenated with
        the raw (un‑enhanced) keys/values of the last ``prefix_length`` clean
        frames.

    During inference denoising, the clean prefix frames are not part of the
    input; instead their raw spatial K,V are provided via a cache.  When
    ``write_cache=True`` the layer returns the raw (un‑enhanced) K,V of the
    current frames for storage, which will serve as prefix for the next
    autoregressive step.
    """

    def __init__(self, dim: int, num_heads: int, prefix_length: int, config) -> None:
        """
        Args:
            dim: Hidden dimension.
            num_heads: Number of attention heads.
            prefix_length: p_prime, e.g. 3.
            config: Global configuration (unused, for interface consistency).
        """
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.prefix_length = prefix_length

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        is_clean_mask: torch.Tensor,
        spatial_cache_k: Optional[torch.Tensor] = None,
        spatial_cache_v: Optional[torch.Tensor] = None,
        write_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor ``(B, L, N, D)``.
            is_clean_mask: Boolean tensor ``(B, L)``, True for clean frames,
                           False for noisy frames.
            spatial_cache_k/v: Optional cached raw keys/values from previous
                               chunk, shape ``(B, P', H, N, d_h)``.  Only
                               provided during inference denoising (all frames
                               noisy).
            write_cache: If True, return the raw (un‑enhanced) K,V of the
                         current frames.

        Returns:
            output: Attended tensor ``(B, L, N, D)``.
            new_cache_k: If write_cache==True, raw K ``(B, L, H, N, d_h)``,
                         else None.
            new_cache_v: Same format.
        """
        B, L, N, D = x.shape
        H = self.num_heads
        d_h = self.head_dim
        P_p = self.prefix_length

        # 1) Compute Q for all frames: (B*L, N, D) after flattening heads later
        x_flat = x.view(B * L, N, D)
        Q = self.q_proj(x_flat).view(B * L, N, H, d_h).transpose(1, 2)  # (B*L, H, N, d_h)

        # 2) Compute base K,V for all frames: (B, L, H, N, d_h)
        kv = self.kv_proj(x)  # (B, L, N, 2*D)
        K_base, V_base = kv.chunk(2, dim=-1)  # each (B, L, N, D)
        K_base = K_base.view(B, L, N, H, d_h).transpose(2, 3)  # (B, L, H, N, d_h)
        V_base = V_base.view(B, L, N, H, d_h).transpose(2, 3)

        # ---- Handle cache writing: all frames are clean, return raw K,V ----
        if write_cache:
            # For clean frames the attention uses self‑repeat; we still need to
            # run the forward pass to maintain correct hidden states, but the
            # returned cache is the *raw* K,V (un‑enhanced).
            # Build enhanced K,V for attention: repeat each frame's K,V P_p times.
            K_enh = K_base.repeat(1, 1, 1, 1, P_p).reshape(B, L, H, N * P_p, d_h)
            V_enh = V_base.repeat(1, 1, 1, 1, P_p).reshape(B, L, H, N * P_p, d_h)

            # Attend
            # Q shape: (B*L, H, N, d_h) -> need to expand to (B*L, H, N, d_h) once per frame
            # K_enh shape: (B, L, H, N*P_p, d_h) -> reshape to (B*L, H, N*P_p, d_h)
            K_enh_flat = K_enh.reshape(B * L, H, N * P_p, d_h)
            V_enh_flat = V_enh.reshape(B * L, H, N * P_p, d_h)

            attn_weights = torch.matmul(Q, K_enh_flat.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, V_enh_flat)  # (B*L, H, N, d_h)

            # Merge heads & project
            attn_output = attn_output.transpose(1, 2).contiguous().view(B * L, N, D)
            attn_output = self.out_proj(attn_output)
            output = attn_output.view(B, L, N, D)

            # Return raw K,V as cache
            new_cache_k = K_base.detach()   # (B, L, H, N, d_h)
            new_cache_v = V_base.detach()
            return output, new_cache_k, new_cache_v

        # ---- Inference denoising: all frames noisy, prefix from cache ----
        if spatial_cache_k is not None and spatial_cache_v is not None:
            # cache shape: (B, P_p, H, N, d_h) – ensure it's exactly P_p
            # For each noisy frame, enhance K = cat(cache, K_base_i)
            # Flatten cache spatially: (B, H, P_p*N, d_h)
            cache_k_flat = spatial_cache_k.permute(0, 2, 1, 3, 4).reshape(B, H, P_p * N, d_h)
            cache_v_flat = spatial_cache_v.permute(0, 2, 1, 3, 4).reshape(B, H, P_p * N, d_h)

            # Expand over L frames: (B, L, H, P_p*N, d_h)
            cache_k_flat = cache_k_flat.unsqueeze(1).expand(-1, L, -1, -1, -1)
            cache_v_flat = cache_v_flat.unsqueeze(1).expand(-1, L, -1, -1, -1)

            # Concatenate along spatial dim -> (B, L, H, N + P_p*N, d_h)
            K_enh = torch.cat([cache_k_flat, K_base], dim=3)
            V_enh = torch.cat([cache_v_flat, V_base], dim=3)

            K_enh_flat = K_enh.reshape(B * L, H, N * (P_p + 1), d_h)
            V_enh_flat = V_enh.reshape(B * L, H, N * (P_p + 1), d_h)

            attn_weights = torch.matmul(Q, K_enh_flat.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, V_enh_flat)

            attn_output = attn_output.transpose(1, 2).contiguous().view(B * L, N, D)
            attn_output = self.out_proj(attn_output)
            output = attn_output.view(B, L, N, D)

            return output, None, None

        # ---- Training mode: mixture of clean and noisy frames ----
        # Separate indices
        clean_mask = is_clean_mask   # (B, L)
        noisy_mask = ~clean_mask

        # Number of clean frames per sample (should be the same across batch? Not necessarily,
        # but we assume all samples have the same split for simplicity, which is true
        # with our dataset design where each batch uses same prefix length.
        # For robustness we handle per‑sample counts.
        B_idx = torch.arange(B, device=x.device)

        # Count clean frames per sample
        P_per_sample = clean_mask.sum(dim=1)   # (B,)

        # Gather indices of clean frames: (total_clean,) indices
        clean_indices = clean_mask.nonzero(as_tuple=False)   # (K, 2) where K = total clean frames

        # Prepare empty output tensor, later scatter
        output = torch.empty(B, L, N, D, device=x.device, dtype=x.dtype)

        # ---------- Process noisy frames ----------
        if noisy_mask.any():
            # For noisy frames we need the last P_p clean frames (raw K,V).
            # If some samples have fewer clean frames than P_p, we pad by
            # repeating the last available clean frame.
            # Implementation: for each sample b, collect the K,V of the last
            # min(P_p, P_b) clean frames, then pad to exactly P_p frames.

            # Create list of per-sample prefix K,V, padded.
            # This is a bit involved but we can vectorize with gather.
            prefix_K_list = []
            prefix_V_list = []
            for b in range(B):
                P_b = P_per_sample[b]
                if P_b == 0:
                    # No clean frames -> prefix is empty; add empty tensor.
                    # We'll handle later: for frames with no prefix, enhanced K = just their own K.
                    # We can represent empty prefix as zero-sized tensor and skip.
                    # But easier: treat as P_p frames of zeros? However, the paper likely
                    # doesn't use prefix enhancement when P=0. We'll add empty placeholder.
                    prefix_K_list.append(None)
                    prefix_V_list.append(None)
                    continue
                # take last min(P_p, P_b) clean frames
                k_p = min(P_p, P_b)
                clean_b_indices = clean_indices[clean_indices[:, 0] == b][-k_p:, 1]  # frame indices
                # gather raw K,V of these frames
                K_clean_b = K_base[b, clean_b_indices]   # (k_p, H, N, d_h)
                V_clean_b = V_base[b, clean_b_indices]
                # if k_p < P_p, pad by repeating last frame
                if k_p < P_p:
                    repeat_times = P_p - k_p
                    K_clean_b = torch.cat([K_clean_b, K_clean_b[-1:].repeat(repeat_times, 1, 1, 1)], dim=0)
                    V_clean_b = torch.cat([V_clean_b, V_clean_b[-1:].repeat(repeat_times, 1, 1, 1)], dim=0)
                # shape (P_p, H, N, d_h)
                prefix_K_list.append(K_clean_b)
                prefix_V_list.append(V_clean_b)

            # Now process each noisy frame: concatenate prefix_K of its sample with its own K
            noisy_frame_indices = noisy_mask.nonzero(as_tuple=False)  # (M, 2)
            if noisy_frame_indices.numel() > 0:
                # We'll process noisy frames in a loop for clarity, although could be batched with
                # advanced indexing, but it's small overhead.
                for b, l_idx in noisy_frame_indices:
                    P_b = P_per_sample[b]
                    if P_b == 0:
                        # no prefix: enhanced = only own K,V
                        K_enh_i = K_base[b, l_idx].unsqueeze(0)   # (1, H, N, d_h)
                        V_enh_i = V_base[b, l_idx].unsqueeze(0)
                    else:
                        prefix_K_b = prefix_K_list[b]   # (P_p, H, N, d_h)
                        prefix_V_b = prefix_V_list[b]
                        # reshape to (H, P_p*N, d_h)
                        prefix_K_b_flat = prefix_K_b.permute(1, 0, 2, 3).reshape(H, P_p * N, d_h)
                        prefix_V_b_flat = prefix_V_b.permute(1, 0, 2, 3).reshape(H, P_p * N, d_h)
                        own_K = K_base[b, l_idx]   # (H, N, d_h)
                        own_V = V_base[b, l_idx]
                        own_K_flat = own_K.reshape(H, N, d_h)
                        own_V_flat = own_V.reshape(H, N, d_h)
                        # concatenate along spatial dim
                        K_enh_i = torch.cat([prefix_K_b_flat, own_K_flat], dim=1)   # (H, N+P_p*N, d_h)
                        V_enh_i = torch.cat([prefix_V_b_flat, own_V_flat], dim=1)
                        K_enh_i = K_enh_i.unsqueeze(0)  # (1, H, ...)
                        V_enh_i = V_enh_i.unsqueeze(0)

                    # Q for this frame: Q was computed for all frames, select this one
                    q_i = Q.view(B, L, H, N, d_h)[b, l_idx]  # (H, N, d_h)
                    # Compute attention for this single frame
                    q_i = q_i.unsqueeze(0)  # (1, H, N, d_h)
                    attn_w = torch.matmul(q_i, K_enh_i.transpose(-2, -1)) * self.scale
                    attn_w = F.softmax(attn_w, dim=-1)
                    attn_out = torch.matmul(attn_w, V_enh_i)   # (1, H, N, d_h)
                    attn_out = attn_out.transpose(1, 2).contiguous().view(1, N, D)
                    attn_out = self.out_proj(attn_out)
                    output[b, l_idx] = attn_out.squeeze(0)

        # ---------- Process clean frames ----------
        if clean_mask.any():
            clean_frame_indices = clean_indices   # (K, 2)
            # For clean frames: enhanced K,V = repeat own K,V P_p times
            for b, l_idx in clean_frame_indices:
                K_clean_frame = K_base[b, l_idx]  # (H, N, d_h)
                V_clean_frame = V_base[b, l_idx]
                # Repeat along spatial dim
                K_enh = K_clean_frame.unsqueeze(0).repeat(P_p, 1, 1, 1).reshape(1, H, N * P_p, d_h)
                V_enh = V_clean_frame.unsqueeze(0).repeat(P_p, 1, 1, 1).reshape(1, H, N * P_p, d_h)

                q_i = Q.view(B, L, H, N, d_h)[b, l_idx].unsqueeze(0)  # (1, H, N, d_h)
                attn_w = torch.matmul(q_i, K_enh.transpose(-2, -1)) * self.scale
                attn_w = F.softmax(attn_w, dim=-1)
                attn_out = torch.matmul(attn_w, V_enh)
                attn_out = attn_out.transpose(1, 2).contiguous().view(1, N, D)
                attn_out = self.out_proj(attn_out)
                output[b, l_idx] = attn_out.squeeze(0)

        return output, None, None


# ---------------------------------------------------------------------------
# CrossAttention
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """
    Standard cross‑attention between video tokens and text (T5) embeddings.

    The query comes from the video hidden states while key/value come from
    text embeddings.  No caching is needed; the text is constant across
    frames and timesteps.
    """

    def __init__(self, dim: int, num_heads: int, config) -> None:
        """
        Args:
            dim: Hidden dimension of video tokens.
            num_heads: Number of attention heads.
            config: Global configuration (unused).
        """
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=False)  # text features are projected to dim if necessary
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Video hidden states ``(B, L, N, D)``.
            text_emb: Text token embeddings ``(B, L_text, D_text)``.
                      The key/value projection expects input dim ``D``,
                      so an external projection may be needed if D_text != D.

        Returns:
            Attended tensor ``(B, L, N, D)``.
        """
        B, L, N, D = x.shape
        L_text = text_emb.size(1)

        # 1) Query: flatten video tokens to (B*L*N, D)
        x_flat = x.reshape(B * L * N, D)
        Q = self.q_proj(x_flat).view(B * L * N, self.num_heads, self.head_dim).transpose(0, 1)  # (H, B*L*N, d_h)

        # 2) Key/Value from text: shape (B, L_text, D)
        #    Assume text_emb already has dimension D (or is projected externally)
        kv = self.kv_proj(text_emb)  # (B, L_text, 2*D)
        K_text, V_text = kv.chunk(2, dim=-1)  # each (B, L_text, D)
        K_text = K_text.view(B, L_text, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, L_text, d_h)
        V_text = V_text.view(B, L_text, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 3) Expand key/value to match the flattened video tokens:
        #    (B, H, L_text, d_h) -> (B*L*N, H, L_text, d_h) flattening
        K = K_text.unsqueeze(1).expand(-1, L * N, -1, -1, -1).reshape(B * L * N, self.num_heads, L_text, self.head_dim)
        V = V_text.unsqueeze(1).expand(-1, L * N, -1, -1, -1).reshape(B * L * N, self.num_heads, L_text, self.head_dim)

        # K,V shape (B*L*N, H, L_text, d_h); Q shape (H, B*L*N, d_h)
        # For efficient matmul, transpose Q to (B*L*N, H, d_h, 1) etc.
        Q = Q.transpose(0, 1)  # (B*L*N, H, d_h)
        Q = Q.unsqueeze(2)      # (B*L*N, H, 1, d_h)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B*L*N, H, 1, L_text)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (B*L*N, H, 1, d_h)

        attn_output = attn_output.squeeze(2)          # (B*L*N, H, d_h)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B * L * N, D)
        attn_output = self.out_proj(attn_output)

        # Reshape back to (B, L, N, D)
        output = attn_output.view(B, L, N, D)
        return output
