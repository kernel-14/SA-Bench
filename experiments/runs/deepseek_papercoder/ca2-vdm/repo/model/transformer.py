## model/transformer.py
"""
Spatial‑Temporal Transformer backbone for Ca2‑VDM.

Implements the core transformer architecture with per‑frame adaptive
layer normalisation (adaLN), causal temporal attention, prefix‑enhanced
spatial attention, cross‑attention, and feed‑forward blocks.

The design aligns with the paper's description:
- Spatial positional embeddings (SPE) are sinusoidal and shared across frames.
- Temporal positional embeddings (TPE) are cyclic, enabling KV‑cache sharing
  beyond the training length.
- Dual timestep embedding: clean prefix frames receive tEmb(0), noisy frames
  receive tEmb(t).
- The forward pass supports both training (full clean+noisy sequence) and
  autoregressive inference with optional KV‑caches.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from model.attention import (
    CausalTemporalAttention,
    CrossAttention,
    PrefixEnhancedSpatialAttention,
)
from utils.positional_encodings import get_sinusoidal_encoding


# ---------------------------------------------------------------------------
# Helper modules
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar diffusion timesteps into a vector via sinusoidal encoding + MLP.

    The input timesteps are first normalised to [0, 1] by dividing by 1000,
    then converted to a fixed‑length frequency embedding, and finally mapped
    to the transformer hidden dimension through a small MLP.
    """

    def __init__(self, hidden_dim: int, freq_dim: int = 256, max_period: float = 10000.0) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestep tensor of shape ``(...,)``.  Values are expected to be
               integers in ``[0, 1000]``.  The leading dimensions are preserved
               and passed through the embedding.

        Returns:
            Tensor of shape ``(*, hidden_dim)``.
        """
        t_norm = t.float() / 1000.0  # normalise to [0,1]
        # Compute sinusoidal embedding
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t_norm.unsqueeze(-1) * freqs  # (..., half)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.freq_dim % 2 == 1:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[..., :1])], dim=-1
            )
        return self.mlp(embedding)


class FrameWiseAdaLN(nn.Module):
    """
    Per‑frame adaptive layer normalisation with modulation (scale, shift, gate).

    Given a per‑frame conditioning embedding ``emb`` of shape ``(B, L, D)``,
    this module applies LayerNorm to the input tensor ``x`` of shape
    ``(B, L, N, D)`` and then modulates the normalised output element‑wise.

    The modulation parameters are predicted by a small MLP from ``emb``.
    A gating vector is also returned to scale the residual pathway, following
    the DiT architecture.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 3 * dim),
        )
        # Initialise near‑identity: zero modulation parameters
        nn.init.constant_(self.modulation[-1].weight, 0)
        nn.init.constant_(self.modulation[-1].bias, 0)

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:   Input tensor ``(B, L, N, D)``.
            emb: Conditioning embedding ``(B, L, D)``.

        Returns:
            modulated output of same shape as ``x``, and a gating tensor of
            shape ``(B, L, 1, 1)`` that can be used to gate the residual connection.
        """
        normed = self.norm(x)
        mod = self.modulation(emb)  # (B, L, 3*D)
        scale, shift, gate = mod.chunk(3, dim=-1)  # each (B, L, D)
        # scale and shift are broadcast over N
        normed = normed * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)
        gate = gate.unsqueeze(2)  # (B, L, 1, D) – broadcast over N
        return normed, gate


class FeedForward(nn.Module):
    """
    Simple two‑layer MLP with GELU activation, used as the feed‑forward
    sub‑layer in each transformer block.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        inner_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    A single transformer block containing:

    - prefix‑enhanced spatial attention,
    - causal temporal attention,
    - cross‑attention (optional, only for text‑conditioned tasks),
    - feed‑forward network.

    Each sub‑layer is preceded by a dedicated FrameWiseAdaLN module that
    modulates the input features according to the per‑frame timestep embedding.
    Residual connections are gated by the modulation gate vector.
    """

    def __init__(self, dim: int, num_heads: int, config: Config) -> None:
        super().__init__()
        p_prime = config.video.p_prime

        # Attention layers
        self.spatial_attn = PrefixEnhancedSpatialAttention(dim, num_heads, p_prime, config)
        self.temporal_attn = CausalTemporalAttention(dim, num_heads, config)
        self.cross_attn: Optional[nn.Module] = None
        if config.task == "t2v":
            self.cross_attn = CrossAttention(dim, num_heads, config)

        # Feed‑forward
        mlp_ratio = config.model.transformer.mlp_ratio
        self.ffn = FeedForward(dim, mlp_ratio)

        # Adaptive layer norm modules (one per sub‑layer)
        self.norm_spatial = FrameWiseAdaLN(dim)
        self.norm_temporal = FrameWiseAdaLN(dim)
        self.norm_cross: Optional[FrameWiseAdaLN] = None
        if self.cross_attn is not None:
            self.norm_cross = FrameWiseAdaLN(dim)
        self.norm_ffn = FrameWiseAdaLN(dim)

    def forward(
        self,
        x: torch.Tensor,
        t_emb_frames: torch.Tensor,
        is_clean_mask: torch.Tensor,
        text_emb: Optional[torch.Tensor],
        temp_k: Optional[torch.Tensor],
        temp_v: Optional[torch.Tensor],
        spat_k: Optional[torch.Tensor],
        spat_v: Optional[torch.Tensor],
        write_cache: bool,
    ) -> Tuple[
        torch.Tensor,
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor]],
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor]],
    ]:
        """
        Args:
            x:              Input hidden states ``(B, L, N, D)``.
            t_emb_frames:   Per‑frame timestep embedding ``(B, L, D)``.
            is_clean_mask:  Boolean mask ``(B, L)`` indicating which frames
                            are clean (timestep 0).
            text_emb:       Optional text token embeddings ``(B, L_txt, D)``.
            temp_k, temp_v: Optional temporal KV‑cache from previous steps,
                            shape ``(B*N, H, P_k, d_h)`` (or ``None``).
            spat_k, spat_v: Optional spatial prefix cache from previous chunk,
                            shape ``(B, P', H, N, d_h)`` (or ``None``).
            write_cache:    If True, the attention layers will return their
                            clean K,V for cache update.

        Returns:
            - Updated hidden tensor ``(B, L, N, D)``.
            - Tuple of new temporal cache tensors ``(temp_k, temp_v)``,
              each either ``None`` or ``(B*N, H, L, d_h)``.
            - Tuple of new spatial cache tensors ``(spat_k, spat_v)``,
              each either ``None`` or ``(B, L, H, N, d_h)``.
        """
        # ------------------- spatial attention -------------------
        norm_x, gate_sp = self.norm_spatial(x, t_emb_frames)
        attn_sp_out, new_spat_k, new_spat_v = self.spatial_attn(
            norm_x,
            is_clean_mask,
            spatial_cache_k=spat_k,
            spatial_cache_v=spat_v,
            write_cache=write_cache,
        )
        x = x + gate_sp * attn_sp_out

        # ------------------- temporal attention -------------------
        norm_x, gate_tp = self.norm_temporal(x, t_emb_frames)
        attn_tp_out, new_temp_k, new_temp_v = self.temporal_attn(
            norm_x,
            cache_k=temp_k,
            cache_v=temp_v,
            write_cache=write_cache,
        )
        x = x + gate_tp * attn_tp_out

        # ------------------- cross‑attention (optional) -----------
        if self.cross_attn is not None and text_emb is not None:
            norm_x, gate_cr = self.norm_cross(x, t_emb_frames)
            attn_cr_out = self.cross_attn(norm_x, text_emb)
            x = x + gate_cr * attn_cr_out

        # ------------------- feed‑forward -------------------------
        norm_x, gate_ff = self.norm_ffn(x, t_emb_frames)
        ff_out = self.ffn(norm_x)
        x = x + gate_ff * ff_out

        return (
            x,
            (new_temp_k, new_temp_v),
            (new_spat_k, new_spat_v),
        )


# ---------------------------------------------------------------------------
# Spatial‑Temporal Transformer
# ---------------------------------------------------------------------------

class SpatialTemporalTransformer(nn.Module):
    """
    The core transformer that processes spatial‑temporal latent tokens.

    It adds fixed spatial and cyclic temporal positional embeddings, embeds
    per‑frame timesteps, and then applies a stack of ``TransformerBlock``s.
    When ``write_cache`` is True, the model returns per‑layer K,V caches
    for both temporal and spatial attention, enabling autoregressive inference
    with KV‑cache sharing.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        dim = config.model.transformer.hidden_dim
        num_heads = config.model.transformer.num_heads
        num_layers = config.model.transformer.num_layers
        latent_size = config.data.latent_size
        L_train = config.video.train_max_len

        # Spatial positional embedding (fixed sinusoidal, 1 per patch)
        num_patches = latent_size * latent_size
        spe = get_sinusoidal_encoding(num_patches, dim)  # (N, D)
        self.register_buffer("spe", spe.unsqueeze(0).unsqueeze(0))  # (1, 1, N, D)

        # Cyclic temporal positional embeddings (base table, L_train positions)
        tpe_base = get_sinusoidal_encoding(L_train, dim)  # (L_train, D)
        self.register_buffer("tpe_base", tpe_base)

        # Timestep embedder
        self.t_embedder = TimestepEmbedder(dim)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads, config) for _ in range(num_layers)]
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(dim)
        self.L_train = L_train

    def prepare_cache_writing(self) -> None:
        """
        Placeholder method to conform to the design interface.
        The actual cache writing mode is controlled via the ``write_cache``
        argument in :meth:`forward`.
        """
        pass

    def forward(
        self,
        hidden: torch.Tensor,
        timestep: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        temporal_cache: Optional[Dict[str, List[Optional[torch.Tensor]]]] = None,
        spatial_cache: Optional[Dict[str, List[Optional[torch.Tensor]]]] = None,
        write_cache: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            hidden:         Latent frames ``(B, L, N, D)``.
            timestep:       Tensor of shape ``(B, L)`` containing diffusion
                            timesteps.  0 = clean frame, t = noisy.
            text_emb:       Optional text token embeddings ``(B, L_txt, D)``.
            temporal_cache: Optional dict with keys ``'k'`` and ``'v'``, each
                            a list of per‑layer tensors (length ``num_layers``,
                            each ``(B*N, H, P_k, d_h)``).  May also contain
                            the key ``'tpe_start_idx'`` (int) for inference
                            to assign cyclic TPEs.
            spatial_cache:  Optional dict with keys ``'k'`` and ``'v'``, each
                            a list of per‑layer tensors (length ``num_layers``,
                            each ``(B, P', H, N, d_h)``).
            write_cache:    If True, the method returns a tuple ``(hidden,
                            cache_out)`` where ``cache_out`` contains the new
                            temporal and spatial K,V for all layers.

        Returns:
            If ``write_cache`` is False: output tensor ``(B, L, N, D)``.
            Otherwise: ``(output, cache_out)`` where ``cache_out`` is a dict
            with ``'temporal': {'k': [...], 'v': [...]}`` and
            ``'spatial': {'k': [...], 'v': [...]}``.
        """
        B, L, N, D = hidden.shape

        # ---- 1. Spatial positional embedding ----
        # spe shape: (1, 1, N, D) broadcasts to (B, L, N, D)
        hidden = hidden + self.spe

        # ---- 2. Temporal positional embedding (cyclic) ----
        if self.training:
            # Sample a random cyclic shift per sample in the batch
            shift = torch.randint(0, self.L_train, (B,), device=hidden.device)
            global_idx = (
                shift.unsqueeze(1) + torch.arange(L, device=hidden.device).unsqueeze(0)
            ) % self.L_train
        else:
            # Inference: determine start index from temporal_cache, default 0.
            tpe_start_idx = 0
            if temporal_cache is not None and isinstance(temporal_cache, dict):
                tpe_start_idx = temporal_cache.get("tpe_start_idx", 0)
            global_idx = (
                tpe_start_idx + torch.arange(L, device=hidden.device)
            ) % self.L_train
            global_idx = global_idx.unsqueeze(0).expand(B, -1)

        global_idx = global_idx.long()
        tpe = self.tpe_base[global_idx]  # (B, L, D)
        hidden = hidden + tpe.unsqueeze(2)  # (B, L, 1, D)

        # ---- 3. Per‑frame timestep embedding ----
        # Ensure timestep has shape (B, L)
        if timestep.dim() == 1:
            # Single scalar t → all frames have the same t
            timestep = timestep.unsqueeze(0).expand(B, L)
        elif timestep.dim() == 2:
            if timestep.shape[0] == 1 and B > 1:
                timestep = timestep.expand(B, -1)
            if timestep.shape != (B, L):
                raise ValueError(
                    f"timestep shape must be (B, L) or (1, L), got {timestep.shape}"
                )
        else:
            raise ValueError(f"timestep must be 1‑D or 2‑D, got shape {timestep.shape}")

        t_emb_frames = self.t_embedder(timestep)  # (B, L, D)
        is_clean_mask = timestep == 0  # (B, L)

        # ---- 4. Prepare per‑layer cache inputs ----
        temp_k_list: Optional[List[Optional[torch.Tensor]]] = None
        temp_v_list: Optional[List[Optional[torch.Tensor]]] = None
        spat_k_list: Optional[List[Optional[torch.Tensor]]] = None
        spat_v_list: Optional[List[Optional[torch.Tensor]]] = None

        if temporal_cache is not None:
            temp_k_list = temporal_cache["k"]
            temp_v_list = temporal_cache["v"]
        if spatial_cache is not None:
            spat_k_list = spatial_cache["k"]
            spat_v_list = spatial_cache["v"]

        # ---- 5. Iterate over blocks ----
        new_temporal_k: List[Optional[torch.Tensor]] = [] if write_cache else []
        new_temporal_v: List[Optional[torch.Tensor]] = [] if write_cache else []
        new_spatial_k: List[Optional[torch.Tensor]] = [] if write_cache else []
        new_spatial_v: List[Optional[torch.Tensor]] = [] if write_cache else []

        for i, block in enumerate(self.blocks):
            temp_k = temp_k_list[i] if temp_k_list else None
            temp_v = temp_v_list[i] if temp_v_list else None
            spat_k = spat_k_list[i] if spat_k_list else None
            spat_v = spat_v_list[i] if spat_v_list else None

            hidden, (t_k, t_v), (s_k, s_v) = block(
                hidden,
                t_emb_frames,
                is_clean_mask,
                text_emb,
                temp_k,
                temp_v,
                spat_k,
                spat_v,
                write_cache,
            )

            if write_cache:
                new_temporal_k.append(t_k)
                new_temporal_v.append(t_v)
                new_spatial_k.append(s_k)
                new_spatial_v.append(s_v)

        # ---- 6. Final norm ----
        hidden = self.final_norm(hidden)

        if write_cache:
            cache_out: Dict[str, Any] = {
                "temporal": {"k": new_temporal_k, "v": new_temporal_v},
                "spatial": {"k": new_spatial_k, "v": new_spatial_v},
            }
            return hidden, cache_out
        else:
            return hidden

