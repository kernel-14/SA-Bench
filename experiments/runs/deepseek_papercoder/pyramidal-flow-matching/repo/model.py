## model.py
"""
Pyramidal Flow Matching loss and MM-DiT model.

This module implements:
- `PyramidalFlowMatchingLoss` : the unified flow matching objective with spatial pyramid stages.
- `MMDiT` : a 2B‑parameter Multimodal Diffusion Transformer extending SD3 Medium with
    temporal RoPE, causal attention, and history conditioning.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (
    downsample,
    nearest_upsample,
    patchify,
    unpatchify,
)


class PyramidalFlowMatchingLoss:
    """
    Computes the pyramidal flow matching loss that jointly optimises generation and
    super‑resolution across K spatial pyramid stages.

    The loss regresses the model's velocity prediction against a target vector field
    defined by a piecewise linear interpolation between a noisier, compressed latent
    and a cleaner, higher‑resolution latent (Eqs. 5‑11).
    """

    def __init__(self, cfg: dict):
        """
        Args:
            cfg: The full configuration dict (must contain `model.pyramid` section).
        """
        pyramid = cfg["model"]["pyramid"]
        self.num_stages: int = pyramid["num_stages"]
        self.s: List[float] = list(pyramid["s"])   # finest‑first (k=0..K-1)
        self.e: List[float] = list(pyramid["e"])
        self.down_mode: str = pyramid.get("down_mode", "bilinear")
        self.up_mode: str = pyramid.get("up_mode", "nearest")
        self.patch_size: Tuple[int, int] = tuple(cfg["model"]["patch_size"])
        self.ph, self.pw = self.patch_size

        # Sanity checks
        assert self.num_stages == len(self.s) == len(self.e)
        for k in range(self.num_stages - 1):
            # Verify renoising recursion: e_{k+1} = 2*s_k / (1+s_k)
            expected_e = (2.0 * self.s[k]) / (1.0 + self.s[k])
            if not abs(self.e[k + 1] - expected_e) < 1e-4:
                raise ValueError(
                    f"Pyramid schedule violates renoising recursion for k={k}. "
                    f"e_{k+1}={self.e[k+1]:.4f} but expected {expected_e:.4f}."
                )

        # Mapping from stage idx (0 finest, K-1 coarsest) to downsample factor for x_e
        self._factors: List[int] = [2 ** (self.num_stages - 1 - k) for k in range(self.num_stages)]

    def sample_endpoints(
        self, clean_latent: torch.Tensor, stage: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Produces the two endpoints (start and end of the piecewise flow) for a given spatial stage.

        Args:
            clean_latent: Full‑resolution target latent, shape (B, C, H, W).
            stage: Stage index (0 = finest, K‑1 = coarsest).

        Returns:
            Tuple (x_e, x_s) both at spatial resolution `H//factor × W//factor`.
        """
        factor = self._factors[stage]
        s_k = self.s[stage]
        e_k = self.e[stage]

        # Downsample the target to current stage resolution
        x1_down = downsample(clean_latent, factor)

        # Lower‑resolution start: upsample from one level coarser
        x1_lower = nearest_upsample(downsample(clean_latent, factor * 2))

        # Shared noise for coupled sampling
        n = torch.randn_like(x1_down)

        x_e = e_k * x1_down + (1.0 - e_k) * n
        x_s = s_k * x1_lower + (1.0 - s_k) * n
        return x_e, x_s

    def compute_loss(
        self,
        model: "MMDiT",
        clean_latent: torch.Tensor,
        text_emb: torch.Tensor,
        history: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Performs a full training step with pyramidal flow matching.

        Args:
            model: The MM‑DiT model.
            clean_latent: Batched full‑resolution target latents, shape (B, C, H, W).
            text_emb: Text embeddings, shape (B, T_text, D_text).
            history: Optional history frames latents, shape (B, T_hist, C, H_h, W_h).
                     (Already compressed according to temporal pyramid and augmented with noise.)

        Returns:
            Scalar loss (MSE).
        """
        B, C, H, W = clean_latent.shape
        device = clean_latent.device

        # ----- 1. Sample stage and timestep -----
        stage = torch.randint(0, self.num_stages, (1,), device=device).item()
        s_k = self.s[stage]
        e_k = self.e[stage]
        t = s_k + (e_k - s_k) * torch.rand(1, device=device).item()

        # ----- 2. Compute endpoints and interpolation -----
        x_e, x_s = self.sample_endpoints(clean_latent, stage)
        t_prime = (t - s_k) / (e_k - s_k) if e_k > s_k else 0.0
        x_t = t_prime * x_e + (1.0 - t_prime) * x_s  # (B, C, H_s, W_s)

        # Target velocity (shape matches x_e/x_s)
        v_target = x_e - x_s

        # ----- 3. Model forward -----
        # Prepare timestep as float tensor
        timestep = torch.full((B,), t, device=device, dtype=clean_latent.dtype)

        # Build causal attention mask for the current batch composition
        if history is not None:
            T_hist = history.shape[1]
            Hh, Wh = history.shape[3], history.shape[4]
        else:
            T_hist = 0
            Hh = Wh = 0

        N_text = text_emb.shape[1]
        H_s, W_s = x_t.shape[2], x_t.shape[3]
        N_hist_per_frame = (Hh // self.ph) * (Wh // self.pw) if T_hist > 0 else 0
        N_curr = (H_s // self.ph) * (W_s // self.pw)

        attention_mask = self._build_attention_mask(
            B, N_text, T_hist, N_hist_per_frame, N_curr, device
        )

        # Call model with the noisy latent and history
        v_pred = model(
            noisy_latent=x_t,
            timestep=timestep,
            context=text_emb,
            history=history,
            attention_mask=attention_mask,
        )  # (B, C, H_s, W_s)

        # ----- 4. Compute loss -----
        loss = F.mse_loss(v_pred, v_target)
        return loss

    def _build_attention_mask(
        self,
        B: int,
        N_text: int,
        T_hist: int,
        N_per_hist: int,
        N_curr: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Builds the causal attention mask for the MM‑DiT transformer.

        Within one sample's sequence, text tokens are visible to all, history tokens are
        visible to earlier frames and text, and the current frame tokens are visible to
        all history and text but not to future frames (no future frames exist).

        The mask has shape (B, 1, total_seq, total_seq) with 0.0 for allowed attention
        and -inf for disallowed. Cross‑sample attention is disabled by the batch dimension.

        Args:
            B: Batch size.
            N_text: Number of text tokens.
            T_hist: Number of history frames.
            N_per_hist: Number of patch tokens per history frame.
            N_curr: Number of patch tokens for the current frame.
            device: Torch device.

        Returns:
            Float mask tensor.
        """
        total = N_text + T_hist * N_per_hist + N_curr
        if total == 0:
            raise ValueError("Total sequence length is zero.")

        # Assign frame indices
        frame_ids = torch.zeros(total, dtype=torch.long, device=device)
        # Text: large negative frame index so that they are visible to all
        frame_ids[:N_text] = -1000

        # History frames: each block gets frame index -T_hist + f (f in [0, T_hist-1])
        for f in range(T_hist):
            start = N_text + f * N_per_hist
            frame_ids[start:start + N_per_hist] = -T_hist + f

        # Current frame: index 0
        frame_ids[-N_curr:] = 0

        # Build mask: token i can attend to token j if frame_ids[j] <= frame_ids[i]
        # Use broadcasting: (total, 1) >= (1, total)
        mask = (frame_ids.unsqueeze(0) >= frame_ids.unsqueeze(1))  # (total, total) bool
        float_mask = torch.where(
            mask,
            torch.tensor(0.0, device=device),
            torch.tensor(-float("inf"), device=device),
        )
        # Expand to (1, 1, total, total) then to (B, 1, total, total)
        float_mask = float_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, total, total).contiguous()
        return float_mask


# -------------------------------------------------------------------------
# Rotary Position Embedding (RoPE) helpers
# -------------------------------------------------------------------------

def precompute_freqs_cis(dim: int, max_pos: int = 2048, theta: float = 10000.0):
    """Precompute complex exponentials for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_pos).float()
    freqs = torch.outer(t, freqs)  # (max_pos, dim//2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex


def apply_rotary_emb_qk(
    q: torch.Tensor, k: torch.Tensor, t_idx: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies RoPE to query and key tensors based on temporal indices.

    Args:
        q: Query tensor, shape (...seq, head_dim) – will be handled per batch element.
        k: Key tensor, same shape.
        t_idx: Temporal indices, shape (B, seq_len). Non‑negative values are used;
               tokens with t_idx < 0 are left unchanged (text tokens).
        freqs_cis: Precomputed complex rotation matrices, shape (max_pos, dim//2).

    Returns:
        Rotated q and k tensors of same shapes.
    """
    B, seq_len, num_heads, head_dim = q.shape
    # Reshape to (B, num_heads, seq_len, head_dim) for easier indexing
    q = q.permute(0, 2, 1, 3).contiguous()  # (B, heads, seq, head_dim)
    k = k.permute(0, 2, 1, 3).contiguous()

    # Only apply to tokens with t_idx >= 0
    apply_mask = t_idx >= 0  # (B, seq)
    # We'll process batch items individually if needed, or vectorise

    # Prepare index lookup: limit t_idx to valid range [0, max_pos)
    max_pos = freqs_cis.shape[0]
    t_idx_clamped = t_idx.clamp(min=0, max=max_pos - 1)  # (B, seq)

    # Gather rotation matrices for each token
    # freqs_cis shape: (max_pos, dim//2) complex
    freqs_cis_selected = freqs_cis[t_idx_clamped]  # (B, seq, dim//2)

    # Convert q, k to complex: treat head_dim as pairs of real/imag
    q_complex = torch.view_as_complex(
        q.float().reshape(B, num_heads, seq_len, head_dim // 2, 2)
    )  # (B, heads, seq, head_dim//2) complex
    k_complex = torch.view_as_complex(
        k.float().reshape(B, num_heads, seq_len, head_dim // 2, 2)
    )

    # Apply rotation (element‑wise multiplication)
    # Expand batch dim to heads if necessary: freqs_cis_selected (B, seq, dim//2) -> (B, 1, seq, dim//2)
    freqs_cis_selected = freqs_cis_selected.unsqueeze(1).expand(B, num_heads, seq_len, -1)

    q_rotated = q_complex * freqs_cis_selected
    k_rotated = k_complex * freqs_cis_selected

    # Convert back to real
    q_out = torch.view_as_real(q_rotated).flatten(3)  # (B, heads, seq, head_dim)
    k_out = torch.view_as_real(k_rotated).flatten(3)

    # Reshape back to original format
    q_out = q_out.permute(0, 2, 1, 3)  # (B, seq, heads, head_dim)
    k_out = k_out.permute(0, 2, 1, 3)

    # For tokens where we should NOT apply RoPE, keep original q,k
    # Use mask: apply_mask (B, seq) -> (B, seq, 1, 1)
    apply_mask = apply_mask.unsqueeze(-1).unsqueeze(-1).float()
    q = q.permute(0, 2, 1, 3) * (1 - apply_mask) + q_out * apply_mask
    k = k.permute(0, 2, 1, 3) * (1 - apply_mask) + k_out * apply_mask

    return q, k


# -------------------------------------------------------------------------
# Transformer Block
# -------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """A single DiT block with adaLN‑modulated self‑attention and feed‑forward."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        causal: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.mlp_ratio = mlp_ratio
        self.causal = causal

        # Normalisation layers
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Attention projection weights
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=True)

        # Feed‑forward (GEGLU as in SD3)
        ffn_hidden = int(hidden_size * mlp_ratio * 2 / 3)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden * 2, bias=True),
            nn.GELU(),  # using GELU as approximation of GEGLU; SD3 uses GEGLU which is GELU gating
            nn.Linear(ffn_hidden, hidden_size, bias=True),
        )

        # adaLN modulation layers
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        t_idx: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input token embeddings (B, N, D).
            c: Conditioning vector (B, D) from timestep embedding.
            attention_mask: Additive mask (B, 1, N, N) with 0 for allowed, -inf for disallowed.
            t_idx: Temporal indices per token (B, N).
            freqs_cis: Precomputed RoPE frequencies.
        Returns:
            Updated token embeddings (B, N, D).
        """
        B, N, D = x.shape
        # adaLN modulation params
        mod_params = self.adaLN_modulation(c)  # (B, 6*D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_params.chunk(6, dim=1)

        # Self‑attention
        normed_x = self.norm1(x)
        normed_x = normed_x * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        qkv = self.qkv(normed_x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, head_dim)

        # Apply RoPE
        q, k = apply_rotary_emb_qk(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), t_idx, freqs_cis)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)

        # Scaled dot‑product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, heads, N, N)

        if attention_mask is not None:
            # attention_mask should be (B, 1, N, N) -> expand to heads
            if attention_mask.dim() == 4:
                attn_weights = attn_weights + attention_mask.expand(-1, self.num_heads, -1, -1)
            else:
                attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # (B, heads, N, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, N, D)
        attn_output = self.attn_out(attn_output)

        # Gate
        x = x + gate_msa[:, None, :] * attn_output

        # Feed‑forward
        normed_x = self.norm2(x)
        normed_x = normed_x * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        ffn_output = self.ffn(normed_x)
        x = x + gate_mlp[:, None, :] * ffn_output

        return x


# -------------------------------------------------------------------------
# MM‑DiT Model
# -------------------------------------------------------------------------
class MMDiT(nn.Module):
    """
    Multimodal Diffusion Transformer (2B parameters) extended from SD3 Medium.

    Incorporates temporal Rotary Position Embeddings, blockwise causal attention,
    and unified processing of text, history, and current noisy latent.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m_cfg = cfg["model"]
        self.hidden_size: int = m_cfg["hidden_size"]
        self.num_heads: int = m_cfg["attention_heads"]
        self.mlp_ratio: float = m_cfg["mlp_ratio"]
        self.num_layers: int = m_cfg["num_layers"]
        self.patch_size: Tuple[int, int] = tuple(m_cfg["patch_size"])
        self.ph, self.pw = self.patch_size
        self.context_dim: int = m_cfg["text_conditioning"]["context_dim"]
        self.causal_attention: bool = m_cfg.get("causal_attention", True)

        # ----- Token embedding layers -----
        # Patch embedding for video frames (both current and history)
        self.patch_embed = nn.Conv2d(
            cfg["vae"]["latent_channels"],
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        # Text projection
        self.text_proj = nn.Linear(self.context_dim, self.hidden_size, bias=True)

        # Timestep embedding
        self.t_embed = nn.Sequential(
            TimestepEmbedding(self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        # Positional encodings
        self.max_spatial_size: int = 128  # maximum spatial grid size (patches)
        self._precompute_spatial_pos_cache()

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                causal=self.causal_attention,
            )
            for _ in range(self.num_layers)
        ])

        # Output head
        output_dim = self.ph * self.pw * cfg["vae"]["latent_channels"]
        self.final_layer = nn.Linear(self.hidden_size, output_dim, bias=True)

        # RoPE precomputation
        self.rope_max_pos = 2048
        self.rope_theta = 10000.0
        self.freqs_cis = precompute_freqs_cis(self.hidden_size // self.num_heads, self.rope_max_pos, self.rope_theta)

        # Null context for CFG
        self.null_embed = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

        # Initialise weights (SD3 Medium pretrained will be loaded later)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _precompute_spatial_pos_cache(self):
        """
        Precomputes a 2D sinusoidal position embedding for a maximum grid size.
        Used for spatial positional encoding of patches.
        """
        max_h, max_w = self.max_spatial_size, self.max_spatial_size
        dim = self.hidden_size
        # Create 2D sinusoidal embeddings (following ViT, SD3)
        half_dim = dim // 2
        emb_h = torch.arange(max_h).float()
        emb_w = torch.arange(max_w).float()
        omega = torch.arange(half_dim // 2).float() / (half_dim // 2 - 1)
        omega = 1.0 / (10000 ** omega)

        # Height embedding
        emb_h = emb_h[:, None] * omega[None, :]  # (max_h, half_dim//2)
        emb_h = torch.cat([emb_h.sin(), emb_h.cos()], dim=-1)  # (max_h, half_dim)

        # Width embedding
        emb_w = emb_w[:, None] * omega[None, :]  # (max_w, half_dim//2)
        emb_w = torch.cat([emb_w.sin(), emb_w.cos()], dim=-1)  # (max_w, half_dim)

        # Combine: outer sum? Use grid layout: each patch at (i,j) gets row emb[i] + col emb[j]
        # We'll compute per position as emb_h[i, :half_dim] + emb_w[j, half_dim:] following common practice
        self.register_buffer(
            "spatial_pos_emb_h", emb_h, persistent=False
        )  # (max_h, half_dim)
        self.register_buffer(
            "spatial_pos_emb_w", emb_w, persistent=False
        )  # (max_w, half_dim)

    def get_2d_pos_embed(self, h_patches: int, w_patches: int) -> torch.Tensor:
        """
        Returns (1, h_patches * w_patches, hidden_size) spatial positional embeddings.
        Supports extrapolation by repeating the maximum available embeddings.
        """
        device = self.spatial_pos_emb_h.device
        # Use cache; if needed, interpolate or repeat
        emb_h = self.spatial_pos_emb_h[:h_patches] if h_patches <= self.max_spatial_size else F.interpolate(
            self.spatial_pos_emb_h[None, None].float(), size=h_patches, mode='linear', align_corners=False
        ).squeeze()
        emb_w = self.spatial_pos_emb_w[:w_patches] if w_patches <= self.max_spatial_size else F.interpolate(
            self.spatial_pos_emb_w[None, None].float(), size=w_patches, mode='linear', align_corners=False
        ).squeeze()

        # Build grid: each position (i,j)
        half_dim = emb_h.shape[-1]
        emb_h = emb_h[:, None, :]  # (h, 1, half_dim)
        emb_w = emb_w[None, :, :]  # (1, w, half_dim)
        # Combine by concatenating across half_dim? Actually standard: cos(row+col) but we can add.
        # We implement as sinusoidal based on row and col separately, then concatenate.
        # From ViT: pos_emb[i,j] = concat( sin( row*omega ), cos(row*omega), sin(col*omega), cos(col*omega) )
        # Our emb_h already contains sin/cos for rows (half_dim), emb_w for cols (half_dim).
        # So position (i,j) = concat( emb_h[i], emb_w[j] ) -> shape (half_dim + half_dim = hidden_size)
        grid_h = emb_h.repeat(1, w_patches, 1)  # (h, w, half_dim)
        grid_w = emb_w.repeat(h_patches, 1, 1)  # (h, w, half_dim)
        pos_embed = torch.cat([grid_h, grid_w], dim=-1)  # (h, w, hidden_size)
        pos_embed = pos_embed.reshape(1, h_patches * w_patches, -1)  # (1, N, hidden_size)
        return pos_embed.to(device)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        context: Optional[torch.Tensor],
        history: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass of the MM‑DiT.

        Args:
            noisy_latent: Current frame's noisy latent, shape (B, C, H_s, W_s).
            timestep: Tensor of shape (B,) with continuous time in [0,1].
            context: Text embedding tensor (B, T_text, D_text); None for unconditional.
            history: Optional history frames' latents, shape (B, T_hist, C, H_h, W_h).
            attention_mask: Additive mask (B, 1, N_total, N_total).

        Returns:
            Velocity prediction of shape (B, C, H_s, W_s).
        """
        B = noisy_latent.shape[0]
        device = noisy_latent.device

        # Timestep embedding
        c_emb = self.t_embed(timestep)  # (B, hidden_size)

        # ---- Tokenisation ----
        # Current frame patches
        curr_tokens = self.patch_embed(noisy_latent)  # (B, D, h_p, w_p)
        h_p, w_p = curr_tokens.shape[2], curr_tokens.shape[3]
        curr_tokens = curr_tokens.flatten(2).transpose(1, 2)  # (B, N_curr, D)
        N_curr = curr_tokens.shape[1]
        # Add spatial pos embed
        curr_tokens = curr_tokens + self.get_2d_pos_embed(h_p, w_p).expand(B, -1, -1)

        # History frames
        if history is not None:
            B, T_h, C_h, H_h, W_h = history.shape
            hist_tokens_list = []
            for f in range(T_h):
                frame = history[:, f]  # (B, C, H_h, W_h)
                tokens_f = self.patch_embed(frame)  # (B, D, h_ph, w_pw)
                h_ph, w_pw = tokens_f.shape[2], tokens_f.shape[3]
                tokens_f = tokens_f.flatten(2).transpose(1, 2)  # (B, N_h_f, D)
                # Spatial pos embed (extrapolate/interpolate)
                tokens_f = tokens_f + self.get_2d_pos_embed(h_ph, w_pw).expand(B, -1, -1)
                hist_tokens_list.append(tokens_f)
            hist_tokens = torch.cat(hist_tokens_list, dim=1)  # (B, T_h * N_h_f, D)
            N_hist = hist_tokens.shape[1]
            N_hist_per_frame = N_hist // T_h
        else:
            hist_tokens = None
            N_hist = 0
            T_h = 0
            N_hist_per_frame = 0

        # Text tokens
        if context is not None:
            txt_tokens = self.text_proj(context)  # (B, T_text, D)
        else:
            # Use learnable null embedding
            txt_tokens = self.null_embed.expand(B, 1, -1)  # (B, 1, D)
        N_text = txt_tokens.shape[1]
        # 1D position embedding for text (optional but used in SD3)
        # We use a simple learned or sinusoidal embedding; here sinusoidal for consistency
        txt_pos = self._get_1d_text_pos_embed(N_text, device)
        txt_tokens = txt_tokens + txt_pos.unsqueeze(0)

        # Concatenate sequence
        seq_parts = [txt_tokens]
        if hist_tokens is not None:
            seq_parts.append(hist_tokens)
        seq_parts.append(curr_tokens)
        x = torch.cat(seq_parts, dim=1)  # (B, total, D)
        total = x.shape[1]

        # Temporal indices for RoPE
        t_idx = torch.zeros(B, total, dtype=torch.long, device=device)
        # Text: set to -1000 (won't apply RoPE)
        t_idx[:, :N_text] = -1000
        # History: assign -T_h to -1
        if T_h > 0:
            for f in range(T_h):
                start = N_text + f * N_hist_per_frame
                t_idx[:, start:start + N_hist_per_frame] = -T_h + f
        # Current: 0 already

        # Apply causal mask if not provided (but we always provide from loss)
        if attention_mask is None:
            # Fallback: trivial mask (all visible)
            attention_mask = torch.zeros((B, 1, total, total), dtype=x.dtype, device=device)

        # Pass through transformer layers
        Freqs_cis = self.freqs_cis.to(device)
        for layer in self.layers:
            x = layer(x, c_emb, attention_mask, t_idx, Freqs_cis)

        # Extract current frame tokens (last N_curr)
        curr_out = x[:, -N_curr:]  # (B, N_curr, D)
        # Project to pixel space
        vec = self.final_layer(curr_out)  # (B, N_curr, patch_dim)
        # Unpatchify
        v_pred = unpatchify(
            vec,
            latent_shape=(noisy_latent.shape[1], 1, noisy_latent.shape[2], noisy_latent.shape[3]),
            patch_size=self.patch_size,
        ).squeeze(2)  # remove temporal dummy dim -> (B, C, H_s, W_s)

        return v_pred

    def _get_1d_text_pos_embed(self, length: int, device: torch.device) -> torch.Tensor:
        """Generates 1D sinusoidal position embedding for text tokens."""
        dim = self.hidden_size
        position = torch.arange(length, device=device).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * -(math.log(10000.0) / dim))
        pe = torch.zeros(length, dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def load_pretrained_weights(self, path: str) -> None:
        """
        Loads pretrained MM‑DiT weights, e.g., from SD3 Medium.
        This method is a placeholder; actual mapping depends on the original checkpoint.
        """
        try:
            state_dict = torch.load(path, map_location="cpu")
            # Perform key mapping if necessary (omitted for brevity)
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {path}. Missing keys: {len(missing)}, unexpected: {len(unexpected)}")
            for k in missing:
                print(f"  missing: {k}")
            for k in unexpected:
                print(f"  unexpected: {k}")
        except Exception as e:
            print(f"Warning: failed to load pretrained weights from {path}: {e}")


class TimestepEmbedding(nn.Module):
    """Time embedding using sinusoidal Fourier features and MLP."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) continuous time in [0,1].
        Returns:
            (B, dim)
        """
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        freqs = freqs.to(t.device)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding

