import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from layers import (
    CausalTemporalAttention,
    PrefixEnhancedSpatialAttention,
    VisualTextCrossAttention,
    FeedForward,
    get_causal_mask,
)
from modules import (
    JointTimestepEmbedding,
    CyclicTPEs,
    SpatialPosEmbed,
)


class TransformerBlock(nn.Module):
    """Single Transformer block with:
    1. Causal temporal attention + AdaLN
    2. Prefix-enhanced spatial attention + AdaLN  
    3. Cross attention + AdaLN
    4. Feed-forward + AdaLN
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        spatial_head_dim: int,
        temporal_head_dim: int,
        cross_head_dim: int,
        cross_attn_dim: int = 4096,
        prefix_len_enhance: int = 3,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Attention layers
        self.temporal_attn = CausalTemporalAttention(hidden_size, num_heads, temporal_head_dim)
        self.spatial_attn = PrefixEnhancedSpatialAttention(hidden_size, num_heads, spatial_head_dim, prefix_len_enhance)
        self.cross_attn = VisualTextCrossAttention(hidden_size, num_heads, cross_head_dim, cross_attn_dim)
        self.ff = FeedForward(hidden_size)

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm4 = nn.LayerNorm(hidden_size, elementwise_affine=False)

        # AdaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        adaLN_emb: torch.Tensor,
        causal_mask: torch.Tensor,
        P: int,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        temporal_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        spatial_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            hidden_states: (B, L, HW, C)
            adaLN_emb: (B, 1, 1, C) or (B, L, 1, C) timestep embedding for modulation
            causal_mask: (L, total_L) attention mask for temporal attention
            P: number of clean prefix frames
            encoder_hidden_states: (B, T_text, cross_attn_dim)
            temporal_kv_cache: cached temporal K, V from clean prefix
            spatial_kv_cache: cached spatial K, V from clean prefix
        
        Returns:
            hidden_states: (B, L, HW, C)
            temporal_kv: new temporal K, V for cache writing
        """
        B, L, HW, C = hidden_states.shape

        # AdaLN parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(adaLN_emb).chunk(6, dim=-1)

        # 1. Causal temporal attention
        norm_x = self.norm1(hidden_states)
        norm_x = norm_x * (1 + scale_msa) + shift_msa
        attn_out, temporal_kv = self.temporal_attn(norm_x, causal_mask, temporal_kv_cache)
        hidden_states = hidden_states + gate_msa * attn_out

        # 2. Prefix-enhanced spatial attention
        # Flatten temporal into batch for spatial attention
        # hidden_states: (B, L, HW, C) -> (B*L, HW, C) but we keep B and L separate
        norm_x = self.norm2(hidden_states)
        # Spatial attention processes each frame with prefix enhancement
        attn_out = self.spatial_attn(norm_x, P, spatial_kv_cache)
        hidden_states = hidden_states + attn_out

        # 3. Cross attention (if text condition available)
        if encoder_hidden_states is not None:
            norm_x = self.norm3(hidden_states)
            # Flatten to (B, L*HW, C)
            norm_x_flat = norm_x.view(B, L * HW, C)
            cross_out = self.cross_attn(norm_x_flat, encoder_hidden_states)
            cross_out = cross_out.view(B, L, HW, C)
            hidden_states = hidden_states + cross_out

        # 4. Feed-forward
        norm_x = self.norm4(hidden_states)
        norm_x = norm_x * (1 + scale_mlp) + shift_mlp
        ff_out = self.ff(norm_x)
        hidden_states = hidden_states + gate_mlp * ff_out

        return hidden_states, temporal_kv


class Ca2VDM(nn.Module):
    """Ca2-VDM: Causal Cache Video Diffusion Model.
    
    Architecture based on spatial-temporal Transformer with:
    - Causal temporal attention
    - Prefix-enhanced spatial attention  
    - KV-cache sharing across denoising steps
    - Cyclic-TPEs for long video generation
    - Partial noising training with clean prefix
    """

    def __init__(
        self,
        hidden_size: int = 1152,
        num_heads: int = 16,
        num_layers: int = 28,
        spatial_head_dim: int = 72,
        temporal_head_dim: int = 72,
        cross_head_dim: int = 72,
        cross_attn_dim: int = 4096,
        prefix_len_enhance: int = 3,
        max_train_len: int = 65,
        patch_size: int = 2,
        latent_channels: int = 4,
        spatial_size: int = 32,
        learn_sigma: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.spatial_size = spatial_size
        self.max_train_len = max_train_len
        self.learn_sigma = learn_sigma
        self.prefix_len_enhance = prefix_len_enhance

        # Compute number of patches
        self.num_patches_h = spatial_size // patch_size
        self.num_patches_w = spatial_size // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Input projection: patches -> hidden_size
        self.patch_embed = nn.Linear(latent_channels * patch_size * patch_size, hidden_size)

        # Spatial positional embeddings
        self.spatial_pos_embed = SpatialPosEmbed(self.num_patches, hidden_size)

        # Cyclic TPEs
        self.cyclic_tpes = CyclicTPEs(max_train_len, hidden_size)

        # Joint timestep embedding (handles t=0 for clean prefix and t=t for noisy target)
        self.t_embedder = JointTimestepEmbedding(hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                spatial_head_dim=spatial_head_dim,
                temporal_head_dim=temporal_head_dim,
                cross_head_dim=cross_head_dim,
                cross_attn_dim=cross_attn_dim,
                prefix_len_enhance=prefix_len_enhance,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.adaLN_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )

        # Output projection
        out_channels = latent_channels * patch_size * patch_size
        if learn_sigma:
            out_channels = out_channels * 2
        self.output_proj = nn.Linear(hidden_size, out_channels)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _patchify(self, z: torch.Tensor) -> torch.Tensor:
        """Convert latent to patch tokens.
        Args:
            z: (B, L, C, H, W)
        Returns:
            tokens: (B, L, num_patches, hidden_size)
        """
        B, L, C, H, W = z.shape
        ph = pw = self.patch_size
        nh = H // ph
        nw = W // pw

        z = z.view(B, L, C, nh, ph, nw, pw)
        z = z.permute(0, 1, 3, 5, 4, 6, 2).contiguous()
        z = z.view(B, L, nh * nw, C * ph * pw)
        return self.patch_embed(z)

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert patch tokens back to latent.
        Args:
            tokens: (B, L, num_patches, hidden_size)
        Returns:
            z: (B, L, C, H, W)
        """
        B, L, NP, _ = tokens.shape
        ph = pw = self.patch_size
        nh = self.num_patches_h
        nw = self.num_patches_w
        C_out = self.latent_channels * 2 if self.learn_sigma else self.latent_channels

        tokens = self.output_proj(tokens)  # (B, L, NP, C_out * ph * pw)
        tokens = tokens.view(B, L, nh, nw, ph, pw, -1)
        tokens = tokens.permute(0, 1, 6, 2, 4, 3, 5).contiguous()
        tokens = tokens.view(B, L, -1, nh * ph, nw * pw)
        return tokens

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        P: int,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        tpe_offset: int = 0,
        temporal_kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        spatial_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            z: (B, L, C, H, W) latent input [clean_prefix; noisy_target]
            t: (B,) timestep
            P: number of clean prefix frames
            encoder_hidden_states: (B, T_text, cross_attn_dim)
            tpe_offset: cyclic TPE offset for training
            temporal_kv_caches: list of (K, V) per layer for KV-cache
            spatial_kv_cache: spatial KV-cache
            return_cache: if True, return KV caches for cache writing
        
        Returns:
            output dict with 'pred' (B, L, C, H, W) and optionally 'temporal_kv_caches', 'spatial_kv'
        """
        B, L, C, H, W = z.shape
        device = z.device

        # Patchify
        tokens = self._patchify(z)  # (B, L, NP, hidden_size)

        # Add spatial positional embeddings
        tokens = tokens + self.spatial_pos_embed().unsqueeze(0).unsqueeze(0)  # (1, 1, NP, C)

        # Add temporal positional embeddings
        tpes = self.cyclic_tpes(L, tpe_offset)  # (L, hidden_size)
        tokens = tokens + tpes.unsqueeze(0).unsqueeze(2)  # (B, L, NP, C)

        # Timestep embedding
        t_emb = self.t_embedder(t, L, P)  # (B, L, C)
        t_emb = t_emb.unsqueeze(2)  # (B, L, 1, C)

        # Build causal attention mask
        total_L = L
        if temporal_kv_caches is not None and len(temporal_kv_caches) > 0:
            cache_L = temporal_kv_caches[0][0].shape[1] if temporal_kv_caches[0] is not None else 0
            total_L = cache_L + L
        else:
            cache_L = 0

        causal_mask = get_causal_mask(total_L, device)  # (total_L, total_L)
        # For the current frames, we only need the last L rows
        if cache_L > 0:
            causal_mask = causal_mask[cache_L:, :]  # (L, total_L) for the denoising target

        # Pass through transformer blocks
        new_temporal_kv_caches = []
        for i, block in enumerate(self.blocks):
            temporal_kv = temporal_kv_caches[i] if temporal_kv_caches is not None else None
            tokens, new_kv = block(
                tokens,
                t_emb,
                causal_mask,
                P,
                encoder_hidden_states,
                temporal_kv_cache=temporal_kv,
                spatial_kv_cache=spatial_kv_cache,
            )
            if return_cache:
                new_temporal_kv_caches.append(new_kv)

        # Final norm and output
        shift, scale = self.adaLN_final(t_emb).chunk(2, dim=-1)
        tokens = self.final_norm(tokens)
        tokens = tokens * (1 + scale) + shift

        # Unpatchify to get prediction
        pred = self._unpatchify(tokens)  # (B, L, Cout, H, W)

        if self.learn_sigma:
            pred, logvar = pred.chunk(2, dim=2)
        else:
            logvar = None

        result = {"pred": pred}
        if return_cache:
            result["temporal_kv_caches"] = new_temporal_kv_caches
        if logvar is not None:
            result["logvar"] = logvar

        return result

    def get_loss(
        self,
        z_0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
        P: int,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        tpe_offset: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined simplified + VLB loss.
        
        Args:
            z_0: clean latent (B, L, C, H, W)
            noise: Gaussian noise (B, L, C, H, W)
            t: timestep (B,)
            P: prefix length
            encoder_hidden_states: text embeddings
            tpe_offset: cyclic TPE offset
        """
        B, L, C, H, W = z_0.shape
        device = z_0.device

        # Noise only the target frames
        alpha_bar_t = self._get_alpha_bar(t)  # (B, 1, 1, 1, 1)
        alpha_bar_t = alpha_bar_t.view(B, 1, 1, 1, 1)

        # Build input: clean prefix + noisy target
        z_t = torch.cat([
            z_0[:, :P],  # clean prefix (no noise)
            alpha_bar_t.sqrt() * z_0[:, P:] + (1 - alpha_bar_t).sqrt() * noise[:, P:],
        ], dim=1)

        # Forward
        output = self.forward(z_t, t, P, encoder_hidden_states, tpe_offset)
        pred = output["pred"]

        # Compute loss mask: 0 for clean prefix, 1 for target
        mask = torch.ones(B, L, 1, 1, 1, device=device)
        mask[:, :P] = 0

        # Simplified loss (MSE on noise prediction)
        target_noise = noise[:, P:]  # only noise on target frames
        pred_noise = pred[:, P:]

        simple_loss = F.mse_loss(pred_noise, target_noise, reduction="none")
        simple_loss = (simple_loss * mask[:, P:]).sum() / mask[:, P:].sum()

        loss = simple_loss

        # VLB loss if using learnable sigma
        if self.learn_sigma and "logvar" in output:
            vlb_loss = self._compute_vlb_loss(
                pred, output["logvar"], z_0, z_t, t, P, alpha_bar_t, mask
            )
            loss = loss + vlb_loss

        return {"loss": loss, "simple_loss": simple_loss}

    def _get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Get alpha_bar (cumulative product of 1-beta) for given timesteps."""
        # Using the DDPM schedule from the paper: beta_1=1e-4, beta_T=0.02
        betas = self._get_betas(t.device)
        alpha_bars = torch.cumprod(1 - betas, dim=0)
        return alpha_bars[t]

    @staticmethod
    def _get_betas(device: torch.device, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
        """Linear schedule for betas."""
        return torch.linspace(beta_start, beta_end, T, device=device)

    def _compute_vlb_loss(
        self,
        pred: torch.Tensor,
        logvar: torch.Tensor,
        z_0: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
        P: int,
        alpha_bar_t: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute VLB loss with learnable covariance.
        
        Following Nichol & Dhariwal 2021 and Peebles & Xie 2023.
        
        KL(N(μ_q, σ_q²) || N(μ_p, σ_p²)) = 0.5 * [log(σ_p²) - log(σ_q²) 
            + (σ_q² + (μ_q - μ_p)²) / σ_p² - 1]
        """
        B, L, C, H, W = z_0.shape
        device = z_0.device

        betas = Ca2VDM._get_betas(device)
        alpha_bars = torch.cumprod(1 - betas, dim=0)
        betas_t = betas[t].view(B, 1, 1, 1, 1)
        alpha_bars_t = alpha_bar_t.view(B, 1, 1, 1, 1)
        alpha_bars_t_1 = torch.cat([torch.ones(1, device=device), alpha_bars[:-1]])[t].view(B, 1, 1, 1, 1)

        # True posterior variance: σ_q² = β_t * (1 - \bar{α}_{t-1}) / (1 - \bar{α}_t)
        true_var = betas_t * (1 - alpha_bars_t_1) / (1 - alpha_bars_t).clamp(min=1e-8)
        true_logvar = torch.log(true_var.clamp(min=1e-20))

        # Model predicted x_0 and mean μ_p
        pred_x0 = (z_t[:, P:] - (1 - alpha_bars_t).sqrt() * pred[:, P:]) / alpha_bars_t.sqrt().clamp(min=1e-8)
        pred_mean = (
            betas_t * alpha_bars_t_1.sqrt() / (1 - alpha_bars_t) * pred_x0
            + (1 - alpha_bars_t_1) * betas_t.sqrt() / (1 - alpha_bars_t) * z_t[:, P:]
        )
        # True mean μ_q (using true x_0)
        true_mean = (
            betas_t * alpha_bars_t_1.sqrt() / (1 - alpha_bars_t) * z_0[:, P:]
            + (1 - alpha_bars_t_1) * betas_t.sqrt() / (1 - alpha_bars_t) * z_t[:, P:]
        )

        # Model variance: σ_p² = exp(logvar)
        model_logvar = logvar[:, P:]
        model_logvar = torch.clamp(model_logvar, max=20.0)

        # KL divergence per element
        kl = 0.5 * (
            model_logvar - true_logvar
            + (true_var + (pred_mean - true_mean).pow(2)) / torch.exp(model_logvar)
            - 1.0
        )
        # Average over C, H, W dimensions
        kl = kl.mean(dim=(2, 3, 4))  # (B, L-P)
        # Apply mask
        vlb = (kl * mask[:, P:, 0, 0, 0]).sum() / mask[:, P:, 0, 0, 0].sum().clamp(min=1)

        return vlb


class Ca2VDM_Bidirectional(nn.Module):
    """Bidirectional baseline model (OS-Fix and OS-Ext).
    
    Uses full temporal attention instead of causal attention.
    No KV-cache capability.
    """

    def __init__(
        self,
        hidden_size: int = 1152,
        num_heads: int = 16,
        num_layers: int = 28,
        head_dim: int = 72,
        cross_attn_dim: int = 4096,
        max_train_len: int = 65,
        patch_size: int = 2,
        latent_channels: int = 4,
        spatial_size: int = 32,
        learn_sigma: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.spatial_size = spatial_size
        self.max_train_len = max_train_len
        self.learn_sigma = learn_sigma

        self.num_patches_h = spatial_size // patch_size
        self.num_patches_w = spatial_size // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        self.patch_embed = nn.Linear(latent_channels * patch_size * patch_size, hidden_size)
        self.spatial_pos_embed = SpatialPosEmbed(self.num_patches, hidden_size)

        # Standard TPEs (not cyclic for fixed-length baseline)
        from modules import get_sinusoidal_positional_encoding
        self.tpe = nn.Parameter(
            get_sinusoidal_positional_encoding(max_train_len, hidden_size),
            requires_grad=False
        )

        self.t_embedder = JointTimestepEmbedding(hidden_size)

        # Full bidirectional attention blocks
        self.blocks = nn.ModuleList([
            BidirectionalBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                cross_attn_dim=cross_attn_dim,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.adaLN_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        out_channels = latent_channels * patch_size * patch_size
        if learn_sigma:
            out_channels = out_channels * 2
        self.output_proj = nn.Linear(hidden_size, out_channels)

    def _patchify(self, z):
        B, L, C, H, W = z.shape
        ph = pw = self.patch_size
        nh = H // ph
        nw = W // pw
        z = z.view(B, L, C, nh, ph, nw, pw)
        z = z.permute(0, 1, 3, 5, 4, 6, 2).contiguous()
        z = z.view(B, L, nh * nw, C * ph * pw)
        return self.patch_embed(z)

    def _unpatchify(self, tokens):
        B, L, NP, _ = tokens.shape
        ph = pw = self.patch_size
        nh = self.num_patches_h
        nw = self.num_patches_w
        C_out = self.latent_channels * 2 if self.learn_sigma else self.latent_channels
        tokens = self.output_proj(tokens)
        tokens = tokens.view(B, L, nh, nw, ph, pw, -1)
        tokens = tokens.permute(0, 1, 6, 2, 4, 3, 5).contiguous()
        tokens = tokens.view(B, L, -1, nh * ph, nw * pw)
        return tokens

    def forward(self, z, t, P, encoder_hidden_states=None, tpe_offset=0):
        B, L, C, H, W = z.shape
        tokens = self._patchify(z)
        tokens = tokens + self.spatial_pos_embed().unsqueeze(0).unsqueeze(0)
        
        tpe_indices = (torch.arange(L, device=z.device) + tpe_offset) % self.max_train_len
        tpes = self.tpe[tpe_indices]
        tokens = tokens + tpes.unsqueeze(0).unsqueeze(2)

        t_emb = self.t_embedder(t, L, P).unsqueeze(2)

        for block in self.blocks:
            tokens = block(tokens, t_emb, encoder_hidden_states)

        shift, scale = self.adaLN_final(t_emb).chunk(2, dim=-1)
        tokens = self.final_norm(tokens)
        tokens = tokens * (1 + scale) + shift
        pred = self._unpatchify(tokens)

        if self.learn_sigma:
            pred, logvar = pred.chunk(2, dim=2)
        else:
            logvar = None

        result = {"pred": pred}
        if logvar is not None:
            result["logvar"] = logvar
        return result


class BidirectionalBlock(nn.Module):
    """Bidirectional Transformer block (baseline) with full temporal attention."""

    def __init__(self, hidden_size, num_heads, head_dim, cross_attn_dim=4096):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim

        # Full temporal attention (bidirectional)
        self.temporal_attn_q = nn.Linear(hidden_size, inner_dim)
        self.temporal_attn_k = nn.Linear(hidden_size, inner_dim)
        self.temporal_attn_v = nn.Linear(hidden_size, inner_dim)
        self.temporal_attn_o = nn.Linear(inner_dim, hidden_size)

        # Spatial attention
        self.spatial_attn_q = nn.Linear(hidden_size, inner_dim)
        self.spatial_attn_k = nn.Linear(hidden_size, inner_dim)
        self.spatial_attn_v = nn.Linear(hidden_size, inner_dim)
        self.spatial_attn_o = nn.Linear(inner_dim, hidden_size)

        # Cross attention
        self.cross_attn_q = nn.Linear(hidden_size, inner_dim)
        self.cross_attn_k = nn.Linear(cross_attn_dim, inner_dim)
        self.cross_attn_v = nn.Linear(cross_attn_dim, inner_dim)
        self.cross_attn_o = nn.Linear(inner_dim, hidden_size)

        # FF
        self.ff = FeedForward(hidden_size)

        # Norms
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm4 = nn.LayerNorm(hidden_size, elementwise_affine=False)

        # AdaLN
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def _full_temporal_attn(self, x):
        B, L, HW, C = x.shape
        inner = self.num_heads * self.head_dim
        # Permute: (B, HW, L, C)
        x = x.permute(0, 2, 1, 3)
        Q = self.temporal_attn_q(x).view(B, HW, L, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4).reshape(B * HW, self.num_heads, L, self.head_dim)
        K = self.temporal_attn_k(x).view(B, HW, L, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4).reshape(B * HW, self.num_heads, L, self.head_dim)
        V = self.temporal_attn_v(x).view(B, HW, L, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4).reshape(B * HW, self.num_heads, L, self.head_dim)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        out = out.reshape(B, HW, self.num_heads, L, self.head_dim).permute(0, 3, 1, 2, 4).reshape(B, L, HW, inner)
        return self.temporal_attn_o(out)

    def _spatial_attn(self, x):
        B, L, HW, C = x.shape
        inner = self.num_heads * self.head_dim
        x = x.view(B * L, HW, C)
        Q = self.spatial_attn_q(x).view(B * L, HW, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.spatial_attn_k(x).view(B * L, HW, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.spatial_attn_v(x).view(B * L, HW, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        out = out.permute(0, 2, 1, 3).reshape(B * L, HW, inner)
        return self.spatial_attn_o(out).view(B, L, HW, C)

    def forward(self, hidden_states, adaLN_emb, encoder_hidden_states=None):
        B, L, HW, C = hidden_states.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(adaLN_emb).chunk(6, dim=-1)

        # Temporal attn
        norm_x = self.norm1(hidden_states)
        norm_x = norm_x * (1 + scale_msa) + shift_msa
        hidden_states = hidden_states + gate_msa * self._full_temporal_attn(norm_x)

        # Spatial attn
        norm_x = self.norm2(hidden_states)
        hidden_states = hidden_states + self._spatial_attn(norm_x)

        # Cross attn
        if encoder_hidden_states is not None:
            norm_x = self.norm3(hidden_states)
            norm_flat = norm_x.view(B, L * HW, C)
            Q = self.cross_attn_q(norm_flat).view(B, L * HW, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            K = self.cross_attn_k(encoder_hidden_states).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            V = self.cross_attn_v(encoder_hidden_states).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            cross_out = torch.matmul(attn, V)
            cross_out = cross_out.permute(0, 2, 1, 3).reshape(B, L * HW, C)
            cross_out = self.cross_attn_o(cross_out).view(B, L, HW, C)
            hidden_states = hidden_states + cross_out

        # FF
        norm_x = self.norm4(hidden_states)
        norm_x = norm_x * (1 + scale_mlp) + shift_mlp
        hidden_states = hidden_states + gate_mlp * self.ff(norm_x)

        return hidden_states
