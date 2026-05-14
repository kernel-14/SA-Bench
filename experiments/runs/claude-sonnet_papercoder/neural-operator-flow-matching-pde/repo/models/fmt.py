```python
## models/fmt.py
"""Flow Marching Transformer (FMT) for generative PDE foundation modeling.

Implements the core generative model from:
"Bridging Neural Operator and Flow Matching for a Generative PDE Foundation Model"

Three main components:
  1. FlowMarchingKernel: Stateless math utility for the location-scale
     interpolation kernel (Section 3.1). Constructs noisy interpolations
     x_t^k and computes the preconditioned flow marching loss (Eq. 9).

  2. DiffusionForcingGRU: Maintains a compressed latent history state h_s
     across physical timesteps via GRU + cross-attention (Section 3.2).
     Implements the diffusion forcing scheme for stable long-horizon rollouts.

  3. FMT: Full Flow Marching Transformer combining SiT backbone (Section 4.1),
     temporal pyramid (Section 3.3), and diffusion forcing (Section 3.2).
     Supports three size variants: FMT-S (6M), FMT-B (42M), FMT-L (138M).

Key design choices from the paper:
  - k-free objective: network g_θ takes (x_t^k, t) only, not k
  - Preconditioned loss: ||(1-t)*g - (x1 - x_t^k)||^2 avoids stiffness near t→1
  - Temporal pyramid: 340 total tokens (4+16+64+256) for 15× efficiency gain
  - AdaLN-Zero conditioning from DiT, RMSNorm+SwiGLU from Llama-2
  - FlashAttention v2 for efficient multi-head self-attention
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.sit import (
    AdaLNZero,
    FinalLayer,
    PatchEmbed,
    RMSNorm,
    SiTBlock,
    SwiGLUFFN,
    TimestepEmbedder,
)
from models.temporal_pyramid import TemporalPyramid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FlowMarchingKernel
# ---------------------------------------------------------------------------


class FlowMarchingKernel:
    """Stateless utility for the flow marching location-scale interpolation kernel.

    Encodes the core mathematical operations from Section 3.1 of the paper:

    Location-scale interpolation (Eq. 1-3):
        x_t^k = μ_t + σ_t * z
        μ_t   = t * x_1 + k * (1 - t) * x_0
        σ_t   = (1 - t) * (1 - k)
        z     ~ N(0, I)

    Two boundary cases:
        k = 1: x_t^1 = t*x_1 + (1-t)*x_0  (deterministic neural operator)
        k = 0: x_t^0 ~ N(t*x_1, (1-t)^2*I) (stochastic flow matching)

    k-free velocity target (Eq. 5, simplified):
        u_t^k = (x_1 - x_t^k) / (1 - t)

    Preconditioned flow marching loss (Eq. 9):
        L_FM = 0.5 * E[ ||(1-t)*g_θ(x_t^k, t) - (x_1 - x_t^k)||^2 ]

    This class has no learnable parameters and no nn.Module inheritance.
    It is instantiated once in FMT.__init__ and reused across training steps.
    """

    def sample_interpolation(
        self,
        x0: Tensor,
        x1: Tensor,
        t: Tensor,
        k: Tensor,
    ) -> Tensor:
        """Construct the noisy interpolated state x_t^k.

        Implements the location-scale interpolation kernel from Eq. 1-3:
            μ_t   = t * x_1 + k * (1 - t) * x_0
            σ_t   = (1 - t) * (1 - k)
            x_t^k = μ_t + σ_t * z,   z ~ N(0, I)

        Boundary cases:
            k = 1: σ_t = 0, so x_t^k = t*x_1 + (1-t)*x_0 (deterministic)
            k = 0: x_t^k = t*x_1 + (1-t)*z (stochastic flow matching)

        Args:
            x0: Previous frame latent of shape (B, C, H, W).
            x1: Next frame latent (target) of shape (B, C, H, W).
            t: Timestep tensor. Accepts (B,), (B,1), or (B,1,1,1).
                Values in [0, 1] sampled from Uniform(0,1) during training.
                Reshaped internally to (B, 1, 1, 1) for broadcasting.
            k: Bridge parameter tensor. Same shape flexibility as t.
                Values in [0, 1] sampled from Uniform(0,1) during training.
                k=1 → deterministic operator, k=0 → stochastic flow matching.

        Returns:
            Noisy interpolated state x_t^k of shape (B, C, H, W).
            Same dtype and device as x0.
        """
        # Reshape t and k to (B, 1, 1, 1) for broadcasting over (C, H, W).
        b = x0.shape[0]
        t_bc: Tensor = t.view(b, 1, 1, 1).to(x0.dtype)
        k_bc: Tensor = k.view(b, 1, 1, 1).to(x0.dtype)

        # Compute mean and standard deviation of the interpolation kernel.
        # μ_t = t * x_1 + k * (1 - t) * x_0
        mu_t: Tensor = t_bc * x1 + k_bc * (1.0 - t_bc) * x0

        # σ_t = (1 - t) * (1 - k)
        sigma_t: Tensor = (1.0 - t_bc) * (1.0 - k_bc)

        # Sample noise z ~ N(0, I) with same shape as x0.
        z: Tensor = torch.randn_like(x0)

        # x_t^k = μ_t + σ_t * z
        x_tk: Tensor = mu_t + sigma_t * z

        return x_tk

    def compute_velocity_target(
        self,
        x_tk: Tensor,
        x1: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Compute the k-free velocity target (x_1 - x_t^k).

        From the k-free objective (Section 3.1, after Eq. 8):
            u_t^k = (x_1 - x_t^k) / (1 - t)

        The (1-t) denominator is absorbed into the preconditioned loss
        (Eq. 9), so this method returns only the numerator (x_1 - x_t^k).
        The t parameter is accepted for interface consistency but unused.

        Args:
            x_tk: Noisy interpolated state of shape (B, C, H, W).
            x1: Target next frame of shape (B, C, H, W).
            t: Timestep tensor (unused in this method, kept for API consistency).

        Returns:
            Velocity target (x_1 - x_t^k) of shape (B, C, H, W).
        """
        return x1 - x_tk

    def compute_loss(
        self,
        g: Tensor,
        x_tk: Tensor,
        x1: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Compute the preconditioned flow marching loss (Eq. 9).

        Implements the numerically stable preconditioned objective:
            L_FM = 0.5 * E[ ||(1-t)*g_θ(x_t^k, t) - (x_1 - x_t^k)||^2 ]

        The (1-t) preconditioning prevents numerical stiffness near t→1
        where the raw velocity target (x_1 - x_t^k)/(1-t) diverges.
        Minimizers of L_FM correspond to minimizers of R_FM (the unweighted
        regression objective) up to the benign (1-t) scaling.

        Args:
            g: Predicted velocity from the network g_θ, shape (B, C, H, W).
                This is the raw network output before any scaling.
            x_tk: Noisy interpolated state of shape (B, C, H, W).
            x1: Target next frame of shape (B, C, H, W).
            t: Timestep tensor. Accepts (B,), (B,1), or (B,1,1,1).
                Reshaped internally to (B, 1, 1, 1) for broadcasting.

        Returns:
            Scalar loss tensor (mean over batch and spatial dimensions).
        """
        b = g.shape[0]
        t_bc: Tensor = t.view(b, 1, 1, 1).to(g.dtype)

        # Target: (x_1 - x_t^k), shape (B, C, H, W)
        target: Tensor = x1 - x_tk

        # Preconditioned residual: (1-t)*g - (x_1 - x_t^k)
        residual: Tensor = (1.0 - t_bc) * g - target

        # Mean squared error over all dimensions.
        loss: Tensor = 0.5 * (residual ** 2).mean()

        return loss


# ---------------------------------------------------------------------------
# DiffusionForcingGRU
# ---------------------------------------------------------------------------


class DiffusionForcingGRU(nn.Module):
    """GRU-based diffusion forcing module for history-conditioned PDE prediction.

    Implements the diffusion forcing scheme from Section 3.2 of the paper.
    Maintains a compressed latent state h_s that summarizes the PDE history
    up to physical timestep s, providing conditioning for the FMT transformer.

    The update rule (Section 3.2):
        h_s ~ p_φ(h_s | h_{s-1}, x_{s,t_s}^{k_s}, t_s)

    Implemented as:
        1. Cross-attention: compress x_{s,t_s}^{k_s} to a single token
           (the current noisy state is "compressed onto a single token by
           cross attention" — paper Section 4.1)
        2. GRUCell: update h_{s-1} with the compressed token

    The GRU hidden dimension equals the FMT embedding dimension (shared
    internal dimension — paper Section 4.1).

    Attributes:
        hidden_size: GRU hidden dimension = FMT embedding dimension.
        proj_in: Linear projection from latent channels (16) to hidden_size.
        query: Learned query parameter for cross-attention pooling.
        cross_attn: Multi-head cross-attention for spatial compression.
        gru: GRUCell for sequential hidden state updates.
    """

    def __init__(self, hidden_size: int) -> None:
        """Initialize DiffusionForcingGRU.

        Args:
            hidden_size: GRU hidden dimension. Must equal the FMT embedding
                dimension (embed_dim) so the hidden state h can be directly
                added to the timestep conditioning vector c in FMT.forward.
                From config: fmt.variants.fmt_b.embed_dim = 512 (for FMT-B).
        """
        super().__init__()

        self.hidden_size: int = hidden_size

        # Project latent channels (C=16) to hidden_size for cross-attention.
        # Applied to each spatial token of the flattened noisy latent.
        # From config: p2vae.latent_channels = 16.
        self.proj_in: nn.Linear = nn.Linear(16, hidden_size, bias=True)

        # Learned query for cross-attention pooling.
        # Shape (1, 1, hidden_size) — expanded to (B, 1, hidden_size) at runtime.
        # Initialized to zeros per design spec; small normal init would also work.
        self.query: nn.Parameter = nn.Parameter(
            torch.zeros(1, 1, hidden_size)
        )

        # Multi-head cross-attention: compresses spatial token sequence to
        # a single summary token via the learned query.
        # num_heads=8 as per design spec (unspecified in paper, reasonable default).
        # For hidden_size=256 (FMT-S): head_dim=32
        # For hidden_size=512 (FMT-B): head_dim=64 (matches paper's head_dim=64)
        # For hidden_size=768 (FMT-L): head_dim=96
        self.cross_attn: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True,
            bias=True,
        )

        # GRUCell for sequential hidden state updates.
        # input_size = hidden_size (compressed token dimension)
        # hidden_size = hidden_size (GRU state dimension)
        self.gru: nn.GRUCell = nn.GRUCell(
            input_size=hidden_size,
            hidden_size=hidden_size,
        )

    def compress_to_token(self, x_noisy: Tensor) -> Tensor:
        """Compress a noisy latent frame to a single context token via cross-attention.

        Implements the "compressed onto a single token by cross attention"
        operation described in paper Section 4.1. The learned query vector
        attends over all spatial positions of the projected latent to produce
        a single summary token.

        Args:
            x_noisy: Noisy latent frame of shape (B, C, H, W) where C=16
                (latent_channels from config) and H, W vary by pyramid level:
                - Frame 0 (oldest): (B, 16, 2, 2) → 4 spatial tokens
                - Frame 1: (B, 16, 4, 4) → 16 spatial tokens
                - Frame 2: (B, 16, 8, 8) → 64 spatial tokens
                - Frame 3 (newest): (B, 16, 16, 16) → 256 spatial tokens
                The cross-attention handles variable sequence lengths.

        Returns:
            Single context token of shape (B, 1, hidden_size) summarizing
            the entire spatial field via learned attention pooling.
        """
        b, c, h, w = x_noisy.shape

        # Step 1: Flatten spatial dimensions.
        # (B, C, H, W) → (B, H*W, C) via permute + reshape.
        x_flat: Tensor = x_noisy.permute(0, 2, 3, 1).reshape(b, h * w, c)

        # Step 2: Project channels from C=16 to hidden_size.
        # (B, H*W, 16) → (B, H*W, hidden_size)
        x_proj: Tensor = self.proj_in(x_flat)

        # Step 3: Expand learned query to batch size.
        # (1, 1, hidden_size) → (B, 1, hidden_size)
        q: Tensor = self.query.expand(b, -1, -1)

        # Step 4: Cross-attention pooling.
        # Query: (B, 1, hidden_size) — single learned query
        # Key/Value: (B, H*W, hidden_size) — projected spatial tokens
        # Output: (B, 1, hidden_size) — single summary token
        token: Tensor
        token, _ = self.cross_attn(
            query=q,
            key=x_proj,
            value=x_proj,
            need_weights=False,
        )

        return token  # (B, 1, hidden_size)

    def update(self, h: Tensor, x_noisy: Tensor) -> Tensor:
        """Update the GRU hidden state with the current noisy frame.

        Implements one step of the diffusion forcing update:
            token = CrossAttn(query, x_noisy)  # compress to single token
            h_new = GRU(token, h)              # update hidden state

        Args:
            h: Previous GRU hidden state of shape (B, hidden_size).
                Initialized to zeros at the start of each trajectory.
            x_noisy: Current noisy latent frame of shape (B, C, H, W).
                Typically the noisy interpolated state x_{s,t_s}^{k_s}
                for the current physical timestep s.

        Returns:
            Updated GRU hidden state h_new of shape (B, hidden_size).
        """
        # Compress noisy frame to single token: (B, 1, hidden_size)
        token: Tensor = self.compress_to_token(x_noisy)

        # Squeeze sequence dimension: (B, 1, hidden_size) → (B, hidden_size)
        token_squeezed: Tensor = token.squeeze(1)

        # GRUCell update: h_new = GRU(input=token, hidden=h)
        h_new: Tensor = self.gru(token_squeezed, h)

        return h_new  # (B, hidden_size)

    def init_hidden(self, batch_size: int, device: torch.device) -> Tensor:
        """Initialize the GRU hidden state to zeros.

        Called at the start of each training sequence and at the beginning
        of autoregressive rollout inference.

        Args:
            batch_size: Number of samples in the batch (B).
            device: Target device for the hidden state tensor.

        Returns:
            Zero-initialized hidden state of shape (B, hidden_size).
        """
        return torch.zeros(batch_size, self.hidden_size, device=device)


# ---------------------------------------------------------------------------
# FMT (Flow Marching Transformer)
# ---------------------------------------------------------------------------


class FMT(nn.Module):
    """Flow Marching Transformer: generative PDE foundation model.

    Unifies deterministic neural operators and stochastic flow matching
    through a single transport field trained with the preconditioned flow
    marching objective (Section 3.1). Operates entirely in the latent space
    of P2VAE (c16p16 = 16 channels, 16×16 spatial).

    Architecture (Section 4.1):
        - SiT backbone: Transformer blocks with RMSNorm + SwiGLU (Llama-2)
          and AdaLN-Zero conditioning (DiT)
        - FlashAttention v2 for efficient multi-head self-attention
        - Temporal pyramid: 340 tokens (4+16+64+256) for 15× efficiency gain
        - GRU-based diffusion forcing for stable long-horizon rollouts

    Three size variants (paper Section 4.1):
        FMT-S: embed_dim=256, depth=6,  num_heads=4  (~6M params)
        FMT-B: embed_dim=512, depth=12, num_heads=8  (~42M params)
        FMT-L: embed_dim=768, depth=24, num_heads=12 (~138M params)

    Training objective (Eq. 11, conditional flow marching):
        L_CFM = 0.5 * E[ Σ_{s=0}^{3} ||(1-t_s)*g_θ(x_{s,t_s}^{k_s}, t_s, h_{s-1})
                         - (x_{s+1} - x_{s,t_s}^{k_s})||^2 ]

    Inference (Section 3.4):
        - Euler ODE sampler with N=100 steps, dt=0.01
        - Deterministic: all k_i=1 (no noise in history)
        - Generative: k_0=k_1=k_2=1, k_3<1 (generate from noise)

    Attributes:
        embed_dim: Token embedding dimension (256/512/768).
        latent_channels: P2VAE latent channel count (16 from config).
        latent_size: P2VAE latent spatial size (16 from config).
        depth: Number of SiT Transformer blocks.
        num_heads: Number of attention heads per block.
        patch_embed: PatchEmbed for projecting latent tokens to embed_dim.
        pos_embed: Learnable positional embedding (1, 340, embed_dim).
        blocks: ModuleList of SiTBlock instances.
        t_embedder: TimestepEmbedder for scalar t → embed_dim conditioning.
        norm_out: RMSNorm applied before output projection.
        proj_out: Linear projection from embed_dim to latent_channels.
        temporal_pyramid: TemporalPyramid for multi-scale token construction.
        gru_forcing: DiffusionForcingGRU for history conditioning.
        flow_kernel: FlowMarchingKernel for loss computation.
    """

    def __init__(self, config: Dict) -> None:
        """Initialize FMT from a configuration dictionary.

        The config dict should contain keys from config.yaml under the
        'fmt' section with the variant-specific values merged in.

        Expected keys (with defaults):
            embed_dim (int): Token embedding dimension. Default: 512 (FMT-B).
            depth (int): Number of SiT blocks. Default: 12 (FMT-B).
            num_heads (int): Attention heads. Default: 8 (FMT-B).
            head_dim (int): Per-head dimension. Default: 64 (from config).
            mlp_ratio (float): FFN hidden dim ratio. Default: 4.0 (from config).
            latent_channels (int): P2VAE latent channels. Default: 16.
            latent_size (int): P2VAE latent spatial size. Default: 16.
            patch_size (int): Patch size for tokenization. Default: 1.

        Args:
            config: Configuration dictionary. All keys have defaults so
                partial configs are supported for testing.
        """
        super().__init__()

        # Extract configuration with defaults from config.yaml.
        self.embed_dim: int = int(config.get("embed_dim", 512))
        self.depth: int = int(config.get("depth", 12))
        self.num_heads: int = int(config.get("num_heads", 8))
        self.head_dim: int = int(config.get("head_dim", 64))
        self.mlp_ratio: float = float(config.get("mlp_ratio", 4.0))
        self.latent_channels: int = int(config.get("latent_channels", 16))
        self.latent_size: int = int(config.get("latent_size", 16))
        self.patch_size: int = int(config.get("patch_size", 1))

        # Validate head_dim consistency.
        if self.embed_dim % self.head_dim != 0:
            raise ValueError(
                f"embed_dim={self.embed_dim} must be divisible by "
                f"head_dim={self.head_dim}."
            )
        expected_heads: int = self.embed_dim // self.head_dim
        if self.num_heads != expected_heads:
            logger.warning(
                "num_heads=%d does not match embed_dim // head_dim = %d. "
                "Using num_heads=%d.",
                self.num_heads,
                expected_heads,
                expected_heads,
            )
            self.num_heads = expected_heads

        # ------------------------------------------------------------------
        # Temporal pyramid (Section 3.3)
        # Downsample factors: [8, 4, 2, 1] → token counts: [4, 16, 64, 256]
        # Total tokens: 340 (config: fmt.temporal_pyramid.total_tokens = 340)
        # ------------------------------------------------------------------
        self.temporal_pyramid: TemporalPyramid = TemporalPyramid(
            latent_size=self.latent_size,
            downsample_factors=[8, 4, 2, 1],
        )
        total_tokens: int = self.temporal_pyramid.total_tokens  # 340

        # ------------------------------------------------------------------
        # Patch embedding (Section 4.1, patch_size=1)
        # Projects each spatial token from latent_channels=16 to embed_dim.
        # Applied to each pyramid level independently in forward().
        # ------------------------------------------------------------------
        self.patch_embed: PatchEmbed = PatchEmbed(
            patch_size=self.patch_size,
            in_channels=self.latent_channels,
            embed_dim=self.embed_dim,
        )

        # ------------------------------------------------------------------
        # Learnable positional embedding for all 340 pyramid tokens.
        # Shape: (1, 340, embed_dim) — broadcast over batch dimension.
        # Initialized with small normal noise following ViT/DiT convention.
        # ------------------------------------------------------------------
        self.pos_embed: nn.Parameter = nn.Parameter(
            torch.zeros(1, total_tokens, self.embed_dim)
        )
        nn.init.normal_(self.pos_embed, std=0.02)

        # ------------------------------------------------------------------
        # SiT Transformer blocks (Section 4.1)
        # RMSNorm + SwiGLU (Llama-2) + AdaLN-Zero (DiT) + FlashAttention v2
        # ------------------------------------------------------------------
        self.blocks: nn.ModuleList = nn.ModuleList([
            SiTBlock(
                hidden_size=self.embed_dim,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_ratio=self.mlp_ratio,
            )
            for _ in range(self.depth)
        ])

        # ------------------------------------------------------------------
        # Timestep embedder (Section 4.1, AdaLN-Zero conditioning)
        # Embeds scalar t ∈ [0,1] → (B, embed_dim) conditioning vector.
        # ------------------------------------------------------------------
        self.t_embedder: TimestepEmbedder = TimestepEmbedder(
            hidden_size=self.embed_dim,
            freq_embed_size=256,
        )

        # ------------------------------------------------------------------
        # Output head
        # norm_out: RMSNorm before projection
        # proj_out: Linear(embed_dim → latent_channels) for velocity prediction
        # Applied only to the last 256 tokens (full-resolution frame 3).
        # Zero-initialized for training stability (DiT convention).
        # ------------------------------------------------------------------
        self.norm_out: RMSNorm = RMSNorm(self.embed_dim)
        self.proj_out: nn.Linear = nn.Linear(
            self.embed_dim,
            self.latent_channels,
            bias=True,
        )
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

        # ------------------------------------------------------------------
        # GRU-based diffusion forcing (Section 3.2, 4.1)
        # Hidden dimension = embed_dim (shared internal dimension).
        # ------------------------------------------------------------------
        self.gru_forcing: DiffusionForcingGRU = DiffusionForcingGRU(
            hidden_size=self.embed_dim
        )

        # ------------------------------------------------------------------
        # Flow marching kernel (Section 3.1)
        # Stateless utility for interpolation and loss computation.
        # ------------------------------------------------------------------
        self.flow_kernel: FlowMarchingKernel = FlowMarchingKernel()

        # Log model configuration.
        n_params: int = sum(p.numel() for p in self.parameters())
        logger.info(
            "FMT initialized: embed_dim=%d, depth=%d, num_heads=%d, "
            "total_tokens=%d, params=%.2fM",
            self.embed_dim,
            self.depth,
            self.num_heads,
            total_tokens,
            n_params / 1e6,
        )

    def forward(
        self,
        latents: List[Tensor],
        ts: List[Tensor],
        h_prev: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass: predict velocity for the last (newest) frame.

        Takes 4 noisy latent frames at full resolution (B, 16, 16, 16),
        builds the temporal pyramid internally, runs the SiT transformer
        conditioned on the timestep and GRU history, and returns the
        predicted velocity for frame 3 (the full-resolution target frame).

        The GRU hidden state is updated using the full-resolution frame 3
        latent (the frame being predicted), and the updated state h_new is
        returned for use in the next autoregressive step.

        Args:
            latents: List of exactly 4 tensors, each of shape
                (B, 16, 16, 16) = (B, latent_channels, latent_size, latent_size).
                These are the noisy interpolated states x_{s,t_s}^{k_s} for
                s=0,1,2,3, constructed by FlowMarchingKernel.sample_interpolation.
                latents[0] = oldest frame (will be downsampled most aggressively)
                latents[3] = newest frame (kept at full resolution)
            ts: List of exactly 4 timestep tensors, each of shape (B,).
                ts[3] is used as the primary conditioning timestep (the timestep
                for the frame whose velocity we're predicting).
            h_prev: Previous GRU hidden state of shape (B, embed_dim).
                Initialized to zeros at the start of each trajectory.
                Encodes the PDE history from all previous physical timesteps.

        Returns:
            Tuple of:
                velocity: Predicted velocity for frame 3, shape (B, 16, 16, 16).
                    This is the raw network output g_θ(x_t^k, t, h) before
                    any scaling. The flow marching loss applies (1-t) scaling.
                h_new: Updated GRU hidden state of shape (B, embed_dim).
                    Computed by updating h_prev with the frame 3 noisy latent.
                    Used as h_prev in the next autoregressive step.

        Raises:
            ValueError: If len(latents) != 4 or len(ts) != 4.
        """
        if len(latents) != 4:
            raise ValueError(
                f"FMT.forward expects exactly 4 latent frames, got {len(latents)}."
            )
        if len(ts) != 4:
            raise ValueError(
                f"FMT.forward expects exactly 4 timestep tensors, got {len(ts)}."
            )

        b: int = latents[0].shape[0]
        device: torch.device = latents[0].device

        # ------------------------------------------------------------------
        # Step 1: Build temporal pyramid from 4 full-resolution noisy latents.
        # pyramid[0]: (B, 16, 2,  2 ) — frame 0 downsampled by factor 8
        # pyramid[1]: (B, 16, 4,  4 ) — frame 1 downsampled by factor 4
        # pyramid[2]: (B, 16, 8,  8 ) — frame 2 downsampled by factor 2
        # pyramid[3]: (B, 16, 16, 16) — frame 3 at full resolution (factor 1)
        # ------------------------------------------------------------------
        pyramid: List[Tensor] = self.temporal_pyramid.build