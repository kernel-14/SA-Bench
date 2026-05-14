```python
## models/mmdit.py
"""Multi-Modal Diffusion Transformer (MM-DiT) for Pyramidal Flow Matching.

Implements the MM-DiT backbone based on SD3 Medium architecture (Esser et al.,
2024) as described in the paper (Section 4.1, Appendix B):
- 24 transformer layers, 2B parameters total
- Dual-stream processing: separate visual and text token streams
- Blockwise causal attention for autoregressive video generation
- AdaLayerNorm conditioning via timestep + stage + CLIP embeddings
- T5-XXL sequence embeddings as text conditioning stream
- Spatial pyramid extrapolation and temporal pyramid interpolation
- Patchify/unpatchify for 2×2 spatial patches

Tensor convention:
- Videos: [B, C, T, H, W] in pixel space
- Latents: [B, latent_channels, T//8, H//8, W//8] in latent space
- Tokens: [B, seq_len, hidden_dim] in transformer space

Usage:
    from models.mmdit import MMDiT

    model = MMDiT(config)
    velocity = model.forward(
        latent=noisy_latent,
        timesteps=t,
        text_cond=text_embeddings,
        frame_token_counts=[256, 256, 256],
        stage_id=0,
        history_latent=compressed_history,
    )
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor

from models.attention import BlockwiseCausalAttention
from models.positional_encoding import PositionalEncoding
from utils.logging import get_logger

## ---------------------------------------------------------------------------
## Module-level logger
## ---------------------------------------------------------------------------
logger = get_logger(__name__)

## ---------------------------------------------------------------------------
## Optional einops import
## ---------------------------------------------------------------------------
try:
    from einops import rearrange  # type: ignore[import]
    _EINOPS_AVAILABLE: bool = True
except ImportError:
    _EINOPS_AVAILABLE = False
    logger.warning(
        "einops not available. Using manual reshape for patchify/unpatchify. "
        "Install with: pip install einops"
    )


## ---------------------------------------------------------------------------
## Helper: sinusoidal timestep embedding
## ---------------------------------------------------------------------------

def _sinusoidal_timestep_embedding(
    timesteps: Tensor,
    embed_dim: int,
    max_period: float = 10000.0,
) -> Tensor:
    """Computes sinusoidal timestep embeddings.

    Standard implementation from Ho et al. (2020) / Vaswani et al. (2017).
    Scales timesteps by 1000 to map [0, 1] → [0, 1000] before embedding.

    Args:
        timesteps: 1D tensor of timestep values [B], in range [0, 1].
        embed_dim: Embedding dimension. Must be even.
        max_period: Controls the minimum frequency. Defaults to 10000.

    Returns:
        Tensor of shape [B, embed_dim] containing sinusoidal embeddings.

    Raises:
        ValueError: If embed_dim is odd.
    """
    if embed_dim % 2 != 0:
        raise ValueError(
            f"embed_dim must be even for sinusoidal embedding, got {embed_dim}."
        )

    half_dim: int = embed_dim // 2
    device: torch.device = timesteps.device
    dtype: torch.dtype = timesteps.dtype

    # Scale timesteps from [0, 1] to [0, 1000]
    scaled_t: Tensor = timesteps.float() * 1000.0

    # Inverse frequencies: 1 / max_period^(2i / embed_dim)
    freqs: Tensor = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half_dim, dtype=torch.float32, device=device)
        / half_dim
    )  # [half_dim]

    # Outer product: [B, half_dim]
    args: Tensor = scaled_t[:, None].float() * freqs[None]

    # Concatenate cos and sin: [B, embed_dim]
    embedding: Tensor = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    return embedding.to(dtype=dtype)


## ---------------------------------------------------------------------------
## AdaLayerNorm
## ---------------------------------------------------------------------------

class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization for timestep/stage conditioning.

    Injects the conditioning vector c into each transformer block via
    learned scale and shift parameters. Implements the formula:
        output = (1 + scale) * LayerNorm(x) + shift
    where scale and shift are derived from c via a linear projection.

    This is the standard DiT/SD3 conditioning mechanism (Peebles & Xie, 2023).

    Attributes:
        hidden_dim: Feature dimension of the input tensor.
        norm: Standard LayerNorm applied before scaling/shifting.
        proj: Linear projection from conditioning dim to 2*hidden_dim.
            Produces [scale, shift] concatenated along the last dimension.
    """

    def __init__(self, hidden_dim: int) -> None:
        """Initializes AdaLayerNorm.

        Args:
            hidden_dim: Feature dimension of the input tensor x.
                Also the dimension of the conditioning vector c.
        """
        super().__init__()

        self.hidden_dim: int = hidden_dim

        # Standard LayerNorm (no affine parameters — scale/shift come from c)
        self.norm: nn.LayerNorm = nn.LayerNorm(
            hidden_dim, elementwise_affine=False, eps=1e-6
        )

        # Project conditioning vector to scale + shift
        # SiLU activation before projection (standard in DiT/SD3)
        self.proj: nn.Sequential = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )

        # Zero-initialize the projection for stable training start
        # This makes AdaLayerNorm behave as standard LayerNorm at init
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """Applies adaptive layer normalization.

        Args:
            x: Input tensor of shape [B, seq_len, hidden_dim].
            c: Conditioning vector of shape [B, hidden_dim].
                Contains combined timestep + stage + CLIP embeddings.

        Returns:
            Normalized and conditioned tensor of shape [B, seq_len, hidden_dim].
        """
        # Project conditioning to scale and shift: [B, 2*hidden_dim]
        scale_shift: Tensor = self.proj(c)  # [B, 2*hidden_dim]

        # Split into scale and shift: each [B, hidden_dim]
        scale: Tensor
        shift: Tensor
        scale, shift = scale_shift.chunk(2, dim=-1)

        # Expand for broadcasting over seq_len dimension
        # [B, hidden_dim] → [B, 1, hidden_dim]
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)

        # Apply: (1 + scale) * LayerNorm(x) + shift
        x_normed: Tensor = self.norm(x)
        return (1.0 + scale) * x_normed + shift


## ---------------------------------------------------------------------------
## MLP (Feed-Forward Network)
## ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Feed-forward network used in each transformer block.

    Standard two-layer MLP with GELU activation:
        x → Linear(D, mlp_dim) → GELU → Linear(mlp_dim, D)

    Attributes:
        fc1: First linear layer (D → mlp_dim).
        fc2: Second linear layer (mlp_dim → D).
    """

    def __init__(self, hidden_dim: int, mlp_dim: int) -> None:
        """Initializes MLP.

        Args:
            hidden_dim: Input and output feature dimension.
            mlp_dim: Hidden dimension of the MLP (= hidden_dim * mlp_ratio).
        """
        super().__init__()

        self.fc1: nn.Linear = nn.Linear(hidden_dim, mlp_dim, bias=True)
        self.fc2: nn.Linear = nn.Linear(mlp_dim, hidden_dim, bias=True)

        # Initialize with Xavier uniform for stable training
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Applies the two-layer MLP.

        Args:
            x: Input tensor of shape [B, seq_len, hidden_dim].

        Returns:
            Output tensor of shape [B, seq_len, hidden_dim].
        """
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.fc2(x)
        return x


## ---------------------------------------------------------------------------
## MMDiTBlock
## ---------------------------------------------------------------------------

class MMDiTBlock(nn.Module):
    """Single MM-DiT transformer block with dual-stream processing.

    Implements the SD3 Medium MM-DiT block structure:
    1. Visual stream: AdaLayerNorm → joint attention → residual
    2. Text stream: AdaLayerNorm → joint attention → residual
    3. Visual MLP: AdaLayerNorm → MLP → residual
    4. Text MLP: LayerNorm → MLP → residual

    Joint attention concatenates visual and text tokens, runs
    BlockwiseCausalAttention with the causal mask, then splits outputs.

    Attributes:
        hidden_dim: Feature dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        mlp_dim: MLP hidden dimension.
        norm1_vis: AdaLayerNorm for visual stream pre-attention.
        norm1_txt: AdaLayerNorm for text stream pre-attention.
        attn: BlockwiseCausalAttention for joint visual+text attention.
        norm2_vis: AdaLayerNorm for visual stream pre-MLP.
        norm2_txt: Standard LayerNorm for text stream pre-MLP.
        mlp_vis: MLP for visual stream.
        mlp_txt: MLP for text stream.
    """

    def __init__(
        self,
        hidden_dim: int = 1152,
        num_heads: int = 16,
        head_dim: int = 72,
        mlp_ratio: float = 4.0,
        context_dim: int = 4096,
        dropout: float = 0.0,
    ) -> None:
        """Initializes MMDiTBlock.

        Args:
            hidden_dim: Feature dimension for both visual and text streams.
                Defaults to 1152 (SD3 Medium standard).
            num_heads: Number of attention heads. Defaults to 16.
            head_dim: Dimension per attention head. Defaults to 72.
            mlp_ratio: MLP hidden dim multiplier. Defaults to 4.0.
            context_dim: Text context dimension for cross-attention.
                Defaults to 4096 (T5-XXL hidden dim).
            dropout: Attention dropout probability. Defaults to 0.0.
        """
        super().__init__()

        self.hidden_dim: int = hidden_dim
        self.num_heads: int = num_heads
        self.head_dim: int = head_dim
        self.mlp_dim: int = int(hidden_dim * mlp_ratio)

        # ----------------------------------------------------------------
        # Pre-attention norms
        # ----------------------------------------------------------------
        # Visual stream: AdaLayerNorm (conditioned on timestep/stage)
        self.norm1_vis: AdaLayerNorm = AdaLayerNorm(hidden_dim)
        # Text stream: AdaLayerNorm (also conditioned)
        self.norm1_txt: AdaLayerNorm = AdaLayerNorm(hidden_dim)

        # ----------------------------------------------------------------
        # Joint attention (visual + text tokens concatenated)
        # BlockwiseCausalAttention handles the causal mask for visual tokens
        # and full bidirectional attention for text tokens.
        # context_dim is set to hidden_dim because in joint attention,
        # both visual and text tokens are projected to hidden_dim before
        # being concatenated — so the "context" is also hidden_dim.
        # ----------------------------------------------------------------
        self.attn: BlockwiseCausalAttention = BlockwiseCausalAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            context_dim=hidden_dim,  # Joint attention: context is also hidden_dim
            dropout=dropout,
        )

        # ----------------------------------------------------------------
        # Pre-MLP norms
        # ----------------------------------------------------------------
        # Visual stream: AdaLayerNorm (conditioned)
        self.norm2_vis: AdaLayerNorm = AdaLayerNorm(hidden_dim)
        # Text stream: standard LayerNorm (SD3 convention for text MLP)
        self.norm2_txt: nn.LayerNorm = nn.LayerNorm(
            hidden_dim, elementwise_affine=True, eps=1e-6
        )

        # ----------------------------------------------------------------
        # MLPs
        # ----------------------------------------------------------------
        self.mlp_vis: MLP = MLP(hidden_dim, self.mlp_dim)
        self.mlp_txt: MLP = MLP(hidden_dim, self.mlp_dim)

    def _build_joint_mask(
        self,
        vis_seq_len: int,
        txt_seq_len: int,
        vis_causal_mask: Optional[Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tensor]:
        """Builds the joint attention mask for visual + text tokens.

        The joint mask has shape [total_seq, total_seq] where
        total_seq = vis_seq_len + txt_seq_len.

        Masking rules:
        - Visual → Visual: blockwise causal (from vis_causal_mask)
        - Visual → Text: always True (visual tokens attend to all text)
        - Text → Visual: always True (text tokens attend to all visual)
        - Text → Text: always True (text tokens fully bidirectional)

        Args:
            vis_seq_len: Number of visual tokens.
            txt_seq_len: Number of text tokens.
            vis_causal_mask: Boolean causal mask [vis_seq, vis_seq] from
                build_causal_mask(). True = can attend. If None, visual
                tokens use fully bidirectional attention.
            device: Target device.
            dtype: Target dtype for additive mask conversion.

        Returns:
            Additive float mask [1, 1, total_seq, total_seq] where
            0.0 = can attend, -inf = blocked. Returns None if no masking
            is needed (all True).
        """
        total_seq: int = vis_seq_len + txt_seq_len

        if vis_causal_mask is None and txt_seq_len == 0:
            return None

        # Start with all-True (all can attend)
        joint_bool_mask: Tensor = torch.ones(
            total_seq, total_seq, dtype=torch.bool, device=device
        )

        # Apply visual causal mask to the visual-to-visual quadrant
        if vis_causal_mask is not None:
            # vis_causal_mask: [vis_seq, vis_seq]
            # Place in top-left quadrant of joint mask
            vis_mask_device: Tensor = vis_causal_mask.to(device=device)
            joint_bool_mask[:vis_seq_len, :vis_seq_len] = vis_mask_device

        # Text → Visual, Visual → Text, Text → Text: all True (already set)
        # No changes needed for these quadrants.

        # Convert bool mask to additive float mask
        # 0.0 = can attend, -inf = blocked
        additive_mask: Tensor = torch.zeros(
            total_seq, total_seq, dtype=dtype, device=device
        )
        additive_mask = additive_mask.masked_fill(~joint_bool_mask, float("-inf"))

        # Expand for broadcasting: [1, 1, total_seq, total_seq]
        return additive_mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        x_vis: Tensor,
        x_txt: Tensor,
        c: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Applies one MM-DiT transformer block.

        Args:
            x_vis: Visual token sequence [B, vis_seq, hidden_dim].
            x_txt: Text token sequence [B, txt_seq, hidden_dim].
            c: Conditioning vector [B, hidden_dim] containing combined
                timestep + stage + CLIP embeddings.
            attn_mask: Optional boolean causal mask [vis_seq, vis_seq]
                from BlockwiseCausalAttention.build_causal_mask().
                True = can attend. If None, fully bidirectional attention.

        Returns:
            Tuple of:
                - x_vis_out: Updated visual tokens [B, vis_seq, hidden_dim].
                - x_txt_out: Updated text tokens [B, txt_seq, hidden_dim].
        """
        batch_size: int = x_vis.shape[0]
        vis_seq_len: int = x_vis.shape[1]
        txt_seq_len: int = x_txt.shape[1]
        device: torch.device = x_vis.device
        dtype: torch.dtype = x_vis.dtype

        # ----------------------------------------------------------------
        # Step 1: Pre-attention normalization
        # ----------------------------------------------------------------
        x_vis_norm: Tensor = self.norm1_vis(x_vis, c)  # [B, vis_seq, D]
        x_txt_norm: Tensor = self.norm1_txt(x_txt, c)  # [B, txt_seq, D]

        # ----------------------------------------------------------------
        # Step 2: Joint attention
        # Concatenate visual and text tokens along sequence dimension
        # ----------------------------------------------------------------
        # Joint sequence: [visual tokens | text tokens]
        x_joint: Tensor = torch.cat(
            [x_vis_norm, x_txt_norm], dim=1
        )  # [B, vis_seq + txt_seq, D]

        # Build joint attention mask
        joint_mask: Optional[Tensor] = self._build_joint_mask(
            vis_seq_len=vis_seq_len,
            txt_seq_len=txt_seq_len,
            vis_causal_mask=attn_mask,
            device=device,
            dtype=dtype,
        )

        # Run joint self-attention
        # BlockwiseCausalAttention.forward() expects:
        # x: [B, seq_len, embed_dim], context=None for self-attention
        # attn_mask: boolean [seq_len, seq_len] — but we pass additive mask directly
        # We bypass the bool→additive conversion in BlockwiseCausalAttention
        # by using the internal SDPA path with our pre-built additive mask.
        # To do this cleanly, we call the underlying attention directly.
        attn_out_joint: Tensor = self._joint_attention(
            x_joint, joint_mask
        )  # [B, vis_seq + txt_seq, D]

        # Split back into visual and text outputs
        attn_out_vis: Tensor = attn_out_joint[:, :vis_seq_len, :]
        attn_out_txt: Tensor = attn_out_joint[:, vis_seq_len:, :]

        # ----------------------------------------------------------------
        # Step 3: Residual connections after attention
        # ----------------------------------------------------------------
        x_vis = x_vis + attn_out_vis
        x_txt = x_txt + attn_out_txt

        # ----------------------------------------------------------------
        # Step 4: MLP with pre-norm
        # ----------------------------------------------------------------
        # Visual stream: AdaLayerNorm → MLP → residual
        x_vis_mlp_in: Tensor = self.norm2_vis(x_vis, c)
        x_vis = x_vis + self.mlp_vis(x_vis_mlp_in)

        # Text stream: standard LayerNorm → MLP → residual
        x_txt_mlp_in: Tensor = self.norm2_txt(x_txt)
        x_txt = x_txt + self.mlp_txt(x_txt_mlp_in)

        return x_vis, x_txt

    def _joint_attention(
        self,
        x_joint: Tensor,
        additive_mask: Optional[Tensor],
    ) -> Tensor:
        """Runs joint self-attention on the concatenated visual+text sequence.

        Uses the Q/K/V projections from self.attn but bypasses the
        bool→additive mask conversion since we pre-build the additive mask.

        Args:
            x_joint: Concatenated [visual | text] tokens [B, total_seq, D].
            additive_mask: Pre-built additive float mask
                [1, 1, total_seq, total_seq] or None.

        Returns:
            Attended output [B, total_seq, D].
        """
        batch_size: int = x_joint.shape[0]
        total_seq: int = x_joint.shape[1]
        device: torch.device = x_joint.device

        # Q, K, V projections (self-attention: all from x_joint)
        q: Tensor = self.attn.q_proj(x_joint)   # [B, total_seq, D]
        k: Tensor = self.attn.k_proj(x_joint)   # [B, total_seq, D]
        v: Tensor = self.attn.v_proj(x_joint)   # [B, total_seq, D]

        # Reshape to multi-head format: [B, num_heads, total_seq, head_dim]
        q = q.reshape(batch_size, total_seq, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)

        k = k.reshape(batch_size, total_seq, self.num_heads, self.head_dim)
        k = k.transpose(1, 2)

        v = v.reshape(batch_size, total_seq, self.num_heads, self.head_dim)
        v = v.transpose(1, 2)

        # Scaled dot-product attention with additive mask
        dropout_p: float = self.attn.dropout if self.training else 0.0
        scale: float = 1.0 / math.sqrt(self.head_dim)

        out: Tensor = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=additive_mask,
            dropout_p=dropout_p,
            is_causal=False,
            scale=scale,
        )  # [B, num_heads, total_seq, head_dim]

        # Reshape back: [B, total_seq, D]
        out = out.transpose(1, 2).reshape(batch_size, total_seq, self.attn.embed_dim)

        # Output projection
        out = self.attn.out_proj(out)

        return out


## ---------------------------------------------------------------------------
## MMDiT
## ---------------------------------------------------------------------------

class MMDiT(nn.Module):
    """Multi-Modal Diffusion Transformer for Pyramidal Flow Matching.

    Full MM-DiT model with 24 transformer blocks, patchify/unpatchify,
    timestep/stage conditioning, and dual-stream visual+text processing.

    Based on SD3 Medium architecture (Esser et al., 2024) with modifications:
    - Blockwise causal attention for autoregressive video generation
    - Stage embedding for pyramid stage conditioning
    - Spatial pyramid extrapolation and temporal pyramid interpolation
    - History latent conditioning for temporal pyramid

    Attributes:
        num_layers: Number of transformer blocks (24 from config).
        hidden_dim: Feature dimension (1152 from config).
        num_heads: Number of attention heads (16 from config).
        head_dim: Dimension per attention head (72 from config).
        mlp_ratio: MLP hidden dim multiplier (4.0 from config).
        patch_size: Spatial patch size (2 from config).
        latent_channels: VAE latent channels (16 from config).
        num_stages: Number of pyramid stages (3 from config).
        t5_embed_dim: T5 embedding dimension (4096 from config).
        clip_embed_dim: CLIP embedding dimension (768 from config).
        t5_max_length: T5 max sequence length (256 from config).
        patch_dim: Tokens per patch = patch_size^2 * latent_channels (64).
        mlp_dim: MLP hidden dimension = hidden_dim * mlp_ratio (4608).
        use_gradient_checkpointing: Whether to use gradient checkpointing.
        patch_embed: Linear projection from patch_dim to hidden_dim.
        t5_proj: Linear projection from t5_embed_dim to hidden_dim.
        clip_proj: Linear projection from clip_embed_dim to hidden_dim.
        timestep_mlp: MLP for timestep embedding.
        stage_embedding: Learned stage embedding table.
        blocks: ModuleList of 24 MMDiTBlocks.
        final_norm: AdaLayerNorm before output projection.
        output_proj: Linear projection from hidden_dim to patch_dim.
        pos_enc: PositionalEncoding module.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initializes MMDiT from the project config.

        Reads all required values from configs/default.yaml via the
        omegaconf DictConfig (or plain dict) passed as ``config``.

        Args:
            config: Project configuration dictionary. Expected keys:
                - config.model.num_layers: 24
                - config.model.hidden_dim: 1152
                - config.model.num_heads: 16
                - config.model.head_dim: 72
                - config.model.mlp_ratio: 4.0
                - config.model.patch_size: 2
                - config.vae.latent_channels: 16
                - config.pyramid.num_stages: 3
                - config.model.text_encoder.t5_embed_dim: 4096
                - config.model.text_encoder.clip_embed_dim: 768
                - config.model.text_encoder.t5_max_length: 256
        """
        super().__init__()

        # ----------------------------------------------------------------
        # Parse configuration
        # ----------------------------------------------------------------
        model_cfg: Dict[str, Any] = config.get("model", {})
        vae_cfg: Dict[str, Any] = config.get("vae", {})
        pyramid_cfg: Dict[str, Any] = config.get("pyramid", {})
        text_enc_cfg: Dict[str, Any] = model_cfg.get("text_encoder", {})

        self.num_layers: int = int(model_cfg.get("num_layers", 24))
        self.hidden_dim: int = int(model_cfg.get("hidden_dim", 1152))
        self.num_heads: int = int(model_cfg.get("num_heads", 16))
        self.head_dim: int = int(model_cfg.get("head_dim", 72))
        self.mlp_ratio: float = float(model_cfg.get("mlp_ratio", 4.0))
        self.patch_size: int = int(model_cfg.get("patch_size", 2))
        self.latent_channels: int = int(vae_cfg.get("latent_channels", 16))
        self.num_stages: int = int(pyramid_cfg.get("num_stages", 3))
        self.t5_embed_dim: int = int(text_enc_cfg.get("t5_embed_dim", 4096))
        self.clip_embed_dim: int = int(text_enc_cfg.get("clip_embed_dim", 768))
        self.t5_max_length: int = int(text_enc_cfg.get("t5_max_length", 256))

        # Derived dimensions
        self.patch_dim: int = (
            self.patch_size * self.patch_size * self.latent_channels
        )  # 2*2*16 = 64
        self.mlp_dim: int = int(self.hidden_dim * self.mlp_ratio)  # 4608

        # Gradient checkpointing flag (memory optimization for large-scale training)
        self.use_gradient_checkpointing: bool = bool(
            model_cfg.get("use_gradient_checkpointing", False)
        )

        # ----------------------------------------------------------------
        # Patch embedding: patch_dim → hidden_dim
        # ----------------------------------------------------------------
        self.patch_embed: nn.Linear = nn.Linear(
            self.patch_dim, self.hidden_dim, bias=True
        )
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)

        # ----------------------------------------------------------------
        # Text conditioning projections
        # ----------------------------------------------------------------
        # T5 sequence embeddings: t5_embed_dim → hidden_dim
        self.t5_proj: nn.Linear = nn.Linear(
            self.t5_embed_dim, self.hidden_dim, bias=True
        )
        nn.init.xavier_uniform_(self.t5_proj.weight)
        nn.init.zeros_(self.t5_proj.bias)

        # CLIP pooled embedding: clip_embed_dim → hidden_dim (added to c)
        self.clip_proj: nn.Linear = nn.Linear(
            self.clip_embed_dim, self.hidden_dim, bias=True
        )
        nn.init.xavier_uniform_(self.clip_proj.weight)
        nn.init.zeros_(self.clip_proj.bias)

        # ----------------------------------------------------------------
        # Timestep embedding MLP
        # Sinusoidal(256) → Linear(256, hidden_dim) → SiLU → Linear(hidden_dim, hidden_dim)
        # ----------------------------------------------------------------
        self._timestep_embed_dim: int = 256
        self.timestep_mlp: nn.Sequential = nn.Sequential(
            nn.Linear(self._timestep_embed_dim, self.hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
        )
        nn.init.xavier_uniform_(self.timestep_mlp[0].weight)
        nn.init.zeros_(self.timestep_mlp[0].bias)
        nn.init.xavier_uniform_(self.timestep_mlp[2].weight)
        nn.init.zeros_(self.timestep_mlp[2].bias)

        # ----------------------------------------------------------------
        # Stage embedding: learned embedding for K=3 pyramid stages
        # ----------------------------------------------------------------
        self.stage_embedding: nn.Embedding = nn.Embedding(
            self.num_stages, self.hidden_dim
        )
        nn.init.normal_(self.stage_embedding.weight, std=0.02)

        # ----------------------------------------------------------------
        # Transformer blocks: 24 MMDiTBlocks
        # ----------------------------------------------------------------
        self.blocks: nn.ModuleList = nn.ModuleList([
            MMDiTBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_ratio=self.mlp_ratio,
                context_dim=self.hidden_dim,  # Joint attention uses hidden_dim
                dropout=0.0,
            )
            for _ in range(self.num_layers)
        ])

        # ----------------------------------------------------------------
        # Output