## model/visual_encoder.py
"""Bidirectional visual transformer encoder for NaViL.

This module implements the visual encoder component of the NaViL native
multimodal architecture (Section 3.1 and 4.1 of the paper). The encoder
converts raw image patches into semantic visual token embeddings using
bidirectional (non-causal) self-attention and 2D-RoPE for spatial position
encoding.

Key design choices (from the paper):
- Bidirectional attention (is_causal=False): all patches attend to all others,
  enabling global spatial context extraction.
- 2D-RoPE: captures row and column positions independently within each
  attention head.
- SwiGLU FFN: same activation as the LLM for architectural consistency.
- Pre-norm residual connections: improves training stability.
- Shared RoPE2D instance across all layers: memory-efficient since RoPE
  has no learned parameters.
- depth=0 degenerate case: encoder reduces to a pure patch embedding layer.

Architecture parameters (NaViL-2B from Table 6):
    depth=24, width=1472, mlp_width=5888, num_heads=23, patch_size=16
    head_dim = 1472 // 23 = 64 (standard 64-dim head)

Architecture parameters (NaViL-9B from Table 6):
    depth=32, width=1792, mlp_width=7168, num_heads=28, patch_size=16
    head_dim = 1792 // 28 = 64 (standard 64-dim head)

Config alignment (configs/navil_2b.yaml):
    model.visual_encoder.depth:     24
    model.visual_encoder.width:     1472
    model.visual_encoder.mlp_width: 5888
    model.visual_encoder.num_heads: 23
    model.visual_encoder.patch_size: 16
    inference.patch_size:            16
    inference.image_pad_multiple:    32
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.rope_2d import RoPE2D


class VisualEncoderLayer(nn.Module):
    """A single bidirectional transformer block for the visual encoder.

    Structurally mirrors an LLM transformer layer but uses full
    (bidirectional) attention and 2D-RoPE for spatial position encoding.
    All projection matrices use bias=False, consistent with the LLM
    architecture convention.

    Architecture:
        x = x + Attention(RMSNorm(x), grid_hw)   # pre-norm + residual
        x = x + FFN(RMSNorm(x))                  # pre-norm + residual

    Attention uses F.scaled_dot_product_attention with is_causal=False
    (bidirectional — all tokens attend to all other tokens).

    FFN uses SwiGLU: down_proj(SiLU(gate_proj(x)) * up_proj(x))

    Args:
        hidden_size: Token embedding dimension (e.g., 1472 for NaViL-2B).
        num_heads:   Number of attention heads (e.g., 23 for NaViL-2B).
                     Must divide hidden_size evenly.
        mlp_width:   FFN intermediate dimension (e.g., 5888 for NaViL-2B).
        rope2d:      Shared RoPE2D instance for 2D rotary position embeddings.
                     Stored as a plain attribute (not a sub-module) since it
                     has no learned parameters and is shared across layers.

    Raises:
        ValueError: If hidden_size is not divisible by num_heads.

    Example::

        rope2d = RoPE2D(dim=64, max_height=256, max_width=256)
        layer = VisualEncoderLayer(
            hidden_size=1472, num_heads=23, mlp_width=5888, rope2d=rope2d
        )
        x = torch.randn(2, 196, 1472)  # (B, N, hidden_size)
        out = layer(x, grid_hw=(14, 14))  # (2, 196, 1472)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_width: int,
        rope2d: RoPE2D,
    ) -> None:
        """Initialise the VisualEncoderLayer with all projection matrices.

        Args:
            hidden_size: Token embedding dimension.
            num_heads:   Number of attention heads. Must divide hidden_size.
            mlp_width:   FFN intermediate (gate/up) dimension.
            rope2d:      Shared RoPE2D instance. Stored as a plain attribute
                         (not registered as a sub-module) to avoid duplicate
                         state_dict entries when shared across layers.

        Raises:
            ValueError: If hidden_size % num_heads != 0.
        """
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by "
                f"num_heads={num_heads}. Got remainder "
                f"{hidden_size % num_heads}."
            )

        self.hidden_size: int = hidden_size
        self.num_heads: int = num_heads
        self.mlp_width: int = mlp_width
        self.head_dim: int = hidden_size // num_heads

        # ------------------------------------------------------------------ #
        # Pre-attention and pre-FFN layer normalisation (RMSNorm).            #
        # torch.nn.RMSNorm is available in PyTorch >= 2.1 (we use 2.3.0).    #
        # eps=1e-6 is standard for LLM-style RMSNorm.                         #
        # ------------------------------------------------------------------ #
        self.norm1: nn.RMSNorm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.norm2: nn.RMSNorm = nn.RMSNorm(hidden_size, eps=1e-6)

        # ------------------------------------------------------------------ #
        # Attention projection matrices.                                       #
        # Fused QKV projection: one Linear(hidden_size, 3*hidden_size)        #
        # is more efficient than three separate projections.                   #
        # ------------------------------------------------------------------ #
        self.qkv_proj: nn.Linear = nn.Linear(
            hidden_size, 3 * hidden_size, bias=False
        )
        self.out_proj: nn.Linear = nn.Linear(
            hidden_size, hidden_size, bias=False
        )

        # ------------------------------------------------------------------ #
        # SwiGLU FFN projection matrices.                                      #
        # gate_proj and up_proj both map hidden_size → mlp_width.             #
        # down_proj maps mlp_width → hidden_size.                             #
        # ------------------------------------------------------------------ #
        self.gate_proj: nn.Linear = nn.Linear(
            hidden_size, mlp_width, bias=False
        )
        self.up_proj: nn.Linear = nn.Linear(
            hidden_size, mlp_width, bias=False
        )
        self.down_proj: nn.Linear = nn.Linear(
            mlp_width, hidden_size, bias=False
        )

        # ------------------------------------------------------------------ #
        # Shared RoPE2D instance.                                              #
        # Stored as a plain Python attribute (not via register_module) so     #
        # that it does not appear multiple times in the parent module's        #
        # state_dict when shared across all VisualEncoderLayer instances.     #
        # RoPE2D has no learned parameters — only precomputed buffers.        #
        # ------------------------------------------------------------------ #
        self.rope2d: RoPE2D = rope2d

    def attention(
        self,
        x: torch.Tensor,
        grid_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """Compute bidirectional multi-head self-attention with 2D-RoPE.

        All tokens attend to all other tokens (no causal mask). 2D-RoPE
        is applied to Q and K before the attention computation to encode
        spatial row/column positions.

        Args:
            x:       Input tensor of shape (B, N, hidden_size) where
                     N = grid_hw[0] * grid_hw[1].
            grid_hw: Tuple (H_grid, W_grid) specifying the patch grid
                     dimensions. Used by RoPE2D to build 2D position
                     frequency tensors.

        Returns:
            Output tensor of shape (B, N, hidden_size).
        """
        B: int
        N: int
        B, N, _ = x.shape

        # ------------------------------------------------------------------ #
        # Step 1: Fused QKV projection.                                        #
        # qkv: (B, N, 3 * hidden_size)                                        #
        # ------------------------------------------------------------------ #
        qkv: torch.Tensor = self.qkv_proj(x)

        # Split into Q, K, V along the last dimension.
        # Each has shape (B, N, hidden_size).
        q: torch.Tensor
        k: torch.Tensor
        v: torch.Tensor
        q, k, v = qkv.chunk(3, dim=-1)

        # ------------------------------------------------------------------ #
        # Step 2: Reshape for multi-head attention.                            #
        # (B, N, hidden_size) → (B, N, num_heads, head_dim)                  #
        # ------------------------------------------------------------------ #
        q = q.view(B, N, self.num_heads, self.head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim)
        v = v.view(B, N, self.num_heads, self.head_dim)

        # ------------------------------------------------------------------ #
        # Step 3: Apply 2D-RoPE to Q and K.                                   #
        # RoPE2D.forward expects (B, N, num_heads, head_dim) and grid_hw.     #
        # Returns the same shape with rotary embeddings applied.              #
        # ------------------------------------------------------------------ #
        q = self.rope2d(q, grid_hw)  # (B, N, num_heads, head_dim)
        k = self.rope2d(k, grid_hw)  # (B, N, num_heads, head_dim)

        # ------------------------------------------------------------------ #
        # Step 4: Transpose to (B, num_heads, N, head_dim) for SDPA.          #
        # ------------------------------------------------------------------ #
        q = q.transpose(1, 2)  # (B, num_heads, N, head_dim)
        k = k.transpose(1, 2)  # (B, num_heads, N, head_dim)
        v = v.transpose(1, 2)  # (B, num_heads, N, head_dim)

        # ------------------------------------------------------------------ #
        # Step 5: Bidirectional scaled dot-product attention.                  #
        # is_causal=False: all tokens attend to all other tokens.             #
        # This is the defining characteristic of the visual encoder vs. LLM. #
        # F.scaled_dot_product_attention uses Flash Attention when available. #
        # ------------------------------------------------------------------ #
        attn_out: torch.Tensor = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        # attn_out: (B, num_heads, N, head_dim)

        # ------------------------------------------------------------------ #
        # Step 6: Reshape and apply output projection.                         #
        # (B, num_heads, N, head_dim) → (B, N, hidden_size)                  #
        # .contiguous() is required before .view() after .transpose().        #
        # ------------------------------------------------------------------ #
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, self.hidden_size)
        out: torch.Tensor = self.out_proj(attn_out)  # (B, N, hidden_size)

        return out

    def ffn(self, x: torch.Tensor) -> torch.Tensor:
        """Compute SwiGLU feed-forward network.

        SwiGLU formulation:
            out = down_proj(SiLU(gate_proj(x)) * up_proj(x))

        The gate and up projections are separate linear layers, and their
        element-wise product forms the gated activation. This is the same
        FFN architecture used in the LLM component.

        Args:
            x: Input tensor of shape (B, N, hidden_size).

        Returns:
            Output tensor of shape (B, N, hidden_size).
        """
        # gate: (B, N, mlp_width) — SiLU-activated gate branch
        gate: torch.Tensor = F.silu(self.gate_proj(x))

        # up: (B, N, mlp_width) — linear up-projection branch
        up: torch.Tensor = self.up_proj(x)

        # Element-wise product then down-projection
        out: torch.Tensor = self.down_proj(gate * up)  # (B, N, hidden_size)

        return out

    def forward(
        self,
        x: torch.Tensor,
        grid_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """Forward pass through the VisualEncoderLayer.

        Applies pre-norm residual connections for both the attention
        sub-layer and the FFN sub-layer:

            x = x + Attention(RMSNorm(x), grid_hw)
            x = x + FFN(RMSNorm(x))

        Args:
            x:       Input tensor of shape (B, N, hidden_size).
            grid_hw: Tuple (H_grid, W_grid) for 2D-RoPE computation.
                     N must equal H_grid * W_grid.

        Returns:
            Output tensor of shape (B, N, hidden_size).
        """
        # Pre-norm attention sub-layer with residual connection.
        x = x + self.attention(self.norm1(x), grid_hw)

        # Pre-norm FFN sub-layer with residual connection.
        x = x + self.ffn(self.norm2(x))

        return x


class VisualEncoder(nn.Module):
    """Full bidirectional visual transformer encoder for NaViL.

    Converts raw image tensors into semantic visual token embeddings via:
    1. Patch embedding (Conv2d with stride=patch_size)
    2. N bidirectional transformer layers with 2D-RoPE (depth=0 skips this)
    3. Final RMSNorm

    The encoder supports the depth=0 degenerate case where it reduces to
    a pure patch embedding layer (as stated in the paper: "the visual
    encoder degenerates to a simple patch embedding layer when d=0").

    A single RoPE2D instance is shared across all VisualEncoderLayer
    instances for memory efficiency (RoPE has no learned parameters).

    Args:
        depth:      Number of transformer layers (e.g., 24 for NaViL-2B,
                    32 for NaViL-9B). Set to 0 for pure patch embedding.
        width:      Token embedding dimension (e.g., 1472 for NaViL-2B,
                    1792 for NaViL-9B).
        mlp_width:  FFN intermediate dimension (e.g., 5888 for NaViL-2B,
                    7168 for NaViL-9B).
        num_heads:  Number of attention heads (e.g., 23 for NaViL-2B,
                    28 for NaViL-9B). Must divide width evenly.
        patch_size: Patch size in pixels (16 for both NaViL-2B and 9B).
                    Images must be padded to multiples of patch_size before
                    being passed to forward().

    Raises:
        ValueError: If width is not divisible by num_heads.

    Example::

        # NaViL-2B visual encoder
        encoder = VisualEncoder(
            depth=24, width=1472, mlp_width=5888,
            num_heads=23, patch_size=16
        )
        images = torch.randn(2, 3, 448, 448)  # (B, C, H, W)
        tokens, grid_hw = encoder(images)
        # tokens: (2, 784, 1472)  — 28*28=784 patches
        # grid_hw: (28, 28)
    """

    def __init__(
        self,
        depth: int = 24,
        width: int = 1472,
        mlp_width: int = 5888,
        num_heads: int = 23,
        patch_size: int = 16,
    ) -> None:
        """Initialise the VisualEncoder with all sub-modules.

        Args:
            depth:      Number of transformer layers. 0 = pure patch embedding.
            width:      Token embedding dimension.
            mlp_width:  FFN intermediate dimension.
            num_heads:  Number of attention heads. Must divide width.
            patch_size: Patch size in pixels (stride of patch embedding).

        Raises:
            ValueError: If width % num_heads != 0.
        """
        super().__init__()

        if width % num_heads != 0:
            raise ValueError(
                f"width={width} must be divisible by num_heads={num_heads}. "
                f"Got remainder {width % num_heads}. "
                f"For NaViL-2B: 1472 / 23 = 64 exactly. "
                f"For NaViL-9B: 1792 / 28 = 64 exactly."
            )

        self.depth: int = depth
        self.width: int = width
        self.mlp_width: int = mlp_width
        self.num_heads: int = num_heads
        self.patch_size: int = patch_size
        self.head_dim: int = width // num_heads  # 64 for both NaViL-2B and 9B

        # ------------------------------------------------------------------ #
        # Patch embedding layer.                                               #
        # Conv2d(3, width, kernel_size=patch_size, stride=patch_size)         #
        # converts (B, 3, H, W) → (B, width, H//patch_size, W//patch_size).  #
        # Non-overlapping patches: stride == kernel_size.                     #
        # ------------------------------------------------------------------ #
        self.patch_embed: nn.Conv2d = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,  # Patch embedding typically uses bias (ViT convention)
        )

        # ------------------------------------------------------------------ #
        # Shared RoPE2D instance.                                              #
        # max_height=max_width=256 supports images up to 256*16=4096 pixels   #
        # per side, well beyond practical use (max_image_patches=4096 in      #
        # config corresponds to a 64*64=4096 patch grid at most).             #
        # ------------------------------------------------------------------ #
        self.rope2d: RoPE2D = RoPE2D(
            dim=self.head_dim,   # 64 for both NaViL-2B and 9B
            max_height=256,
            max_width=256,
            base=10000.0,
        )

        # ------------------------------------------------------------------ #
        # Transformer layers.                                                  #
        # All layers share the same rope2d instance.                          #
        # When depth=0, this is an empty ModuleList (no transformer layers).  #
        # ------------------------------------------------------------------ #
        self.layers: nn.ModuleList = nn.ModuleList(
            [
                VisualEncoderLayer(
                    hidden_size=width,
                    num_heads=num_heads,
                    mlp_width=mlp_width,
                    rope2d=self.rope2d,
                )
                for _ in range(depth)
            ]
        )

        # ------------------------------------------------------------------ #
        # Final layer normalisation applied after all transformer layers.     #
        # ------------------------------------------------------------------ #
        self.norm: nn.RMSNorm = nn.RMSNorm(width, eps=1e-6)

    def get_grid_size(self, H: int, W: int) -> Tuple[int, int]:
        """Compute the patch grid dimensions for an image of size (H, W).

        Args:
            H: Image height in pixels. Should be a multiple of patch_size
               (images are padded to multiples of 32 per config, and
               patch_size=16 divides 32 evenly).
            W: Image width in pixels. Same constraint as H.

        Returns:
            Tuple (H_grid, W_grid) where:
                H_grid = H // patch_size
                W_grid = W // patch_size

        Example::

            encoder = VisualEncoder(patch_size=16)
            grid_hw = encoder.get_grid_size(448, 448)
            # Returns (28, 28)
        """
        return (H // self.patch_size, W // self.patch_size)

    def forward(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Encode a batch of images into visual token embeddings.

        Pipeline:
            1. Patch embedding: (B, 3, H, W) → (B, width, H_grid, W_grid)
            2. Flatten spatial dims: (B, width, H_grid, W_grid) → (B, N, width)
            3. If depth=0: apply norm and return (pure patch embedding mode)
            4. Pass through all transformer layers with 2D-RoPE
            5. Apply final RMSNorm
            6. Return (tokens, grid_hw)

        Args:
            images: Float tensor of shape (B, 3, H, W). Images should be
                    pre-processed (padded to multiples of 32, normalised)
                    by ImagePreprocessor before being passed here.
                    H and W must be multiples of patch_size (16).

        Returns:
            A tuple (tokens, grid_hw) where:
                tokens:  Float tensor of shape (B, N, width) containing
                         the encoded visual token embeddings.
                         N = H_grid * W_grid = (H // patch_size) * (W // patch_size).
                grid_hw: Tuple (H_grid, W_grid) specifying the spatial
                         grid dimensions. Needed by Connector (pixel shuffle)
                         and MultiScalePacking (end_of_line insertion).

        Example::

            encoder = VisualEncoder(
                depth=24, width=1472, mlp_width=5888,
                num_heads=23, patch_size=16
            )
            images = torch.randn(2, 3, 448, 448)
            tokens, grid_hw = encoder(images)
            # tokens.shape: (2, 784, 1472)
            # grid_hw: (28, 28)
        """
        # ------------------------------------------------------------------ #
        # Step 1: Patch embedding.                                             #
        # (B, 3, H, W) → (B, width, H_grid, W_grid)                         #
        # ------------------------------------------------------------------ #
        x: torch.Tensor = self.patch_embed(images)
        # x: (B, width, H_grid, W_grid)

        # ------------------------------------------------------------------ #
        # Step 2: Extract grid dimensions and flatten spatial dims.            #
        # ------------------------------------------------------------------ #
        B: int
        C: int
        H_grid: int
        W_grid: int
        B, C, H_grid, W_grid = x.shape
        grid_hw: Tuple[int, int] = (H_grid, W_grid)

        # Flatten spatial dimensions: (B, width, H_grid, W_grid) → (B, N, width)
        # .flatten(2) merges dims 2 and 3: (B, width, H_grid*W_grid)
        # .transpose(1, 2) swaps channel and sequence dims: (B, N, width)
        x = x.flatten(2).transpose(1, 2)
        # x: (B, H_grid * W_grid, width)

        # ------------------------------------------------------------------ #
        # Step 3: Handle depth=0 degenerate case.                              #
        # When depth=0, the encoder is a pure patch embedding layer.          #
        # Apply final norm and return immediately without any transformer      #
        # layers (self.layers is an empty ModuleList).                        #
        # ------------------------------------------------------------------ #
        if self.depth == 0:
            x = self.norm(x)
            return x, grid_hw

        # ------------------------------------------------------------------ #
        # Step 4: Pass through all bidirectional transformer layers.           #
        # Each layer receives the full token sequence and grid dimensions      #
        # for 2D-RoPE computation.                                            #
        # ------------------------------------------------------------------ #
        layer: VisualEncoderLayer
        for layer in self.layers:
            x = layer(x, grid_hw)
        # x: (B, N, width)

        # ------------------------------------------------------------------ #
        # Step 5: Apply final RMSNorm.                                         #
        # ------------------------------------------------------------------ #
        x = self.norm(x)
        # x: (B, N, width)

        # ------------------------------------------------------------------ #
        # Step 6: Return tokens and grid dimensions.                           #
        # grid_hw is needed downstream by Connector and MultiScalePacking.   #
        # ------------------------------------------------------------------ #
        return x, grid_hw
