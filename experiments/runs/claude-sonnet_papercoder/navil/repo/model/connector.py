## model/connector.py
"""Connector module bridging the visual encoder and LLM in NaViL.

This module implements the ``Connector`` class, which performs two operations
to bridge the visual encoder's output token space to the LLM's input embedding
space:

1. **Pixel shuffle (space-to-depth):** Merges r×r neighboring tokens in the
   spatial grid into a single token with r² times the channel dimension. This
   reduces the visual token sequence length by a factor of r², which is critical
   for keeping the multimodal sequence within the LLM's context window.

2. **MLP projection:** Projects the merged tokens from ``visual_dim * r²``
   dimensions to ``llm_dim`` via a two-layer MLP with GELU activation.

Architecture (from the paper, Section 3.1):
    "C is the connector which downsamples the encoded image embeddings through
    pixel shuffle and projects them to the LLM's feature space by a MLP."

Config alignment (configs/navil_2b.yaml):
    model.connector.pixel_shuffle_factor: 2
    model.visual_encoder.width:           1472  → visual_dim
    model.llm.width:                      2048  → llm_dim

    After pixel shuffle with r=2:
        - Token count reduced by 4× (r²=4)
        - Channel dimension increased from 1472 to 1472*4=5888
        - MLP maps 5888 → 2048

Pixel shuffle implementation note:
    We implement token-space pixel shuffle (space-to-depth) via reshape and
    permute operations, NOT torch.nn.PixelShuffle (which operates on image
    tensors and performs the inverse spatial operation).

    The reshape sequence for a grid of (H, W) tokens each of dimension D:
        (B, H*W, D)
        → (B, H, W, D)                    # restore spatial grid
        → (B, H//r, r, W//r, r, D)        # group r×r blocks
        → (B, H//r, W//r, r, r, D)        # permute block dims adjacent to D
        → (B, H//r, W//r, r*r*D)          # merge block dims with D
        → (B, (H//r)*(W//r), r*r*D)       # flatten spatial dims

Interaction with other components:
    - Upstream: VisualEncoder.forward returns (tokens, grid_hw) where
      tokens.shape = (B, H*W, visual_dim). Connector receives these directly.
    - Downstream: NaViLModel.build_multimodal_embeds uses the returned
      projected_tokens (B, N_compressed, llm_dim) to fill image token
      positions. The returned new_grid_hw drives <end_of_line> insertion
      in MultiScalePacking.pack_image_tokens.
"""

from typing import Tuple

import torch
import torch.nn as nn


class Connector(nn.Module):
    """Bridges visual encoder output to LLM input via pixel shuffle and MLP.

    Performs token-space pixel shuffle (space-to-depth) to compress the
    visual token sequence by merging r×r neighboring tokens, then projects
    the merged tokens to the LLM's hidden dimension via a two-layer MLP.

    The pixel shuffle factor ``r`` controls the compression ratio:
    - Token count is reduced by r² (e.g., r=2 → 4× fewer tokens)
    - Channel dimension is increased by r² before MLP projection

    Args:
        visual_dim:           Output dimension of the visual encoder
                              (e.g., 1472 for NaViL-2B). This is the
                              ``width`` parameter of ``VisualEncoder``.
        llm_dim:              Input embedding dimension of the LLM
                              (e.g., 2048 for NaViL-2B). This is the
                              ``width`` parameter of ``MoELLM``.
        pixel_shuffle_factor: Spatial downsampling factor r. The token
                              grid is compressed by r× in each spatial
                              dimension, reducing token count by r².
                              Defaults to 2 (from configs/navil_2b.yaml:
                              model.connector.pixel_shuffle_factor=2).

    Raises:
        ValueError: If ``pixel_shuffle_factor`` is less than 1.
        ValueError: If ``visual_dim`` or ``llm_dim`` is not positive.

    Example::

        # NaViL-2B connector
        connector = Connector(
            visual_dim=1472,
            llm_dim=2048,
            pixel_shuffle_factor=2,
        )
        # visual_tokens from VisualEncoder: (B, H*W, 1472)
        # grid_hw from VisualEncoder: (H, W)
        projected, new_grid = connector(visual_tokens, grid_hw=(28, 28))
        # projected.shape: (2, 196, 2048)  — 28*28=784 → 14*14=196 tokens
        # new_grid: (14, 14)
    """

    def __init__(
        self,
        visual_dim: int = 1472,
        llm_dim: int = 2048,
        pixel_shuffle_factor: int = 2,
    ) -> None:
        """Initialise the Connector with pixel shuffle factor and MLP projector.

        Args:
            visual_dim:           Visual encoder output dimension.
                                  Default: 1472 (NaViL-2B visual encoder width).
            llm_dim:              LLM input embedding dimension.
                                  Default: 2048 (NaViL-2B LLM width).
            pixel_shuffle_factor: Spatial compression factor r ≥ 1.
                                  Default: 2 (from navil_2b.yaml config).

        Raises:
            ValueError: If any argument is out of valid range.
        """
        super().__init__()

        if pixel_shuffle_factor < 1:
            raise ValueError(
                f"pixel_shuffle_factor must be >= 1, got {pixel_shuffle_factor}. "
                "A factor of 1 means no spatial compression (identity pixel shuffle)."
            )
        if visual_dim <= 0:
            raise ValueError(
                f"visual_dim must be positive, got {visual_dim}."
            )
        if llm_dim <= 0:
            raise ValueError(
                f"llm_dim must be positive, got {llm_dim}."
            )

        self.visual_dim: int = visual_dim
        self.llm_dim: int = llm_dim
        self.pixel_shuffle_factor: int = pixel_shuffle_factor

        # ------------------------------------------------------------------ #
        # Compute MLP input dimension after pixel shuffle.                     #
        # Pixel shuffle merges r×r tokens → channel dim multiplied by r².    #
        # Example (NaViL-2B): 1472 * 2² = 1472 * 4 = 5888                   #
        # ------------------------------------------------------------------ #
        r: int = pixel_shuffle_factor
        mlp_input_dim: int = visual_dim * (r * r)

        # ------------------------------------------------------------------ #
        # Two-layer MLP projector.                                             #
        # Architecture: Linear → GELU → Linear                               #
        # Input:  mlp_input_dim = visual_dim * r²  (e.g., 5888 for NaViL-2B) #
        # Hidden: llm_dim                          (e.g., 2048 for NaViL-2B) #
        # Output: llm_dim                          (e.g., 2048 for NaViL-2B) #
        #                                                                      #
        # GELU activation is consistent with InternLM2's connector style.    #
        # bias=True is the default and appropriate for projection layers.     #
        # ------------------------------------------------------------------ #
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(mlp_input_dim, llm_dim, bias=True),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim, bias=True),
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        grid_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Apply pixel shuffle compression and MLP projection to visual tokens.

        Compresses the spatial token grid by merging r×r neighboring tokens
        into a single token (space-to-depth), then projects to the LLM's
        embedding dimension.

        Reshape sequence (r = pixel_shuffle_factor):
            (B, H*W, D)
            → (B, H, W, D)                 restore spatial grid
            → (B, H//r, r, W//r, r, D)     group r×r blocks
            → (B, H//r, W//r, r, r, D)     permute block dims adjacent to D
            → (B, H//r, W//r, r*r*D)       merge block dims with channel dim
            → (B, (H//r)*(W//r), r*r*D)    flatten spatial dims
            → MLP → (B, (H//r)*(W//r), llm_dim)

        Args:
            visual_tokens: Float tensor of shape (B, H*W, visual_dim)
                           containing the visual encoder's output tokens.
                           B is the batch size, H*W is the total number of
                           patches (must equal grid_hw[0] * grid_hw[1]),
                           and visual_dim is the encoder's output dimension.
            grid_hw:       Tuple (H, W) specifying the spatial grid dimensions
                           of the visual tokens. Both H and W must be divisible
                           by ``pixel_shuffle_factor``.

        Returns:
            A tuple (projected_tokens, new_grid_hw) where:
                projected_tokens: Float tensor of shape
                                  (B, (H//r)*(W//r), llm_dim) containing
                                  the compressed and projected visual tokens
                                  ready for the LLM.
                new_grid_hw:      Tuple (H//r, W//r) specifying the spatial
                                  grid dimensions after compression. Used by
                                  MultiScalePacking to insert <end_of_line>
                                  tokens after each row of the compressed grid.

        Raises:
            AssertionError: If H or W is not divisible by pixel_shuffle_factor,
                            indicating that image padding was not applied
                            correctly upstream (ImagePreprocessor.pad_to_multiple
                            should ensure this).
            AssertionError: If the number of tokens in visual_tokens does not
                            match H * W from grid_hw.

        Example::

            connector = Connector(visual_dim=1472, llm_dim=2048,
                                  pixel_shuffle_factor=2)
            # Simulate VisualEncoder output for a 448×448 image
            # with patch_size=16: grid = 28×28 = 784 patches
            visual_tokens = torch.randn(2, 784, 1472)
            projected, new_grid = connector(visual_tokens, grid_hw=(28, 28))
            # projected.shape: (2, 196, 2048)  — 14*14=196 compressed tokens
            # new_grid: (14, 14)
        """
        # ------------------------------------------------------------------ #
        # Unpack inputs and validate dimensions.                               #
        # ------------------------------------------------------------------ #
        H: int
        W: int
        H, W = grid_hw

        B: int
        N: int
        D: int
        B, N, D = visual_tokens.shape

        r: int = self.pixel_shuffle_factor

        # Validate that the token count matches the declared grid dimensions.
        assert N == H * W, (
            f"visual_tokens sequence length {N} does not match "
            f"grid_hw product {H}*{W}={H * W}. "
            "Ensure grid_hw is consistent with the visual encoder output."
        )

        # Validate that H and W are divisible by the pixel shuffle factor.
        # This is guaranteed when images are padded to multiples of 32 and
        # patch_size=16 (grid dims are multiples of 2), with r=2.
        assert H % r == 0, (
            f"Grid height H={H} is not divisible by pixel_shuffle_factor={r}. "
            "Ensure ImagePreprocessor.pad_to_multiple(image, 32) is applied "
            "before encoding, so that grid dimensions are multiples of r."
        )
        assert W % r == 0, (
            f"Grid width W={W} is not divisible by pixel_shuffle_factor={r}. "
            "Ensure ImagePreprocessor.pad_to_multiple(image, 32) is applied "
            "before encoding, so that grid dimensions are multiples of r."
        )

        # ------------------------------------------------------------------ #
        # Pixel shuffle (space-to-depth) via reshape and permute.             #
        # ------------------------------------------------------------------ #

        # Step 1: Restore spatial grid from flat token sequence.
        # (B, H*W, D) → (B, H, W, D)
        x: torch.Tensor = visual_tokens.view(B, H, W, D)

        # Step 2: Group r×r spatial blocks.
        # (B, H, W, D) → (B, H//r, r, W//r, r, D)
        # This splits each spatial dimension into (num_blocks, block_size).
        H_out: int = H // r
        W_out: int = W // r
        x = x.view(B, H_out, r, W_out, r, D)

        # Step 3: Permute to bring block dimensions adjacent to channel dim.
        # (B, H//r, r, W//r, r, D) → (B, H//r, W//r, r, r, D)
        # Permutation: (0, 1, 3, 2, 4, 5) — swap dims 2 and 3
        x = x.permute(0, 1, 3, 2, 4, 5)
        # x: (B, H//r, W//r, r, r, D)

        # .contiguous() is required before .view() after .permute() because
        # permute creates a non-contiguous tensor in memory.
        x = x.contiguous()

        # Step 4: Merge block dimensions with channel dimension.
        # (B, H//r, W//r, r, r, D) → (B, H//r, W//r, r*r*D)
        merged_dim: int = r * r * D
        x = x.view(B, H_out, W_out, merged_dim)

        # Step 5: Flatten spatial dimensions.
        # (B, H//r, W//r, r*r*D) → (B, (H//r)*(W//r), r*r*D)
        N_out: int = H_out * W_out
        x = x.view(B, N_out, merged_dim)
        # x: (B, (H//r)*(W//r), visual_dim * r²)

        # ------------------------------------------------------------------ #
        # MLP projection: visual_dim * r² → llm_dim                          #
        # ------------------------------------------------------------------ #
        projected_tokens: torch.Tensor = self.mlp(x)
        # projected_tokens: (B, (H//r)*(W//r), llm_dim)

        # ------------------------------------------------------------------ #
        # Compute and return new grid dimensions.                              #
        # ------------------------------------------------------------------ #
        new_grid_hw: Tuple[int, int] = (H_out, W_out)

        return projected_tokens, new_grid_hw
