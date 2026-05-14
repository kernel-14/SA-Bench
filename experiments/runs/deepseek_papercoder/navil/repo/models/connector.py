# models/connector.py

"""
Connector module that bridges the visual encoder and the modality‑aware LLM
in the NaViL architecture.

The paper (Sec. 3.1) describes the connector as a component that
“*downsamples the encoded image embeddings through pixel shuffle and
projects them to the LLM’s feature space by a MLP*”.  This implementation
faithfully follows that description:

* Spatial downsampling via :class:`torch.nn.PixelUnshuffle` (the inverse of
  pixel shuffle, which reduces the number of visual tokens by aggregating
  spatial information).
* A 2‑layer MLP with SiLU activation that projects the resulting features
  from the visual encoder’s hidden dimension to the LLM’s hidden dimension.

The module is instantiated once for the entire model and processes each
image (or each scale in multi‑scale mode) independently through a
sample‑wise loop.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class Connector(nn.Module):
    """
    Connector that applies pixel unshuffle + MLP projection to a batch of
    visual tokens.

    Args:
        in_dim: Hidden dimension of the visual encoder (e.g., 1472 for
            NaViL‑2B).
        out_dim: Hidden dimension of the LLM (e.g., 2048 for
            InternLM2‑1.8B).
        shuffle_ratio: Spatial downscaling factor.  Each spatial
            dimension is divided by this value.  Default is 2,
            resulting in a 4× token reduction.
        mlp_hidden_dim: Internal dimension of the intermediate MLP layer.
            If ``None`` (default), it is set to ``out_dim``, making the MLP
            projection effectively a single linear layer with an
            interleaved SiLU activation.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        shuffle_ratio: int = 2,
        mlp_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.shuffle_ratio = shuffle_ratio

        # Pixel unshuffle expects input of shape (B, C, H, W).
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=shuffle_ratio)

        # After unshuffle, the channel dimension grows by ratio^2.
        intermediate_dim = in_dim * (shuffle_ratio**2)

        if mlp_hidden_dim is None:
            mlp_hidden_dim = out_dim  # collapse to a single projection with SiLU

        # MLP: linear → SiLU → linear
        self.mlp = nn.Sequential(
            nn.Linear(intermediate_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, out_dim),
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        patch_grid_sizes: List[Tuple[int, int]],
    ) -> List[torch.Tensor]:
        """
        Downsample and project the visual tokens for each sample in the batch.

        Args:
            visual_tokens: Output of the visual encoder, shape
                ``(B, N, in_dim)``.  ``N`` is the total number of patch
                positions (product of the patch grid dimensions).
            patch_grid_sizes: A list of ``B`` tuples ``(h_patches, w_patches)``
                giving the 2D grid layout of the patches for each sample.
                ``h_patches * w_patches`` must equal the corresponding
                ``N``.

        Returns:
            A list of ``B`` tensors, the *i*‑th of shape
            ``(new_N_i, out_dim)`` where
            ``new_N_i = (h_patches // ratio) * (w_patches // ratio)``.
        """
        B = len(patch_grid_sizes)
        outputs: List[torch.Tensor] = []

        for i in range(B):
            tokens_i = visual_tokens[i]  # (N_i, in_dim)
            h, w = patch_grid_sizes[i]
            N_i = tokens_i.shape[0]
            assert (
                h * w == N_i
            ), f"Sample {i}: expected {h}*{w}={h*w} tokens, got {N_i}"

            # 1. Reshape to 2D feature map
            # tokens_i: (h*w, in_dim) → (h, w, in_dim) → (in_dim, h, w)
            feat_map = tokens_i.reshape(h, w, -1).permute(2, 0, 1)

            # 2. Pixel unshuffle (needs a batch dimension)
            # Insert batch dim → (1, in_dim, h, w) → unshuffle
            feat_map = feat_map.unsqueeze(0)
            feat_map = self.pixel_unshuffle(feat_map)  # (1, intermediate_dim, h//r, w//r)

            # 3. Remove batch dim and flatten spatial dimensions
            feat_map = feat_map.squeeze(0)  # (intermediate_dim, new_h, new_w)
            # For the MLP we need (num_tokens, intermediate_dim)
            feat_ch = feat_map.shape[0]
            feat_map = feat_map.reshape(feat_ch, -1).transpose(0, 1)  # (new_N, intermediate_dim)

            # 4. Apply MLP projection
            projected = self.mlp(feat_map)  # (new_N, out_dim)
            outputs.append(projected)

        return outputs

