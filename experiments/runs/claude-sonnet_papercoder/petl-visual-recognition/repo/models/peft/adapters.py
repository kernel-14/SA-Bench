## Code: models/peft/adapters.py

```python
## models/peft/adapters.py
"""Adapter-based PEFT methods for the PEFT Visual Recognition reproduction study.

This module implements all 5 adapter-based PEFT methods described in Section 2.2
and Appendix B.2.2 of the paper:

    "Lessons and Insights from a Unifying Study of Parameter-Efficient
    Fine-Tuning (PEFT) in Visual Recognition"

Implemented methods:
- HoulsbyAdapter: two sequential adapters per layer (after MSA and after MLP)
- PfeifferAdapter: one sequential adapter per layer (after MLP only)
- AdaptFormer: one parallel adapter per layer (parallel to MLP)
- ConvPassAdapter: two parallel convolutional adapters per layer
- RepAdapter: two linear group-wise adapters per layer (before MSA and before MLP)

All adapters share the bottleneck structure:
    Adapter(h) = s · W_up(σ(W_down(h))) + h
where W_down ∈ R^{r×D}, W_up ∈ R^{D×r}, s is a scalar scale factor,
σ is GELU activation, and r ≪ D is the bottleneck dimension.

Paper config references (config.yaml):
    peft_methods.houlsby_adapter.search_grid: scale [0.01,0.1,1,10], bottleneck [4,8,16,32]
    peft_methods.pfeiffer_adapter.search_grid: scale [0.01,0.1,1,10], bottleneck [4,8,16,32]
    peft_methods.adaptformer.search_grid: scale [0.05,0.1,0.2], bottleneck [4,16,32]
    peft_methods.convpass.search_grid: scale [0.01,0.1,1,10,100], bottleneck [8,16]
    peft_methods.repadapter.search_grid: scale [0.1,0.5,1,5,10], bottleneck [8,16,32]
    peft_methods.repadapter.groups: 8
    backbones.imagenet21k_vit.embed_dim: 768
    backbones.imagenet21k_vit.num_layers: 12

Typical usage (called by PEFTFactory):
    import copy
    backbone = copy.deepcopy(vit_wrapper.get_backbone())
    # freeze_backbone() called before this

    adapter_modules = apply_houlsby_adapter(
        backbone=backbone,
        embed_dim=768,
        bottleneck=8,
        scale=0.1,
    )
    # adapter_modules is a list of HoulsbyAdapterBlock (one per layer)
    # Each block's parameters are trainable; backbone params remain frozen
"""

import logging
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architecture constants (config.yaml: backbones.imagenet21k_vit)
# ---------------------------------------------------------------------------
_DEFAULT_EMBED_DIM: int = 768    # config.yaml: backbones.imagenet21k_vit.embed_dim
_DEFAULT_NUM_LAYERS: int = 12    # config.yaml: backbones.imagenet21k_vit.num_layers

# Spatial grid size for ViT-B/16 with 224×224 input and patch size 16.
# N = (224/16)^2 = 196 patch tokens; spatial grid = 14×14.
_DEFAULT_SPATIAL_SIZE: int = 14  # sqrt(196) = 14

# Default number of groups for RepAdapter group-wise projection.
# config.yaml: peft_methods.repadapter.groups: 8
_DEFAULT_GROUPS: int = 8


# ===========================================================================
# Base class: BottleneckAdapter
# ===========================================================================

class BottleneckAdapter(nn.Module):
    """Bottleneck adapter module: W_down → GELU → W_up + residual.

    Implements the core adapter computation shared by Houlsby, Pfeiffer,
    and AdaptFormer:
        Adapter(h) = s · W_up(σ(W_down(h))) + h

    Initialization ensures identity mapping at the start of training:
    - W_down: kaiming_uniform_ (standard He init)
    - W_up: zeros — ensures W_up(σ(W_down(h))) = 0 at init, so output = h

    This is the standard LoRA-style initialization for adapters and is
    critical for stable training from a pretrained backbone.

    Attributes:
        embed_dim: Input/output feature dimension D (768 for ViT-B/16).
        bottleneck: Bottleneck dimension r (r ≪ D).
        scale: Fixed scalar scale factor s (hyperparameter, not learned).
        W_down: Linear projection D → r (no bias).
        act: GELU activation function.
        W_up: Linear projection r → D (no bias), initialized to zeros.
    """

    def __init__(
        self,
        embed_dim: int = _DEFAULT_EMBED_DIM,
        bottleneck: int = 8,
        scale: float = 1.0,
    ) -> None:
        """Initialises the bottleneck adapter with identity-preserving init.

        Args:
            embed_dim: Input and output feature dimension D. Default: 768
                (config.yaml: backbones.imagenet21k_vit.embed_dim).
            bottleneck: Bottleneck dimension r. Must satisfy r < embed_dim.
                Search grids from config.yaml:
                - Houlsby/Pfeiffer: {4, 8, 16, 32}
                - AdaptFormer: {4, 16, 32}
            scale: Fixed scalar scale factor s applied to the adapter output
                before the residual addition. This is a hyperparameter chosen
                during grid search, not a learned parameter.
                Search grids from config.yaml:
                - Houlsby/Pfeiffer: {0.01, 0.1, 1.0, 10.0}
                - AdaptFormer: {0.05, 0.1, 0.2}

        Raises:
            ValueError: If bottleneck >= embed_dim or either is non-positive.
        """
        super().__init__()

        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}.")
        if bottleneck <= 0:
            raise ValueError(f"bottleneck must be positive, got {bottleneck}.")
        if bottleneck >= embed_dim:
            raise ValueError(
                f"bottleneck ({bottleneck}) must be less than embed_dim ({embed_dim}) "
                "to achieve parameter efficiency."
            )

        self.embed_dim: int = embed_dim
        self.bottleneck: int = bottleneck
        self.scale: float = scale

        # Down-projection: D → r (no bias for parameter efficiency)
        self.W_down: nn.Linear = nn.Linear(embed_dim, bottleneck, bias=False)

        # Nonlinear activation (GELU, consistent with ViT MLP)
        self.act: nn.GELU = nn.GELU()

        # Up-projection: r → D (no bias for parameter efficiency)
        self.W_up: nn.Linear = nn.Linear(bottleneck, embed_dim, bias=False)

        # ------------------------------------------------------------------
        # Identity-preserving initialization:
        # W_down: kaiming_uniform_ (standard He init for linear layers)
        # W_up: zeros — ensures adapter output = 0 at init → Adapter(h) = h
        # ------------------------------------------------------------------
        nn.init.kaiming_uniform_(self.W_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.W_up.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Applies the bottleneck adapter with residual connection.

        Args:
            h: Input feature tensor of shape (B, seq_len, D) where:
                B = batch size, seq_len = N+1 (CLS + patch tokens), D = embed_dim.

        Returns:
            Output tensor of the same shape (B, seq_len, D):
                s · W_up(GELU(W_down(h))) + h
        """
        # Down-project: (B, seq_len, D) → (B, seq_len, r)
        down: torch.Tensor = self.W_down(h)

        # Nonlinear activation: shape unchanged
        activated: torch.Tensor = self.act(down)

        # Up-project: (B, seq_len, r) → (B, seq_len, D)
        up: torch.Tensor = self.W_up(activated)

        # Scale and residual: s · up + h
        return self.scale * up + h


# ===========================================================================
# HoulsbyAdapter
# ===========================================================================

class HoulsbyAdapterBlock(nn.Module):
    """Container for Houlsby Adapter: two bottleneck adapters per Transformer layer.

    Houlsby et al. [36] insert two adapters per layer:
    - adapter_msa: applied sequentially after MSA + residual (at h5)
    - adapter_mlp: applied sequentially after MLP + residual (at h9)

    Paper equation (Appendix B.2.2):
        h5 = Adapter1(h5)    # after MSA output + residual
        h9 = Adapter2(h9)    # after MLP output + residual

    Config reference (config.yaml):
        peft_methods.houlsby_adapter.search_grid:
            adapter_scale: [0.01, 0.1, 1.0, 10.0]
            adapter_bottleneck: [4, 8, 16, 32]
        peft_methods.houlsby_adapter.params_range_M: [0.165, 1.198]

    Attributes:
        adapter_msa: BottleneckAdapter applied after MSA block.
        adapter_mlp: BottleneckAdapter applied after MLP block.
    """

    def __init__(
        self,
        embed_dim: int = _DEFAULT_EMBED_DIM,
        bottleneck: int = 8,
        scale: float = 1.0,
    ) -> None:
        """Initialises two bottleneck adapters for one Transformer layer.

        Args:
            embed_dim: Feature dimension D. Default: 768.
            bottleneck: Bottleneck dimension r. Search grid: {4, 8, 16, 32}.
            scale: Scale factor s. Search grid: {0.01, 0.1, 1.0, 10.0}.
        """
        super().__init__()

        self.adapter_msa: BottleneckAdapter = BottleneckAdapter(
            embed_dim=embed_dim,
            bottleneck=bottleneck,
            scale=scale,
        )
        self.adapter_mlp: BottleneckAdapter = BottleneckAdapter(
            embed_dim=embed_dim,
            bottleneck=bottleneck,
            scale=scale,
        )


class PfeifferAdapterBlock(nn.Module):
    """Container for Pfeiffer Adapter: one bottleneck adapter per Transformer layer.

    Pfeiffer et al. [74] insert the adapter solely after the MLP block,
    a strategy shown effective in recent studies [37].

    Paper equation (Appendix B.2.2):
        h9 = Adapter(h9)    # after MLP output + residual only

    Config reference (config.yaml):
        peft_methods.pfeiffer_adapter.search_grid:
            adapter_scale: [0.01, 0.1, 1.0, 10.0]
            adapter_bottleneck: [4, 8, 16, 32]
        peft_methods.pfeiffer_adapter.params_range_M: [0.082, 0.599]

    Attributes:
        adapter_mlp: BottleneckAdapter applied after MLP block.
    """

    def __init__(
        self,
        embed_dim: int = _DEFAULT_EMBED_DIM,
        bottleneck: int = 8,
        scale: float = 1.0,
    ) -> None:
        """Initialises one bottleneck adapter for one Transformer layer.

        Args:
            embed_dim: Feature dimension D. Default: 768.
            bottleneck: Bottleneck dimension r. Search grid: {4, 8, 16, 32}.
            scale: Scale factor s. Search grid: {0.01, 0.1, 1.0, 10.0}.
        """
        super().__init__()

        self.adapter_mlp: BottleneckAdapter = BottleneckAdapter(
            embed_dim=embed_dim,
            bottleneck=bottleneck,
            scale=scale,
        )


class AdaptFormerBlock(nn.Module):
    """Container for AdaptFormer: one parallel adapter per Transformer layer.

    AdaptFormer [11] inserts the adapter in parallel with the MLP block.
    The adapter takes the same input as the MLP (post-LN2 features) and
    its output is added to the MLP output before the residual connection.

    Paper equation (Appendix B.2.2):
        h9 = h9 + Adapter(h7)    # parallel to MLP; h7 = norm2(h5) = MLP input

    The parallel design allows domain-specific features from the adapter to
    complement domain-agnostic features from the original MLP block.

    Config reference (config.yaml):
        peft_methods.adaptformer.search_grid:
            adapter_scale: [0.05, 0.1, 0.2]
            adapter_bottleneck: [4, 16, 32]
        peft_methods.adaptformer.params_range_M: [0.082, 0.599]

    Attributes:
        adapter: BottleneckAdapter applied in parallel with MLP.
    """

    def __init__(
        self,
        embed_dim: int = _DEFAULT_EMBED_DIM,
        bottleneck: int = 16,
        scale: float = 0.1,
    ) -> None:
        """Initialises one parallel bottleneck adapter for one Transformer layer.

        Args:
            embed_dim: Feature dimension D. Default: 768.
            bottleneck: Bottleneck dimension r. Search grid: {4, 16, 32}.
            scale: Scale factor s. Search grid: {0.05, 0.1, 0.2}.
        """
        super().__init__()

        self.adapter: BottleneckAdapter = BottleneckAdapter(
            embed_dim=embed_dim,
            bottleneck=bottleneck,
            scale=scale,
        )


# ===========================================================================
# ConvPassAdapter
# ===========================================================================

class ConvPassAdapter(nn.Module):
    """Convolutional bypass adapter for Vision Transformers.

    ConvPass [43] addresses the lack of visual inductive bias in standard
    adapters by introducing a convolutional bottleneck module. It runs in
    parallel with both MSA and MLP blocks.

    Architecture:
        Convpass(h) = s · W_up(σ(Conv2d(σ(W_down(h)))))
    where:
        W_down: 1×1 conv, D → r (channel reduction)
        Conv2d: 3×3 depthwise conv, r → r (spatial processing)
        W_up:   1×1 conv, r → D (channel expansion)
        σ: GELU activation (applied twice)
        s: scalar scale factor

    Paper equation (Appendix B.2.2):
        h5 = Convpass1(h2) + h5    # parallel to MSA; h2 = norm1(h1)
        h9 = Convpass2(h7) + h9    # parallel to MLP; h7 = norm2(h5)

    Critical: patch tokens are reshaped to 2D spatial grid (14×14 for ViT-B/16)
    for the Conv2d operation. The CLS token is excluded from spatial processing
    and concatenated back after convolution.

    Config reference (config.yaml):
        peft_methods.convpass.search_grid:
            convpass_scale: [0.01, 0.1, 1.0, 10.0, 100.0]
            convpass_bottleneck: [8, 16]
            xavier_init: [true, false]
        peft_methods.convpass.params_range_M: [0.327, 0.664]

    Attributes:
        embed_dim: Input/output feature dimension D.
        bottleneck: Bottleneck channel dimension r.
        scale: Fixed scalar scale factor s.
        spatial_size: Spatial grid size (14 for ViT-B/16 with 224×224 input).
        W_down: 1×1 Conv2d for channel reduction (D → r).
        conv: 3×3 Conv2d for spatial processing (r → r, same padding).
        W_up: 1×1 Conv2d for channel expansion (r → D).
        act: GELU activation.
    """

    def __init__(
        self,
        embed_dim: int = _DEFAULT_EMBED_DIM,
        bottleneck: int = 8,
        scale: float = 1.0,
        xavier_init: bool = True,
        spatial_size: int = _DEFAULT_SPATIAL_SIZE,
    ) -> None:
        """Initialises the convolutional bypass adapter.

        Args:
            embed_dim: Input/output feature dimension D. Default: 768.
            bottleneck: Bottleneck channel dimension r. Search grid: {8, 16}.
            scale: Fixed scalar scale factor s.
                Search grid: {0.01, 0.1, 1.0, 10.0, 100.0}.
            xavier_init: If True, initialise W_down and W_up with xavier_uniform_.
                If False, use default PyTorch Conv2d initialization (kaiming_uniform).
                Search grid: {True, False}.
            spatial_size: Spatial grid size for 2D reshape. Default: 14
                (for ViT-B/16 with 224×224 input and patch size 16:
                N = (224/16)^2 = 196, sqrt(196) = 14).

        Raises:
            ValueError: If embed_dim, bottleneck, or spatial_size are non-positive.
        """
        super().__init__()

        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}.")
        if bottleneck <= 0:
            raise ValueError(f"bottleneck must be positive, got {bottleneck}.")
        if spatial_size <= 0:
            raise ValueError(f"spatial_size must be positive, got {spatial_size}.")

        self.embed_dim: int = embed_dim
        self.bottleneck: int = bottleneck
        self.scale: float = scale
        self.spatial_size: int = spatial_size

        # 1×1 convolution: channel reduction D → r
        self.W_down: nn.Conv2d = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=bottleneck,
            kernel_size=1,
            bias=False,
        )

        # 3×3 convolution: spatial processing r → r (same padding preserves size)
        self.conv: nn.Conv2d = nn.Conv2d(
            in_channels=bottleneck,
            out_channels=bottleneck,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        # 1×1 convolution: channel expansion r → D
        self.W_up: nn.Conv2d = nn.Conv2d(
            in_channels=bottleneck,
            out_channels=embed_dim,
            kernel_size=1,
            bias=False,
        )

        # GELU activation (applied after W_down and after conv)
        self.act: nn.GELU = nn.GELU()

        # ------------------------------------------------------------------
        # Initialization.
        # xavier_init=True: xavier_uniform_ for W_down and W_up.
        # xavier_init=False: default PyTorch Conv2d init (kaiming_uniform).
        # The 3×3 conv always uses default init.
        # Note: Unlike BottleneckAdapter, W_up is NOT zeroed here.
        # The scale factor s controls the initial contribution magnitude.
        # ------------------------------------------------------------------
        if xavier_init:
            nn.init.xavier_uniform_(self.W_down.weight)
            nn.init.xavier_uniform_(self.W_up.weight)
            _logger.debug("ConvPassAdapter: using xavier_uniform_ initialization.")
        else:
            # Default PyTorch Conv2d init (kaiming_uniform_) — no action needed.
            _logger.debug("ConvPassAdapter: using default (kaiming_uniform_) initialization.")

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Applies the convolutional bypass adapter.

        Handles the 2D reshape required for Conv2d:
        1. Separate CLS token from patch tokens.
        2. Reshape patch tokens to 2D spatial grid (B, D, H, W).
        3. Apply W_down → GELU → Conv2d → GELU → W_up.
        4. Reshape back to sequence format (B, N, D).
        5. Concatenate CLS token back.
        6. Scale and return (without residual — caller adds residual).

        Note: This method returns the adapter output WITHOUT the residual
        addition. The caller (wrapped block forward) adds the residual:
            h5 = Convpass1(h2) + h5

        Args:
            h: Input feature tensor of shape (B, N+1, D) where:
                B = batch size, N+1 = 197 (1 CLS + 196 patches), D = embed_dim.

        Returns:
            Adapter output tensor of shape (B, N+1, D).
            The CLS token position contains a zero vector (CLS is not processed
            by convolution; the caller adds the original h5 as residual which
            includes the CLS contribution).
        """
        batch_size: int = h.shape[0]
        seq_len: int = h.shape[1]  # N+1 = 197
        n_patches: int = seq_len - 1  # N = 196

        # ------------------------------------------------------------------
        # Step 1: Separate CLS token and patch tokens.
        # ------------------------------------------------------------------
        cls_token: torch.Tensor = h[:, 0:1, :]      # (B, 1, D)
        patch_tokens: torch.Tensor = h[:, 1:, :]    # (B, N, D)

        # ------------------------------------------------------------------
        # Step 2: Reshape patch tokens to 2D spatial grid.
        # (B, N, D) → (B, D, H, W) where H = W = spatial_size = 14
        # ------------------------------------------------------------------
        # Compute spatial size dynamically in case input resolution differs.
        spatial_h: int = int(math.sqrt(n_patches))
        spatial_w: int = spatial_h

        if spatial_h * spatial_w != n_patches:
            # Non-square patch grid — use stored spatial_size as fallback.
            spatial_h = self.spatial_size
            spatial_w = self.spatial_size
            if spatial_h * spatial_w != n_patches:
                raise ValueError(
                    f"Cannot reshape {n_patches} patch tokens into a square 2D grid. "
                    f"Expected N = spatial_size^2 = {self.spatial_size}^2 = "
                    f"{self.spatial_size**2}, got N = {n_patches}."
                )

        # Reshape: (B, N, D) → (B, D, H, W)
        # Step 1: (B, N, D) → (B, H, W, D)
        # Step 2: (B, H, W, D) → (B, D, H, W) via permute
        spatial_tokens: torch.Tensor = (
            patch_tokens
            .view(batch_size, spatial_h, spatial_w, self.embed_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        # spatial_tokens shape: (B, D, H, W) = (B, 768, 14, 14)

        # ------------------------------------------------------------------
        # Step 3: Apply convolutional pipeline.
        # W_down → GELU → Conv2d → GELU → W_up
        # ------------------------------------------------------------------
        # 1×1 conv: (B, D, H, W) → (B, r, H, W)
        x: torch.Tensor = self.W_down(spatial_tokens)

        # GELU activation
        x = self.act(x)

        # 3×3 conv: (B, r, H, W) → (B, r, H, W) (same padding)
        x = self.conv(x)

        # GELU activation
        x = self.act(x)

        # 1×1 conv: (B, r, H, W) → (B, D, H, W)
        x = self.W_up(x)

        # ------------------------------------------------------------------
        # Step 4: Reshape back to sequence format.
        # (B, D, H, W) → (B, N, D)
        # ------------------------------------------------------------------
        # (B, D, H, W) → (B, H, W, D) → (B, N, D)
        conv_out: torch.Tensor = (
            x
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(batch_size, n_patches, self.embed_dim)
        )
        # conv_out shape: (B, N, D) = (B, 196, 768)

        # ------------------------------------------------------------------
        # Step 5: Concatenate CLS token placeholder.
        # The CLS token is not processed by convolution.
        # We prepend a zero tensor at the CLS position; the caller adds
        # the original h as residual, which contributes the CLS features.
        # ------------------------------------------------------------------
        cls_placeholder: torch.Tensor = torch.zeros_like(cls_token)
        output: torch.Tensor = torch.cat([cls_placeholder, conv_out], dim=1)
        # output shape: (B, N+1, D) = (B, 197, 768)

        # ------------------------------------------------------------------
        # Step 6: Apply scale factor.
        # Note: NO residual addition here — caller handles: Convpass(h2) + h5
        # ------------------------------------------------------------------
        return self.scale * output


class ConvPassBlock(nn.Module):
    """Container for ConvPass: two convolutional adapters per Transformer layer.

    Holds convpass_msa (parallel to MSA) and convpass_mlp (parallel to MLP).

    Attributes:
        convpass_msa: ConvPassAdapter applied in parallel with MSA block.
        convpass_mlp: ConvPassAdapter applied in parallel with MLP block.
    """

    def __init__(
        self,
        embed_dim: int = _DEFAULT_EMBED_DIM,
        bottleneck: int = 8,
        scale: float = 1.0,
        xavier_init: bool = True,
        spatial_size: int = _DEFAULT_SPATIAL_SIZE,
    ) -> None:
        """Initialises two convolutional adapters for one Transformer layer.

        Args:
            embed_dim: Feature dimension D. Default: 768.
            bottleneck: Bottleneck channel dimension r. Search grid: {8, 16}.
            scale: Scale factor s. Search grid: {0.01, 0.1, 1.0, 10.0, 100.0}.
            xavier_init: If True, use xavier_uniform_ init. Search grid: {True, False}.
            spatial_size: Spatial grid size. Default: 14 for ViT-B/16.
        """
        super().__init__()

        self.convpass_msa: ConvPassAdapter = ConvPassAdapter(
            embed_dim=embed_dim,
            bottleneck=bottleneck,
            scale=scale,
            xavier_init=xavier_init,
            spatial_size=spatial_size,
        )
        self.convpass_mlp: ConvPassAdapter = ConvPassAdapter(
            embed_dim=embed_dim,
            bottleneck=bottleneck,
            scale=scale,
            xavier_init=xavier_init,
            spatial_size=spatial_size,
        )


# ===========================================================================
# RepAdapter
# ===========================================================================

class RepAdapterModule(nn.Module):
    """Linear adapter with group-wise transformation (RepAdapter).

    RepAdapter [63] found that removing the nonlinear activation in adapters
    does not degrade performance for vision tasks. It uses a linear adapter
    with group-wise transformation, placed sequentially before MSA and MLP.

    Architecture:
        RepAdapter(h) = s · φ_up(φ_down(h)) + h
        φ_down(h) = W_down · h                          (W_down ∈ R^{r×D})
        φ_up(h̃) = [W_g1·h̃_g1, ..., W_gG·h̃_gG]       (group-wise projection)

    where h̃_gi ∈ R^{r/G × (N+1)} is the i-th group of h̃ ∈ R^{r × (N+1)},
    and W_gi ∈ R^{D/G × r/G} is the group projection weight.

    Due to its linearity and sequential placement, RepAdapter can be
    re-parameterized into the original MSA/MLP weights at inference,
    incurring zero additional inference overhead.

    Paper equation (Appendix B.2.2, Table 6):
        h2 = RepAdapter1(h2)    # modifies post-LN1 features before MSA
        h7 = RepAdapter2(h7)    # modifies post-LN2 features before MLP

    Config reference (config.yaml):
        peft_methods.repadapter.search_grid:
            repadapter_scale: [0.1, 0.5, 1.0, 5.0, 10.0]
            repadapter_bottleneck: [8, 16, 32]
        peft_methods.repadapter.groups: 8
        peft_methods.repadapter.params_range_M: [0.239, 0.903]

    Attributes:
        embed_dim: Input/output feature dimension D.
        bottleneck: Bottleneck dimension r.
        scale: Fixed scalar scale factor s.
        groups: Number of groups G for group-wise projection.
        W_down: Linear projection D → r (no bias).
        W_up_groups: ModuleList of G Linear(r//G, D//G) projections.
    """

    def __init__(
        self,
        embed_dim: int = _DEFAULT_EMBED_DIM,
        bottleneck: int = 8,
        scale: float = 1.0,
        groups: int = _DEFAULT_GROUPS,
    ) -> None:
        """Initialises the RepAdapter with group-wise projection.

        Args:
            embed_dim: Input/output feature dimension D. Default: 768.
            bottleneck: Bottleneck dimension r. Search grid: {8, 16, 32}.
                Must be divisible by groups. All valid values {8, 16, 32}
                are divisible by the default groups=8.
            scale: Fixed scalar scale factor s.
                Search grid: {0.1, 0.5, 1.0, 5.0, 10.0}.
            groups: Number of groups G for group-wise projection. Default: 8
                (config.yaml: peft_methods.repadapter.groups: 8).
                Must divide both bottleneck and embed_dim.

        Raises: