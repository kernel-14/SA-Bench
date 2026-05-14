# models/router.py
"""Router-gating network for the MoE-POT architecture.

Implements the RouterGating class, which computes per-sample routing
distributions over the routed experts in each MoE layer. The router is
a CNN-based spatial feature extractor that collapses the token grid into
a single probability vector via global average pooling.

From the paper (Section 4):
    "These features are then passed to a router-gating network G^l(z_0^l(x)),
    which computes a vector of routing logits s^l(z_0^l(x)) ∈ R^{N_r},
    where N_r is the number of routed experts (e.g., N_r = 16). The gating
    weights are computed via a softmax function:
        w^l(z_0^l(x)) = Softmax(s^l(z_0^l(x))) ∈ R^{N_r}."

From Appendix B.2:
    "Both expert networks and router-gating networks are implemented using
    convolutional neural networks (CNNs) to preserve spatial information."

Architecture:
    Conv2d(embed_dim, embed_dim//2, kernel=3, padding=1) → GELU
    → Conv2d(embed_dim//2, embed_dim//4, kernel=3, padding=1) → GELU
    → AdaptiveAvgPool2d(1)
    → Flatten()
    → Linear(embed_dim//4, num_routed_experts)
    → (logits, Softmax(logits))

The full softmax over all num_routed_experts is always returned (not just
the top-K selected entries). This is required for:
  1. Load balancing loss: Importance_i = sum_b w_{i,b} over full distribution
  2. Interpretability: routing fingerprint Y_{ij} ∈ R^{16} per sample
     (Appendix B.4, dataset classification via cross-entropy distance)
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RouterGating(nn.Module):
    """CNN-based router-gating network for the MoE layer.

    Produces a per-sample probability distribution over num_routed_experts
    routed experts from a spatial feature map. The routing decision is
    global (one vector per sample, not per spatial location), achieved via
    AdaptiveAvgPool2d(1) which aggregates information from all spatial
    positions before the final linear projection.

    The router always returns the full softmax distribution over all
    num_routed_experts experts. Top-K selection and masking are handled
    externally by MoELayer._apply_topk_routing(), keeping the router's
    responsibility cleanly separated.

    The router is frozen during fine-tuning (MoEPOT.freeze_router() sets
    requires_grad=False on all router parameters), preserving the expert
    assignment strategy learned during pre-training while allowing expert
    networks to adapt to the target dataset.

    Attributes:
        embed_dim: Number of input channels. Equals attn_dim from config:
            512 (Tiny), 1024 (Small/Medium).
        num_routed_experts: Number of routed experts N_r. Default 16
            (config.yaml models.*.num_routed_experts).
        conv1: First CNN layer: embed_dim → embed_dim//2, kernel=3, padding=1.
            Preserves spatial dimensions H'×W' (typically 16×16).
        conv2: Second CNN layer: embed_dim//2 → embed_dim//4, kernel=3, padding=1.
            Preserves spatial dimensions H'×W'.
        pool: AdaptiveAvgPool2d(1) — collapses (H', W') → (1, 1) via global
            average pooling, producing one spatial summary per sample.
        fc: Linear(embed_dim//4, num_routed_experts) — final projection to
            routing logits space.
    """

    def __init__(
        self,
        embed_dim: int,
        num_routed_experts: int = 16,
    ) -> None:
        """Initializes the RouterGating network.

        Constructs the CNN reduction pipeline:
            embed_dim → embed_dim//2 → embed_dim//4 → num_routed_experts

        Uses PyTorch default weight initialization (Kaiming uniform for
        Conv2d, uniform for Linear). No special initialization is needed;
        the load balancing loss with w_bal=0.1 naturally drives the router
        toward uniform expert utilization during training.

        No BatchNorm or LayerNorm is used inside the router because:
          - Total batch size is 20 across 8 GPUs (~2-3 samples per GPU),
            making BatchNorm statistics unreliable.
          - The paper does not specify any normalization inside the router.

        Args:
            embed_dim: Number of input channels. Must be divisible by 4
                to allow the channel reduction embed_dim → embed_dim//2
                → embed_dim//4. Corresponds to attn_dim in config.yaml:
                  - Tiny:   512  (config.yaml models.tiny.attn_dim)
                  - Small:  1024 (config.yaml models.small.attn_dim)
                  - Medium: 1024 (config.yaml models.medium.attn_dim)
            num_routed_experts: Number of routed experts N_r. The router
                produces a probability distribution over this many experts.
                Default is 16 (config.yaml models.*.num_routed_experts).
                Ablation studies test N_r ∈ {8, 16, 32} (Table 4).

        Raises:
            ValueError: If embed_dim <= 0 or not divisible by 4.
            ValueError: If num_routed_experts <= 0.
        """
        super().__init__()

        # --- Input validation ---
        if embed_dim <= 0:
            raise ValueError(
                f"embed_dim must be positive, got {embed_dim}."
            )
        if embed_dim % 4 != 0:
            raise ValueError(
                f"embed_dim must be divisible by 4 for the channel reduction "
                f"embed_dim → embed_dim//2 → embed_dim//4. "
                f"Got embed_dim={embed_dim}."
            )
        if num_routed_experts <= 0:
            raise ValueError(
                f"num_routed_experts must be positive, got {num_routed_experts}."
            )

        self.embed_dim: int = embed_dim
        self.num_routed_experts: int = num_routed_experts

        # Intermediate channel sizes for the CNN reduction pipeline.
        mid_channels: int = embed_dim // 2    # e.g., 256 (Tiny), 512 (Small)
        small_channels: int = embed_dim // 4  # e.g., 128 (Tiny), 256 (Small)

        # --- Layer 1: First spatial convolution ---
        # embed_dim → embed_dim//2, kernel=3, padding=1 (same convolution).
        # Preserves spatial dimensions H'×W' = 16×16 after patchification.
        # Integrates local spatial context before global pooling.
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=mid_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        # --- Layer 2: Second spatial convolution ---
        # embed_dim//2 → embed_dim//4, kernel=3, padding=1 (same convolution).
        # Further compresses the representation while preserving H'×W'.
        self.conv2: nn.Conv2d = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=small_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        # --- Global average pooling ---
        # Collapses (H', W') → (1, 1), producing one spatial summary per sample.
        # This is the critical step that converts a spatial feature map into
        # a single per-sample routing vector. Global average pooling is chosen
        # because it aggregates information from all spatial locations equally,
        # reflecting the global character of the PDE field (the routing decision
        # should capture the overall PDE type, not a local patch feature).
        # AdaptiveAvgPool2d(1) handles any input spatial size gracefully.
        self.pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)

        # --- Final linear projection ---
        # embed_dim//4 → num_routed_experts.
        # Maps the compressed spatial summary to routing logits.
        # The linear layer has bias=True (default) to allow the router to
        # learn expert-specific baseline activation levels.
        self.fc: nn.Linear = nn.Linear(
            in_features=small_channels,
            out_features=num_routed_experts,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes routing logits and full softmax distribution for the input.

        Applies the CNN reduction pipeline to extract a global spatial
        summary, then projects to routing logits and normalizes via softmax.

        The full softmax over all num_routed_experts is returned (not just
        the top-K selected entries). This is required for:
          1. Load balancing loss computation in MoELayer:
             Importance_i = sum_b w_{i,b} over the full distribution.
          2. Interpretability analysis in InterpretabilityAnalyzer:
             The full 16-dimensional routing vector Y_{ij} is used as a
             fingerprint for dataset classification (Appendix B.4).

        Pipeline:
            (B, embed_dim, H', W')
            → conv1 → GELU → (B, embed_dim//2, H', W')
            → conv2 → GELU → (B, embed_dim//4, H', W')
            → pool  → (B, embed_dim//4, 1, 1)
            → flatten → (B, embed_dim//4)
            → fc → (B, num_routed_experts)   [logits]
            → softmax → (B, num_routed_experts)   [full_softmax]

        Args:
            x: Input feature map of shape (B, embed_dim, H', W') where:
                - B: Batch size (up to 20 for pre-training, config.yaml
                  pretraining.batch_size).
                - embed_dim: Must match self.embed_dim (attn_dim from config).
                - H': Token grid height, typically 16 (= 128 / patch_size=8,
                  config.yaml architecture.target_resolution=128,
                  architecture.patch_size=8).
                - W': Token grid width, typically 16.
                This is z_0^l(x) from the paper — the output of the Fourier
                layer passed to the MoE layer.

        Returns:
            A tuple (logits, full_softmax) where:
              - logits: Raw unnormalized routing scores of shape
                (B, num_routed_experts). These are the pre-softmax values
                s^l(z_0^l(x)) from the paper.
              - full_softmax: Normalized routing weights of shape
                (B, num_routed_experts). These are w^l(z_0^l(x)) =
                Softmax(s^l(z_0^l(x))) from the paper. Values are in
                (0, 1) and sum to 1.0 along dim=-1 for each sample.
        """
        # --- Step 1: First CNN layer with GELU activation ---
        # Input:  (B, embed_dim, H', W')
        # Output: (B, embed_dim//2, H', W')
        # Spatial dimensions H'×W' are preserved by padding=1.
        out: torch.Tensor = self.conv1(x)
        out = F.gelu(out)

        # --- Step 2: Second CNN layer with GELU activation ---
        # Input:  (B, embed_dim//2, H', W')
        # Output: (B, embed_dim//4, H', W')
        # Further channel compression while preserving spatial structure.
        out = self.conv2(out)
        out = F.gelu(out)

        # --- Step 3: Global average pooling ---
        # Input:  (B, embed_dim//4, H', W')
        # Output: (B, embed_dim//4, 1, 1)
        # Collapses all spatial positions into a single summary vector.
        # This produces one routing decision per sample in the batch,
        # not per spatial location — the routing is global.
        out = self.pool(out)

        # --- Step 4: Flatten spatial singleton dimensions ---
        # Input:  (B, embed_dim//4, 1, 1)
        # Output: (B, embed_dim//4)
        # start_dim=1 preserves the batch dimension.
        out = out.flatten(start_dim=1)

        # --- Step 5: Linear projection to routing logits ---
        # Input:  (B, embed_dim//4)
        # Output: (B, num_routed_experts)
        # These are the raw unnormalized routing scores s^l(z_0^l(x)).
        logits: torch.Tensor = self.fc(out)

        # --- Step 6: Softmax normalization over expert dimension ---
        # Input:  (B, num_routed_experts)
        # Output: (B, num_routed_experts)
        # dim=-1 normalizes over the expert axis (dim=1) independently
        # for each sample in the batch. Values are in (0, 1) and sum to
        # 1.0 per sample: w^l(z_0^l(x)) = Softmax(s^l(z_0^l(x))).
        # The full distribution (all 16 experts) is returned here.
        # Top-K masking is applied externally in MoELayer._apply_topk_routing().
        full_softmax: torch.Tensor = F.softmax(logits, dim=-1)

        return logits, full_softmax
