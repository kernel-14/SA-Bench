# models/expert.py
"""Expert CNN subnetwork for the MoE-POT architecture.

Implements the ExpertCNN class, which serves as the atomic building block
for both shared experts (always active) and routed experts (dynamically
selected via top-K routing) within the MoE layer.

From the paper (Section 4):
    "Each expert E_i^{l(s)} or E_j^{l(r)} is a convolutional subnetwork
    that takes z_0^l(x) as input and maps it to an output feature map of
    the same shape."

From Appendix B.2:
    "Both expert networks and router-gating networks are implemented using
    convolutional neural networks (CNNs) to preserve spatial information."

Architecture:
    Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        → GELU()
        → Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)

The two-layer CNN with 3×3 "same" convolutions preserves the spatial
dimensions of the PDE token grid (16×16 after patchification with P=8
on 128×128 input). No normalization layers are used to avoid instability
with small per-expert effective batch sizes during top-K routing.
"""

import torch
import torch.nn as nn


class ExpertCNN(nn.Module):
    """Two-layer CNN expert subnetwork for the MoE layer.

    Implements a spatially-preserving feature transformation used as both
    shared experts (N_s=2, always active) and routed experts (N_r=16,
    top-K=4 activated per input) in each MoE block.

    The expert maps an input feature map to an output of identical shape,
    enabling direct weighted aggregation in the MoE output formula:

        z^{l+1}(x) = (1/N_s) * Σ_i E_i^{l(s)}(z_0^l(x))
                   + Σ_k w_k^l * E_{i_k}^{l(r)}(z_0^l(x))

    Design choices:
      - kernel_size=3, padding=1: "Same" convolution preserving H'×W'.
      - GELU activation: Consistent with the paper's σ(·) notation and
        the transformer convention used throughout the architecture.
      - No BatchNorm/LayerNorm: Avoids instability with small per-expert
        effective batch sizes (batch_size=20 total, ~5 samples per routed
        expert on average with top-K=4 out of 16).
      - No internal residual: The skip connection is applied at the
        MoEBlock level after the full MoE layer output, not inside each
        individual expert.
      - Named attributes (not nn.Sequential): Required by the interface
        specification for direct attribute access (conv1, conv2, activation).

    Attributes:
        in_channels: Number of input (and output) channels. Equals
            embed_dim (attn_dim from config): 512 for Tiny, 1024 for
            Small and Medium.
        hidden_channels: Number of intermediate channels in the bottleneck.
            Equals mlp_dim from config: 512 for Tiny, 1024 for Small,
            2048 for Medium.
        conv1: First convolutional layer mapping in_channels → hidden_channels.
            Conv2d with kernel_size=3, padding=1 (same convolution).
        conv2: Second convolutional layer mapping hidden_channels → in_channels.
            Conv2d with kernel_size=3, padding=1 (same convolution).
        activation: GELU non-linearity applied between conv1 and conv2.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
    ) -> None:
        """Initializes the ExpertCNN with two convolutional layers and GELU.

        Uses PyTorch default weight initialization (Kaiming uniform for
        Conv2d), which is appropriate for GELU-activated networks and
        consistent with the paper's lack of custom initialization details.

        Args:
            in_channels: Number of input channels C. Also the number of
                output channels, ensuring shape preservation. Corresponds
                to embed_dim (attn_dim) in the model config:
                  - Tiny:   512  (config.yaml models.tiny.attn_dim)
                  - Small:  1024 (config.yaml models.small.attn_dim)
                  - Medium: 1024 (config.yaml models.medium.attn_dim)
            hidden_channels: Number of intermediate channels in the hidden
                layer. Corresponds to mlp_dim in the model config:
                  - Tiny:   512  (config.yaml models.tiny.mlp_dim)
                  - Small:  1024 (config.yaml models.small.mlp_dim)
                  - Medium: 2048 (config.yaml models.medium.mlp_dim)

        Raises:
            ValueError: If in_channels <= 0 or hidden_channels <= 0.
        """
        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                f"in_channels must be positive, got {in_channels}."
            )
        if hidden_channels <= 0:
            raise ValueError(
                f"hidden_channels must be positive, got {hidden_channels}."
            )

        self.in_channels: int = in_channels
        self.hidden_channels: int = hidden_channels

        # First convolutional layer: in_channels → hidden_channels.
        # kernel_size=3, padding=1 implements a "same" convolution:
        # output spatial size equals input spatial size for any (H', W').
        # This is critical for the shape-preserving contract of the expert.
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        # GELU activation: σ(·) in the paper's notation.
        # GELU is the standard choice in transformer-based architectures
        # (ViT, DPOT) and is consistent with the Fourier layer's activation.
        self.activation: nn.GELU = nn.GELU()

        # Second convolutional layer: hidden_channels → in_channels.
        # Restores the channel dimension to in_channels, ensuring the
        # output shape (B, in_channels, H', W') matches the input shape.
        self.conv2: nn.Conv2d = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the two-layer CNN transformation to the input feature map.

        Computes:
            out = conv2(GELU(conv1(x)))

        The spatial dimensions H' and W' are preserved throughout by the
        "same" convolution (kernel_size=3, padding=1). The channel dimension
        returns to in_channels after conv2.

        Args:
            x: Input feature map of shape (B, in_channels, H', W') where:
                - B: Batch size (up to batch_size=20 for pre-training).
                - in_channels: Must match self.in_channels (embed_dim).
                - H': Token grid height, typically 16 (= 128 / patch_size=8).
                - W': Token grid width, typically 16 (= 128 / patch_size=8).

        Returns:
            Output feature map of shape (B, in_channels, H', W').
            Identical shape to the input, satisfying the shape-preserving
            contract required by the MoE aggregation formula.
        """
        # Step 1: First convolution — expand to hidden dimension.
        # Input:  (B, in_channels, H', W')
        # Output: (B, hidden_channels, H', W')
        out: torch.Tensor = self.conv1(x)

        # Step 2: Non-linear activation.
        # Shape unchanged: (B, hidden_channels, H', W')
        out = self.activation(out)

        # Step 3: Second convolution — project back to input dimension.
        # Input:  (B, hidden_channels, H', W')
        # Output: (B, in_channels, H', W')
        out = self.conv2(out)

        return out
