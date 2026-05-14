## agents/conv_lstm_cell.py
"""ConvLSTM cell building blocks for the DRC agent architecture.

This module implements the two core neural network modules used in the Deep
Repeated ConvLSTM (DRC) architecture described in Guez et al. (2019) and
analyzed in the paper "Interpreting Emergent Planning in Model-Free RL".

The two modules are:

1. **PoolAndInject**: A global pooling mechanism that allows spatial information
   to spread rapidly across the entire 8×8 grid within a single computational
   tick. Without this, a 3×3 kernel ConvLSTM can only propagate information one
   cell per tick — too slow for planning across an 8×8 board. By globally pooling
   the output state and broadcasting back, every position receives a summary of
   the entire cell state (paper Section E.3).

2. **ConvLSTMCell**: A modified LSTM cell operating on 3D spatial tensors
   (B, C, H, W) instead of 1D vectors. Incorporates the pool-and-inject
   mechanism and uses a single fused convolution for all four LSTM gates.
   The cell state `c` (not the output state `h`) is what linear probes operate
   on throughout the interpretability pipeline (paper Section 4.1).

Architecture reference (paper Section E.3):
    Pool-and-inject:
        m = [MeanPool_{H,W}(h), MaxPool_{H,W}(h)]^T  ∈ R^{2G}
        p̂ = W_p * m + b_p  ∈ R^{H*W*G}
        p = Reshape_{H×W×G}(p̂)  ∈ R^{H×W×G}

    ConvLSTM gates (fused):
        combined = cat([x, h, p], dim=1)  ∈ R^{(input_dim + 2G) × H × W}
        [i, f, g, o] = conv_gates(combined)  (each ∈ R^{G × H × W})
        c_new = sigmoid(f) * c + sigmoid(i) * tanh(g)
        h_new = sigmoid(o) * tanh(c_new)

Spatial dimension preservation:
    All convolutions use padding = kernel_size // 2 = 1 (for kernel_size=3),
    ensuring 8×8 inputs produce 8×8 outputs. This is the "single layer of
    input zero padding" from Section 2.3 and E.3, and is what enables the
    spatial bijection between cell state positions and Sokoban grid squares —
    the core assumption underlying the 1×1 probe design.

Input dimension accounting for DRCAgent:
    The input `x` passed to ConvLSTMCell.forward() is pre-concatenated by
    DRCAgent before calling this cell. For DRC(3,3) with hidden_dim=32:
    - Layer 1: x = cat([i_t, top_down_skip]) → input_dim = 32 + 32 = 64
    - Layers 2,3: x = cat([i_t, h_prev_layer]) → input_dim = 32 + 32 = 64
    So conv_gates input channels = input_dim + hidden_dim + hidden_dim
                                  = 64 + 32 + 32 = 128 for all layers.

Critical note on what gets probed:
    The paper consistently probes the CELL STATE `c` (denoted g_t^d in the
    paper), not the output state `h`. This module returns (h_new, c_new) so
    that DRCAgent can explicitly expose c_new for the probing pipeline.

Example:
    >>> import torch
    >>> pool_inject = PoolAndInject(hidden_dim=32, spatial_h=8, spatial_w=8)
    >>> h = torch.zeros(2, 32, 8, 8)
    >>> p = pool_inject(h)
    >>> p.shape
    torch.Size([2, 32, 8, 8])

    >>> cell = ConvLSTMCell(input_dim=64, hidden_dim=32, kernel_size=3,
    ...                     spatial_h=8, spatial_w=8)
    >>> x = torch.zeros(2, 64, 8, 8)
    >>> h = torch.zeros(2, 32, 8, 8)
    >>> c = torch.zeros(2, 32, 8, 8)
    >>> h_new, c_new = cell(x, h, c)
    >>> h_new.shape, c_new.shape
    (torch.Size([2, 32, 8, 8]), torch.Size([2, 32, 8, 8]))
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class PoolAndInject(nn.Module):
    """Global pooling mechanism for rapid spatial information propagation.

    Implements the pool-and-inject operation from paper Section E.3:

        m = [MeanPool_{H,W}(h), MaxPool_{H,W}(h)]^T  ∈ R^{2G}
        p̂ = W_p * m + b_p  ∈ R^{H*W*G}
        p = Reshape_{H×W×G}(p̂)  ∈ R^{H×W×G}

    The output `p` has the same spatial shape as the input `h`, enabling
    direct concatenation in the ConvLSTM gate computation. By broadcasting
    global statistics (mean and max) back to every spatial position, this
    mechanism allows information to propagate across the entire 8×8 grid
    in a single computational tick — critical for efficient planning.

    The pooling operates on the **output state** `h` (not the cell state `c`),
    matching the paper's formula which uses h_{t,n-1}^d.

    Attributes:
        hidden_dim: Number of channels G in the hidden state (default 32).
        spatial_h: Spatial height H of the hidden state (default 8).
        spatial_w: Spatial width W of the hidden state (default 8).
        fc: Linear layer mapping R^{2G} → R^{H*W*G}. This is a large
            projection (64 → 2048 for default params) that broadcasts
            global statistics to every spatial position.

    Example:
        >>> pool_inject = PoolAndInject(hidden_dim=32, spatial_h=8, spatial_w=8)
        >>> h = torch.randn(4, 32, 8, 8)  # batch=4, G=32, H=8, W=8
        >>> p = pool_inject(h)
        >>> p.shape
        torch.Size([4, 32, 8, 8])
        >>> # p has same shape as h, ready for concatenation in gate computation
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        spatial_h: int = 8,
        spatial_w: int = 8,
    ) -> None:
        """Initialize the PoolAndInject module.

        Args:
            hidden_dim: Number of channels G in the hidden state. Matches
                config.yaml agent.hidden_dim = 32.
            spatial_h: Spatial height H of the hidden state. Matches
                config.yaml agent.grid_h = 8.
            spatial_w: Spatial width W of the hidden state. Matches
                config.yaml agent.grid_w = 8.
        """
        super().__init__()

        self.hidden_dim: int = hidden_dim
        self.spatial_h: int = spatial_h
        self.spatial_w: int = spatial_w

        # Linear layer: R^{2G} → R^{H*W*G}
        # Input: concatenated mean-pool and max-pool vectors (2G = 64 for defaults)
        # Output: flattened spatial broadcast (H*W*G = 2048 for defaults)
        self.fc: nn.Linear = nn.Linear(
            in_features=2 * hidden_dim,
            out_features=spatial_h * spatial_w * hidden_dim,
            bias=True,
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Compute the pool-and-inject output from the hidden state.

        Performs global mean and max pooling over the spatial dimensions,
        concatenates the results, applies a linear projection, and reshapes
        back to the original spatial dimensions.

        Args:
            h: Output state tensor of shape (B, G, H, W) where:
                - B is the batch size
                - G = hidden_dim = 32 is the channel dimension
                - H = spatial_h = 8 is the spatial height
                - W = spatial_w = 8 is the spatial width
                This is the **output state** h_{t,n-1}^d, not the cell state.

        Returns:
            Pool-and-inject output tensor of shape (B, G, H, W), matching
            the shape of `h`. This can be directly concatenated with `h`
            and the input `x` in the ConvLSTM gate computation.

        Note:
            The output has the same shape as `h`, so it can be concatenated
            along the channel dimension (dim=1) without any reshaping.
        """
        batch_size: int = h.shape[0]

        # Step 1: Global mean pooling over spatial dimensions (H, W).
        # h shape: (B, G, H, W) → mean_pool shape: (B, G)
        mean_pool: torch.Tensor = torch.mean(h, dim=[2, 3])

        # Step 2: Global max pooling over spatial dimensions (H, W).
        # h shape: (B, G, H, W) → max_pool shape: (B, G)
        max_pool: torch.Tensor = torch.amax(h, dim=[2, 3])

        # Step 3: Concatenate mean and max pool vectors.
        # m shape: (B, 2G)
        m: torch.Tensor = torch.cat([mean_pool, max_pool], dim=1)

        # Step 4: Linear projection from R^{2G} to R^{H*W*G}.
        # p_flat shape: (B, H*W*G)
        p_flat: torch.Tensor = self.fc(m)

        # Step 5: Reshape to spatial tensor (B, G, H, W).
        # This broadcasts the global statistics to every spatial position.
        p: torch.Tensor = p_flat.view(
            batch_size, self.hidden_dim, self.spatial_h, self.spatial_w
        )

        return p


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell with pool-and-inject for the DRC architecture.

    Implements a modified ConvLSTM cell (Shi et al., 2015) with the pool-and-
    inject mechanism from Guez et al. (2019). This cell is the core recurrent
    unit of the DRC agent, processing 3D spatial tensors (B, C, H, W) instead
    of 1D vectors.

    The key modifications from a standard ConvLSTM are:
    1. **Pool-and-inject**: The output state `h` is globally pooled and
       broadcast back to every position before the gate computation, enabling
       rapid spatial information propagation (paper Section E.3).
    2. **Pre-concatenated input**: The input `x` already contains the bottom-up
       skip (encoded observation i_t) and top-down skip (previous final layer
       output) concatenated by DRCAgent. This cell just processes the combined
       input without needing to know the skip connection structure.

    The cell uses a single fused convolution for all four LSTM gates (i, f, g, o)
    for computational efficiency. The gate split order is [i, f, g, o] following
    the standard LSTM convention.

    Critical: This cell returns BOTH (h_new, c_new). The cell state `c_new`
    (denoted g_t^d in the paper) is what linear probes operate on throughout
    the interpretability pipeline. The output state `h_new` is used for
    skip connections and the final policy/value computation.

    Spatial dimension preservation:
        padding = kernel_size // 2 = 1 (for kernel_size=3) ensures that
        8×8 inputs produce 8×8 outputs, maintaining the spatial bijection
        between cell state positions and Sokoban grid squares.

    Attributes:
        input_dim: Channel dimension of the pre-concatenated input x.
            For DRC(3,3): input_dim = 64 (32 from i_t + 32 from skip).
        hidden_dim: Channel dimension G of the hidden states h and c.
            Matches config.yaml agent.hidden_dim = 32.
        kernel_size: Convolutional kernel size. Matches config.yaml
            agent.kernel_size = 3.
        spatial_h: Spatial height H. Matches config.yaml agent.grid_h = 8.
        spatial_w: Spatial width W. Matches config.yaml agent.grid_w = 8.
        pool_inject: PoolAndInject module for global information propagation.
        conv_gates: Fused convolution computing all 4 LSTM gates simultaneously.
            Input channels: input_dim + hidden_dim + hidden_dim
                          = input_dim + 2 * hidden_dim
            Output channels: 4 * hidden_dim (split into i, f, g, o gates)

    Example:
        >>> cell = ConvLSTMCell(input_dim=64, hidden_dim=32, kernel_size=3,
        ...                     spatial_h=8, spatial_w=8)
        >>> x = torch.zeros(2, 64, 8, 8)   # pre-concatenated input
        >>> h = torch.zeros(2, 32, 8, 8)   # previous output state
        >>> c = torch.zeros(2, 32, 8, 8)   # previous cell state
        >>> h_new, c_new = cell(x, h, c)
        >>> h_new.shape
        torch.Size([2, 32, 8, 8])
        >>> c_new.shape
        torch.Size([2, 32, 8, 8])
        >>> # c_new is the cell state g_t^d used by linear probes
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 32,
        kernel_size: int = 3,
        spatial_h: int = 8,
        spatial_w: int = 8,
    ) -> None:
        """Initialize the ConvLSTMCell.

        Args:
            input_dim: Channel dimension of the pre-concatenated input tensor
                `x` passed to forward(). For DRC(3,3) with hidden_dim=32:
                - Layer 1: input_dim = 64 (i_t[32] + top_down_skip[32])
                - Layers 2,3: input_dim = 64 (i_t[32] + h_prev_layer[32])
                The total gate convolution input channels will be
                input_dim + 2 * hidden_dim = 64 + 64 = 128.
            hidden_dim: Number of channels G in the hidden states h and c.
                Matches config.yaml agent.hidden_dim = 32.
            kernel_size: Convolutional kernel size for the gate convolution.
                Matches config.yaml agent.kernel_size = 3. The padding is
                automatically set to kernel_size // 2 to preserve spatial dims.
            spatial_h: Spatial height H of the hidden states. Matches
                config.yaml agent.grid_h = 8. Passed to PoolAndInject.
            spatial_w: Spatial width W of the hidden states. Matches
                config.yaml agent.grid_w = 8. Passed to PoolAndInject.
        """
        super().__init__()

        self.input_dim: int = input_dim
        self.hidden_dim: int = hidden_dim
        self.kernel_size: int = kernel_size
        self.spatial_h: int = spatial_h
        self.spatial_w: int = spatial_w

        # Pool-and-inject module: operates on the output state h.
        # Owned by this cell (instantiated here, not passed in).
        self.pool_inject: PoolAndInject = PoolAndInject(
            hidden_dim=hidden_dim,
            spatial_h=spatial_h,
            spatial_w=spatial_w,
        )

        # Fused gate convolution computing all 4 LSTM gates simultaneously.
        # Input channels breakdown:
        #   - input_dim: pre-concatenated input x (bottom-up + top-down skips)
        #   - hidden_dim: previous output state h
        #   - hidden_dim: pool-and-inject output p (same shape as h)
        # Output channels: 4 * hidden_dim (split into i, f, g, o gates)
        # Padding = kernel_size // 2 = 1 preserves 8×8 spatial dimensions.
        gate_in_channels: int = input_dim + hidden_dim + hidden_dim
        gate_out_channels: int = 4 * hidden_dim
        padding: int = kernel_size // 2

        self.conv_gates: nn.Conv2d = nn.Conv2d(
            in_channels=gate_in_channels,
            out_channels=gate_out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one ConvLSTM tick with pool-and-inject.

        Computes the new output state h_new and cell state c_new from the
        pre-concatenated input x, previous output state h, and previous
        cell state c.

        The computation follows the standard LSTM equations adapted for
        convolutional operations and augmented with pool-and-inject:

            p = pool_inject(h)                          # global broadcast
            combined = cat([x, h, p], dim=1)            # fuse all inputs
            [i_gate, f_gate, g_gate, o_gate] = conv_gates(combined).chunk(4)
            c_new = sigmoid(f_gate) * c + sigmoid(i_gate) * tanh(g_gate)
            h_new = sigmoid(o_gate) * tanh(c_new)

        Args:
            x: Pre-concatenated input tensor of shape (B, input_dim, H, W).
                This is constructed by DRCAgent before calling this method.
                For DRC(3,3): x = cat([i_t, skip], dim=1) where skip is
                either the top-down skip (layer 1) or the previous layer's
                output h (layers 2, 3). input_dim = 64 for all layers.
            h: Previous output state tensor of shape (B, hidden_dim, H, W).
                This is h_{t,n-1}^d from the paper. Used in gate computation
                and as input to pool_inject.
            c: Previous cell state tensor of shape (B, hidden_dim, H, W).
                This is g_{t,n-1}^d from the paper (using the paper's notation
                where g denotes the cell state). Updated via the forget and
                input gates.

        Returns:
            Tuple (h_new, c_new) where:
            - h_new: New output state of shape (B, hidden_dim, H, W).
              Used for skip connections (top-down skip to layer 1, bottom-up
              to next layer) and for the final policy/value computation.
            - c_new: New cell state of shape (B, hidden_dim, H, W).
              This is g_t^d in the paper's notation. **This is what linear
              probes operate on** — DRCAgent must explicitly return c_new
              from each layer at each tick for the probing pipeline.

        Note:
            The gate split order [i, f, g, o] follows the standard LSTM
            convention. Deviating from this order would produce a broken LSTM
            (e.g., applying sigmoid to the cell gate instead of tanh).
        """
        # Step 1: Compute pool-and-inject from the previous output state h.
        # p shape: (B, hidden_dim, H, W) — same as h, ready for concatenation.
        p: torch.Tensor = self.pool_inject(h)

        # Step 2: Concatenate all inputs along the channel dimension.
        # combined shape: (B, input_dim + hidden_dim + hidden_dim, H, W)
        #               = (B, 64 + 32 + 32, 8, 8) = (B, 128, 8, 8) for defaults
        combined: torch.Tensor = torch.cat([x, h, p], dim=1)

        # Step 3: Apply fused gate convolution.
        # gates shape: (B, 4 * hidden_dim, H, W) = (B, 128, 8, 8) for defaults
        gates: torch.Tensor = self.conv_gates(combined)

        # Step 4: Split into 4 gates along the channel dimension.
        # Each gate shape: (B, hidden_dim, H, W) = (B, 32, 8, 8) for defaults
        # Split order: [i_gate, f_gate, g_gate, o_gate] — standard LSTM convention.
        i_gate: torch.Tensor
        f_gate: torch.Tensor
        g_gate: torch.Tensor
        o_gate: torch.Tensor
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)

        # Step 5: Apply gate activations.
        # Input gate: sigmoid (controls how much new information to store)
        i_act: torch.Tensor = torch.sigmoid(i_gate)
        # Forget gate: sigmoid (controls how much old cell state to retain)
        f_act: torch.Tensor = torch.sigmoid(f_gate)
        # Cell gate: tanh (new candidate cell state values)
        g_act: torch.Tensor = torch.tanh(g_gate)
        # Output gate: sigmoid (controls what to output from cell state)
        o_act: torch.Tensor = torch.sigmoid(o_gate)

        # Step 6: Update cell state.
        # c_new = forget_gate * c + input_gate * cell_gate
        # This is g_t^d in the paper's notation — what probes operate on.
        c_new: torch.Tensor = f_act * c + i_act * g_act

        # Step 7: Compute new output state.
        # h_new = output_gate * tanh(c_new)
        h_new: torch.Tensor = o_act * torch.tanh(c_new)

        return h_new, c_new
