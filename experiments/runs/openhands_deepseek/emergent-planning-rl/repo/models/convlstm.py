"""
ConvLSTM cell and layer implementations based on Shi et al. (2015).

Used in Deep Repeated ConvLSTM (DRC) agents as described in
Guez et al. (2019), "An Investigation of Model-Free Planning".

The ConvLSTM uses 3D hidden states (H, W, C) and convolutional connections
with kernel size 3 and input zero padding to preserve spatial dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ConvLSTMCell(nn.Module):
    """
    A single ConvLSTM cell with convolutional input-to-hidden and
    hidden-to-hidden transitions.

    For each input x_t and previous state (h_{t-1}, c_{t-1}):
        i_t = sigmoid(conv_i(x_t) + conv_hi(h_{t-1}) + bias_i)
        f_t = sigmoid(conv_f(x_t) + conv_hf(h_{t-1}) + bias_f)
        o_t = sigmoid(conv_o(x_t) + conv_ho(h_{t-1}) + bias_o)
        g_t = tanh(conv_g(x_t) + conv_hg(h_{t-1}) + bias_g)
        c_t = f_t * c_{t-1} + i_t * g_t
        h_t = o_t * tanh(c_t)

    All convolutions use kernel_size=3, padding=1 to preserve spatial dims,
    and output `hidden_channels` channels.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        # Input-to-hidden convolutions (for i, f, o, g)
        self.conv_x = nn.Conv2d(
            input_channels, hidden_channels * 4,
            kernel_size=kernel_size, padding=padding, bias=False
        )
        # Hidden-to-hidden convolutions (for i, f, o, g)
        self.conv_h = nn.Conv2d(
            hidden_channels, hidden_channels * 4,
            kernel_size=kernel_size, padding=padding, bias=True
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: input tensor of shape (B, input_channels, H, W)
            state: previous hidden and cell states, each (B, hidden_channels, H, W)

        Returns:
            h: output hidden state (B, hidden_channels, H, W)
            (h, c): new hidden and cell states
        """
        B, _, H, W = x.shape

        if state is None:
            h_prev = torch.zeros(B, self.hidden_channels, H, W,
                                 device=x.device, dtype=x.dtype)
            c_prev = torch.zeros(B, self.hidden_channels, H, W,
                                 device=x.device, dtype=x.dtype)
        else:
            h_prev, c_prev = state

        # Input-to-hidden
        x_gates = self.conv_x(x)  # (B, 4*hidden_channels, H, W)
        # Hidden-to-hidden
        h_gates = self.conv_h(h_prev)  # (B, 4*hidden_channels, H, W)

        gates = x_gates + h_gates
        i_gate, f_gate, o_gate, g_gate = gates.chunk(4, dim=1)

        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        o = torch.sigmoid(o_gate)
        g = torch.tanh(g_gate)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, (h, c)


class ConvLSTMLayer(nn.Module):
    """
    A ConvLSTM layer wrapping a single ConvLSTMCell.

    This layer receives multiple inputs (from bottom-up skip connections)
    and optionally includes pool-and-inject.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        pool_and_inject: bool = False,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.pool_and_inject = pool_and_inject

        self.cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size, padding)

        if pool_and_inject:
            # Mean and max pooling spatially, then affine transform back
            self.pool_proj = nn.Linear(2 * hidden_channels, hidden_channels * 8 * 8)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: input tensor (B, input_channels, H, W)
            state: previous (h, c) states

        Returns:
            h: output (B, hidden_channels, H, W)
            (h, c): new states
        """
        B, C, H, W = x.shape

        if state is not None:
            h_prev, c_prev = state
        else:
            h_prev = torch.zeros(B, self.hidden_channels, H, W,
                                 device=x.device, dtype=x.dtype)
            c_prev = torch.zeros(B, self.hidden_channels, H, W,
                                 device=x.device, dtype=x.dtype)

        if self.pool_and_inject:
            # Pool-and-inject: provide spatially-pooled version of prev output
            h_mean = h_prev.mean(dim=[2, 3])  # (B, hidden_channels)
            h_max = h_prev.amax(dim=[2, 3])    # (B, hidden_channels)
            pooled = torch.cat([h_mean, h_max], dim=1)  # (B, 2*hidden_channels)
            injection = self.pool_proj(pooled)  # (B, hidden_channels*H*W)
            injection = injection.view(B, self.hidden_channels, H, W)
            h_prev = h_prev + injection

        h, new_state = self.cell(x, (h_prev, c_prev))
        return h, new_state
