"""
Deep Repeated ConvLSTM (DRC) Agent Architecture.

Based on Guez et al. (2019), "An Investigation of Model-Free Planning".

The DRC agent processes Sokoban observations through:
1. A convolutional encoder
2. A stack of ConvLSTM layers with internal ticks
3. Policy and value heads

Key features:
- Bottom-up skip connections: input encoding provided to all layers
- Top-down skip connections: final layer output fed back to first layer next tick
- Pool-and-inject: spatially pooled previous output injected into cell
- Multiple internal ticks (N) per environment step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

from .convlstm import ConvLSTMLayer


class DRCNet(nn.Module):
    """
    Deep Repeated ConvLSTM network.

    Architecture:
        Encoder: Conv2d(7 -> 32, kernel=3, padding=1) + ReLU
        Stack of D ConvLSTM layers, each with hidden_channels=32.
        Each layer performs N internal ticks per environment step.

    Input: observation (B, 7, 8, 8)
    Output: policy logits (B, 5), value (B,)
    """

    def __init__(
        self,
        input_channels: int = 7,
        hidden_channels: int = 32,
        num_layers: int = 3,      # D
        num_ticks: int = 3,       # N
        num_actions: int = 5,
        kernel_size: int = 3,
        padding: int = 1,
        grid_size: int = 8,
        bottom_up_skip: bool = True,
        top_down_skip: bool = True,
        pool_and_inject: bool = True,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_ticks = num_ticks
        self.num_actions = num_actions
        self.grid_size = grid_size
        self.bottom_up_skip = bottom_up_skip
        self.top_down_skip = top_down_skip
        self.pool_and_inject = pool_and_inject

        # Encoder: Conv2d + ReLU, maintaining spatial dimensions
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size,
                      padding=padding),
            nn.ReLU(inplace=True),
        )

        # ConvLSTM layers
        self.layers = nn.ModuleList()
        for d in range(num_layers):
            # Each layer receives input_channels from bottom-up skip
            # plus hidden_channels from previous layer
            # Layer 0 also receives hidden_channels from top-down skip
            in_ch = 0
            if d == 0:
                in_ch += hidden_channels  # from encoder (bottom-up)
                if top_down_skip:
                    in_ch += hidden_channels  # from top-down
            else:
                in_ch += hidden_channels  # from encoder (bottom-up)
                in_ch += hidden_channels  # from previous layer

            self.layers.append(
                ConvLSTMLayer(
                    input_channels=in_ch,
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    pool_and_inject=pool_and_inject,
                )
            )

        # Policy head: affine transformation on concatenated [output, encoding]
        policy_input_dim = hidden_channels * grid_size * grid_size + hidden_channels * grid_size * grid_size
        self.policy_fc = nn.Sequential(
            nn.Linear(policy_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_actions),
        )

        # Value head
        self.value_fc = nn.Sequential(
            nn.Linear(policy_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def _init_states(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> List[Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Initialize hidden states for all layers to None."""
        return [None for _ in range(self.num_layers)]

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass for a single environment step with N internal ticks.

        Args:
            x: observation (B, 7, 8, 8)
            states: list of (h, c) tuples for each layer, or None

        Returns:
            policy_logits: (B, num_actions)
            value: (B, 1)
            new_states: list of (h, c) tuples
        """
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        if states is None:
            states = self._init_states(B, device, dtype)
        # Pad states to have num_layers entries
        while len(states) < self.num_layers:
            states.append(None)

        # Encode observation
        encoding = self.encoder(x)  # (B, hidden_channels, 8, 8)

        # Process through N internal ticks
        for tick in range(self.num_ticks):
            # For top-down skip: get output of last layer from previous tick
            top_down = None
            if self.top_down_skip and tick > 0:
                # output of last layer from previous tick
                # This is stored in the hidden state h of the last layer
                if states[self.num_layers - 1] is not None:
                    top_down = states[self.num_layers - 1][0]  # h state of last layer

            prev_output = None
            for d in range(self.num_layers):
                # Build input to this layer
                layer_inputs = []

                # Bottom-up skip: always provide encoder output
                if self.bottom_up_skip:
                    layer_inputs.append(encoding)

                if d == 0:
                    # First layer: also gets top-down skip if available
                    if self.top_down_skip and top_down is not None:
                        layer_inputs.append(top_down)
                else:
                    # Subsequent layers: get output from previous layer
                    if prev_output is not None:
                        layer_inputs.append(prev_output)

                inp = torch.cat(layer_inputs, dim=1)  # (B, in_channels, 8, 8)

                h, new_state = self.layers[d](inp, states[d])
                prev_output = h
                states[d] = new_state

        # Final layer output
        final_h = states[self.num_layers - 1][0]  # (B, hidden_channels, 8, 8)
        final_h_flat = final_h.reshape(B, -1)  # (B, hidden_channels * 8 * 8)
        encoding_flat = encoding.reshape(B, -1)  # (B, hidden_channels * 8 * 8)

        # Concatenate final output with encoding
        combined = torch.cat([final_h_flat, encoding_flat], dim=1)

        # Policy and value heads
        policy_logits = self.policy_fc(combined)  # (B, num_actions)
        value = self.value_fc(combined)  # (B, 1)

        return policy_logits, value, states

    def get_cell_states(
        self,
        states: List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
    ) -> List[Optional[torch.Tensor]]:
        """Extract cell states from the layer states."""
        return [s[1] if s is not None else None for s in states]
