# drc_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np


class ConvLSTMCell(nn.Module):
    """
    A single ConvLSTM cell used in the DRCModel's ConvLSTM layers.
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int, padding: int):
        """
        Initializes a ConvLSTM cell.

        Args:
            input_dim (int): Number of channels in the input.
            hidden_dim (int): Number of hidden channels in the ConvLSTM.
            kernel_size (int): Size of the convolutional kernel.
            padding (int): Padding for the convolutional operations.
        """
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim

        # Define gates: input, forget, output, candidate
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the ConvLSTM cell.

        Args:
            x (torch.Tensor): Input tensor at the current step.
            h (torch.Tensor): Hidden state from the previous time step.
            c (torch.Tensor): Cell state from the previous time step.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated hidden state and cell state.
        """
        combined = torch.cat([x, h], dim=1)  # Concatenate input and hidden state
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_output, chunks=4, dim=1)  # Split into gates

        # Compute LSTM gates
        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        o = torch.sigmoid(cc_o)  # Output gate
        g = torch.tanh(cc_g)     # Candidate cell state

        # Update cell state and hidden state
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size: int, spatial_dims: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initializes the hidden and cell states with zeros.

        Args:
            batch_size (int): Batch size.
            spatial_dims (Tuple[int, int]): Input spatial dimensions (H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Initialized hidden state and cell state.
        """
        height, width = spatial_dims
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        return h, c


class DRCModel(nn.Module):
    """
    Implementation of the Deep Repeated ConvLSTM (DRC) model.
    Processes Sokoban input and outputs policy logits and value estimates.
    """

    def __init__(self, config: dict):
        """
        Initializes the DRCModel.

        Args:
            config (dict): Configuration dictionary specifying DRC parameters.
        """
        super(DRCModel, self).__init__()
        self.config = config

        # Parse configuration
        self.convlstm_layers = config["agent"]["convlstm_layers"]
        self.recurrent_ticks = config["agent"]["recurrent_ticks"]
        self.hidden_dim = config["agent"]["convlstm_channels"]
        self.kernel_size = config["agent"]["kernel_size"]
        self.padding = config["agent"]["padding"]
        self.input_dim = 7  # Sokoban symbolic input (8x8x7)

        # Convolutional Encoder
        self.encoder = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding
        )

        # ConvLSTM Layers
        self.convlstm_layers_list = nn.ModuleList([
            ConvLSTMCell(
                input_dim=self.hidden_dim if i > 0 else self.hidden_dim,
                hidden_dim=self.hidden_dim,
                kernel_size=self.kernel_size,
                padding=self.padding
            ) for i in range(self.convlstm_layers)
        ])

        # Policy and Value heads
        self.policy_head = nn.Linear(self.hidden_dim * 8 * 8, 5)  # Discrete Sokoban actions
        self.value_head = nn.Linear(self.hidden_dim * 8 * 8, 1)   # Scalar value estimate

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to process input and predict policy logits and value estimate.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 7, 8, 8).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Policy logits (B, 5) and state value (B, 1).
        """
        batch_size, _, height, width = x.size()
        device = x.device

        # Encode the input
        x_encoded = self.encoder(x)

        # Initialize hidden and cell states for ConvLSTM layers
        hidden_states = []
        cell_states = []
        for layer in self.convlstm_layers_list:
            h, c = layer.init_hidden(batch_size, (height, width))
            hidden_states.append(h)
            cell_states.append(c)

        # Perform recurrent ticks
        for tick in range(self.recurrent_ticks):
            for l in range(self.convlstm_layers):
                input_tensor = x_encoded if l == 0 else hidden_states[l - 1]
                h, c = self.convlstm_layers_list[l](input_tensor, hidden_states[l], cell_states[l])
                hidden_states[l], cell_states[l] = h, c

        # Use the final hidden state from the last ConvLSTM layer
        final_hidden_state = hidden_states[-1]  # Shape: (B, hidden_dim, 8, 8)
        flattened_state = final_hidden_state.view(batch_size, -1)  # Flatten spatial dimensions

        # Pass through policy head and value head
        policy_logits = self.policy_head(flattened_state)  # Shape: (B, 5)
        state_value = self.value_head(flattened_state)     # Shape: (B, 1)

        return policy_logits, state_value

    def get_hidden_states(self) -> List[torch.Tensor]:
        """
        Returns the hidden states for probing.

        Returns:
            List[torch.Tensor]: List of hidden states from all ConvLSTM layers.
        """
        return self.hidden_states if hasattr(self, 'hidden_states') else []
