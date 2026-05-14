import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ConvLSTMCell(nn.Module):
    """
    ConvLSTM cell as used in the DRC architecture (Guez et al., 2019).
    
    Implements a standard ConvLSTM with convolutional gates operating on
    3D (H x W x C) hidden states. The input to each cell includes:
      - The bottom-up encoding i_t (broadcast to all layers)
      - The top-down skip connection from the final layer's previous tick output
      - The pool-and-inject signal from this cell's own previous output
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        height: int = 8,
        width: int = 8,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.height = height
        self.width = width

        # Gates: input, forget, output, cell — all computed jointly
        # Input to gates: [input, hidden, pool_inject, top_down_skip]
        # We concatenate all inputs along channel dim before the conv
        gate_input_channels = input_channels + hidden_channels

        self.conv_gates = nn.Conv2d(
            gate_input_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        # Pool-and-inject: mean+max pool -> affine -> reshape
        self.pool_inject_linear = nn.Linear(
            2 * hidden_channels,
            height * width * hidden_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: combined input (B, C_in, H, W) — includes encoding + skip connections
            h: previous hidden state (B, C_h, H, W)
            c: previous cell state (B, C_h, H, W)
        Returns:
            h_new: (B, C_h, H, W)
            c_new: (B, C_h, H, W)
        """
        B, _, H, W = h.shape

        # Pool-and-inject
        mean_pool = h.mean(dim=(2, 3))  # (B, C_h)
        max_pool = h.amax(dim=(2, 3))   # (B, C_h)
        pooled = torch.cat([mean_pool, max_pool], dim=1)  # (B, 2*C_h)
        p = self.pool_inject_linear(pooled)               # (B, H*W*C_h)
        p = p.view(B, self.hidden_channels, H, W)         # (B, C_h, H, W)

        combined = torch.cat([x, h + p], dim=1)
        gates = self.conv_gates(combined)

        i_gate, f_gate, o_gate, g_gate = gates.chunk(4, dim=1)
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        o_gate = torch.sigmoid(o_gate)
        g_gate = torch.tanh(g_gate)

        c_new = f_gate * c + i_gate * g_gate
        h_new = o_gate * torch.tanh(c_new)

        return h_new, c_new


class DRCEncoder(nn.Module):
    """
    Convolutional encoder that maps observation x_t to encoding i_t.
    Produces output with same spatial dimensions as input (8x8).
    """

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 32,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W) observation
        Returns:
            (B, C_out, H, W) encoding
        """
        return F.relu(self.conv(x))


class DRCStack(nn.Module):
    """
    Stack of D ConvLSTM layers with DRC-specific modifications:
      - Bottom-up skip: encoding i_t fed to all layers
      - Top-down skip: final layer output h^D_{t,n-1} fed to layer 1 at tick n
      - Pool-and-inject: spatial pooling of each layer's own hidden state
    
    Performs N ticks of recurrent computation per environment time step.
    """

    def __init__(
        self,
        num_layers: int = 3,
        num_ticks: int = 3,
        encoding_channels: int = 32,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        padding: int = 1,
        height: int = 8,
        width: int = 8,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_ticks = num_ticks
        self.hidden_channels = hidden_channels
        self.height = height
        self.width = width

        # Layer 1 receives: encoding + top-down skip from final layer
        # Layers 2..D receive: encoding only (bottom-up)
        # All layers also receive pool-and-inject from their own h (handled in cell)
        layer1_input_channels = encoding_channels + hidden_channels
        other_input_channels = encoding_channels

        self.cells = nn.ModuleList()
        for d in range(num_layers):
            in_ch = layer1_input_channels if d == 0 else other_input_channels
            self.cells.append(
                ConvLSTMCell(
                    input_channels=in_ch,
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    height=height,
                    width=width,
                )
            )

    def forward(
        self,
        encoding: torch.Tensor,
        hidden_states: Optional[list] = None,
        cell_states: Optional[list] = None,
        return_all_ticks: bool = False,
    ) -> Tuple[list, list, Optional[list]]:
        """
        Args:
            encoding: (B, C_enc, H, W)
            hidden_states: list of D tensors (B, C_h, H, W), or None for zeros
            cell_states: list of D tensors (B, C_h, H, W), or None for zeros
            return_all_ticks: if True, return cell states at every tick
        Returns:
            hidden_states: list of D tensors (B, C_h, H, W) after N ticks
            cell_states: list of D tensors (B, C_h, H, W) after N ticks
            all_tick_cell_states: list of N lists of D tensors, or None
        """
        B = encoding.shape[0]
        device = encoding.device

        if hidden_states is None:
            hidden_states = [
                torch.zeros(B, self.hidden_channels, self.height, self.width, device=device)
                for _ in range(self.num_layers)
            ]
        if cell_states is None:
            cell_states = [
                torch.zeros(B, self.hidden_channels, self.height, self.width, device=device)
                for _ in range(self.num_layers)
            ]

        all_tick_cell_states = [] if return_all_ticks else None

        for n in range(self.num_ticks):
            new_hidden = []
            new_cell = []

            # Top-down skip: final layer's output from previous tick
            top_down = hidden_states[-1]

            for d in range(self.num_layers):
                if d == 0:
                    x = torch.cat([encoding, top_down], dim=1)
                else:
                    x = encoding

                h_new, c_new = self.cells[d](x, hidden_states[d], cell_states[d])
                new_hidden.append(h_new)
                new_cell.append(c_new)

            hidden_states = new_hidden
            cell_states = new_cell

            if return_all_ticks:
                all_tick_cell_states.append([c.detach() for c in cell_states])

        return hidden_states, cell_states, all_tick_cell_states

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[list, list]:
        h = [
            torch.zeros(batch_size, self.hidden_channels, self.height, self.width, device=device)
            for _ in range(self.num_layers)
        ]
        c = [
            torch.zeros(batch_size, self.hidden_channels, self.height, self.width, device=device)
            for _ in range(self.num_layers)
        ]
        return h, c
