## model.py
"""
Deep Repeated ConvLSTM (DRC) Agent Architecture.

Implements the DRC(3,3) model from the paper
"Interpreting Emergent Planning in Model‑Free Reinforcement Learning".

The architecture consists of:
- A convolutional encoder (7→32 channels, 3×3 kernel, padding=1).
- D stacked ConvLSTM blocks, each with skip connections and pool‑and‑inject.
- A post‑processing MLP and separate policy/value heads.

The network can be unrolled for N internal ticks per environment step.
All hidden/cell states have the same spatial dimensions (8×8) as the input grid.

Typical usage:
    config = Config.from_yaml('config.yaml')
    model = DRCNetwork(config.agent)
    state = model.initial_state(batch_size)
    logits, value, new_state = model(obs, state, num_ticks=3)
"""

from typing import List, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper module: Pool‑and‑Inject
# ---------------------------------------------------------------------------
class PoolInject(nn.Module):
    """
    Pool‑and‑Inject module as described in Appendix E.3.
    Computes mean and max spatial pooling of the previous hidden state,
    passes the concatenated vector through a linear layer, and reshapes
    back to the spatial feature map shape.
    """
    def __init__(self, channels: int, spatial_size: Tuple[int, int]):
        super().__init__()
        self.channels = channels
        self.spatial_size = spatial_size
        self.linear = nn.Linear(
            2 * channels,
            channels * spatial_size[0] * spatial_size[1]
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B = h.size(0)
        # h shape: (B, C, H, W)
        mean_pool = h.mean(dim=[-2, -1])   # (B, C)
        max_pool  = h.amax(dim=[-2, -1])   # (B, C)
        concat = torch.cat([mean_pool, max_pool], dim=-1)   # (B, 2C)
        out = self.linear(concat)                          # (B, C*H*W)
        return out.view(B, self.channels, *self.spatial_size)


# ---------------------------------------------------------------------------
# Convolutional LSTM cell
# ---------------------------------------------------------------------------
class ConvLSTMCell(nn.Module):
    """
    A single ConvLSTM cell operating on 3D feature maps.
    Implements standard LSTM gating with 2D convolutions.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int,
        padding: int,
        bias: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # One convolution to compute all four gates at once
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x   : (B, input_dim, H, W)
            h   : (B, hidden_dim, H, W)  – previous hidden state
            c   : (B, hidden_dim, H, W)  – previous cell state
        Returns:
            h_new : (B, hidden_dim, H, W)
            c_new : (B, hidden_dim, H, W)
        """
        combined = torch.cat([x, h], dim=1)          # (B, input_dim+hidden_dim, H, W)
        gates = self.conv(combined)                   # (B, 4*hidden_dim, H, W)
        i, f, g, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, c_new


# ---------------------------------------------------------------------------
# One DRC block (ConvLSTM + skips + pool‑and‑inject)
# ---------------------------------------------------------------------------
class DRCBlock(nn.Module):
    """
    A single layer of the Deep Repeated ConvLSTM stack.
    Incorporates:
      - bottom‑up skip connection (i_t is always concatenated with input x)
      - top‑down skip connection (optional, only for bottom layer after tick 0)
      - pool‑and‑inject on the previous hidden state.
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        padding: int,
        use_pool_inject: bool = True,
        spatial_size: Tuple[int, int] = (8, 8)
    ):
        super().__init__()
        self.channels = channels
        self.spatial_size = spatial_size

        # ConvLSTM cell: input to the cell is concatenation of x (after possible
        # top‑down addition) and the bottom‑up encoding i_t.
        self.conv_lstm = ConvLSTMCell(
            input_dim=2 * channels,
            hidden_dim=channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

        # Top‑down projection (1×1 conv used only when top_down is not None)
        self.top_down_proj = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            bias=False
        )

        # Pool‑and‑inject module (optional)
        if use_pool_inject:
            self.pool_inject = PoolInject(channels, spatial_size)
        else:
            self.pool_inject = None

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        bottom_up: torch.Tensor,
        top_down: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x          : main input from previous layer (or i_t for first layer)
            h, c       : previous hidden/cell states for this layer
            bottom_up  : the encoded observation i_t (always provided)
            top_down   : (optional) hidden state of top layer from previous tick
                         (only used for bottom layer on ticks > 0)

        Returns:
            h_new, c_new : updated hidden and cell states for this layer.
        """
        # Incorporate top‑down connection if present
        if top_down is not None:
            x = x + self.top_down_proj(top_down)

        # Concatenate x (with possible top‑down) and bottom_up for the ConvLSTM input
        cell_input = torch.cat([x, bottom_up], dim=1)

        # ConvLSTM step
        h_new, c_new = self.conv_lstm(cell_input, h, c)

        # Pool‑and‑inject using the *previous* hidden state h
        if self.pool_inject is not None:
            h_new = h_new + self.pool_inject(h)

        return h_new, c_new


# ---------------------------------------------------------------------------
# Full DRC actor‑critic network
# ---------------------------------------------------------------------------
class DRCNetwork(nn.Module):
    """
    Deep Repeated ConvLSTM agent.

    Args:
        config : a dictionary or object with the following possible keys:
            channels (int, default 32)
            layers (int, default 3)
            internal_ticks (int, default 3)
            kernel_size (int, default 3)
            padding (int, default 1)
            pool_inject (bool, default True)
            hidden_size (int, default 256)
            num_actions (int, default 5)
            c_in (int, default 7)        # observation channels
    """
    def __init__(self, config):
        super().__init__()

        # Helper to read config whether it is a dict or an object with attributes
        def _get(key, default):
            if isinstance(config, dict):
                return config.get(key, default)
            else:
                return getattr(config, key, default)

        # Read architecture hyperparameters
        channels = _get('channels', 32)
        layers = _get('layers', 3)
        self.internal_ticks = _get('internal_ticks', 3)
        kernel_size = _get('kernel_size', 3)
        padding = _get('padding', 1)
        use_pool_inject = _get('pool_inject', True)
        hidden_size = _get('hidden_size', 256)
        num_actions = _get('num_actions', 5)
        c_in = _get('c_in', 7)

        # Spatial dimensions are fixed for the Sokoban environment (8×8 grid)
        self.spatial_size = (8, 8)

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        self.encoder = nn.Conv2d(
            in_channels=c_in,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

        # ------------------------------------------------------------------
        # DRC blocks
        # ------------------------------------------------------------------
        self.drc_blocks = nn.ModuleList()
        for _ in range(layers):
            self.drc_blocks.append(
                DRCBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    use_pool_inject=use_pool_inject,
                    spatial_size=self.spatial_size
                )
            )

        # ------------------------------------------------------------------
        # Post‑processing MLP
        # ------------------------------------------------------------------
        # Concatenation of final hidden state (channels) and encoded input (channels)
        flat_dim = 2 * channels * self.spatial_size[0] * self.spatial_size[1]
        self.post_linear = nn.Linear(flat_dim, hidden_size)

        # ------------------------------------------------------------------
        # Policy and value heads
        # ------------------------------------------------------------------
        self.policy_head = nn.Linear(hidden_size, num_actions)
        self.value_head = nn.Linear(hidden_size, 1)

        # Store hyperparams for convenience
        self.layers = layers
        self.channels = channels

    # ------------------------------------------------------------------
    # Recurrent state initialisation
    # ------------------------------------------------------------------
    def initial_state(
        self, batch_size: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns a list of (h, c) zero tensors for each layer,
        shaped (batch_size, channels, H, W).
        """
        device = next(self.parameters()).device
        Z = torch.zeros(
            batch_size,
            self.channels,
            self.spatial_size[0],
            self.spatial_size[1],
            device=device,
            dtype=torch.float32
        )
        return [(Z.clone(), Z.clone()) for _ in range(self.layers)]

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        obs: torch.Tensor,
        state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        num_ticks: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Process an observation (or sequence of observations) through the DRC.

        Args:
            obs        : tensor of shape (B, H, W, C_in) where H=W=8, C_in=7.
            state      : recurrent state from previous step (list of (h,c) per layer).
                         If None, initial zero states are used.
            num_ticks  : number of internal ticks (defaults to self.internal_ticks).

        Returns:
            logits     : action logits, shape (B, num_actions)
            value      : state value estimate, shape (B, 1)
            new_state  : updated recurrent state (list of (h,c) per layer)
        """
        if num_ticks is None:
            num_ticks = self.internal_ticks

        batch_size = obs.size(0)

        # Convert observation from NHWC to NCHW (channel‑first)
        obs = obs.permute(0, 3, 1, 2).float()     # (B, 7, 8, 8)

        # Encode observation once per step
        i_t = self.encoder(obs)                    # (B, C, 8, 8)

        # Initialise state if not provided
        if state is None:
            state = self.initial_state(batch_size)

        # Keep track of the top‑layer hidden state from the previous tick
        prev_h_last = None

        # Unroll internal ticks
        for tick in range(num_ticks):
            new_h_list, new_c_list = [], []

            for d, block in enumerate(self.drc_blocks):
                # --- Input x for this layer ---
                if d == 0:
                    x = i_t
                else:
                    x = new_h_list[d - 1]

                # --- Previous state ---
                h_prev, c_prev = state[d]

                # --- Top‑down connection (only bottom layer after tick 0) ---
                top_down = None
                if d == 0 and tick > 0 and prev_h_last is not None:
                    top_down = prev_h_last

                # --- DRC block forward ---
                h_new, c_new = block(
                    x=x,
                    h=h_prev,
                    c=c_prev,
                    bottom_up=i_t,
                    top_down=top_down
                )

                new_h_list.append(h_new)
                new_c_list.append(c_new)

            # Prepare state for next tick
            state = list(zip(new_h_list, new_c_list))

            # Remember the top layer’s hidden state for the next tick’s top‑down
            prev_h_last = new_h_list[-1]

        # ------------------------------------------------------------------
        # After the last tick: build output
        # ------------------------------------------------------------------
        h_final = state[-1][0]                 # top‑layer hidden state, (B, C, 8, 8)
        combined = torch.cat([h_final, i_t], dim=1)   # (B, 2C, 8, 8)
        combined_flat = combined.flatten(start_dim=1)  # (B, 2C*8*8)

        o_t = F.relu(self.post_linear(combined_flat))   # (B, hidden_size)

        logits = self.policy_head(o_t)                  # (B, num_actions)
        value = self.value_head(o_t)                    # (B, 1)

        return logits, value, state

    # ------------------------------------------------------------------
    # Interpretability helpers
    # ------------------------------------------------------------------
    def get_final_cell_states(
        self,
        obs: torch.Tensor,
        state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> List[torch.Tensor]:
        """
        Return a list of cell states (one per layer) after the final internal tick.

        Args:
            obs   : observation tensor (B, H, W, C_in)
            state : optional initial state; if None, zeros are used.

        Returns:
            list of cell state tensors, each of shape (B, channels, H, W),
            ordered from layer 0 (bottom) to layer L-1 (top).
        """
        _, _, new_state = self.forward(obs, state)
        # new_state is a list of (h, c) per layer; extract the cell states
        return [c for (_, c) in new_state]

    def forward_with_all_cell_states(
        self,
        obs: torch.Tensor,
        state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        num_ticks: Optional[int] = None
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        List[Tuple[torch.Tensor, torch.Tensor]],
        List[List[torch.Tensor]]
    ]:
        """
        Same as forward(), but additionally returns all cell states for each
        tick and each layer.

        Returns:
            logits, value, new_state, cell_states_per_tick
              cell_states_per_tick : list of length num_ticks, each element is a
                                     list of length L of cell state tensors
                                     (B, channels, H, W) for that tick.
        """
        if num_ticks is None:
            num_ticks = self.internal_ticks

        batch_size = obs.size(0)
        obs = obs.permute(0, 3, 1, 2).float()
        i_t = self.encoder(obs)

        if state is None:
            state = self.initial_state(batch_size)

        prev_h_last = None
        cell_states_per_tick = []

        for tick in range(num_ticks):
            new_h_list, new_c_list = [], []

            for d, block in enumerate(self.drc_blocks):
                if d == 0:
                    x = i_t
                else:
                    x = new_h_list[d - 1]

                h_prev, c_prev = state[d]

                top_down = None
                if d == 0 and tick > 0 and prev_h_last is not None:
                    top_down = prev_h_last

                h_new, c_new = block(
                    x=x,
                    h=h_prev,
                    c=c_prev,
                    bottom_up=i_t,
                    top_down=top_down
                )

                new_h_list.append(h_new)
                new_c_list.append(c_new)

            state = list(zip(new_h_list, new_c_list))
            prev_h_last = new_h_list[-1]

            # Record cell states for this tick
            cell_states_per_tick.append(new_c_list.copy())

        # Final output as in forward()
        h_final = state[-1][0]
        combined = torch.cat([h_final, i_t], dim=1)
        combined_flat = combined.flatten(start_dim=1)
        o_t = F.relu(self.post_linear(combined_flat))
        logits = self.policy_head(o_t)
        value = self.value_head(o_t)

        return logits, value, state, cell_states_per_tick

