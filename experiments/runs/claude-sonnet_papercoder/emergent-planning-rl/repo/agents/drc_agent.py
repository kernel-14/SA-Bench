## Code: agents/drc_agent.py

```python
## agents/drc_agent.py
"""Deep Repeated ConvLSTM (DRC) agent for the emergent planning interpretability pipeline.

This module implements the DRC(D,N) agent architecture introduced by Guez et al. (2019)
and analyzed in "Interpreting Emergent Planning in Model-Free Reinforcement Learning".

The DRC agent is a model-free recurrent actor-critic that performs N internal ticks of
ConvLSTM computation per environment step. The key architectural features are:

1. **Convolutional encoder**: Maps symbolic 8×8×7 observations to 8×8×32 encodings,
   preserving spatial dimensions for the spatial bijection with the Sokoban grid.

2. **ConvLSTM stack with skip connections**:
   - Bottom-up skip: encoding i_t fed to ALL layers at every tick
   - Top-down skip: final layer output h fed back to layer 0 at next tick
   - Pool-and-inject: handled internally by ConvLSTMCell

3. **Multiple ticks per step**: N=3 ticks of recurrent computation per environment step,
   enabling iterative plan refinement within a single observation.

4. **Explicit cell state exposure**: The cell state g_t^d (not output state h_t^d) is
   returned at every layer and tick, since all probes and interventions operate on it.

Critical design constraint (paper Section 4.1):
    Linear probes operate on the CELL STATE g_t^d (denoted c in LSTM notation),
    not the output state h_t^d. The forward() method must return all_cell_states
    as a List[List[Tensor]] indexed by [layer][tick], where each tensor is the
    cell state c (not h) of shape (B, 32, 8, 8).

Architecture reference (paper Sections 2.3, E.3):
    Encoder: Conv(7→32, k=3, p=1) → ReLU → Conv(32→32, k=3, p=1) → ReLU
    ConvLSTM stack: D=3 layers, N=3 ticks, hidden_dim=32, kernel=3, padding=1
    Skip connections:
        - Bottom-up: i_t → all layers (concatenated with layer input)
        - Top-down: h_{t,n-1}^{D-1} → layer 0 input at tick n
    Output: cat(h_final_flat, i_t_flat) → Linear → ReLU → policy_head, value_head

Input dimension per ConvLSTMCell (all layers identical):
    x = cat([i_t OR h_prev_layer, skip], dim=1) → input_dim = 32 + 32 = 64
    ConvLSTMCell gate input = input_dim + hidden_dim + hidden_dim = 64 + 32 + 32 = 128

Example:
    >>> import torch
    >>> agent = DRCAgent()
    >>> obs = torch.zeros(1, 7, 8, 8)
    >>> state = agent.initial_state(batch_size=1)
    >>> logits, value, new_state, cell_states = agent.forward(obs, state)
    >>> logits.shape
    torch.Size([1, 5])
    >>> len(cell_states), len(cell_states[0])
    (3, 3)
    >>> cell_states[0][0].shape  # layer 0, tick 0
    torch.Size([1, 32, 8, 8])
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.conv_lstm_cell import ConvLSTMCell


class DRCAgent(nn.Module):
    """Deep Repeated ConvLSTM agent with explicit cell state exposure.

    Implements the DRC(D,N) architecture from Guez et al. (2019) with the
    modifications described in paper Appendix E.3. The agent is a recurrent
    actor-critic that performs N internal ticks of ConvLSTM computation per
    environment step.

    The agent is designed for interpretability: it explicitly returns cell states
    at every layer and tick so that the probing and intervention pipeline can
    access them without re-running the forward pass.

    Attributes:
        obs_channels: Number of input observation channels (7 for symbolic Sokoban).
        hidden_dim: Channel dimension G for encoder and all ConvLSTM layers (32).
        num_layers: Number of ConvLSTM layers D (3 for DRC(3,3)).
        num_ticks: Number of internal ticks per step N (3 for DRC(3,3)).
        grid_h: Spatial height of the grid (8).
        grid_w: Spatial width of the grid (8).
        encoder: Two-layer convolutional encoder mapping obs → i_t.
        conv_lstm_layers: ModuleList of D ConvLSTMCell instances.
        output_fc: Linear layer fusing final hidden state and encoding.
        policy_head: Linear layer mapping to action logits.
        value_head: Linear layer mapping to scalar value estimate.

    Example:
        >>> agent = DRCAgent(obs_channels=7, hidden_dim=32, num_layers=3,
        ...                  num_ticks=3, grid_h=8, grid_w=8)
        >>> obs = torch.zeros(2, 7, 8, 8)  # batch=2
        >>> state = agent.initial_state(batch_size=2)
        >>> logits, value, new_state, cell_states = agent.forward(obs, state)
        >>> logits.shape
        torch.Size([2, 5])
        >>> value.shape
        torch.Size([2, 1])
        >>> len(new_state)  # D layers
        3
        >>> len(cell_states), len(cell_states[0])  # [D layers][N ticks]
        (3, 3)
    """

    def __init__(
        self,
        obs_channels: int = 7,
        hidden_dim: int = 32,
        num_layers: int = 3,
        num_ticks: int = 3,
        grid_h: int = 8,
        grid_w: int = 8,
        kernel_size: int = 3,
        n_actions: int = 5,
    ) -> None:
        """Initialize the DRC agent.

        Constructs the encoder, ConvLSTM stack, and output heads. All ConvLSTM
        layers have identical architecture (input_dim=64, hidden_dim=32) because
        both the bottom-up skip (i_t, 32 channels) and the layer-below output
        (h or top-down skip, 32 channels) contribute 32 channels each.

        Args:
            obs_channels: Number of channels in the symbolic observation.
                Matches config.yaml agent.obs_channels = 7.
            hidden_dim: Channel dimension G for encoder output and all ConvLSTM
                hidden states. Matches config.yaml agent.hidden_dim = 32.
            num_layers: Number of ConvLSTM layers D.
                Matches config.yaml agent.num_layers = 3.
            num_ticks: Number of internal ticks per environment step N.
                Matches config.yaml agent.num_ticks = 3.
            grid_h: Spatial height of the grid.
                Matches config.yaml agent.grid_h = 8.
            grid_w: Spatial width of the grid.
                Matches config.yaml agent.grid_w = 8.
            kernel_size: Convolutional kernel size for encoder and ConvLSTM cells.
                Matches config.yaml agent.kernel_size = 3.
            n_actions: Number of discrete actions.
                Matches config.yaml env.n_actions = 5.
        """
        super().__init__()

        self.obs_channels: int = obs_channels
        self.hidden_dim: int = hidden_dim
        self.num_layers: int = num_layers
        self.num_ticks: int = num_ticks
        self.grid_h: int = grid_h
        self.grid_w: int = grid_w
        self.kernel_size: int = kernel_size
        self.n_actions: int = n_actions

        # ------------------------------------------------------------------
        # Convolutional encoder: obs (B, 7, 8, 8) → i_t (B, 32, 8, 8)
        # Two conv layers with ReLU, same-padding preserves 8×8 spatial dims.
        # ------------------------------------------------------------------
        padding: int = kernel_size // 2  # = 1 for kernel_size=3
        self.encoder: nn.Sequential = nn.Sequential(
            nn.Conv2d(obs_channels, hidden_dim, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.ReLU(),
        )

        # ------------------------------------------------------------------
        # ConvLSTM stack: D layers with untied parameters.
        #
        # Input dimension for ALL layers = 64:
        #   - Layer 0: x = cat(i_t[32], top_down_skip[32]) → 64 channels
        #   - Layers 1..D-1: x = cat(h_prev_layer[32], i_t[32]) → 64 channels
        #
        # The ConvLSTMCell internally handles pool-and-inject (adds hidden_dim
        # channels) and the previous hidden state (adds hidden_dim channels),
        # so the gate convolution input is 64 + 32 + 32 = 128 channels.
        # ------------------------------------------------------------------
        conv_lstm_input_dim: int = hidden_dim + hidden_dim  # 32 + 32 = 64

        self.conv_lstm_layers: nn.ModuleList = nn.ModuleList([
            ConvLSTMCell(
                input_dim=conv_lstm_input_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                spatial_h=grid_h,
                spatial_w=grid_w,
            )
            for _ in range(num_layers)
        ])

        # ------------------------------------------------------------------
        # Output head: fuses final layer output with encoding.
        #
        # From paper Appendix E.3:
        #   "the output h_{t,N}^D of the final ConvLSTM cell at the final tick N
        #    is concatenated with the input encoding i_t and undergoes an affine
        #    transformation followed by a ReLU non-linearity to generate o_t"
        #
        # Dimensions:
        #   h_final_flat: (B, hidden_dim * grid_h * grid_w) = (B, 32*8*8) = (B, 2048)
        #   i_t_flat:     (B, hidden_dim * grid_h * grid_w) = (B, 2048)
        #   concat:       (B, 4096)
        #   output_fc:    (B, 4096) → (B, 2048)
        # ------------------------------------------------------------------
        spatial_flat_dim: int = hidden_dim * grid_h * grid_w  # 32 * 8 * 8 = 2048

        self.output_fc: nn.Linear = nn.Linear(
            in_features=2 * spatial_flat_dim,   # 4096
            out_features=spatial_flat_dim,       # 2048
            bias=True,
        )

        # Policy head: (B, 2048) → (B, n_actions=5)
        self.policy_head: nn.Linear = nn.Linear(
            in_features=spatial_flat_dim,
            out_features=n_actions,
            bias=True,
        )

        # Value head: (B, 2048) → (B, 1)
        self.value_head: nn.Linear = nn.Linear(
            in_features=spatial_flat_dim,
            out_features=1,
            bias=True,
        )

    def initial_state(
        self, batch_size: int = 1
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Create the initial (zero) recurrent state for a new episode.

        Returns a list of D (h, c) tuples, one per ConvLSTM layer, where both
        h (output state) and c (cell state) are initialized to zeros. The cell
        state c is what probes operate on; the output state h is used for skip
        connections and the final output computation.

        The device is inferred from the agent's parameters so that the initial
        state is automatically on the correct device (CPU or GPU).

        Args:
            batch_size: Number of parallel environments / batch size.
                Use 1 for single-environment inference, N for batched training.

        Returns:
            List of D tuples, each containing:
            - h: Output state tensor of shape (batch_size, hidden_dim, grid_h, grid_w)
                 = (batch_size, 32, 8, 8). Initialized to zeros.
            - c: Cell state tensor of shape (batch_size, hidden_dim, grid_h, grid_w)
                 = (batch_size, 32, 8, 8). Initialized to zeros.
            The list has length num_layers (D=3 for DRC(3,3)).

        Example:
            >>> state = agent.initial_state(batch_size=4)
            >>> len(state)  # D=3 layers
            3
            >>> state[0][0].shape  # h for layer 0
            torch.Size([4, 32, 8, 8])
            >>> state[0][1].shape  # c for layer 0
            torch.Size([4, 32, 8, 8])
        """
        # Infer device from model parameters.
        device: torch.device = next(self.parameters()).device

        state: List[Tuple[torch.Tensor, torch.Tensor]] = [
            (
                torch.zeros(batch_size, self.hidden_dim, self.grid_h, self.grid_w,
                            device=device),
                torch.zeros(batch_size, self.hidden_dim, self.grid_h, self.grid_w,
                            device=device),
            )
            for _ in range(self.num_layers)
        ]
        return state

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode a symbolic observation into a spatial feature map.

        Convenience method for modules that only need the encoding i_t without
        running the full recurrent forward pass (e.g., ConceptLabeler for
        baseline probe comparisons).

        Handles both HWC format (B, 8, 8, 7) from SokobanEnv and CHW format
        (B, 7, 8, 8) expected by the convolutional encoder.

        Args:
            obs: Symbolic observation tensor. Accepted shapes:
                - (B, 7, 8, 8): CHW format (already transposed)
                - (B, 8, 8, 7): HWC format (from SokobanEnv.get_symbolic_obs())
                - (8, 8, 7): Single unbatched HWC observation (auto-batched)

        Returns:
            Encoding tensor i_t of shape (B, hidden_dim, grid_h, grid_w)
            = (B, 32, 8, 8). Preserves spatial dimensions via same-padding.

        Example:
            >>> obs = torch.zeros(2, 8, 8, 7)  # HWC from SokobanEnv
            >>> i_t = agent.encode(obs)
            >>> i_t.shape
            torch.Size([2, 32, 8, 8])
        """
        obs = self._normalize_obs(obs)
        return self.encoder(obs)

    def forward(
        self,
        obs: torch.Tensor,
        state: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        List[Tuple[torch.Tensor, torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        """Run the full DRC forward pass for one environment step.

        Performs the following computation:
        1. Encode observation: obs → i_t (B, 32, 8, 8)
        2. For each tick n in [0, N):
           a. Layer 0: x = cat(i_t, top_down_skip) → ConvLSTMCell → (h_0, c_0)
           b. Layers 1..D-1: x = cat(h_{d-1}, i_t) → ConvLSTMCell → (h_d, c_d)
           c. Update top_down_skip = h_{D-1} for next tick
           d. Store cell states c_d for all layers
        3. Compute output: cat(h_{D-1}_flat, i_t_flat) → FC → ReLU → o_t
        4. Policy: policy_head(o_t) → logits (B, 5)
        5. Value: value_head(o_t) → value (B, 1)

        The top-down skip at tick n=0 uses the final layer's output state from
        the previous environment step (state[D-1][0]), enabling information to
        flow from the previous step's computation into the current step's first tick.

        Args:
            obs: Symbolic observation tensor. Accepted shapes:
                - (B, 7, 8, 8): CHW format
                - (B, 8, 8, 7): HWC format from SokobanEnv
                - (8, 8, 7): Single unbatched HWC (auto-batched to (1, 7, 8, 8))
            state: Recurrent state from the previous step. List of D tuples
                (h^d, c^d), each of shape (B, 32, 8, 8). Use initial_state()
                for the first step of an episode.

        Returns:
            Tuple of four elements:
            1. policy_logits: Tensor of shape (B, n_actions=5). Raw logits
               (not softmaxed) for the action distribution. Used by the trainer
               for loss computation and by get_action for sampling.
            2. value: Tensor of shape (B, 1). Scalar value estimate V(s_t).
               Used by V-trace for advantage computation.
            3. new_state: Updated recurrent state. List of D tuples (h^d, c^d),
               each of shape (B, 32, 8, 8). Pass to the next call to forward().
            4. all_cell_states: Cell states at every layer and tick.
               List[List[Tensor]] with shape [num_layers][num_ticks].
               all_cell_states[d][n] is the cell state c of layer d after tick n+1,
               shape (B, 32, 8, 8). This is g_t^d in the paper's notation.
               - For probing: use all_cell_states[d][-1] (final tick)
               - For thinking-step analysis: use all_cell_states[d][n] for each n
               - For interventions: modify state[d][1] before calling forward()

        Example:
            >>> obs = torch.zeros(1, 8, 8, 7)  # HWC from SokobanEnv
            >>> state = agent.initial_state(1)
            >>> logits, value, new_state, cell_states = agent.forward(obs, state)
            >>> logits.shape
            torch.Size([1, 5])
            >>> value.shape
            torch.Size([1, 1])
            >>> cell_states[2][-1].shape  # layer 3, final tick
            torch.Size([1, 32, 8, 8])
        """
        # ------------------------------------------------------------------
        # Step 1: Normalize observation to (B, C, H, W) format.
        # ------------------------------------------------------------------
        obs_chw: torch.Tensor = self._normalize_obs(obs)
        batch_size: int = obs_chw.shape[0]

        # ------------------------------------------------------------------
        # Step 2: Encode observation to spatial feature map.
        # i_t shape: (B, hidden_dim, grid_h, grid_w) = (B, 32, 8, 8)
        # ------------------------------------------------------------------
        i_t: torch.Tensor = self.encoder(obs_chw)

        # ------------------------------------------------------------------
        # Step 3: Initialize tick-level state from the recurrent state.
        # current_h[d] and current_c[d] are the h and c for layer d.
        # top_down is the output state of the final layer from the previous
        # step's final tick — used as the top-down skip at tick n=0.
        # ------------------------------------------------------------------
        current_h: List[torch.Tensor] = [state[d][0] for d in range(self.num_layers)]
        current_c: List[torch.Tensor] = [state[d][1] for d in range(self.num_layers)]

        # Top-down skip: h from final layer, previous step's final tick.
        # At the very first step (initial_state), this is zeros.
        top_down: torch.Tensor = state[self.num_layers - 1][0]

        # ------------------------------------------------------------------
        # Step 4: Initialize cell state collection.
        # all_cell_states[d][n] = cell state c of layer d after tick n+1.
        # Initialized with None placeholders; filled during tick loop.
        # ------------------------------------------------------------------
        all_cell_states: List[List[Optional[torch.Tensor]]] = [
            [None] * self.num_ticks for _ in range(self.num_layers)
        ]

        # ------------------------------------------------------------------
        # Step 5: Run N ticks of recurrent computation.
        # ------------------------------------------------------------------
        for tick_idx in range(self.num_ticks):
            # Temporary storage for this tick's new h values (needed for
            # bottom-up skip to higher layers within the same tick).
            new_h: List[torch.Tensor] = [None] * self.num_layers  # type: ignore
            new_c: List[torch.Tensor] = [None] * self.num_layers  # type: ignore

            for layer_idx in range(self.num_layers):
                if layer_idx == 0:
                    # Layer 0: receives i_t (bottom-up) + top_down (top-down skip).
                    # x shape: (B, 64, 8, 8)
                    x: torch.Tensor = torch.cat([i_t, top_down], dim=1)
                else:
                    # Layers 1..D-1: receive h from layer below (bottom-up)
                    # + i_t (bottom-up skip to all layers).
                    # x shape: (B, 64, 8, 8)
                    x = torch.cat([new_h[layer_idx - 1], i_t], dim=1)

                # Run one ConvLSTM tick.
                h_new: torch.Tensor
                c_new: torch.Tensor
                h_new, c_new = self.conv_lstm_layers[layer_idx](
                    x, current_h[layer_idx], current_c[layer_idx]
                )

                new_h[layer_idx] = h_new
                new_c[layer_idx] = c_new

                # Store cell state for this layer and tick.
                # c_new is g_t^d in the paper's notation — what probes use.
                all_cell_states[layer_idx][tick_idx] = c_new

            # Update current h and c for all layers.
            current_h = new_h
            current_c = new_c

            # Update top-down skip for the next tick:
            # h of the final layer at this tick feeds back to layer 0 next tick.
            top_down = current_h[self.num_layers - 1]

        # ------------------------------------------------------------------
        # Step 6: Compute output from final layer, final tick.
        # ------------------------------------------------------------------
        # h_final: output state of final layer after final tick (B, 32, 8, 8)
        h_final: torch.Tensor = current_h[self.num_layers - 1]

        # Flatten spatial dimensions for the output FC layer.
        h_final_flat: torch.Tensor = h_final.flatten(start_dim=1)   # (B, 2048)
        i_t_flat: torch.Tensor = i_t.flatten(start_dim=1)            # (B, 2048)

        # Concatenate and apply affine + ReLU to get o_t.
        # o_t shape: (B, 2048)
        o_t: torch.Tensor = F.relu(
            self.output_fc(torch.cat([h_final_flat, i_t_flat], dim=1))
        )

        # Policy logits: (B, n_actions=5)
        policy_logits: torch.Tensor = self.policy_head(o_t)

        # Value estimate: (B, 1)
        value: torch.Tensor = self.value_head(o_t)

        # ------------------------------------------------------------------
        # Step 7: Construct new recurrent state for the next step.
        # ------------------------------------------------------------------
        new_state: List[Tuple[torch.Tensor, torch.Tensor]] = [
            (current_h[d], current_c[d]) for d in range(self.num_layers)
        ]

        # Cast all_cell_states to the correct type (remove Optional).
        # At this point all entries have been filled by the tick loop.
        cell_states_typed: List[List[torch.Tensor]] = [
            [all_cell_states[d][n] for n in range(self.num_ticks)]  # type: ignore
            for d in range(self.num_layers)
        ]

        return policy_logits, value, new_state, cell_states_typed

    def get_action(
        self,
        obs: torch.Tensor,
        state: List[Tuple[torch.Tensor, torch.Tensor]],
        greedy: bool = False,
    ) -> Tuple[
        int,
        torch.Tensor,
        torch.Tensor,
        List[Tuple[torch.Tensor, torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        """Select an action given an observation and recurrent state.

        Wraps forward() to provide a convenient interface for environment
        interaction. Handles both stochastic sampling (training) and greedy
        selection (evaluation) via the greedy flag.

        During training (greedy=False): samples from the categorical distribution
        parameterized by policy_logits, as described in paper Section E.4.
        At test time (greedy=True): selects the action with the highest logit,
        as described in paper Section E.4 ("acts greedily by always performing
        the action with the greatest logit").

        Args:
            obs: Symbolic observation tensor. Accepted shapes:
                - (B, 7, 8, 8): CHW format
                - (B, 8, 8, 7): HWC format from SokobanEnv
                - (8, 8, 7): Single unbatched HWC (auto-batched)
                - (1, 8, 8, 7): Single batched HWC
            state: Recurrent state from the previous step. List of D tuples
                (h^d, c^d), each of shape (B, 32, 8, 8).
            greedy: If True, select the action with the highest logit (argmax).
                If False, sample from the categorical distribution.
                Default False (training mode). Set True for evaluation.

        Returns:
            Tuple of five elements:
            1. action: int, the selected action in {0, 1, 2, 3, 4}.
               For batched inputs (B>1), returns the action for the first
               element (index 0). The IMPALATrainer handles batching separately.
            2. log_prob: Tensor of shape (B,), log probability of the selected
               action under the current policy. Used by V-trace for importance
               sampling ratios.
            3. value: Tensor of shape (B,), scalar value estimate (squeezed
               from (B, 1) to (B,)). Used by V-trace for advantage computation.
            4. new_state: Updated recurrent state. List of D tuples (h^d, c^d).
            5. all_cell_states: Cell states at every layer and tick.
               List[List[Tensor]] with shape [num_layers][num_ticks].
               all_cell_states[d][n] shape: (B, 32, 8, 8).

        Example:
            >>> obs = torch.zeros(1, 8, 8, 7)
            >>> state = agent.initial_state(1)
            >>> action, log_prob, value, new_state, cell_states = agent.get_action(
            ...     obs, state, greedy=True
            ... )
            >>> isinstance(action, int)
            True
            >>> log_prob.shape
            torch.Size([1])
            >>> value.shape
            torch.Size([1])
        """
        # Run full forward pass.
        policy_logits: torch.Tensor
        value_raw: torch.Tensor
        new_state: List[Tuple[torch.Tensor, torch.Tensor]]
        all_cell_states: List[List[torch.Tensor]]

        policy_logits, value_raw, new_state, all_cell_states = self.forward(
            obs, state
        )

        # Build categorical distribution from logits.
        dist: Categorical = Categorical(logits=policy_logits)

        # Select action.
        if greedy:
            action_tensor: torch.Tensor = policy_logits.argmax(dim=-1)
        else:
            action_tensor = dist.sample()

        # Compute log probability of the selected action.
        log_prob: torch.Tensor = dist.log_prob(action_tensor)

        # Squeeze value from (B, 1) to (B,).
        value: torch.Tensor = value_raw.squeeze(-1)

        # Return scalar action (int) for single-environment interaction.
        # For batched environments, the trainer handles the batch dimension.
        action: int = int(action_tensor[0].item())

        return action, log_prob, value, new_state, all_cell_states

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observation tensor to (B, C, H, W) CHW format.

        SokobanEnv.get_symbolic_obs() returns observations in HWC format
        (H, W, C) = (8, 8, 7). PyTorch convolutions expect BCHW format.
        This method handles the transposition and auto-batching.

        Accepted input shapes:
        - (8, 8, 7): Single unbatched HWC → (1, 7, 8, 8)
        - (B, 8, 8, 7): Batched HWC → (B, 7, 8, 8)
        - (B, 7, 8, 8): Already in CHW format → returned as-is
        - (7, 8, 8): Single unbatched CHW → (1, 7, 8, 8)

        Args:
            obs: Observation tensor in any of the above formats.

        Returns:
            Observation tensor in (B, C, H, W) = (B, 7, 8, 8) format,
            on the same device as the input tensor.

        Raises:
            ValueError: If the observation shape is not recognized.
        """
        if obs.dim() == 3:
            # Single unbatched observation: either (H, W, C) or (C, H, W).
            if obs.shape[-1] == self.obs_channels:
                # (H, W, C) = (8, 8, 7) → (1, 7, 8, 8)
                obs = obs.permute(2, 0, 1).unsqueeze(0)
            elif obs.shape[0] == self.obs_channels:
                # (C, H, W) = (7, 8, 8) → (1, 7