"""
Deep Repeated ConvLSTM (DRC) Agent implementation.
Based on: Guez et al. (2019) "An Investigation of Model-Free Planning"

Architecture (from Appendix E.3):
- Convolutional encoder: obs -> encoding i_t in R^{H x W x G_0}
- Stack of D ConvLSTM layers with N ticks per step
- Policy and value heads

Key features:
1. Bottom-up skip connections: input encoding i_t is provided to ALL ConvLSTM units
2. Top-down skip connections: output of final ConvLSTM on current tick is provided
   as additional input to the bottom ConvLSTM on the NEXT tick
3. Pool-and-inject: each ConvLSTM cell receives spatially pooled version of its
   own output from the prior tick
4. N ticks of recurrent computation per environment step

The agent studied in the paper: DRC(3,3) with 32 channels, kernel size 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


GRID_SIZE = 8
NUM_CHANNELS = 7  # Symbolic observation channels
NUM_ACTIONS = 5   # noop, up, down, left, right


class ConvLSTMCell(nn.Module):
    """
    ConvLSTM cell with pool-and-inject mechanism.
    
    From Appendix E.3:
    Each ConvLSTM cell receives:
    1. Input encoding i_t (bottom-up skip)
    2. Top-down skip from previous tick's final layer (only for first layer)
    3. Previous hidden state h
    4. Pool-and-inject: spatially pooled version of h from prior tick
    
    The pool-and-inject is computed as:
    m = [MeanPool(h), MaxPool(h)]  # (2*G_d,)
    p_hat = W_p * m + b_p          # (H*W*G_d,)
    p = Reshape(p_hat)             # (H, W, G_d)
    """
    
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        """
        Args:
            input_channels: Number of input channels (encoding + optional top-down skip)
            hidden_channels: Number of hidden channels (G_d in paper)
            kernel_size: Convolution kernel size (3 in paper)
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2  # Same padding
        
        # Pool-and-inject: project pooled hidden state back to spatial dimensions
        # Input: concatenated mean and max pool -> 2*hidden_channels
        # Output: H*W*hidden_channels (reshaped to H x W x hidden_channels)
        self.pool_inject = nn.Linear(
            2 * hidden_channels, 
            GRID_SIZE * GRID_SIZE * hidden_channels
        )
        
        # ConvLSTM gates: input + hidden + pool_inject -> 4 * hidden (for i, f, g, o gates)
        # Total input channels: input_channels + hidden_channels + hidden_channels (pool inject)
        total_input = input_channels + hidden_channels + hidden_channels
        self.conv_gates = nn.Conv2d(
            total_input, 
            4 * hidden_channels, 
            kernel_size=kernel_size, 
            padding=padding
        )
    
    def forward(
        self, 
        x: torch.Tensor,           # (B, input_channels, H, W)
        h: torch.Tensor,           # (B, hidden_channels, H, W)
        c: torch.Tensor,           # (B, hidden_channels, H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of ConvLSTM cell.
        
        Args:
            x: Input tensor (B, input_channels, H, W)
            h: Previous hidden state (B, hidden_channels, H, W)
            c: Previous cell state (B, hidden_channels, H, W)
            
        Returns:
            (new_h, new_c): New hidden and cell states
        """
        B, _, H, W = x.shape
        
        # Pool-and-inject: create global context from hidden state
        # Mean pool and max pool over spatial dimensions
        h_mean = h.mean(dim=[2, 3])  # (B, hidden_channels)
        h_max = h.max(dim=3)[0].max(dim=2)[0]  # (B, hidden_channels)
        m = torch.cat([h_mean, h_max], dim=1)  # (B, 2*hidden_channels)
        
        # Project to spatial dimensions
        p_flat = self.pool_inject(m)  # (B, H*W*hidden_channels)
        p = p_flat.view(B, self.hidden_channels, H, W)  # (B, hidden_channels, H, W)
        
        # Concatenate input, hidden state, and pool-inject
        combined = torch.cat([x, h, p], dim=1)  # (B, total_input, H, W)
        
        # Compute gates
        gates = self.conv_gates(combined)  # (B, 4*hidden_channels, H, W)
        
        # Split into 4 gates
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)
        
        # Apply activations
        i_gate = torch.sigmoid(i_gate)  # input gate
        f_gate = torch.sigmoid(f_gate)  # forget gate
        g_gate = torch.tanh(g_gate)     # cell gate
        o_gate = torch.sigmoid(o_gate)  # output gate
        
        # Update cell and hidden state
        new_c = f_gate * c + i_gate * g_gate
        new_h = o_gate * torch.tanh(new_c)
        
        return new_h, new_c
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and cell states to zeros."""
        h = torch.zeros(batch_size, self.hidden_channels, GRID_SIZE, GRID_SIZE, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, GRID_SIZE, GRID_SIZE, device=device)
        return h, c


class DRCAgent(nn.Module):
    """
    Deep Repeated ConvLSTM (DRC) Agent.
    
    Architecture (from Appendix E.3):
    1. Convolutional encoder: obs -> encoding i_t
    2. Stack of D ConvLSTM layers with N ticks per step
       - All layers receive i_t (bottom-up skip)
       - First layer also receives output of last layer from previous tick (top-down skip)
       - Each layer has pool-and-inject
    3. Output: concat(h_{t,N}^D, i_t) -> affine -> ReLU -> o_t
    4. Policy head: o_t -> action logits
    5. Value head: o_t -> value estimate
    
    The agent studied in the paper is DRC(3,3) with:
    - D=3 ConvLSTM layers
    - N=3 ticks per step
    - 32 channels
    - Kernel size 3
    """
    
    def __init__(
        self, 
        D: int = 3,              # Number of ConvLSTM layers
        N: int = 3,              # Number of ticks per step
        hidden_channels: int = 32,  # G_d in paper
        obs_channels: int = 7,   # Input observation channels
        num_actions: int = 5,    # Number of actions
        kernel_size: int = 3,    # Convolution kernel size
    ):
        """
        Args:
            D: Number of ConvLSTM layers
            N: Number of ticks per step
            hidden_channels: Number of hidden channels per ConvLSTM
            obs_channels: Number of observation channels
            num_actions: Number of possible actions
            kernel_size: Convolution kernel size
        """
        super().__init__()
        self.D = D
        self.N = N
        self.hidden_channels = hidden_channels
        self.num_actions = num_actions
        
        # Convolutional encoder: obs -> encoding
        # Input: (B, obs_channels, H, W)
        # Output: (B, hidden_channels, H, W)
        padding = kernel_size // 2
        self.encoder = nn.Sequential(
            nn.Conv2d(obs_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
        )
        
        # Stack of ConvLSTM cells
        # Bottom-up skip: ALL cells receive encoding i_t
        # Top-down skip: first cell also receives output of last cell from previous tick
        # So first cell input = encoding + top-down = 2 * hidden_channels
        # Other cells input = encoding = hidden_channels
        self.convlstm_cells = nn.ModuleList()
        for d in range(D):
            if d == 0:
                # First cell also receives top-down skip from last cell
                input_channels = hidden_channels + hidden_channels  # encoding + top-down
            else:
                # Other cells receive only encoding (bottom-up skip)
                input_channels = hidden_channels
            
            cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size)
            self.convlstm_cells.append(cell)
        
        # Output layer: combine final hidden state with encoding
        # h_{t,N}^D concatenated with i_t -> affine -> ReLU -> o_t
        # Input: (hidden_channels + hidden_channels) * H * W
        # Output: hidden_channels (as a vector)
        output_input_size = (hidden_channels + hidden_channels) * GRID_SIZE * GRID_SIZE
        self.output_layer = nn.Sequential(
            nn.Linear(output_input_size, hidden_channels),
            nn.ReLU(),
        )
        
        # Policy head: hidden_channels -> num_actions
        self.policy_head = nn.Linear(hidden_channels, num_actions)
        
        # Value head: hidden_channels -> 1
        self.value_head = nn.Linear(hidden_channels, 1)
    
    def forward(
        self,
        obs: torch.Tensor,
        hidden_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_cell_states: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], Optional[List]]:
        """
        Forward pass of DRC agent.
        
        Args:
            obs: Observation tensor (B, H, W, obs_channels) - channel-last format
            hidden_states: List of (h, c) tuples for each ConvLSTM layer, or None for zeros
            return_cell_states: If True, return cell states at each tick
            
        Returns:
            (policy_logits, value, new_hidden_states, cell_states_per_tick)
            - policy_logits: (B, num_actions)
            - value: (B, 1)
            - new_hidden_states: List of (h, c) for each layer
            - cell_states_per_tick: List of [layer_cell_states] at each tick (if return_cell_states)
              cell_states_per_tick[tick][layer] = cell state tensor (B, C, H, W)
        """
        B = obs.shape[0]
        device = obs.device
        
        # Convert from channel-last to channel-first
        # (B, H, W, C) -> (B, C, H, W)
        x = obs.permute(0, 3, 1, 2).float()
        
        # Encode observation: i_t
        i_t = self.encoder(x)  # (B, hidden_channels, H, W)
        
        # Initialize hidden states if not provided
        if hidden_states is None:
            hidden_states = [
                cell.init_hidden(B, device) 
                for cell in self.convlstm_cells
            ]
        
        # Store cell states per tick if requested
        cell_states_per_tick = [] if return_cell_states else None
        
        # Perform N ticks of recurrent computation
        current_states = list(hidden_states)
        
        # Top-down skip connection: output of final layer from previous tick
        # Initialize to zeros for first tick
        top_down = torch.zeros_like(i_t)
        
        for n in range(self.N):
            new_states = []
            
            for d, cell in enumerate(self.convlstm_cells):
                h, c = current_states[d]
                
                if d == 0:
                    # First cell: receives encoding + top-down skip
                    cell_input = torch.cat([i_t, top_down], dim=1)
                else:
                    # Other cells: receive only encoding (bottom-up skip)
                    cell_input = i_t
                
                new_h, new_c = cell(cell_input, h, c)
                new_states.append((new_h, new_c))
            
            current_states = new_states
            
            # Update top-down skip connection for next tick
            # = output (h) of final layer at current tick
            top_down = current_states[-1][0]  # h of last layer
            
            # Store cell states if requested
            if return_cell_states:
                tick_cell_states = [state[1] for state in current_states]  # cell states (c)
                cell_states_per_tick.append(tick_cell_states)
        
        # Get final hidden state of last layer: h_{t,N}^D
        final_h = current_states[-1][0]  # (B, hidden_channels, H, W)
        
        # Combine final hidden state with encoding: concat(h_{t,N}^D, i_t)
        # Flatten spatial dimensions
        combined = torch.cat([
            final_h.flatten(1),  # (B, hidden_channels * H * W)
            i_t.flatten(1),      # (B, hidden_channels * H * W)
        ], dim=1)
        
        # Apply affine + ReLU to get o_t
        o_t = self.output_layer(combined)  # (B, hidden_channels)
        
        # Policy and value heads
        policy_logits = self.policy_head(o_t)  # (B, num_actions)
        value = self.value_head(o_t)            # (B, 1)
        
        return policy_logits, value, current_states, cell_states_per_tick
    
    def get_action(
        self, 
        obs: torch.Tensor,
        hidden_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        greedy: bool = True,
    ) -> Tuple[int, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Get action for a single observation.
        
        Args:
            obs: Observation (H, W, C) or (1, H, W, C)
            hidden_states: Previous hidden states
            greedy: If True, take argmax; otherwise sample
            
        Returns:
            (action, new_hidden_states)
        """
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        
        with torch.no_grad():
            logits, _, new_hidden_states, _ = self.forward(obs, hidden_states)
            
            if greedy:
                action = logits.argmax(dim=-1).item()
            else:
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()
        
        return action, new_hidden_states
    
    def get_cell_states(
        self,
        obs: torch.Tensor,
        hidden_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[List[torch.Tensor]]]:
        """
        Get cell states at each tick for interpretability analysis.
        
        Args:
            obs: Observation (B, H, W, C)
            hidden_states: Previous hidden states
            
        Returns:
            (new_hidden_states, cell_states_per_tick)
            cell_states_per_tick[tick][layer] = cell state tensor (B, C, H, W)
        """
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        
        with torch.no_grad():
            _, _, new_hidden_states, cell_states_per_tick = self.forward(
                obs, hidden_states, return_cell_states=True
            )
        
        return new_hidden_states, cell_states_per_tick
    
    def init_hidden(
        self, 
        batch_size: int = 1, 
        device: Optional[torch.device] = None
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Initialize hidden states for all layers."""
        if device is None:
            device = next(self.parameters()).device
        return [cell.init_hidden(batch_size, device) for cell in self.convlstm_cells]
