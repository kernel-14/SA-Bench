import torch
import torch.nn as nn
import torch.nn.functional as F

from config import AgentConfig
from layers import ConvLSTMCell, PoolAndInject, ResBlock

class ConvEncoder(nn.Module):
    """
    Convolutional encoder to process observation into an encoding.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super(ConvEncoder, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, channels, H, W)
        return F.relu(self.conv(x)) # (batch_size, out_channels, H_0, W_0)

class DRC(nn.Module):
    """
    Deep Repeated ConvLSTM (DRC) agent architecture.
    A recurrent actor-critic architecture based on ConvLSTMs.
    """
    def __init__(self, config: AgentConfig):
        super(DRC, self).__init__()
        self.config = config

        self.encoder = ConvEncoder(config.OBSERVATION_CHANNELS, config.CHANNELS, config.KERNEL_SIZE, config.PADDING)

        self.convlstm_layers = nn.ModuleList()
        self.pool_and_inject_layers = nn.ModuleList()

        for d in range(config.D_CONVLSTM_LAYERS):
            input_dim_for_cell = config.CHANNELS + config.CHANNELS # For i_t and p_t_n-1_d
            if d == 0: # Only the first ConvLSTM layer receives top-down skip connection
                input_dim_for_cell += config.CHANNELS
            self.convlstm_layers.append(
                ConvLSTMCell(input_dim=input_dim_for_cell,
                             hidden_dim=config.CHANNELS,
                             kernel_size=config.KERNEL_SIZE,
                             padding=config.PADDING)
            )
            self.pool_and_inject_layers.append(
                PoolAndInject(config.CHANNELS, config.GRID_SIZE)
            )

        self.output_transform = nn.Sequential(
            nn.Conv2d(config.CHANNELS * 2, config.CHANNELS * 2, kernel_size=1), # Equivalent to affine on concatenated feature maps
            nn.ReLU()
        )
        flattened_size = config.CHANNELS * 2 * config.GRID_SIZE * config.GRID_SIZE
        self.policy_head = nn.Linear(flattened_size, config.NUM_ACTIONS)
        self.value_head = nn.Linear(flattened_size, 1)

    def forward(self, x_t: torch.Tensor, prev_states: list[tuple[torch.Tensor, torch.Tensor]] | None = None) -> \
            tuple[torch.Tensor, torch.Tensor, list[list[tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Performs N internal ticks of computation for a single environment step.
        x_t: current observation (batch_size, channels, H, W)
        prev_states: list of (h, c) tuples for each ConvLSTM layer from previous env step.
                     If None, initialize with zeros.
        Returns:
            policy_logits: logits for actions
            value: estimated state value
            new_states: list of (h, c) tuples for each ConvLSTM layer after N ticks
            all_cell_states_per_tick: list of (list of (h,c) per layer) after each internal tick, for probing
        """
        batch_size, _, H, W = x_t.shape
        i_t = self.encoder(x_t) # (batch_size, CHANNELS, H, W)

        if prev_states is None:
            # Initialize hidden and cell states with zeros
            h_init = torch.zeros(batch_size, self.config.CHANNELS, H, W, device=x_t.device)
            c_init = torch.zeros(batch_size, self.config.CHANNELS, H, W, device=x_t.device)
            prev_states = [(h_init, c_init) for _ in range(self.config.D_CONVLSTM_LAYERS)]

        current_states = prev_states
        all_cell_states_per_tick = [] # To store all (h,c) for all layers after each internal tick, for probing

        for n in range(self.config.N_INTERNAL_TICKS):
            next_states_tick = []
            # h_D_n_minus_1 is the output of the final ConvLSTM layer from the *previous* internal tick
            h_D_n_minus_1 = all_cell_states_per_tick[-1][-1][0] if n > 0 else None 
            
            current_tick_layer_states = [] # Store (h,c) for all layers in current tick for probing
            
            for d in range(self.config.D_CONVLSTM_LAYERS):
                h_d_n_minus_1, c_d_n_minus_1 = current_states[d]

                # Pool-and-Inject: p_t_n-1_d from current layer d, previous tick
                p_d_n_minus_1 = self.pool_and_inject_layers[d](h_d_n_minus_1)

                # Assemble input_tensor for ConvLSTMCell
                input_to_cell = [i_t, p_d_n_minus_1]
                if d == 0 and h_D_n_minus_1 is not None: # Top-down skip to the first ConvLSTM layer
                    input_to_cell.append(h_D_n_minus_1)
                
                effective_input_tensor = torch.cat(input_to_cell, dim=1)

                h_d_n, c_d_n = self.convlstm_layers[d](effective_input_tensor, (h_d_n_minus_1, c_d_n_minus_1))
                next_states_tick.append((h_d_n, c_d_n))
                current_tick_layer_states.append((h_d_n, c_d_n))
            
            current_states = next_states_tick
            all_cell_states_per_tick.append(current_tick_layer_states)
        
        # After N ticks, the final output for policy and value heads
        h_D_N = current_states[-1][0] # Final hidden state of the last ConvLSTM layer

        # Concatenate final hidden state with input encoding
        output_concat = torch.cat([h_D_N, i_t], dim=1) # (batch_size, 2*CHANNELS, H, W)
        
        o_t = self.output_transform(output_concat) # (batch_size, 2*CHANNELS, H, W)
        
        # Flatten for policy and value heads
        o_t_flat = o_t.view(batch_size, -1)

        policy_logits = self.policy_head(o_t_flat)
        value = self.value_head(o_t_flat).squeeze(-1) # Squeeze to (batch_size,)

        return policy_logits, value, current_states, all_cell_states_per_tick

class ResNetAgent(nn.Module):
    """
    ResNet agent architecture for Sokoban (Appendix G).
    """
    def __init__(self, config: AgentConfig, num_res_blocks: int = 24):
        super(ResNetAgent, self).__init__()
        self.config = config

        self.initial_conv = nn.Conv2d(config.OBSERVATION_CHANNELS, config.CHANNELS, 
                                      kernel_size=config.KERNEL_SIZE, padding=config.PADDING)
        
        self.res_blocks = nn.ModuleList([
            ResBlock(config.CHANNELS, config.GRID_SIZE, config.KERNEL_SIZE, config.PADDING) 
            for _ in range(num_res_blocks)
        ])

        flattened_size = config.CHANNELS * config.GRID_SIZE * config.GRID_SIZE
        self.mlp = nn.Linear(flattened_size, 256) # MLP of dimensionality 256

        self.policy_head = nn.Linear(256, config.NUM_ACTIONS)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        x_t: current observation (batch_size, channels, H, W)
        Returns:
            policy_logits: logits for actions
            value: estimated state value
            hidden_states_per_layer: list of hidden states after each residual block's final ReLU, for probing
        """
        batch_size = x_t.shape[0]
        
        x = F.relu(self.initial_conv(x_t)) # (batch_size, CHANNELS, H, W)
        
        hidden_states_per_layer = [] # Store hidden states after each res block's final ReLU
        for block in self.res_blocks:
            x = block(x)
            hidden_states_per_layer.append(x) # Store the output of the block after final ReLU
        
        # Flatten after all residual blocks
        x_flat = x.view(batch_size, -1)
        
        # Pass through MLP
        mlp_out = F.relu(self.mlp(x_flat))
        
        policy_logits = self.policy_head(mlp_out)
        value = self.value_head(mlp_out).squeeze(-1)

        return policy_logits, value, hidden_states_per_layer
