import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
from typing import Any, Tuple, Optional, List, Union

# Attempt to import Config from config.py
# If config.py is not yet available (e.g., during isolated testing or initial setup),
# a dummy Config class will be used to prevent circular imports and allow this module to be tested.
try:
    from config import Config
    from agents.base_agent import BaseAgentModel
except ImportError:
    # Dummy Config class for self-testing or if config.py is not yet available
    class Config:
        def __init__(self, data: dict = None):
            self._data = data if data is not None else {}
        def get(self, key: str, default: Any = None) -> Any:
            keys = key.split('.')
            current = self._data
            for k in keys:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return default
            return current
        def set(self, key: str, value: Any) -> None:
            keys = key.split('.')
            d = self._data
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
        def save(self, output_path: str) -> None: pass
    print("Warning: Could not import 'Config' from 'config.py'. Using a dummy Config class.")

    # Dummy BaseAgentModel for self-testing
    class BaseAgentModel(nn.Module):
        def __init__(self, config: Config) -> None:
            super().__init__()
            self.config = config
            self.device = torch.device("cpu")
            self.action_space_size = 5
            self.grid_height = 8
            self.grid_width = 8
            self.obs_channels = 7
            self.observation_space_shape = (8, 8, 7)
        def _build_model_architecture(self) -> None: pass
        def forward(self, obs: torch.Tensor, hidden_state: Any = None) -> Tuple[torch.Tensor, Any]: pass
        def get_cell_state(self, layer_idx: int, tick_idx: int = -1) -> torch.Tensor: pass
        def set_cell_state(self, layer_idx: int, tick_idx: int, new_state: torch.Tensor) -> None: pass
        def act(self, obs: np.ndarray, hidden_state: Any = None, greedy: bool = True) -> Tuple[int, Any, torch.Tensor, torch.Tensor]:
            # Dummy act method
            return 0, None, torch.zeros(self.action_space_size), torch.zeros(1)
    print("Warning: Could not import 'BaseAgentModel'. Using a dummy BaseAgentModel class.")


class _ConvEncoder(nn.Module):
    """
    Convolutional encoder to process raw symbolic observations into an encoding i_t.
    """
    def __init__(self, config: Config) -> None:
        super().__init__()
        env_name: str = config.get("environment.name", "Sokoban")
        if env_name == "Sokoban":
            in_channels: int = config.get("environment.sokoban.observation_channels", 7)
        elif env_name == "MiniPacMan":
            in_channels: int = config.get("environment.mini_pacman.observation_channels", 14)
        else:
            raise ValueError(f"Unsupported environment configured: {env_name} for ConvEncoder.")

        out_channels: int = config.get("agent.drc_agent.encoder.out_channels", 32)
        kernel_size: int = config.get("agent.drc_agent.encoder.kernel_size", 3)
        padding: int = config.get("agent.drc_agent.encoder.padding", 1) # Assumes padding to maintain spatial dims

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        # Activation function can be customized. Default to ReLU if not specified or unrecognized.
        activation_str: str = config.get("agent.drc_agent.encoder.activation", "ReLU")
        if activation_str == "ReLU":
            self.activation = nn.ReLU()
        elif activation_str == "Identity":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation function for encoder: {activation_str}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input observation tensor. Shape (B, C_obs, H, W).
        Returns:
            torch.Tensor: Encoded tensor i_t. Shape (B, G0, H0, W0).
        """
        return self.activation(self.conv(x))


class _ConvLSTMCell(nn.Module):
    """
    A single ConvLSTM unit as described in Shi et al. 2015 and adapted for DRC.
    Its forward method must accept an additional `pooled_injection` tensor.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        # Convolutional layers for the gates (input, forget, cell, output)
        # Each gate's convolution takes the concatenated (x_curr + h_prev) as input
        # So, the input channels for these convolutions are (in_channels + out_channels)
        self.conv_gates = nn.Conv2d(
            in_channels=self.in_channels + self.out_channels,
            out_channels=4 * self.out_channels, # For i, f, g, o gates
            kernel_size=self.kernel_size,
            padding=self.padding
        )

    def forward(self, x_curr: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor,
                pooled_injection: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_curr (torch.Tensor): Current input to the ConvLSTM cell. Shape (B, in_channels, H, W).
                                   This will be i_t (and possibly top_down input) + pooled_injection.
            h_prev (torch.Tensor): Previous hidden state. Shape (B, out_channels, H, W).
            c_prev (torch.Tensor): Previous cell state. Shape (B, out_channels, H, W).
            pooled_injection (Optional[torch.Tensor]): Spatially pooled and reshaped output from prior tick.
                                                      Shape (B, out_channels, H, W). Added to x_curr.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (h_curr, c_curr) - Current hidden and cell states.
        """
        if pooled_injection is not None:
            # "element-wise addition to the input of the convolutional operations within the ConvLSTM cell"
            x_curr = x_curr + pooled_injection

        combined_input = torch.cat([x_curr, h_prev], dim=1) # Concatenate along channel dimension
        combined_conv = self.conv_gates(combined_input)

        # Split into gates
        i, f, g, o = torch.split(combined_conv, self.out_channels, dim=1)

        # Apply activations
        i = torch.sigmoid(i) # Input gate
        f = torch.sigmoid(f) # Forget gate
        g = torch.tanh(g)    # Cell gate (candidate)
        o = torch.sigmoid(o) # Output gate

        # Compute current cell state
        c_curr = f * c_prev + i * g
        # Compute current hidden state
        h_curr = o * torch.tanh(c_curr)

        return h_curr, c_curr


class _ConvLSTMStack(nn.Module):
    """
    Manages D ConvLSTMCell instances and orchestrates the recurrent computations
    across N internal ticks per environment step, including skip connections
    and Pool-and-Inject mechanism.
    """
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.D: int = config.get("agent.drc_agent.D", 3) # Number of ConvLSTM layers
        self.N: int = config.get("agent.drc_agent.N", 3) # Number of internal ticks

        self.convlstm_channels: int = config.get("agent.drc_agent.convlstm_channels", 32)
        self.convlstm_kernel_size: int = config.get("agent.drc_agent.convlstm_kernel_size", 3)
        self.convlstm_padding: int = config.get("agent.drc_agent.convlstm_padding", 1)

        # Create D ConvLSTM cells
        self.convlstm_cells = nn.ModuleList([
            _ConvLSTMCell(
                in_channels=self.convlstm_channels, # i_t + top-down for layer 0, i_t for others
                out_channels=self.convlstm_channels,
                kernel_size=self.convlstm_kernel_size,
                padding=self.convlstm_padding
            ) for _ in range(self.D)
        ])
        
        # Pool-and-Inject mechanism for each layer
        self.pool_and_inject_linears = nn.ModuleList()
        self.pool_and_inject_biases = nn.ParameterList() # Use ParameterList for individual bias tensors

        for _ in range(self.D):
            # input to linear layer: concatenation of mean_pool(h) and max_pool(h)
            # each pool is G_d channels, so input is 2 * G_d
            self.pool_and_inject_linears.append(
                nn.Linear(2 * self.convlstm_channels, self.convlstm_channels * 8 * 8) # Reshape output to G_d * H * W
            )
            self.pool_and_inject_biases.append(
                nn.Parameter(torch.randn(self.convlstm_channels * 8 * 8)) # Initialize bias
            )

    def forward(self, i_t: torch.Tensor,
                h_prev_states_list: List[torch.Tensor],
                c_prev_states_list: List[torch.Tensor]) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """
        Performs N internal ticks of computation across D ConvLSTM layers.

        Args:
            i_t (torch.Tensor): Encoded observation from the _ConvEncoder. Shape (B, G0, H0, W0).
            h_prev_states_list (List[torch.Tensor]): List of hidden states (one for each layer D) from the previous
                                                 environment step. Shape D x (B, G_d, H_d, W_d).
            c_prev_states_list (List[torch.Tensor]): List of cell states (one for each layer D) from the previous
                                                 environment step. Shape D x (B, G_d, H_d, W_d).

        Returns:
            Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
                - _h_states_per_tick_layer: List of lists, storing h_states for each (tick_idx, layer_idx).
                                            Outer list size (N+1), inner list size D.
                - _c_states_per_tick_layer: List of lists, storing c_states for each (tick_idx, layer_idx).
                                            Outer list size (N+1), inner list size D.
        """
        batch_size, _, grid_h, grid_w = i_t.shape

        # Initialize storage for states across all ticks and layers.
        # tick_idx=0 stores the states from the previous environment step (s_{t-1})
        self._h_states_per_tick_layer: List[List[torch.Tensor]] = [h_prev_states_list]
        self._c_states_per_tick_layer: List[List[torch.Tensor]] = [c_prev_states_list]

        # Loop N internal ticks
        for n in range(1, self.N + 1):
            h_tick_n: List[torch.Tensor] = []
            c_tick_n: List[torch.Tensor] = []
            top_down_input: Optional[torch.Tensor] = None

            # Top-Down Skip Connections: output of the final ConvLSTM unit from the prior tick
            # is provided as an additional input to the bottom (1st) ConvLSTM unit on the next tick.
            if n > 0: # For n=1..N, use h from tick (n-1), final layer (D-1)
                top_down_input = self._h_states_per_tick_layer[n-1][self.D-1]

            # Loop D ConvLSTM layers
            for d in range(self.D):
                h_prev: torch.Tensor = self._h_states_per_tick_layer[n-1][d]
                c_prev: torch.Tensor = self._c_states_per_tick_layer[n-1][d]

                # Pool-and-Inject mechanism: spatially pooled version of h_prev from prior tick (n-1)
                mean_pooled = h_prev.mean(dim=[-2, -1]) # MeanPool_H,W (h) -> (B, G_d)
                max_pooled = h_prev.max(dim=-1)[0].max(dim=-1)[0] # MaxPool_H,W (h) -> (B, G_d)
                # Concatenate mean and max pooled vectors
                m_d = torch.cat([mean_pooled, max_pooled], dim=1) # (B, 2 * G_d)

                # Affine transformation and reshape
                p_hat_d = self.pool_and_inject_linears[d](m_d) + self.pool_and_inject_biases[d]
                # Reshape to (B, G_d, H_d, W_d)
                pooled_injection = p_hat_d.reshape(batch_size, self.convlstm_channels, grid_h, grid_w)

                # Construct x_curr for current ConvLSTMCell
                x_curr_input = i_t # Bottom-Up Skip Connections: i_t is input to all ConvLSTMs

                if d == 0 and top_down_input is not None: # Apply Top-Down to first layer
                    x_curr_input = x_curr_input + top_down_input # Summing as often done for skip connections

                # Call ConvLSTM cell for current layer and tick
                h_curr, c_curr = self.convlstm_cells[d](x_curr=x_curr_input,
                                                        h_prev=h_prev,
                                                        c_prev=c_prev,
                                                        pooled_injection=pooled_injection)
                h_tick_n.append(h_curr)
                c_tick_n.append(c_curr)
            
            self._h_states_per_tick_layer.append(h_tick_n)
            self._c_states_per_tick_layer.append(c_tick_n)
        
        return self._h_states_per_tick_layer, self._c_states_per_tick_layer


class _PolicyValueHeads(nn.Module):
    """
    Policy and Value heads, accepting the combined feature vector o_t.
    """
    def __init__(self, config: Config, feature_dim: int, action_space: int) -> None:
        super().__init__()
        mlp_hidden_dim: int = config.get("agent.drc_agent.policy_value_head.mlp_hidden_dim", 256)

        # First affine transformation followed by ReLU
        self.affine_transform = nn.Linear(feature_dim, mlp_hidden_dim)
        self.activation = nn.ReLU()

        # Policy head
        self.policy_head = nn.Linear(mlp_hidden_dim, action_space)

        # Value head
        self.value_head = nn.Linear(mlp_hidden_dim, 1)

    def forward(self, concat_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            concat_features (torch.Tensor): Concatenated and flattened features (h_D_N and i_t).
                                           Shape (B, feature_dim).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (policy_logits, value_estimate).
        """
        o_t = self.activation(self.affine_transform(concat_features))
        policy_logits = self.policy_head(o_t)
        value_estimate = self.value_head(o_t)
        return policy_logits, value_estimate


class DRCAgent(BaseAgentModel):
    """
    Deep Repeated ConvLSTM (DRC) agent model.
    """
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        
        self.D: int = config.get("agent.drc_agent.D", 3) # Number of ConvLSTM layers
        self.N: int = config.get("agent.drc_agent.N", 3) # Number of internal ticks
        self.convlstm_channels: int = config.get("agent.drc_agent.convlstm_channels", 32)
        
        # Initialize previous (h, c) states to zeros. These represent s_{t-1}.
        # Will be (D) x (B, G_d, H_d, W_d)
        self._prev_h_states: List[torch.Tensor] = self._init_hidden_states(self.device, self.convlstm_channels, self.grid_height, self.grid_width)
        self._prev_c_states: List[torch.Tensor] = self._init_hidden_states(self.device, self.convlstm_channels, self.grid_height, self.grid_width)
        
        # Internal storage for all h and c states across ticks and layers within the current step (s_t,n,d)
        # These will be lists of lists: [tick_idx][layer_idx]
        self._h_states_per_tick_layer: List[List[torch.Tensor]] = []
        self._c_states_per_tick_layer: List[List[torch.Tensor]] = []

    def _init_hidden_states(self, device: torch.device, channels: int, H: int, W: int) -> List[torch.Tensor]:
        """Helper to initialize a list of zero tensors for D layers."""
        return [torch.zeros(1, channels, H, W, device=device) for _ in range(self.D)]

    def _build_model_architecture(self) -> None:
        """
        Builds the DRC agent's specific architecture components.
        """
        self.encoder = _ConvEncoder(self.config).to(self.device)
        self.convlstm_stack = _ConvLSTMStack(self.config).to(self.device)

        # Calculate feature dimension for PolicyValueHeads
        # h_t,N^D is (B, convlstm_channels, H, W)
        # i_t is (B, encoder_out_channels, H, W)
        # After flattening and concatenating:
        # convlstm_channels (G_d) and encoder_out_channels (G0) are both 32.
        feature_dim: int = (self.grid_height * self.grid_width * self.convlstm_channels) + \
                           (self.grid_height * self.grid_width * self.config.get("agent.drc_agent.encoder.out_channels", 32))
        
        self.policy_value_heads = _PolicyValueHeads(self.config, feature_dim, self.action_space_size).to(self.device)
        
        # Assign policy and value heads to instance attributes for BaseAgentModel compliance
        self._policy_head = self.policy_value_heads.policy_head
        self._value_head = self.policy_value_heads.value_head


    def forward(self, obs: torch.Tensor,
                hidden_state: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None) -> \
                Tuple[torch.Tensor, torch.Tensor, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """
        Forward pass for the DRCAgent.

        Args:
            obs (torch.Tensor): A batch of observations. Expected shape (B, C, H, W).
            hidden_state (Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]]):
                          (h_prev_states_list, c_prev_states_list) from the previous environment step.
                          If None, uses internal _prev_h_states and _prev_c_states (initialized to zeros).
        Returns:
            Tuple[torch.Tensor, torch.Tensor, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
                - policy_logits (torch.Tensor): Logits for actions. Shape (B, action_space_size).
                - value_estimate (torch.Tensor): Estimated state value. Shape (B, 1).
                - new_hidden_state (Tuple[List[torch.Tensor], List[torch.Tensor]]):
                                    (h_final_states_list, c_final_states_list) for the next environment step.
        """
        batch_size = obs.shape[0]

        # Ensure internal _prev_h_states and _prev_c_states match batch_size if they were only (1,...)
        if self._prev_h_states[0].shape[0] != batch_size:
            self._prev_h_states = [h.repeat(batch_size, 1, 1, 1) for h in self._prev_h_states]
            self._prev_c_states = [c.repeat(batch_size, 1, 1, 1) for c in self._prev_c_states]

        # Use provided hidden_state or internal _prev_h_states/_prev_c_states
        h_prev_states_input: List[torch.Tensor]
        c_prev_states_input: List[torch.Tensor]
        if hidden_state is None:
            h_prev_states_input = self._prev_h_states
            c_prev_states_input = self._prev_c_states
        else:
            h_prev_states_input, c_prev_states_input = hidden_state
            # Ensure they are on the correct device if provided externally
            h_prev_states_input = [h.to(self.device) for h in h_prev_states_input]
            c_prev_states_input = [c.to(self.device) for c in c_prev_states_input]


        # 1. Encoder pass
        i_t = self.encoder(obs) # (B, G0, H0, W0)

        # 2. ConvLSTM Stack pass
        h_all_ticks_layers, c_all_ticks_layers = self.convlstm_stack(i_t, h_prev_states_input, c_prev_states_input)
        
        # Update internal storage for `get_cell_state` and `set_cell_state`
        self._h_states_per_tick_layer = h_all_ticks_layers
        self._c_states_per_tick_layer = c_all_ticks_layers

        # 3. Extract output for Policy/Value Heads
        # Final hidden state from the last layer (D-1) of the last tick (N)
        h_D_N = self._h_states_per_tick_layer[self.N][self.D-1] # (B, G_d, H_d, W_d)

        # Flatten h_D_N and i_t
        h_D_N_flat = h_D_N.view(batch_size, -1)
        i_t_flat = i_t.view(batch_size, -1)

        # Concatenate flattened features for the policy/value heads input
        concat_features = torch.cat([h_D_N_flat, i_t_flat], dim=1) # (B, feature_dim)

        # 4. Policy and Value Heads
        policy_logits, value_estimate = self.policy_value_heads(concat_features)

        # 5. Update _prev_h_states and _prev_c_states for the next environment step
        # These store the final states (s_t) to be used as s_{t-1} for the next step.
        self._prev_h_states = self._h_states_per_tick_layer[self.N]
        self._prev_c_states = self._c_states_per_tick_layer[self.N]
        
        new_hidden_state = (self._prev_h_states, self._prev_c_states)

        return policy_logits, value_estimate, new_hidden_state


    def get_cell_state(self, layer_idx: int, tick_idx: int = -1) -> torch.Tensor:
        """
        Retrieves the cell state (g_t^d) for a specific layer and tick.

        Args:
            layer_idx (int): The 0-indexed ConvLSTM layer number (0 to D-1).
            tick_idx (int, optional): The 0-indexed internal computational tick (0 to N).
                                      0 refers to states from s_{t-1}. -1 refers to the final tick N.
                                      Defaults to -1.

        Returns:
            torch.Tensor: The cell state tensor (H, W, G_d). Note: This extracts a single item from batch.
        """
        if not (0 <= layer_idx < self.D):
            raise ValueError(f"layer_idx {layer_idx} out of bounds for D={self.D} layers.")
        
        actual_tick_idx = tick_idx if tick_idx != -1 else self.N
        if not (0 <= actual_tick_idx <= self.N):
            raise ValueError(f"tick_idx {actual_tick_idx} out of bounds for N={self.N} ticks (0 to N).")

        # Return the cell state, unsqueezing batch dimension if it's (1, ...)
        return self._c_states_per_tick_layer[actual_tick_idx][layer_idx].squeeze(0)


    def set_cell_state(self, layer_idx: int, tick_idx: int, new_state: torch.Tensor) -> None:
        """
        Modifies the cell state (g_t^d) for a specific layer and tick.
        This directly modifies the internally stored states that the ConvLSTMStack uses.

        Args:
            layer_idx (int): The 0-indexed ConvLSTM layer number (0 to D-1).
            tick_idx (int): The 0-indexed internal computational tick (0 to N).
                            0 refers to states from s_{t-1}. -1 refers to the final tick N.
            new_state (torch.Tensor): The new cell state tensor to set. Shape (H, W, G_d).
        """
        if not (0 <= layer_idx < self.D):
            raise ValueError(f"layer_idx {layer_idx} out of bounds for D={self.D} layers.")
        
        actual_tick_idx = tick_idx if tick_idx != -1 else self.N
        if not (0 <= actual_tick_idx <= self.N):
            raise ValueError(f"tick_idx {actual_tick_idx} out of bounds for N={self.N} ticks (0 to N).")

        # The new_state comes as (H,W,G), need to convert to (1,G,H,W) to match internal format
        # And ensure it's on the correct device
        formatted_new_state = new_state.permute(2, 0, 1).unsqueeze(0).to(self.device)

        if formatted_new_state.shape != self._c_states_per_tick_layer[actual_tick_idx][layer_idx].shape:
            raise ValueError(f"Shape mismatch for new_state at layer {layer_idx}, tick {actual_tick_idx}. "
                             f"Expected {self._c_states_per_tick_layer[actual_tick_idx][layer_idx].shape}, got {formatted_new_state.shape}.")

        # Directly update the stored state.
        # This assumes that the convlstm_stack's forward pass accesses these lists by reference,
        # ensuring that this modification impacts subsequent computations within the same forward call.
        self._c_states_per_tick_layer[actual_tick_idx][layer_idx] = formatted_new_state

        # For consistency, also update the _prev_c_states if this is the final tick of the step
        if actual_tick_idx == self.N:
            self._prev_c_states[layer_idx] = formatted_new_state
        
        # Note: Intervening on 'c' state only implies 'h' state would be re-calculated based on this new 'c'
        # if the ConvLSTMCell logic correctly updates 'h' from 'c'. Otherwise, 'h' might also need intervention.
        # Paper implies intervention on cell state `g_x,y` only.


if __name__ == '__main__':
    print("--- Testing DRCAgent ---")

    # Dummy Config for testing
    dummy_config_data = {
        'experiment_name': 'test_drc_agent',
        'environment': {
            'name': 'Sokoban',
            'sokoban': {
                'grid_size': [8, 8],
                'observation_channels': 7,
                'action_space_size': 5, # Up, Down, Left, Right, No-op
            }
        },
        'agent': {
            'drc_agent': {
                'D': 3,
                'N': 3,
                'convlstm_channels': 32,
                'convlstm_kernel_size': 3,
                'convlstm_padding': 1,
                'encoder': {
                    'type': "Conv2d",
                    'out_channels': 32,
                    'kernel_size': 3,
                    'padding': 1,
                    'activation': "ReLU"
                },
                'policy_value_head': {
                    'mlp_hidden_dim': 256
                }
            }
        }
    }
    dummy_config = Config(dummy_config_data)

    # Instantiate DRCAgent
    drc_agent = DRCAgent(dummy_config)
    print(f"DRCAgent initialized on device: {drc_agent.device}")
    print(f"Number of ConvLSTM layers (D): {drc_agent.D}")
    print(f"Number of internal ticks (N): {drc_agent.N}")
    print(f"ConvLSTM channels: {drc_agent.convlstm_channels}")

    # Create dummy observation
    dummy_obs_np = np.random.rand(8, 8, 7).astype(np.float32)
    dummy_obs_tensor = torch.from_numpy(dummy_obs_np).permute(2, 0, 1).unsqueeze(0).to(drc_agent.device)
    print(f"\nDummy observation shape: {dummy_obs_tensor.shape}")

    # Test forward pass without initial hidden state
    policy_logits, value_estimate, next_hidden_state = drc_agent.forward(dummy_obs_tensor)
    print(f"\nPolicy logits shape: {policy_logits.shape}")
    print(f"Value estimate shape: {value_estimate.shape}")
    print(f"Next hidden state (h, c) - list of D layers. First h state shape: {next_hidden_state[0][0].shape}")
    print(f"Next hidden state (h, c) - list of D layers. First c state shape: {next_hidden_state[1][0].shape}")

    # Test forward pass with provided hidden state
    policy_logits_2, value_estimate_2, next_hidden_state_2 = drc_agent.forward(dummy_obs_tensor, next_hidden_state)
    print(f"\nSecond forward pass using previous hidden state.")
    print(f"Policy logits shape: {policy_logits_2.shape}")

    # Test get_cell_state
    print("\n--- Testing get_cell_state ---")
    
    # Get initial (t-1) state for layer 0
    c_state_t_minus_1_l0 = drc_agent.get_cell_state(layer_idx=0, tick_idx=0)
    print(f"Cell state (tick 0, layer 0) shape: {c_state_t_minus_1_l0.shape}")
    
    # Get final (t, N) state for layer 2
    c_state_t_N_l2 = drc_agent.get_cell_state(layer_idx=2, tick_idx=-1)
    print(f"Cell state (final tick N, layer 2) shape: {c_state_t_N_l2.shape}")
    
    # Test invalid layer_idx
    try:
        drc_agent.get_cell_state(layer_idx=drc_agent.D)
    except ValueError as e:
        print(f"Caught expected error for invalid layer_idx: {e}")

    # Test invalid tick_idx
    try:
        drc_agent.get_cell_state(layer_idx=0, tick_idx=drc_agent.N + 1)
    except ValueError as e:
        print(f"Caught expected error for invalid tick_idx: {e}")

    # Test set_cell_state
    print("\n--- Testing set_cell_state ---")
    original_c_state_t_N_l2 = drc_agent.get_cell_state(layer_idx=2, tick_idx=-1).clone()
    new_c_state = torch.rand_like(original_c_state_t_N_l2) # Create a random tensor of same shape
    drc_agent.set_cell_state(layer_idx=2, tick_idx=-1, new_state=new_c_state)
    
    retrieved_new_c_state = drc_agent.get_cell_state(layer_idx=2, tick_idx=-1)
    # Check if the state was indeed updated
    assert torch.equal(retrieved_new_c_state, new_c_state), "set_cell_state did not update correctly!"
    print("set_cell_state updated cell state successfully.")
    
    # Verify _prev_c_states also updated if it was the final tick
    assert torch.equal(drc_agent._prev_c_states[2].squeeze(0), new_c_state), "_prev_c_states not updated correctly!"
    print("_prev_c_states also updated successfully for final tick.")

    # Test act method (inherited from BaseAgentModel, but uses DRCAgent's forward)
    print("\n--- Testing act method ---")
    action, new_hidden, logit_output, value_output = drc_agent.act(dummy_obs_np, greedy=True)
    print(f"Action (greedy): {action}")
    print(f"Logits shape from act: {logit_output.shape}")
    print(f"Value from act: {value_output.item()}")

    action_sampled, _, _, _ = drc_agent.act(dummy_obs_np, greedy=False)
    print(f"Action (sampled): {action_sampled}")

    print("\n--- DRCAgent testing complete ---")

