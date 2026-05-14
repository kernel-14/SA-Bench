import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
from typing import Any, Tuple, Optional, List, Dict

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
        def forward(self, obs: torch.Tensor, hidden_state: Any = None) -> Tuple[torch.Tensor, Any]: return torch.randn(1, 256), None
        def get_cell_state(self, layer_idx: int, tick_idx: int = -1) -> torch.Tensor: return torch.randn(self.grid_height, self.grid_width, 32)
        def set_cell_state(self, layer_idx: int, tick_idx: int, new_state: torch.Tensor) -> None: pass
        def get_action_logits(self, model_output_vector: torch.Tensor) -> torch.Tensor: return nn.Linear(model_output_vector.shape[-1], self.action_space_size)(model_output_vector)
        def get_value_estimate(self, model_output_vector: torch.Tensor) -> torch.Tensor: return nn.Linear(model_output_vector.shape[-1], 1)(model_output_vector)
        def act(self, obs: np.ndarray, hidden_state: Any = None, greedy: bool = True) -> Tuple[int, Any, torch.Tensor, torch.Tensor]:
            # Dummy act method
            dummy_obs_tensor = torch.from_numpy(obs).float().to(self.device).permute(2, 0, 1).unsqueeze(0)
            model_output_vector, _ = self.forward(dummy_obs_tensor, hidden_state)
            policy_logits = self.get_action_logits(model_output_vector).squeeze(0)
            value_estimate = self.get_value_estimate(model_output_vector).squeeze(0)
            action = torch.argmax(policy_logits, dim=-1).item() if greedy else dist.Categorical(logits=policy_logits).sample().item()
            return action, None, policy_logits, value_estimate
    print("Warning: Could not import 'BaseAgentModel'. Using a dummy BaseAgentModel class.")


class _ResBlock(nn.Module):
    """
    A single simplified residual block as described in Appendix G of the paper.
    Input -> Conv -> LayerNorm -> ReLU -> Conv -> LayerNorm -> + (Input) -> ReLU -> Output
    """
    def __init__(self, channels: int, kernel_size: int, padding: int,
                 grid_height: int, grid_width: int, activation_str: str = 'ReLU') -> None:
        """
        Initializes a residual block.

        Args:
            channels (int): The number of input and output channels for the convolutional layers.
            kernel_size (int): The kernel size for the convolutional layers.
            padding (int): The padding for the convolutional layers to maintain spatial dimensions.
            grid_height (int): The height of the feature map.
            grid_width (int): The width of the feature map.
            activation_str (str): Name of the activation function ('ReLU').
        """
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        # LayerNorm applied over (channels, height, width) dimensions
        self.norm1 = nn.LayerNorm([channels, grid_height, grid_width])

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.LayerNorm([channels, grid_height, grid_width])

        if activation_str == "ReLU":
            self.activation = nn.ReLU()
        elif activation_str == "Identity": # For cases where no activation is desired
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation function for _ResBlock: {activation_str}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the residual block.

        Args:
            x (torch.Tensor): Input tensor. Shape (B, channels, H, W).

        Returns:
            torch.Tensor: Output tensor. Shape (B, channels, H, W).
        """
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += identity  # Skip connection: add original input
        out = self.activation(out)  # Final ReLU

        return out


class _MLPPolicyValueHeads(nn.Module):
    """
    MLP-based policy and value heads for the ResNet agent.
    Takes the flattened output of the ResNet backbone and produces action logits and a value estimate.
    """
    def __init__(self, in_features: int, mlp_hidden_dim: int, action_space_size: int) -> None:
        """
        Initializes the MLP policy and value heads.

        Args:
            in_features (int): Number of input features (flattened output of ResNet backbone).
            mlp_hidden_dim (int): Dimension of the hidden layer in the MLP.
            action_space_size (int): Number of possible actions in the environment.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, mlp_hidden_dim),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(mlp_hidden_dim, action_space_size)
        self.value_head = nn.Linear(mlp_hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass for the policy and value heads.

        Args:
            x (torch.Tensor): Flattened input tensor from the ResNet backbone. Shape (B, in_features).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - policy_logits (torch.Tensor): Logits for each action. Shape (B, action_space_size).
                - value_estimate (torch.Tensor): Estimated value of the state. Shape (B, 1).
        """
        hidden = self.mlp(x)
        policy_logits = self.policy_head(hidden)
        value_estimate = self.value_head(hidden)
        return policy_logits, value_estimate


class ResNetAgent(BaseAgentModel):
    """
    Implements the full ResNet agent model, including the backbone and policy/value heads,
    and methods for interaction and interpretability.
    """
    def __init__(self, config: Config) -> None:
        """
        Initializes the ResNet agent.

        Args:
            config (Config): Configuration object containing agent and environment settings.
        """
        super().__init__(config)  # This will call _build_model_architecture

        # Store internal activations for get_cell_state for probing.
        # Key: block_idx, Value: tensor (B, C, H, W)
        self._cached_activations: Dict[int, torch.Tensor] = {}

        # Store intervention vectors for set_cell_state.
        # Key: block_idx, Value: tensor (C, H, W) to be added
        self._intervention_vectors: Dict[int, torch.Tensor] = {}

    def _build_model_architecture(self) -> None:
        """
        Builds the specific ResNet architecture components based on configuration.
        """
        self.num_residual_blocks: int = self.config.get("agent.resnet_agent.num_residual_blocks", 24)
        self.block_channels: int = self.config.get("agent.resnet_agent.block_channels", 32)
        mlp_output_dim: int = self.config.get("agent.resnet_agent.mlp_output_dim", 256)
        
        # Initial convolutional layer to map observation channels to block channels
        # (e.g., 7 channels for Sokoban observation to 32 block channels)
        # Kernel size and padding for feature map spatial dimension preservation
        self.initial_conv = nn.Conv2d(
            in_channels=self.obs_channels,
            out_channels=self.block_channels,
            kernel_size=3,
            padding=1 # Maintain H, W for 8x8 grid
        )

        self.blocks = nn.ModuleList([
            _ResBlock(
                channels=self.block_channels,
                kernel_size=3, # Assuming 3x3 kernels for res blocks
                padding=1, # Assuming padding to maintain spatial dimensions
                grid_height=self.grid_height,
                grid_width=self.grid_width,
                activation_str='ReLU' # Default activation
            ) for _ in range(self.num_residual_blocks)
        ])

        # Calculate input features for MLP heads
        mlp_in_features: int = self.block_channels * self.grid_height * self.grid_width
        self.mlp_heads = _MLPPolicyValueHeads(mlp_in_features, mlp_output_dim, self.action_space_size)

        # Assign policy and value heads to instance attributes for BaseAgentModel compliance
        self._policy_head = self.mlp_heads.policy_head
        self._value_head = self.mlp_heads.value_head

    def forward(self, obs: torch.Tensor, hidden_state: Any = None) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Forward pass for the ResNetAgent.

        Args:
            obs (torch.Tensor): A batch of observations. Expected shape (B, C, H, W).
            hidden_state (Any, optional): Ignored for ResNet as it is a feedforward network. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Any]:
                - policy_logits (torch.Tensor): Logits for actions. Shape (B, action_space_size).
                - value_estimate (torch.Tensor): Estimated state value. Shape (B, 1).
                - None: No recurrent hidden state for a feedforward agent.
        """
        batch_size = obs.shape[0]
        
        # Clear cached activations at the beginning of each forward pass
        self._cached_activations.clear()

        # Apply initial convolution
        x = self.initial_conv(obs) # (B, block_channels, H, W)

        # Pass through residual blocks
        for block_idx, block in enumerate(self.blocks):
            x = block(x)
            
            # Apply intervention if configured for this block
            if block_idx in self._intervention_vectors:
                # Intervention vector is stored as (C, H, W), unsqueeze to (1, C, H, W) to add to batch
                x = x + self._intervention_vectors[block_idx].unsqueeze(0).to(self.device)
            
            # Cache the output of the block (after potential intervention)
            self._cached_activations[block_idx] = x.clone().detach() # Detach to prevent gradients flowing to probes

        # Flatten the final output for the MLP heads
        x_flattened = x.view(batch_size, -1) # (B, block_channels * H * W)

        # Policy and Value Heads
        policy_logits, value_estimate = self.mlp_heads(x_flattened)

        return policy_logits, value_estimate, None # ResNet is feedforward, no hidden state

    def get_cell_state(self, layer_idx: int, tick_idx: int = -1) -> torch.Tensor:
        """
        Retrieves the activations after a specific residual block for probing.

        Args:
            layer_idx (int): The 0-indexed residual block number (0 to num_residual_blocks - 1).
            tick_idx (int, optional): Ignored for ResNet as it has no internal ticks. Defaults to -1.

        Returns:
            torch.Tensor: The activations tensor after the specified block. Shape (H, W, C).
                          Note: This extracts a single item from batch (assumes batch_size=1).

        Raises:
            ValueError: If layer_idx is out of bounds or activations are not cached.
        """
        if not (0 <= layer_idx < self.num_residual_blocks):
            raise ValueError(f"layer_idx {layer_idx} out of bounds for {self.num_residual_blocks} blocks.")
        
        # Retrieve from cached activations, assuming single-batch (squeeze(0))
        activations = self._cached_activations.get(layer_idx)
        if activations is None:
            raise ValueError(f"Activations for layer {layer_idx} not found. Ensure forward pass has been run.")
        
        # Convert from (B, C, H, W) to (H, W, C) for consistency with design (probe input expectation)
        return activations.squeeze(0).permute(1, 2, 0)

    def set_cell_state(self, layer_idx: int, tick_idx: int, new_state: torch.Tensor) -> None:
        """
        Stores an intervention vector to be added to the activations of a specific
        residual block during the next forward pass.

        Args:
            layer_idx (int): The 0-indexed residual block number (0 to num_residual_blocks - 1).
            tick_idx (int): Ignored for ResNet.
            new_state (torch.Tensor): The new tensor (intervention vector) to add. Expected shape (H, W, C).

        Raises:
            ValueError: If layer_idx is out of bounds or shape mismatch.
        """
        if not (0 <= layer_idx < self.num_residual_blocks):
            raise ValueError(f"layer_idx {layer_idx} out of bounds for {self.num_residual_blocks} blocks.")
        
        # Convert new_state from (H, W, C) to (C, H, W) for internal storage and element-wise addition
        # It's expected to be a single (non-batched) tensor.
        formatted_new_state = new_state.permute(2, 0, 1).to(self.device)

        # Check shape consistency (assuming square grid and block_channels)
        expected_shape = (self.block_channels, self.grid_height, self.grid_width)
        if formatted_new_state.shape != expected_shape:
            raise ValueError(f"Shape mismatch for new_state at layer {layer_idx}. "
                             f"Expected {expected_shape}, got {formatted_new_state.shape}.")

        self._intervention_vectors[layer_idx] = formatted_new_state

    # The get_action_logits, get_value_estimate, and act methods are inherited and
    # use the outputs of the forward pass consistently with BaseAgentModel.


if __name__ == '__main__':
    print("--- Testing ResNetAgent ---")

    # Dummy Config for testing
    dummy_config_data = {
        'experiment_name': 'test_resnet_agent',
        'environment': {
            'name': 'Sokoban',
            'sokoban': {
                'grid_size': [8, 8],
                'observation_channels': 7,
                'action_space_size': 5, # Up, Down, Left, Right, No-op
            }
        },
        'agent': {
            'resnet_agent': {
                'num_residual_blocks': 3, # Use a small number for testing
                'block_channels': 32,
                'mlp_output_dim': 256
            }
        }
    }
    dummy_config = Config(dummy_config_data)

    # Instantiate ResNetAgent
    resnet_agent = ResNetAgent(dummy_config)
    print(f"ResNetAgent initialized on device: {resnet_agent.device}")
    print(f"Number of residual blocks: {resnet_agent.num_residual_blocks}")
    print(f"Block channels: {resnet_agent.block_channels}")
    print(f"Observation shape: {resnet_agent.observation_space_shape}")
    print(f"Action space size: {resnet_agent.action_space_size}")


    # Create dummy observation (e.g., a single Sokoban 8x8x7 observation)
    dummy_obs_np = np.random.rand(*resnet_agent.observation_space_shape).astype(np.float32)
    
    # Test act method (which calls forward internally)
    print("\n--- Testing act method ---")
    action, hidden_state_output, policy_logits_tensor, value_estimate_tensor = resnet_agent.act(dummy_obs_np, greedy=True)
    print(f"Action (greedy): {action}")
    print(f"Hidden state output (should be None): {hidden_state_output}")
    print(f"Policy logits shape: {policy_logits_tensor.shape}")
    print(f"Value estimate shape: {value_estimate_tensor.shape}")

    action_sampled, _, _, _ = resnet_agent.act(dummy_obs_np, greedy=False)
    print(f"Action (sampled): {action_sampled}")

    # Test get_cell_state
    print("\n--- Testing get_cell_state ---")
    
    # After act (forward) has been run, _cached_activations should be populated
    # Retrieve activations from the last block (index 2 for num_residual_blocks=3)
    try:
        activations_block2 = resnet_agent.get_cell_state(layer_idx=2)
        print(f"Activations from block 2 shape (H, W, C): {activations_block2.shape}")
        expected_shape = (resnet_agent.grid_height, resnet_agent.grid_width, resnet_agent.block_channels)
        assert activations_block2.shape == expected_shape, "Shape mismatch for retrieved activations."
        print(f"Activations from block 2 head: {activations_block2.flatten()[:5]}")
    except ValueError as e:
        print(f"Error getting activations: {e}")

    # Test invalid layer_idx for get_cell_state
    try:
        resnet_agent.get_cell_state(layer_idx=resnet_agent.num_residual_blocks)
    except ValueError as e:
        print(f"Caught expected error for invalid layer_idx: {e}")

    # Test set_cell_state and its effect in forward pass
    print("\n--- Testing set_cell_state and intervention ---")
    intervention_layer_idx = 1 # Intervene on the second block
    
    # Create a dummy intervention vector (H, W, C)
    dummy_intervention_vector_np = np.ones(expected_shape, dtype=np.float32) * 0.1
    dummy_intervention_vector = torch.from_numpy(dummy_intervention_vector_np)
    
    # Store the intervention vector
    resnet_agent.set_cell_state(layer_idx=intervention_layer_idx, tick_idx=-1, new_state=dummy_intervention_vector)
    print(f"Intervention vector stored for layer {intervention_layer_idx}.")

    # Run forward again (via act) to trigger the intervention
    print("\n--- Running forward with intervention ---")
    # First, get original activations BEFORE intervention
    _ = resnet_agent.act(dummy_obs_np, greedy=True) # Populate cache
    original_activations_block1 = resnet_agent.get_cell_state(layer_idx=intervention_layer_idx)
    
    # Now set intervention and run forward
    resnet_agent.set_cell_state(layer_idx=intervention_layer_idx, tick_idx=-1, new_state=dummy_intervention_vector)
    _ = resnet_agent.act(dummy_obs_np, greedy=True) # Run to apply intervention
    
    intervened_activations_block1 = resnet_agent.get_cell_state(layer_idx=intervention_layer_idx)

    # Check if the activations changed by the intervention amount
    # Note: `get_cell_state` detaches and clones, so we compare numerical values
    # The `_cached_activations` are updated directly by forward with intervention,
    # so `get_cell_state` after intervention will show the intervened value.
    # To properly check, we need to compare the output of block 1 without intervention
    # to the output of block 1 with intervention.

    # Re-run a controlled test:
    # 1. Get activations without intervention
    resnet_agent._intervention_vectors.clear() # Ensure no previous intervention is active
    _ = resnet_agent.act(dummy_obs_np, greedy=True)
    no_intervention_activations = resnet_agent.get_cell_state(layer_idx=intervention_layer_idx).cpu().numpy()

    # 2. Get activations with intervention
    resnet_agent.set_cell_state(layer_idx=intervention_layer_idx, tick_idx=-1, new_state=dummy_intervention_vector)
    _ = resnet_agent.act(dummy_obs_np, greedy=True)
    with_intervention_activations = resnet_agent.get_cell_state(layer_idx=intervention_layer_idx).cpu().numpy()

    # The expected change is approximately the intervention vector itself,
    # because the intervention happens *after* the block's computation but *before* caching.
    # The identity mapping of the resblock is also affected.
    # So we're testing if the stored value was added to the block's output before caching.
    # Let's verify that the output is different, and the `_intervention_vectors` dict is cleared at the start of forward.
    # Wait, `_cached_activations` is cleared, but `_intervention_vectors` is *not* cleared by `forward`.
    # `_intervention_vectors` should persist until `set_cell_state` is called again or cleared externally.
    # This aligns with the paper's description "We repeat the ‘short-route’ intervention every step".

    # Let's adjust the test to check if the intervention vector was added.
    print(f"Difference (with - without): {np.mean(with_intervention_activations - no_intervention_activations)}")
    
    # The actual addition is `x = x + self._intervention_vectors[block_idx]`.
    # `x` is the output of `block(x)`. So the cached activation is `block(original_x) + intervention_vector`.
    # The `no_intervention_activations` is `block(original_x)`.
    # So, `with_intervention_activations` should be `no_intervention_activations + dummy_intervention_vector_np`.
    
    assert np.allclose(with_intervention_activations, no_intervention_activations + dummy_intervention_vector_np, atol=1e-5), \
        "Intervention did not correctly modify activations."
    print("Intervention successfully applied to activations in forward pass.")
    
    print("\n--- ResNetAgent testing complete ---")

