"""
model.py
Defines the MR.Q architecture including state encoder, state-action encoder, policy head, and value head.
Supports both vector and image observations as described in the MR.Q plan from the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Implements the MR.Q model, including state and state-action encoders, policy head, and value head.
    """

    def __init__(self, params: dict):
        """
        Initialize the MR.Q model with configurations.

        Args:
            params (dict): A dictionary of model-related hyperparameters from `config.yaml`.
        """
        super(Model, self).__init__()

        # Extract configuration parameters
        self.zs_dim = params.get("zs_dim", 512)
        self.zsa_dim = params.get("zsa_dim", 512)
        self.hidden_dim = params.get("hidden_dim", 512)
        self.activation_function = params.get("activation_function", "ELU")
        self.observation_shape = params.get("observation_shape", (84, 84))
        self.is_image_input = params.get("is_image_input", False)
        self.action_dim = params.get("action_dim", None)
        self.discrete_action_space = params.get("discrete_action_space", False)

        # Build encoders
        self.state_encoder = self.build_encoder(self.observation_shape, mode="state")
        self.state_action_encoder = self.build_encoder((self.zs_dim + self.action_dim,), mode="state_action")

        # Build heads
        self.policy_head = self.build_policy_head(input_dim=self.zs_dim, output_dim=self.action_dim)
        self.value_head = self.build_value_head(input_dim=self.zsa_dim)

        # Activation (defaults to ELU as per `config.yaml`)
        self.activ = nn.ELU()

    def build_encoder(self, input_shape: tuple, mode: str) -> nn.Module:
        """
        Build the encoder for either states or state-action pairs.

        Args:
            input_shape (tuple): Shape of the input (e.g., (84, 84) for images or feature vector length).
            mode (str): 'state' or 'state_action', specifying which encoder to create.

        Returns:
            nn.Module: The encoder network.
        """
        if mode == "state" and self.is_image_input:
            # CNN-based encoder for image inputs
            layers = [
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2),
                nn.ELU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2),
                nn.ELU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2),
                nn.ELU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
                nn.ELU(),
                nn.Flatten(),
                nn.Linear(1568, self.zs_dim),  # Assumes 84x84 input reduces to flattened 1568 after convolutions
                nn.LayerNorm(self.zs_dim),
                nn.ELU(),
            ]
        elif mode == "state":
            # MLP-based encoder for vector inputs
            layers = [
                nn.Linear(input_shape[0], self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ELU(),
                nn.Linear(self.hidden_dim, self.zs_dim),
                nn.LayerNorm(self.zs_dim),
            ]
        elif mode == "state_action":
            # MLP-based encoder for state-action inputs
            layers = [
                nn.Linear(input_shape[0], self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ELU(),
                nn.Linear(self.hidden_dim, self.zsa_dim),
                nn.LayerNorm(self.zsa_dim),
            ]
        else:
            raise ValueError(f"Unsupported encoder mode: {mode}")

        return nn.Sequential(*layers)

    def build_policy_head(self, input_dim: int, output_dim: int) -> nn.Module:
        """
        Build the policy head that outputs actions.

        Args:
            input_dim (int): Dimension of input embeddings (state embeddings).
            output_dim (int): Dimension of action space.

        Returns:
            nn.Module: The policy head network.
        """
        layers = [
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, output_dim),
        ]
        if self.discrete_action_space:
            # Gumbel-Softmax required for discrete actions
            layers.append(nn.Softmax(dim=-1))
        else:
            # Tanh activation for continuous actions
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def build_value_head(self, input_dim: int) -> nn.Module:
        """
        Build the value head that outputs scalar Q(s, a) values.

        Args:
            input_dim (int): Dimension of input embeddings (state-action embeddings).

        Returns:
            nn.Module: The value head network.
        """
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, 1),  # Single scalar output
        )

    def forward_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for state embeddings (z_s).

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: State embedding tensor (z_s).
        """
        return self.state_encoder(state)

    def forward_state_action(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for state-action embeddings (z_sa).

        Args:
            state (torch.Tensor): Input state tensor.
            action (torch.Tensor): Input action tensor.

        Returns:
            torch.Tensor: State-action embedding tensor (z_sa).
        """
        z_s = self.forward_state(state)
        z_a = self.activ(nn.Linear(self.action_dim, self.hidden_dim)(action))
        z_combined = torch.cat([z_s, z_a], dim=-1)
        return self.state_action_encoder(z_combined)
