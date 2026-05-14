import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from utils.common_utils import CNNEncoder, MLPBlock, ResNetEncoder, init_weights


# --- Helper for potential (B, H, W, C) to (B, C, H, W) conversion ---
def _permute_if_needed(x: torch.Tensor) -> torch.Tensor:
    """
    Permutes a tensor from (B, H, W, C) to (B, C, H, W) if it's 4D and
    the last dimension is 3 (assuming RGB image) and C is not already the first.
    Otherwise, returns the tensor as is.
    """
    if x.ndim == 4 and x.shape[-1] == 3 and x.shape[1] != 3: # Assuming RGB channel is last
        return x.permute(0, 3, 1, 2)
    return x


# --- ICM Curiosity Networks ---

class FeatureEncoder(nn.Module):
    """
    Maps a raw observation (state s) to a compressed, latent feature representation h(s).
    """

    def __init__(self, config: Config, state_dim: Union[int, Tuple[int, ...]], output_dim: int, pixel_based: bool):
        """
        Initializes the FeatureEncoder.

        Args:
            config (Config): Configuration object.
            state_dim (Union[int, Tuple[int, ...]]): Dimension of the state observation.
                                                      (e.g., int for state-based, Tuple[C,H,W] for pixel-based).
            output_dim (int): The dimension of the latent feature vector h(s).
            pixel_based (bool): True if observations are pixel-based.
        """
        super().__init__()
        self.config: Config = config
        self.pixel_based: bool = pixel_based
        self.output_dim: int = output_dim # Store it for potential external use

        if self.pixel_based:
            if not isinstance(state_dim, tuple) or len(state_dim) != 3:
                raise ValueError(f"For pixel_based=True, state_dim must be (C, H, W), got {state_dim}")
            
            # Use CNNEncoder for pixel observations
            self.encoder = CNNEncoder(
                input_shape=state_dim,
                output_dim=output_dim,
                num_filters=[32, 32, 32, 32], # Default values, can be configured in common_utils if needed
                kernel_sizes=[3, 3, 3, 3],
                strides=[2, 1, 1, 1],
                activation_fn_name="ReLU",
            )
        else:
            if not isinstance(state_dim, int):
                state_dim = state_dim[0] # Assume state_dim is (vector_dim,)
            # Use MLPBlock for state vector observations
            hidden_layers: int = config.get_hyperparam('relevance_function.icm_net.feature_encoder_hidden_layers')
            hidden_units: int = config.get_hyperparam('relevance_function.icm_net.feature_encoder_hidden_units')
            self.encoder = MLPBlock(
                input_dim=state_dim,
                output_dim=output_dim,
                hidden_units=hidden_units,
                num_hidden_layers=hidden_layers,
                activation_fn_name="ReLU",
            )
        self.apply(init_weights)
        self.to(self.config.get_hyperparam('experiment.device'))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the FeatureEncoder.

        Args:
            state (torch.Tensor): The input state observation.

        Returns:
            torch.Tensor: The latent feature vector h(s).
        """
        if self.pixel_based:
            state = _permute_if_needed(state) # Ensure (B, C, H, W)
        return self.encoder(state)


class ForwardDynamicsModel(nn.Module):
    """
    Predicts the latent representation of the next state h(s') given the current
    latent state h(s) and the action a.
    """

    def __init__(self, config: Config, latent_dim: int, action_dim: int):
        """
        Initializes the ForwardDynamicsModel.

        Args:
            config (Config): Configuration object.
            latent_dim (int): The dimension of the latent state features (output of FeatureEncoder).
            action_dim (int): The dimension of the action space.
        """
        super().__init__()
        self.config: Config = config

        input_dim: int = latent_dim + action_dim
        output_dim: int = latent_dim # Predicting the next latent state

        hidden_layers: int = config.get_hyperparam('relevance_function.icm_net.forward_model_hidden_layers')
        hidden_units: int = config.get_hyperparam('relevance_function.icm_net.forward_model_hidden_units')

        self.model = MLPBlock(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_units=hidden_units,
            num_hidden_layers=hidden_layers,
            activation_fn_name="ReLU",
        )
        self.apply(init_weights)
        self.to(self.config.get_hyperparam('experiment.device'))

    def forward(self, latent_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the ForwardDynamicsModel.

        Args:
            latent_state (torch.Tensor): The latent representation of the current state h(s).
            action (torch.Tensor): The action taken.

        Returns:
            torch.Tensor: The predicted latent representation of the next state h(s').
        """
        x = torch.cat([latent_state, action], dim=-1)
        return self.model(x)


# --- RND Curiosity Networks ---

class TargetNetwork(nn.Module):
    """
    A randomly initialized neural network that remains fixed during training.
    It computes a target feature representation for the next state s'.
    """

    def __init__(self, config: Config, state_dim: Union[int, Tuple[int, ...]], pixel_based: bool):
        """
        Initializes the TargetNetwork.

        Args:
            config (Config): Configuration object.
            state_dim (Union[int, Tuple[int, ...]]): Dimension of the next state observation s'.
            pixel_based (bool): True if observations are pixel-based.
        """
        super().__init__()
        self.config: Config = config
        self.pixel_based: bool = pixel_based

        output_dim: int = config.get_hyperparam('relevance_function.rnd_net.feature_output_dim')

        if self.pixel_based:
            if not isinstance(state_dim, tuple) or len(state_dim) != 3:
                raise ValueError(f"For pixel_based=True, state_dim must be (C, H, W), got {state_dim}")
            
            # As per paper: "three-layer CNNs, with bottleneck latent dimension 64"
            # Our CNNEncoder takes num_filters as a list, and output_dim is the final linear proj.
            feature_bottleneck_dim: int = config.get_hyperparam('relevance_function.rnd_net.feature_bottleneck_dim')
            self.network = CNNEncoder(
                input_shape=state_dim,
                output_dim=output_dim,
                num_filters=[feature_bottleneck_dim] * 3, # Three layers, fixed bottleneck dim
                kernel_sizes=[3, 3, 3], # Common choice
                strides=[1, 1, 1], # Common choice for feature extractors
                activation_fn_name="ReLU",
            )
        else:
            if not isinstance(state_dim, int):
                state_dim = state_dim[0] # Assume state_dim is (vector_dim,)
            # Default MLP parameters if not specifically in config for RND state-based
            hidden_layers: int = 2
            hidden_units: int = 256
            try:
                hidden_layers = config.get_hyperparam('relevance_function.rnd_net.state_hidden_layers')
                hidden_units = config.get_hyperparam('relevance_function.rnd_net.state_hidden_units')
            except KeyError:
                pass # Use defaults if not specified
            
            self.network = MLPBlock(
                input_dim=state_dim,
                output_dim=output_dim,
                hidden_units=hidden_units,
                num_hidden_layers=hidden_layers,
                activation_fn_name="ReLU",
            )
        
        self.apply(init_weights)

        # Freeze all parameters
        for param in self.network.parameters():
            param.requires_grad = False
        
        self.to(self.config.get_hyperparam('experiment.device'))

    def forward(self, next_state: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the TargetNetwork.

        Args:
            next_state (torch.Tensor): The next state observation s'.

        Returns:
            torch.Tensor: The fixed target feature vector.
        """
        if self.pixel_based:
            next_state = _permute_if_needed(next_state) # Ensure (B, C, H, W)
        return self.network(next_state)


class PredictorNetwork(nn.Module):
    """
    A trainable neural network that learns to predict the output of the TargetNetwork
    given the next state s'.
    """

    def __init__(self, config: Config, state_dim: Union[int, Tuple[int, ...]], pixel_based: bool):
        """
        Initializes the PredictorNetwork.

        Args:
            config (Config): Configuration object.
            state_dim (Union[int, Tuple[int, ...]]): Dimension of the next state observation s'.
            pixel_based (bool): True if observations are pixel-based.
        """
        super().__init__()
        self.config: Config = config
        self.pixel_based: bool = pixel_based

        output_dim: int = config.get_hyperparam('relevance_function.rnd_net.feature_output_dim')

        if self.pixel_based:
            if not isinstance(state_dim, tuple) or len(state_dim) != 3:
                raise ValueError(f"For pixel_based=True, state_dim must be (C, H, W), got {state_dim}")
            
            feature_bottleneck_dim: int = config.get_hyperparam('relevance_function.rnd_net.feature_bottleneck_dim')
            self.network = CNNEncoder(
                input_shape=state_dim,
                output_dim=output_dim,
                num_filters=[feature_bottleneck_dim] * 3, # Three layers, fixed bottleneck dim
                kernel_sizes=[3, 3, 3], # Common choice
                strides=[1, 1, 1], # Common choice for feature extractors
                activation_fn_name="ReLU",
            )
        else:
            if not isinstance(state_dim, int):
                state_dim = state_dim[0] # Assume state_dim is (vector_dim,)
            hidden_layers: int = 2
            hidden_units: int = 256
            try:
                hidden_layers = config.get_hyperparam('relevance_function.rnd_net.state_hidden_layers')
                hidden_units = config.get_hyperparam('relevance_function.rnd_net.state_hidden_units')
            except KeyError:
                pass # Use defaults if not specified

            self.network = MLPBlock(
                input_dim=state_dim,
                output_dim=output_dim,
                hidden_units=hidden_units,
                num_hidden_layers=hidden_layers,
                activation_fn_name="ReLU",
            )
        
        self.apply(init_weights) # Trainable by default
        self.to(self.config.get_hyperparam('experiment.device'))

    def forward(self, next_state: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the PredictorNetwork.

        Args:
            next_state (torch.Tensor): The next state observation s'.

        Returns:
            torch.Tensor: The predicted feature vector.
        """
        if self.pixel_based:
            next_state = _permute_if_needed(next_state) # Ensure (B, C, H, W)
        return self.network(next_state)


# --- EPIC Curiosity Networks ---

class EmbeddingNetwork(nn.Module):
    """
    Embeds an observation (state s) into a high-dimensional feature space for Episodic Curiosity (EPIC).
    For pixel-based DMLab, it follows a ResNet-18 architecture with an MLP projection.
    """

    def __init__(self, config: Config, state_dim: Union[int, Tuple[int, ...]], pixel_based: bool):
        """
        Initializes the EmbeddingNetwork.

        Args:
            config (Config): Configuration object.
            state_dim (Union[int, Tuple[int, ...]]): Dimension of the state observation s.
            pixel_based (bool): True if observations are pixel-based.
        """
        super().__init__()
        self.config: Config = config
        self.pixel_based: bool = pixel_based
        
        embedder_output_dim: int = config.get_hyperparam('relevance_function.epic_net.embedder_output_dim')

        if self.pixel_based:
            if not isinstance(state_dim, tuple) or len(state_dim) != 3:
                raise ValueError(f"For pixel_based=True, state_dim must be (C, H, W), got {state_dim}")
            
            # ResNet-18 architecture with output dimension 512, followed by a four-layer MLP,
            # also with feature and output dimensions of 512.
            self.resnet_encoder = ResNetEncoder(
                input_shape=state_dim,
                output_dim=embedder_output_dim, # Output of ResNet part
                num_blocks_per_stage=[2, 2, 2, 2], # Corresponds to ResNet-18
                base_channels=64 # Standard ResNet base
            )
            # Four-layer MLP projection
            self.mlp_projection = MLPBlock(
                input_dim=embedder_output_dim,
                output_dim=embedder_output_dim, # Final output dim
                hidden_units=embedder_output_dim,
                num_hidden_layers=4, # Four-layer MLP
                activation_fn_name="ReLU"
            )
            self.network = nn.Sequential(self.resnet_encoder, self.mlp_projection)

        else:
            if not isinstance(state_dim, int):
                state_dim = state_dim[0] # Assume state_dim is (vector_dim,)
            # Default MLP parameters for EPIC state-based
            hidden_layers: int = 4
            hidden_units: int = 256
            try:
                hidden_layers = config.get_hyperparam('relevance_function.epic_net.state_hidden_layers')
                hidden_units = config.get_hyperparam('relevance_function.epic_net.state_hidden_units')
            except KeyError:
                pass # Use defaults if not specified

            self.network = MLPBlock(
                input_dim=state_dim,
                output_dim=embedder_output_dim,
                hidden_units=hidden_units,
                num_hidden_layers=hidden_layers,
                activation_fn_name="ReLU",
            )
        
        self.apply(init_weights)
        self.to(self.config.get_hyperparam('experiment.device'))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the EmbeddingNetwork.

        Args:
            state (torch.Tensor): The input state observation s.

        Returns:
            torch.Tensor: The embedding vector E(s).
        """
        if self.pixel_based:
            state = _permute_if_needed(state) # Ensure (B, C, H, W)
        return self.network(state)


class Comparator(nn.Module):
    """
    Compares two embedding vectors to output a similarity score (logistic regression).
    """

    def __init__(self, config: Config, embedding_dim: int):
        """
        Initializes the Comparator network.

        Args:
            config (Config): Configuration object.
            embedding_dim (int): The dimension of the embedding vectors E(s).
        """
        super().__init__()
        self.config: Config = config

        input_dim: int = 2 * embedding_dim # Concatenate two embeddings
        output_dim: int = 1 # Output a single similarity score

        # A simple MLP for logistic regression, as per paper implies.
        # Default with 1 hidden layer with embedding_dim units is a common choice.
        self.model = MLPBlock(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_units=embedding_dim, # Common for simple comparator
            num_hidden_layers=1, # Simple comparator
            activation_fn_name="ReLU",
            output_activation_fn_name="Sigmoid" # For probability-like score (0 to 1)
        )
        self.apply(init_weights)
        self.to(self.config.get_hyperparam('experiment.device'))

    def forward(self, embedding_1: torch.Tensor, embedding_2: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the Comparator.

        Args:
            embedding_1 (torch.Tensor): The first embedding vector.
            embedding_2 (torch.Tensor): The second embedding vector (e.g., from memory M).

        Returns:
            torch.Tensor: The similarity score between the two embeddings. Shape (batch_size,).
        """
        x = torch.cat([embedding_1, embedding_2], dim=-1)
        # Squeeze the output to remove the last dimension if it's 1, resulting in (batch_size,)
        return self.model(x).squeeze(-1)

