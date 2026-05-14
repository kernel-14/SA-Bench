
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Optional, Union, Tuple

class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) for policy networks, Q-functions,
    or other dense computations.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int,
                 num_layers: int, activation: nn.Module = nn.ReLU):
        super().__init__()
        self.num_layers = num_layers
        self.activation = activation()

        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                layers.append(self.activation)
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CNNEncoder(nn.Module):
    """
    Convolutional Neural Network Encoder for pixel-based observations.
    Based on DrQ-v2 architecture mentioned in the paper.
    """
    def __init__(self, observation_shape: Tuple[int, int, int], output_dim: int):
        super().__init__()
        assert len(observation_shape) == 3 # C, H, W
        self.observation_shape = observation_shape
        self.output_dim = output_dim

        self.convs = nn.Sequential(
            nn.Conv2d(observation_shape[0], 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # Calculate the output dimension after conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, *observation_shape)
            n_flatten = self.convs(dummy_input).flatten(1).shape[1]
        
        self.fc = nn.Linear(n_flatten, output_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (B, C, H, W) or (C, H, W)
        if len(obs.shape) == 3: # (C, H, W)
            obs = obs.unsqueeze(0) # Add batch dimension -> (1, C, H, W)
        
        # If the input tensor is (B, H, W, C), permute to (B, C, H, W)
        # This check assumes C is self.observation_shape[0]
        if obs.shape[1] != self.observation_shape[0]:
            # This heuristic checks if the 2nd dim (H) or 3rd dim (W) match the expected channels
            # If so, it's likely channel-last and needs permuting.
            if obs.shape[3] == self.observation_shape[0]: # (B, H, W, C)
                obs = obs.permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Unexpected observation shape. Expected (B, C, H, W) or (B, H, W, C) where C={self.observation_shape[0]}. Got {obs.shape}")

        h = self.convs(obs)
        h = h.flatten(1)
        latent_features = self.fc(h)
        return latent_features

class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for time encoding in diffusion models.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """
    A residual block for the diffusion U-Net, similar to what's found in common
    diffusion architectures.
    """
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: Optional[int] = None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        if time_embed_dim is not None:
            self.time_mlp = nn.Linear(time_embed_dim, out_channels)
        else:
            self.time_mlp = None

    def forward(self, x: torch.Tensor, time_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.relu(self.bn1(self.conv1(x)))
        if self.time_mlp is not None and time_embed is not None:
            h += self.time_mlp(time_embed).unsqueeze(-1) # Add time embedding
        h = self.bn2(self.conv2(h))
        return self.relu(h + self.residual_conv(x))

class AttentionBlock(nn.Module):
    """
    Attention block for diffusion models.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels) # Using GroupNorm as is common
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1) # Split into query, key, value

        # Permute for attention (B, C, L) -> (B, L, C)
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)

        # Scaled dot-product attention
        # (B, L, C) @ (B, C, L) -> (B, L, L)
        attn = torch.bmm(q, k.transpose(1, 2)) * (q.shape[-1] ** -0.5)
        attn = F.softmax(attn, dim=-1)

        # (B, L, L) @ (B, L, C) -> (B, L, C)
        h = torch.bmm(attn, v)
        
        # Permute back (B, L, C) -> (B, C, L)
        h = h.permute(0, 2, 1)
        return x + self.proj_out(h)

class UNetDiffusionModel(nn.Module):
    """
    A simplified U-Net style diffusion model.
    The paper mentions "conditional diffusion models" and "strong diffusion model architectures",
    and refers to DDPM (Ho et al., 2020).
    It also mentions "residual MLP denoising diffusion model" in Appendix A.2,
    suggesting a 1D convolution approach for sequence data or flattened state/action representations.
    We'll assume a 1D U-Net over the flattened transition features (s, a, s', r).
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 time_embed_dim: int,
                 cond_embed_dim: int, # Embedding dimension for the relevance function 'c'
                 hidden_dims: Tuple[int, ...] = (128, 256, 512),
                 num_res_blocks: int = 2,
                 attention_resolutions: Tuple[int, ...] = (2,), # Apply attention at these resolutions (indices)
                 model_capacity_factor: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.time_embed_dim = time_embed_dim
        self.cond_embed_dim = cond_embed_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.ReLU()
        )

        # Condition embedding (for relevance function c)
        self.cond_mlp = nn.Sequential(
            nn.Linear(1, cond_embed_dim), # Relevance 'c' is typically a scalar
            nn.ReLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim)
        )

        # Initial projection
        init_dim = int(hidden_dims[0] * model_capacity_factor)
        self.initial_conv = nn.Conv1d(input_dim, init_dim, kernel_size=3, padding=1)

        current_dim = init_dim
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Downsampling path
        for i, dim in enumerate(hidden_dims):
            dim = int(dim * model_capacity_factor)
            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(current_dim, dim, time_embed_dim + cond_embed_dim))
                if i in attention_resolutions:
                    self.downs.append(AttentionBlock(dim))
                current_dim = dim
            if i < len(hidden_dims) - 1:
                self.downs.append(nn.Conv1d(current_dim, current_dim, kernel_size=4, stride=2, padding=1)) # Downsample

        # Bottleneck
        self.mid_res1 = ResidualBlock(current_dim, current_dim, time_embed_dim + cond_embed_dim)
        self.mid_attn = AttentionBlock(current_dim)
        self.mid_res2 = ResidualBlock(current_dim, current_dim, time_embed_dim + cond_embed_dim)

        # Upsampling path
        for i, dim in enumerate(reversed(hidden_dims)):
            dim = int(dim * model_capacity_factor)
            for _ in range(num_res_blocks):
                self.ups.append(ResidualBlock(current_dim * 2 if i > 0 else current_dim, dim, time_embed_dim + cond_embed_dim)) # Skip connection
                if i in attention_resolutions:
                    self.ups.append(AttentionBlock(dim))
                current_dim = dim
            if i < len(hidden_dims) - 1:
                self.ups.append(nn.ConvTranspose1d(current_dim, current_dim, kernel_size=4, stride=2, padding=1)) # Upsample

        # Final output layer
        self.final_conv = nn.Conv1d(current_dim, output_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, time: torch.Tensor, condition: torch.Tensor,
                uncond_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch_size, input_dim) - representing flattened (s,a,s',r)
        # time: (batch_size,) - diffusion timestep
        # condition: (batch_size,) - scalar relevance score

        time_embed = self.time_mlp(time)
        cond_embed = self.cond_mlp(condition.unsqueeze(-1)) # Add last dim for linear layer

        if uncond_mask is not None:
            # Apply classifier-free guidance: replace condition with a null condition
            cond_embed = cond_embed * (1 - uncond_mask).unsqueeze(-1) # Mask out condition for some samples

        # Concatenate time and condition embeddings
        combined_embed = torch.cat([time_embed, cond_embed], dim=-1)

        # Assuming x is (B, F) where F is the flattened feature dimension
        # U-Net typically operates on (B, C, L) for 1D or (B, C, H, W) for 2D.
        # Here, we treat the input_dim as "channels" and sequence length as 1,
        # or expand it to (B, input_dim, 1) and then process.
        # The paper mentions "residual MLP denoising diffusion model" and 1D convs are common for sequential data.
        # Let's assume input_dim is the feature dimension of the flattened transition, and we treat it as a sequence of 1.
        # x = x.unsqueeze(-1) # (B, input_dim, 1)

        skips = []
        h = self.initial_conv(x.unsqueeze(-1)).squeeze(-1) # Initial convolution, then squeeze for residual block

        for layer in self.downs:
            if isinstance(layer, ResidualBlock):
                h = layer(h.unsqueeze(-1), combined_embed).squeeze(-1) # Pass combined_embed
            elif isinstance(layer, AttentionBlock):
                h = layer(h.unsqueeze(-1)).squeeze(-1)
            else: # Conv1d for downsampling
                h = layer(h.unsqueeze(-1)).squeeze(-1)
            skips.append(h) # Store for skip connections

        h = self.mid_res1(h.unsqueeze(-1), combined_embed).squeeze(-1)
        h = self.mid_attn(h.unsqueeze(-1)).squeeze(-1)
        h = self.mid_res2(h.unsqueeze(-1), combined_embed).squeeze(-1)

        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                h = torch.cat([h, skips.pop()], dim=1) # Concatenate with skip connection
                h = layer(h.unsqueeze(-1), combined_embed).squeeze(-1)
            elif isinstance(layer, AttentionBlock):
                h = layer(h.unsqueeze(-1)).squeeze(-1)
            else: # ConvTranspose1d for upsampling
                h = layer(h.unsqueeze(-1)).squeeze(-1)
        
        output = self.final_conv(h.unsqueeze(-1)).squeeze(-1) # Final convolution
        return output

class ValueNetwork(nn.Module):
    """
    Common value network for Q-functions (critic) and policy (actor)
    Used in SAC/REDQ.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.q_net = MLP(obs_dim + action_dim, 1, hidden_dim, num_layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q_net(torch.cat([obs, action], dim=-1))



class FeatureEncoder(nn.Module):
    """
    Feature encoder h for ICM and RND. Can be an MLP or CNN based on observation type.
    """
    def __init__(self, observation_space, feature_dim: int):
        super().__init__()
        if len(observation_space.shape) == 1: # State-based
            obs_dim = observation_space.shape[0]
            self.encoder = MLP(obs_dim, feature_dim, 256, 2) # Example MLP
        elif len(observation_space.shape) == 3: # Pixel-based
            self.encoder = CNNEncoder(observation_space.shape, feature_dim)
        else:
            raise ValueError("Unsupported observation space shape")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

class ForwardDynamicsModel(nn.Module):
    """
    Forward dynamics model g for ICM. Predicts next latent state.
    """
    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = MLP(feature_dim + action_dim, feature_dim, hidden_dim, 2) # Example MLP

    def forward(self, latent_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([latent_state, action], dim=-1))

class InverseDynamicsModel(nn.Module):
    """
    Inverse dynamics model for ICM (optional, usually for action prediction).
    Not explicitly used for curiosity in the paper, but often part of ICM.
    """
    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = MLP(feature_dim * 2, action_dim, hidden_dim, 2) # Predict action from (s, s') latents

    def forward(self, latent_state: torch.Tensor, latent_next_state: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([latent_state, latent_next_state], dim=-1))

class RandomNetworkDistillation(nn.Module):
    """
    Random Network Distillation (RND) module.
    Consists of a fixed target network and a trainable predictor network.
    """
    def __init__(self, observation_space, feature_dim: int, latent_dim: int):
        super().__init__()
        # Target network (fixed)
        self.target_net = FeatureEncoder(observation_space, feature_dim)
        for param in self.target_net.parameters():
            param.requires_grad = False

        # Predictor network (trainable)
        self.predictor_net = nn.Sequential(
            FeatureEncoder(observation_space, latent_dim),
            MLP(latent_dim, feature_dim, 256, 2) # Additional MLP projection
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            target_features = self.target_net(obs)
        predicted_features = self.predictor_net(obs)
        return predicted_features, target_features

# Placeholder for Context-Tree Switching (CTS) Density Model
# This is typically a non-neural network model for counting and density estimation.
# We'll represent it as a conceptual module for now.
class CTSDensityModel(nn.Module):
    def __init__(self, image_size: int, context_bins: int):
        super().__init__()
        self.image_size = image_size
        self.context_bins = context_bins
        print(f"Initialized conceptual CTSDensityModel with image_size={image_size}, context_bins={context_bins}")
        # In a full implementation, this would involve a complex data structure
        # for storing counts and estimating densities.
        # For now, it's a placeholder.

    def update(self, obs_action_pair: Tuple[np.ndarray, np.ndarray]):
        # Conceptual update method
        pass

    def get_pseudo_count(self, obs_action_pair: Tuple[np.ndarray, np.ndarray]) -> float:
        # Conceptual method to get pseudo-count
        return 1.0 # Placeholder value


# Placeholder for Episodic Curiosity (ECO)
class ECOModule(nn.Module):
    """
    Episodic Curiosity (ECO) module, based on reachability.
    Conceptual implementation, as it involves a memory buffer M and a comparator C.
    """
    def __init__(self, obs_dim: int, embedder_out_dim: int, mlp_layers: int, mlp_dim: int,
                 memory_size: int = 200):
        super().__init__()
        self.embedder = FeatureEncoder(obs_dim, embedder_out_dim) # E in paper
        self.comparator = MLP(embedder_out_dim * 2, 1, mlp_dim, mlp_layers) # C in paper (input for 2 embeddings)
        self.memory_size = memory_size
        self.memory_buffer = [] # Placeholder for M

    def update_memory(self, obs_embedding: torch.Tensor):
        if len(self.memory_buffer) >= self.memory_size:
            self.memory_buffer.pop(0) # FIFO
        self.memory_buffer.append(obs_embedding.detach().cpu().numpy()) # Store numpy for now

    def get_novelty_score(self, current_obs_embed: torch.Tensor) -> float:
        if not self.memory_buffer:
            return 1.0 # Very novel if memory is empty

        scores = []
        for mem_embed_np in self.memory_buffer:
            mem_embed = torch.from_numpy(mem_embed_np).to(current_obs_embed.device)
            # Assuming comparator takes concatenation of two embeddings
            combined_embed = torch.cat([current_obs_embed, mem_embed], dim=-1)
            similarity = torch.sigmoid(self.comparator(combined_embed)).item() # Probability of being 'close'
            scores.append(similarity)
        
        # Novelty is high if similarity is low to existing memories
        # The paper defines F as alpha * (beta - F(C(E(s), E(s_i))))
        # This implies lower C(E(s), E(s_i)) leads to higher F, i.e., higher novelty.
        # So we want to find the minimum similarity to say it's novel
        # F = alpha * (beta - percentile-90(C))
        
        # For now, a simplified novelty score (e.g., inverse of average similarity)
        return 1.0 - np.mean(scores) if scores else 1.0
