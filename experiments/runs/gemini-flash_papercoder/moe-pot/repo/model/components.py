# model/components.py
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming Config and Utils are in the parent directory or directly accessible
# from 'config' and 'utils' module names.
from config import Config
from utils import get_activation


class PatchificationLayer(nn.Module):
    """
    Transforms raw input frames into spatial patches and adds learnable positional embeddings.
    """
    def __init__(self, config: Config):
        """
        Initializes the PatchificationLayer.

        Args:
            config: The global configuration object.
        """
        super().__init__()
        self.in_channels = config.model.input_channels
        self.embed_dim = config.model.attention_dim
        self.patch_size = config.model.patch_size
        self.input_spatial_resolution = config.model.input_spatial_resolution

        if self.input_spatial_resolution % self.patch_size != 0:
            raise ValueError(
                f"Input spatial resolution ({self.input_spatial_resolution}) "
                f"must be divisible by patch size ({self.patch_size})."
            )

        self.H_prime = self.input_spatial_resolution // self.patch_size
        self.W_prime = self.input_spatial_resolution // self.patch_size

        # Convolutional layer for patch extraction and linear projection
        # Kernel size and stride equal to patch_size create non-overlapping patches.
        self.proj = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # Learnable 2D positional embeddings for spatial patches
        # Shape: (1, embed_dim, H_prime, W_prime) for broadcasting
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.embed_dim, self.H_prime, self.W_prime)
        )
        nn.init.trunc_normal_(self.spatial_pos_embed, std=.02) # Initialize like ViT

    def forward(self, u_t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PatchificationLayer.

        Args:
            u_t: A single input frame. Shape (batch_size, in_channels, H, W).

        Returns:
            A tensor of patch embeddings with added positional information.
            Shape (batch_size, embed_dim, H_prime, W_prime).
        """
        # Apply convolution to extract patches and project to embed_dim
        # conv_out shape: (batch_size, embed_dim, H_prime, W_prime)
        conv_out = self.proj(u_t)

        # Add learnable spatial positional embeddings
        output = conv_out + self.spatial_pos_embed

        return output


class TemporalAggregationLayer(nn.Module):
    """
    Aggregates features across multiple timesteps of patch embeddings to capture
    temporal dynamics, producing a single aggregated feature map.
    """
    def __init__(self, config: Config):
        """
        Initializes the TemporalAggregationLayer.

        Args:
            config: The global configuration object.
        """
        super().__init__()
        self.embed_dim = config.model.attention_dim
        self.T_in = config.model.T_in
        self.activation = get_activation(config.model.activation)

        # Learnable MLP transformation (Wt) implemented as a 1x1 Conv2d.
        # This is shared across all timesteps for efficiency.
        self.mlp_transform = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1),
            self.activation,
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1)
        )

        # Learnable temporal embeddings, one for each input timestep.
        # Shape: (T_in, embed_dim). These will be broadcast across spatial dimensions.
        self.temporal_embeddings = nn.Parameter(
            torch.zeros(self.T_in, self.embed_dim)
        )
        nn.init.trunc_normal_(self.temporal_embeddings, std=.02) # Initialize like ViT

    def forward(self, patches_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TemporalAggregationLayer.

        Args:
            patches_seq: A sequence of patch embeddings.
                         Shape (batch_size, T_in, embed_dim, H_prime, W_prime).

        Returns:
            An aggregated feature map. Shape (batch_size, embed_dim, H_prime, W_prime).
        """
        batch_size, _, H_prime, W_prime = patches_seq.shape[0], patches_seq.shape[2], patches_seq.shape[3], patches_seq.shape[4]
        
        # Initialize an output tensor for aggregated features
        aggregated_features = torch.zeros(
            batch_size, self.embed_dim, H_prime, W_prime,
            device=patches_seq.device, dtype=patches_seq.dtype
        )

        for t in range(self.T_in):
            # Extract patch embedding for the current timestep
            z_p_t = patches_seq[:, t, :, :, :]  # Shape: (batch_size, embed_dim, H_prime, W_prime)

            # Add learnable temporal embedding. Reshape for broadcasting.
            # temporal_embeddings[t] shape (embed_dim) -> (1, embed_dim, 1, 1)
            z_p_t_with_time_embed = z_p_t + self.temporal_embeddings[t].view(1, self.embed_dim, 1, 1)

            # Apply the shared MLP transformation (Wt)
            transformed_features = self.mlp_transform(z_p_t_with_time_embed)

            # Sum up transformed features across time
            aggregated_features += transformed_features

        return aggregated_features


class FourierLayer(nn.Module):
    """
    Implements a multi-head Fourier neural operator (FNO) block, which learns
    kernel-based integral transformations in the frequency domain.
    """
    def __init__(self, config: Config):
        """
        Initializes the FourierLayer.

        Args:
            config: The global configuration object.
        """
        super().__init__()
        self.embed_dim = config.model.attention_dim
        self.num_heads = config.model.num_heads
        self.activation = get_activation(config.model.activation)

        if self.embed_dim % self.num_heads != 0:
            raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})")

        self.head_dim = self.embed_dim // self.num_heads

        # Layer normalization applied before FFT
        self.norm = nn.LayerNorm(normalized_shape=[self.embed_dim, config.model.input_spatial_resolution // config.model.patch_size, config.model.input_spatial_resolution // config.model.patch_size])

        # MLP-like transformations in the Fourier domain for each head
        # These operate on complex numbers, so we handle real and imaginary parts
        # concatenated along the channel dimension. Thus, 2 * head_dim in/out channels.
        self.fourier_mlps = nn.ModuleList()
        for _ in range(self.num_heads):
            self.fourier_mlps.append(
                nn.Sequential(
                    nn.Conv2d(self.head_dim * 2, self.head_dim * 2, kernel_size=1), # W1
                    self.activation,
                    nn.Conv2d(self.head_dim * 2, self.head_dim * 2, kernel_size=1)  # W2
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FourierLayer.

        Args:
            x: Input tensor. Shape (batch_size, embed_dim, H_prime, W_prime).

        Returns:
            Output tensor after Fourier transformations and residual connection.
            Shape (batch_size, embed_dim, H_prime, W_prime).
        """
        # Keep original for residual connection
        x_orig = x

        # Layer normalization (applied to the last three dimensions: C, H, W)
        # Permute to (B, H', W', C) for LayerNorm, then permute back
        x = x.permute(0, 2, 3, 1) # (B, H_prime, W_prime, embed_dim)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2) # (B, embed_dim, H_prime, W_prime)


        batch_size, _, H_prime, W_prime = x.shape
        
        output_heads = []
        # Split input into heads along the embed_dim
        x_heads = x.chunk(self.num_heads, dim=1) # List of (B, head_dim, H_prime, W_prime)

        for i, x_head in enumerate(x_heads):
            # 1. Real Fast Fourier Transform (RFFT2)
            # rfft2 outputs complex tensor of shape (B, head_dim, H_prime, W_prime//2 + 1)
            fft_out = torch.fft.rfft2(x_head, norm="ortho")

            # 2. Separate real and imaginary parts and concatenate
            # This prepares the complex tensor for real-valued Conv2d operations
            fft_out_real = fft_out.real
            fft_out_imag = fft_out.imag
            fft_concat = torch.cat([fft_out_real, fft_out_imag], dim=1) # (B, 2*head_dim, H_prime, W_prime//2 + 1)

            # 3. Apply MLP-like transformations in the frequency domain
            mlp_output = self.fourier_mlps[i](fft_concat)

            # 4. Split back into real and imaginary parts
            mlp_output_real, mlp_output_imag = mlp_output.chunk(2, dim=1)
            
            # 5. Reconstruct complex tensor
            # The .complex() method requires two real tensors of the same shape
            transformed_fft_out = torch.complex(mlp_output_real, mlp_output_imag)

            # 6. Inverse Real Fast Fourier Transform (IRFFT2)
            # irfft2 outputs real tensor of shape (B, head_dim, H_prime, W_prime)
            irfft_out = torch.fft.irfft2(transformed_fft_out, s=(H_prime, W_prime), norm="ortho")
            output_heads.append(irfft_out)

        # Concatenate outputs from all heads
        # Shape: (batch_size, embed_dim, H_prime, W_prime)
        concatenated_head_outputs = torch.cat(output_heads, dim=1)

        # Residual connection
        output = x_orig + concatenated_head_outputs
        
        return output


class Expert(nn.Module):
    """
    Represents a single expert network (either shared or routed) in the MoE layer.
    Implemented as a simple convolutional subnetwork (effectively a point-wise MLP).
    """
    def __init__(self, config: Config):
        """
        Initializes an Expert network.

        Args:
            config: The global configuration object.
        """
        super().__init__()
        self.embed_dim = config.model.attention_dim
        self.mlp_dim = config.model.mlp_dim
        self.cnn_layers = config.model.expert_cnn_layers
        self.cnn_kernel_size = config.model.expert_cnn_kernel_size
        self.activation = get_activation(config.model.activation)

        layers = []
        in_channels = self.embed_dim
        for i in range(self.cnn_layers):
            out_channels = self.mlp_dim if i < self.cnn_layers - 1 else self.embed_dim
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=self.cnn_kernel_size,
                    padding=self.cnn_kernel_size // 2 # Ensure same output size
                )
            )
            if i < self.cnn_layers - 1: # No activation after the last layer
                layers.append(self.activation)
            in_channels = out_channels
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert network.

        Args:
            x: Input tensor. Shape (batch_size, embed_dim, H_prime, W_prime).

        Returns:
            Output tensor of the expert.
            Shape (batch_size, embed_dim, H_prime, W_prime).
        """
        return self.net(x)


class RouterGatingNetwork(nn.Module):
    """
    Determines expert selection and their corresponding gating weights based on the input feature map.
    """
    def __init__(self, config: Config):
        """
        Initializes the RouterGatingNetwork.

        Args:
            config: The global configuration object.
        """
        super().__init__()
        self.embed_dim = config.model.attention_dim
        self.num_routed_experts = config.model.num_routed_experts
        self.cnn_layers = config.model.router_cnn_layers
        self.cnn_kernel_size = config.model.router_cnn_kernel_size
        self.activation = get_activation(config.model.activation)

        layers = []
        in_channels = self.embed_dim
        for i in range(self.cnn_layers):
            out_channels = self.embed_dim # Keep channel dim same throughout CNNs
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=self.cnn_kernel_size,
                    padding=self.cnn_kernel_size // 2
                )
            )
            layers.append(self.activation)
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)

        # After convolutions, apply adaptive average pooling to get a (1,1) spatial dimension
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Final linear layer to produce logits for each routed expert
        self.final_proj = nn.Conv2d(self.embed_dim, self.num_routed_experts, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the RouterGatingNetwork.

        Args:
            x: Input tensor. Shape (batch_size, embed_dim, H_prime, W_prime).

        Returns:
            A tuple (logits, gating_weights).
            logits: Shape (batch_size, num_routed_experts).
            gating_weights: Shape (batch_size, num_routed_experts) (after softmax).
        """
        # Pass through convolutional layers
        x = self.conv_layers(x) # Shape: (B, embed_dim, H_prime, W_prime)

        # Apply adaptive average pooling to reduce spatial dimensions to 1x1
        x = self.avg_pool(x) # Shape: (B, embed_dim, 1, 1)

        # Apply final projection to get logits
        logits = self.final_proj(x) # Shape: (B, num_routed_experts, 1, 1)

        # Squeeze the spatial dimensions to get (batch_size, num_routed_experts)
        logits = logits.squeeze(-1).squeeze(-1)

        # Apply softmax to get gating weights
        gating_weights = F.softmax(logits, dim=-1)

        return logits, gating_weights


class MoELayer(nn.Module):
    """
    Combines shared experts, dynamically selected routed experts, and the router-gating
    mechanism to produce the output for a single MoE block.
    """
    def __init__(self, config: Config):
        """
        Initializes the MoELayer.

        Args:
            config: The global configuration object.
        """
        super().__init__()
        self.embed_dim = config.model.attention_dim
        self.num_routed_experts = config.model.num_routed_experts
        self.num_shared_experts = config.model.num_shared_experts
        self.top_k = config.model.top_k

        if self.top_k > self.num_routed_experts:
            raise ValueError(f"Top-K ({self.top_k}) cannot be greater than "
                             f"the number of routed experts ({self.num_routed_experts}).")

        # Router-Gating Network
        self.router = RouterGatingNetwork(config)

        # Shared Experts (ModuleList allows indexing)
        self.shared_experts = nn.ModuleList(
            [Expert(config) for _ in range(self.num_shared_experts)]
        )

        # Routed Experts (ModuleList allows indexing)
        self.routed_experts = nn.ModuleList(
            [Expert(config) for _ in range(self.num_routed_experts)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the MoELayer.

        Args:
            x: Input tensor. Shape (batch_size, embed_dim, H_prime, W_prime).

        Returns:
            A tuple (final_output, gating_weights).
            final_output: Output of the MoE layer.
                          Shape (batch_size, embed_dim, H_prime, W_prime).
            gating_weights: Softmax probabilities from the router for all routed experts.
                            Shape (batch_size, num_routed_experts).
        """
        batch_size, embed_dim, H_prime, W_prime = x.shape

        # Step 1: Router-Gating Network determines expert weights
        # gating_logits are returned for load balancing loss calculation
        gating_logits, gating_weights = self.router(x) # (B, N_r), (B, N_r)

        # Step 2: Process through Shared Experts
        shared_expert_output = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_expert_output += expert(x)
        shared_expert_output /= float(self.num_shared_experts)

        # Step 3: Process through Routed Experts
        # Select Top-K experts based on gating_weights
        top_k_weights, top_k_indices = torch.topk(gating_weights, k=self.top_k, dim=-1) # (B, K), (B, K)

        routed_expert_output = torch.zeros_like(x)

        # Calculate outputs for all routed experts on the full batch once for efficiency
        # This results in a list of (B, E, H', W') tensors
        all_routed_expert_outputs = [expert(x) for expert in self.routed_experts]
        # Stack to form a single tensor: (B, N_r, E, H', W')
        all_routed_expert_outputs_tensor = torch.stack(all_routed_expert_outputs, dim=1)

        # Efficiently combine selected routed expert outputs
        # For each of the K selected experts:
        for k in range(self.top_k):
            # Extract the k-th selected expert index and weight for each sample in the batch
            expert_indices_k = top_k_indices[:, k] # Shape: (B,)
            weights_k = top_k_weights[:, k]         # Shape: (B,)

            # Use torch.gather to select the output corresponding to the k-th chosen expert
            # for each sample in the batch.
            # The index must match the dimensions of the source tensor after the first (batch) dim.
            # all_routed_expert_outputs_tensor: (B, N_r, E, H', W')
            # expert_indices_k.view(B, 1, 1, 1, 1) broadcasts the index across other dimensions.
            
            # The expand() is needed to match the dimensions of all_routed_expert_outputs_tensor
            # before gather, effectively selecting a slice (N_r=1) for each batch element.
            selected_expert_output_for_k = torch.gather(
                all_routed_expert_outputs_tensor,
                dim=1, # Gather along the expert dimension
                index=expert_indices_k.view(batch_size, 1, 1, 1, 1).expand(
                    batch_size, 1, embed_dim, H_prime, W_prime
                )
            ).squeeze(1) # Squeeze out the N_r dimension (which is now 1) -> (B, E, H', W')

            # Multiply by the corresponding weights and accumulate
            # weights_k.view(-1, 1, 1, 1) reshapes (B,) to (B, 1, 1, 1) for broadcasting
            routed_expert_output += selected_expert_output_for_k * weights_k.view(-1, 1, 1, 1)

        # Step 4: Combine shared and routed outputs
        final_output = shared_expert_output + routed_expert_output

        return final_output, gating_weights

