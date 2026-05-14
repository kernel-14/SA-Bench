## moe_pot_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from utilities import inject_gaussian_noise


class MoEPOTModel(nn.Module):
    """
    Implementation of the Mixture-of-Experts Operator Transformer (MoE-POT) architecture.

    This model includes Patchification, Fourier Layer, and Mixture-of-Experts Layer components
    to handle pretraining and inference tasks for PDE datasets.
    """

    def __init__(self, params: Dict):
        """
        Initialize the MoE-POT model components based on the provided configuration.

        Args:
            params (Dict): Dictionary containing hyperparameters and architecture settings.
        """
        super(MoEPOTModel, self).__init__()
        
        # Load configuration for architecture settings
        self.patch_size = params.get("patch_size", 8)
        self.attention_heads = params.get("attention_heads", 4)
        self.fourier_dim = params.get("fourier_dim", 512)
        self.moe_layers = params.get("mo_layers", 4)
        self.routed_experts = params.get("routed_experts", 16)
        self.shared_experts = params.get("shared_experts", 2)
        self.top_k = params.get("top_k", 4)
        self.load_balancing_weight = params.get("w_balance", 0.1)

        # Patchification Layer: Convert spatial data into tokenized patches.
        self.patchify_layer = nn.Conv2d(
            in_channels=params["input_channels"],
            out_channels=self.fourier_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # Fourier Layer: Integral transform in the frequency domain.
        self.fourier_layer = FourierLayer(self.fourier_dim, self.attention_heads)

        # Mixture-of-Experts Layer: Routed and shared expert mechanisms.
        self.router_gating_network = RouterGatingNetwork(self.routed_experts, self.fourier_dim)
        self.shared_expert_networks = nn.ModuleList([
            ExpertBlock(self.fourier_dim) for _ in range(self.shared_experts)
        ])
        self.routed_expert_networks = nn.ModuleList([
            ExpertBlock(self.fourier_dim) for _ in range(self.routed_experts)
        ])

        # Transformer Layers: Stacked processing structure combining Fourier and MoE layers.
        self.transformer_layers = nn.ModuleList([
            nn.ModuleDict({
                "fourier_layer": FourierLayer(self.fourier_dim, self.attention_heads),
                "moe_layer": MixtureOfExpertsLayer(
                    self.router_gating_network, self.shared_expert_networks, self.routed_expert_networks, self.top_k
                )
            }) for _ in range(self.moe_layers)
        ])

    def patchify(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Convert input tensors into patch tokens for transformer layers.

        Args:
            inputs (Tensor): Input tensor [Batch, Time, Channels, Height, Width].

        Returns:
            Tensor: Patchified tensor [Batch, Time, Num_Patches, Fourier_Dim].
        """
        batch_size, time_steps, channels, height, width = inputs.shape
        reshaped = inputs.view(batch_size * time_steps, channels, height, width)  # Combine batch and time dimensions
        patch_tokens = self.patchify_layer(reshaped)  # Apply convolution
        patch_tokens = patch_tokens.view(batch_size, time_steps, -1, self.fourier_dim)  # Flatten patches
        return patch_tokens

    def apply_fourier_layer(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier transformation and feature extraction.

        Args:
            inputs (Tensor): Feature tensor [Batch, Time, Num_Patches, Fourier_Dim].

        Returns:
            Tensor: Transformed tensor in Fourier domain.
        """
        return self.fourier_layer(inputs)

    def moe_layer(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Apply the Mixture-of-Experts mechanism to handle routed and shared experts.

        Args:
            inputs (Tensor): Feature tensor [Batch, Time, Num_Patches, Fourier_Dim].

        Returns:
            Tensor: Aggregated output tensor after MoE computation.
        """
        return self.router_gating_network(inputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the complete end-to-end forward pass.

        Args:
            x (Tensor): Input tensor representing PDE data.

        Returns:
            Tensor: Predicted tensor for the next timestep.
        """
        # Step 1: Patchification
        patch_tokens = self.patchify(x)

        # Step 2: Sequential processing through transformer layers
        for layer in self.transformer_layers:
            patch_tokens = layer["fourier_layer"](patch_tokens)
            patch_tokens = layer["moe_layer"](patch_tokens)

        # Final output (predicted next timestep)
        return patch_tokens


class FourierLayer(nn.Module):
    """
    Implements integral transformation using Fourier domain operators.
    """

    def __init__(self, fourier_dim: int, attention_heads: int):
        super(FourierLayer, self).__init__()
        self.fourier_dim = fourier_dim
        self.attention_heads = attention_heads

        # Fourier kernel transformation parameters
        self.kernel_transform = nn.Parameter(
            torch.randn(attention_heads, fourier_dim // attention_heads, fourier_dim // attention_heads)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, num_patches, _ = inputs.shape

        # Split input into attention heads
        head_features = inputs.view(batch_size, time_steps, num_patches, self.attention_heads, -1)
        transformed_heads = []

        for head_idx in range(self.attention_heads):
            head = head_features[:, :, :, head_idx, :]
            # Apply Fourier transformation
            frequency_domain = torch.fft.fftn(head, dim=(-1))
            transformed_frequency = frequency_domain @ self.kernel_transform[head_idx]
            spatial_domain = torch.fft.ifftn(transformed_frequency, dim=(-1)).real
            transformed_heads.append(spatial_domain)

        # Concatenate across attention head dimensions
        return torch.cat(transformed_heads, dim=-1)


class ExpertBlock(nn.Module):
    """
    Defines an individual expert block used for both shared and routed experts.
    """

    def __init__(self, input_dim: int):
        super(ExpertBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)


class RouterGatingNetwork(nn.Module):
    """
    Implements the router gating network for expert selection in the MoE layer.
    """

    def __init__(self, num_experts: int, input_dim: int):
        super(RouterGatingNetwork, self).__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim

        # Fully connected layers for gating logits computation
        self.gating_fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_experts)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute sparse routing probabilities for experts.

        Args:
            inputs (Tensor): Feature tensor.

        Returns:
            Tensor: Routing probabilities [Batch, Time, Num_Patches, Num_Experts].
        """
        routing_logits = self.gating_fc(inputs.mean(dim=1))  # Compute logits
        softmax_gating = F.softmax(routing_logits, dim=-1)  # Apply softmax
        return softmax_gating


class MixtureOfExpertsLayer(nn.Module):
    """
    MoE Layer combining routed and shared experts based on dynamic activation.
    """

    def __init__(self, router: RouterGatingNetwork, shared_experts, routed_experts, top_k: int):
        super(MixtureOfExpertsLayer, self).__init__()
        self.router = router
        self.shared_experts = shared_experts
        self.routed_experts = routed_experts
        self.top_k = top_k

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Compute router weights
        routing_weights = self.router(inputs)
        top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1).indices

        # Shared expert contribution
        shared_contribution = torch.mean(torch.stack([expert(inputs) for expert in self.shared_experts]), dim=0)

        # Routed expert contribution (top-k)
        routed_contribution = torch.stack([
            routing_weights[..., idx] * self.routed_experts[idx](inputs)
            for idx in top_k_indices
        ]).sum(dim=0)

        return shared_contribution + routed_contribution
