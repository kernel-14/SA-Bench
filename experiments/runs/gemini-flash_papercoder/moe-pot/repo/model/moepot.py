import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from model.components import PatchificationLayer, TemporalAggregationLayer, FourierLayer, MoELayer
from utils import get_activation


class MoEPOT(nn.Module):
    """
    Mixture-of-Experts Operator Transformer for Large-Scale PDE Pre-Training.

    This class implements the full MoE-POT model architecture, integrating
    patchification, temporal aggregation, Fourier layers, and MoE layers.
    """
    def __init__(self, config: Config):
        """
        Initializes the MoEPOT model.

        Args:
            config: The global configuration object, containing all model hyperparameters.
        """
        super().__init__()
        self.config = config

        # Validate that input/output channels have been set by the data module
        if self.config.model.input_channels == 0 or self.config.model.output_channels == 0:
            raise ValueError(
                "Config.model.input_channels and Config.model.output_channels "
                "must be set by the PDEDataModule before initializing MoEPOT."
            )

        self.embed_dim = config.model.attention_dim
        self.num_layers = config.model.num_layers
        self.patch_size = config.model.patch_size
        self.input_spatial_resolution = config.model.input_spatial_resolution
        self.output_channels = config.model.output_channels
        self.activation = get_activation(config.model.activation)

        # Calculate patched spatial dimensions
        self.H_patched = self.input_spatial_resolution // self.patch_size
        self.W_patched = self.input_spatial_resolution // self.patch_size

        # 1. Patchification Layer
        self.patchification_layer = PatchificationLayer(config)

        # 2. Temporal Aggregation Layer
        self.temporal_aggregation_layer = TemporalAggregationLayer(config)

        # 3. N Blocks (Fourier Layer + MoE Layer)
        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            self.blocks.append(
                nn.ModuleDict({
                    'fourier': FourierLayer(config),
                    'moe': MoELayer(config)
                })
            )

        # 4. Final Prediction Head (maps feature dimension back to output channels)
        # The output is at the patched spatial resolution (H_patched, W_patched)
        self.prediction_head = nn.Conv2d(
            in_channels=self.embed_dim,
            out_channels=self.output_channels,
            kernel_size=1
        )

    def forward(self, u_seq: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for the MoEPOT model.

        Args:
            u_seq: Input sequence of PDE frames.
                   Shape: (batch_size, T_in, C_in, H, W).
            noise: Optional noise tensor to be added to u_seq during pre-training.
                   Shape: (batch_size, T_in, C_in, H, W) or None.

        Returns:
            A tuple containing:
            - predicted_next_frame: The model's prediction for the next frame.
                                    Shape: (batch_size, C_out, H_patched, W_patched).
            - router_weights_per_layer: A list of softmax probabilities from the
                                        router of each MoE layer.
                                        Each element in the list has shape (batch_size, num_routed_experts).
        """
        if noise is not None:
            # Noise is assumed to be pre-generated and scaled (e.g., from utils.inject_noise)
            u_seq = u_seq + noise

        batch_size, T_in, C_in, H, W = u_seq.shape

        # 1. Patchification for each frame
        patched_frames = []
        for t in range(T_in):
            u_t = u_seq[:, t, :, :, :]  # (batch_size, C_in, H, W)
            
            # PatchificationLayer already includes its spatial positional embeddings.
            patched_features_t = self.patchification_layer(u_t) # (batch_size, embed_dim, H_patched, W_patched)
            patched_frames.append(patched_features_t)

        # Stack patched frames to form a sequence (B, T_in, embed_dim, H_patched, W_patched)
        patches_seq_tensor = torch.stack(patched_frames, dim=1)

        # 2. Temporal Aggregation
        # TemporalAggregationLayer already incorporates its temporal embeddings.
        x = self.temporal_aggregation_layer(patches_seq_tensor) # (batch_size, embed_dim, H_patched, W_patched)

        # Store router weights for load balancing loss
        router_weights_per_layer = []

        # 3. Pass through N Blocks (Fourier + MoE)
        for i, block in enumerate(self.blocks):
            # Fourier Layer
            x = block['fourier'](x)

            # MoE Layer
            x, gating_weights = block['moe'](x)
            router_weights_per_layer.append(gating_weights)

        # 4. Final Prediction Head
        predicted_next_frame = self.prediction_head(x) # (batch_size, C_out, H_patched, W_patched)

        return predicted_next_frame, router_weights_per_layer

    def freeze_router(self) -> None:
        """
        Freezes the parameters of all RouterGatingNetwork instances within the model.
        This is typically done during fine-tuning.
        """
        for block in self.blocks:
            for param in block['moe'].router.parameters():
                param.requires_grad = False
        if torch.distributed.get_rank() == 0:
            print("Router-Gating Networks frozen.")

    def unfreeze_router(self) -> None:
        """
        Unfreezes the parameters of all RouterGatingNetwork instances within the model.
        """
        for block in self.blocks:
            for param in block['moe'].router.parameters():
                param.requires_grad = True
        if torch.distributed.get_rank() == 0:
            print("Router-Gating Networks unfrozen.")

