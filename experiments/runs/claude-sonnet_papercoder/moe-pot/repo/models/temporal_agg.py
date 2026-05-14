# models/temporal_agg.py
"""Temporal aggregation layer for the MoE-POT architecture.

Implements the TemporalAggregation class, which collapses T input timesteps
into a single aggregated feature map encoding the temporal dynamics of the
PDE trajectory.

From the paper (Section 4, Input Encoding and Temporal Aggregation):
    "To capture temporal dynamics, we employ a temporal aggregation layer
    that extracts information across adjacent time steps. For each local
    node feature z_p^t ∈ R^C in Z_p^t, we apply a learnable MLP
    transformation W_t combined with Fourier feature constant γ ∈ R^C:
        z_agg = Σ_t W_t · z_p^t · e^{-iγt}
    This aggregation enables the model to implicitly infer the underlying
    PDE governing parameters."

Since all computations are real-valued, the complex exponential is
approximated by its real part:
    Re(e^{-iγt}) = cos(γt)

giving the real-valued implementation:
    z_agg = Σ_t W_t(z_p^t) · cos(γt)

From config.yaml:
    architecture.input_timesteps: 10    (T=10 frames as input)
    architecture.target_resolution: 128 (H=W=128)
    architecture.patch_size: 8          (P=8, token grid = 16×16)
    models.tiny.attn_dim: 512           (embed_dim for Tiny model)
    models.small.attn_dim: 1024         (embed_dim for Small/Medium models)

Data flow:
    Input:  (B, T=10, embed_dim, H'=16, W'=16)
    Output: (B, embed_dim, H'=16, W'=16)
"""

import torch
import torch.nn as nn


class TemporalAggregation(nn.Module):
    """Temporal aggregation layer that collapses T timesteps into one feature map.

    Applies per-timestep learnable linear transformations modulated by
    learnable Fourier frequency coefficients, then sums across the time
    dimension to produce a single spatially-resolved feature map.

    The aggregation formula (real-valued approximation):
        z_agg = Σ_{t=0}^{T-1} W_t(z_p^t) · cos(γ · t)

    where:
      - W_t: Independent Linear(embed_dim, embed_dim) per timestep t.
             Allows each timestep to have a completely independent feature
             transformation, capturing timestep-specific PDE dynamics.
      - γ ∈ R^{embed_dim}: Learnable Fourier frequency vector. Each channel
             dimension has its own frequency, enabling the model to learn
             different temporal frequencies for different feature channels.
      - cos(γ · t): Real part of the complex exponential e^{-iγt}, computed
             element-wise over the embed_dim channel dimension.

    Design choices:
      - Separate Linear layers per timestep (not shared): Matches the paper's
        W_t notation with subscript t, allowing timestep-specific transformations.
      - Real-valued cos modulation: Takes Re(e^{-iγt}) = cos(γt), discarding
        the imaginary sin component for simplicity and real-valued consistency.
      - No activation function: The aggregation is a smooth weighted sum;
        nonlinearities come from the subsequent Fourier and MoE layers.
      - Efficient batched implementation: Reshapes to (B, H'*W', T, embed_dim)
        to apply all Linear layers via batched matrix multiplications without
        a Python loop over spatial positions.

    Attributes:
        embed_dim: Feature dimension. Corresponds to attn_dim in config.yaml:
            512 (Tiny), 1024 (Small/Medium).
        input_timesteps: Number of input timesteps T. Default 10
            (config.yaml architecture.input_timesteps).
        temporal_weights: ModuleList of input_timesteps Linear(embed_dim,
            embed_dim) layers — one independent transformation W_t per
            timestep. All are registered as model parameters.
        fourier_gamma: Learnable parameter γ ∈ R^{embed_dim}, initialized
            from N(0, 1). Optimized during training to learn the most
            informative temporal frequencies for the PDE dynamics.
    """

    def __init__(
        self,
        embed_dim: int,
        input_timesteps: int = 10,
    ) -> None:
        """Initializes the TemporalAggregation layer.

        Creates input_timesteps independent Linear(embed_dim, embed_dim)
        layers and a learnable Fourier frequency vector γ ∈ R^{embed_dim}.

        Args:
            embed_dim: Feature dimension for the patch tokens. Corresponds
                to attn_dim in config.yaml model configurations:
                  - Tiny:   512  (config.yaml models.tiny.attn_dim)
                  - Small:  1024 (config.yaml models.small.attn_dim)
                  - Medium: 1024 (config.yaml models.medium.attn_dim)
                This is both the input and output dimension of each W_t
                linear layer, and the dimension of γ.
            input_timesteps: Number of input timesteps T. One independent
                Linear layer W_t is created for each timestep. Default 10
                (config.yaml architecture.input_timesteps). The model takes
                T=10 consecutive frames as input and predicts the next frame.

        Raises:
            ValueError: If embed_dim <= 0.
            ValueError: If input_timesteps <= 0.
        """
        super().__init__()

        # --- Input validation ---
        if embed_dim <= 0:
            raise ValueError(
                f"embed_dim must be positive, got {embed_dim}."
            )
        if input_timesteps <= 0:
            raise ValueError(
                f"input_timesteps must be positive, got {input_timesteps}."
            )

        self.embed_dim: int = embed_dim
        self.input_timesteps: int = input_timesteps

        # ----------------------------------------------------------------
        # Per-timestep linear transformations W_t
        # ----------------------------------------------------------------
        # Create input_timesteps independent Linear(embed_dim, embed_dim)
        # layers. Each W_t has its own parameters, allowing the model to
        # learn timestep-specific feature transformations.
        #
        # Using nn.ModuleList ensures all W_t parameters are properly
        # registered as model parameters and included in optimizer updates.
        #
        # bias=True (default): Standard for linear layers; allows each
        # timestep transformation to have an independent bias term.
        #
        # Memory: T × embed_dim × embed_dim × 4 bytes
        #   Tiny (T=10, d=512):  10 × 512 × 512 × 4 ≈ 10 MB
        #   Small (T=10, d=1024): 10 × 1024 × 1024 × 4 ≈ 40 MB
        self.temporal_weights: nn.ModuleList = nn.ModuleList(
            [
                nn.Linear(
                    in_features=embed_dim,
                    out_features=embed_dim,
                    bias=True,
                )
                for _ in range(input_timesteps)
            ]
        )

        # ----------------------------------------------------------------
        # Learnable Fourier frequency vector γ ∈ R^{embed_dim}
        # ----------------------------------------------------------------
        # Initialized from N(0, 1) to provide diverse initial frequencies.
        # During training, γ is optimized to learn the most informative
        # temporal frequencies for the PDE dynamics being modeled.
        #
        # The modulation cos(γ · t) is computed element-wise:
        #   - γ has shape (embed_dim,)
        #   - t is a scalar (timestep index 0, 1, ..., T-1)
        #   - γ * t has shape (embed_dim,) — per-channel frequency scaling
        #   - cos(γ * t) has shape (embed_dim,) — per-channel modulation
        #
        # At t=0: cos(γ * 0) = cos(0) = 1.0 for all channels, so the
        # first timestep always contributes with unit weight regardless of γ.
        self.fourier_gamma: nn.Parameter = nn.Parameter(
            torch.randn(embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregates T timesteps into a single feature map via weighted sum.

        Implements the real-valued temporal aggregation formula:
            z_agg = Σ_{t=0}^{T-1} W_t(z_p^t) · cos(γ · t)

        Processing pipeline:
            (B, T, embed_dim, H', W')
            → permute to (B, T, H', W', embed_dim)   [embed_dim last for Linear]
            → reshape to (B, T, H'*W', embed_dim)     [flatten spatial dims]
            → permute to (B, H'*W', T, embed_dim)     [group by spatial location]
            → for t in range(T):
                  z_t = W_t(x[:, :, t, :])            [(B, H'*W', embed_dim)]
                  cos_t = cos(γ * t)                   [(embed_dim,)]
                  z_agg += z_t * cos_t                 [broadcast over (B, H'*W')]
            → reshape to (B, H', W', embed_dim)
            → permute to (B, embed_dim, H', W')

        Args:
            x: Patchified feature tensor of shape (B, T, embed_dim, H', W')
                where:
                - B: Batch size (up to 20 for pre-training, config.yaml
                  pretraining.batch_size).
                - T: Number of input timesteps. Typically 10 (config.yaml
                  architecture.input_timesteps). Must be <= self.input_timesteps.
                - embed_dim: Feature dimension. Must match self.embed_dim.
                - H': Token grid height = target_resolution / patch_size = 16
                  (config.yaml architecture.target_resolution=128,
                  architecture.patch_size=8).
                - W': Token grid width = 16.
                This is Z_p^t from the paper — the output of PatchifyLayer.

        Returns:
            Aggregated feature map of shape (B, embed_dim, H', W').
            This is z_agg from the paper, ready for the first MoEBlock's
            FourierLayer. The temporal dimension T has been collapsed into
            a single feature map encoding the PDE trajectory dynamics.
        """
        batch_size: int = x.shape[0]
        t_actual: int = x.shape[1]
        # embed_dim: int = x.shape[2]  # must equal self.embed_dim
        h_prime: int = x.shape[3]
        w_prime: int = x.shape[4]

        # ----------------------------------------------------------------
        # Step 1: Reshape for efficient batched linear operations
        # ----------------------------------------------------------------
        # Rearrange dimensions so that embed_dim is last (required by
        # nn.Linear) and spatial locations are flattened into a single
        # dimension for batched processing.
        #
        # (B, T, embed_dim, H', W')
        # → permute(0, 1, 3, 4, 2): (B, T, H', W', embed_dim)
        # → reshape: (B, T, H'*W', embed_dim)
        # → permute(0, 2, 1, 3): (B, H'*W', T, embed_dim)
        #
        # The final shape (B, H'*W', T, embed_dim) groups all spatial
        # locations together, allowing the Linear layers to process all
        # spatial positions in a single batched matrix multiply.
        x_perm: torch.Tensor = x.permute(0, 1, 3, 4, 2).contiguous()
        # Shape: (B, T, H', W', embed_dim)

        x_flat: torch.Tensor = x_perm.reshape(
            batch_size, t_actual, h_prime * w_prime, self.embed_dim
        )
        # Shape: (B, T, H'*W', embed_dim)

        x_spatial: torch.Tensor = x_flat.permute(0, 2, 1, 3).contiguous()
        # Shape: (B, H'*W', T, embed_dim)

        # ----------------------------------------------------------------
        # Step 2: Initialize accumulator for the weighted sum
        # ----------------------------------------------------------------
        # z_agg accumulates Σ_t W_t(z_p^t) · cos(γ · t).
        # Shape: (B, H'*W', embed_dim) — same as a single timestep's output.
        # Initialized to zeros; each timestep's contribution is added in-place.
        z_agg: torch.Tensor = torch.zeros(
            batch_size,
            h_prime * w_prime,
            self.embed_dim,
            dtype=x.dtype,
            device=x.device,
        )

        # ----------------------------------------------------------------
        # Step 3: Apply per-timestep transformation and accumulate
        # ----------------------------------------------------------------
        # Loop over T timesteps (T=10 is small, loop overhead is negligible).
        # For each timestep t:
        #   1. Extract features at timestep t: x_t = x_spatial[:, :, t, :]
        #      Shape: (B, H'*W', embed_dim)
        #   2. Apply per-timestep linear transformation W_t:
        #      z_t = temporal_weights[t](x_t)
        #      Shape: (B, H'*W', embed_dim)
        #   3. Compute Fourier modulation coefficient:
        #      cos_t = cos(γ * t)
        #      Shape: (embed_dim,) — per-channel cosine weight
        #   4. Accumulate weighted contribution:
        #      z_agg += z_t * cos_t
        #      Broadcasting: (B, H'*W', embed_dim) * (embed_dim,)
        #                   → (B, H'*W', embed_dim)
        for t_idx in range(t_actual):
            # Extract features at timestep t_idx.
            # x_spatial shape: (B, H'*W', T, embed_dim)
            # x_t shape:       (B, H'*W', embed_dim)
            x_t: torch.Tensor = x_spatial[:, :, t_idx, :]

            # Apply the per-timestep linear transformation W_t.
            # temporal_weights[t_idx] is Linear(embed_dim, embed_dim).
            # Input:  (B, H'*W', embed_dim)
            # Output: (B, H'*W', embed_dim)
            # nn.Linear operates on the last dimension, so the batched
            # (B, H'*W') dimensions are handled automatically.
            z_t: torch.Tensor = self.temporal_weights[t_idx](x_t)

            # Compute the Fourier modulation coefficient for timestep t_idx.
            # fourier_gamma shape: (embed_dim,)
            # t_idx is a Python int scalar; γ * t_idx broadcasts correctly.
            # cos_t shape: (embed_dim,)
            #
            # At t_idx=0: cos(γ * 0) = cos(0) = 1.0 for all channels,
            # ensuring the first timestep always contributes with full weight.
            cos_t: torch.Tensor = torch.cos(
                self.fourier_gamma * float(t_idx)
            )
            # Shape: (embed_dim,)

            # Accumulate weighted contribution.
            # z_t shape:   (B, H'*W', embed_dim)
            # cos_t shape: (embed_dim,)  — broadcasts over (B, H'*W')
            # Result:      (B, H'*W', embed_dim)
            z_agg = z_agg + z_t * cos_t

        # ----------------------------------------------------------------
        # Step 4: Reshape output back to spatial format
        # ----------------------------------------------------------------
        # z_agg shape: (B, H'*W', embed_dim)
        # → reshape to (B, H', W', embed_dim)
        # → permute to (B, embed_dim, H', W')  [channel-first format]
        #
        # The channel-first format (B, embed_dim, H', W') is required by
        # the subsequent FourierLayer, which uses Conv2d-style operations.
        z_agg_spatial: torch.Tensor = z_agg.reshape(
            batch_size, h_prime, w_prime, self.embed_dim
        )
        # Shape: (B, H', W', embed_dim)

        # Permute to channel-first format.
        # (B, H', W', embed_dim) → (B, embed_dim, H', W')
        z_out: torch.Tensor = z_agg_spatial.permute(0, 3, 1, 2).contiguous()
        # Shape: (B, embed_dim, H', W')

        return z_out
