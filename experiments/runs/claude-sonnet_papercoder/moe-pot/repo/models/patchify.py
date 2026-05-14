# models/patchify.py
"""Patchification layer with learnable positional encoding for MoE-POT.

Implements the PatchifyLayer class, which is the first processing stage
in the MoEPOT architecture. It converts raw spatiotemporal PDE fields
into patch-embedded tokens with positional encodings, following the
Vision Transformer (ViT) patchification paradigm adapted for PDE data.

From the paper (Section 4, Input Encoding and Temporal Aggregation):
    "To encode spatial features, we apply a patchification layer with
    positional embeddings inspired by vision transformers:
        Z_p^t = P(u^t + p^t),  t = 1, ..., T
    where P is a convolutional layer, and p^t_{i,j} = W_p(x_i, y_j, t)
    denotes learnable positional encodings."

From config.yaml:
    architecture.patch_size: 8          (P=8, optimal from ablation Table 8)
    architecture.target_resolution: 128 (H=W=128)
    architecture.max_channels: 4        (C=4, padded to match CNS dataset)
    architecture.input_timesteps: 10    (T=10 frames as input)
    models.tiny.attn_dim: 512           (embed_dim for Tiny model)
    models.small.attn_dim: 1024         (embed_dim for Small/Medium models)

Data flow:
    Input:  (B, T, C=4, H=128, W=128)
    Output: (B, T, embed_dim, H/P=16, W/P=16)
"""

import torch
import torch.nn as nn


class PatchifyLayer(nn.Module):
    """Patchification layer with learnable 3D positional encoding.

    Converts a batch of spatiotemporal PDE fields into patch-embedded
    token sequences. Each spatial patch of size P×P is projected to an
    embed_dim-dimensional vector via a strided convolution, then augmented
    with a learnable positional encoding that encodes the patch's (x, y)
    location and timestep t.

    The positional encoding formula from the paper:
        p^t_{i,j} = W_p(x_i, y_j, t)
    is implemented as a Linear(3, embed_dim) layer applied to normalized
    3D coordinates (x_i, y_j, t_norm) at patch resolution.

    The paper formula Z_p^t = P(u^t + p^t) is interpreted as:
        1. Apply Conv2d to u^t: z_conv = conv(u^t)  → (B, embed_dim, H/P, W/P)
        2. Compute positional encoding: p^t = pos_embed_proj([x_i, y_j, t])
           → (embed_dim, H/P, W/P) per timestep
        3. Add: Z_p^t = z_conv + p^t  → (B, embed_dim, H/P, W/P)

    This is the standard ViT-style approach where positional encoding is
    added in the embedding space (after the conv projection), consistent
    with the paper's intent and the DPOT baseline architecture.

    Attributes:
        patch_size: Spatial patch size P. Both kernel_size and stride of
            the embedding conv. Default 8 (config.yaml architecture.patch_size).
        in_channels: Number of input channels C. Default 4 (config.yaml
            architecture.max_channels, padded to match CNS dataset).
        embed_dim: Output embedding dimension. Corresponds to attn_dim in
            config.yaml: 512 (Tiny), 1024 (Small/Medium).
        img_size: Input spatial resolution H=W. Default 128 (config.yaml
            architecture.target_resolution).
        input_timesteps: Number of input timesteps T. Default 10 (config.yaml
            architecture.input_timesteps).
        num_patches_h: Number of patches along height = img_size // patch_size.
            Typically 16 (= 128 // 8).
        num_patches_w: Number of patches along width = img_size // patch_size.
            Typically 16 (= 128 // 8).
        conv: Conv2d(in_channels, embed_dim, kernel_size=patch_size,
            stride=patch_size) — the spatial patch embedding projection P.
        pos_embed_proj: Linear(3, embed_dim) — maps (x_i, y_j, t_norm)
            coordinates to embed_dim-dimensional positional encodings.
            Corresponds to W_p ∈ R^{embed_dim × 3} in the paper.
        coord_buffer: Precomputed coordinate tensor of shape
            (input_timesteps, num_patches_h, num_patches_w, 3) registered
            as a non-trainable buffer. Contains normalized (x_i, y_j, t_norm)
            coordinates for all patch positions and timesteps.
    """

    def __init__(
        self,
        patch_size: int = 8,
        in_channels: int = 4,
        embed_dim: int = 512,
        img_size: int = 128,
        input_timesteps: int = 10,
    ) -> None:
        """Initializes the PatchifyLayer.

        Constructs the spatial embedding convolution, positional encoding
        projection, and precomputes the normalized coordinate buffer for
        all patch positions and timesteps.

        Args:
            patch_size: Spatial patch size P. Used as both kernel_size and
                stride of the embedding convolution, producing non-overlapping
                patches. Default 8 (config.yaml architecture.patch_size).
                Ablation studies test P ∈ {4, 8, 16} (Table 8 in paper).
            in_channels: Number of input channels C after preprocessing.
                Default 4 (config.yaml architecture.max_channels). All
                datasets are padded to this channel count in PDEDataset.
            embed_dim: Embedding dimension for patch tokens. Corresponds to
                attn_dim in config.yaml model configurations:
                  - Tiny:   512  (config.yaml models.tiny.attn_dim)
                  - Small:  1024 (config.yaml models.small.attn_dim)
                  - Medium: 1024 (config.yaml models.medium.attn_dim)
            img_size: Input spatial resolution (H = W). Default 128
                (config.yaml architecture.target_resolution). All datasets
                are resized to this resolution in Preprocessor.resize().
            input_timesteps: Number of input timesteps T. Default 10
                (config.yaml architecture.input_timesteps). Used to
                precompute temporal coordinates in the positional encoding.

        Raises:
            ValueError: If img_size is not divisible by patch_size.
            ValueError: If patch_size <= 0, in_channels <= 0, embed_dim <= 0,
                img_size <= 0, or input_timesteps <= 0.
        """
        super().__init__()

        # --- Input validation ---
        if patch_size <= 0:
            raise ValueError(
                f"patch_size must be positive, got {patch_size}."
            )
        if in_channels <= 0:
            raise ValueError(
                f"in_channels must be positive, got {in_channels}."
            )
        if embed_dim <= 0:
            raise ValueError(
                f"embed_dim must be positive, got {embed_dim}."
            )
        if img_size <= 0:
            raise ValueError(
                f"img_size must be positive, got {img_size}."
            )
        if input_timesteps <= 0:
            raise ValueError(
                f"input_timesteps must be positive, got {input_timesteps}."
            )
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by patch_size "
                f"({patch_size}). Got remainder {img_size % patch_size}."
            )

        # Store configuration attributes.
        self.patch_size: int = patch_size
        self.in_channels: int = in_channels
        self.embed_dim: int = embed_dim
        self.img_size: int = img_size
        self.input_timesteps: int = input_timesteps

        # Derived spatial dimensions of the token grid after patchification.
        # With img_size=128 and patch_size=8: num_patches_h = num_patches_w = 16.
        self.num_patches_h: int = img_size // patch_size
        self.num_patches_w: int = img_size // patch_size

        # ----------------------------------------------------------------
        # Spatial Embedding Convolution (the P operator in the paper)
        # ----------------------------------------------------------------
        # Conv2d with kernel_size=stride=patch_size implements non-overlapping
        # patch projection: each P×P spatial patch → one embed_dim token.
        # Input:  (B*T, in_channels, H, W)
        # Output: (B*T, embed_dim, H/P, W/P) = (B*T, embed_dim, 16, 16)
        # bias=True: standard for patch embedding projections.
        self.conv: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        # ----------------------------------------------------------------
        # Positional Encoding Projection (W_p in the paper)
        # ----------------------------------------------------------------
        # Linear(3, embed_dim) maps normalized 3D coordinates (x_i, y_j, t)
        # to embed_dim-dimensional positional encodings.
        # Paper notation: W_p ∈ R^{n×3} where n = embed_dim.
        # This is the learnable component of the positional encoding;
        # the coordinate grid itself is a fixed precomputed buffer.
        self.pos_embed_proj: nn.Linear = nn.Linear(
            in_features=3,
            out_features=embed_dim,
            bias=True,
        )

        # ----------------------------------------------------------------
        # Precomputed Coordinate Buffer
        # ----------------------------------------------------------------
        # Build the full (T, H/P, W/P, 3) coordinate tensor once in __init__
        # and register as a non-trainable buffer. This avoids repeated
        # construction during each forward pass and ensures the tensor
        # automatically moves to the correct device with the model.
        #
        # Coordinate normalization:
        #   - Spatial: x_i ∈ [0, 1] for i = 0, ..., num_patches_h - 1
        #              y_j ∈ [0, 1] for j = 0, ..., num_patches_w - 1
        #   - Temporal: t_norm = t / max(T-1, 1) ∈ [0, 1] for t = 0, ..., T-1
        # Normalization ensures the linear projection receives inputs in a
        # consistent range regardless of resolution or sequence length.
        coord_buffer: torch.Tensor = self._build_coord_buffer()
        self.register_buffer("coord_buffer", coord_buffer)

    def _build_coord_buffer(self) -> torch.Tensor:
        """Precomputes the normalized 3D coordinate tensor for all patches.

        Builds a tensor of shape (T, H/P, W/P, 3) containing normalized
        (x_i, y_j, t_norm) coordinates for every combination of timestep
        and patch position. This tensor is registered as a buffer and
        used in every forward pass to compute positional encodings.

        Coordinate ranges:
          - x_i: Normalized patch center x-coordinate ∈ [0, 1].
            Computed as linspace(0, 1, num_patches_h).
          - y_j: Normalized patch center y-coordinate ∈ [0, 1].
            Computed as linspace(0, 1, num_patches_w).
          - t_norm: Normalized timestep ∈ [0, 1].
            Computed as t / max(T-1, 1) for t = 0, ..., T-1.
            Uses max(T-1, 1) to handle the edge case T=1 without division
            by zero.

        Returns:
            Float32 tensor of shape (input_timesteps, num_patches_h,
            num_patches_w, 3). The last dimension contains [x_i, y_j, t_norm]
            for each (t, i, j) combination.
        """
        t: int = self.input_timesteps
        h: int = self.num_patches_h
        w: int = self.num_patches_w

        # Normalized spatial coordinates at patch resolution.
        # x_coords shape: (num_patches_h,) — values in [0, 1]
        # y_coords shape: (num_patches_w,) — values in [0, 1]
        x_coords: torch.Tensor = torch.linspace(0.0, 1.0, h)
        y_coords: torch.Tensor = torch.linspace(0.0, 1.0, w)

        # Create 2D spatial meshgrid.
        # grid_x shape: (num_patches_h, num_patches_w) — x varies along dim 0
        # grid_y shape: (num_patches_h, num_patches_w) — y varies along dim 1
        # indexing='ij': grid_x[i, j] = x_coords[i], grid_y[i, j] = y_coords[j]
        grid_x: torch.Tensor
        grid_y: torch.Tensor
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="ij")
        # Both shapes: (num_patches_h, num_patches_w)

        # Normalized temporal coordinates.
        # t_norm shape: (T,) — values in [0, 1]
        # max(T-1, 1) prevents division by zero when T=1.
        t_indices: torch.Tensor = torch.arange(t, dtype=torch.float32)
        t_norm: torch.Tensor = t_indices / max(t - 1, 1)

        # Build the full (T, H/P, W/P, 3) coordinate tensor.
        # Strategy: expand spatial grid across T, expand temporal coord
        # across (H/P, W/P), then stack along the last dimension.

        # Expand spatial grids to (T, H/P, W/P):
        # grid_x: (H/P, W/P) → (1, H/P, W/P) → (T, H/P, W/P)
        grid_x_expanded: torch.Tensor = grid_x.unsqueeze(0).expand(t, h, w)
        grid_y_expanded: torch.Tensor = grid_y.unsqueeze(0).expand(t, h, w)

        # Expand temporal coordinate to (T, H/P, W/P):
        # t_norm: (T,) → (T, 1, 1) → (T, H/P, W/P)
        t_norm_expanded: torch.Tensor = t_norm.view(t, 1, 1).expand(t, h, w)

        # Stack along last dimension to get (T, H/P, W/P, 3).
        # coord_buffer[t_idx, i, j] = [x_i, y_j, t_norm_t_idx]
        coord_buffer: torch.Tensor = torch.stack(
            [grid_x_expanded, grid_y_expanded, t_norm_expanded],
            dim=-1,
        )
        # Shape: (T, num_patches_h, num_patches_w, 3)

        return coord_buffer.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies patch embedding and positional encoding to the input.

        Implements the paper formula Z_p^t = P(u^t + p^t) as:
            1. Apply Conv2d to all timesteps: z_conv = conv(u^t)
            2. Compute positional encodings: p^t = pos_embed_proj(coords^t)
            3. Add: Z_p^t = z_conv + p^t

        The positional encoding is added in the embedding space (after the
        conv projection), which is the standard ViT-style approach and
        consistent with the paper's intent.

        Processing pipeline:
            (B, T, C, H, W)
            → reshape to (B*T, C, H, W)
            → Conv2d → (B*T, embed_dim, H/P, W/P)
            → reshape to (B, T, embed_dim, H/P, W/P)
            → + pos_enc (T, embed_dim, H/P, W/P)  [broadcast over B]
            → (B, T, embed_dim, H/P, W/P)

        Args:
            x: Input spatiotemporal tensor of shape (B, T, C, H, W) where:
                - B: Batch size (up to 20 for pre-training, config.yaml
                  pretraining.batch_size).
                - T: Number of input timesteps = 10 (config.yaml
                  architecture.input_timesteps).
                - C: Number of channels = 4 (config.yaml
                  architecture.max_channels, padded in PDEDataset).
                - H: Spatial height = 128 (config.yaml
                  architecture.target_resolution).
                - W: Spatial width = 128 (config.yaml
                  architecture.target_resolution).

        Returns:
            Patch-embedded token tensor of shape (B, T, embed_dim, H/P, W/P)
            = (B, 10, embed_dim, 16, 16) with default config values.
            This is Z_p^t from the paper, ready for TemporalAggregation.
        """
        batch_size: int = x.shape[0]
        t: int = x.shape[1]
        # c: int = x.shape[2]  # in_channels, not used directly
        # h: int = x.shape[3]  # img_size, not used directly
        # w: int = x.shape[4]  # img_size, not used directly

        # ----------------------------------------------------------------
        # Step 1: Reshape for batched convolution
        # ----------------------------------------------------------------
        # Merge batch and time dimensions so Conv2d processes all (B*T)
        # frames in a single batched operation.
        # Input:  (B, T, C, H, W)
        # Output: (B*T, C, H, W)
        x_reshaped: torch.Tensor = x.view(batch_size * t, self.in_channels,
                                           self.img_size, self.img_size)

        # ----------------------------------------------------------------
        # Step 2: Apply spatial patch embedding convolution
        # ----------------------------------------------------------------
        # Conv2d with kernel_size=stride=patch_size projects each P×P patch
        # to an embed_dim-dimensional token. No overlap between patches.
        # Input:  (B*T, in_channels, H, W)
        # Output: (B*T, embed_dim, H/P, W/P) = (B*T, embed_dim, 16, 16)
        z_conv: torch.Tensor = self.conv(x_reshaped)

        # Reshape back to separate batch and time dimensions.
        # (B*T, embed_dim, H/P, W/P) → (B, T, embed_dim, H/P, W/P)
        z_conv = z_conv.view(
            batch_size, t, self.embed_dim,
            self.num_patches_h, self.num_patches_w
        )

        # ----------------------------------------------------------------
        # Step 3: Compute positional encodings
        # ----------------------------------------------------------------
        # The precomputed coord_buffer has shape (T, H/P, W/P, 3) and
        # contains normalized (x_i, y_j, t_norm) coordinates for all
        # patch positions and timesteps.
        #
        # Apply pos_embed_proj (Linear 3 → embed_dim) to the coordinate
        # tensor. The Linear layer operates on the last dimension.
        # Input:  (T, H/P, W/P, 3)
        # Output: (T, H/P, W/P, embed_dim)
        #
        # self.coord_buffer is automatically on the correct device because
        # it was registered via register_buffer in __init__.
        pos_enc_raw: torch.Tensor = self.pos_embed_proj(self.coord_buffer)
        # Shape: (T, num_patches_h, num_patches_w, embed_dim)

        # Permute to (T, embed_dim, H/P, W/P) for broadcasting with z_conv.
        # The embed_dim dimension must be at position 1 (channel-first format).
        pos_enc: torch.Tensor = pos_enc_raw.permute(0, 3, 1, 2).contiguous()
        # Shape: (T, embed_dim, num_patches_h, num_patches_w)

        # Handle the case where the forward pass uses a different T than
        # the precomputed buffer (e.g., during inference with variable T).
        # Slice the buffer to match the actual T in the input.
        if t != self.input_timesteps:
            # Use only the first t timesteps of the precomputed buffer.
            # This handles edge cases where T < input_timesteps.
            # If T > input_timesteps, we need to recompute — but this
            # should not happen in normal usage.
            if t <= self.input_timesteps:
                pos_enc = pos_enc[:t]
            else:
                # Recompute for the larger T on-the-fly.
                # This is a fallback for unusual inference scenarios.
                pos_enc = self._compute_pos_enc_for_t(t, x.device)

        # ----------------------------------------------------------------
        # Step 4: Add positional encoding to conv output
        # ----------------------------------------------------------------
        # z_conv shape:  (B, T, embed_dim, H/P, W/P)
        # pos_enc shape: (T, embed_dim, H/P, W/P)
        #
        # Unsqueeze pos_enc to (1, T, embed_dim, H/P, W/P) for broadcasting
        # over the batch dimension B.
        pos_enc_broadcast: torch.Tensor = pos_enc.unsqueeze(0)
        # Shape: (1, T, embed_dim, num_patches_h, num_patches_w)

        # Element-wise addition with broadcasting over B.
        # z_conv:              (B, T, embed_dim, H/P, W/P)
        # pos_enc_broadcast:   (1, T, embed_dim, H/P, W/P)
        # Result:              (B, T, embed_dim, H/P, W/P)
        z_out: torch.Tensor = z_conv + pos_enc_broadcast

        return z_out

    def _compute_pos_enc_for_t(
        self,
        t: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Computes positional encodings for an arbitrary number of timesteps.

        Fallback method for cases where the forward pass uses a different
        number of timesteps than the precomputed buffer. Recomputes the
        coordinate tensor on-the-fly for the given T.

        This method is not called in normal training/evaluation (where T
        is always equal to input_timesteps=10). It handles edge cases
        during inference with variable-length sequences.

        Args:
            t: Number of timesteps to compute encodings for.
            device: Target device for the output tensor.

        Returns:
            Positional encoding tensor of shape (t, embed_dim, H/P, W/P)
            on the specified device.
        """
        h: int = self.num_patches_h
        w: int = self.num_patches_w

        # Normalized spatial coordinates.
        x_coords: torch.Tensor = torch.linspace(0.0, 1.0, h, device=device)
        y_coords: torch.Tensor = torch.linspace(0.0, 1.0, w, device=device)
        grid_x: torch.Tensor
        grid_y: torch.Tensor
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="ij")

        # Normalized temporal coordinates.
        t_indices: torch.Tensor = torch.arange(
            t, dtype=torch.float32, device=device
        )
        t_norm: torch.Tensor = t_indices / max(t - 1, 1)

        # Expand and stack to (t, H/P, W/P, 3).
        grid_x_exp: torch.Tensor = grid_x.unsqueeze(0).expand(t, h, w)
        grid_y_exp: torch.Tensor = grid_y.unsqueeze(0).expand(t, h, w)
        t_norm_exp: torch.Tensor = t_norm.view(t, 1, 1).expand(t, h, w)

        coords: torch.Tensor = torch.stack(
            [grid_x_exp, grid_y_exp, t_norm_exp], dim=-1
        )
        # Shape: (t, H/P, W/P, 3)

        # Apply positional encoding projection.
        # Input:  (t, H/P, W/P, 3)
        # Output: (t, H/P, W/P, embed_dim)
        pos_enc_raw: torch.Tensor = self.pos_embed_proj(coords)

        # Permute to (t, embed_dim, H/P, W/P).
        pos_enc: torch.Tensor = pos_enc_raw.permute(0, 3, 1, 2).contiguous()

        return pos_enc
