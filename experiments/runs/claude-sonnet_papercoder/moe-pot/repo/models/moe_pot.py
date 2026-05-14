```python
# models/moe_pot.py
"""Top-level MoE-POT model assembly.

Implements MoEBlock (a single transformer block with Fourier + MoE sublayers)
and MoEPOT (the full model) as described in Section 4 of the paper:

    "Our proposed model architecture begins by processing raw data through a
    patchification layer and a temporal aggregation layer, which reduces
    spatial-temporal resolution and extracts dynamic structures inherent to
    PDEs. The processed features are then passed through N blocks, each of
    which contains a Fourier layer and a MoE layer."

Architecture overview:
    Input (B, T, C, H, W)
        → PatchifyLayer       → (B, T, embed_dim, H/P, W/P)
        → TemporalAggregation → (B, embed_dim, H/P, W/P)
        → N × MoEBlock        → (B, embed_dim, H/P, W/P)
        → ConvTranspose2d     → (B, C, H, W)

Each MoEBlock:
    x → pre-norm → FourierLayer → residual → pre-norm → MoELayer → residual

From config.yaml (architecture section):
    patch_size: 8
    input_timesteps: 10
    target_resolution: 128
    max_channels: 4
    modes_x: 8
    modes_y: 8
    load_balance_weight: 0.1

From config.yaml (models section, e.g. tiny):
    attn_dim: 512
    mlp_dim: 512
    num_layers: 4
    num_heads: 4
    num_routed_experts: 16
    num_shared_experts: 2
    top_k: 4
"""

from typing import Any, Tuple

import torch
import torch.nn as nn

from models.fourier_layer import FourierLayer
from models.moe_layer import MoELayer
from models.patchify import PatchifyLayer
from models.temporal_agg import TemporalAggregation


class MoEBlock(nn.Module):
    """Single transformer block containing a Fourier sublayer and a MoE sublayer.

    Implements the repeating unit stacked N times in MoEPOT. Each block
    applies pre-norm before each sublayer and adds a residual connection
    after each sublayer:

        x → LayerNorm → FourierLayer → + x  (residual)
          → LayerNorm → MoELayer     → + x  (residual)

    The MoELayer returns a (output, balance_loss) tuple; the balance_loss
    is passed up to MoEPOT for accumulation across all N blocks.

    Attributes:
        fourier_layer: Multi-head spectral convolution layer. Computes
            z_0^l from z^l via the Fourier domain two-layer MLP.
        moe_layer: Sparse MoE layer with shared and routed CNN experts.
            Computes z^{l+1} from z_0^l via expert aggregation.
        norm1: LayerNorm(embed_dim) applied before the Fourier sublayer.
        norm2: LayerNorm(embed_dim) applied before the MoE sublayer.
        embed_dim: Feature dimension for this block.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        mlp_dim: int = 512,
        num_heads: int = 4,
        modes_x: int = 8,
        modes_y: int = 8,
        num_routed_experts: int = 16,
        num_shared_experts: int = 2,
        top_k: int = 4,
        load_balance_weight: float = 0.1,
    ) -> None:
        """Initializes a MoEBlock with Fourier and MoE sublayers.

        Args:
            embed_dim: Feature dimension d_z. Corresponds to attn_dim in
                config.yaml model configurations:
                  - Tiny:   512  (config.yaml models.tiny.attn_dim)
                  - Small:  1024 (config.yaml models.small.attn_dim)
                  - Medium: 1024 (config.yaml models.medium.attn_dim)
            mlp_dim: Hidden channel dimension inside each ExpertCNN.
                Corresponds to mlp_dim in config.yaml:
                  - Tiny:   512  (config.yaml models.tiny.mlp_dim)
                  - Small:  1024 (config.yaml models.small.mlp_dim)
                  - Medium: 2048 (config.yaml models.medium.mlp_dim)
            num_heads: Number of spectral heads in FourierLayer.
                Corresponds to num_heads in config.yaml:
                  - Tiny:   4 (config.yaml models.tiny.num_heads)
                  - Small:  8 (config.yaml models.small.num_heads)
                  - Medium: 8 (config.yaml models.medium.num_heads)
            modes_x: Number of Fourier modes in x-direction. Default 8
                (config.yaml architecture.modes_x).
            modes_y: Number of Fourier modes in y-direction. Default 8
                (config.yaml architecture.modes_y).
            num_routed_experts: Number of routed experts N_r. Default 16
                (config.yaml models.*.num_routed_experts).
            num_shared_experts: Number of shared experts N_s. Default 2
                (config.yaml models.*.num_shared_experts).
            top_k: Number of routed experts activated per input K. Default 4
                (config.yaml models.*.top_k).
            load_balance_weight: Scaling factor w_bal for CV² loss. Default 0.1
                (config.yaml architecture.load_balance_weight).

        Raises:
            ValueError: If embed_dim <= 0 or mlp_dim <= 0.
            ValueError: If num_heads <= 0 or embed_dim % num_heads != 0.
        """
        super().__init__()

        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}.")
        if mlp_dim <= 0:
            raise ValueError(f"mlp_dim must be positive, got {mlp_dim}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads "
                f"({num_heads})."
            )

        self.embed_dim: int = embed_dim

        # Pre-norm LayerNorm instances.
        # LayerNorm(embed_dim) normalizes over the last dimension.
        # Since spatial tensors are (B, embed_dim, H', W'), we permute
        # to (B, H', W', embed_dim) before applying norm, then permute back.
        self.norm1: nn.LayerNorm = nn.LayerNorm(embed_dim)
        self.norm2: nn.LayerNorm = nn.LayerNorm(embed_dim)

        # Fourier sublayer: global spectral convolution with skip connection.
        # Input/output shape: (B, embed_dim, H', W')
        self.fourier_layer: FourierLayer = FourierLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            modes_x=modes_x,
            modes_y=modes_y,
        )

        # MoE sublayer: sparse expert aggregation.
        # Input shape: (B, embed_dim, H', W')
        # Output: ((B, embed_dim, H', W'), scalar balance_loss)
        self.moe_layer: MoELayer = MoELayer(
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            num_routed_experts=num_routed_experts,
            num_shared_experts=num_shared_experts,
            top_k=top_k,
            load_balance_weight=load_balance_weight,
        )

    def _apply_prenorm(self, x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        """Applies LayerNorm to a spatial tensor (B, C, H', W').

        LayerNorm expects the normalized dimension to be last. This helper
        permutes the spatial tensor to (B, H', W', C), applies norm, then
        permutes back to (B, C, H', W').

        Args:
            x: Input tensor of shape (B, embed_dim, H', W').
            norm: LayerNorm(embed_dim) instance.

        Returns:
            Normalized tensor of shape (B, embed_dim, H', W').
        """
        # (B, embed_dim, H', W') → (B, H', W', embed_dim)
        x_perm: torch.Tensor = x.permute(0, 2, 3, 1).contiguous()
        # Apply LayerNorm over the last (embed_dim) dimension.
        x_normed: torch.Tensor = norm(x_perm)
        # (B, H', W', embed_dim) → (B, embed_dim, H', W')
        return x_normed.permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the Fourier and MoE sublayers with pre-norm and residuals.

        Implements the pre-norm transformer block:
            x = x + FourierLayer(norm1(x))
            x, balance_loss = x + MoELayer(norm2(x))

        Args:
            x: Input feature map of shape (B, embed_dim, H', W') where:
                - B: Batch size.
                - embed_dim: Feature dimension (attn_dim from config).
                - H': Token grid height, typically 16 (= 128 / 8).
                - W': Token grid width, typically 16.

        Returns:
            A tuple (output, balance_loss) where:
              - output: Feature map of shape (B, embed_dim, H', W').
                Same shape as input, enabling stacking of N blocks.
              - balance_loss: Scalar tensor — the load balancing loss for
                this block's MoE layer. Accumulated across all N blocks
                in MoEPOT.forward() and added to the prediction loss.
        """
        # ----------------------------------------------------------------
        # Fourier sublayer with pre-norm and residual connection
        # ----------------------------------------------------------------
        # Pre-norm: normalize over embed_dim (last dim after permute).
        x_norm1: torch.Tensor = self._apply_prenorm(x, self.norm1)

        # Apply Fourier layer: global spectral convolution + skip.
        # Input/output: (B, embed_dim, H', W')
        fourier_out: torch.Tensor = self.fourier_layer(x_norm1)

        # Residual connection: x = x + FourierLayer(norm1(x))
        x = x + fourier_out

        # ----------------------------------------------------------------
        # MoE sublayer with pre-norm and residual connection
        # ----------------------------------------------------------------
        # Pre-norm: normalize over embed_dim.
        x_norm2: torch.Tensor = self._apply_prenorm(x, self.norm2)

        # Apply MoE layer: sparse expert aggregation.
        # Returns (output, balance_loss) per the MoELayer contract.
        moe_out: torch.Tensor
        balance_loss: torch.Tensor
        moe_out, balance_loss = self.moe_layer(x_norm2)

        # Residual connection: x = x + MoELayer(norm2(x))
        x = x + moe_out

        return x, balance_loss


class MoEPOT(nn.Module):
    """Mixture-of-Experts Pre-training Operator Transformer.

    Full model that processes spatiotemporal PDE solution trajectories
    auto-regressively. Takes T input frames and predicts the next frame.

    Architecture:
        Input (B, T, C, H, W)
            → PatchifyLayer       → (B, T, embed_dim, H/P, W/P)
            → TemporalAggregation → (B, embed_dim, H/P, W/P)
            → N × MoEBlock        → (B, embed_dim, H/P, W/P)
            → ConvTranspose2d     → (B, C, H, W)

    The model is designed for pre-training on 6 heterogeneous PDE datasets
    simultaneously. The MoE architecture decouples model capacity from
    inference cost: total parameters scale with num_routed_experts, but
    only top_k experts are activated per input (25% of routed experts).

    Key properties:
      - All datasets padded to max_channels=4 (CNS has 4: rho, vx, vy, p).
      - Spatial resolution standardized to 128×128 before input.
      - Patch size P=8 gives 16×16 token grid after patchification.
      - T=10 input frames; predicts frame T+1.
      - Load balancing loss accumulated across all N blocks.
      - Router frozen during fine-tuning (freeze_router() method).

    Attributes:
        config: Configuration object with all hyperparameters.
        in_channels: Number of input/output channels (max_channels=4).
        embed_dim: Feature dimension (attn_dim from config).
        patch_size: Spatial patch size P (default 8).
        patchify: PatchifyLayer for spatial patch embedding + positional enc.
        temporal_agg: TemporalAggregation for collapsing T timesteps.
        blocks: ModuleList of N MoEBlock instances.
        output_proj: ConvTranspose2d for upsampling back to original resolution.
    """

    def __init__(self, config: Any) -> None:
        """Initializes MoEPOT from a configuration object.

        Reads all hyperparameters from config attributes, which correspond
        to the YAML configuration file keys. The config object must have
        the following attributes (all present in config.yaml):

        From config.yaml models.{size} section:
            config.attn_dim: int          (embed_dim)
            config.mlp_dim: int           (expert hidden dim)
            config.num_layers: int        (number of MoEBlocks)
            config.num_heads: int         (Fourier layer heads)
            config.num_routed_experts: int
            config.num_shared_experts: int
            config.top_k: int

        From config.yaml architecture section:
            config.patch_size: int        (default 8)
            config.input_timesteps: int   (default 10)
            config.target_resolution: int (default 128)
            config.max_channels: int      (default 4)
            config.modes_x: int           (default 8)
            config.modes_y: int           (default 8)
            config.load_balance_weight: float (default 0.1)

        Args:
            config: Configuration object. Typically a dataclass or
                SimpleNamespace loaded from config.yaml via Config.from_yaml().
                Must expose all attributes listed above.

        Raises:
            AttributeError: If config is missing required attributes.
            ValueError: If derived dimensions are inconsistent (e.g.,
                embed_dim not divisible by num_heads).
        """
        super().__init__()

        # Store config for use in count_parameters() and get_router_weights().
        self.config: Any = config

        # ----------------------------------------------------------------
        # Extract hyperparameters from config
        # ----------------------------------------------------------------
        # Model architecture dimensions (from config.yaml models.{size}).
        embed_dim: int = int(config.attn_dim)
        mlp_dim: int = int(config.mlp_dim)
        num_layers: int = int(config.num_layers)
        num_heads: int = int(config.num_heads)
        num_routed_experts: int = int(config.num_routed_experts)
        num_shared_experts: int = int(config.num_shared_experts)
        top_k: int = int(config.top_k)

        # Architecture hyperparameters (from config.yaml architecture).
        patch_size: int = int(config.patch_size)
        input_timesteps: int = int(config.input_timesteps)
        target_resolution: int = int(config.target_resolution)
        in_channels: int = int(config.max_channels)
        modes_x: int = int(config.modes_x)
        modes_y: int = int(config.modes_y)
        load_balance_weight: float = float(config.load_balance_weight)

        # Store frequently accessed dimensions as instance attributes.
        self.in_channels: int = in_channels
        self.embed_dim: int = embed_dim
        self.patch_size: int = patch_size
        self.input_timesteps: int = input_timesteps
        self.target_resolution: int = target_resolution
        self.num_layers: int = num_layers

        # ----------------------------------------------------------------
        # Patchification Layer
        # ----------------------------------------------------------------
        # Converts (B, T, C, H, W) → (B, T, embed_dim, H/P, W/P).
        # Uses Conv2d(in_channels, embed_dim, kernel=P, stride=P) for
        # non-overlapping patch projection, plus learnable 3D positional
        # encoding W_p(x_i, y_j, t) → embed_dim.
        self.patchify: PatchifyLayer = PatchifyLayer(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            img_size=target_resolution,
            input_timesteps=input_timesteps,
        )

        # ----------------------------------------------------------------
        # Temporal Aggregation Layer
        # ----------------------------------------------------------------
        # Collapses (B, T, embed_dim, H', W') → (B, embed_dim, H', W').
        # Uses per-timestep Linear(embed_dim, embed_dim) layers modulated
        # by learnable Fourier frequencies γ ∈ R^{embed_dim}:
        #   z_agg = Σ_t W_t(z_p^t) · cos(γ · t)
        self.temporal_agg: TemporalAggregation = TemporalAggregation(
            embed_dim=embed_dim,
            input_timesteps=input_timesteps,
        )

        # ----------------------------------------------------------------
        # N MoEBlocks (the core transformer stack)
        # ----------------------------------------------------------------
        # Each block: pre-norm → FourierLayer → residual
        #             pre-norm → MoELayer     → residual
        # All blocks share the same architecture but have independent params.
        self.blocks: nn.ModuleList = nn.ModuleList(
            [
                MoEBlock(
                    embed_dim=embed_dim,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    modes_x=modes_x,
                    modes_y=modes_y,
                    num_routed_experts=num_routed_experts,
                    num_shared_experts=num_shared_experts,
                    top_k=top_k,
                    load_balance_weight=load_balance_weight,
                )
                for _ in range(num_layers)
            ]
        )

        # ----------------------------------------------------------------
        # Output Projection (inverse of patchification)
        # ----------------------------------------------------------------
        # ConvTranspose2d(embed_dim, in_channels, kernel=P, stride=P)
        # upsamples (B, embed_dim, H/P, W/P) → (B, in_channels, H, W).
        # With P=8: 16×16 → 128×128 in a single learned transposed conv.
        # This is the natural inverse of the Conv2d patchification.
        self.output_proj: nn.ConvTranspose2d = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=in_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(
        self,
        u_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts the next PDE frame from T input frames.

        Implements the auto-regressive operator G_w(u^{<T}) = u^T from
        Section 2.2 of the paper. The model takes T consecutive frames as
        input and predicts the next frame.

        Processing pipeline:
            (B, T, C, H, W)
            → PatchifyLayer       → (B, T, embed_dim, H/P, W/P)
            → TemporalAggregation → (B, embed_dim, H/P, W/P)
            → N × MoEBlock        → (B, embed_dim, H/P, W/P)
            → ConvTranspose2d     → (B, C, H, W)

        The total balance loss is the sum of per-block balance losses:
            total_balance_loss = Σ_{l=1}^{N} L_balance^l

        This is added to the prediction loss in the training loop:
            L = L_pred + total_balance_loss

        Args:
            u_input: Input spatiotemporal tensor of shape (B, T, C, H, W)
                where:
                - B: Batch size (up to 20 for pre-training, config.yaml
                  pretraining.batch_size).
                - T: Number of input timesteps = 10 (config.yaml
                  architecture.input_timesteps).
                - C: Number of channels = 4 (config.yaml
                  architecture.max_channels, padded in PDEDataset).
                - H: Spatial height = 128 (config.yaml
                  architecture.target_resolution).
                - W: Spatial width = 128.

        Returns:
            A tuple (u_pred, total_balance_loss) where:
              - u_pred: Predicted next frame of shape (B, C, H, W) =
                (B, 4, 128, 128). This is u^T = G_w(u^0, ..., u^{T-1}).
              - total_balance_loss: Scalar tensor — sum of load balancing
                losses across all N MoEBlocks. Added to prediction loss
                during pre-training. During fine-tuning with frozen router,
                this is still returned but typically ignored or zero-weighted.
        """
        # ----------------------------------------------------------------
        # Step 1: Patchification with positional encoding
        # ----------------------------------------------------------------
        # Input:  (B, T, C, H, W)
        # Output: (B, T, embed_dim, H/P, W/P) = (B, 10, embed_dim, 16, 16)
        # Each timestep is independently patchified and augmented with
        # learnable 3D positional encodings p^t_{i,j} = W_p(x_i, y_j, t).
        z_patches: torch.Tensor = self.patchify(u_input)
        # Shape: (B, T, embed_dim, H', W')

        # ----------------------------------------------------------------
        # Step 2: Temporal aggregation
        # ----------------------------------------------------------------
        # Input:  (B, T, embed_dim, H', W')
        # Output: (B, embed_dim, H', W') = (B, embed_dim, 16, 16)
        # Collapses T timesteps into a single feature map encoding the
        # PDE trajectory dynamics via: z_agg = Σ_t W_t(z_p^t) · cos(γ·t)
        z_agg: torch.Tensor = self.temporal_agg(z_patches)
        # Shape: (B, embed_dim, H', W')

        # ----------------------------------------------------------------
        # Step 3: N MoEBlocks (transformer stack)
        # ----------------------------------------------------------------
        # Each block applies: pre-norm → FourierLayer → residual
        #                     pre-norm → MoELayer     → residual
        # Balance losses are accumulated across all N blocks.
        x: torch.Tensor = z_agg
        total_balance_loss: torch.Tensor = torch.tensor(
            0.0, dtype=x.dtype, device=x.device
        )

        block: MoEBlock
        for block in self.blocks:
            block_out: torch.Tensor
            block_balance_loss: torch.Tensor
            block_out, block_balance_loss = block(x)
            x = block_out
            total_balance_loss = total_balance_loss + block_balance_loss
        # x shape: (B, embed_dim, H', W')

        # ----------------------------------------------------------------
        # Step 4: Output projection (inverse patchification)
        # ----------------------------------------------------------------
        # ConvTranspose2d upsamples (B, embed_dim, H/P, W/P) → (B, C, H, W).
        # With embed_dim=512/1024, in_channels=4, P=8:
        #   (B, embed_dim, 16, 16) → (B, 4, 128, 128)
        u_pred: torch.Tensor = self.output_proj(x)
        # Shape: (B, in_channels, H, W) = (B, 4, 128, 128)

        return u_pred, total_balance_loss

    def get_router_weights(
        self,
        x: torch.Tensor,
        block_idx: int,
    ) -> torch.Tensor:
        """Extracts the full softmax routing distribution from a specific block.

        Runs a partial forward pass up to block_idx and returns the router's
        full softmax output (all num_routed_experts=16 weights) for the
        given input. Used by InterpretabilityAnalyzer to compute routing
        fingerprints for dataset classification (Appendix B.4).

        The routing vector Y_{ij} ∈ R^{16} for sample j in dataset i is
        used to compute the mean routing vector Y_i = (1/N_i) Σ_j Y_{ij},
        which serves as the dataset's routing fingerprint. New inputs are
        classified by finding the nearest Y_i via cross-entropy distance.

        This method does NOT apply torch.no_grad() internally — the caller
        is responsible for wrapping in torch.no_grad() during evaluation.

        Args:
            x: Input tensor of shape (B, T, C, H, W). Same format as
                the input to forward(). Typically a single batch from
                a test DataLoader.
            block_idx: Index of the block from which to extract routing
                weights. 0-indexed: block_idx=0 extracts from the first
                block, block_idx=1 from the second, etc.
                Paper reports 97.7% classification accuracy at block_idx=1
                (Block 2 in 1-indexed notation, config.yaml
                interpretability.analysis_block_idx: 2 → 0-indexed: 1).

        Returns:
            Full softmax routing distribution of shape (B, num_routed_experts)
            = (B, 16). Values are in (0, 1) and sum to 1.0 along dim=-1
            for each sample. This is w^l(z_0^l(x)) = Softmax(s^l(z_0^l(x)))
            from the paper, computed at the specified block.

        Raises:
            IndexError: If block_idx is out of range [0, num_layers).
        """
        # Validate block_idx.
        if block_idx < 0 or block_idx >= len(self.blocks):
            raise IndexError(
                f"block_idx={block_idx} is out of range. "
                f"Model has {len(self.blocks)} blocks (indices 0 to "
                f"{len(self.blocks) - 1})."
            )

        # ----------------------------------------------------------------
        # Partial forward pass: patchify → temporal_agg → blocks[0..block_idx-1]
        # ----------------------------------------------------------------
        # Step 1: Patchification.
        # Input:  (B, T, C, H, W)
        # Output: (B, T, embed_dim, H', W')
        z_patches: torch.Tensor = self.patchify(x)

        # Step 2: Temporal aggregation.
        # Input:  (B, T, embed_dim, H', W')
        # Output: (B, embed_dim, H', W')
        z_agg: torch.Tensor = self.temporal_agg(z_patches)

        # Step 3: Run blocks before block_idx (if any).
        # These blocks transform the feature map but we don't need their
        # routing weights — only the routing at block_idx matters.
        feat: torch.Tensor = z_agg
        for i in range(block_idx):
            feat_out: torch.Tensor
            _: torch.Tensor
            feat_out, _ = self.blocks[i](feat)
            feat = feat_out
        # feat shape: (B, embed_dim, H', W')

        # ----------------------------------------------------------------
        # Extract routing weights from block_idx
        # ----------------------------------------------------------------
        # At block_idx, we need to:
        #   1. Apply norm1 → FourierLayer → residual (Fourier sublayer)
        #   2. Apply norm2 (pre-norm for MoE sublayer)
        #   3. Pass through the router to get routing weights
        # We do NOT run the full MoE layer — just the router portion.

        target_block: MoEBlock = self.blocks[block_idx]

        # Step 4a: Fourier sublayer (same as MoEBlock.forward step 1).
        # Pre-norm before Fourier layer.
        feat_norm1: torch.Tensor = target_block._apply_prenorm(
            feat, target_block.norm1
        )
        # Apply Fourier layer.
        fourier_out: torch.Tensor = target_block.fourier_layer(feat_norm1)
        # Residual connection.
        feat_after_fourier: torch.Tensor = feat + fourier_out

        # Step 4b: Pre-norm before MoE layer.
        feat_norm2: torch.Tensor = target_block._apply_prenorm(
            feat_after_fourier, target_block.norm2
        )
        # feat_norm2 shape: (B, embed_dim, H', W')
        # This is z_0^l(x) from the paper — the input to the MoE layer.

        # Step 4c: Run only the router (not the full MoE layer).
        # RouterGating.forward() returns (logits, full_softmax).
        # We return only full_softmax for interpretability analysis.
        _logits: torch.Tensor
        full_softmax: torch.Tensor
        _logits, full_softmax = target_block.moe_layer.router(feat_norm2)
        # full_softmax shape: (B, num_routed_experts) = (B, 16)

        return full_softmax

    def freeze_router(self) -> None:
        """Freezes all router-gating network parameters across all blocks.

        Sets requires_grad=False for all parameters in each block's
        RouterGating network. Called by Finetuner.__init__() before
        optimizer creation to exclude router parameters from optimization.

        From paper Appendix B.3:
            "we freeze the parameters of the router-gating network during
            fine-tuning to preserve the expert assignment strategy obtained
            from the joint training stage."

        Only RouterGating parameters are frozen. ExpertCNN parameters
        (both shared and routed experts) remain trainable, allowing them
        to adapt to the target dataset while preserving the routing behavior
        learned during pre-training.

        After calling this method, the optimizer created in Finetuner
        will only update expert parameters (and other non-router params),
        since frozen parameters are excluded from optimizer parameter groups
        when filter(lambda p: p.requires_grad, model.parameters()) is used.
        """
        block: MoEBlock