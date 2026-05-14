## Code: models/mask_decoder.py

```python
## models/mask_decoder.py
"""Mask decoder for SAM 2: extends SAM's two-way transformer with occlusion head,
skip connections from the hierarchical image encoder, sigmoid-activated IoU
predictions, and the mask output token serving as the object pointer.

Architecture (Section 4, Appendix D.1, Figure 8):
    - TwoWayTransformer: stacked two-way attention blocks updating both prompt
      tokens and frame embeddings (identical to SAM's design)
    - Additional occlusion_token + occlusion_prediction_head (SAM 2 addition)
    - Skip connections from stride-4 and stride-8 Hiera features into upsampling
    - Sigmoid-activated IoU predictions (SAM 2 modification from raw logits)
    - Mask output token (index 0) doubles as the object pointer for memory bank

Config references:
    model.num_multimask_outputs: 3
    model.mask_threshold: 0.0
    model.fpn_out_channels: 256  → transformer_dim

Paper references:
    Section 4: "Our decoder design largely follows SAM. We stack 'two-way'
        transformer blocks that update prompt and frame embeddings."
    Section 4: "we add an additional head that predicts whether the object of
        interest is present on the current frame."
    Appendix D.1: "we also introduce an occlusion prediction head."
    Appendix D.1: "we use the mask token corresponding to the output mask as
        the object pointer token for the frame."
    Appendix D.1: "we also include the stride 4 and 8 features from the image
        encoder during upsampling."
    Appendix D.2.1: "apply a sigmoid activation to the IoU logits to restrict
        the output into the range between 0 and 1."
"""

import logging
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LayerNorm2d helper (channels-first format)
# ---------------------------------------------------------------------------


class LayerNorm2d(nn.Module):
    """Layer normalization for 2D feature maps in [B, C, H, W] format.

    Permutes to channels-last, applies LayerNorm over C, then permutes back.
    Used in the upsampling pathway of MaskDecoder.

    Args:
        num_channels: Number of channels C to normalize over.
        eps: Epsilon for numerical stability. Defaults to 1e-6.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm: nn.LayerNorm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization over the channel dimension.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Normalized tensor of shape [B, C, H, W].
        """
        x = x.permute(0, 2, 3, 1)   # [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)   # [B, C, H, W]
        return x


# ---------------------------------------------------------------------------
# MLP helper
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Simple multi-layer perceptron used for IoU, occlusion, and hypernetwork heads.

    Architecture: Linear → ReLU (repeated num_layers-1 times) → Linear.
    Optional sigmoid on the final output for probability outputs.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Output feature dimension.
        num_layers: Total number of linear layers (including input and output).
            Must be >= 1. With num_layers=1, only a single linear layer is used.
        sigmoid_output: If True, apply sigmoid to the final output.
            Defaults to False; sigmoid is applied externally for IoU/occlusion
            to keep this class generic.

    Example:
        mlp = MLP(256, 256, 4, num_layers=3)
        x = torch.randn(2, 256)
        out = mlp(x)  # [2, 4]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()

        self.num_layers: int = num_layers
        self.sigmoid_output: bool = sigmoid_output

        # Build layer dimensions
        if num_layers == 1:
            h_dims: List[int] = []
        else:
            h_dims = [hidden_dim] * (num_layers - 1)

        # Construct layers: input → hidden... → output
        layer_dims: List[int] = [input_dim] + h_dims + [output_dim]
        layers: List[nn.Module] = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                layers.append(nn.ReLU(inplace=True))

        self.layers: nn.Sequential = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize linear layers with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP.

        Args:
            x: Input tensor of shape [..., input_dim].

        Returns:
            Output tensor of shape [..., output_dim].
        """
        x = self.layers(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


# ---------------------------------------------------------------------------
# Attention helper
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Multi-head attention with optional downscaling of internal dimension.

    Used within TwoWayAttentionBlock for both self-attention and cross-attention.
    Supports an internal dimension downscaling factor to reduce compute in
    cross-attention layers (SAM convention).

    Args:
        embedding_dim: Input and output embedding dimension.
        num_heads: Number of attention heads.
        downsample_rate: Factor by which to reduce the internal QKV dimension.
            Defaults to 1 (no downscaling). SAM uses 2 for cross-attention.

    Example:
        attn = Attention(embedding_dim=256, num_heads=8, downsample_rate=2)
        q = torch.randn(2, 10, 256)
        k = torch.randn(2, 4096, 256)
        v = torch.randn(2, 4096, 256)
        out = attn(q, k, v)  # [2, 10, 256]
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()

        self.embedding_dim: int = embedding_dim
        self.num_heads: int = num_heads
        self.internal_dim: int = embedding_dim // downsample_rate

        if self.internal_dim % num_heads != 0:
            raise ValueError(
                f"internal_dim ({self.internal_dim}) must be divisible by "
                f"num_heads ({num_heads}). Got embedding_dim={embedding_dim}, "
                f"downsample_rate={downsample_rate}."
            )

        self.head_dim: int = self.internal_dim // num_heads
        self.scale: float = self.head_dim ** -0.5

        self.q_proj: nn.Linear = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj: nn.Linear = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj: nn.Linear = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj: nn.Linear = nn.Linear(self.internal_dim, embedding_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize projection layers with Xavier uniform."""
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def _separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape [B, L, internal_dim] → [B, num_heads, L, head_dim]."""
        B, L, _ = x.shape
        x = x.reshape(B, L, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # [B, num_heads, L, head_dim]

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape [B, num_heads, L, head_dim] → [B, L, internal_dim]."""
        B, _, L, _ = x.shape
        x = x.permute(0, 2, 1, 3)  # [B, L, num_heads, head_dim]
        return x.reshape(B, L, self.internal_dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-head attention.

        Args:
            q: Query tensor of shape [B, L_q, embedding_dim].
            k: Key tensor of shape [B, L_k, embedding_dim].
            v: Value tensor of shape [B, L_v, embedding_dim]. L_v must equal L_k.

        Returns:
            Attention output of shape [B, L_q, embedding_dim].
        """
        # Project to internal dimension
        q_proj = self.q_proj(q)   # [B, L_q, internal_dim]
        k_proj = self.k_proj(k)   # [B, L_k, internal_dim]
        v_proj = self.v_proj(v)   # [B, L_k, internal_dim]

        # Separate into heads
        q_heads = self._separate_heads(q_proj)   # [B, nhead, L_q, head_dim]
        k_heads = self._separate_heads(k_proj)   # [B, nhead, L_k, head_dim]
        v_heads = self._separate_heads(v_proj)   # [B, nhead, L_k, head_dim]

        # Scaled dot-product attention
        # Use PyTorch's built-in for potential FlashAttention dispatch
        attn_out = F.scaled_dot_product_attention(
            q_heads, k_heads, v_heads,
            dropout_p=0.0,
            is_causal=False,
        )  # [B, nhead, L_q, head_dim]

        # Recombine heads and project to output
        attn_out = self._recombine_heads(attn_out)  # [B, L_q, internal_dim]
        return self.out_proj(attn_out)               # [B, L_q, embedding_dim]


# ---------------------------------------------------------------------------
# TwoWayAttentionBlock
# ---------------------------------------------------------------------------


class TwoWayAttentionBlock(nn.Module):
    """Single block of the two-way transformer used in SAM 2's mask decoder.

    Performs bidirectional attention between sparse prompt tokens and the
    image embedding, following SAM's original design (Kirillov et al., 2023).

    Operations per block:
        1. Self-attention on sparse tokens (tokens attend to each other)
        2. Cross-attention: tokens (Q) → image embedding (K, V)
        3. MLP on tokens
        4. Cross-attention: image embedding (Q) → tokens (K, V)

    All sub-operations use pre-norm (LayerNorm before the operation) with
    residual connections.

    Args:
        embedding_dim: Embedding dimension throughout. Defaults to 256.
        num_heads: Number of attention heads. Defaults to 8.
        mlp_dim: Hidden dimension of the token MLP. Defaults to 2048.
        activation: Activation class for the MLP. Defaults to nn.ReLU.
        attention_downsample_rate: Downsampling factor for cross-attention
            internal dimension. Defaults to 2 (SAM convention).
        skip_first_layer_pe: If True, skip adding positional encoding to
            queries in the first self-attention layer. Used for the first
            block when tokens already have PE added externally.
            Defaults to False.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()

        self.skip_first_layer_pe: bool = skip_first_layer_pe

        # 1. Self-attention on tokens
        self.self_attn: Attention = Attention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
        )
        self.norm1: nn.LayerNorm = nn.LayerNorm(embedding_dim)

        # 2. Cross-attention: tokens → image
        self.cross_attn_token_to_image: Attention = Attention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            downsample_rate=attention_downsample_rate,
        )
        self.norm2: nn.LayerNorm = nn.LayerNorm(embedding_dim)

        # 3. MLP on tokens
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            activation(),
            nn.Linear(mlp_dim, embedding_dim),
        )
        self.norm3: nn.LayerNorm = nn.LayerNorm(embedding_dim)

        # 4. Cross-attention: image → tokens
        self.cross_attn_image_to_token: Attention = Attention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            downsample_rate=attention_downsample_rate,
        )
        self.norm4: nn.LayerNorm = nn.LayerNorm(embedding_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize MLP layers with Xavier uniform."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        query_pe: torch.Tensor,
        key_pe: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply two-way attention block.

        Args:
            queries: Sparse prompt token embeddings of shape [B, N_tokens, C].
            keys: Image embedding tokens of shape [B, H*W, C].
            query_pe: Positional encoding for queries, same shape as queries.
            key_pe: Positional encoding for keys (image PE), same shape as keys.

        Returns:
            Tuple of:
                - Updated queries: [B, N_tokens, C]
                - Updated keys: [B, H*W, C]
        """
        # ------------------------------------------------------------------
        # 1. Self-attention on sparse tokens
        # ------------------------------------------------------------------
        if self.skip_first_layer_pe:
            queries = queries + self.self_attn(queries, queries, queries)
        else:
            q_with_pe = queries + query_pe
            attn_out = self.self_attn(q_with_pe, q_with_pe, queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # ------------------------------------------------------------------
        # 2. Cross-attention: tokens (Q) → image embedding (K, V)
        # ------------------------------------------------------------------
        q_with_pe = queries + query_pe
        k_with_pe = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q_with_pe, k_with_pe, keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # ------------------------------------------------------------------
        # 3. MLP on tokens
        # ------------------------------------------------------------------
        queries = queries + self.mlp(queries)
        queries = self.norm3(queries)

        # ------------------------------------------------------------------
        # 4. Cross-attention: image embedding (Q) → tokens (K, V)
        # ------------------------------------------------------------------
        q_with_pe = queries + query_pe
        k_with_pe = keys + key_pe
        attn_out = self.cross_attn_image_to_token(k_with_pe, q_with_pe, queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


# ---------------------------------------------------------------------------
# TwoWayTransformer
# ---------------------------------------------------------------------------


class TwoWayTransformer(nn.Module):
    """Two-way transformer for SAM 2's mask decoder.

    Stacks `depth` TwoWayAttentionBlock layers followed by a final
    cross-attention layer (tokens → image) and a final LayerNorm on tokens.

    This is SAM's original design carried over unchanged into SAM 2.
    From Section 4: "Our decoder design largely follows SAM. We stack
    'two-way' transformer blocks that update prompt and frame embeddings."

    Args:
        depth: Number of TwoWayAttentionBlock layers. Defaults to 2
            (SAM's original design; not re-specified for SAM 2).
        embedding_dim: Embedding dimension. Defaults to 256.
        num_heads: Number of attention heads. Defaults to 8.
        mlp_dim: Hidden dimension of the token MLP. Defaults to 2048.
        activation: Activation class for MLPs. Defaults to nn.ReLU.
        attention_downsample_rate: Downsampling factor for cross-attention.
            Defaults to 2.

    Example:
        transformer = TwoWayTransformer(depth=2, embedding_dim=256, num_heads=8)
        image_embed = torch.randn(2, 256, 64, 64)
        image_pe = torch.randn(2, 256, 64, 64)
        point_embed = torch.randn(2, 5, 256)
        tokens_out, image_out = transformer(image_embed, image_pe, point_embed)
        # tokens_out: [2, 5, 256], image_out: [2, 4096, 256]
    """

    def __init__(
        self,
        depth: int = 2,
        embedding_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()

        self.depth: int = depth
        self.embedding_dim: int = embedding_dim
        self.num_heads: int = num_heads
        self.mlp_dim: int = mlp_dim

        # Stack of two-way attention blocks
        self.layers: nn.ModuleList = nn.ModuleList([
            TwoWayAttentionBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                activation=activation,
                attention_downsample_rate=attention_downsample_rate,
                skip_first_layer_pe=(i == 0),  # skip PE in first block's self-attn
            )
            for i in range(depth)
        ])

        # Final cross-attention: tokens attend to image one last time
        self.final_attn_token_to_image: Attention = Attention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            downsample_rate=attention_downsample_rate,
        )
        self.norm_final_attn: nn.LayerNorm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: torch.Tensor,
        image_pe: torch.Tensor,
        point_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the two-way transformer.

        Args:
            image_embedding: Frame embedding of shape [B, C, H, W].
                Will be flattened to [B, H*W, C] internally.
            image_pe: Positional encoding for the image, same shape [B, C, H, W].
            point_embedding: Sparse prompt token embeddings [B, N_tokens, C].

        Returns:
            Tuple of:
                - Updated token embeddings: [B, N_tokens, C]
                - Updated image embedding (flattened): [B, H*W, C]
        """
        B, C, H, W = image_embedding.shape

        # Flatten spatial dimensions: [B, C, H, W] → [B, H*W, C]
        image_flat: torch.Tensor = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe_flat: torch.Tensor = image_pe.flatten(2).permute(0, 2, 1)

        # Initialize queries (tokens) and keys (image)
        queries: torch.Tensor = point_embedding   # [B, N_tokens, C]
        keys: torch.Tensor = image_flat           # [B, H*W, C]

        # Run stacked two-way attention blocks
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=torch.zeros_like(queries),  # PE added inside blocks
                key_pe=image_pe_flat,
            )

        # Final cross-attention: tokens → image (one last update of tokens)
        q_with_pe = queries + torch.zeros_like(queries)  # no additional PE for tokens
        k_with_pe = keys + image_pe_flat
        attn_out = self.final_attn_token_to_image(q_with_pe, k_with_pe, keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


# ---------------------------------------------------------------------------
# MaskDecoder
# ---------------------------------------------------------------------------


class MaskDecoder(nn.Module):
    """Mask decoder for SAM 2 with occlusion head and skip connections.

    Extends SAM's mask decoder with four SAM 2-specific additions:
    1. Occlusion token + occlusion prediction head (new output mode for PVS task)
    2. Skip connections from stride-4 and stride-8 Hiera features into upsampling
    3. Sigmoid-activated IoU predictions (restricted to [0, 1])
    4. Mask output token (index 0) returned as object pointer for memory bank

    From Appendix D.1 (Figure 8): "The design largely follows SAM, and we
    additionally include the stride 4 and 8 features from the image encoder
    during upsampling. We also use the mask token corresponding to the output
    mask as an object pointer and generate an occlusion score which indicates
    if the object of interest is visible in the current frame."

    Token layout fed to TwoWayTransformer:
        [iou_token | mask_tokens (num_multimask_outputs+1) | occlusion_token | sparse_prompts]
        Indices:
            0: iou_token
            1..num_multimask_outputs+1: mask_tokens
            num_multimask_outputs+2: occlusion_token
            num_multimask_outputs+3..: sparse prompt tokens

    Object pointer:
        mask_tokens_out[:, 0, :] — the single-mask output token (index 0 of
        mask_tokens). This is the "mask token corresponding to the output mask"
        referenced in Appendix D.1.

    Args:
        transformer_dim: Embedding dimension throughout. Defaults to 256
            (matches config.model.fpn_out_channels).
        transformer: TwoWayTransformer instance for bidirectional attention.
        num_multimask_outputs: Number of masks for ambiguous prompts.
            Defaults to 3 (config.model.num_multimask_outputs).
        skip_s4_channels: Channel count of Stage 1 (stride-4) Hiera features.
            Depends on encoder size. Defaults to 112 (Hiera-B+ Stage 1).
        skip_s8_channels: Channel count of Stage 2 (stride-8) Hiera features.
            Depends on encoder size. Defaults to 224 (Hiera-B+ Stage 2).
        iou_head_depth: Number of layers in the IoU prediction MLP.
            Defaults to 3.
        iou_head_hidden_dim: Hidden dimension of the IoU prediction MLP.
            Defaults to 256.

    Example:
        transformer = TwoWayTransformer(depth=2, embedding_dim=256, num_heads=8)
        decoder = MaskDecoder(
            transformer_dim=256,
            transformer=transformer,
            num_multimask_outputs=3,
            skip_s4_channels=112,
            skip_s8_channels=224,
        )
        image_embed = torch.randn(2, 256, 64, 64)
        image_pe = torch.randn(1, 256, 64, 64)
        sparse = torch.randn(2, 3, 256)
        dense = torch.randn(2, 256, 64, 64)
        skip_s4 = torch.randn(2, 112, 256, 256)
        skip_s8 = torch.randn(2, 224, 128, 128)
        masks, iou, occ, ptr = decoder(
            image_embed, image_pe, sparse, dense, [skip_s4, skip_s8], True
        )
    """

    def __init__(
        self,
        transformer_dim: int = 256,
        transformer: Optional[TwoWayTransformer] = None,
        num_multimask_outputs: int = 3,
        skip_s4_channels: int = 112,
        skip_s8_channels: int = 224,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.transformer_dim: int = transformer_dim
        self.num_multimask_outputs: int = num_multimask_outputs

        # ------------------------------------------------------------------
        # Two-way transformer (SAM's original design)
        # ------------------------------------------------------------------
        if transformer is None:
            self.transformer: TwoWayTransformer = TwoWayTransformer(
                depth=2,
                embedding_dim=transformer_dim,
                num_heads=8,
                mlp_dim=2048,
            )
        else:
            self.transformer = transformer

        # ------------------------------------------------------------------
        # Output token embeddings
        # ------------------------------------------------------------------
        # IoU prediction token (1 token)
        self.iou_token: nn.Embedding = nn.Embedding(1, transformer_dim)

        # Mask output tokens: num_multimask_outputs + 1
        # Index 0: single-mask output (used when multimask_output=False)
        # Indices 1..num_multimask_outputs: multi-mask outputs
        self.mask_tokens: nn.Embedding = nn.Embedding(
            num_multimask_outputs + 1, transformer_dim
        )

        # Occlusion prediction token (SAM 2 addition)
        # Predicts whether the object is visible in the current frame
        self.occlusion_token: nn.Embedding = nn.Embedding(1, transformer_dim)

        # ------------------------------------------------------------------
        # Upsampling pathway with skip connections (SAM 2 addition)
        # ------------------------------------------------------------------
        # Stage 1: stride-16 → stride-8 (2× upsample)
        # Input: [B, transformer_dim, H/16, W/16]
        # Output: [B, transformer_dim//4, H/8, W/8]
        self.upscale_conv1: nn.ConvTranspose2d = nn.ConvTranspose2d(
            transformer_dim,
            transformer_dim // 4,
            kernel_size=2,
            stride=2,
        )

        # Skip connection lateral conv for stride-8 features
        # Projects Stage 2 Hiera features to match upscale_conv1 output channels
        self.skip_conv_s8: nn.Conv2d = nn.Conv2d(
            skip_s8_channels,
            transformer_dim // 4,
            kernel_size=1,
            bias=False,
        )

        # Normalization and activation after first upsample + skip merge
        self.norm1: LayerNorm2d = LayerNorm2d(transformer_dim // 4)

        # Stage 2: stride-8 → stride-4 (2× upsample)
        # Input: [B, transformer_dim//4, H/8, W/8]
        # Output: [B, transformer_dim//8, H/4, W/4]
        self.upscale_conv2: nn.ConvTranspose2d = nn.ConvTranspose2d(
            transformer_dim // 4,
            transformer_dim // 8,
            kernel_size=2,
            stride=2,
        )

        # Skip connection lateral conv for stride-4 features
        # Projects Stage 1 Hiera features to match upscale_conv2 output channels
        self.skip_conv_s4: nn.Conv2d = nn.Conv2d(
            skip_s4_channels,
            transformer_dim // 8,
            kernel_size=1,
            bias=False,
        )

        # Normalization and activation after second upsample + skip merge
        self.norm2: LayerNorm2d = LayerNorm2d(transformer_dim // 8)

        # ------------------------------------------------------------------
        # Output hypernetwork MLPs (one per mask token)
        # Maps each mask token embedding to a weight vector that is
        # dot-producted with upscaled features to produce mask logits
        # ------------------------------------------------------------------
        # Total mask tokens: num_multimask_outputs + 1
        self.output_hypernetworks_mlps: nn.ModuleList = nn.ModuleList([
            MLP(
                input_dim=transformer_dim,
                hidden_dim=transformer_dim,
                output_dim=transformer_dim // 8,
                num_layers=3,
            )
            for _ in range(num_multimask_outputs + 1)
        ])

        # ------------------------------------------------------------------
        # IoU prediction head (SAM 2 modification: sigmoid applied externally)
        # Input: iou_token output [B, transformer_dim]
        # Output: [B, num_multimask_outputs + 1] — one score per mask
        # ------------------------------------------------------------------
        self.iou_prediction_head: MLP = MLP(
            input_dim=transformer_dim,