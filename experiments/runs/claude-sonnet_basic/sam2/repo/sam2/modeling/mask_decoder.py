"""
Mask decoder for SAM 2.

The decoder uses "two-way" transformer blocks that update both prompt and frame embeddings.
Key additions over SAM:
1. Skip connections from stride 4 and 8 features of the Hiera encoder
2. Occlusion prediction head (predicts if object is visible in current frame)
3. Object pointer token (mask token used as object pointer for memory bank)
4. Multi-mask prediction for ambiguous prompts
"""

import math
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoWayAttentionBlock(nn.Module):
    """
    A transformer block with four layers:
    1. Self-attention on sparse inputs (prompts)
    2. Cross-attention from sparse inputs to dense inputs (image features)
    3. MLP on sparse inputs
    4. Cross-attention from dense inputs to sparse inputs
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ):
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        query_pe: torch.Tensor,
        key_pe: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class TwoWayTransformer(nn.Module):
    """
    Two-way transformer for the mask decoder.
    Processes both sparse (prompt) and dense (image) embeddings.
    """

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ):
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: torch.Tensor,
        image_pe: torch.Tensor,
        point_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_embedding: [B, C, H, W] image features
            image_pe: [B, C, H, W] positional encoding for image
            point_embedding: [B, N, C] sparse prompt embeddings

        Returns:
            queries: [B, N, C] updated prompt embeddings
            keys: [B, H*W, C] updated image embeddings
        """
        # Flatten spatial dimensions
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)  # B HW C
        image_pe = image_pe.flatten(2).permute(0, 2, 1)  # B HW C

        # Prepare queries and keys
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply final attention layer from the tokens to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class Attention(nn.Module):
    """Multi-head attention with optional downsampling of internal dimensions."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class MLPBlock(nn.Module):
    """MLP block for the transformer."""

    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class MaskDecoder(nn.Module):
    """
    SAM 2 mask decoder.

    Predicts segmentation masks from image embeddings and prompt embeddings.
    Key features:
    - Two-way transformer for joint processing of prompts and image features
    - Multi-mask prediction for ambiguous prompts
    - Occlusion prediction head
    - Object pointer token for memory bank
    - Skip connections from stride 4 and 8 image encoder features
    """

    def __init__(
        self,
        transformer_dim: int = 256,
        transformer: Optional[nn.Module] = None,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = True,
        skip_dims: Tuple[int, int] = (64, 128),  # stride 4 and 8 feature dims
    ):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_multimask_outputs = num_multimask_outputs
        self.use_high_res_features = use_high_res_features

        if transformer is None:
            self.transformer = TwoWayTransformer(
                depth=2,
                embedding_dim=transformer_dim,
                mlp_dim=2048,
                num_heads=8,
            )
        else:
            self.transformer = transformer

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1  # +1 for single mask
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # Occlusion prediction token (new in SAM 2)
        self.occlusion_token = nn.Embedding(1, transformer_dim)

        # Upscaling network for mask prediction
        if use_high_res_features:
            # With skip connections from stride 4 and 8 features
            self.output_upscaling = nn.Sequential(
                nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                LayerNorm2d(transformer_dim // 4),
                activation(),
                nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                activation(),
            )
            # Fuse with stride 8 skip features
            self.skip_fuse_8 = nn.Conv2d(
                transformer_dim // 4 + skip_dims[1], transformer_dim // 4, kernel_size=1
            )
            # Fuse with stride 4 skip features
            self.skip_fuse_4 = nn.Conv2d(
                transformer_dim // 8 + skip_dims[0], transformer_dim // 8, kernel_size=1
            )
        else:
            self.output_upscaling = nn.Sequential(
                nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                LayerNorm2d(transformer_dim // 4),
                activation(),
                nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                activation(),
            )

        # MLP heads for each mask output
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            for _ in range(self.num_mask_tokens)
        ])

        # IoU prediction head
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        # Occlusion prediction head (new in SAM 2)
        self.occlusion_head = MLP(transformer_dim, transformer_dim // 4, 1, 3)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Args:
            image_embeddings: [B, C, H, W] image features (conditioned by memory attention)
            image_pe: [B, C, H, W] positional encoding for image
            sparse_prompt_embeddings: [B, N, C] sparse prompt embeddings
            dense_prompt_embeddings: [B, C, H, W] dense prompt embeddings
            multimask_output: if True, return multiple masks for ambiguous prompts
            high_res_features: optional list of [stride4, stride8] features for skip connections

        Returns:
            masks: [B, num_masks, H, W] predicted masks
            iou_pred: [B, num_masks] predicted IoU scores
            mask_tokens_out: [B, num_masks, C] mask token outputs (used as object pointers)
            occlusion_pred: [B, 1] occlusion prediction score
        """
        masks, iou_pred, mask_tokens_out, occlusion_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            high_res_features=high_res_features,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)

        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        mask_tokens_out = mask_tokens_out[:, mask_slice, :]

        return masks, iou_pred, mask_tokens_out, occlusion_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict masks for all mask tokens."""
        # Concatenate output tokens
        output_tokens = torch.cat(
            [
                self.iou_token.weight,
                self.mask_tokens.weight,
                self.occlusion_token.weight,
            ],
            dim=0,
        )
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0] // image_embeddings.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0] // image_pe.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]
        occlusion_token_out = hs[:, 1 + self.num_mask_tokens, :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        if self.use_high_res_features and high_res_features is not None:
            skip4, skip8 = high_res_features[0], high_res_features[1]
            # First upscale: stride 16 -> stride 8
            upscaled = self.output_upscaling[0](src)  # ConvTranspose2d
            upscaled = self.output_upscaling[1](upscaled)  # LayerNorm2d
            upscaled = self.output_upscaling[2](upscaled)  # activation
            # Fuse with stride 8 skip features
            upscaled = torch.cat([upscaled, skip8], dim=1)
            upscaled = self.skip_fuse_8(upscaled)
            # Second upscale: stride 8 -> stride 4
            upscaled = self.output_upscaling[3](upscaled)  # ConvTranspose2d
            upscaled = self.output_upscaling[4](upscaled)  # activation
            # Fuse with stride 4 skip features
            upscaled = torch.cat([upscaled, skip4], dim=1)
            upscaled = self.skip_fuse_4(upscaled)
        else:
            upscaled = self.output_upscaling(src)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # B x num_mask_tokens x C

        b, c, h, w = upscaled.shape
        masks = (hyper_in @ upscaled.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = torch.sigmoid(self.iou_prediction_head(iou_token_out))

        # Generate occlusion prediction
        occlusion_pred = self.occlusion_head(occlusion_token_out)

        return masks, iou_pred, mask_tokens_out, occlusion_pred


class MLP(nn.Module):
    """Simple MLP with configurable depth."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


class LayerNorm2d(nn.Module):
    """Layer normalization for 2D feature maps."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
