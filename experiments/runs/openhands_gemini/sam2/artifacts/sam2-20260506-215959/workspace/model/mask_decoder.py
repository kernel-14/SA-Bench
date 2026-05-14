
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Type

from .layers import MLP

class TwoWayTransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(embedding_dim)
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=drop, bias=qkv_bias, batch_first=False)
        self.norm2 = norm_layer(embedding_dim)
        self.cross_attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=drop, bias=qkv_bias, batch_first=False)
        self.norm3 = norm_layer(embedding_dim)
        self.mlp = MLP(
            in_features=embedding_dim,
            hidden_features=int(embedding_dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(
        self,
        query: torch.Tensor, # (N_query, B, C)
        key_value: torch.Tensor, # (N_kv, B, C)
    ) -> torch.Tensor:
        # Self-attention on query
        q = self.norm1(query)
        attn_out, _ = self.attn(q, q, q)
        query = query + attn_out

        # Cross-attention query to key_value
        q = self.norm2(query)
        kv = self.norm2(key_value)
        cross_attn_out, _ = self.cross_attn(q, kv, kv)
        query = query + cross_attn_out

        # MLP
        query = query + self.mlp(self.norm3(query))
        return query

class MaskDecoder(nn.Module):
    def __init__(
        self,
        transformer_dim: int,
        transformer_num_heads: int,
        transformer_num_layers: int,
        iou_head_depth: int,
        iou_head_hidden_dim: int,
        num_mask_tokens: int = 4, # For multiple mask prediction
        output_upscaling_factor: int = 4, # Upsample mask output
    ):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.output_upscaling_factor = output_upscaling_factor

        self.num_mask_tokens = num_mask_tokens
        self.mask_tokens = nn.Embedding(num_mask_tokens, transformer_dim)
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.no_object_token = nn.Embedding(1, transformer_dim) # For occlusion prediction

        self.transformer = nn.ModuleList(
            [
                TwoWayTransformerBlock(
                    embedding_dim=transformer_dim,
                    num_heads=transformer_num_heads,
                )
                for _ in range(transformer_num_layers)
            ]
        )

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            nn.GroupNorm(1, transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            nn.GroupNorm(1, transformer_dim // 8),
            nn.GELU(),
        )
        self.output_hypernet_proj = nn.Conv2d(transformer_dim // 8, num_mask_tokens, kernel_size=1)

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, num_mask_tokens, iou_head_depth
        )
        self.occlusion_prediction_head = nn.Linear(transformer_dim, 1)

        self.fcn_mask_output = nn.Conv2d(transformer_dim // 8, 1, kernel_size=1) # For final mask output

    def forward(
        self,
        image_embeddings: torch.Tensor, # (B, C, H_embed, W_embed)
        sparse_prompt_embeddings: torch.Tensor, # (B, N_sparse, C)
        dense_prompt_embeddings: torch.Tensor, # (B, C, H_embed, W_embed)
        multiscale_features: List[torch.Tensor], # From image encoder (stride 4, 8)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predicts masks, IoU scores, and occlusion scores.

        Args:
            image_embeddings (torch.Tensor): The image embeddings from the image encoder,
                potentially conditioned by memory attention. (B, C, H_embed, W_embed)
            sparse_prompt_embeddings (torch.Tensor): Sparse prompt embeddings (points, boxes). (B, N_sparse, C)
            dense_prompt_embeddings (torch.Tensor): Dense prompt embeddings (masks). (B, C, H_embed, W_embed)
            multiscale_features (List[torch.Tensor]): High-resolution features from the image encoder.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - masks (torch.Tensor): Predicted masks. (B, N_masks, H_orig, W_orig)
                - iou_predictions (torch.Tensor): Predicted IoU scores. (B, N_masks)
                - occlusion_predictions (torch.Tensor): Predicted occlusion scores. (B, 1)
        """
        # Concatenate image and dense prompt embeddings
        image_embedding_flat = image_embeddings.flatten(2).permute(2, 0, 1) # (H_embed*W_embed, B, C)
        dense_prompt_flat = dense_prompt_embeddings.flatten(2).permute(2, 0, 1) # (H_embed*W_embed, B, C)
        combined_embeddings = image_embedding_flat + dense_prompt_flat

        # Prepare tokens
        batch_size = image_embeddings.shape[0]
        mask_tokens = self.mask_tokens.weight.unsqueeze(0).repeat(batch_size, 1, 1) # (B, N_mask_tokens, C)
        iou_token = self.iou_token.weight.unsqueeze(0).repeat(batch_size, 1, 1) # (B, 1, C)
        no_object_token = self.no_object_token.weight.unsqueeze(0).repeat(batch_size, 1, 1) # (B, 1, C)

        # Concatenate with sparse prompts
        # (B, N_sparse + N_mask_tokens + 1 + 1, C)
        all_tokens = torch.cat([sparse_prompt_embeddings, mask_tokens, iou_token, no_object_token], dim=1)
        
        # Permute for transformer (N_tokens, B, C)
        all_tokens = all_tokens.permute(1, 0, 2)

        # Transformer blocks
        for block in self.transformer:
            all_tokens = block(all_tokens, combined_embeddings)

        # Split tokens back
        sparse_prompts_out = all_tokens[:sparse_prompt_embeddings.shape[1]]
        mask_tokens_out = all_tokens[sparse_prompt_embeddings.shape[1] : sparse_prompt_embeddings.shape[1] + self.num_mask_tokens]
        iou_token_out = all_tokens[sparse_prompt_embeddings.shape[1] + self.num_mask_tokens : sparse_prompt_embeddings.shape[1] + self.num_mask_tokens + 1]
        no_object_token_out = all_tokens[sparse_prompt_embeddings.shape[1] + self.num_mask_tokens + 1:]

        # Reshape to (B, N_tokens, C)
        mask_tokens_out = mask_tokens_out.permute(1, 0, 2)
        iou_token_out = iou_token_out.permute(1, 0, 2)
        no_object_token_out = no_object_token_out.permute(1, 0, 2)

        # Predict masks
        upscaled_features = self.output_upscaling(image_embeddings)
        # Add skip connections (multiscale features from stride 4 and 8)
        # Assuming multiscale_features = [stride4_feat, stride8_feat]
        if len(multiscale_features) > 0:
            upscaled_features = upscaled_features + F.interpolate(
                multiscale_features[1], # Stride 8
                size=upscaled_features.shape[2:],
                mode="bilinear",
                align_corners=False
            )
        if len(multiscale_features) > 1:
            upscaled_features = upscaled_features + F.interpolate(
                multiscale_features[0], # Stride 4
                size=upscaled_features.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        hypernet_out = self.output_hypernet_proj(upscaled_features) # (B, N_mask_tokens, H_up, W_up)

        masks = torch.einsum("bqc,bchw->bqhw", mask_tokens_out, hypernet_out)
        
        # Predict IoU and occlusion scores
        iou_predictions = self.iou_prediction_head(iou_token_out) # (B, N_mask_tokens)
        occlusion_predictions = self.occlusion_prediction_head(no_object_token_out) # (B, 1)

        return masks, iou_predictions.squeeze(1), occlusion_predictions.squeeze(1)

