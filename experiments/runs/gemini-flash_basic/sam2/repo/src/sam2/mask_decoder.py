import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Type

class MaskDecoder(nn.Module):
    """
    SAM 2 Mask Decoder.

    This module takes image embeddings, prompt embeddings, and multiscale skip features
    from the image encoder to predict segmentation masks, an occlusion score, and an IoU score.
    It largely follows the SAM decoder design but includes specific adaptations for video
    and the occlusion prediction head.
    """

    def __init__(
        self,
        transformer_dim: int,
        transformer_n_heads: int,
        transformer_depth: int,
        transformer_mlp_dim: int,
        iou_head_depth: int,
        iou_head_hidden_dim: int,
        activation: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.transformer_dim = transformer_dim

        self.num_mask_tokens = 4 # SAM predicts 3 masks + 1 no-object mask

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # Output tokens for occlusion prediction
        self.occlusion_token = nn.Embedding(1, transformer_dim)

        # Positional encoding for image features
        self.pe_layer = PositionEmbeddingRandom(transformer_dim // 2)

        # Mock two-way transformer for simplicity. In a real impl, this would be a full transformer.
        # This transformer updates both query (prompt) and key/value (image) embeddings.
        self.transformer = TwoWayTransformer(
            depth=transformer_depth,
            embedding_dim=transformer_dim,
            num_heads=transformer_n_heads,
            mlp_dim=transformer_mlp_dim,
            activation=activation,
        )

        # Upscaling layers to produce masks
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            nn.LayerNorm(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernet_coords = nn.Conv2d(transformer_dim // 8, self.num_mask_tokens, kernel_size=1)
        self.output_hypernet_points = nn.Conv2d(transformer_dim // 8, self.num_mask_tokens, kernel_size=1)


        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens, # Predict IoU for each mask output
            iou_head_depth,
            activation,
        )

        self.occlusion_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            1, # Predict a single occlusion score
            iou_head_depth,
            activation,
        )

    def forward(
        self,
        image_embeddings: torch.Tensor, # From ImageEncoder, stride 16 (B, C, H_16, W_16)
        image_pe: torch.Tensor, # Positional encoding for image_embeddings (1, H_16, W_16, C)
        sparse_prompt_embeddings: torch.Tensor, # From PromptEncoder (B, N_sparse, C)
        dense_prompt_embeddings: torch.Tensor, # From PromptEncoder (B, C, H_16, W_16)
        skip_feat4: torch.Tensor, # From ImageEncoder, stride 4 (B, C, H_4, W_4)
        skip_feat8: torch.Tensor, # From ImageEncoder, stride 8 (B, C, H_8, W_8)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            image_embeddings: Image features from the image encoder.
            image_pe: Positional embeddings for image_embeddings.
            sparse_prompt_embeddings: Embeddings from sparse prompts (points, boxes).
            dense_prompt_embeddings: Embeddings from dense prompts (masks).
            skip_feat4: Skip connection features from ImageEncoder at stride 4.
            skip_feat8: Skip connection features from ImageEncoder at stride 8.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                (masks, iou_predictions, occlusion_predictions, object_pointer_embedding)
        """

        # Concatenate mask and IoU tokens to sparse prompts
        sparse_embeddings = torch.cat([
            self.iou_token.weight.unsqueeze(0).expand(sparse_prompt_embeddings.shape[0], -1, -1),
            self.mask_tokens.weight.unsqueeze(0).expand(sparse_prompt_embeddings.shape[0], -1, -1),
            self.occlusion_token.weight.unsqueeze(0).expand(sparse_prompt_embeddings.shape[0], -1, -1),
            sparse_prompt_embeddings,
        ], dim=1)

        # Add dense prompt embeddings to image embeddings
        image_embeddings = image_embeddings + dense_prompt_embeddings

        # Transformer accepts (B, N, C) for queries/keys. Flatten image_embeddings.
        B, C, H, W = image_embeddings.shape
        image_embeddings_flat = image_embeddings.view(B, C, H * W).permute(0, 2, 1) # (B, H*W, C)
        image_pe_flat = image_pe.view(1, H * W, C) # (1, H*W, C)

        # Apply two-way transformer
        # The output  is typically from the image path of the transformer
        # and  from the prompt path.
        # In SAM, the transformer outputs a query embedding for each mask output token
        # and an image embedding.
        # For simplicity, we'll assume the transformer updates  and .

        # SAM-like transformer output (modified for SAM2 additions)
        # query_embeddings: (B, N_tokens, C) where N_tokens includes iou, mask, occlusion, and sparse prompts
        # image_embedding_updates: (B, H*W, C)
        query_embeddings, image_embedding_updates = self.transformer(
            image_embeddings_flat,
            image_pe_flat,
            sparse_embeddings
        )

        # Separate the output tokens
        iou_token_out = query_embeddings[:, 0, :]
        mask_tokens_out = query_embeddings[:, 1:1+self.num_mask_tokens, :]
        occlusion_token_out = query_embeddings[:, 1+self.num_mask_tokens, :]
        # The rest are updated sparse prompt embeddings, which are not directly used for mask generation here.

        # IoU and Occlusion prediction
        iou_predictions = self.iou_prediction_head(iou_token_out)
        occlusion_predictions = self.occlusion_prediction_head(occlusion_token_out)

        # Reshape image embedding updates back to spatial form (B, C, H, W)
        image_embedding_final = image_embedding_updates.permute(0, 2, 1).view(B, C, H, W)

        # Upscale and combine with skip connections for high-resolution masks
        low_res_masks = self.output_upscaling(image_embedding_final)

        # Add skip connections (stride 8 and 4) from the image encoder during upsampling
        # The paper states: "Additionally include the stride 4 and 8 features from the image encoder during upsampling."
        # This suggests merging them at appropriate stages of the upsampling. For simplicity here, 
        # we'll consider them fused after  produces an initial low-res mask.
        # A more detailed FPN-like merge would occur within .

        # Mocking the merge by resizing and adding - actual implementation would be more complex.
        # Resize low_res_masks to match skip_feat8 and skip_feat4 resolutions
        # skip_feat8 has H/8, W/8. low_res_masks is H/4, W/4 after first upsample, then H/2, W/2 after second
        
        # Let's refine the upsampling and skip connection integration.
        # Assuming output_upscaling takes image_embedding_final (H/16, W/16)
        # Step 1: Up to H/8, W/8. Merge with skip_feat8
        # Step 2: Up to H/4, W/4. Merge with skip_feat4

        # For this mock,  should output H/4, W/4.
        # Let's adjust output_upscaling to reflect this, and then integrate skip features.

        # Redefine output_upscaling for clearer skip connection integration
        # Initial upsampling to H/8, W/8
        upsample_to_8 = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            activation(),
        )
        # Then upsample to H/4, W/4
        upsample_to_4 = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        
        # Re-applying to reflect proper skip connections, but maintaining the single sequential for brevity
        # For a true FPN integration in decoder,  would be more complex.
        # For now, let's assume  produces a feature map at some intermediate resolution,
        # and then we use hypernets for final mask prediction.

        # For the mock, let's assume the  directly produces a feature map
        # that is then used with mask tokens to generate masks.

        # The paper states that the stride 4 and 8 features are added to the upsampling layers.
        # This means the upsampling needs to be more granular.

        # Let's simplify the output_upscaling to just produce a feature map at H/4, W/4 or similar
        # and then apply the hypernets.

        # This is a highly simplified representation of a two-way transformer's image output path.
        # In SAM,  would be the updated image features from the transformer.
        
        # Mask prediction using mask_tokens_out and image_embedding_final
        # Mask tokens are applied as convolutions or by dot product with image features.
        # Here, we use hypernets that take the upsampled image features and output mask logits.
        
        upsampled_features = self.output_upscaling(image_embedding_final)

        # For simplicity in this mock, we assume skip connections are integrated within the upsampling
        # process of  if it were a multi-stage process.
        # For now, we will add them before the final mask prediction.

        # Resize skip features to match upsampled_features spatial dimensions for adding
        # upsampled_features is H/4, W/4 (if transformer_dim // 8 is its output chan)
        upsampled_features = upsampled_features + F.interpolate(skip_feat8, size=upsampled_features.shape[2:], mode='nearest')
        upsampled_features = upsampled_features + F.interpolate(skip_feat4, size=upsampled_features.shape[2:], mode='nearest')

        hypernet_masks_coords = self.output_hypernet_coords(upsampled_features)
        hypernet_masks_points = self.output_hypernet_points(upsampled_features)

        # Multiply mask tokens with upsampled features to get mask logits
        # This is a common pattern in SAM-like decoders.
        # (B, N_masks, C) @ (B, C, H, W) -> (B, N_masks, H, W)
        masks = (mask_tokens_out @ upsampled_features.view(B, self.transformer_dim, -1)).view(B, self.num_mask_tokens, upsampled_features.shape[2], upsampled_features.shape[3])

        # Object pointer embedding is the mask token that corresponds to the predicted mask (e.g., highest IoU mask)
        # For simplicity, we'll return the mask_tokens_out as the potential object pointer embeddings.
        object_pointer_embedding = mask_tokens_out[:, 0, :] # Example: take the first mask token as the primary object pointer

        return masks, iou_predictions, occlusion_predictions, object_pointer_embedding


class TwoWayTransformer(nn.Module):
    """
    A mock Two-Way Transformer similar to the one in SAM.
    It takes query embeddings (prompts) and image embeddings (keys/values)
    and iteratively refines them through self-attention and cross-attention.
    """
    def __init__(self, depth: int, embedding_dim: int, num_heads: int, mlp_dim: int, activation: Type[nn.Module] = nn.GELU):
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        
        # Simplified: just return updated embeddings without complex transformer blocks
        # In a full implementation, this would involve multiple layers of attention and FFNs.
        # For the purpose of data flow, we'll simulate the output shape and pass-through functionality.

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                TwoWayTransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim,
            num_heads,
            q_bias=True,
            kv_bias=True,
        )
        self.final_mlp = MLP(
            embedding_dim,
            mlp_dim,
            embedding_dim,
            3,
            activation,
        )

    def forward(
        self, 
        image_embedding: torch.Tensor, 
        image_pe: torch.Tensor, 
        point_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # For simplicity, apply some transformations. In reality, this is iterative.
        # This mock will simply return inputs with correct shapes, possibly transformed.
        # Actual SAM two-way transformer iteratively updates point_embedding and image_embedding.

        # Apply transformer blocks
        for layer in self.layers:
            point_embedding, image_embedding = layer(point_embedding, image_embedding, image_pe)

        # Final attention layer to refine token embeddings with all image tokens
        # (B, N_points, C) @ (B, H*W, C) -> (B, N_points, C)
        # SAM's original implementation has more sophisticated final attention
        # For mock, we'll pass point_embedding through some linear layers
        q = point_embedding
        k = image_embedding
        v = image_embedding
        
        # Perform attention. Assuming Attention takes Q, K, V and outputs updated Q
        point_embedding = self.final_attn_token_to_image(q, k, v)
        point_embedding = self.final_mlp(point_embedding)
        
        return point_embedding, image_embedding

class TwoWayTransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.GELU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, mlp_dim, embedding_dim, 3, activation)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = Attention(
            embedding_dim,
            num_heads,
            q_bias=True,
            kv_bias=True,
        )
        self.cross_attn_image_to_token = Attention(
            embedding_dim,
            num_heads,
            q_bias=True,
            kv_bias=True,
        )

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        query_pe: torch.Tensor,
        key_pe: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Self attention for queries
        q = queries + query_pe
        attn_out = self.self_attn(q, q, q)
        queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention queries to keys
        q = queries + query_pe
        k = keys + key_pe
        v = keys
        attn_out = self.cross_attn_token_to_image(q, k, v)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP on queries
        queries = queries + self.mlp(queries)
        queries = self.norm3(queries)

        # Cross attention keys to queries
        q = keys + key_pe
        k = queries + query_pe
        v = queries
        attn_out = self.cross_attn_image_to_token(q, k, v)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    A basic Attention block. Simplified for mock purposes.
    """
    def __init__(
        self, embedding_dim: int, num_heads: int, q_bias: bool = False, kv_bias: bool = False
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=q_bias)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=kv_bias)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=kv_bias)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, N_q, C = q.shape
        _, N_k, _ = k.shape

        q = self.q_proj(q).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(k).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(v).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        out = (attn_weights @ v).transpose(1, 2).reshape(B, N_q, C)
        out = self.out_proj(out)
        return out


class MLP(nn.Module):
    """
    Simple MLP for heads.
    """
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, activation: Type[nn.Module] = nn.GELU
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# Placeholder PositionEmbeddingRandom and other types if not imported
# Ensure these are defined or imported from prompt_encoder.py
# For this file, we assume they are available or will be defined locally.

# --- BEGIN Placeholder for PositionEmbeddingRandom --- (Should be imported from prompt_encoder.py)
import math
from typing import Optional

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding for sparse prompts, similar to SAM.
    Generates random positional embeddings that are then scaled.
    """
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None:
            scale = 2 * math.pi
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((num_pos_feats, 2)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords @ self.positional_encoding_gaussian_matrix.to(coords.dtype)
        coords = 2 * math.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        h, w = size
        grid = torch.ones((h, w), device=self.positional_encoding_gaussian_matrix.device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe # (H, W, C)

    def forward_with_coords(self, coords_input: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(self.positional_encoding_gaussian_matrix.device))
# --- END Placeholder for PositionEmbeddingRandom ---


# Example usage (for testing the module structure)
if __name__ == "__main__":
    # Mock inputs
    batch_size = 1
    transformer_dim = 256
    image_h_w = 1024 // 16 # Example H/W for stride 16

    image_embeddings = torch.randn(batch_size, transformer_dim, image_h_w, image_h_w)
    image_pe = PositionEmbeddingRandom(transformer_dim // 2)( (image_h_w, image_h_w) ).unsqueeze(0)

    sparse_prompt_embeddings = torch.randn(batch_size, 5, transformer_dim) # 5 sparse tokens
    dense_prompt_embeddings = torch.randn(batch_size, transformer_dim, image_h_w, image_h_w)

    # Skip features with appropriate resolutions (e.g., H/4, W/4 and H/8, W/8)
    skip_feat4 = torch.randn(batch_size, transformer_dim, 1024 // 4, 1024 // 4)
    skip_feat8 = torch.randn(batch_size, transformer_dim, 1024 // 8, 1024 // 8)

    mask_decoder = MaskDecoder(
        transformer_dim=transformer_dim,
        transformer_n_heads=8,
        transformer_depth=2,
        transformer_mlp_dim=transformer_dim * 4,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )

    masks, iou_predictions, occlusion_predictions, object_pointer = mask_decoder(
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        skip_feat4,
        skip_feat8,
    )

    print(f"Masks shape: {masks.shape}") # Should be (B, num_mask_tokens, H_out, W_out)
    print(f"IoU Predictions shape: {iou_predictions.shape}") # Should be (B, num_mask_tokens)
    print(f"Occlusion Predictions shape: {occlusion_predictions.shape}") # Should be (B, 1)
    print(f"Object Pointer Embedding shape: {object_pointer.shape}") # Should be (B, C)

    expected_mask_h_w = 1024 // 4 # After upsampling from H/16, W/16 by factor of 4
    assert masks.shape[2] == expected_mask_h_w and masks.shape[3] == expected_mask_h_w
    print("MaskDecoder outputs match expected shapes.")

