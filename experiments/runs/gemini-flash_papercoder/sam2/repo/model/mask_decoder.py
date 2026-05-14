import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Dict, List, Optional, Tuple, Union

# Placeholder for Config type hint to avoid circular import with config.py
Config = Any

# Importing common building blocks from memory_attention to maintain consistency
# Assuming these are defined similarly to SAM's components and are generic enough.
from model.memory_attention import MLP, Attention


class TwoWayTransformerBlock(nn.Module):
    """
    A single block of the Two-Way Transformer as used in SAM's mask decoder.
    It performs self-attention on image embeddings, self-attention on prompt embeddings,
    and cross-attention between them.
    """

    def __init__(self, hidden_dim: int, num_heads: int, drop_rate: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn_image = Attention(
            q_dim=hidden_dim, kv_dim=hidden_dim, hidden_dim=hidden_dim, num_heads=num_heads, drop_rate=drop_rate
        )

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.self_attn_prompt = Attention(
            q_dim=hidden_dim, kv_dim=hidden_dim, hidden_dim=hidden_dim, num_heads=num_heads, drop_rate=drop_rate
        )

        self.norm3 = nn.LayerNorm(hidden_dim)
        # Cross-attention from image to prompt tokens
        self.cross_attn_image_to_prompt = Attention(
            q_dim=hidden_dim, kv_dim=hidden_dim, hidden_dim=hidden_dim, num_heads=num_heads, drop_rate=drop_rate
        )

        self.norm4 = nn.LayerNorm(hidden_dim)
        # Cross-attention from prompt to image tokens
        self.cross_attn_prompt_to_image = Attention(
            q_dim=hidden_dim, kv_dim=hidden_dim, hidden_dim=hidden_dim, num_heads=num_heads, drop_rate=drop_rate
        )

        self.norm5 = nn.LayerNorm(hidden_dim)
        self.mlp_image = MLP(hidden_dim, hidden_dim * 4, drop=drop_rate)

        self.norm6 = nn.LayerNorm(hidden_dim)
        self.mlp_prompt = MLP(hidden_dim, hidden_dim * 4, drop=drop_rate)

    def forward(
        self,
        image_tokens: torch.Tensor,  # (B, L_img, hidden_dim)
        prompt_tokens: torch.Tensor,  # (B, L_prompt, hidden_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single TwoWayTransformerBlock.

        Args:
            image_tokens (torch.Tensor): Image feature tokens.
            prompt_tokens (torch.Tensor): Prompt feature tokens.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated image and prompt tokens.
        """
        # Self-attention for image tokens
        norm_image_sa = self.norm1(image_tokens)
        image_tokens_sa = self.self_attn_image(norm_image_sa, norm_image_sa, norm_image_sa)
        image_tokens = image_tokens + image_tokens_sa

        # Self-attention for prompt tokens
        norm_prompt_sa = self.norm2(prompt_tokens)
        prompt_tokens_sa = self.self_attn_prompt(norm_prompt_sa, norm_prompt_sa, norm_prompt_sa)
        prompt_tokens = prompt_tokens + prompt_tokens_sa

        # Cross-attention: Image to Prompt
        norm_prompt_cross = self.norm3(prompt_tokens)
        prompt_tokens_cross = self.cross_attn_image_to_prompt(norm_prompt_cross, image_tokens, image_tokens)
        prompt_tokens = prompt_tokens + prompt_tokens_cross

        # Cross-attention: Prompt to Image
        norm_image_cross = self.norm4(image_tokens)
        image_tokens_cross = self.cross_attn_prompt_to_image(norm_image_cross, prompt_tokens, prompt_tokens)
        image_tokens = image_tokens + image_tokens_cross

        # MLPs for both streams
        image_tokens = image_tokens + self.mlp_image(self.norm5(image_tokens))
        prompt_tokens = prompt_tokens + self.mlp_prompt(self.norm6(prompt_tokens))

        return image_tokens, prompt_tokens


class TwoWayTransformer(nn.Module):
    """
    Stacks multiple TwoWayTransformerBlocks to allow interaction between
    image features and prompt features.
    """

    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int, drop_rate: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            TwoWayTransformerBlock(hidden_dim, num_heads, drop_rate)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        image_tokens: torch.Tensor,
        prompt_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_tokens (torch.Tensor): Image feature tokens (B, L_img, hidden_dim).
            prompt_tokens (torch.Tensor): Prompt feature tokens (B, L_prompt, hidden_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated image and prompt tokens.
        """
        for layer in self.layers:
            image_tokens, prompt_tokens = layer(image_tokens, prompt_tokens)
        return image_tokens, prompt_tokens


class MaskDecoder(nn.Module):
    """
    Decodes segmentation masks, IoU scores, and occlusion probabilities from
    conditioned image features and prompt embeddings.
    """

    def __init__(self, config: Config):
        """
        Initializes the MaskDecoder.

        Args:
            config (Config): The global configuration object.
        """
        super().__init__()
        self._config = config

        self.transformer_layers: int = self._config.get("model.mask_decoder.transformer_layers", 2)
        self.use_occlusion_head: bool = self._config.get("model.mask_decoder.use_occlusion_head", True)
        self.predict_multiple_masks: bool = self._config.get("model.mask_decoder.predict_multiple_masks", True)
        # num_mask_tokens in config.yaml refers to number of mask variants, typically 3 for SAM.
        self.num_mask_variants: int = self._config.get("model.mask_decoder.num_mask_tokens", 3)

        # Image embedding dimension (from ImageEncoder/MemoryAttention)
        self.image_embedding_dim: int = self._config.get("model.memory_attention.hidden_dim", 256)
        # Prompt embedding dimension (from PromptEncoder, assumed same as image_embedding_dim for consistency)
        self.prompt_embedding_dim: int = self._config.get("model.memory_attention.hidden_dim", 256)
        # Number of heads for attention in transformer blocks
        self.num_heads: int = self._config.get("model.memory_attention.num_heads", 8)

        if self.image_embedding_dim != self.prompt_embedding_dim:
            # This is a strong assumption in SAM-like architectures
            raise ValueError(
                f"Image embedding dim ({self.image_embedding_dim}) must match "
                f"prompt embedding dim ({self.prompt_embedding_dim}) for TwoWayTransformer."
            )

        # Learned Query Tokens
        # These are tokens that interact with image features to produce outputs
        self.mask_tokens = nn.Parameter(torch.randn(1, self.num_mask_variants, self.prompt_embedding_dim))
        self.iou_token = nn.Parameter(torch.randn(1, 1, self.prompt_embedding_dim))
        
        if self.use_occlusion_head:
            self.occlusion_token = nn.Parameter(torch.randn(1, 1, self.prompt_embedding_dim))

        # Two-way Transformer for interaction between image and prompt embeddings
        self.transformer = TwoWayTransformer(
            hidden_dim=self.image_embedding_dim,
            num_layers=self.transformer_layers,
            num_heads=self.num_heads
        )

        # Output Prediction Heads
        # IoU Head: predicts the quality of the predicted mask
        self.iou_head = nn.Sequential(
            nn.Linear(self.prompt_embedding_dim, self.prompt_embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.prompt_embedding_dim // 2, 1)
        )

        # Occlusion Head: predicts if the object is present/visible in the frame
        if self.use_occlusion_head:
            self.occlusion_head = nn.Sequential(
                nn.Linear(self.prompt_embedding_dim, self.prompt_embedding_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.prompt_embedding_dim // 2, 1)
            )

        # Mask Upsampling Path
        # The conditioned image_embedding from MemoryAttention is typically at stride 16.
        # We need to upsample it and combine with skip connections from ImageEncoder
        # C1 (stride 4), C2 (stride 8), image_embedding (stride 16)
        
        # Output channels for upsampling path (to match hidden_dim for consistency)
        upsample_channels = self.image_embedding_dim

        # Skip connection from ImageEncoder's stride 8 features (C2)
        # Needs to match `image_embedding_dim` before adding.
        # Hiera-B+ C2_channels = 256. If image_embedding_dim = 256, then it's a direct match,
        # otherwise a conv layer is needed.
        hiera_type = self._config.get("model.image_encoder.type", "Hiera-B+")
        hiera_config = _HIERA_CONFIGS.get(hiera_type, _HIERA_CONFIGS["Hiera-B+"])
        c2_channels_from_image_encoder = hiera_config["out_channels"][1] # Channels for stride 8

        if c2_channels_from_image_encoder != upsample_channels:
            self.skip_conv_c2 = nn.Conv2d(c2_channels_from_image_encoder, upsample_channels, kernel_size=1)
        else:
            self.skip_conv_c2 = nn.Identity()

        # Skip connection from ImageEncoder's stride 4 features (C1)
        # Hiera-B+ C1_channels = 128.
        c1_channels_from_image_encoder = hiera_config["out_channels"][0] # Channels for stride 4

        if c1_channels_from_image_encoder != upsample_channels:
            self.skip_conv_c1 = nn.Conv2d(c1_channels_from_image_encoder, upsample_channels, kernel_size=1)
        else:
            self.skip_conv_c1 = nn.Identity()


        self.upsample_layers = nn.ModuleList([
            # Upsample from stride 16 to stride 8
            nn.Sequential(
                nn.ConvTranspose2d(self.image_embedding_dim, upsample_channels, kernel_size=2, stride=2),
                nn.GroupNorm(min(32, upsample_channels), upsample_channels),
                nn.ReLU(inplace=True)
            ),
            # Upsample from stride 8 to stride 4
            nn.Sequential(
                nn.ConvTranspose2d(upsample_channels, upsample_channels, kernel_size=2, stride=2),
                nn.GroupNorm(min(32, upsample_channels), upsample_channels),
                nn.ReLU(inplace=True)
            ),
        ])

        # Hypernetwork for dynamic mask head
        # It takes the mask tokens and generates weights and biases for a 1x1 conv layer.
        # The output conv layer transforms `upsample_channels` to 1 output channel.
        # So, it needs to generate `num_mask_variants * (upsample_channels * 1 + 1)` parameters.
        # Each mask variant gets its own 1x1 conv.
        self.output_hypernetwork_mlp = nn.Sequential(
            nn.Linear(self.prompt_embedding_dim, upsample_channels * 1 + 1), # weights and bias for a 1x1 conv
            nn.ReLU(inplace=True),
            nn.Linear(upsample_channels * 1 + 1, self.num_mask_variants * (upsample_channels * 1 + 1))
        )
        
        # We need a placeholder for the final dynamic convolution.
        # The weights/bias will be populated in the forward pass.


    def forward(
        self,
        image_embedding: torch.Tensor,  # (B, C_img, H_img, W_img) (from MemoryAttention, e.g., stride 16)
        prompt_embeddings: torch.Tensor,  # (B, N_prompts, C_prompt) (from PromptEncoder)
        multi_scale_features: List[torch.Tensor],  # [C1_feats (stride 4), C2_feats (stride 8)] from ImageEncoder
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Performs the forward pass of the MaskDecoder.

        Args:
            image_embedding (torch.Tensor): Conditioned image feature map.
                                            Shape: (B, image_embedding_dim, H_orig/16, W_orig/16).
            prompt_embeddings (torch.Tensor): Embeddings of user-provided prompts.
                                              Shape: (B, N_prompts, prompt_embedding_dim).
            multi_scale_features (List[torch.Tensor]): High-resolution features for skip connections.
                                                        Expected: [C1 (B, C_C1, H_orig/4, W_orig/4),
                                                                   C2 (B, C_C2, H_orig/8, W_orig/8)].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                - masks_probs (torch.Tensor): Predicted segmentation masks.
                                            Shape: (B, num_mask_variants, H_orig/4, W_orig/4).
                - iou_preds_probs (torch.Tensor): Predicted IoU scores. Shape: (B, 1).
                - occlusion_preds_probs (Optional[torch.Tensor]): Predicted occlusion probabilities.
                                                                  Shape: (B, 1) or None.
        """
        B, C_img, H_img, W_img = image_embedding.shape
        H_orig_div_16, W_orig_div_16 = H_img, W_img

        C1_feats, C2_feats = multi_scale_features[0], multi_scale_features[1]

        # 1. Prepare Transformer Inputs
        # Image tokens: Flatten spatial dimensions
        image_tokens = image_embedding.view(B, C_img, -1).permute(0, 2, 1)  # (B, H_img*W_img, C_img)

        # Query tokens: Concatenate prompt_embeddings with learned tokens
        learned_tokens = [self.mask_tokens.expand(B, -1, -1), self.iou_token.expand(B, -1, -1)]
        if self.use_occlusion_head:
            learned_tokens.append(self.occlusion_token.expand(B, -1, -1))
        
        all_learned_tokens = torch.cat(learned_tokens, dim=1)
        query_tokens = torch.cat([prompt_embeddings, all_learned_tokens], dim=1) # (B, N_total_queries, C_prompt)

        # 2. Two-way Transformer Interaction
        image_tokens_out, query_tokens_out = self.transformer(image_tokens, query_tokens)

        # 3. Extract Output Token Embeddings
        # The order of learned tokens: mask_tokens, iou_token, occlusion_token (if enabled)
        # So we can slice query_tokens_out accordingly.
        # Start index for learned tokens after prompt_embeddings
        learned_tokens_start_idx = prompt_embeddings.shape[1]

        mask_tokens_out = query_tokens_out[:, learned_tokens_start_idx : learned_tokens_start_idx + self.num_mask_variants]
        iou_token_out = query_tokens_out[:, learned_tokens_start_idx + self.num_mask_variants : learned_tokens_start_idx + self.num_mask_variants + 1]
        
        occlusion_preds_probs: Optional[torch.Tensor] = None
        if self.use_occlusion_head:
            occlusion_token_out = query_tokens_out[:, learned_tokens_start_idx + self.num_mask_variants + 1 : learned_tokens_start_idx + self.num_mask_variants + 2]
            # 5. Predict Occlusion
            occlusion_logits = self.occlusion_head(occlusion_token_out.squeeze(1))
            occlusion_preds_probs = torch.sigmoid(occlusion_logits)


        # 4. Predict IoU
        iou_logits = self.iou_head(iou_token_out.squeeze(1))
        iou_preds_probs = torch.sigmoid(iou_logits) # Sigmoid for probability between 0 and 1

        # 6. Generate Mask Logits
        # Reshape image_tokens_out back to spatial dimensions
        upsampled_image_features = image_tokens_out.permute(0, 2, 1).view(B, C_img, H_orig_div_16, W_orig_div_16)

        # Upsample 1 (from stride 16 to stride 8) and add C2 skip connection
        upsampled_image_features = self.upsample_layers[0](upsampled_image_features) # (B, upsample_channels, H/8, W/8)
        c2_processed = self.skip_conv_c2(C2_feats) # (B, upsample_channels, H/8, W/8)
        upsampled_image_features = upsampled_image_features + c2_processed

        # Upsample 2 (from stride 8 to stride 4) and add C1 skip connection
        upsampled_image_features = self.upsample_layers[1](upsampled_image_features) # (B, upsample_channels, H/4, W/4)
        c1_processed = self.skip_conv_c1(C1_feats) # (B, upsample_channels, H/4, W/4)
        upsampled_image_features = upsampled_image_features + c1_processed

        # Dynamic Mask Head (Hypernetwork)
        # mask_tokens_out shape (B, num_mask_variants, prompt_embedding_dim)
        # Hypernetwork output generates weights/biases for num_mask_variants 1x1 convs.
        hypernetwork_output = self.output_hypernetwork_mlp(mask_tokens_out.flatten(0, 1)) # (B * num_mask_variants, params_per_conv)

        # Reshape hypernetwork output to get individual weights and biases for each mask variant
        # Params per conv: (upsample_channels * 1) for weights + 1 for bias
        num_params_per_conv = upsample_channels + 1 
        weights = hypernetwork_output[:, :-1].reshape(B * self.num_mask_variants, 1, upsample_channels, 1, 1) # (B*N_v, out_C, in_C, kH, kW)
        biases = hypernetwork_output[:, -1:].reshape(B * self.num_mask_variants, 1) # (B*N_v, out_C)

        # Apply the dynamically generated convolutions
        # Need to reshape upsampled_image_features to apply batched convolution
        # (B, C, H, W) -> (B*N_v, C, H, W) by repeating
        upsampled_image_features_repeated = upsampled_image_features.unsqueeze(1).repeat(1, self.num_mask_variants, 1, 1, 1).flatten(0, 1)

        mask_logits = F.conv2d(
            upsampled_image_features_repeated,
            weights,
            biases.flatten(), # F.conv2d expects 1D bias
            padding=0,
            stride=1,
            groups=B * self.num_mask_variants # Apply each 1x1 conv independently
        )
        mask_logits = mask_logits.view(B, self.num_mask_variants, upsampled_image_features.shape[-2], upsampled_image_features.shape[-1])
        
        # 7. Apply Activations and Return
        masks_probs = torch.sigmoid(mask_logits)

        return masks_probs, iou_preds_probs, occlusion_preds_probs

# Mock _HIERA_CONFIGS from ImageEncoder for internal use in MaskDecoder.
# This should ideally be a shared config/constant across modules if needed.
_HIERA_CONFIGS = {
    "Hiera-T": {
        "out_channels": [96, 192, 384, 768],
        "patch_size": 16,
        "global_attn_blocks": [5, 7, 9],
    },
    "Hiera-S": {
        "out_channels": [96, 192, 384, 768],
        "patch_size": 16,
        "global_attn_blocks": [7, 10, 13],
    },
    "Hiera-B+": {
        "out_channels": [128, 256, 512, 1024],
        "patch_size": 16,
        "global_attn_blocks": [12, 16, 20],
    },
    "Hiera-L": {
        "out_channels": [192, 384, 768, 1536],
        "patch_size": 16,
        "global_attn_blocks": [23, 33, 43],
    },
}

