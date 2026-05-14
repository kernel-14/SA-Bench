import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import PatchEmbedding, TemporalAggregation, FourierLayer, MoELayer
from .utils import calculate_load_balancing_loss

class MoE_POT(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 patch_size, 
                 embed_dim, 
                 img_size, 
                 fourier_feature_constant_dim, 
                 num_fourier_heads, 
                 num_moe_blocks, 
                 num_routed_experts, 
                 num_shared_experts, 
                 moe_top_k,
                 weight_balance_loss=0.1
                ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.img_size = img_size # (H, W)
        self.fourier_feature_constant_dim = fourier_feature_constant_dim
        self.num_fourier_heads = num_fourier_heads
        self.num_moe_blocks = num_moe_blocks
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.moe_top_k = moe_top_k
        self.weight_balance_loss = weight_balance_loss

        # Section 4: "It begins by processing raw data through a patchification layer and a temporal aggregation layer [15]"
        # Input: u^{<T} in R^(H x W x T x C) -> (B, T, C, H, W)
        # PatchEmbedding expects (B, C_in, H, W) for *each* time step.
        # TemporalAggregation expects (B, T, embed_dim, H_patches, W_patches)
        
        # Assuming C (in_channels) is the channel dimension of the raw input u^t.
        # For patchification, we need to apply it to each time step independently.
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim, img_size)
        
        # H_patches, W_patches are the output dimensions after patchification.
        num_patches_h = img_size[0] // patch_size
        num_patches_w = img_size[1] // patch_size
        self.img_size_patches = (num_patches_h, num_patches_w)

        self.temporal_aggregation = TemporalAggregation(embed_dim, fourier_feature_constant_dim)

        # Section 4: "The processed features are then passed through N blocks, each of which contains a Fourier layer [13] and a MoE layer"
        self.moe_blocks = nn.ModuleList()
        for _ in range(num_moe_blocks):
            self.moe_blocks.append(nn.ModuleDict({
                'fourier_layer': FourierLayer(embed_dim, num_fourier_heads, self.img_size_patches),
                'moe_layer': MoELayer(embed_dim, num_routed_experts, num_shared_experts, moe_top_k)
            }))
            
        # Final projection layer to map the output features back to the desired output channels.
        # Assuming the output of the last MoE layer is (B, embed_dim, H_p, W_p) (real part).
        # We need to upsample to original H, W and change channels to out_channels.
        # This could be a convolutional transpose or a series of convolutions.
        # For simplicity, let's use a Conv2dTranspose for upsampling.
        # It needs to output (B, out_channels, H, W) to match u^t.
        
        # The patchification layer reduced H,W by patch_size.
        # We need to reverse this.
        self.output_proj = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, out_channels, kernel_size=1)
        )

    def forward(self, u_t_minus_T_to_t_minus_1):
        # u_t_minus_T_to_t_minus_1: (B, T, C_in, H, W) - sequence of T previous frames
        B, T, C_in, H, W = u_t_minus_T_to_t_minus_1.shape

        # 1. Patchification for each time step
        # Apply patch_embed to each time step. Reshape input to (B*T, C_in, H, W) for batch processing.
        # Then reshape back to (B, T, embed_dim, H_p, W_p)
        patched_frames = []
        for t_idx in range(T):
            u_t = u_t_minus_T_to_t_minus_1[:, t_idx, :, :, :]
            patched_frames.append(self.patch_embed(u_t)) # (B, embed_dim, H_p, W_p)
        z_p_t_sequence = torch.stack(patched_frames, dim=1) # (B, T, embed_dim, H_p, W_p)

        # 2. Temporal Aggregation
        z_agg = self.temporal_aggregation(z_p_t_sequence) # (B, embed_dim, H_p, W_p) complex
        
        # Initialize z_l with the aggregated output. The MoE layer expects real input.
        # The Fourier layer outputs complex, but the MoE layer uses the real part.
        # So we pass the complex output from temporal_aggregation to the first Fourier layer.
        current_features = z_agg # This is (B, embed_dim, H_p, W_p) complex.

        # Collect load balancing losses from each MoE layer
        all_load_balancing_losses = []

        # 3. N MoE Blocks
        for block_idx, moe_block in enumerate(self.moe_blocks):
            # Fourier Layer operates on complex features and outputs complex features.
            fourier_output = moe_block['fourier_layer'](current_features)
            
            # MoE Layer expects real input. We will take the real part of the Fourier output
            # for the expert networks, but the router might implicitly use magnitude or other features.
            # For now, let's explicitly pass the real part to the MoE layer's convolutional expert networks.
            moe_output = moe_block['moe_layer'](fourier_output) # MoE layer output is real (B, embed_dim, H_p, W_p)
            current_features = moe_output # For next block, we feed the real output of MoE.
            
            # Retrieve and store the load balancing loss for this block
            gating_weights_all = moe_block['moe_layer'].gating_weights_all_for_loss
            loss_balance = calculate_load_balancing_loss(gating_weights_all, self.num_routed_experts, self.weight_balance_loss)
            all_load_balancing_losses.append(loss_balance)

        # 4. Final Output Projection
        # The last `current_features` is real (B, embed_dim, H_p, W_p).
        output_frame = self.output_proj(current_features) # (B, out_channels, H, W)

        # Sum all load balancing losses
        total_load_balancing_loss = sum(all_load_balancing_losses) if all_load_balancing_losses else torch.tensor(0.0, device=output_frame.device)

        return output_frame, total_load_balancing_loss
