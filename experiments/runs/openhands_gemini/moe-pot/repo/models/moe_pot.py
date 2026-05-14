
import torch
import torch.nn as nn
from einops import rearrange

from models.layers import PatchEmbedding, TemporalAggregation, FourierLayer
from models.experts import MoELayer

class MoEPOTBlock(nn.Module):
    """
    A single block of MoE-POT, consisting of a Fourier Layer and a MoE Layer.
    """
    def __init__(self, embed_dim: int, mlp_dim: int, num_heads: int, num_routed_experts: int, num_shared_experts: int, top_k: int, H_prime: int, W_prime: int):
        super().__init__()
        self.fourier_layer = FourierLayer(embed_dim, num_heads, H_prime, W_prime)
        self.moe_layer = MoELayer(embed_dim, mlp_dim, num_routed_experts, num_shared_experts, top_k)
        
        # Add Normalization layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        # x: (B, H', W', embed_dim)
        
        # Fourier Layer
        fourier_out = self.norm1(x + self.fourier_layer(x)) # Residual connection, then normalization
        
        # MoE Layer
        moe_out, lb_loss, expert_weights_flat = self.moe_layer(fourier_out)
        output = self.norm2(fourier_out + moe_out) # Residual connection, then normalization
        
        return output, lb_loss, expert_weights_flat

class MoEPOT(nn.Module):
    """
    The main Mixture-of-Experts Pre-training Operator Transformer (MoE-POT) model.
    """
    def __init__(self, 
                 patch_size: int, 
                 in_channels: int, 
                 out_channels: int, 
                 embed_dim: int, 
                 mlp_dim: int, 
                 num_layers: int, 
                 num_heads: int, 
                 num_routed_experts: int, 
                 num_shared_experts: int, 
                 top_k: int, 
                 H: int, 
                 W: int, 
                 time_steps: int):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.H = H
        self.W = W
        self.time_steps = time_steps

        self.H_prime = H // patch_size
        self.W_prime = W // patch_size

        # Input Encoding and Temporal Aggregation
        self.patch_embedding = PatchEmbedding(patch_size, in_channels, embed_dim, H, W)
        self.temporal_aggregation = TemporalAggregation(embed_dim, time_steps)

        # N MoE-POT Blocks
        self.blocks = nn.ModuleList([
            MoEPOTBlock(embed_dim, mlp_dim, num_heads, num_routed_experts, num_shared_experts, top_k, self.H_prime, self.W_prime)
            for _ in range(num_layers)
        ])

        # Output projection (e.g., a linear layer or CNN to map back to original channels)
        # Assuming a 1x1 convolution as a projector
        self.output_projection = nn.Conv2d(embed_dim, out_channels, kernel_size=1)


    def forward(self, u_t_minus_T: torch.Tensor):
        # u_t_minus_T: (B, T, H, W, C) - input sequence of T frames

        # Input Encoding and Temporal Aggregation
        # z_p: (B, T, H', W', embed_dim)
        z_p = self.patch_embedding(u_t_minus_T)
        
        # z_agg: (B, H', W', embed_dim) - aggregated features
        z_agg = self.temporal_aggregation(z_p)

        # Pass through N MoE-POT Blocks
        current_features = z_agg
        total_lb_loss = 0.0
        all_expert_weights = []

        for block in self.blocks:
            current_features, lb_loss, expert_weights_flat = block(current_features)
            total_lb_loss += lb_loss
            all_expert_weights.append(expert_weights_flat)
        
        # Output projection
        # current_features: (B, H', W', embed_dim)
        # Permute to (B, embed_dim, H', W') for Conv2d
        output = rearrange(current_features, 'b h w c -> b c h w')
        output = self.output_projection(output)
        
        # Upscale output to original H, W if H' != H, W' != W
        # Assuming output should match original spatial resolution for prediction
        # If patch_size > 1, the output will be H' x W'. Need to upsample.
        # The paper uses a patchification layer then "predicts the next frame".
        # This implies the output should be (B, C_out, H, W) or (B, H, W, C_out)
        
        if self.H_prime != self.H or self.W_prime != self.W:
            # Using interpolation to upsample
            output = F.interpolate(output, size=(self.H, self.W), mode='bilinear', align_corners=False)
        
        # Output: (B, C_out, H, W). Rearrange to (B, H, W, C_out) for consistency with input
        output = rearrange(output, 'b c h w -> b h w c')

        return output, total_lb_loss, all_expert_weights

if __name__ == '__main__':
    from config import config
    
    # Example usage with config
    config.model.patch_size = 8
    config.model.in_channels = 3 # Example, e.g., (vx, vy, pressure)
    config.model.out_channels = 3 # Predict next state
    config.model.attention_dim = 32 # embed_dim
    config.model.mlp_dim = 64
    config.model.num_layers = 2
    config.model.num_heads = 4
    config.model.num_routed_experts = 16
    config.model.num_shared_experts = 2
    config.model.top_k_experts = 4
    config.data.h_resolution = 64 # H
    config.data.time_steps = 5 # T

    model = MoEPOT(
        patch_size=config.model.patch_size,
        in_channels=C_in,
        out_channels=C_out,
        embed_dim=config.model.attention_dim,
        mlp_dim=config.model.mlp_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        num_routed_experts=config.model.num_routed_experts,
        num_shared_experts=config.model.num_shared_experts,
        top_k=config.model.top_k_experts,
        H=config.data.h_resolution,
        W=config.data.h_resolution,
        time_steps=config.data.time_steps
    )

    # Example input: (B, T, H, W, C_in)
    input_data = torch.randn(B, T, H, W, C_in)

    output, total_lb_loss, all_expert_weights = model(input_data)

    print(f"MoE-POT output shape: {output.shape}")
    print(f"Total Load Balancing Loss: {total_lb_loss.item()}")
    print(f"Number of expert weights lists (per block): {len(all_expert_weights)}")
    if len(all_expert_weights) > 0:
        print(f"Shape of first expert weights list: {all_expert_weights[0].shape}")

    assert output.shape == (B, H, W, C_out)
    assert len(all_expert_weights) == config.model.num_layers

    print("MoE-POT model tested successfully!")
