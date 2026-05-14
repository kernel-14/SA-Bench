import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple, List, Dict, Any, Optional, Union

# Assuming Config class is available from config.py
# This is a stub for local development/testing without full config.py
try:
    from config import Config, VaeConfig
    # Import utility functions, assuming they handle 3D tensors correctly
    from utils import downsample, upsample
except ImportError:
    print("Warning: config.py or utils.py not found. Using stub classes.")
    # Stubs for local development/testing
    class VaeConfig:
        name: str = "VideoVAE"
        compression_rate: List[int] = [8, 8, 8] # [Temporal, Height, Width]
        latent_channels: int = 4
        base_channels: int = 128
        num_res_blocks: int = 2
        attn_resolutions: List[int] = [16] # Spatial resolutions where attention blocks are added
        use_causal_conv3d: bool = True
        kl_regularization: bool = True

    class ModelConfig:
        vae: VaeConfig = VaeConfig()

    class Config:
        model: ModelConfig = ModelConfig()

    # Minimal stub for downsample/upsample if utils.py is not available
    # Note: This stub only handles spatial dimensions (H, W).
    # Actual utils.py implementation should handle 3D tensors as designed.
    def downsample(tensor: torch.Tensor, factor: int, mode: str = "bilinear") -> torch.Tensor:
        # Assuming tensor is (B, C, T, H, W) or (B, C, H, W)
        if tensor.ndim == 5: # Video (B, C, T, H, W)
            return F.interpolate(tensor, size=(tensor.shape[2] // factor, tensor.shape[3] // factor, tensor.shape[4] // factor), mode="trilinear", align_corners=False)
        elif tensor.ndim == 4: # Image (B, C, H, W)
            return F.interpolate(tensor, size=(tensor.shape[2] // factor, tensor.shape[3] // factor), mode=mode, align_corners=False)
        else:
            raise NotImplementedError("Stub downsample for other tensor dimensions not implemented.")

    def upsample(tensor: torch.Tensor, factor: int, mode: str = "bilinear") -> torch.Tensor:
        # Assuming tensor is (B, C, T, H, W) or (B, C, H, W)
        if tensor.ndim == 5: # Video (B, C, T, H, W)
            return F.interpolate(tensor, size=(tensor.shape[2] * factor, tensor.shape[3] * factor, tensor.shape[4] * factor), mode="trilinear", align_corners=False)
        elif tensor.ndim == 4: # Image (B, C, H, W)
            return F.interpolate(tensor, size=(tensor.shape[2] * factor, tensor.shape[3] * factor), mode=mode, align_corners=False)
        else:
            raise NotImplementedError("Stub upsample for other tensor dimensions not implemented.")


# --- Helper Modules ---

class CausalConv3d(nn.Module):
    """
    A 3D convolutional layer that ensures causality along the temporal dimension.
    Padding is applied only to the 'past' temporal dimension.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int, int]],
                 stride: Union[int, Tuple[int, int, int]] = 1, padding: Union[int, Tuple[int, int, int]] = 0,
                 dilation: Union[int, Tuple[int, int, int]] = 1, groups: int = 1, bias: bool = True):
        super().__init__()
        
        # Ensure kernel_size, stride, padding, dilation are tuples
        kernel_size = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        stride = (stride, stride, stride) if isinstance(stride, int) else stride
        padding = (padding, padding, padding) if isinstance(padding, int) else padding
        dilation = (dilation, dilation, dilation) if isinstance(dilation, int) else dilation

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        
        # Causal padding for temporal dimension: (kernel_size_t - 1) * dilation_t
        self.temporal_padding = (self.kernel_size[0] - 1) * self.dilation[0]
        
        # The actual nn.Conv3d will have zero temporal padding explicitly
        # and spatial padding as specified.
        self.conv = nn.Conv3d(in_channels, out_channels, self.kernel_size, stride=self.stride,
                              padding=(0, padding[1], padding[2]), dilation=self.dilation,
                              groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad only the beginning of the temporal dimension
        x = F.pad(x, (0, 0, 0, 0, self.temporal_padding, 0))
        return self.conv(x)


class ResnetBlock3D(nn.Module):
    """
    3D Residual block with GroupNorm and SiLU activation.
    """
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.nin_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return self.nin_shortcut(x) + h


class AttentionBlock3D(nn.Module):
    """
    3D Attention block that performs spatial self-attention.
    Flattens the temporal dimension into the batch dimension for 2D spatial attention.
    """
    def __init__(self, in_channels: int, num_heads: int = 8, dim_head: int = 64, num_groups: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.hidden_dim = num_heads * dim_head # For multihead attention

        self.norm = nn.GroupNorm(num_groups, in_channels)
        self.q = CausalConv3d(in_channels, self.hidden_dim, kernel_size=1)
        self.k = CausalConv3d(in_channels, self.hidden_dim, kernel_size=1)
        self.v = CausalConv3d(in_channels, self.hidden_dim, kernel_size=1)
        self.proj_out = CausalConv3d(self.hidden_dim, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        h_ = self.norm(x)

        # Flatten T into B for 2D spatial attention
        h_ = rearrange(h_, 'b c t h w -> (b t) c h w')

        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Reshape for multi-head attention: (B*T, num_heads, C_head, H*W)
        q = rearrange(q, 'bt (heads dim_head) h w -> bt heads dim_head (h w)', heads=self.num_heads)
        k = rearrange(k, 'bt (heads dim_head) h w -> bt heads dim_head (h w)', heads=self.num_heads)
        v = rearrange(v, 'bt (heads dim_head) h w -> bt heads dim_head (h w)', heads=self.num_heads)

        # Permute for batch matrix multiplication: (B*T, heads, H*W, dim_head)
        q = q.permute(0, 1, 3, 2) 
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        
        # compute attention scores: (B*T, heads, H*W, H*W)
        attn = torch.matmul(q, k.transpose(-1, -2)) * (self.dim_head ** -0.5)
        attn = F.softmax(attn, dim=-1)

        # Apply attention to V: (B*T, heads, H*W, dim_head)
        h_ = torch.matmul(attn, v)
        
        # Reshape back to (B*T, C, H, W)
        h_ = rearrange(h_.permute(0, 1, 3, 2), 'bt heads dim_head (h w) -> bt (heads dim_head) h w', h=H, w=W)

        h_ = self.proj_out(h_)
        
        # Reshape back to 5D
        h_ = rearrange(h_, '(b t) c h w -> b c t h w', b=B, t=T)

        return x + h_


class Downsample3D(nn.Module):
    """
    Downsampling layer for 3D feature maps using a strided causal convolution.
    Reduces temporal, spatial height, and spatial width dimensions by 2x.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Stride 2 for all dimensions (temporal, height, width)
        self.conv = CausalConv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    """
    Upsampling layer for 3D feature maps using nearest interpolation and a causal convolution.
    Increases temporal, spatial height, and spatial width dimensions by 2x.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = CausalConv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        # Interpolate across temporal, height, and width by a factor of 2
        x = F.interpolate(x, size=(T * 2, H * 2, W * 2), mode='nearest')
        return self.conv(x)


# --- Main VideoVAE Class ---

class VideoVAE(nn.Module):
    """
    3D Variational Autoencoder (VAE) for video compression and reconstruction.
    Architecture similar to MAGVIT-v2, incorporating 3D causal convolutions.
    """
    def __init__(self, config: Config):
        super().__init__()
        vae_config: VaeConfig = config.model.vae
        
        # Input channels for raw video (e.g., RGB is 3)
        self.input_channels = 3 
        self.latent_channels = vae_config.latent_channels # Latent space channels (e.g., 4)
        self.base_channels = vae_config.base_channels
        self.num_res_blocks = vae_config.num_res_blocks
        self.attn_resolutions = vae_config.attn_resolutions # Spatial resolutions for attention
        self.kl_regularization = vae_config.kl_regularization

        # Calculate number of downsampling stages from compression_rate [T_comp, H_comp, W_comp]
        # Assuming each stage downsamples by 2x in T, H, W.
        if not (vae_config.compression_rate[0] == vae_config.compression_rate[1] == vae_config.compression_rate[2]):
            raise ValueError("All dimensions in compression_rate must be equal for consistent downsampling.")
        
        self.num_down_stages = int(torch.log2(torch.tensor(float(vae_config.compression_rate[0]))).item())
        if not (2**self.num_down_stages == vae_config.compression_rate[0]):
             raise ValueError("Compression rate must be a power of 2 for each dimension.")

        # Channel multipliers for UNet-like structure
        # E.g., for num_down_stages=3, channel_mult = [1, 2, 4, 4] implies:
        # base_channels -> 2*base_channels -> 4*base_channels -> 4*base_channels (deepest)
        self.channel_mult = [2**i for i in range(self.num_down_stages)] + [2**(self.num_down_stages - 1)] 
        
        # --- Encoder ---
        self.encoder = nn.ModuleList()
        # Initial convolution (e.g., 3 input channels to base_channels)
        self.encoder.append(CausalConv3d(self.input_channels, self.base_channels, kernel_size=3, padding=1))
        
        curr_channels = self.base_channels
        
        for i in range(self.num_down_stages):
            out_channels = self.base_channels * self.channel_mult[i]
            
            # Encoder block for current stage
            encoder_block = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                encoder_block.append(ResnetBlock3D(curr_channels, out_channels))
                curr_channels = out_channels
            
            # Add attention if current (approximate) spatial resolution matches
            # This calculation assumes that initial H/W are multiples of 2^num_down_stages.
            # And attn_resolutions are relative to the initial resolution after initial downsampling stages.
            # A more robust check might involve inferring actual H/W based on input_video.shape
            # and downsampling steps. For now, we assume attn_resolutions are targets for H/W.
            # We use a rough estimate: (e.g. 128 / 2^i)
            # If initial H/W is say 128, and attn_resolutions is [16].
            # Stage 0: 128 -> (Resnet, Attn(if 128 in attn_res)) -> Downsample (64)
            # Stage 1: 64  -> (Resnet, Attn(if 64 in attn_res)) -> Downsample (32)
            # Stage 2: 32  -> (Resnet, Attn(if 32 in attn_res)) -> Downsample (16)
            # This logic needs adjustment or clarification. For now, let's assume `attn_resolutions`
            # are the spatial sizes at which attention should occur after convolution but before downsampling.
            # We'll need to pass the actual `spatial_res` to AttentionBlock3D if needed for its internal logic.
            # For simplicity, we assume `attn_resolutions` are the spatial sizes where we want to apply attention.
            # The current implementation of AttentionBlock3D doesn't depend on `spatial_res` directly,
            # it just takes `in_channels`. So we don't need a dynamic `current_spatial_res` in its init.
            
            # The paper says: "attn_resolutions: [16] # Placeholder, common".
            # This implies attention should be added when the *spatial* dimension is 16.
            # For a 128x128 input, this would be after 3 downsampling stages.
            # We'll use a placeholder logic that attempts to match resolutions for attention placement.
            # This part is a common point of ambiguity in paper reproductions without exact details.
            
            # Instead of deriving current_spatial_res, let's just add attention if `curr_channels`
            # leads to a specific target resolution in the VAE structure.
            # Given that VAE often has symmetric structure and attention at certain scales:
            # Let's add attention at resolution 16x16, which is usually after final downsampling (before bottleneck)
            # or before middle block. This is a heuristic.
            if i == self.num_down_stages - 1 and self.attn_resolutions and self.attn_resolutions[0] == 16:
                encoder_block.append(AttentionBlock3D(curr_channels)) # Add attention at deepest encoder stage
            
            if i < self.num_down_stages - 1: # Downsample in all but the last encoder stage
                # out_channels for Downsample3D is usually the next curr_channels from channel_mult
                encoder_block.append(Downsample3D(curr_channels, self.base_channels * self.channel_mult[i+1]))
                curr_channels = self.base_channels * self.channel_mult[i+1] # Update curr_channels after downsampling
            
            self.encoder.append(nn.Sequential(*encoder_block))
        
        # Middle block for encoder (after all downsamplings, before quantization)
        # curr_channels is now the channels of the deepest level (e.g., 4*base_channels)
        self.mid_block_encoder = nn.Sequential(
            ResnetBlock3D(curr_channels, curr_channels),
            AttentionBlock3D(curr_channels), # Attention in the middle block
            ResnetBlock3D(curr_channels, curr_channels)
        )
        self.encoder_norm = nn.GroupNorm(32, curr_channels) # GroupNorm before final encoder output
        self.encoder_out = CausalConv3d(curr_channels, curr_channels, kernel_size=3, padding=1) # Processes output of middle block for quantization

        # --- Quantization / De-quantization ---
        # quant_conv maps from the final encoder feature map (curr_channels) to latent_channels * 2 (mean, log_var)
        self.quant_conv = CausalConv3d(curr_channels, self.latent_channels * 2, kernel_size=1)
        # post_quant_conv maps from latent_channels to the channels expected by the decoder's middle block (curr_channels)
        self.post_quant_conv = CausalConv3d(self.latent_channels, curr_channels, kernel_size=1)

        # --- Decoder ---
        self.decoder = nn.ModuleList()
        
        # Middle block for decoder (symmetric to encoder's middle block)
        self.mid_block_decoder = nn.Sequential(
            ResnetBlock3D(curr_channels, curr_channels),
            AttentionBlock3D(curr_channels), # Attention in the middle block
            ResnetBlock3D(curr_channels, curr_channels)
        )
        
        # Upsampling blocks (reverse order of encoder's downsampling stages)
        # curr_channels remains from the deepest encoder stage
        for i in reversed(range(self.num_down_stages)):
            # Determine channels for this upsampling stage
            out_channels = self.base_channels * self.channel_mult[i] # Target channels after upsampling/resnet blocks
            
            decoder_block = nn.ModuleList()
            
            # Upsample first in the decoder stage, then apply resnet/attention
            if i < self.num_down_stages - 1: # Upsample in all but the final decoder stage
                # Upsample3D's `out_channels` is simply for the conv after interpolation.
                # It sets the channel count for the subsequent ResnetBlocks.
                decoder_block.append(Upsample3D(curr_channels, curr_channels))
            
            for _ in range(self.num_res_blocks):
                decoder_block.append(ResnetBlock3D(curr_channels, out_channels))
                curr_channels = out_channels # Update curr_channels after resnet blocks

            # Add attention at appropriate spatial resolutions for decoder (symmetric to encoder)
            if i == self.num_down_stages - 1 and self.attn_resolutions and self.attn_resolutions[0] == 16:
                decoder_block.append(AttentionBlock3D(curr_channels)) # Add attention at deepest decoder stage
            
            self.decoder.insert(0, nn.Sequential(*decoder_block)) # Insert at beginning to build in reverse order

        # Final layers for decoder (from base_channels back to input_channels)
        self.decoder_norm = nn.GroupNorm(32, curr_channels)
        self.decoder_out = CausalConv3d(curr_channels, self.input_channels, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encodes an input video into its latent representation (mean, log_var)
        and samples a latent code using the reparameterization trick.
        Input x: (batch_size, input_channels, T, H, W)
        Output latent_sample: (batch_size, latent_channels, T_latent, H_latent, W_latent)
        """
        h = x
        for module in self.encoder:
            h = module(h)

        h = self.mid_block_encoder(h)
        h = self.encoder_norm(h)
        h = F.silu(h)
        h = self.encoder_out(h)

        # Map to mean and log_var
        moments = self.quant_conv(h)
        mean, log_var = torch.chunk(moments, 2, dim=1) # Split channels into two halves
        
        if self.kl_regularization:
            latent_sample = self.sample_latent(mean, log_var)
        else:
            latent_sample = mean # If no KL regularization, just use mean as latent

        return mean, log_var, latent_sample

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decodes a latent representation back into a pixel-space video.
        Input latents: (batch_size, latent_channels, T_latent, H_latent, W_latent)
        Output video_recon: (batch_size, input_channels, T, H, W)
        """
        h = self.post_quant_conv(latents)
        
        h = self.mid_block_decoder(h)

        for module in self.decoder:
            h = module(h)
        
        h = self.decoder_norm(h)
        h = F.silu(h)
        video_recon = self.decoder_out(h)
        
        return video_recon

    def sample_latent(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Implements the reparameterization trick to sample from the latent distribution.
        """
        # Clamp log_var for numerical stability to prevent extremely large or small std values
        log_var = torch.clamp(log_var, -30.0, 20.0) 
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mean + std * epsilon


# Example Usage for testing
if __name__ == "__main__":
    # Create a dummy config for testing
    dummy_vae_config = VaeConfig(
        name="VideoVAE",
        compression_rate=[8, 8, 8], # 3 stages of 2x downsampling
        latent_channels=4,
        base_channels=128,
        num_res_blocks=2,
        attn_resolutions=[16], # Attention at 16x16 spatial resolution (deepest stage)
        use_causal_conv3d=True,
        kl_regularization=True
    )
    dummy_config_obj = Config()
    dummy_config_obj.model.vae = dummy_vae_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate VAE
    vae = VideoVAE(dummy_config_obj).to(device)
    vae.eval() # Set to evaluation mode for consistent behavior

    # Dummy input video (Batch, Channels, Frames, Height, Width)
    # Original resolution: 3x64x64
    input_video = torch.randn(1, 3, 16, 128, 128).to(device) # B, C, T, H, W
    input_image = torch.randn(1, 3, 128, 128).to(device) # B, C, H, W (for image-like behavior, if needed)

    print(f"Input video shape: {input_video.shape}")

    with torch.no_grad():
        mean, log_var, latent_sample = vae.encode(input_video)
        print(f"Encoded latent mean shape: {mean.shape}")
        print(f"Encoded latent log_var shape: {log_var.shape}")
        print(f"Sampled latent shape: {latent_sample.shape}")

        # Expected latent shape for 8x8x8 compression:
        # T_latent = 16 // 8 = 2
        # H_latent = 128 // 8 = 16
        # W_latent = 128 // 8 = 16
        # C_latent = 4 (from vae_config.latent_channels)
        assert latent_sample.shape == (1, 4, 2, 16, 16), f"Expected (1, 4, 2, 16, 16), got {latent_sample.shape}"
        
        reconstructed_video = vae.decode(latent_sample)
        print(f"Reconstructed video shape: {reconstructed_video.shape}")

        assert reconstructed_video.shape == input_video.shape, f"Expected {input_video.shape}, got {reconstructed_video.shape}"

    print("\nVideoVAE forward pass (encode -> decode) test successful!")

    # Test CausalConv3d padding logic
    print("\nTesting CausalConv3d padding logic...")
    causal_conv = CausalConv3d(1, 1, kernel_size=(3, 3, 3), padding=(1,1,1)).to(device) # Padding (t, h, w)
    test_tensor_t = 5
    test_tensor = torch.randn(1, 1, test_tensor_t, 10, 10).to(device)
    output_causal = causal_conv(test_tensor)
    # For a causal conv with kernel_size_t=3, padding_t=0 (internal to conv, handled by F.pad), stride_t=1
    # output_temporal_dim = input_temporal_dim + (2 * actual_padding_t) - (kernel_size_t - 1)*dilation_t
    # (after F.pad) = input_temporal_dim + (2 * 0) - kernel_size_t + 1 = input_temporal_dim - 2
    # But F.pad adds `kernel_size[0]-1` to front. So input to conv is `T + kernel_size[0]-1`.
    # conv (padding=0, stride=1) output will be `T_padded - kernel_size[0] + 1`
    # = `(T + kernel_size[0]-1) - kernel_size[0] + 1 = T`.
    print(f"CausalConv3d test - Input T: {test_tensor.shape[2]}, Output T: {output_causal.shape[2]}")
    assert output_causal.shape[2] == test_tensor.shape[2] # temporal dimension should be preserved with causal padding
    
    # Test Downsample3D and Upsample3D with dummy data
    print("\nTesting Downsample3D and Upsample3D...")
    downsample_layer = Downsample3D(3, 6).to(device)
    upsample_layer = Upsample3D(6, 3).to(device)
    
    small_video = torch.randn(1, 3, 4, 32, 32).to(device) # T=4, H=32, W=32
    down_output = downsample_layer(small_video) # Should be T=2, H=16, W=16
    print(f"Downsample3D - Input shape: {small_video.shape}, Output shape: {down_output.shape}")
    assert down_output.shape == (1, 6, 2, 16, 16), f"Expected (1, 6, 2, 16, 16), got {down_output.shape}"

    up_output = upsample_layer(down_output) # Should be T=4, H=32, W=32
    print(f"Upsample3D - Input shape: {down_output.shape}, Output shape: {up_output.shape}")
    assert up_output.shape == (1, 3, 4, 32, 32), f"Expected (1, 3, 4, 32, 32), got {up_output.shape}"
    print("Helper modules (CausalConv3d, Downsample3D, Upsample3D) tests successful!")

    print("\nAll VideoVAE and helper module tests completed.")

