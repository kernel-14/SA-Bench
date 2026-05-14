import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    """
    Hiera-based Image Encoder.

    This module is responsible for taking an input image and producing multi-scale
    feature maps, as well as a combined image embedding. It abstracts the Hiera
    backbone and a Feature Pyramid Network (FPN) for multiscale feature fusion.

    The paper mentions using an MAE pre-trained Hiera image encoder and a FPN
    to fuse stride 16 and 32 features for the main image embedding.
    Stride 4 and 8 features are provided as skip connections to the mask decoder.
    No relative positional encoding is used; instead, interpolated global positional
    embeddings are adopted.
    """
    def __init__(self,
                 img_size: int = 1024,
                 in_chans: int = 3,
                 patch_size: int = 16,
                 embed_dim: int = 768, # Corresponds to Hiera-B+
                 depth: int = 12, # Corresponds to Hiera-B+
                 num_heads: int = 12, # Corresponds to Hiera-B+
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 out_chans: int = 256, # Output channels for FPN
                 ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_chans = out_chans

        # Mock Hiera backbone - In a real implementation, this would be a full Hiera model.
        # We'll simulate its output by producing features at different strides.
        # Hiera provides features at different stages. Let's assume 4 stages for typical hierarchical backbones.
        # The paper mentions stride 4, 8, 16, and 32 features.
        # We'll represent these as mock convolutional layers for simplicity.

        # Stage 1: Stride 4 features
        self.stage1 = nn.Conv2d(in_chans, embed_dim // 4, kernel_size=4, stride=4)
        # Stage 2: Stride 8 features
        self.stage2 = nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=2, stride=2)
        # Stage 3: Stride 16 features
        self.stage3 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=2, stride=2)
        # Stage 4: Stride 32 features
        self.stage4 = nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=2, stride=2)


        # Mock FPN for fusing stride 16 and 32 features
        # Lateral connections
        self.lateral3 = nn.Conv2d(embed_dim, out_chans, kernel_size=1)
        self.lateral4 = nn.Conv2d(embed_dim * 2, out_chans, kernel_size=1)

        # Output convolutions
        self.output3 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1)
        self.output4 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1)

        # Convolutions to adjust channels for skip connections if needed
        self.skip_conv4 = nn.Conv2d(embed_dim // 4, out_chans, kernel_size=1)
        self.skip_conv8 = nn.Conv2d(embed_dim // 2, out_chans, kernel_size=1)


    def forward(self, x: torch.Tensor):
        # x: (B, C, H, W)

        # Mock Hiera feature extraction
        feat4 = self.stage1(x) # (B, embed_dim//4, H/4, W/4)
        feat8 = self.stage2(feat4) # (B, embed_dim//2, H/8, W/8)
        feat16 = self.stage3(feat8) # (B, embed_dim, H/16, W/16)
        feat32 = self.stage4(feat16) # (B, embed_dim*2, H/32, W/32)

        # FPN for image embedding (fusing stride 16 and 32 features)
        # Lateral connections
        lat3 = self.lateral3(feat16)
        lat4 = self.lateral4(feat32)

        # Upsample and add
        # We need to ensure that the feature maps have compatible spatial dimensions
        # for addition. F.interpolate will handle the upsampling.
        p4 = self.output4(lat4)
        p3 = self.output3(lat3 + F.interpolate(p4, size=lat3.shape[2:], mode='nearest'))

        # The image embedding for the mask decoder is typically from the FPN's highest resolution output
        # or a combination. The paper states "fuse the stride 16 and 32 features ... to produce the image embeddings".
        # Let's consider p3 as the primary image embedding, and p4 (upsampled) as part of it implicitly.
        # The paper says "multiscale features during decoding", so we might pass all P-levels or specific ones.
        # For SAM2, the main frame embedding from the image encoder is passed to memory attention.
        # Let's assume the highest resolution FPN output (p3) is the primary image embedding.
        image_embedding = p3 # (B, out_chans, H/16, W/16)

        # Skip connections for mask decoder
        # These are directly from Hiera stages, potentially adjusted for channel dimension
        skip_feat4 = self.skip_conv4(feat4) # (B, out_chans, H/4, W/4)
        skip_feat8 = self.skip_conv8(feat8) # (B, out_chans, H/8, W/8)

        return image_embedding, skip_feat4, skip_feat8

# Example usage (for testing the module structure)
if __name__ == "__main__":
    # Create a dummy input image
    batch_size = 1
    image_size = 1024
    dummy_image = torch.randn(batch_size, 3, image_size, image_size)

    # Initialize the image encoder
    image_encoder = ImageEncoder(img_size=image_size)

    # Forward pass
    image_embedding, skip_feat4, skip_feat8 = image_encoder(dummy_image)

    print(f"Input image shape: {dummy_image.shape}")
    print(f"Image embedding shape (stride 16 equivalent): {image_embedding.shape}")
    print(f"Skip feature shape (stride 4): {skip_feat4.shape}")
    print(f"Skip feature shape (stride 8): {skip_feat8.shape}")

    # Verify strides
    expected_h_16 = image_size // 16
    expected_w_16 = image_size // 16
    assert image_embedding.shape[2] == expected_h_16 and image_embedding.shape[3] == expected_w_16
    
    expected_h_4 = image_size // 4
    expected_w_4 = image_size // 4
    assert skip_feat4.shape[2] == expected_h_4 and skip_feat4.shape[3] == expected_w_4

    expected_h_8 = image_size // 8
    expected_w_8 = image_size // 8
    assert skip_feat8.shape[2] == expected_h_8 and skip_feat8.shape[3] == expected_w_8

    print("ImageEncoder outputs match expected shapes and strides.")
