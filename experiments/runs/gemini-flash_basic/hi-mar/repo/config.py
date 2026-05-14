import dataclasses

@dataclasses.dataclass
class HiMARConfig:
    img_size_low: int = 128
    img_size_high: int = 256
    patch_size: int = 8 # VAE downsampling factor, e.g., 8 for 128x128 -> 16x16 latent
    in_chans_vae_latent: int = 4 # Channels in VAE latent space (e.g., 4 from LDM/Stable Diffusion VAE)
    embed_dim: int = 768
    depth: int = 24
    num_heads: int = 12 # Common for transformers, derived from hidden_size / 64 for DIT, so 768/12 = 64
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_scale: float = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    num_classes: int = 1000 # For ImageNet
    context_dim: int = None # For text-to-image (e.g., CLIP text embedding dimension)
    diff_head1_out_dim: int = 4 # Predicts noise in VAE latent space
    diff_head2_num_layers: int = 6

# Configurations for Hi-MAR variants based on Table 1
HiMAR_B_Config = HiMARConfig(
    embed_dim=768,
    depth=24,
    num_heads=12, # 768 / 64 = 12
    diff_head2_num_layers=6,
)

HiMAR_L_Config = HiMARConfig(
    embed_dim=1024,
    depth=32,
    num_heads=16, # 1024 / 64 = 16
    diff_head2_num_layers=8,
)

HiMAR_H_Config = HiMARConfig(
    embed_dim=1280,
    depth=40,
    num_heads=20, # 1280 / 64 = 20
    diff_head2_num_layers=12,
)

# For Hi-MAR-S (Small) as mentioned in MS-COCO results, let's derive one
# Paper mentions "light-weight version of our Hi-MAR with comparable model size" to U-ViT-S/2
# U-ViT-S/2 often has smaller dims and layers.
HiMAR_S_Config = HiMARConfig(
    img_size_low=128, # Assuming same for now
    img_size_high=256, # Assuming same for now
    patch_size=8, 
    in_chans_vae_latent=4, 
    embed_dim=384, # Example smaller dim
    depth=12, # Example smaller depth
    num_heads=6,  # 384 / 64 = 6
    mlp_ratio=4.0,
    diff_head2_num_layers=4, # Example smaller layers
)

END_OF_PYTHON_CODE'
