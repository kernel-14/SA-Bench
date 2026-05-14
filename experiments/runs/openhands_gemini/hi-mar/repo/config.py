import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Hi-MAR Configuration")

    # General
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory.")
    parser.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet", "mscoco"], help="Dataset to use.")
    parser.add_argument("--image_size", type=int, default=256, help="Image size for high resolution.")
    parser.add_argument("--low_res_image_size", type=int, default=128, help="Image size for low resolution.")

    # Model Architecture
    parser.add_argument("--model_type", type=str, default="Hi-MAR-B", choices=["Hi-MAR-B", "Hi-MAR-L", "Hi-MAR-H"], help="Hi-MAR model variant.")
    parser.add_argument("--vae_path", type=str, default="stabilityai/sd-vae-ft-mse", help="Path to pre-trained VAE model.")
    parser.add_argument("--patch_size", type=int, default=8, help="Patch size for VAE.") # Assuming common VAE setup, adjust if paper specifies.
    parser.add_argument("--in_channels", type=int, default=3, help="Input image channels.")
    parser.add_argument("--out_channels", type=int, default=3, help="Output image channels (for diffusion head prediction).")
    parser.add_argument("--num_classes", type=int, default=1000, help="Number of classes for class-conditional generation (ImageNet).")
    parser.add_argument("--conditioning_dim", type=int, default=768, help="Dimension of conditioning tokens (e.g., text embeddings).")

    # Hi-MAR Transformer specific
    parser.add_argument("--transformer_layers_B", type=int, default=24, help="Number of transformer layers for Hi-MAR-B.")
    parser.add_argument("--transformer_hidden_size_B", type=int, default=768, help="Hidden size for Hi-MAR-B transformer.")
    parser.add_argument("--transformer_layers_L", type=int, default=32, help="Number of transformer layers for Hi-MAR-L.")
    parser.add_argument("--transformer_hidden_size_L", type=int, default=1024, help="Hidden size for Hi-MAR-L transformer.")
    parser.add_argument("--transformer_layers_H", type=int, default=40, help="Number of transformer layers for Hi-MAR-H.")
    parser.add_argument("--transformer_hidden_size_H", type=int, default=1280, help="Hidden size for Hi-MAR-H transformer.")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads.") # Common default, adjust if paper specifies.

    # Diffusion Head 1 (low-res) specific
    parser.add_argument("--diff_head1_layers_B", type=int, default=6, help="Number of layers for Diffusion Head 1 (low-res) for Hi-MAR-B.")
    parser.add_argument("--diff_head1_hidden_size_B", type=int, default=1024, help="Hidden size for Diffusion Head 1 (low-res) for Hi-MAR-B.")
    parser.add_argument("--diff_head1_layers_L", type=int, default=8, help="Number of layers for Diffusion Head 1 (low-res) for Hi-MAR-L.")
    parser.add_argument("--diff_head1_hidden_size_L", type=int, default=1280, help="Hidden size for Diffusion Head 1 (low-res) for Hi-MAR-L.")
    parser.add_argument("--diff_head1_layers_H", type=int, default=12, help="Number of layers for Diffusion Head 1 (low-res) for Hi-MAR-H.")
    parser.add_argument("--diff_head1_hidden_size_H", type=int, default=1536, help="Hidden size for Diffusion Head 1 (low-res) for Hi-MAR-H.")

    # Diffusion Head 2 (high-res) specific
    parser.add_argument("--diff_head2_layers_B", type=int, default=6, help="Number of layers for Diffusion Head 2 (high-res) for Hi-MAR-B.")
    parser.add_argument("--diff_head2_hidden_size_B", type=int, default=512, help="Hidden size for Diffusion Head 2 (high-res) for Hi-MAR-B.")
    parser.add_argument("--diff_head2_layers_L", type=int, default=8, help="Number of layers for Diffusion Head 2 (high-res) for Hi-MAR-L.")
    parser.add_argument("--diff_head2_hidden_size_L", type=int, default=512, help="Hidden size for Diffusion Head 2 (high-res) for Hi-MAR-L.")
    parser.add_argument("--diff_head2_layers_H", type=int, default=12, help="Number of layers for Diffusion Head 2 (high-res) for Hi-MAR-H.")
    parser.add_argument("--diff_head2_hidden_size_H", type=int, default=768, help="Hidden size for Diffusion Head 2 (high-res) for Hi-MAR-H.")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=800, help="Number of training epochs for ImageNet.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for ImageNet.")
    parser.add_argument("--lr_t2i", type=float, default=8e-4, help="Learning rate for text-to-image (MS-COCO).")
    parser.add_argument("--weight_decay", type=float, default=0.02, help="Weight decay for ImageNet.")
    parser.add_argument("--weight_decay_t2i", type=float, default=0.03, help="Weight decay for text-to-image (MS-COCO).")
    parser.add_argument("--warmup_epochs", type=int, default=100, help="Linear warmup epochs for ImageNet.")
    parser.add_argument("--warmup_steps_t2i", type=int, default=8000, help="Linear warmup steps for text-to-image (MS-COCO).")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.") # Assuming 128 based on speed/accuracy trade-off fig.
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer to use.")
    parser.add_argument("--ema_momentum", type=float, default=0.9999, help="EMA momentum for text-to-image.")

    # Masking strategies
    parser.add_argument("--masking_ratio_phase1_min", type=float, default=0.7, help="Min masking ratio for phase 1 (ImageNet).")
    parser.add_argument("--masking_ratio_phase1_max", type=float, default=1.0, help="Max masking ratio for phase 1 (ImageNet).")
    parser.add_argument("--masking_strategy_phase2", type=str, default="cosine", help="Masking strategy for phase 2 (ImageNet).")
    parser.add_argument("--beta_dist_alpha", type=float, default=4.0, help="Alpha parameter for Beta distribution masking (MS-COCO).")
    parser.add_argument("--beta_dist_beta", type=float, default=1.0, help="Beta parameter for Beta distribution masking (MS-COCO).")

    # Inference parameters
    parser.add_argument("--inference_steps_phase1", type=int, default=32, help="Inference steps for phase 1.")
    parser.add_argument("--inference_steps_phase2", type=int, default=4, help="Inference steps for phase 2.")
    parser.add_argument("--cfg_guidance_scale", type=float, default=None, help="Classifier-free guidance scale. None if not used.")
    parser.add_argument("--save_interval", type=int, default=50, help="Save model checkpoint every N epochs.")

    # CLIP for text-to-image
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-large-patch14", help="CLIP model for text encoding.")

    args = parser.parse_args()
    return args

def get_model_config(model_type):
    config = {}
    if model_type == "Hi-MAR-B":
        config["transformer_layers"] = 24
        config["transformer_hidden_size"] = 768
        config["diff_head1_layers"] = 6
        config["diff_head1_hidden_size"] = 1024
        config["diff_head2_layers"] = 6
        config["diff_head2_hidden_size"] = 512
    elif model_type == "Hi-MAR-L":
        config["transformer_layers"] = 32
        config["transformer_hidden_size"] = 1024
        config["diff_head1_layers"] = 8
        config["diff_head1_hidden_size"] = 1280
        config["diff_head2_layers"] = 8
        config["diff_head2_hidden_size"] = 512
    elif model_type == "Hi-MAR-H":
        config["transformer_layers"] = 40
        config["transformer_hidden_size"] = 1280
        config["diff_head1_layers"] = 12
        config["diff_head1_hidden_size"] = 1536
        config["diff_head2_layers"] = 12
        config["diff_head2_hidden_size"] = 768
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return config