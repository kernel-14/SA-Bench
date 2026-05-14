
import argparse

def get_config():
    parser = argparse.ArgumentParser(description="NFIG: Multi-Scale Autoregressive Image Generation via Frequency Ordering")

    # Dataset
    parser.add_argument('--dataset', type=str, default='imagenet', help='Dataset to use (e.g., imagenet)')
    parser.add_argument('--image_size', type=int, default=256, help='Image size for training')
    parser.add_argument('--data_path', type=str, default='./data/imagenet', help='Path to dataset')

    # Dataset
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of classes in the dataset (e.g., 1000 for ImageNet)')

    # FR-VAE (Image Tokenizer)
    parser.add_argument('--fr_vae_vqgan_config', type=str, default='xqgan', help='VQGAN architecture config (adopted from XQGAN)')
    parser.add_argument('--fr_vae_encoder_pretrained', type=str, default='dinov2-base', help='Pretrained weights for image encoder (DINOv2-base)')
    parser.add_argument('--fr_vae_codebook_size', type=int, default=4096, help='Codebook size for FR-VAE')
    parser.add_argument('--fr_vae_embedding_dim', type=int, default=256, help='Embedding dimension for codebook vectors') # Assuming from typical VQ-GANs
    parser.add_argument('--fr_vae_n_residual_layers', type=int, default=8, help='Number of residual layers in FR-VAE') # Placeholder, need to find in paper or common VQGAN

    # Frequency Residual Quantizer
    parser.add_argument('--frequency_scaling_factors', type=list, default=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16],
                        help='Multiple scaling factors across different frequency bands')
    parser.add_argument('--vocabulary_size', type=int, default=680, help='Total vocabulary size for frequency tokens')

    # NFIG Transformer (Image Generator)
    parser.add_argument('--transformer_backbone', type=str, default='var_transformer', help='Transformer backbone (VAR Transformer)')
    parser.add_argument('--transformer_depth', type=int, default=16, help='Depth of VAR Transformer')
    parser.add_argument('--transformer_heads', type=int, default=8, help='Number of attention heads in Transformer') # Placeholder
    parser.add_argument('--transformer_dim', type=int, default=512, help='Dimension of Transformer embeddings') # Placeholder

    # Training
    parser.add_argument('--epochs', type=int, default=350, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=8e-5, help='Learning rate for Adam optimizer')
    parser.add_argument('--batch_size', type=int, default=768, help='Batch size for training')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loader workers')

    # Inference
    parser.add_argument('--cfg_scale', type=float, default=4.5, help='Classifier Free Guidance scale')
    parser.add_argument('--top_k', type=int, default=990, help='Top-k sampling for inference')
    parser.add_argument('--num_inference_steps', type=int, default=10, help='Number of inference steps (for next-frequency prediction)')

    # Logging and Checkpointing
    parser.add_argument('--log_interval', type=int, default=100, help='Steps between logging training metrics')
    parser.add_argument('--save_interval', type=int, default=10, help='Epochs between saving model checkpoints')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory for logs and checkpoints')

    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (cuda or cpu)')

    return parser.parse_args()

if __name__ == '__main__':
    config = get_config()
    print("NFIG Configuration:")
    for arg in vars(config):
        print(f"  {arg}: {getattr(config, arg)}")
