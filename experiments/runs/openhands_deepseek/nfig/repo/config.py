"""
NFIG: Next-Frequency Image Generation Configuration
Based on "Multi-Scale Autoregressive Image Generation via Frequency Ordering"
"""

class FRVAEConfig:
    # Architecture
    image_size: int = 256
    latent_channels: int = 256
    downsampling_factor: int = 16  # 256/16 = 16
    feature_map_size: int = 16  # H' = W' = 16
    codebook_size: int = 4096
    codebook_dim: int = 32
    num_frequency_bands: int = 10

    # Scale factors for each frequency band (matching paper)
    # Each s_i defines the resolution h_i = s_i, w_i = s_i for band i
    scale_factors: tuple = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    total_tokens: int = 680  # sum of s_i^2

    # VQ-GAN components
    discriminator_type: str = "dino"
    perceptual_loss_weight: float = 1.0
    gan_loss_weight: float = 0.5
    commitment_loss_weight: float = 0.25
    quantizer_type: str = "residual_frequency"

    # Training
    vae_batch_size: int = 64
    vae_learning_rate: float = 1e-4
    vae_adam_betas: tuple = (0.5, 0.9)
    vae_training_steps: int = 500000

    # Optimizer
    use_ema: bool = True
    ema_decay: float = 0.999


class NFIGTransformerConfig:
    # Architecture
    hidden_dim: int = 1024
    num_heads: int = 16
    num_layers: int = 16
    dropout: float = 0.1
    vocab_size: int = 4096
    max_sequence_length: int = 680

    # Scale factors and token distribution
    scale_factors: tuple = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    num_frequency_bands: int = 10
    total_tokens: int = 680
    feature_map_size: int = 16

    # Conditioning
    num_classes: int = 1000
    class_dropout_prob: float = 0.1
    use_adaln: bool = True

    # Training
    batch_size: int = 768
    learning_rate: float = 8e-5
    adam_betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.01
    max_epochs: int = 350
    warmup_steps: int = 10000
    gradient_clip: float = 1.0

    # Diffusion/guidance
    cfg_prob: float = 0.1  # probability of dropping class for CFG

    # Inference
    cfg_scale: float = 4.5
    top_k: int = 990
    temperature: float = 1.0


class DataConfig:
    dataset: str = "imagenet"
    image_size: int = 256
    random_crop: bool = True
    random_flip: bool = True
    num_workers: int = 8
    pin_memory: bool = True
    data_path: str = "/datasets/ImageNet"


class ExperimentConfig:
    fr_vae: FRVAEConfig = FRVAEConfig()
    transformer: NFIGTransformerConfig = NFIGTransformerConfig()
    data: DataConfig = DataConfig()

    # Mixed precision
    use_amp: bool = True
    compile_model: bool = False

    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000

    # Paths
    output_dir: str = "./checkpoints"
    vae_checkpoint: str = "./checkpoints/fr_vae_best.pt"

    # Evaluation
    eval_batch_size: int = 64
    num_eval_samples: int = 50000
    fid_use_torch: bool = True
