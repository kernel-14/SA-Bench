
import argparse

class P2VAEConfig:
    """Configuration for P2VAE (Pretrained Physics Variational Autoencoder)."""
    IMAGE_SIZE = 128
    IN_CHANNELS = 3
    OUT_CHANNELS = 3 # For reconstruction
    LATENT_CHANNELS = 16 # c16p16
    COMPRESSION_RATE = 12 # From c3p128 to c16p16
    BETA_KL_LOSS = 1e-3
    BASE_DIM_16M = 64
    BASE_DIM_87M = 128

    # Training
    LEARNING_RATE = 1e-4
    BETAS = (0.9, 0.995)
    WEIGHT_DECAY = 1e-4
    TRAINING_STEPS = 100_000
    BATCH_SIZE = 256 # Base batch size, LR is adjusted accordingly


class FMTConfig:
    """Configuration for FMT (Flow Marching Transformer)."""
    # Architecture
    HEAD_DIM = 64
    # Embedding dimensions for different model sizes
    EMBED_DIM_SMALL = 256 # FMT-S-6M
    EMBED_DIM_BASE = 512  # FMT-B-42M
    EMBED_DIM_LARGE = 768 # FMT-L-138M

    # Training
    LEARNING_RATE = 1e-4
    BETAS = (0.9, 0.95)
    WEIGHT_DECAY = 0.01
    TRAINING_STEPS = 100_000
    BATCH_SIZE = 256 # Base batch size, LR is adjusted accordingly

    # Flow Marching
    T_UNIFORM_MIN = 0.0
    T_UNIFORM_MAX = 1.0
    K_UNIFORM_MIN = 0.0
    K_UNIFORM_MAX = 1.0

    # Prediction & Generation
    EULER_N_DISCRETIZATION = 100
    EULER_DT = 0.01
    K_DETERMINISTIC = 1.0
    K_GENERATIVE_DEFAULT = 0.5 # Example value, can be varied


class DataConfig:
    """Configuration for dataset loading and preprocessing."""
    DATASET_PATH = "./data" # Placeholder
    TRAJECTORY_LENGTH = 4
    SPATIAL_RESOLUTION = 128
    NUM_CHANNELS = 3
    PRECISION = "float16" # Data stored in float16

    # Dataset splits
    TRAIN_RATIO = 0.8
    VALID_RATIO = 0.1
    TEST_RATIO = 0.1

    # Number of distinct PDE families
    NUM_PDE_FAMILIES = 12


class TrainingConfig:
    """General training configurations."""
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" # Default, can be changed based on availability
    LOG_INTERVAL = 100
    EVAL_INTERVAL = 1000
    SAVE_INTERVAL = 5000


def parse_args():
    parser = argparse.ArgumentParser(description="Neural Operator and Flow Matching for Generative PDE Foundation Model")

    # General
    parser.add_argument("--device", type=str, default=TrainingConfig.DEVICE,
                        help="Device to use for training (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=TrainingConfig.SEED,
                        help="Random seed for reproducibility")

    # P2VAE
    parser.add_argument("--p2vae_model_size", type=str, default="16M", choices=["16M", "87M"],
                        help="Size of the P2VAE model (16M or 87M)")
    parser.add_argument("--p2vae_lr", type=float, default=P2VAEConfig.LEARNING_RATE,
                        help="Learning rate for P2VAE training")
    parser.add_argument("--p2vae_batch_size", type=int, default=P2VAEConfig.BATCH_SIZE,
                        help="Batch size for P2VAE training")
    parser.add_argument("--p2vae_training_steps", type=int, default=P2VAEConfig.TRAINING_STEPS,
                        help="Number of training steps for P2VAE")
    parser.add_argument("--beta_kl_loss", type=float, default=P2VAEConfig.BETA_KL_LOSS,
                        help="Weight for the KL divergence term in P2VAE loss")

    # FMT
    parser.add_argument("--fmt_model_size", type=str, default="Small", choices=["Small", "Base", "Large"],
                        help="Size of the FMT model (Small, Base, or Large)")
    parser.add_argument("--fmt_lr", type=float, default=FMTConfig.LEARNING_RATE,
                        help="Learning rate for FMT training")
    parser.add_argument("--fmt_batch_size", type=int, default=FMTConfig.BATCH_SIZE,
                        help="Batch size for FMT training")
    parser.add_argument("--fmt_training_steps", type=int, default=FMTConfig.TRAINING_STEPS,
                        help="Number of training steps for FMT")

    # Data
    parser.add_argument("--dataset_path", type=str, default=DataConfig.DATASET_PATH,
                        help="Path to the dataset directory")

    return parser.parse_args()

