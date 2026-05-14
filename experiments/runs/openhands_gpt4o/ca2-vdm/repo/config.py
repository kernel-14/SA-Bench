# Configuration file for Ca2-VDM

CONFIG = {
    # Model parameters
    "model": {
        "latent_dim": 256,
        "num_layers": 12,
        "num_heads": 8,
        "dropout": 0.1,
        "causal_attention": True,
        "prefix_enhanced_attention": True,
        "max_condition_length": 49,  # P_max
        "chunk_length": 16,  # l
    },

    # Training parameters
    "training": {
        "batch_size": 144,
        "learning_rate": 2e-5,
        "num_steps": 21000,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "scheduler": "linear",
        "gradient_clipping": 1.0,
    },

    # Dataset parameters
    "dataset": {
        "name": "SkyTimelapse",
        "resolution": (256, 256),
        "train_split": "train",
        "test_split": "test",
        "num_workers": 4,
    },

    # Diffusion parameters
    "diffusion": {
        "num_timesteps": 1000,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "schedule": "linear",
    },

    # Evaluation parameters
    "evaluation": {
        "metrics": ["FVD"],
        "num_samples": 512,
    },

    # Logging and checkpointing
    "logging": {
        "log_dir": "logs/",
        "checkpoint_dir": "checkpoints/",
        "save_interval": 1000,
    },
}