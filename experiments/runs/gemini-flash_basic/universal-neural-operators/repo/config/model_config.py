# This file will contain configuration dictionaries for different models and training scenarios.
# These are illustrative values and would typically be tuned during actual experimentation.

MODEL_CONFIGS = {
    "FNO_1D": {
        "core_model_type": "FNO",
        "shared_model_params": {
            "modes": 12,
            "width": 64,
            "num_fno_layers": 4
        },
        "lifting_params": {
            "in_channels": None, # Will be set dynamically based on dataset
            "out_channels": 64
        },
        "projection_params": {
            "in_channels": 64,
            "out_channels": None # Will be set dynamically based on dataset
        },
        "data_dim": 1
    },
    "MambaFNO_1D": {
        "core_model_type": "MambaFNO",
        "shared_model_params": {
            "modes": 12,
            "width": 64,
            "num_fno_layers": 4,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2
        },
        "lifting_params": {
            "in_channels": None,
            "out_channels": 64
        },
        "projection_params": {
            "in_channels": 64,
            "out_channels": None
        },
        "data_dim": 1
    },
    "PerceiverFNO_1D": {
        "core_model_type": "PerceiverFNO",
        "shared_model_params": {
            "modes": 12,
            "width": 64,
            "num_fno_layers": 4,
            "latent_dim": 64,
            "num_latent_tokens": 16,
            "num_heads": 8,
            "mlp_ratio": 4.
        },
        "lifting_params": {
            "in_channels": None,
            "out_channels": 64
        },
        "projection_params": {
            "in_channels": 64,
            "out_channels": None
        },
        "data_dim": 1
    },
    "FNO_2D": {
        "core_model_type": "FNO",
        "shared_model_params": {
            "modes1": 12,
            "modes2": 12,
            "width": 64,
            "num_fno_layers": 4
        },
        "lifting_params": {
            "in_channels": None, 
            "out_channels": 64
        },
        "projection_params": {
            "in_channels": 64,
            "out_channels": None 
        },
        "data_dim": 2
    },
    "MambaFNO_2D": {
        "core_model_type": "MambaFNO",
        "shared_model_params": {
            "modes1": 12,
            "modes2": 12,
            "width": 64,
            "num_fno_layers": 4,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2
        },
        "lifting_params": {
            "in_channels": None,
            "out_channels": 64
        },
        "projection_params": {
            "in_channels": 64,
            "out_channels": None
        },
        "data_dim": 2
    },
    "PerceiverFNO_2D": {
        "core_model_type": "PerceiverFNO",
        "shared_model_params": {
            "modes1": 12,
            "modes2": 12,
            "width": 64,
            "num_fno_layers": 4,
            "latent_dim": 64,
            "num_latent_tokens": 16,
            "num_heads": 8,
            "mlp_ratio": 4.
        },
        "lifting_params": {
            "in_channels": None,
            "out_channels": 64
        },
        "projection_params": {
            "in_channels": 64,
            "out_channels": None
        },
        "data_dim": 2
    },
}

TRAINING_CONFIGS = {
    "pretrain": {
        "epochs": 100,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "loss_fn": "mse",
        "optimizer": "adam",
        "save_interval": 10,
        "eval_interval": 5
    },
    "finetune": {
        "epochs": 50,
        "learning_rate": 1e-4,
        "batch_size": 16,
        "loss_fn": "mse",
        "optimizer": "adam",
        "save_interval": 5,
        "eval_interval": 2
    }
}
