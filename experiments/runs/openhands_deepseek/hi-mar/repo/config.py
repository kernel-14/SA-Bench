"""Configuration for Hi-MAR model variants and training hyperparameters."""

# Model architecture configs (Table 1 from the paper)
MODEL_CONFIGS = {
    'Hi-MAR-B': {
        'token_dim': 16,
        'low_res_num_tokens': 256,
        'high_res_num_tokens': 1024,
        'transformer_depth': 24,
        'transformer_num_heads': 12,
        'transformer_dim': 768,
        'mlp_ratio': 4.0,
        'dropout': 0.0,
        'head1_hidden_dim': 1024,
        'head1_depth': 6,
        'head2_hidden_dim': 512,
        'head2_depth': 6,
        'head2_num_heads': 8,
        'num_classes': 1000,
        'num_train_timesteps': 1000,
        'beta_start': 1e-4,
        'beta_end': 0.02,
    },
    'Hi-MAR-L': {
        'token_dim': 16,
        'low_res_num_tokens': 256,
        'high_res_num_tokens': 1024,
        'transformer_depth': 32,
        'transformer_num_heads': 16,
        'transformer_dim': 1024,
        'mlp_ratio': 4.0,
        'dropout': 0.0,
        'head1_hidden_dim': 1280,
        'head1_depth': 8,
        'head2_hidden_dim': 512,
        'head2_depth': 8,
        'head2_num_heads': 8,
        'num_classes': 1000,
        'num_train_timesteps': 1000,
        'beta_start': 1e-4,
        'beta_end': 0.02,
    },
    'Hi-MAR-H': {
        'token_dim': 16,
        'low_res_num_tokens': 256,
        'high_res_num_tokens': 1024,
        'transformer_depth': 40,
        'transformer_num_heads': 16,
        'transformer_dim': 1280,
        'mlp_ratio': 4.0,
        'dropout': 0.0,
        'head1_hidden_dim': 1536,
        'head1_depth': 12,
        'head2_hidden_dim': 768,
        'head2_depth': 12,
        'head2_num_heads': 12,
        'num_classes': 1000,
        'num_train_timesteps': 1000,
        'beta_start': 1e-4,
        'beta_end': 0.02,
    },
}

# Training hyperparameters for ImageNet (Section 4.2)
IMAGENET_TRAIN_CONFIG = {
    'lr': 1e-4,
    'beta1': 0.9,
    'beta2': 0.95,
    'weight_decay': 0.02,
    'warmup_epochs': 100,
    'total_epochs': 800,
    'p1_mask_min_ratio': 0.7,
    'p1_mask_max_ratio': 1.0,
    'p2_mask_schedule': 'cosine',  # MaskGIT-style
    'ema_decay': 0.9999,
    'grad_clip': 1.0,
    'inference_p1_steps': 32,
    'inference_p2_steps': 4,
}

# Training hyperparameters for MS-COCO (Section 4.2)
COCO_TRAIN_CONFIG = {
    'lr': 8e-4,
    'beta1': 0.9,
    'beta2': 0.95,
    'weight_decay': 0.03,
    'warmup_steps': 8000,
    'p1_mask_min_ratio': 0.7,
    'p1_mask_max_ratio': 1.0,
    'p2_mask_schedule': 'beta',  # Beta(4, 1)
    'p2_beta_alpha': 4.0,
    'p2_beta_beta': 1.0,
    'ema_decay': 0.9999,
    'grad_clip': 1.0,
    'inference_p1_steps': 32,
    'inference_p2_steps': 4,
}

# VAE latent encoding
VAE_CONFIG = {
    'kl_version': 'KL-16',
    'latent_channels': 16,
    'scaling_factor': 0.18215,
}

# Image sizes
IMAGE_SIZES = {
    'high_res': 256,
    'low_res': 128,
    'high_res_latent_size': 32,   # 256 / 8
    'low_res_latent_size': 16,    # 128 / 8
}
