"""
Configuration for nGPT experiments.

Contains model configurations, optimization parameters, and hyperparameters
as described in the paper (Tables 2 and 3, Appendix A.6).
"""

import math

# Model configurations matching the paper
MODEL_CONFIGS = {
    '0.5B': {
        'n_layers': 24,
        'd_model': 1024,
        'n_heads': 16,
        'd_k': 64,          # d_model / n_heads = 1024/16
        'd_mlp': 4096,      # 4 * d_model
        'gpt_params': 468.2,  # Millions
        'ngpt_params': 468.4,
    },
    '1B': {
        'n_layers': 36,
        'd_model': 1280,
        'n_heads': 20,
        'd_k': 64,          # d_model / n_heads = 1280/20
        'd_mlp': 5120,      # 4 * d_model
        'gpt_params': 1025.7,
        'ngpt_params': 1026.1,
    },
}

# Optimization parameters (Table 3)
OPTIM_CONFIG = {
    'gpt': {
        'optimizer': 'AdamW',
        'weight_decay': 0.1,
        'warmup_steps': 2000,
        'lr_schedule': 'cosine_annealing',
        'final_lr': 0.0,
        'betas': (0.9, 0.95),
        'eps': 1e-8,
    },
    'ngpt': {
        'optimizer': 'Adam',  # AdamW with weight_decay=0.0
        'weight_decay': 0.0,
        'warmup_steps': 0,
        'lr_schedule': 'cosine_annealing',
        'final_lr': 0.0,
        'betas': (0.9, 0.95),
        'eps': 1e-8,
    },
}

# Training setup (Appendix A.6)
TRAINING_CONFIG = {
    'global_batch_size': 512,
    'num_gpus': 64,         # 8 nodes x 8 GPUs (A100)
    'dataset': 'OpenWebText',
    'tokenizer': 'LLaMA-2',  # 32k tokens
    'vocab_size': 32000,
    'dtype': 'bfloat16',
    'rope_base': 10000,
}

# nGPT-specific initialization parameters (Section 2.6)
NGPT_INIT_CONFIG = {
    # Eigen learning rates
    'alpha_A_init': 0.05,           # ~1/n_layers order
    'alpha_M_init': 0.05,
    'alpha_scale': '1/sqrt(d_model)',  # Computed per model

    # QK scaling
    's_qk_init': 1.0,
    's_qk_scale': '1/sqrt(d_model)',

    # MLP scaling
    's_u_init': 1.0,
    's_u_scale': 1.0,
    's_v_init': 1.0,
    's_v_scale': 1.0,

    # Logit scaling
    's_z_init': 1.0,
    's_z_scale': '1/sqrt(d_model)',
}

# GPT initialization (Appendix A.6)
GPT_INIT_CONFIG = {
    'init_std': 0.02,
    'output_std_scale': 'sqrt(2 * n_layers)',  # As per Radford et al. (2018)
}

# Initialization for nGPT (Appendix A.6)
NGPT_INIT_STD = '1/sqrt(d_model)'  # Matrix parameters initialized with this std

# Ablation study configurations (Appendix A.9)
ABLATION_CONFIGS = {
    'baseline': {
        's_qk_init': 1.0,
        's_qk_scale': '1/sqrt(d_model)',
        's_u_init': 1.0,
        's_u_scale': 1.0,
        's_v_init': 1.0,
        's_v_scale': 1.0,
        's_z_init': 1.0,
        's_z_scale': '1/sqrt(d_model)',
    },
    'no_qk_norm': {
        'use_qk_norm': False,
    },
    'slerp': {
        'use_slerp': True,  # Use SLERP instead of LERP
    },
    'fixed_s_qk': {
        's_qk_learnable': False,
        's_qk_fixed_value': 1.0,
    },
    'scalar_alpha': {
        'alpha_per_dim': False,  # Use scalar instead of per-dimension
    },
}


def get_alpha_scale(d_model):
    """Get the alpha scale value for a given model dimension."""
    return 1.0 / math.sqrt(d_model)


def get_s_scale(d_model, param_name):
    """Get the scale value for a given scaling parameter."""
    if param_name in ['s_qk', 's_z', 'alpha']:
        return 1.0 / math.sqrt(d_model)
    elif param_name in ['s_u', 's_v']:
        return 1.0
    else:
        return 1.0 / math.sqrt(d_model)


def get_init_std_ngpt(d_model):
    """Get initialization std for nGPT matrix parameters."""
    return 1.0 / math.sqrt(d_model)


def get_init_std_gpt():
    """Get initialization std for GPT matrix parameters."""
    return 0.02
