import yaml
import os
from copy import deepcopy

DEFAULT_CONFIG = {
    'wavelet': {
        'type_1d': 'bior2.4',
        'type_2d': 'bior1.3',
        'mode_1d': 'periodization',
        'mode_2d': 'zero',
    },
    'diffusion': {
        'num_timesteps': 1000,
        'beta_start': 1.0e-4,
        'beta_end': 0.02,
        'schedule': 'linear',
    },
    'ddim': {
        'sampling_steps': 50,
        'eta': 1.0,
    },
    'unet_1d': {
        'init_dim': 128,
        'down_up_layers': 4,
        'kernel_size': 3,
        'dim_mult_phi': [1, 2, 4, 8],
        'dim_mult_theta': [1, 2, 4, 8],
        'resnet_groups': 8,
        'attn_hidden_dim': 32,
        'attn_heads': 4,
    },
    'unet_3d': {
        'init_dim': 100,
        'kernel_size': [3, 3, 3],
        'kernel_padding': [1, 1, 1],
        'attn_heads': 4,
    },
    'multires': {
        'num_levels': 3,
        'sr_steps': [1, 2, 3],
    },
    'training': {
        'batch_size': 16,
        'optimizer': 'adam',
        'learning_rate': 1.0e-4,
        'training_steps': 190000,
        'lr_scheduler': 'cosine',
    },
    'control': {
        'guidance_weight': 120000,
        'guidance_scheduler': 'cosine',
    },
    'data': {
        'burgers_dir': './data/burgers/',
        'advection_dir': './data/advection/',
        'navier_stokes_dir': './data/navier_stokes/',
        'fluid_2d_dir': './data/fluid_2d/',
        'era5_dir': './data/era5/',
    },
}


def load_config(config_path=None):
    """Load configuration from YAML file, falling back to defaults."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        config = deepcopy(DEFAULT_CONFIG)
        _deep_update(config, user_config)
        return config
    return deepcopy(DEFAULT_CONFIG)


def _deep_update(base, update):
    """Recursively update a nested dict."""
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def get_experiment_config(experiment_name):
    """Get specialized config for each experiment."""
    config = deepcopy(DEFAULT_CONFIG)
    
    if experiment_name == 'burgers':
        pass
    elif experiment_name == 'navier_stokes':
        config['ddim']['sampling_steps'] = 50
        config['ddim']['eta'] = 1.0
    elif experiment_name == 'fluid_2d':
        config['unet_3d']['init_dim'] = 100
        config['training']['batch_size'] = 4
        config['ddim']['sampling_steps'] = 100
    elif experiment_name == 'era5':
        config['training']['batch_size'] = 4
        config['ddim']['sampling_steps'] = 100
    
    return config
