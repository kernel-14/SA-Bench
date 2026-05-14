"""Configuration for all experiments described in the paper.

Provides default configs for:
- Individual PDE problems (Burgers, Gray-Scott, Navier-Stokes, Heat, RD)
- Model architectures (FNO, MambaFNO, PerceiverIOFNO, CoDANO, SwinV2FNO)
- Pretraining, fine-tuning, and full experiment suites (Tables 1 & 2)
"""

# ===========================================================================
#  Core model configurations
# ===========================================================================

CORE_CONFIG = {
    'hidden_channels': 64,
    'n_layers': 4,
    'modes': 12,
    'modes1': 12,
    'modes2': 12,
    'ndim': 1,
    'num_heads': 8,
    'num_latents': 128,
    'mamba_d_state': 16,
    'mamba_d_conv': 4,
    'mamba_expand': 2,
    'swin_window_size': 8,
    'local_attn_window': 16,
}

# ===========================================================================
#  Problem configurations
# ===========================================================================

BURGERS_CONFIG = {
    'problem': 'burgers',
    'in_channels': 2,       # u0(x), nu
    'out_channels': 1,      # u(x, T)
    'ndim': 1,
    'nx': 256,
    'L': 1.0,
    'T': 1.0,
    'nu_min': 0.001,
    'nu_max': 0.1,
    'dt': 0.001,
    'train_samples': 1000,
    'test_samples': 200,
    'batch_size': 16,
    'lift_mode': 'mlp',
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'n_epochs': 100,
    'pretrain_variants': ['burgers_low_nu', 'burgers_mid_nu', 'burgers_high_nu'],
}

GRAYSCOTT_CONFIG = {
    'problem': 'grayscott',
    'in_channels': 6,       # u0, v0, Du, Dv, F, k
    'out_channels': 2,      # u, v
    'ndim': 2,
    'nx': 64,
    'L': 2.5,
    'T': 5.0,
    'Du': 0.16,
    'Dv': 0.08,
    'F': 0.035,
    'k': 0.065,
    'dt': 0.1,
    'train_samples': 500,
    'test_samples': 100,
    'batch_size': 8,
    'lift_mode': 'mlp',
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'n_epochs': 100,
    'pretrain_variants': ['grayscott_low_F', 'grayscott_mid_F', 'grayscott_high_F'],
}

NAVIERSTOKES_CONFIG = {
    'problem': 'navierstokes',
    'in_channels': 2,       # w0, nu
    'out_channels': 1,      # w(x, y, T)
    'ndim': 2,
    'nx': 64,
    'L': 6.283185307179586,  # 2*pi
    'T': 1.0,
    'nu_min': 1e-4,
    'nu_max': 1e-3,
    'dt': 0.01,
    'train_samples': 500,
    'test_samples': 100,
    'batch_size': 8,
    'lift_mode': 'mlp',
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'n_epochs': 100,
    'pretrain_variants': ['ns_low_nu', 'ns_high_nu'],
}

HEAT_CONFIG = {
    'problem': 'heat',
    'in_channels': 2,       # u0, alpha
    'out_channels': 1,      # u(x, y, T)
    'ndim': 2,
    'nx': 64,
    'L': 1.0,
    'T': 0.5,
    'with_convection': False,
    'dt': 0.001,
    'train_samples': 500,
    'test_samples': 100,
    'batch_size': 8,
    'lift_mode': 'mlp',
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'n_epochs': 100,
}

HEAT_CONVECTION_CONFIG = {
    'problem': 'heat_convection',
    'in_channels': 3,       # u0, alpha, beta
    'out_channels': 1,
    'ndim': 2,
    'nx': 64,
    'L': 1.0,
    'T': 0.5,
    'with_convection': True,
    'dt': 0.001,
    'train_samples': 500,
    'test_samples': 100,
    'batch_size': 8,
    'lift_mode': 'mlp',
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'n_epochs': 100,
}

RD_CONFIG = {
    'problem': 'rd_advection',
    'in_channels': 4,       # u0, v0, Du, Dv
    'out_channels': 2,
    'ndim': 2,
    'nx': 64,
    'L': 1.0,
    'T': 2.0,
    'with_advection': False,
    'dt': 0.005,
    'train_samples': 500,
    'test_samples': 100,
    'batch_size': 8,
    'lift_mode': 'mlp',
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'n_epochs': 100,
}

RD_ADVECTION_CONFIG = {
    'problem': 'rd_advection',
    'in_channels': 5,       # u0, v0, Du, Dv, beta
    'out_channels': 2,
    'ndim': 2,
    'nx': 64,
    'L': 1.0,
    'T': 2.0,
    'with_advection': True,
    'dt': 0.005,
    'train_samples': 500,
    'test_samples': 100,
    'batch_size': 8,
    'lift_mode': 'mlp',
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'n_epochs': 100,
}

ADVECTION_CONFIG = {
    'problem': 'advection',
    'in_channels': 2,       # u0, beta
    'out_channels': 1,
    'ndim': 1,
    'nx': 256,
    'L': 1.0,
    'T': 1.0,
    'dt': 0.001,
    'train_samples': 1000,
    'test_samples': 200,
    'batch_size': 16,
    'lift_mode': 'mlp',
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'n_epochs': 100,
}

# ===========================================================================
#  Pretraining and fine-tuning configurations
# ===========================================================================

PRETRAIN_CONFIG = {
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'n_epochs': 100,
    'batch_size': 16,
    'device': 'cuda',
}

FINETUNE_CONFIG = {
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'n_epochs': 50,
    'batch_size': 16,
    'device': 'cuda',
}

SCRATCH_CONFIG = {
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'n_epochs': 100,
    'batch_size': 16,
    'device': 'cuda',
}

# ===========================================================================
#  Full experiment suite (Tables 1 & 2)
# ===========================================================================

TABLE1_EXPERIMENT = {
    'mode': 'table1',
    'table1_problems': ['burgers', 'grayscott', 'navierstokes'],
    'models': ['mamba_fno', 'perceiver_io_fno', 'fno', 'codano'],
    'core_config': CORE_CONFIG,
    'problem_configs': {
        'burgers': BURGERS_CONFIG,
        'grayscott': GRAYSCOTT_CONFIG,
        'navierstokes': NAVIERSTOKES_CONFIG,
    },
    'pretrain_config': PRETRAIN_CONFIG,
    'finetune_config': FINETUNE_CONFIG,
    'scratch_config': SCRATCH_CONFIG,
}

TABLE2_EXPERIMENT = {
    'mode': 'table2',
    'table2_models': ['mamba_fno', 'perceiver_io_fno', 'fno', 'codano'],
    'core_config': CORE_CONFIG,
    'problem_configs': {
        'heat': HEAT_CONFIG,
        'heat_convection': HEAT_CONVECTION_CONFIG,
        'rd': RD_CONFIG,
        'rd_advection': RD_ADVECTION_CONFIG,
        'advection': ADVECTION_CONFIG,
        'burgers': BURGERS_CONFIG,
    },
    'multiphysics_source': ['advection', 'burgers'],
    'multiphysics_target': 'rd',
    'pretrain_config': PRETRAIN_CONFIG,
    'finetune_config': FINETUNE_CONFIG,
    'scratch_config': SCRATCH_CONFIG,
}


def get_config(experiment='table1'):
    """Return config for the requested experiment."""
    if experiment == 'table1':
        return dict(TABLE1_EXPERIMENT)
    elif experiment == 'table2':
        return dict(TABLE2_EXPERIMENT)
    elif experiment == 'burgers':
        cfg = dict(BURGERS_CONFIG)
        cfg['model_type'] = 'fno'
        cfg['mode'] = 'train_single'
        return cfg
    elif experiment == 'all':
        return {
            'table1': dict(TABLE1_EXPERIMENT),
            'table2': dict(TABLE2_EXPERIMENT),
        }
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
