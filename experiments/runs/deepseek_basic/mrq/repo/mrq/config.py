"""
Default hyperparameter configuration for MR.Q.
Based on Table 3 in the paper.

All values are fixed across all benchmarks (Gym, DMC Proprioceptive, 
DMC Visual, Atari) as described in the paper.
"""

MRQ_CONFIG = {
    # Architecture
    'zs_dim': 512,
    'za_dim': 256,
    'zsa_dim': 512,
    'hidden_dim': 512,
    
    # Encoder
    'num_reward_bins': 65,
    'encoder_horizon': 5,
    'lambda_dynamics': 0.1,
    'lambda_reward': 0.1,
    'lambda_terminal': 0.1,
    'lambda_pre_activ': 1e-5,
    
    # TD3
    'multi_step_horizon': 3,
    'target_policy_noise_std': 0.2,
    'target_policy_noise_clip': 0.3,
    
    # LAP (Fujimoto et al., 2020)
    'lap_alpha': 0.4,
    'lap_min_priority': 1.0,
    
    # Exploration
    'initial_random_steps': 10000,
    'exploration_noise_std': 0.2,
    
    # Common
    'discount': 0.99,
    'replay_capacity': int(1e6),
    'batch_size': 256,
    'target_update_freq': 250,
    'replay_ratio': 1,  # 1 update per environment step
    
    # Optimizer (AdamW, Loshchilov & Hutter 2019)
    'weight_decay': 1e-4,
    'grad_clip_norm': 20.0,
    
    # Encoder optimizer
    'encoder_lr': 1e-4,
    
    # Value optimizer
    'value_lr': 3e-4,
    
    # Policy optimizer
    'policy_lr': 3e-4,
    
    # Activation functions
    # - Encoder: ELU (Clevert et al., 2015)
    # - Value: ELU
    # - Policy: ReLU
    
    # Weight initialization
    # - Xavier uniform (Glorot & Bengio, 2010)
    # - Bias: 0
    
    # Reward bins
    # - 65 bins with symexp spacing
    # - Effective range: [-22026, 22026]
    
    # Gumbel-Softmax temperature (Jang et al., 2017)
    'gumbel_softmax_tau': 10,
    
    # Reward range for two-hot encoding
    'symexp_bound': 10.0,
}

# Benchmark-specific environment settings
BENCHMARK_CONFIGS = {
    'gym_locomotion': {
        'envs': [
            'Ant-v4', 'HalfCheetah-v4', 'Hopper-v4', 
            'Humanoid-v4', 'Walker2d-v4'
        ],
        'total_steps': 1_000_000,
        'eval_freq': 5000,
        'image_observations': False,
    },
    'dmc_proprioceptive': {
        'envs': [
            'acrobot-swingup', 'ball_in_cup-catch', 'cartpole-balance',
            'cartpole-balance_sparse', 'cartpole-swingup', 
            'cartpole-swingup_sparse', 'cheetah-run', 'dog-run',
            'dog-stand', 'dog-trot', 'dog-walk', 'finger-spin',
            'finger-turn_easy', 'finger-turn_hard', 'fish-swim',
            'hopper-hop', 'hopper-stand', 'humanoid-run',
            'humanoid-stand', 'humanoid-walk', 'pendulum-swingup',
            'quadruped-run', 'quadruped-walk', 'reacher-easy',
            'reacher-hard', 'walker-run', 'walker-stand', 'walker-walk'
        ],
        'total_steps': 500_000,
        'eval_freq': 5000,
        'image_observations': False,
        'action_repeat': 2,
    },
    'dmc_visual': {
        'envs': [
            'acrobot-swingup', 'ball_in_cup-catch', 'cartpole-balance',
            'cartpole-balance_sparse', 'cartpole-swingup',
            'cartpole-swingup_sparse', 'cheetah-run', 'dog-run',
            'dog-stand', 'dog-trot', 'dog-walk', 'finger-spin',
            'finger-turn_easy', 'finger-turn_hard', 'fish-swim',
            'hopper-hop', 'hopper-stand', 'humanoid-run',
            'humanoid-stand', 'humanoid-walk', 'pendulum-swingup',
            'quadruped-run', 'quadruped-walk', 'reacher-easy',
            'reacher-hard', 'walker-run', 'walker-stand', 'walker-walk'
        ],
        'total_steps': 500_000,
        'eval_freq': 5000,
        'image_observations': True,
        'action_repeat': 2,
        'image_size': 84,
        'state_channels': 3,  # RGB
        'frame_stack': 3,  # Use previous 3 observations
    },
    'atari': {
        'envs': None,  # 57 games from DreamerV3
        'total_steps': 2_500_000,
        'eval_freq': 100_000,
        'image_observations': True,
        'action_repeat': 4,
        'image_size': 84,
        'state_channels': 4,  # 4 stacked frames
        'frame_stack': 4,
        'grayscale': True,
        'sticky_actions': True,
    },
}

# Normalization reference scores from the paper
TD3_NORMALIZATION = {
    'Ant-v4': {'random': -70.288, 'td3': 3942},
    'HalfCheetah-v4': {'random': -289.415, 'td3': 10574},
    'Hopper-v4': {'random': 18.791, 'td3': 3226},
    'Humanoid-v4': {'random': 120.423, 'td3': 5165},
    'Walker2d-v4': {'random': 2.791, 'td3': 3946},
}
