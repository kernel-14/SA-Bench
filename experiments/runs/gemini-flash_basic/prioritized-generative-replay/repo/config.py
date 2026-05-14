import torch

class Config:
    # Environment
    env_name = "CartPole-v1" # Placeholder
    obs_dim = 4 # Placeholder
    action_dim = 2 # Placeholder

    # Replay Buffer
    replay_buffer_capacity = 1_000_000
    synthetic_buffer_capacity = 1_000_000

    # Training
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_timesteps = 1_000_000
    evaluation_interval = 10_000
    batch_size = 256
    gamma = 0.99
    lr = 3e-4
    tau = 0.005 # For target network updates
    update_policy_every_steps = 1 # Policy network update frequency

    # PGR Specific
    generative_model_train_freq = 10000 # Train generative model every X environment steps
    generative_model_updates_per_step = 1 # Number of gradient steps for generative model per update freq
    p_uncond = 0.25 # Probability of dropping condition for CFG during diffusion training
    guidance_scale = 1.0 # Classifier-free guidance scale during sampling
    synthetic_data_ratio = 0.5 # Ratio of synthetic to real data in mixed batch
    num_synthetic_samples = 128 # Number of synthetic samples to generate per update

    # Diffusion Model
    diffusion_hidden_dim = 256
    diffusion_num_layers = 4
    diffusion_dropout = 0.1
    diffusion_n_timesteps = 1000
    diffusion_condition_dim = 1 # Relevance score is a scalar

    # SAC Agent
    sac_alpha = 0.2 # Initial alpha for SAC
    sac_hidden_dim = 256

    # Curiosity Relevance Function
    feature_encoder_latent_dim = 512
    curiosity_optimizer_lr = 1e-3
