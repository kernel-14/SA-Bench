"""Training loop for PGR and all baselines.

Implements the outer loop from Algorithm 1:
- Collect real transitions
- Periodically run inner loop (train diffusion, generate synthetic data)
- Train policy on mixed real+synthetic data
"""

import os
import time
import argparse
from typing import Dict, Any
import numpy as np
import torch
from tqdm import tqdm

from config import RunConfig, DiffusionConfig, PolicyConfig, ReplayConfig, CuriosityConfig, EnvConfig
from pgr import PGR, SYNTHER
from envs import make_env
from utils import Logger, evaluate_policy, compute_dormant_ratio


def run_experiment(config: RunConfig, log_dir: str):
    """Run a full PGR experiment."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"[PGR] Using device: {device}")

    # Create environment
    env = make_env(config)
    is_pixel = "pixel" in config.experiment

    # Get observation/action dimensions
    if is_pixel:
        obs_shape = env.observation_space.shape
        state_dim = config.policy.latent_dim
        action_dim = env.action_space.shape[0]
    else:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

    print(f"[PGR] State dim: {state_dim}, Action dim: {action_dim}, Pixel: {is_pixel}")

    # Create PGR instance
    pgr = PGR(config, state_dim, action_dim, is_pixel=is_pixel)

    # Logger
    logger = Logger(log_dir)

    # Training loop
    total_steps = config.total_env_steps
    obs, _ = env.reset()
    episode_return = 0.0
    episode_steps = 0

    pbar = tqdm(total=total_steps, desc="Env steps")

    while pgr.total_env_steps < total_steps:
        # Select action
        action = pgr.select_action(obs)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_return += reward
        episode_steps += 1

        # Store transition
        pgr.observe(obs, action, reward, next_obs, float(done))

        obs = next_obs

        # Episode end
        if done:
            pgr.end_episode()
            obs, _ = env.reset()
            if len(pgr.episode_returns) % 10 == 0:
                logger.log_scalar("episode_return", pgr.episode_returns[-1])
            episode_return = 0.0
            episode_steps = 0

        # Periodic inner loop (Algorithm 1, step 4-8)
        if pgr.should_update():
            pgr.run_inner_loop()

            # Log metrics after inner loop
            metrics = pgr.get_metrics()
            logger.log_scalars(metrics)

        # Train policy (step 7 - performed continuously)
        if pgr.total_env_steps > config.replay.n_seed_steps:
            pgr.update_policy()

        # Periodic evaluation
        if pgr.total_env_steps % config.eval_frequency == 0 and pgr.total_env_steps > 0:
            avg_return, std_return = evaluate_policy(
                env, pgr.policy,
                num_episodes=config.n_eval_episodes,
                is_pixel=is_pixel,
                device=device,
            )
            logger.log_scalar("eval/avg_return", avg_return, pgr.total_env_steps)
            logger.log_scalar("eval/std_return", std_return, pgr.total_env_steps)
            print(f"[Step {pgr.total_env_steps:6d}] Eval return: {avg_return:.2f} ± {std_return:.2f}")

        pbar.update(1)

    pbar.close()

    # Final evaluation
    avg_return, std_return = evaluate_policy(
        env, pgr.policy,
        num_episodes=config.n_eval_episodes * 2,
        is_pixel=is_pixel,
        device=device,
    )
    print(f"[Final] Return: {avg_return:.2f} ± {std_return:.2f}")

    # Save model
    model_path = os.path.join(log_dir, "final_model.pt")
    pgr.save(model_path)
    print(f"[PGR] Model saved to {model_path}")

    logger.close()
    return avg_return, std_return


def run_synther_baseline(config: RunConfig, log_dir: str):
    """Run SYNTHER baseline (unconditional generative replay)."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    env = make_env(config)
    is_pixel = "pixel" in config.experiment

    if is_pixel:
        state_dim = config.policy.latent_dim
        action_dim = env.action_space.shape[0]
    else:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

    synther = SYNTHER(config, state_dim, action_dim, is_pixel=is_pixel)

    logger = Logger(log_dir)

    total_steps = config.total_env_steps
    obs, _ = env.reset()

    pbar = tqdm(total=total_steps, desc="SYNTHER env steps")

    while synther.total_env_steps < total_steps:
        action = synther.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        synther.observe(obs, action, reward, next_obs, float(done))
        obs = next_obs

        if done:
            synther.end_episode()
            obs, _ = env.reset()

        if synther.should_update():
            synther.run_inner_loop()
            metrics = synther.get_metrics()
            logger.log_scalars(metrics)

        if synther.total_env_steps > config.replay.n_seed_steps:
            synther.update_policy()

        pbar.update(1)

    pbar.close()

    avg_return, std_return = evaluate_policy(
        env, synther.policy,
        num_episodes=config.n_eval_episodes * 2,
        is_pixel=is_pixel,
        device=device,
    )
    print(f"[SYNTHER Final] Return: {avg_return:.2f} ± {std_return:.2f}")

    model_path = os.path.join(log_dir, "synther_final.pt")
    synther.save(model_path)

    logger.close()
    return avg_return, std_return


def run_model_free_baseline(config: RunConfig, log_dir: str):
    """Run model-free baseline (REDQ or SAC)."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    env = make_env(config)
    is_pixel = "pixel" in config.experiment

    if is_pixel:
        state_dim = config.policy.latent_dim
        action_dim = env.action_space.shape[0]
    else:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

    from models.policy import REDQPolicy, SACPolicy
    from replay import ReplayBuffer

    if config.noisy_nets:
        policy = REDQPolicy(
            state_dim=state_dim, action_dim=action_dim,
            hidden_dim=config.policy.hidden_dims,
            n_layers=config.policy.n_hidden_layers,
            n_critics=config.policy.n_critics,
            n_target_critics=config.policy.n_target_critics,
            gamma=config.policy.gamma,
            tau=config.policy.tau,
            noisy=True,
            bootstrapped=config.bootstrapped_q,
        ).to(device)
    else:
        if config.experiment == "sac":
            policy = SACPolicy(
                state_dim=state_dim, action_dim=action_dim,
                hidden_dim=config.policy.hidden_dims,
                n_layers=config.policy.n_hidden_layers,
                gamma=config.policy.gamma,
                tau=config.policy.tau,
            ).to(device)
        else:
            policy = REDQPolicy(
                state_dim=state_dim, action_dim=action_dim,
                hidden_dim=config.policy.hidden_dims,
                n_layers=config.policy.n_hidden_layers,
                n_critics=config.policy.n_critics,
                n_target_critics=config.policy.n_target_critics,
                gamma=config.policy.gamma,
                tau=config.policy.tau,
                bootstrapped=config.bootstrapped_q,
            ).to(device)

    actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=config.policy.actor_lr)
    if hasattr(policy, 'critics'):
        critic_params = sum([list(c.parameters()) for c in policy.critics], [])
    else:
        critic_params = list(policy.critic1.parameters()) + list(policy.critic2.parameters())
    critic_optimizer = torch.optim.Adam(critic_params, lr=config.policy.critic_lr)
    alpha_optimizer = torch.optim.Adam([policy.log_alpha], lr=config.policy.actor_lr)

    # Use PER or standard replay
    if config.use_per:
        from replay import PrioritizedReplayBuffer
        buffer = PrioritizedReplayBuffer(
            capacity=config.replay.real_buffer_capacity,
            state_dim=state_dim,
            action_dim=action_dim,
        )

        if config.relevance_fn == "curiosity":
            from models.curiosity import ICM
            relevance_fn = ICM(
                state_dim=state_dim, action_dim=action_dim,
                feature_dim=config.curiosity.feature_dim,
                hidden_dim=config.curiosity.hidden_dim,
                lr=config.curiosity.lr,
            ).to(device)
        else:
            relevance_fn = None
    else:
        buffer = ReplayBuffer(
            capacity=config.replay.real_buffer_capacity,
            state_dim=state_dim,
            action_dim=action_dim,
            pixel_based=is_pixel,
        )
        relevance_fn = None

    logger = Logger(log_dir)
    total_steps = config.total_env_steps
    obs, _ = env.reset()

    pbar = tqdm(total=total_steps, desc="Model-free steps")
    total_env_steps = 0

    while total_env_steps < total_steps:
        # Select action
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action = policy.get_action(obs_tensor, deterministic=False)
        action = action.squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Compute priority for PER
        priority = None
        if config.use_per and relevance_fn is not None:
            with torch.no_grad():
                s = torch.FloatTensor(obs).unsqueeze(0).to(device)
                a = torch.FloatTensor(action).unsqueeze(0).to(device)
                ns = torch.FloatTensor(next_obs).unsqueeze(0).to(device)
                priority = relevance_fn.compute_relevance(s, a, ns).item()

        buffer.push(obs, action, reward, next_obs, float(done), priority)

        obs = next_obs
        total_env_steps += 1

        if done:
            obs, _ = env.reset()

        # Train policy
        if total_env_steps > config.replay.n_seed_steps and len(buffer) >= config.policy.batch_size:
            if config.use_per:
                s, a, r, ns, d, weights, indices = buffer.sample(
                    config.policy.batch_size, device
                )
            else:
                s, a, r, ns, d = buffer.sample(
                    config.policy.batch_size, device
                )
                weights = None
                indices = None

            utd = config.scaling_utd if config.scaling_utd > 0 else config.policy.utd
            for _ in range(utd):
                critic_loss = policy.critic_loss(s, a, r, ns, d)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                actor_loss, alpha_loss = policy.actor_loss(s)
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()

                policy.update_targets()

            # Update PER priorities
            if config.use_per and indices is not None:
                with torch.no_grad():
                    new_priorities = (critic_loss.detach().cpu().numpy() + 1e-6)
                buffer.update_priorities(indices, new_priorities)

        pbar.update(1)

    pbar.close()

    avg_return, std_return = evaluate_policy(
        env, policy,
        num_episodes=config.n_eval_episodes * 2,
        is_pixel=is_pixel,
        device=device,
    )
    print(f"[Model-free Final] Return: {avg_return:.2f} ± {std_return:.2f}")

    logger.close()
    return avg_return, std_return


def run_dormant_ratio_analysis(config: RunConfig, log_dir: str):
    """Compute dormant ratio over training (Section 5.2, Fig. 6a)."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    env = make_env(config)
    is_pixel = "pixel" in config.experiment

    if is_pixel:
        state_dim = config.policy.latent_dim
    else:
        state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    pgr = PGR(config, state_dim, action_dim, is_pixel=is_pixel)
    logger = Logger(log_dir)

    total_steps = config.total_env_steps
    obs, _ = env.reset()
    dormant_ratios = []

    pbar = tqdm(total=total_steps, desc="DR Analysis")

    while pgr.total_env_steps < total_steps:
        action = pgr.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        pgr.observe(obs, action, reward, next_obs, float(done))
        obs = next_obs

        if done:
            pgr.end_episode()
            obs, _ = env.reset()

        if pgr.should_update():
            pgr.run_inner_loop()

        if pgr.total_env_steps > config.replay.n_seed_steps:
            pgr.update_policy()

        # Compute dormant ratio periodically
        if pgr.total_env_steps % 5000 == 0:
            with torch.no_grad():
                sample_obs = torch.randn(256, state_dim, device=device)
                dr = compute_dormant_ratio(pgr.policy.actor, sample_obs)
                dormant_ratios.append(dr)
                logger.log_scalar("dormant_ratio", dr, pgr.total_env_steps)

        pbar.update(1)

    pbar.close()
    logger.close()
    return dormant_ratios


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prioritized Generative Replay")
    parser.add_argument("--experiment", type=str, default="dmc_state",
                        choices=["dmc_state", "dmc_pixel", "gym", "dmlab", "scaling"])
    parser.add_argument("--relevance_fn", type=str, default="curiosity",
                        choices=["curiosity", "td_error", "return", "reward", "rnd", "cts", "eco"])
    parser.add_argument("--baseline", type=str, default="pgr",
                        choices=["pgr", "synther", "model_free", "per", "dormant_ratio"])
    parser.add_argument("--env_domain", type=str, default="quadruped")
    parser.add_argument("--env_task", type=str, default="walk")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--noisy_nets", action="store_true")
    parser.add_argument("--bootstrapped_q", action="store_true")
    parser.add_argument("--scaling_larger_network", action="store_true")
    parser.add_argument("--scaling_higher_ratio", action="store_true")
    args = parser.parse_args()

    # Build config
    if args.experiment == "dmc_state":
        from config import get_dmc_state_config
        config = get_dmc_state_config()
    elif args.experiment == "dmc_pixel":
        from config import get_dmc_pixel_config
        config = get_dmc_pixel_config()
    elif args.experiment == "gym":
        from config import get_gym_config
        config = get_gym_config()
    elif args.experiment == "dmlab":
        from config import get_dmlab_config
        config = get_dmlab_config()
    elif args.experiment == "scaling":
        from config import get_scaling_config
        config = get_scaling_config()

    config.relevance_fn = args.relevance_fn
    config.seed = args.seed
    config.total_env_steps = args.total_steps
    config.env.dmc_domain = args.env_domain
    config.env.dmc_task = args.env_task
    config.noisy_nets = args.noisy_nets
    config.bootstrapped_q = args.bootstrapped_q

    if args.scaling_larger_network:
        config.scaling_larger_network = True
        config.policy.hidden_dims = 512
        config.policy.n_hidden_layers = 3
        config.policy.batch_size = 1024

    if args.scaling_higher_ratio:
        config.policy.batch_size = 512
        config.policy.synthetic_ratio = 0.75

    if args.baseline == "per":
        config.use_per = True

    # Create log directory
    log_dir = os.path.join(
        args.log_dir,
        f"{args.experiment}_{args.baseline}_{args.relevance_fn}_{args.env_domain}_{args.env_task}_seed{args.seed}"
    )
    os.makedirs(log_dir, exist_ok=True)

    print(f"[Train] Configuration: {config}")
    print(f"[Train] Log directory: {log_dir}")

    # Run selected baseline
    if args.baseline == "pgr":
        run_experiment(config, log_dir)
    elif args.baseline == "synther":
        run_synther_baseline(config, log_dir)
    elif args.baseline == "model_free":
        run_model_free_baseline(config, log_dir)
    elif args.baseline == "per":
        run_model_free_baseline(config, log_dir)
    elif args.baseline == "dormant_ratio":
        run_dormant_ratio_analysis(config, log_dir)
    else:
        raise ValueError(f"Unknown baseline: {args.baseline}")
