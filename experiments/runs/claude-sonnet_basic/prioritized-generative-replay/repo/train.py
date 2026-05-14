"""
Main training script for Prioritized Generative Replay (PGR).

Supports:
- State-based DMC tasks (quadruped-walk, cheetah-run, reacher-hard, finger-turn-hard)
- OpenAI Gym tasks (Walker2d-v2, HalfCheetah-v2, Hopper-v2)
- Multiple relevance functions (curiosity, return, td_error, reward)
- Comparison baselines (REDQ, SYNTHER)

Usage:
    python train.py --env quadruped-walk --relevance curiosity --seed 0
    python train.py --env HalfCheetah-v2 --relevance curiosity --seed 0
    python train.py --env quadruped-walk --mode synther --seed 0  # unconditional baseline
"""

import argparse
import os
import json
import numpy as np
import torch

from pgr import PGRTrainer


def make_env(env_name: str, seed: int = 0):
    """Create environment by name."""
    try:
        import gymnasium as gym
        # Try OpenAI Gym environments
        if env_name in ["Walker2d-v2", "HalfCheetah-v2", "Hopper-v2",
                        "Walker2d-v4", "HalfCheetah-v4", "Hopper-v4"]:
            env = gym.make(env_name)
            env.reset(seed=seed)
            return env
    except ImportError:
        pass

    try:
        import gym as old_gym
        if env_name in ["Walker2d-v2", "HalfCheetah-v2", "Hopper-v2"]:
            env = old_gym.make(env_name)
            env.seed(seed)
            return env
    except ImportError:
        pass

    # Try DMC environments
    try:
        from dm_control import suite
        import gymnasium as gym
        from gymnasium.wrappers import TimeLimit

        dmc_map = {
            "quadruped-walk": ("quadruped", "walk"),
            "cheetah-run": ("cheetah", "run"),
            "reacher-hard": ("reacher", "hard"),
            "finger-turn-hard": ("finger", "turn_hard"),
            "walker-walk": ("walker", "walk"),
            "hopper-hop": ("hopper", "hop"),
        }

        if env_name in dmc_map:
            domain, task = dmc_map[env_name]
            from pgr.dmc_wrapper import DMCWrapper
            env = DMCWrapper(domain, task, seed=seed)
            return env
    except ImportError:
        pass

    raise ValueError(f"Unknown environment: {env_name}")


def get_env_dims(env):
    """Get observation and action dimensions."""
    import numpy as np
    obs_space = env.observation_space
    act_space = env.action_space

    if hasattr(obs_space, 'shape'):
        obs_dim = int(np.prod(obs_space.shape))
    else:
        obs_dim = obs_space.n

    if hasattr(act_space, 'shape'):
        action_dim = int(np.prod(act_space.shape))
    else:
        action_dim = act_space.n

    return obs_dim, action_dim


def parse_args():
    parser = argparse.ArgumentParser(description="Prioritized Generative Replay")

    # Environment
    parser.add_argument("--env", type=str, default="quadruped-walk",
                        help="Environment name")
    parser.add_argument("--seed", type=int, default=0)

    # Mode
    parser.add_argument("--mode", type=str, default="pgr",
                        choices=["pgr", "synther", "redq"],
                        help="Training mode: pgr (ours), synther (unconditional), redq (model-free)")
    parser.add_argument("--relevance", type=str, default="curiosity",
                        choices=["curiosity", "return", "td_error", "reward"],
                        help="Relevance function for PGR")

    # Training
    parser.add_argument("--total_steps", type=int, default=100_000)
    parser.add_argument("--eval_freq", type=int, default=5_000)
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=5_000)

    # PGR hyperparameters
    parser.add_argument("--synthetic_ratio", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--utd_ratio", type=int, default=20)
    parser.add_argument("--inner_loop_freq", type=int, default=10_000)
    parser.add_argument("--guidance_scale", type=float, default=1.2)
    parser.add_argument("--p_uncond", type=float, default=0.25)

    # Diffusion
    parser.add_argument("--diffusion_hidden_dim", type=int, default=256)
    parser.add_argument("--diffusion_n_layers", type=int, default=4)
    parser.add_argument("--diffusion_n_timesteps", type=int, default=100)
    parser.add_argument("--diffusion_train_steps", type=int, default=200_000)
    parser.add_argument("--diffusion_lr", type=float, default=3e-4)

    # RL
    parser.add_argument("--rl_hidden_dim", type=int, default=256)
    parser.add_argument("--rl_n_layers", type=int, default=2)
    parser.add_argument("--n_q_networks", type=int, default=10)
    parser.add_argument("--rl_lr", type=float, default=3e-4)

    # Scaling experiments (Section 5.3)
    parser.add_argument("--large_network", action="store_true",
                        help="Use larger networks (3 layers, 512 hidden) for scaling experiments")
    parser.add_argument("--high_synthetic_ratio", action="store_true",
                        help="Use higher synthetic ratio (0.75) for scaling experiments")
    parser.add_argument("--high_utd", action="store_true",
                        help="Use UTD=40 for scaling experiments")

    # Misc
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="pgr")

    return parser.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")
    print(f"Environment: {args.env}")
    print(f"Mode: {args.mode}, Relevance: {args.relevance}")
    print(f"Seed: {args.seed}")

    # Create environments
    env = make_env(args.env, seed=args.seed)
    eval_env = make_env(args.env, seed=args.seed + 100)

    obs_dim, action_dim = get_env_dims(env)
    print(f"Obs dim: {obs_dim}, Action dim: {action_dim}")

    # Scaling experiment overrides
    rl_hidden_dim = args.rl_hidden_dim
    rl_n_layers = args.rl_n_layers
    batch_size = args.batch_size
    utd_ratio = args.utd_ratio
    synthetic_ratio = args.synthetic_ratio
    n_syn_samples = 1_000_000
    syn_buffer_size = 1_000_000

    if args.large_network:
        rl_hidden_dim = 512
        rl_n_layers = 3
        batch_size = 1024  # Scale batch size with network size

    if args.high_synthetic_ratio:
        synthetic_ratio = 0.75
        batch_size = 512

    if args.high_utd:
        utd_ratio = 40
        n_syn_samples = 2_000_000
        syn_buffer_size = 2_000_000

    # Mode-specific settings
    if args.mode == "redq":
        # Pure model-free REDQ (no generative replay)
        synthetic_ratio = 0.0
        diffusion_train_steps = 0
    elif args.mode == "synther":
        # Unconditional generative replay (SynthER baseline)
        # Use guidance_scale=1.0 to disable CFG (unconditional generation)
        args.guidance_scale = 1.0
        args.relevance = "reward"  # Dummy relevance (not used for conditioning)

    # Save directory
    run_name = f"{args.env}_{args.mode}_{args.relevance}_seed{args.seed}"
    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save config
    config = vars(args)
    config.update({
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "device": device,
        "rl_hidden_dim": rl_hidden_dim,
        "rl_n_layers": rl_n_layers,
        "batch_size": batch_size,
        "utd_ratio": utd_ratio,
        "synthetic_ratio": synthetic_ratio,
    })
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Initialize WandB if requested
    if args.log_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=config,
            )
        except ImportError:
            print("WandB not available, skipping logging")
            args.log_wandb = False

    # Create PGR trainer
    trainer = PGRTrainer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        relevance_type=args.relevance,
        synthetic_ratio=synthetic_ratio,
        batch_size=batch_size,
        utd_ratio=utd_ratio,
        inner_loop_freq=args.inner_loop_freq,
        diffusion_hidden_dim=args.diffusion_hidden_dim,
        diffusion_n_layers=args.diffusion_n_layers,
        diffusion_n_timesteps=args.diffusion_n_timesteps,
        diffusion_lr=args.diffusion_lr,
        diffusion_batch_size=batch_size,
        diffusion_train_steps=args.diffusion_train_steps if args.mode != "redq" else 0,
        p_uncond=args.p_uncond,
        guidance_scale=args.guidance_scale,
        rl_hidden_dim=rl_hidden_dim,
        rl_n_layers=rl_n_layers,
        n_q_networks=args.n_q_networks,
        rl_lr=args.rl_lr,
        n_syn_samples=n_syn_samples,
        syn_buffer_size=syn_buffer_size,
        device=device,
        seed=args.seed,
    )

    # Train
    metrics = trainer.train(
        env=env,
        eval_env=eval_env,
        total_env_steps=args.total_steps,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        warmup_steps=args.warmup_steps,
        save_dir=save_dir,
    )

    # Save final results
    results = {
        "env": args.env,
        "mode": args.mode,
        "relevance": args.relevance,
        "seed": args.seed,
        "final_eval_reward": metrics["episode_reward"][-1][1] if metrics["episode_reward"] else 0.0,
        "eval_rewards": metrics["episode_reward"],
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining complete!")
    if metrics["episode_reward"]:
        print(f"Final eval reward: {metrics['episode_reward'][-1][1]:.2f}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
