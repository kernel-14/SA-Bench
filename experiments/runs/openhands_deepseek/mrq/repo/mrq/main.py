"""Main entry point for running MR.Q experiments."""

import argparse
from mrq.config import gym_locomotion_config, dmc_proprio_config, dmc_visual_config, atari_config
from mrq.trainer import (
    run_training,
    GYM_LOCOMOTION_ENVS,
    DMC_PROPRIO_ENVS,
    ATARI_ENVS,
)


def main():
    parser = argparse.ArgumentParser(description="MR.Q: Model-based Representations for Q-learning")
    parser.add_argument("--benchmark", type=str, default="gym",
                        choices=["gym", "dmc_proprio", "dmc_visual", "atari"],
                        help="Benchmark to run")
    parser.add_argument("--env", type=str, default=None,
                        help="Specific environment name (default: all in benchmark)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    args = parser.parse_args()

    if args.benchmark == "gym":
        config_fn = gym_locomotion_config
        envs = GYM_LOCOMOTION_ENVS
    elif args.benchmark == "dmc_proprio":
        config_fn = dmc_proprio_config
        envs = DMC_PROPRIO_ENVS
    elif args.benchmark == "dmc_visual":
        config_fn = dmc_visual_config
        envs = DMC_PROPRIO_ENVS
    elif args.benchmark == "atari":
        config_fn = atari_config
        envs = ATARI_ENVS
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")

    if args.env is not None:
        envs = [args.env]

    for env_name in envs:
        cfg = config_fn()
        cfg.env_name = env_name
        cfg.seed = args.seed
        print(f"\n{'=' * 60}")
        print(f"Benchmark: {args.benchmark}, Env: {env_name}")
        print(f"{'=' * 60}")
        result = run_training(cfg, env_name, device=args.device)
        print(f"Final eval results: {result['eval_results'][-5:] if result['eval_results'] else 'None'}")


if __name__ == "__main__":
    main()
