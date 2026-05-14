#!/usr/bin/env python3
"""
Script to run MR.Q experiments across all benchmarks.

Reproduces the main results from:
"Towards General-Purpose Model-Free RL (MR.Q)" - Fujimoto et al., 2025

Usage:
    # Run all Gym experiments (5 envs x 10 seeds)
    python run_experiments.py --benchmark gym --seeds 10

    # Run single environment
    python run_experiments.py --benchmark gym --env HalfCheetah-v4 --seeds 3

    # Run DMC proprioceptive
    python run_experiments.py --benchmark dmc_proprio --seeds 10

    # Run Atari
    python run_experiments.py --benchmark atari --seeds 10

    # Run all benchmarks
    python run_experiments.py --benchmark all --seeds 10
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


GYM_ENVS = [
    "Ant-v4",
    "HalfCheetah-v4",
    "Hopper-v4",
    "Humanoid-v4",
    "Walker2d-v4",
]

DMC_ENVS = [
    "acrobot-swingup",
    "ball_in_cup-catch",
    "cartpole-balance",
    "cartpole-balance_sparse",
    "cartpole-swingup",
    "cartpole-swingup_sparse",
    "cheetah-run",
    "dog-run",
    "dog-stand",
    "dog-trot",
    "dog-walk",
    "finger-spin",
    "finger-turn_easy",
    "finger-turn_hard",
    "fish-swim",
    "hopper-hop",
    "hopper-stand",
    "humanoid-run",
    "humanoid-stand",
    "humanoid-walk",
    "pendulum-swingup",
    "quadruped-run",
    "quadruped-walk",
    "reacher-easy",
    "reacher-hard",
    "walker-run",
    "walker-stand",
    "walker-walk",
]

ATARI_ENVS = [
    "Alien", "Amidar", "Assault", "Asterix", "Asteroids", "Atlantis",
    "BankHeist", "BattleZone", "BeamRider", "Berzerk", "Bowling", "Boxing",
    "Breakout", "Centipede", "ChopperCommand", "CrazyClimber", "Defender",
    "DemonAttack", "DoubleDunk", "Enduro", "FishingDerby", "Freeway",
    "Frostbite", "Gopher", "Gravitar", "Hero", "IceHockey", "Jamesbond",
    "Kangaroo", "Krull", "KungFuMaster", "MontezumaRevenge", "MsPacman",
    "NameThisGame", "Phoenix", "Pitfall", "Pong", "PrivateEye", "Qbert",
    "Riverraid", "RoadRunner", "Robotank", "Seaquest", "Skiing", "Solaris",
    "SpaceInvaders", "StarGunner", "Surround", "Tennis", "TimePilot",
    "Tutankham", "UpNDown", "Venture", "VideoPinball", "WizardOfWor",
    "YarsRevenge", "Zaxxon",
]

BENCHMARK_ENV_TYPE = {
    "gym": "gym",
    "dmc_proprio": "dmc_proprio",
    "dmc_visual": "dmc_visual",
    "atari": "atari",
}

BENCHMARK_ENVS = {
    "gym": GYM_ENVS,
    "dmc_proprio": DMC_ENVS,
    "dmc_visual": DMC_ENVS,
    "atari": ATARI_ENVS,
}


def run_single(env_type, env_name, seed, output_dir, cpu=False):
    """Run a single experiment."""
    cmd = [
        sys.executable, "train.py",
        "--env_type", env_type,
        "--env_name", env_name,
        "--seed", str(seed),
        "--output_dir", output_dir,
    ]
    if cpu:
        cmd.append("--cpu")

    print(f"Running: {env_type} {env_name} seed={seed}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: {env_type} {env_name} seed={seed}")
        print(result.stderr[-500:])
    else:
        print(f"Done: {env_type} {env_name} seed={seed}")

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run MR.Q experiments")
    parser.add_argument("--benchmark", type=str, default="gym",
                        choices=["gym", "dmc_proprio", "dmc_visual", "atari", "all"],
                        help="Benchmark to run")
    parser.add_argument("--env", type=str, default=None,
                        help="Specific environment (optional)")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of seeds (paper uses 10)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Number of parallel workers")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU usage")
    args = parser.parse_args()

    benchmarks = ["gym", "dmc_proprio", "dmc_visual", "atari"] if args.benchmark == "all" else [args.benchmark]

    # Build job list
    jobs = []
    for benchmark in benchmarks:
        env_type = BENCHMARK_ENV_TYPE[benchmark]
        envs = BENCHMARK_ENVS[benchmark]
        if args.env:
            envs = [args.env]

        for env_name in envs:
            for seed in range(args.seeds):
                jobs.append((env_type, env_name, seed, args.output_dir, args.cpu))

    print(f"Total jobs: {len(jobs)}")

    if args.n_workers > 1:
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {
                executor.submit(run_single, *job): job
                for job in jobs
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    success = future.result()
                except Exception as e:
                    print(f"Exception for {job}: {e}")
    else:
        for job in jobs:
            run_single(*job)

    print("\nAll experiments complete!")
    print(f"Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
