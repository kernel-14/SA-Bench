"""
main.py

Main entry point for reproducing the experiments of the Robotic World Model (RWM)
paper.  It orchestrates data collection, world‑model pretraining, MBPO‑PPO policy
optimisation, and evaluation.  All hyperparameters are read from the configuration
file (default configs/anydrive.yaml).  The script supports running multiple random
seeds and logs metrics to TensorBoard.

Usage:
    python main.py --config configs/anydrive.yaml --seeds 5 --device cuda
"""

import argparse
import datetime
import os
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

# Import project‑specific modules (must be on PYTHONPATH)
from env_utils import (
    IsaacEnvWrapper,
    ROBOT_CONFIGS,
    compute_reward,
    get_priv_split,
)
from dataset import TrajectoryBuffer
from world_model import RWM
from ppo_agent import PPOAgent
from mbrl_trainer import MBPOTrainer
from evaluation import Evaluation


# ------------------------------------------------------------------------------
# Helper: set random seeds
# ------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------------------
# Helper: GAE computation (for model‑free PPO collector)
# ------------------------------------------------------------------------------
def compute_gae(
    rewards: torch.Tensor,   # (T,)
    values: torch.Tensor,    # (T,)
    dones: torch.Tensor,     # (T,) bool
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE) and returns.

    Args:
        rewards: 1‑D tensor of rewards.
        values: 1‑D tensor of state values (V(s_t)).
        dones: 1‑D boolean tensor, True if episode terminated after this step.
        gamma: Discount factor.
        lam: GAE lambda.

    Returns:
        advantages: 1‑D tensor of advantage estimates.
        returns: 1‑D tensor of discounted returns.
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value * (~dones[t]) - values[t]
        gae = delta + gamma * lam * (~dones[t]) * gae
        advantages[t] = gae
        returns[t] = gae + values[t]
        next_value = values[t]
    return advantages, returns


# ------------------------------------------------------------------------------
# Model‑free PPO training for data‑collection policy
# ------------------------------------------------------------------------------
def train_collector_policy(
    env: IsaacEnvWrapper,
    config: Dict,
    robot: str,
    robot_cfg: Dict,
    device: torch.device,
    total_steps: int = 1_000_000,
    seed: int = 42,
    log_dir: Optional[str] = None,
) -> PPOAgent:
    """
    Train a PPO agent from scratch in the real environment to serve as
    the behaviour policy for data collection.  Returns the trained agent.

    The agent uses the same architecture as specified in the config
    (policy_network section).  Learning rate and other hyperparameters are
    fixed to standard values.
    """
    set_seed(seed)

    # Dimensions from the environment
    obs_dim = env.get_observation_space().shape[0]
    act_dim = env.get_action_space().shape[0]
    policy_cfg = config["policy_network"]

    # Build the agent
    agent = PPOAgent(
        obs_dim,
        act_dim,
        hidden_dims=policy_cfg["hidden_dims"],
        activation=policy_cfg["activation"],
        lr=0.001,               # typical model‑free PPO learning rate
        clip_range=0.2,
        entropy_coef=0.01,      # slightly higher entropy for exploration
        device=device,
    )

    writer = SummaryWriter(log_dir=log_dir) if log_dir else None

    global_step = 0
    update_freq = 2048         # steps per update (collect, then update)
    episode_rewards = []

    while global_step < total_steps:
        # ── Collect an episode ──
        policy_obs, info = env.reset()
        # Fetch the initial world observation and privileged info
        world_obs = torch.tensor(info["world_model_obs"], dtype=torch.float32, device=device)  # (obs_dim,)
        joint_vel_slice = robot_cfg["world_slices"]["joint_vel"]

        prev_action = torch.zeros(act_dim, device=device).unsqueeze(0)  # (1, act_dim) for prev step
        prev_joint_vel = world_obs[joint_vel_slice[0]:joint_vel_slice[1]]

        ep_obs = []
        ep_act = []
        ep_logp = []
        ep_val = []
        ep_rew = []
        ep_done = []

        done = False
        while not done:
            # Current policy observation (provided by env.reset/step)
            obs_tensor = torch.from_numpy(policy_obs).float().unsqueeze(0).to(device)  # (1, obs_dim)
            # Extract command from current policy observation (for reward later)
            cmd_slice = robot_cfg["command_slice"]
            command = torch.from_numpy(policy_obs[cmd_slice[0]:cmd_slice[1]]).float().unsqueeze(0).to(device)  # (1,3)

            # Get action and value from policy
            with torch.no_grad():
                _, _, value = agent.evaluate(obs_tensor, torch.zeros(1, act_dim, device=device))
            # evaluate might need an action; we have not yet sampled action at this call, but value doesn't depend on action. So we can compute value using a dummy action? Actually evaluate takes obs and act, returns action_mean, log_prob of given action, and value. The value is independent of the action argument; it's from the critic. So we can pass any action (e.g., zeros) and still get correct value. But to be safe, we can first sample action, then evaluate value after sampling, but then we'd lose speed? We can evaluate value with the action that we will take. So:
            action, log_prob = agent.act(obs_tensor)
            action_np = action.squeeze(0).cpu().numpy()
            _, _, value = agent.evaluate(obs_tensor, action)   # now action is correct

            # Step environment
            next_policy_obs, env_reward, term, trunc, info = env.step(action_np)
            done = term or trunc

            # New world observation and privileged info
            next_world_obs = torch.tensor(info["world_model_obs"], dtype=torch.float32, device=device)
            next_priv = torch.tensor(info["privileged"], dtype=torch.float32, device=device)
            joint_vel_next = next_world_obs[joint_vel_slice[0]:joint_vel_slice[1]]

            # Compute reward using the paper's reward function
            reward_tensor = compute_reward(
                world_obs=next_world_obs.unsqueeze(0),
                privileged=next_priv.unsqueeze(0),
                command=command,
                prev_action=prev_action,
                action=action,
                prev_joint_vel=prev_joint_vel.unsqueeze(0),
                joint_vel=joint_vel_next.unsqueeze(0),
                step_time=env.step_time,
                robot=robot,
                robot_cfg=robot_cfg,
            )
            reward = reward_tensor.item()

            # Store transition
            ep_obs.append(policy_obs.copy())
            ep_act.append(action_np.copy())
            ep_logp.append(log_prob.item())
            ep_val.append(value.item())
            ep_rew.append(reward)
            ep_done.append(done)

            # Prepare for next step
            policy_obs = next_policy_obs
            prev_action = action
            prev_joint_vel = joint_vel_next

        # ── Compute GAE ──
        T = len(ep_rew)
        rewards_t = torch.tensor(ep_rew, device=device)
        values_t = torch.tensor(ep_val, device=device)
        dones_t = torch.tensor(ep_done, device=device)
        advantages_t, returns_t = compute_gae(rewards_t, values_t, dones_t, gamma=0.99, lam=0.95)

        # ── Prepare rollout buffer for PPO update ──
        rollout = {
            "obs": torch.from_numpy(np.stack(ep_obs)).float().to(device),
            "act": torch.from_numpy(np.stack(ep_act)).float().to(device),
            "old_log_prob": torch.tensor(ep_logp, device=device).unsqueeze(-1),
            "adv": advantages_t.unsqueeze(-1),
            "ret": returns_t.unsqueeze(-1),
        }

        # ── Update the agent (several epochs) ──
        ppo_logs = agent.update(
            rollout_buffer=rollout,
            value_loss_coef=0.5,
            max_grad_norm=0.5,
            target_kl=None,         # no early stopping for collector
            ppo_epochs=8,           # more epochs for better sample efficiency
            ppo_minibatches=4,
        )

        global_step += T
        episode_rewards.append(sum(ep_rew))

        if writer:
            writer.add_scalar("collector/avg_reward", np.mean(episode_rewards[-100:]), global_step)
            writer.add_scalar("collector/policy_loss", ppo_logs["policy_loss"], global_step)
            writer.add_scalar("collector/value_loss", ppo_logs["value_loss"], global_step)

        if global_step % 50_000 == 0:
            print(f"  Collector step {global_step:8d}/{total_steps}  "
                  f"avg reward: {np.mean(episode_rewards[-100:]):.3f}")

    if writer:
        writer.close()

    return agent


# ------------------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Reproduce RWM & MBPO-PPO experiments.")
    parser.add_argument("--config", type=str, default="configs/anydrive.yaml",
                        help="Path to the YAML configuration file.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="List of random seeds to run (e.g., --seeds 0 1 2).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu).")
    parser.add_argument("--skip-collection", action="store_true",
                        help="Skip data collection if pre‑existing buffer found.")
    parser.add_argument("--collector-steps", type=int, default=2_000_000,
                        help="Number of environment steps for training the collector policy.")
    parser.add_argument("--pretrain-transitions", type=int, default=6_000_000,
                        help="Number of transitions to collect for world‑model pretraining.")
    parser.add_argument("--output-dir", type=str, default="experiments",
                        help="Root directory for outputs (checkpoints, logs).")
    args = parser.parse_args()

    # ── Load configuration ──
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    robot = config["environment"]["robot"]
    if robot not in ROBOT_CONFIGS:
        raise ValueError(f"Unsupported robot '{robot}'. Choose from {list(ROBOT_CONFIGS.keys())}.")
    robot_cfg = ROBOT_CONFIGS[robot]

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # Create base output directory (timestamped experiment folder)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(args.output_dir) / f"{robot}_{config['environment']['task']}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 0: Prepare environment ──
    env = IsaacEnvWrapper(robot=robot, task=config["environment"]["task"], config=config)

    # ── Phase 1: Data collection (only once, not per seed) ──
    pretrain_buffer_path = exp_dir / "pretrain_buffer.pt"
    if not args.skip_collection or not pretrain_buffer_path.exists():
        print("=== Phase 1: Data collection for world‑model pretraining ===")

        # Create collector PPO agent (train or load)
        collector_checkpoint = config.get("data_collection", {}).get("pretrained_policy_path", None)
        if collector_checkpoint and os.path.exists(collector_checkpoint):
            print(f"Loading collector policy from {collector_checkpoint}")
            obs_dim = env.get_observation_space().shape[0]
            act_dim = env.get_action_space().shape[0]
            collect_agent = PPOAgent(obs_dim, act_dim, hidden_dims=config["policy_network"]["hidden_dims"],
                                     activation=config["policy_network"]["activation"], device=device)
            collect_agent.load(collector_checkpoint)
        else:
            print("Training a new collector policy ...")
            collect_agent = train_collector_policy(
                env, config, robot, robot_cfg, device,
                total_steps=args.collector_steps,
                seed=42,  # fixed seed for data collection
                log_dir=str(exp_dir / "collector_logs"),
            )

        # Create buffer for pretraining data (unlimited capacity)
        pretrain_buffer = TrajectoryBuffer(
            capacity_transitions=None,
            obs_dim=robot_cfg["world_obs_dim"],
            act_dim=robot_cfg["action_dim"],
            priv_dim=robot_cfg["priv_dim"],
            device="cpu",
        )

        # Run data collection using the trained collector
        collected = 0
        target = args.pretrain_transitions
        print(f"Collecting {target} transitions ...")
        while collected < target:
            policy_obs, info = env.reset()
            world_obs_before = torch.from_numpy(info["world_model_obs"]).float()
            priv_before = torch.from_numpy(info["privileged"]).float()
            joint_vel_slice = robot_cfg["world_slices"]["joint_vel"]
            prev_action = torch.zeros(robot_cfg["action_dim"])
            prev_joint_vel = world_obs_before[joint_vel_slice[0]:joint_vel_slice[1]]

            done = False
            while not done and collected < target:
                obs_tensor = torch.from_numpy(policy_obs).float().unsqueeze(0).to(device)
                action, _ = collect_agent.act(obs_tensor)
                action_np = action.squeeze(0).cpu().numpy()

                next_policy_obs, _, term, trunc, info = env.step(action_np)
                done = term or trunc

                next_world_obs = torch.from_numpy(info["world_model_obs"]).float()
                next_priv = torch.from_numpy(info["privileged"]).float()

                # Compute reward using next state
                cmd_slice = robot_cfg["command_slice"]
                command = torch.from_numpy(policy_obs[cmd_slice[0]:cmd_slice[1]]).float().unsqueeze(0).to(device)
                reward_tensor = compute_reward(
                    world_obs=next_world_obs.unsqueeze(0).to(device),
                    privileged=next_priv.unsqueeze(0).to(device),
                    command=command,
                    prev_action=prev_action.unsqueeze(0).to(device),
                    action=action,
                    prev_joint_vel=prev_joint_vel.unsqueeze(0).to(device),
                    joint_vel=next_world_obs[joint_vel_slice[0]:joint_vel_slice[1]].unsqueeze(0).to(device),
                    step_time=env.step_time,
                    robot=robot,
                    robot_cfg=robot_cfg,
                )
                reward_val = reward_tensor.item()

                # Push transition
                pretrain_buffer.push(
                    obs=world_obs_before,
                    act=action.squeeze(0).cpu(),
                    rew=reward_val,
                    next_obs=next_world_obs,
                    priv=next_priv,
                    done=done,
                )

                collected += 1

                # Update for next step
                policy_obs = next_policy_obs
                world_obs_before = next_world_obs
                prev_action = action.squeeze(0).cpu()
                prev_joint_vel = next_world_obs[joint_vel_slice[0]:joint_vel_slice[1]]

            if collected % 100_000 == 0:
                print(f"  Collected {collected:7d}/{target} transitions")

        # Save buffer
        pretrain_buffer.save(str(pretrain_buffer_path))
        print(f" Pretrain buffer saved to {pretrain_buffer_path}")
        env.close()
    else:
        print("Skipping data collection (buffer already exists).")

    # ── Seed loop for subsequent phases ──
    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Running with seed {seed}")
        print(f"{'='*60}")
        set_seed(seed)

        seed_dir = exp_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(seed_dir / "logs"))

        # Reload environment (fresh seed for each run)
        env = IsaacEnvWrapper(robot=robot, task=config["environment"]["task"], config=config)

        # ── Phase 2: Pretrain world model (RWM) ──
        print("=== Phase 2: World‑model pretraining ===")
        # Load pretrain buffer
        pretrain_buffer = TrajectoryBuffer(
            capacity_transitions=None,
            obs_dim=robot_cfg["world_obs_dim"],
            act_dim=robot_cfg["action_dim"],
            priv_dim=robot_cfg["priv_dim"],
            device="cpu",
        )
        pretrain_buffer.load(str(pretrain_buffer_path))

        # Build RWM
        num_bin, num_cont = get_priv_split(robot)
        wm_model = RWM(
            obs_dim=robot_cfg["world_obs_dim"],
            act_dim=robot_cfg["action_dim"],
            priv_dim=robot_cfg["priv_dim"],
            num_binary_priv=num_bin,
            num_cont_priv=num_cont,
            history_len=config["world_model"]["history_len"],
            forecast_len=config["world_model"]["forecast_len"],
            gru_hidden_size=config["world_model"]["gru_hidden_size"],
            head_hidden_size=config["world_model"]["head_hidden_size"],
            activation=config["world_model"]["activation"],
        )
        wm_model.to(device)

        # Pretrain
        wm_model.pretrain(
            buffer=pretrain_buffer,
            batch_size=config["world_model"]["batch_size"],
            max_iterations=config["world_model"]["max_iterations"],
            learning_rate=config["world_model"]["learning_rate"],
            weight_decay=config["world_model"]["weight_decay"],
            device=device,
        )

        # Save world model checkpoint
        wm_ckpt_path = str(seed_dir / "rwm_pretrained.pt")
        torch.save(wm_model.state_dict(), wm_ckpt_path)
        print(f"World model saved to {wm_ckpt_path}")

        # ── Phase 3: MBPO‑PPO training ──
        print("=== Phase 3: MBPO‑PPO policy optimization ===")
        # Create a fresh PPO agent for MBRL
        policy_obs_dim = env.get_observation_space().shape[0]
        act_dim = env.get_action_space().shape[0]
        mbrl_ppo = PPOAgent(
            obs_dim=policy_obs_dim,
            act_dim=act_dim,
            hidden_dims=config["policy_network"]["hidden_dims"],
            activation=config["policy_network"]["activation"],
            lr=config["mbrl_ppo"]["learning_rate"],
            clip_range=config["mbrl_ppo"]["clip_range"],
            entropy_coef=config["mbrl_ppo"]["entropy_coef"],
            device=device,
        )

        # Instantiate the MBPO trainer
        trainer = MBPOTrainer(
            env=env,
            world_model=wm_model,
            ppo_agent=mbrl_ppo,
            config=config,
        )

        # Run training
        trainer.train(n_iterations=config["mbrl_ppo"]["max_iterations"])

        # Save final policy and world model
        policy_ckpt_path = str(seed_dir / "mbrl_policy_final.pt")
        mbrl_ppo.save(policy_ckpt_path)
        final_wm_ckpt = str(seed_dir / "rwm_fine_tuned.pt")
        torch.save(wm_model.state_dict(), final_wm_ckpt)

        # ── Phase 4: Evaluation ──
        print("=== Phase 4: Evaluation ===")
        evaluator = Evaluation(env, wm_model, mbrl_ppo, config)

        # Evaluate prediction accuracy (using held‑out episodes?)
        # For simplicity, we use a subset of the pretrain buffer as test set.
        # Create a small test buffer from first few episodes of pretrain_buffer.
        test_buffer = TrajectoryBuffer(
            capacity_transitions=None,
            obs_dim=robot_cfg["world_obs_dim"],
            act_dim=robot_cfg["action_dim"],
            priv_dim=robot_cfg["priv_dim"],
            device="cpu",
        )
        # Use first 5 episodes for evaluation (avoid contamination? acceptable for quick metric)
        for ep_idx in range(min(5, len(pretrain_buffer.episodes))):
            test_buffer.episodes.append(pretrain_buffer.episodes[ep_idx])
        test_buffer._total_transitions = sum(ep["obs"].shape[0] for ep in test_buffer.episodes)

        pred_res = evaluator.evaluate_prediction(
            buffer=test_buffer,
            num_steps=50,
            noise_std=0.0,
            max_windows=100,
        )
        print(f"  Prediction error (mean): {pred_res['mean_error']:.4f}")

        # Evaluate policy in the real simulator
        pol_res = evaluator.evaluate_policy(n_episodes=20, deterministic=True)
        print(f"  Policy tracking reward: {pol_res['mean_reward']:.3f} ± {pol_res['std_reward']:.3f}")

        # TensorBoard logging
        writer.add_scalar("eval/pred_error", pred_res["mean_error"], 0)
        writer.add_scalar("eval/mean_tracking_reward", pol_res["mean_reward"], 0)
        writer.add_scalar("eval/std_tracking_reward", pol_res["std_reward"], 0)

        writer.close()
        env.close()

    print("\nAll seeds completed.")


if __name__ == "__main__":
    main()

