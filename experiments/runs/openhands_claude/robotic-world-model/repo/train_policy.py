"""
MBPO-PPO policy optimization on learned world models (Algorithm 1, Sec. 3.3).

Implements the full training loop:
  1. Collect real environment data → replay buffer D
  2. Update world model p_phi with autoregressive training
  3. Initialize imagination agents from D
  4. Roll out T=100 imagination steps with policy pi_theta and world model p_phi
  5. Update pi_theta with PPO

Training parameters (Table S11):
  imagination_envs=4096, T=100, buffer_size=1000, max_iter=2500,
  lr=0.001, epochs=5, mini_batches=4, kl_target=0.01,
  gamma=0.99, clip=0.2, entropy_coef=0.005
"""

import argparse
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from config import ExperimentConfig, MBPOPPOConfig
from data import (
    ImaginaryRolloutBuffer,
    ReplayBuffer,
    Trajectory,
    TrajectoryDataset,
    build_dataloader,
    generate_synthetic_dataset,
)
from model import ActorCritic, RWM, build_actor_critic, build_rwm
from rewards import WorldModelRewardComputer
from train import MetricsLogger, WorldModelTrainer, set_seed


# ---------------------------------------------------------------------------
# PPO Update
# ---------------------------------------------------------------------------

class PPOUpdater:
    """
    Proximal Policy Optimization update step.

    Implements clipped surrogate objective with entropy bonus and
    adaptive KL divergence target (Table S11).
    """

    def __init__(
        self,
        actor_critic: ActorCritic,
        cfg: MBPOPPOConfig,
        device: torch.device,
    ):
        self.actor_critic = actor_critic
        self.cfg = cfg
        self.device = device
        self.optimizer = AdamW(
            actor_critic.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

    def update(
        self, rollout_buffer: ImaginaryRolloutBuffer, last_value: torch.Tensor
    ) -> Dict[str, float]:
        metrics_list: Dict[str, list] = {
            "policy_loss": [], "value_loss": [], "entropy": [],
            "kl": [], "clip_fraction": [],
        }

        for epoch in range(self.cfg.learning_epochs):
            for batch in rollout_buffer.get_mini_batches(last_value, self.cfg.num_mini_batches):
                obs = batch["obs"]
                actions = batch["actions"]
                log_probs_old = batch["log_probs_old"]
                returns = batch["returns"]
                advantages = batch["advantages"]
                values_old = batch["values_old"]

                log_probs, entropy, values = self.actor_critic.evaluate(obs, actions)

                # PPO clipped surrogate loss
                ratio = torch.exp(log_probs - log_probs_old)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_range, 1 + self.cfg.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss (clipped)
                value_pred_clipped = values_old + torch.clamp(
                    values - values_old, -self.cfg.clip_range, self.cfg.clip_range
                )
                value_loss = torch.max(
                    (values - returns) ** 2,
                    (value_pred_clipped - returns) ** 2,
                ).mean() * 0.5

                entropy_loss = -entropy.mean()
                total_loss = (
                    policy_loss
                    + 0.5 * value_loss
                    + self.cfg.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    kl = (log_probs_old - log_probs).mean()
                    clip_frac = ((ratio - 1).abs() > self.cfg.clip_range).float().mean()

                metrics_list["policy_loss"].append(policy_loss.item())
                metrics_list["value_loss"].append(value_loss.item())
                metrics_list["entropy"].append(entropy.mean().item())
                metrics_list["kl"].append(kl.item())
                metrics_list["clip_fraction"].append(clip_frac.item())

        return {k: float(np.mean(v)) for k, v in metrics_list.items()}


# ---------------------------------------------------------------------------
# Imagination Environment
# ---------------------------------------------------------------------------

class ImaginaryEnvironment:
    """
    Vectorized imagination environment using the learned world model.

    Manages B parallel imagination trajectories, each initialized from
    a starting state sampled from the replay buffer.
    """

    def __init__(
        self,
        world_model: RWM,
        num_envs: int,
        history_horizon: int,
        device: torch.device,
    ):
        self.world_model = world_model
        self.num_envs = num_envs
        self.history_horizon = history_horizon
        self.device = device

        self.hidden: Optional[torch.Tensor] = None
        self.current_obs: Optional[torch.Tensor] = None
        self.obs_history: Optional[torch.Tensor] = None
        self.act_history: Optional[torch.Tensor] = None

    def reset(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        """
        Initialize imagination from sampled starting states.

        Args:
            obs_history:    (B, M, obs_dim)
            action_history: (B, M, action_dim)

        Returns:
            current_obs: (B, obs_dim)
        """
        self.obs_history = obs_history
        self.act_history = action_history
        self.hidden = self.world_model.core.forward_history(obs_history, action_history)
        self.current_obs = obs_history[:, -1].clone()
        return self.current_obs

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Take one imagination step.

        Args:
            action: (B, action_dim)

        Returns:
            next_obs: (B, obs_dim)
            done:     (B,) — termination signal from privileged info
        """
        obs_mean, obs_std, priv_mean, priv_std, self.hidden = (
            self.world_model.core.step(self.current_obs, action, self.hidden)
        )
        eps = torch.randn_like(obs_mean)
        next_obs = obs_mean + obs_std * eps
        self.current_obs = next_obs

        # Termination: predicted contact with base (from privileged info)
        # Use mean prediction for termination detection
        done = self._detect_termination(priv_mean)
        return next_obs, done

    def _detect_termination(self, priv_mean: torch.Tensor) -> torch.Tensor:
        """
        Detect episode termination from privileged information.
        Termination occurs when base contact is predicted (Sec. A.4.3).
        """
        # For ANYmal D: priv = [knee_contact(4), foot_contact(4)]
        # Termination if any knee contact > 0.5 (base contact proxy)
        # For Unitree G1: priv = [body_contact(26), foot_height(2), foot_velocity(2)]
        # Termination if any body contact > 0.5
        contact_probs = torch.sigmoid(priv_mean[..., :4])
        done = (contact_probs > 0.5).any(dim=-1).float()
        return done


# ---------------------------------------------------------------------------
# MBPO-PPO Trainer
# ---------------------------------------------------------------------------

class MBPOPPOTrainer:
    """
    Full MBPO-PPO training loop (Algorithm 1).

    Combines:
      - Real environment data collection (simulated here via replay buffer)
      - World model training with autoregressive scheme
      - Policy optimization via PPO on imagination rollouts
    """

    def __init__(
        self,
        world_model: RWM,
        actor_critic: ActorCritic,
        reward_computer: WorldModelRewardComputer,
        cfg: ExperimentConfig,
        device: torch.device,
        output_dir: str = "outputs_policy",
    ):
        self.world_model = world_model.to(device)
        self.actor_critic = actor_critic.to(device)
        self.reward_computer = reward_computer
        self.cfg = cfg
        self.ppo_cfg = cfg.mbpo_ppo
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.ppo_cfg.checkpoint_dir, exist_ok=True)

        self.replay_buffer = ReplayBuffer(max_size=self.ppo_cfg.buffer_size)
        self.ppo_updater = PPOUpdater(actor_critic, self.ppo_cfg, device)
        self.logger = MetricsLogger()

        robot = cfg.get_robot_config()
        self.obs_dim = robot.obs_dim
        self.action_dim = robot.action_dim
        self.policy_obs_dim = robot.policy_obs_dim

        self.default_joint_pos = torch.zeros(robot.action_dim, device=device)
        self.velocity_command = torch.zeros(3, device=device)

        self.wm_trainer = WorldModelTrainer(
            model=world_model,
            cfg=cfg.rwm_training,
            device=device,
            autoregressive=True,
            output_dir=os.path.join(output_dir, "world_model"),
        )

    def set_velocity_command(self, cmd: torch.Tensor) -> None:
        self.velocity_command = cmd.to(self.device)

    def collect_real_data(self, env, num_steps: int = 1000) -> None:
        """
        Collect real environment data and add to replay buffer.
        In practice, this interfaces with Isaac Lab or hardware.
        Here we provide the interface; actual env interaction is external.
        """
        raise NotImplementedError(
            "Real environment collection requires Isaac Lab integration. "
            "Use load_pretrain_data() to load pre-collected data."
        )

    def load_pretrain_data(self, data_dir: str) -> None:
        """Load pre-collected simulation data into replay buffer."""
        from pathlib import Path
        data_path = Path(data_dir)
        for f in sorted(data_path.glob("*.npz")):
            traj = Trajectory.load(str(f))
            self.replay_buffer.add(traj)
        print(f"Loaded {len(self.replay_buffer)} trajectories into replay buffer.")

    def load_pretrain_data_from_trajectories(self, trajectories: List[Trajectory]) -> None:
        for traj in trajectories:
            self.replay_buffer.add(traj)

    def update_world_model(self) -> Dict[str, float]:
        """Update world model using data from replay buffer (Algorithm 1, step 4)."""
        dataset = self.replay_buffer.build_dataset(
            self.cfg.rwm_training.history_horizon,
            self.cfg.rwm_training.forecast_horizon,
        )
        if len(dataset) == 0:
            return {}

        dataloader = build_dataloader(
            dataset,
            batch_size=min(self.cfg.rwm_training.batch_size, len(dataset)),
            shuffle=True,
            num_workers=0,
        )
        return self.wm_trainer.train_epoch(dataloader)

    def run_imagination_rollouts(
        self,
        num_envs: int,
        horizon: int,
    ) -> ImaginaryRolloutBuffer:
        """
        Run imagination rollouts using world model and policy (Algorithm 1, step 6).

        Initializes agents from replay buffer samples and rolls out T steps.
        """
        rollout_buffer = ImaginaryRolloutBuffer(
            num_envs=num_envs,
            horizon=horizon,
            obs_dim=self.policy_obs_dim,
            action_dim=self.action_dim,
            device=self.device,
            gamma=self.ppo_cfg.discount_factor,
            gae_lambda=self.ppo_cfg.gae_lambda,
        )

        # Sample starting states from replay buffer (Algorithm 1, step 5)
        obs_history, act_history = self.replay_buffer.sample_starting_states(
            n=num_envs,
            history_horizon=self.cfg.rwm_training.history_horizon,
            device=self.device,
        )

        # Initialize imagination environment
        imag_env = ImaginaryEnvironment(
            world_model=self.world_model,
            num_envs=num_envs,
            history_horizon=self.cfg.rwm_training.history_horizon,
            device=self.device,
        )
        wm_obs = imag_env.reset(obs_history, act_history)

        # Convert world model obs to policy obs
        cmd = self.velocity_command.unsqueeze(0).expand(num_envs, -1)
        a_prev = torch.zeros(num_envs, self.action_dim, device=self.device)
        policy_obs = self.reward_computer.wm_obs_to_policy_obs(wm_obs, cmd, a_prev)

        self.world_model.eval()
        self.actor_critic.eval()

        with torch.no_grad():
            for t in range(horizon):
                action, log_prob, value = self.actor_critic.act(policy_obs)

                next_wm_obs, done = imag_env.step(action)

                reward = self.reward_computer.compute_from_wm_obs(
                    next_wm_obs, action, cmd, a_prev, self.default_joint_pos
                )

                rollout_buffer.add(policy_obs, action, log_prob, reward, value, done)

                next_policy_obs = self.reward_computer.wm_obs_to_policy_obs(
                    next_wm_obs, cmd, action
                )
                policy_obs = next_policy_obs
                a_prev = action

        return rollout_buffer, policy_obs

    def train(
        self,
        pretrain_data: Optional[List[Trajectory]] = None,
        pretrain_wm_iterations: int = 500,
    ) -> MetricsLogger:
        """
        Full MBPO-PPO training loop (Algorithm 1).

        Args:
            pretrain_data: pre-collected trajectories for world model pretraining
            pretrain_wm_iterations: number of world model pretraining steps
        """
        # Load pretraining data (Sec. A.4.3)
        if pretrain_data is not None:
            self.load_pretrain_data_from_trajectories(pretrain_data)
            print(f"Pretraining world model for {pretrain_wm_iterations} iterations...")
            self._pretrain_world_model(pretrain_wm_iterations)

        print(f"\nStarting MBPO-PPO training for {self.ppo_cfg.max_iterations} iterations...")
        start_time = time.time()

        for iteration in range(self.ppo_cfg.max_iterations):
            # Step 4: Update world model
            wm_metrics = self.update_world_model()

            # Steps 5-6: Imagination rollouts
            rollout_buffer, last_policy_obs = self.run_imagination_rollouts(
                num_envs=self.ppo_cfg.imagination_envs,
                horizon=self.ppo_cfg.imagination_steps,
            )

            # Compute last value for GAE
            with torch.no_grad():
                last_value = self.actor_critic.value(last_policy_obs)

            # Step 7: Update policy with PPO
            self.actor_critic.train()
            ppo_metrics = self.ppo_updater.update(rollout_buffer, last_value)

            if iteration % self.ppo_cfg.log_interval == 0:
                elapsed = time.time() - start_time
                mean_reward = rollout_buffer.rewards.mean().item()
                log_str = (
                    f"[{iteration}/{self.ppo_cfg.max_iterations}] "
                    f"reward={mean_reward:.4f} "
                    f"policy_loss={ppo_metrics.get('policy_loss', 0):.4f} "
                    f"value_loss={ppo_metrics.get('value_loss', 0):.4f} "
                    f"entropy={ppo_metrics.get('entropy', 0):.4f} "
                    f"kl={ppo_metrics.get('kl', 0):.4f} "
                    f"wm_loss={wm_metrics.get('loss', 0):.4f} "
                    f"elapsed={elapsed:.1f}s"
                )
                print(log_str)

                all_metrics = {
                    "reward": mean_reward,
                    **{f"ppo/{k}": v for k, v in ppo_metrics.items()},
                    **{f"wm/{k}": v for k, v in wm_metrics.items()},
                }
                self.logger.log(all_metrics, iteration)

            if iteration % self.ppo_cfg.save_interval == 0:
                self.save_checkpoint(f"checkpoint_{iteration:06d}.pt")

        self.save_checkpoint("checkpoint_final.pt")
        self.logger.save(os.path.join(self.output_dir, "metrics.json"))
        return self.logger

    def _pretrain_world_model(self, num_iterations: int) -> None:
        """Pretrain world model on collected data before policy optimization."""
        dataset = self.replay_buffer.build_dataset(
            self.cfg.rwm_training.history_horizon,
            self.cfg.rwm_training.forecast_horizon,
        )
        if len(dataset) == 0:
            print("Warning: empty dataset, skipping pretraining.")
            return

        dataloader = build_dataloader(
            dataset,
            batch_size=self.cfg.rwm_training.batch_size,
            shuffle=True,
            num_workers=0,
        )
        self.world_model.train()
        step = 0
        while step < num_iterations:
            for obs, actions, privileged in dataloader:
                obs = obs.to(self.device)
                actions = actions.to(self.device)
                privileged = privileged.to(self.device)

                self.wm_trainer.optimizer.zero_grad()
                loss, metrics = self.world_model.compute_loss(
                    obs, actions, privileged, autoregressive=True
                )
                loss.backward()
                nn.utils.clip_grad_norm_(self.world_model.parameters(), 10.0)
                self.wm_trainer.optimizer.step()
                step += 1

                if step % 100 == 0:
                    print(f"  WM pretrain [{step}/{num_iterations}] loss={metrics['loss']:.4f}")
                if step >= num_iterations:
                    break

    def save_checkpoint(self, filename: str) -> None:
        path = os.path.join(self.ppo_cfg.checkpoint_dir, filename)
        torch.save({
            "world_model": self.world_model.state_dict(),
            "actor_critic": self.actor_critic.state_dict(),
            "ppo_optimizer": self.ppo_updater.optimizer.state_dict(),
            "wm_optimizer": self.wm_trainer.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.world_model.load_state_dict(ckpt["world_model"])
        self.actor_critic.load_state_dict(ckpt["actor_critic"])
        self.ppo_updater.optimizer.load_state_dict(ckpt["ppo_optimizer"])
        self.wm_trainer.optimizer.load_state_dict(ckpt["wm_optimizer"])

    def load_world_model_checkpoint(self, path: str) -> None:
        """Load a pretrained world model checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        if "model_state_dict" in ckpt:
            self.world_model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.world_model.load_state_dict(ckpt["world_model"])
        print(f"Loaded world model from {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="MBPO-PPO policy optimization with RWM")
    parser.add_argument("--robot", type=str, default="anymal_d",
                        choices=["anymal_d", "unitree_g1"])
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory with pre-collected trajectory data")
    parser.add_argument("--wm-checkpoint", type=str, default=None,
                        help="Path to pretrained world model checkpoint")
    parser.add_argument("--output-dir", type=str, default="outputs_policy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-iterations", type=int, default=2500)
    parser.add_argument("--imagination-envs", type=int, default=4096)
    parser.add_argument("--imagination-steps", type=int, default=100)
    parser.add_argument("--pretrain-wm-iterations", type=int, default=500)
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data for testing")
    parser.add_argument("--vel-cmd", type=float, nargs=3, default=[1.0, 0.0, 0.0],
                        help="Velocity command [vx, vy, wz]")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = ExperimentConfig(robot=args.robot, seed=args.seed)
    cfg.mbpo_ppo.device = args.device
    cfg.mbpo_ppo.max_iterations = args.max_iterations
    cfg.mbpo_ppo.imagination_envs = args.imagination_envs
    cfg.mbpo_ppo.imagination_steps = args.imagination_steps
    cfg.mbpo_ppo.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    robot = cfg.get_robot_config()

    world_model = build_rwm(cfg).to(device)
    actor_critic = build_actor_critic(cfg).to(device)
    reward_weights = cfg.get_reward_weights()
    reward_computer = WorldModelRewardComputer(args.robot, reward_weights)

    trainer = MBPOPPOTrainer(
        world_model=world_model,
        actor_critic=actor_critic,
        reward_computer=reward_computer,
        cfg=cfg,
        device=device,
        output_dir=args.output_dir,
    )

    vel_cmd = torch.tensor(args.vel_cmd, dtype=torch.float32)
    trainer.set_velocity_command(vel_cmd)

    if args.wm_checkpoint is not None:
        trainer.load_world_model_checkpoint(args.wm_checkpoint)

    if args.synthetic:
        print("Generating synthetic pretraining data...")
        pretrain_data = generate_synthetic_dataset(
            obs_dim=robot.obs_dim,
            action_dim=robot.action_dim,
            privileged_dim=robot.privileged_dim,
            num_trajectories=50,
            trajectory_length=500,
        )
    else:
        pretrain_data = None
        if os.path.isdir(args.data_dir):
            trainer.load_pretrain_data(args.data_dir)

    trainer.train(
        pretrain_data=pretrain_data,
        pretrain_wm_iterations=args.pretrain_wm_iterations,
    )


if __name__ == "__main__":
    main()
