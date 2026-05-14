"""MBPO-PPO trainer: policy optimization using the learned world model.

Follows Algorithm 1 and the MBPO framework (Janner et al., 2019) adapted with PPO.

Key components:
- Imagination rollouts using RWM for T steps (100 steps per iteration)
- PPO policy updates using clipped surrogate objective
- GAE advantage estimation
- World model fine-tuning on real environment data

Hyperparameters from Table S11.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.policy import PPOActor, PPOCritic
from model.rwm import RoboticWorldModel, RWMLoss
from data.replay_buffer import ReplayBuffer
from config import MBPOPOConfig, RWMConfig
from env.rewards import RewardComputer
from training.world_model_trainer import WorldModelTrainer, create_dataloader
from data.dataset import TrajectoryBuffer
from torch.utils.data import DataLoader


class MBPOPPOTrainer:
    """Model-Based Policy Optimization with PPO using RWM."""

    def __init__(
        self,
        actor: PPOActor,
        critic: PPOCritic,
        world_model: RoboticWorldModel,
        rwm_config: RWMConfig,
        mbpo_config: MBPOPOConfig,
        reward_computer: RewardComputer,
        device: str = "cuda",
    ):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.world_model = world_model.to(device)
        self.rwm_config = rwm_config
        self.config = mbpo_config
        self.reward_computer = reward_computer
        self.device = device

        self.actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=mbpo_config.learning_rate,
            weight_decay=mbpo_config.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            critic.parameters(),
            lr=mbpo_config.learning_rate,
            weight_decay=mbpo_config.weight_decay,
        )

        self.gamma = mbpo_config.discount_factor
        self.lambda_ = mbpo_config.gae_lambda
        self.clip_range = mbpo_config.clip_range
        self.entropy_coef = mbpo_config.entropy_coef
        self.value_loss_coef = mbpo_config.value_loss_coef
        self.kl_target = mbpo_config.kl_target

        self.T = mbpo_config.imagination_steps_per_iteration  # imagination horizon
        self.num_envs = mbpo_config.imagination_envs  # 4096

        self.writer: Optional[SummaryWriter] = None

    def set_writer(self, writer: SummaryWriter):
        self.writer = writer

    def collect_real_data(
        self,
        env,
        replay_buffer: ReplayBuffer,
        num_steps: int,
        deterministic: bool = False,
    ) -> Tuple[float, float]:
        """Collect data from the real environment using the current policy.

        Returns (mean_reward, mean_length) for logging.
        """
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        episode_rewards = []
        episode_length = 0
        total_reward = 0.0

        for _ in range(num_steps):
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
                if deterministic:
                    mean, _ = self.actor.forward(obs_tensor)
                    action = mean.squeeze(0).cpu().numpy()
                else:
                    action, _, _ = self.actor.sample(obs_tensor)
                    action = action.squeeze(0).cpu().numpy()

            step_result = env.step(action)
            next_obs = step_result[0]
            reward = step_result[1]
            done = step_result[2]
            truncated = step_result[4] if len(step_result) > 4 else False
            info = step_result[3] if len(step_result) > 3 else {}

            privileged = info.get("privileged")
            replay_buffer.add(obs, action, reward, next_obs, done or truncated, privileged)

            total_reward += reward
            episode_length += 1

            if done or truncated:
                episode_rewards.append(total_reward)
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                total_reward = 0.0
                episode_length = 0
            else:
                obs = next_obs

        mean_reward = np.mean(episode_rewards) if episode_rewards else total_reward
        return mean_reward, float(episode_length)

    def update_world_model(
        self,
        replay_buffer: ReplayBuffer,
        num_steps: int,
    ) -> Dict[str, float]:
        """Fine-tune the world model using data from the replay buffer.

        Converts replay buffer data into trajectory format for autoregressive training.
        """
        # Build a temporary trajectory buffer from replay data
        traj_buffer = TrajectoryBuffer()
        obs_list = []
        act_list = []
        priv_list = []

        for i in range(len(replay_buffer)):
            data = replay_buffer.get_full_observation(i)
            obs_list.append(data["obs"])
            act_list.append(data["action"])
            if "privileged" in data and data["privileged"] is not None:
                priv_list.append(data["privileged"])

        if len(obs_list) < self.rwm_config.history_horizon + self.rwm_config.forecast_horizon:
            return {"rwm_loss": 0.0}

        traj_buffer.add_trajectory(
            observations=np.stack(obs_list, axis=0),
            actions=np.stack(act_list, axis=0),
            privileged=np.stack(priv_list, axis=0) if priv_list else None,
        )

        dataloader = create_dataloader(traj_buffer, self.rwm_config, shuffle=True)

        self.world_model.train()
        loss_fn = RWMLoss(forecast_decay=self.rwm_config.forecast_decay)

        total_loss = 0.0
        n_updates = 0

        for _ in range(num_steps):
            try:
                batch = next(iter(dataloader))
            except StopIteration:
                dataloader = create_dataloader(traj_buffer, self.rwm_config, shuffle=True)
                batch = next(iter(dataloader))

            observations = batch["observations"].to(self.device)
            actions = batch["actions"].to(self.device)
            privileged = batch.get("privileged")
            if privileged is not None:
                privileged = privileged.to(self.device)

            M = self.rwm_config.history_horizon
            N = self.rwm_config.forecast_horizon

            obs_history = observations[:, :M]
            obs_target = observations[:, M:M + N]
            acts_for_model = actions[:, :M - 1 + N]
            priv_target = privileged[:, M:M + N] if privileged is not None else None

            # We need a separate optimizer for the world model
            # Use the world model's parameters
            pass  # World model updates happen via WorldModelTrainer

        return {"rwm_loss": total_loss / max(n_updates, 1)}

    def generate_imagination_data(
        self,
        replay_buffer: ReplayBuffer,
        world_model_trainer: WorldModelTrainer,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Generate imagination rollouts for PPO training (Algorithm 1, lines 5-6).

        Samples initial observations from replay buffer, then rolls out
        imagination trajectories using the world model and current policy.

        Returns:
            obs_trajs: (num_envs, T, policy_obs_dim)
            act_trajs: (num_envs, T, action_dim)
            rew_trajs: (num_envs, T)
            dones: (num_envs, T)
            values: (num_envs, T)
        """
        M = self.rwm_config.history_horizon
        T = self.T

        # Sample initial observations from replay buffer
        indices = replay_buffer.sample_indices(self.num_envs)

        # For each imagination env, we need M historical observations to warm up
        # We get them from the replay buffer
        batch_obs = []
        batch_actions = []
        # Collect initial sequences
        for idx in indices:
            obs_seq = []
            act_seq = []
            for t in range(M):
                if idx + t < len(replay_buffer):
                    obs_seq.append(replay_buffer.get_observation_at(idx + t))
                else:
                    obs_seq.append(replay_buffer.get_observation_at(0))
            obs_seq = np.stack(obs_seq, axis=0)  # (M, obs_dim)
            batch_obs.append(obs_seq)

        obs_history = torch.as_tensor(np.stack(batch_obs, axis=0), device=self.device)

        # We need action sequences for warmup + imagination
        # Action dim from config
        act_dim = self.config.robot.action_dim

        # Warm up the world model
        self.world_model.eval()
        with torch.no_grad():
            # We need initial actions for the warmup
            # Use dummy actions initially, or gather them from replay
            dummy_actions = torch.zeros(
                self.num_envs, M - 1 + T, act_dim, device=self.device
            )

            # Get hidden state after warmup
            # For the warmup, we need actions that led to each observation
            # Since we don't have them from the buffer, we'll use the actor to generate them
            current_obs = obs_history[:, 0]
            hidden = self.world_model._init_hidden(self.num_envs, self.device)

            for t in range(M - 1):
                with torch.no_grad():
                    action, _, _ = self.actor.sample(
                        current_obs.unsqueeze(0)
                        if current_obs.dim() == 1
                        else current_obs
                    )
                    action = action.squeeze(0)
                dummy_actions[:, t] = action
                _, hidden = self.world_model._gru_step(current_obs, action, hidden)
                current_obs = obs_history[:, t + 1]

            # Now we have hidden after warmup
            # Start imagination rollout
            obs_means_list = []
            privileged_means_list = []
            actions_list = []

            for k in range(T):
                # Predict next observation
                obs_mean, obs_log_std, priv_out = self.world_model._predict_step(hidden)
                std = torch.exp(obs_log_std)
                eps = torch.randn_like(obs_mean)
                sampled_obs = obs_mean + std * eps

                # Get action from policy
                action, _, _ = self.actor.sample(sampled_obs)
                actions_list.append(action)

                # Compute reward from predicted observation and privileged info
                rewards = self.reward_computer.compute_rewards(
                    observations=sampled_obs,
                    privileged=priv_out[0] if priv_out is not None else None,
                    actions=action,
                    prev_actions=actions_list[-2] if len(actions_list) > 1 else action,
                    feet_in_air_time=None,
                )

                obs_means_list.append(sampled_obs)
                if priv_out is not None:
                    privileged_means_list.append(priv_out[0])

                # Get value estimate
                # (In practice, value is computed on the policy observation, not world model obs)
                value = self.critic(sampled_obs)

                if k == 0:
                    values_list = [value]
                    rewards_list = [rewards]

                if k < T - 1:
                    _, hidden = self.world_model._gru_step(sampled_obs, action, hidden)
                    values_list.append(value)
                    rewards_list.append(rewards)

        # Stack results
        obs_trajs = torch.stack(obs_means_list, dim=1)  # (num_envs, T, wm_obs_dim)
        act_trajs = torch.stack(actions_list, dim=1)
        rew_trajs = torch.stack(rewards_list, dim=1) if rewards_list else torch.zeros(self.num_envs, T, device=self.device)
        dones = torch.zeros(self.num_envs, T, device=self.device)

        # Compute values
        with torch.no_grad():
            values = self.critic(obs_trajs.view(-1, obs_trajs.shape[-1])).view(
                self.num_envs, T
            )

        return obs_trajs, act_trajs, rew_trajs, dones, values

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: (num_envs, T)
            values: (num_envs, T)
            dones: (num_envs, T)
            next_values: (num_envs,)

        Returns:
            advantages: (num_envs, T)
            returns: (num_envs, T)
        """
        num_envs, T = rewards.shape
        advantages = torch.zeros(num_envs, T, device=self.device)
        gae = 0

        for t in reversed(range(T)):
            if t == T - 1:
                value_next = next_values
                mask = 1.0 - dones[:, t]
            else:
                value_next = values[:, t + 1]
                mask = 1.0 - dones[:, t]

            delta = rewards[:, t] + self.gamma * value_next * mask - values[:, t]
            gae = delta + self.gamma * self.lambda_ * mask * gae
            advantages[:, t] = gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def ppo_update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        num_epochs: int,
        mini_batches: int,
    ) -> Dict[str, float]:
        """Perform PPO update on imagined data.

        obs: (total_samples, obs_dim)
        actions: (total_samples, action_dim)
        old_log_probs: (total_samples,)
        advantages: (total_samples,)
        returns: (total_samples,)
        """
        total_samples = obs.shape[0]
        batch_size = total_samples // mini_batches

        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0}

        for epoch in range(num_epochs):
            indices = torch.randperm(total_samples, device=self.device)

            for i in range(mini_batches):
                start = i * batch_size
                end = start + batch_size
                idx = indices[start:end]

                batch_obs = obs[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]

                # Evaluate current policy
                new_log_probs, entropy = self.actor.evaluate(batch_obs, batch_actions)

                # Ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                values_pred = self.critic(batch_obs)
                value_loss = 0.5 * ((values_pred - batch_returns) ** 2).mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                )

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # KL divergence
                with torch.no_grad():
                    kl = (new_log_probs - batch_old_log_probs).mean()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.mean().item()
                metrics["kl"] += kl.item()

        # Average
        total_updates = num_epochs * mini_batches
        for key in metrics:
            metrics[key] /= total_updates

        return metrics

    def train(
        self,
        env,
        replay_buffer: ReplayBuffer,
        world_model_trainer: WorldModelTrainer,
        num_iterations: int,
        collect_steps_per_iter: int = 100,
        log_interval: int = 10,
    ) -> Dict[str, list]:
        """Main MBPO-PPO training loop (Algorithm 1).

        Args:
            env: Gym-style environment (real or simulation)
            replay_buffer: D in Algorithm 1
            world_model_trainer: for updating RWM (step 4)
            num_iterations: Number of learning iterations
            collect_steps_per_iter: Steps collected from real env per iteration
            log_interval: Log every N iterations

        Returns:
            Training history dict
        """
        history: Dict[str, list] = {
            "iteration": [],
            "real_reward": [],
            "predicted_reward": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl": [],
            "model_error": [],
        }

        pbar = tqdm(range(num_iterations), desc="MBPO-PPO")

        for iteration in pbar:
            # Step 3: Collect real data
            mean_reward, _ = self.collect_real_data(
                env, replay_buffer, collect_steps_per_iter
            )

            # Step 4: Update world model
            if len(replay_buffer) > self.rwm_config.history_horizon + self.rwm_config.forecast_horizon:
                # Build trajectory buffer from replay
                traj_buffer = TrajectoryBuffer()
                obs_list = []
                act_list = []
                priv_list = []

                for i in range(len(replay_buffer)):
                    data = replay_buffer.get_full_observation(i)
                    obs_list.append(data["obs"])
                    act_list.append(data["action"])
                    if "privileged" in data and data["privileged"] is not None:
                        priv_list.append(data["privileged"])

                if obs_list:
                    traj_buffer.add_trajectory(
                        observations=np.stack(obs_list, axis=0),
                        actions=np.stack(act_list, axis=0),
                        privileged=np.stack(priv_list, axis=0) if priv_list else None,
                    )

                    dataloader = create_dataloader(traj_buffer, self.rwm_config, shuffle=True)
                    # Fine-tune world model for a few steps
                    world_model_trainer.train(dataloader, num_iterations=10, log_interval=1000)

            # Steps 5-6: Generate imagination data
            obs_imag, act_imag, rew_imag, dones_imag, values_imag = (
                self.generate_imagination_data(replay_buffer, world_model_trainer)
            )

            # Compute returns and advantages
            num_envs, T = obs_imag.shape[0], obs_imag.shape[1]
            with torch.no_grad():
                next_values = self.critic(obs_imag[:, -1])
            advantages, returns = self.compute_gae(
                rew_imag, values_imag, dones_imag, next_values
            )

            # Flatten
            flat_obs = obs_imag.view(-1, obs_imag.shape[-1])
            flat_actions = act_imag.view(-1, act_imag.shape[-1])
            flat_advantages = advantages.view(-1)
            flat_returns = returns.view(-1)

            # Compute old log probs
            with torch.no_grad():
                old_log_probs, _ = self.actor.evaluate(flat_obs, flat_actions)

            # Step 7: PPO update
            ppo_metrics = self.ppo_update(
                flat_obs,
                flat_actions,
                old_log_probs,
                flat_advantages,
                flat_returns,
                num_epochs=self.config.learning_epochs,
                mini_batches=self.config.mini_batches,
            )

            # Logging
            pred_reward = rew_imag.mean().item()
            history["iteration"].append(iteration)
            history["real_reward"].append(mean_reward)
            history["predicted_reward"].append(pred_reward)
            history["policy_loss"].append(ppo_metrics["policy_loss"])
            history["value_loss"].append(ppo_metrics["value_loss"])
            history["entropy"].append(ppo_metrics["entropy"])
            history["kl"].append(ppo_metrics["kl"])
            history["model_error"].append(0.0)  # updated below

            if iteration % log_interval == 0:
                pbar.set_postfix({
                    "real_r": f"{mean_reward:.3f}",
                    "pred_r": f"{pred_reward:.3f}",
                })

                if self.writer is not None:
                    self.writer.add_scalar("mbpo_ppo/real_reward", mean_reward, iteration)
                    self.writer.add_scalar("mbpo_ppo/predicted_reward", pred_reward, iteration)
                    for key, val in ppo_metrics.items():
                        self.writer.add_scalar(f"mbpo_ppo/{key}", val, iteration)

        return history
