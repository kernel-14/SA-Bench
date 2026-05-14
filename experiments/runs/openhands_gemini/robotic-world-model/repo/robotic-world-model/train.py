
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from collections import deque
import random
from tqdm import tqdm
from typing import Tuple, Dict, List

from config import GlobalConfig, RWMConfig, MBPOPPOConfig
from model import RoboticWorldModel, ActorCritic
from data import TrajectoryDataset, ReplayBuffer, MockRobotEnvironment

class RWMTrainer:
    """
    Handles the self-supervised autoregressive training of the Robotic World Model.
    """
    def __init__(self, cfg: GlobalConfig, rwm_model: RoboticWorldModel):
        self.cfg = cfg
        self.rwm_model = rwm_model
        self.optimizer = optim.Adam(rwm_model.parameters(),
                                    lr=cfg.rwm_config.learning_rate_rwm,
                                    weight_decay=cfg.rwm_config.weight_decay_rwm)
        self.device = cfg.device
        self.rwm_model.to(self.device)

    def _compute_rwm_loss(self,
                          predicted_obs_dists: List[torch.distributions.Normal],
                          predicted_priv_info_dists: List[torch.distributions.Normal],
                          observations_target: torch.Tensor, # (N, obs_dim)
                          privileged_info_target: torch.Tensor # (N, priv_info_dim)
        ) -> torch.Tensor:
        """
        Computes the RWM loss as described in Equation 2:
        L = (1/N) * sum_{k=1 to N} alpha^k * [L_o(o'_t+k, o_t+k) + L_c(c'_t+k, c_t+k)]
        where L_o and L_c are negative log likelihoods for Gaussian distributions.
        """
        total_loss = torch.tensor(0.0, device=self.device)
        N = self.cfg.rwm_config.forecast_horizon_N
        alpha = self.cfg.rwm_config.forecast_decay_alpha

        for k in range(N):
            # Predicted distributions for step t+k
            obs_dist_k = predicted_obs_dists[k]
            priv_info_dist_k = predicted_priv_info_dists[k]

            # True targets for step t+k
            obs_target_k = observations_target[:, k, :]
            priv_info_target_k = privileged_info_target[:, k, :]

            # Negative log likelihood loss
            loss_obs_k = -obs_dist_k.log_prob(obs_target_k).sum(dim=-1).mean()
            loss_priv_info_k = -priv_info_dist_k.log_prob(priv_info_target_k).sum(dim=-1).mean()

            # Weighted sum for this step
            step_loss = (alpha ** (k + 1)) * (loss_obs_k + loss_priv_info_k)
            total_loss += step_loss

        return total_loss / N

    def train_rwm_epoch(self, dataloader: DataLoader) -> float:
        """
        Trains the RWM for one epoch over the provided dataloader.
        """
        self.rwm_model.train()
        total_epoch_loss = 0.0
        for batch in dataloader:
            observations_history = batch["observations_history"].to(self.device)
            actions_history = batch["actions_history"].to(self.device)
            actions_forecast = batch["actions_forecast"].to(self.device)
            observations_target = batch["observations_target"].to(self.device)
            privileged_info_target = batch["privileged_info_target"].to(self.device)

            self.optimizer.zero_grad()

            predicted_obs_dists, predicted_priv_info_dists, _ = self.rwm_model(
                observations_history, actions_history, actions_forecast
            )

            loss = self._compute_rwm_loss(
                predicted_obs_dists, predicted_priv_info_dists,
                observations_target, privileged_info_target
            )
            loss.backward()
            self.optimizer.step()
            total_epoch_loss += loss.item()

        return total_epoch_loss / len(dataloader)


class MBPPOPPOAgent:
    """
    Implements the MBPO-PPO algorithm for policy optimization.
    """
    def __init__(self, cfg: GlobalConfig, actor_critic: ActorCritic, rwm_model: RoboticWorldModel):
        self.cfg = cfg
        self.actor_critic = actor_critic
        self.rwm_model = rwm_model
        self.optimizer_actor_critic = optim.Adam(actor_critic.parameters(),
                                                 lr=cfg.mbpo_ppo_config.learning_rate_mbpo_ppo,
                                                 weight_decay=cfg.mbpo_ppo_config.weight_decay_mbpo_ppo)
        self.device = cfg.device
        self.actor_critic.to(self.device)
        self.rwm_model.eval() # RWM is used for imagination, not updated here

        self.gamma = cfg.mbpo_ppo_config.discount_factor_gamma
        self.epsilon = cfg.mbpo_ppo_config.clip_range_epsilon
        self.entropy_coeff = cfg.mbpo_ppo_config.entropy_coefficient
        self.learning_epochs = cfg.mbpo_ppo_config.learning_epochs
        self.mini_batches = cfg.mbpo_ppo_config.mini_batches
        self.imagination_steps_per_iteration = cfg.mbpo_ppo_config.imagination_steps_per_iteration

        # Reward functions based on Appendix A.1.2
        # For simplicity, these are placeholders and would need actual implementation
        # based on robot state for real reward calculation.
        # For imagination, we'll assume the world model can output rewards or
        # that rewards can be computed from predicted observations.
        # The paper says: "Rewards are computed from imagined observations and privileged information."
        # We will use a mock reward function for now.

    def _compute_rewards_from_imagined_states(self,
                                                policy_obs_t: torch.Tensor, # (batch_size, policy_obs_dim)
                                                rwm_obs_t_plus_1: torch.Tensor, # (batch_size, rwm_obs_dim)
                                                rwm_priv_info_t_plus_1: torch.Tensor, # (batch_size, priv_info_dim)
                                                action_t: torch.Tensor, # (batch_size, action_dim)
                                                action_t_minus_1: torch.Tensor # (batch_size, action_dim)
        ) -> torch.Tensor:
        """
        Computes rewards from imagined observations and privileged information as described in
        Appendix A.1.2 and Table S6.
        Args:
            policy_obs_t (torch.Tensor): Policy observation at time t. Contains velocity commands.
            rwm_obs_t_plus_1 (torch.Tensor): Predicted RWM observation at time t+1.
            rwm_priv_info_t_plus_1 (torch.Tensor): Predicted privileged info at time t+1.
            action_t (torch.Tensor): Action taken at time t.
            action_t_minus_1 (torch.Tensor): Action taken at time t-1.
        Returns:
            torch.Tensor: Total reward. Shape: (batch_size, 1)
        """
        rewards = torch.zeros((rwm_obs_t_plus_1.shape[0], 1), device=self.device)
        reward_weights = self.cfg.reward_weights
        sigma_v_xy = self.cfg.rwm_config.sigma_v_xy
        sigma_omega_z = self.cfg.rwm_config.sigma_omega_z

        # Extract components from rwm_obs_t_plus_1 (Table S2)
        # Using ANYmal D as example; Unitree G1 would have different slicing.
        rwm_obs_slices = self.cfg.rwm_obs_slices
        rwm_priv_info_slices = self.cfg.rwm_priv_info_slices
        policy_obs_slices = self.cfg.policy_obs_slices

        # Extract components from rwm_obs_t_plus_1 using dynamic slicing (Table S2)
        v_xy = rwm_obs_t_plus_1[:, rwm_obs_slices['v_xy'][0]:rwm_obs_slices['v_xy'][1]]
        v_z = rwm_obs_t_plus_1[:, rwm_obs_slices['v_z'][0]:rwm_obs_slices['v_z'][1]]
        omega_xy = rwm_obs_t_plus_1[:, rwm_obs_slices['omega_xy'][0]:rwm_obs_slices['omega_xy'][1]]
        omega_z = rwm_obs_t_plus_1[:, rwm_obs_slices['omega_z'][0]:rwm_obs_slices['omega_z'][1]]
        tau = rwm_obs_t_plus_1[:, rwm_obs_slices['tau'][0]:rwm_obs_slices['tau'][1]]
        q_pos = rwm_obs_t_plus_1[:, rwm_obs_slices['q_pos'][0]:rwm_obs_slices['q_pos'][1]]

        # Placeholder for joint acceleration, not directly available in RWM obs.
        # In a real setup, this would be derived from (q_t+1 - q_t) / delta_t - q_dot_t / delta_t
        q_dot_dot = torch.zeros_like(tau)

        # Extract commands from policy_obs_t using dynamic slicing (Table S5)
        # velocity command: c (9:12) -> c_xy (9:11), c_z (11:12)
        c_xy = policy_obs_t[:, policy_obs_slices['c'][0]:policy_obs_slices['c'][0]+2] # x,y components
        c_z = policy_obs_t[:, policy_obs_slices['c'][0]+2:policy_obs_slices['c'][1]] # z component
        g_xy = policy_obs_t[:, policy_obs_slices['g'][0]:policy_obs_slices['g'][0]+2] # x,y components of projected gravity

        # --- Reward Term Calculations ---
        # 1. Linear velocity tracking x, y: r_v_xy
        r_v_xy = reward_weights['w_v_xy'] * torch.exp(-torch.norm(c_xy - v_xy, dim=-1, keepdim=True)**2 / (sigma_v_xy**2))
        rewards += r_v_xy

        # 2. Angular velocity tracking z: r_omega_z
        r_omega_z = reward_weights['w_omega_z'] * torch.exp(-torch.norm(c_z - omega_z, dim=-1, keepdim=True)**2 / (sigma_omega_z**2))
        rewards += r_omega_z

        # 3. Linear velocity z: r_v_z
        r_v_z = reward_weights['w_v_z'] * torch.norm(v_z, dim=-1, keepdim=True)**2
        rewards += r_v_z

        # 4. Angular velocity x, y: r_omega_xy
        r_omega_xy = reward_weights['w_omega_xy'] * torch.norm(omega_xy, dim=-1, keepdim=True)**2
        rewards += r_omega_xy

        # 5. Joint torque: r_q_tau
        r_q_tau = reward_weights['w_q_tau'] * torch.norm(tau, dim=-1, keepdim=True)**2
        rewards += r_q_tau

        # 6. Joint acceleration: r_q_ddot (using a proxy or assuming zero for now)
        r_q_ddot = reward_weights['w_q_ddot'] * torch.norm(q_dot_dot, dim=-1, keepdim=True)**2
        rewards += r_q_ddot

        # 7. Action rate: r_a_dot
        r_a_dot = reward_weights['w_a_dot'] * torch.norm(action_t - action_t_minus_1, dim=-1, keepdim=True)**2
        rewards += r_a_dot

        # 8. Feet air time: r_f_a (needs foot contact information, from priv_info)
        foot_contact_threshold = 0.1 # Example threshold
        r_f_a = torch.tensor(0.0, device=self.device)
        if self.cfg.robot_type == "ANYmal D":
            foot_contact_slice = rwm_priv_info_slices['foot_contact']
            foot_contacts = rwm_priv_info_t_plus_1[:, foot_contact_slice[0]:foot_contact_slice[1]]
            # Assuming if sum of contacts is less than N_feet (e.g., 4), some feet are in air
            feet_in_air = (foot_contacts < foot_contact_threshold).sum(dim=-1, keepdim=True)
            r_f_a = reward_weights['w_f_a'] * feet_in_air # This would typically be sum of time in air, here it's count
        # Unitree G1 does not have a direct 'feet air time' entry in its priv info, so we skip for now
        rewards += r_f_a

        # 9. Undesired contacts: r_c (needs body contact information, from priv_info)
        undesired_contact_threshold = 0.5
        r_c = torch.tensor(0.0, device=self.device)
        if self.cfg.robot_type == "ANYmal D":
            knee_contact_slice = rwm_priv_info_slices['knee_contact']
            knee_contacts = rwm_priv_info_t_plus_1[:, knee_contact_slice[0]:knee_contact_slice[1]]
            undesired_contacts = (knee_contacts > undesired_contact_threshold).any(dim=-1, keepdim=True).float()
            r_c = reward_weights['w_c'] * undesired_contacts
        elif self.cfg.robot_type == "Unitree G1":
            body_contact_slice = rwm_priv_info_slices['body_contact']
            body_contacts = rwm_priv_info_t_plus_1[:, body_contact_slice[0]:body_contact_slice[1]]
            undesired_contacts = (body_contacts > undesired_contact_threshold).any(dim=-1, keepdim=True).float()
            r_c = reward_weights['w_c'] * undesired_contacts
        rewards += r_c

        # 10. Flat orientation: r_g
        r_g = reward_weights['w_g'] * torch.norm(g_xy, dim=-1, keepdim=True)**2
        rewards += r_g

        # 11. Foot clearance: r_f_c (needs specific impl based on foot height, from priv_info)
        r_f_c = torch.tensor(0.0, device=self.device)
        if self.cfg.robot_type == "Unitree G1":
            foot_height_slice = rwm_priv_info_slices['foot_height']
            foot_height = rwm_priv_info_t_plus_1[:, foot_height_slice[0]:foot_height_slice[1]]
            # This would reward sufficient clearance (e.g., above 0.05m)
            r_f_c = reward_weights['w_f_c'] * torch.mean(torch.clamp(foot_height - 0.05, min=0), dim=-1, keepdim=True)
        rewards += r_f_c

        # 12. Joint deviation: r_q_d (needs q_0, default joint positions)
        q_0 = self.cfg.q0.expand_as(q_pos) # Use q0 from config, expanded to batch size
        r_q_d = reward_weights['w_q_d'] * torch.norm(q_pos - q_0, p=1, dim=-1, keepdim=True)
        rewards += r_q_d

        return rewards

    def _calculate_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, next_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates Generalized Advantage Estimation (GAE) and returns.
        Args:
            rewards (torch.Tensor): (batch_size, 1)
            values (torch.Tensor): (batch_size, 1)
            dones (torch.Tensor): (batch_size, 1)
            next_value (torch.Tensor): (batch_size, 1) or (1, 1) for last step
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Advantages and Returns
        """
        # Append next_value to values for easier computation
        values = torch.cat([values, next_value], dim=0)

        advantages = torch.zeros_like(rewards).to(self.device)
        last_gae_lam = 0
        lambda_gae = 0.95 # Common GAE lambda value, not specified in paper, assume default

        for t in reversed(range(rewards.shape[0])):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * lambda_gae * (1 - dones[t]) * last_gae_lam

        returns = advantages + values[:-1] # Remove the appended next_value

        return advantages, returns

    def _rollout_imagination_trajectories(self,
                                            initial_policy_obs_batch: torch.Tensor,
                                            rwm_hidden_state_batch: torch.Tensor
        ) -> Dict[str, torch.Tensor]:
        """
        Rolls out imagination trajectories using the RWM and the current policy.
        Args:
            initial_policy_obs_batch (torch.Tensor): Batch of initial policy observations.
                                                     Shape: (imagination_environments, policy_obs_dim)
            rwm_hidden_state_batch (torch.Tensor): Batch of initial RWM hidden states.
                                                   Shape: (1, imagination_environments, rwm_gru_hidden_size)
        Returns:
            Dict[str, torch.Tensor]: Collected imagined trajectories including
                                     policy_observations, actions, log_probs, rewards, values, dones.
        """
        self.actor_critic.eval()
        self.rwm_model.eval()

        imagined_policy_obs_list = []
        imagined_actions_list = []
        imagined_log_probs_list = []
        imagined_rewards_list = []
        imagined_values_list = []
        imagined_dones_list = []

        current_policy_obs = initial_policy_obs_batch
        current_rwm_hidden_state = rwm_hidden_state_batch

        for t in range(self.imagination_steps_per_iteration):
            # Policy acts on current policy observation
            action_dist, value = self.actor_critic(current_policy_obs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(dim=-1, keepdim=True)

            imagined_policy_obs_list.append(current_policy_obs)
            imagined_actions_list.append(action)
            imagined_log_probs_list.append(log_prob)
            imagined_values_list.append(value)

            # RWM predicts next state
            # Note: For simplicity, assume `current_policy_obs` is derived from `current_rwm_obs`
            # and `current_rwm_obs` needs to be reconstructed from `current_policy_obs` for RWM input.
            # This is a simplification, as policy_obs and RWM obs have different structures.
            # In a full implementation, there would be a mapping or RWM would predict policy_obs directly.
            # For now, we need to extract/construct the RWM obs from the policy obs.
            # Based on Table S2 and S5, RWM obs space is a subset of policy obs space + torques.
            # We'll use a placeholder `mock_rwm_obs` for now.

            # Construct mock RWM observation for the RWM from the policy observation.
            # This is a simplified mapping. In a real scenario, this would involve
            # a more precise extraction or a dedicated state estimator.
            # For now, extract components from policy_obs_t to form RWM obs.
            rwm_obs_components = []
            policy_obs_slices = self.cfg.policy_obs_slices
            rwm_obs_slices = self.cfg.rwm_obs_slices

            # Base linear and angular velocities (RWM: 0:6, Policy: 0:6)
            rwm_obs_components.append(current_policy_obs[:, 0:6])
            # Projected gravity (RWM: 6:9, Policy: 6:9)
            rwm_obs_components.append(current_policy_obs[:, 6:9])
            # Joint positions (RWM: 9:21/38, Policy: 12:24/41)
            rwm_obs_components.append(current_policy_obs[:, policy_obs_slices['q_pos'][0]:policy_obs_slices['q_pos'][1]])
            # Joint velocities (RWM: 21/38:33/67, Policy: 24/41:36/70)
            rwm_obs_components.append(current_policy_obs[:, policy_obs_slices['q_vel'][0]:policy_obs_slices['q_vel'][1]])
            # Joint torques (RWM: 33/67:45/96) - not in policy obs, so assume zero or predict
            rwm_obs_components.append(torch.zeros(current_policy_obs.shape[0], rwm_obs_slices['tau'][1] - rwm_obs_slices['tau'][0], device=self.device))
            
            mock_rwm_obs_for_rwm_input = torch.cat(rwm_obs_components, dim=-1)

            # Predict next RWM obs and privileged info
            obs_dist, priv_info_dist, next_rwm_hidden_state = self.rwm_model.predict_next_step(
                current_obs=mock_rwm_obs_for_rwm_input,
                current_action=action,
                current_hidden_state=current_rwm_hidden_state
            )
            predicted_rwm_obs = obs_dist.sample()
            predicted_priv_info = priv_info_dist.sample()

            # Compute imagined reward
            # This mock reward function needs to be replaced with the actual reward calculation
            # using predicted_rwm_obs, predicted_priv_info, action, and previous action.
            # prev_action would be `imagined_actions_list[-1]` if t > 0, else some initial_prev_action
            prev_action = imagined_actions_list[-1] if t > 0 else torch.zeros_like(action)
            reward = self._compute_rewards_from_imagined_states(current_policy_obs, predicted_rwm_obs, predicted_priv_info, action, prev_action)
            imagined_rewards_list.append(reward)

            # Determine done state (e.g., if predicted privileged info indicates collision)
            # The paper mentions: "We explicitly train RWM to predict such terminations in its privileged information prediction head."
            # "During policy optimization, MBPO-PPO treats these termination predictions as episode-ending events in imagination rollouts"
            # For simplicity, let's assume a random termination or based on some threshold in priv_info
            done = torch.zeros_like(reward, dtype=torch.bool)
            # Example: if any priv_info element exceeds a threshold, consider it a termination
            # if (predicted_priv_info > 0.5).any(dim=-1): done = torch.ones_like(reward, dtype=torch.bool)
            imagined_dones_list.append(done)

            # Update current policy observation for the next step
            # This is another critical simplification. The next policy_obs needs to be
            # constructed from `predicted_rwm_obs` and `predicted_priv_info` and other elements
            # (like velocity command, last action, etc.) as per Table S5.
            # Update current policy observation for the next step using dynamic slicing
            # This constructs the next_policy_obs from the predicted_rwm_obs and current action.
            next_policy_obs_components = []
            
            # Base linear and angular velocities (Policy: 0:6, RWM: 0:6)
            next_policy_obs_components.append(predicted_rwm_obs[:, 0:6])
            # Projected gravity (Policy: 6:9, RWM: 6:9)
            next_policy_obs_components.append(predicted_rwm_obs[:, 6:9])
            # Velocity command (Policy: 9:12) - should persist or be updated by an external command
            next_policy_obs_components.append(current_policy_obs[:, policy_obs_slices['c'][0]:policy_obs_slices['c'][1]]) # Keep the same command
            # Joint positions (Policy: 12:X, RWM: 9:Y)
            next_policy_obs_components.append(predicted_rwm_obs[:, rwm_obs_slices['q_pos'][0]:rwm_obs_slices['q_pos'][1]])
            # Joint velocities (Policy: X:Y, RWM: Y:Z)
            next_policy_obs_components.append(predicted_rwm_obs[:, rwm_obs_slices['q_vel'][0]:rwm_obs_slices['q_vel'][1]])
            # Last actions (Policy: Y:Z) - current action becomes last action for next policy_obs
            next_policy_obs_components.append(action)

            next_policy_obs = torch.cat(next_policy_obs_components, dim=-1)

            current_policy_obs = next_policy_obs
            current_rwm_hidden_state = next_rwm_hidden_state

        # Get final value for advantage calculation
        _, last_value = self.actor_critic(current_policy_obs)
        imagined_values_list.append(last_value)

        # Stack and flatten trajectories
        imagined_data = {
            "policy_observations": torch.cat(imagined_policy_obs_list, dim=0),
            "actions": torch.cat(imagined_actions_list, dim=0),
            "log_probs": torch.cat(imagined_log_probs_list, dim=0),
            "rewards": torch.cat(imagined_rewards_list, dim=0),
            "values": torch.cat(imagined_values_list[:-1], dim=0), # Exclude last value
            "dones": torch.cat(imagined_dones_list, dim=0),
            "next_value": imagined_values_list[-1]
        }
        return imagined_data


    def update_policy(self, imagined_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Updates the policy and value function using PPO.
        """
        self.actor_critic.train()

        policy_observations = imagined_data["policy_observations"]
        actions = imagined_data["actions"]
        old_log_probs = imagined_data["log_probs"]
        rewards = imagined_data["rewards"]
        values = imagined_data["values"]
        dones = imagined_data["dones"]
        next_value = imagined_data["next_value"]

        advantages, returns = self._calculate_gae(rewards, values, dones, next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Create dataset for PPO updates
        ppo_dataset = torch.utils.data.TensorDataset(
            policy_observations, actions, old_log_probs, returns, advantages
        )
        ppo_dataloader = DataLoader(ppo_dataset, batch_size=len(ppo_dataset) // self.mini_batches, shuffle=True)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy_loss = 0.0

        for epoch in range(self.learning_epochs):
            for batch_ppo in ppo_dataloader:
                obs_batch, act_batch, old_log_prob_batch, return_batch, adv_batch = batch_ppo

                action_dist, value_pred = self.actor_critic(obs_batch)
                
                # Critic loss
                critic_loss = F.mse_loss(value_pred, return_batch)

                # Actor loss
                log_prob_batch = action_dist.log_prob(act_batch).sum(dim=-1, keepdim=True)
                ratio = torch.exp(log_prob_batch - old_log_prob_batch)
                
                clip_advantage = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv_batch
                actor_loss = -torch.min(ratio * adv_batch, clip_advantage).mean()

                # Entropy loss
                entropy_loss = action_dist.entropy().mean()

                loss = actor_loss + 0.5 * critic_loss - self.entropy_coeff * entropy_loss

                self.optimizer_actor_critic.zero_grad()
                loss.backward()
                self.optimizer_actor_critic.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy_loss += entropy_loss.item()
        
        num_updates = self.learning_epochs * self.mini_batches
        return {
            "actor_loss": total_actor_loss / num_updates,
            "critic_loss": total_critic_loss / num_updates,
            "entropy_loss": total_entropy_loss / num_updates
        }

    def train(self, env: MockRobotEnvironment, replay_buffer: ReplayBuffer, rwm_trainer: RWMTrainer):
        """
        Main training loop for MBPO-PPO.
        Algorithm 1:
        1: Initialize policy πθ, world model pφ, and replay buffer D
        2: for learning iterations = 1, 2, . . . do
        3:   Collect observation-action pairs in D by interacting with the environment using πθ
        4:   Update pφ with autoregressive training using data sampled from D according to Eq. 2
        5:   Initialize imagination agents with observations sampled from D
        6:   Roll out imagination trajectories using πθ and pφ for T steps according to Eq. 3
        7:   Update πθ using PPO or another reinforcement learning algorithm end for
        """
        initial_rwm_hidden_state = torch.zeros(1, self.cfg.mbpo_ppo_config.imagination_environments,
                                               self.cfg.rwm_config.rwm_gru_hidden_size, device=self.device)

        # Initial data collection (Line 3 in Algorithm 1)
        print("Collecting initial environment data...")
        obs, priv_info, policy_obs = env.reset()
        for _ in tqdm(range(self.cfg.mbpo_ppo_config.buffer_size // 2)): # Fill half buffer initially
            action_dist, _ = self.actor_critic(torch.from_numpy(policy_obs).float().to(self.device).unsqueeze(0))
            action = action_dist.sample().squeeze(0).cpu().numpy()
            next_obs, next_priv_info, reward, done, info = env.step(action)
            replay_buffer.add(obs, action, reward, next_obs, done, priv_info, policy_obs)
            obs, priv_info, policy_obs = (next_obs, next_priv_info, info['policy_obs']) if not done else env.reset()
        print(f"Collected {replay_buffer.size} initial transitions.")

        for iteration in tqdm(range(self.cfg.mbpo_ppo_config.max_iterations_mbpo_ppo), desc="MBPO-PPO Iterations"):
            # Line 3: Collect observation-action pairs in D
            self.actor_critic.eval()
            obs, priv_info, policy_obs = env.reset()
            # Collect a small number of new transitions each iteration
            for _ in range(self.cfg.mbpo_ppo_config.imagination_environments // 10): # Example: collect 1/10th of imagination envs steps
                action_dist, _ = self.actor_critic(torch.from_numpy(policy_obs).float().to(self.device).unsqueeze(0))
                action = action_dist.sample().squeeze(0).cpu().numpy()
                next_obs, next_priv_info, reward, done, info = env.step(action)
                replay_buffer.add(obs, action, reward, next_obs, done, priv_info, policy_obs)
                obs, priv_info, policy_obs = (next_obs, next_priv_info, info['policy_obs']) if not done else env.reset()

            # Line 4: Update pφ (RWM)
            print(f"  Updating RWM at iteration {iteration}...")
            rwm_observations, rwm_actions, rwm_priv_info = replay_buffer.get_all_rwm_data()
            rwm_dataset = TrajectoryDataset(self.cfg, rwm_observations, rwm_actions, rwm_priv_info)
            rwm_dataloader = DataLoader(rwm_dataset, batch_size=self.cfg.rwm_config.batch_size_rwm, shuffle=True)
            rwm_loss = rwm_trainer.train_rwm_epoch(rwm_dataloader)
            print(f"  RWM Epoch Loss: {rwm_loss:.4f}")

            # Line 5: Initialize imagination agents with observations sampled from D
            # Sample initial policy observations from the replay buffer
            # And also prepare initial RWM hidden states (e.g., all zeros or from a context encoder)
            sampled_transitions = replay_buffer.sample(self.cfg.mbpo_ppo_config.imagination_environments)
            initial_policy_obs_batch = sampled_transitions["policy_observations"].to(self.device)
            # For a proper RWM hidden state initialization, one would process the history
            # leading up to these `initial_policy_obs_batch`. For now, we use a zero tensor.
            # In a real setup, `initial_rwm_hidden_state` would be obtained by processing a history
            # from the real environment for each imagination agent.
            
            # Line 6: Roll out imagination trajectories
            print(f"  Rolling out imagination trajectories at iteration {iteration}...")
            imagined_data = self._rollout_imagination_trajectories(initial_policy_obs_batch, initial_rwm_hidden_state)
            
            # Line 7: Update πθ using PPO
            print(f"  Updating policy with PPO at iteration {iteration}...")
            ppo_losses = self.update_policy(imagined_data)
            print(f"  PPO Losses - Actor: {ppo_losses['actor_loss']:.4f}, Critic: {ppo_losses['critic_loss']:.4f}, Entropy: {ppo_losses['entropy_loss']:.4f}")

            # Optionally, evaluate policy performance periodically (not explicitly in Algorithm 1, but good practice)


if __name__ == "__main__":
    cfg = GlobalConfig()
    cfg.parse_args() # Parse command line arguments if any

    print(f"Using device: {cfg.device}")
    print(f"Robot Type: {cfg.robot_type}")

    # Initialize RWM
    rwm_model = RoboticWorldModel(cfg)
    rwm_trainer = RWMTrainer(cfg, rwm_model)
    print("RWM model and trainer initialized.")

    # Initialize Actor-Critic (Policy and Value Function)
    actor_critic_model = ActorCritic(cfg)
    print("Actor-Critic model initialized.")

    # Initialize Replay Buffer
    replay_buffer = ReplayBuffer(
        capacity=cfg.mbpo_ppo_config.buffer_size,
        obs_dim=cfg.rwm_obs_dim,
        action_dim=cfg.rwm_action_dim,
        priv_info_dim=cfg.rwm_priv_info_dim,
        policy_obs_dim=cfg.policy_obs_dim
    )
    print("Replay buffer initialized.")

    # Initialize Mock Environment
    env = MockRobotEnvironment(cfg)
    print("Mock environment initialized.")

    # Initialize MBPO-PPO Agent
    mbpo_ppo_agent = MBPPOPPOAgent(cfg, actor_critic_model, rwm_model)
    print("MBPO-PPO agent initialized.")

    # Start training
    mbpo_ppo_agent.train(env, replay_buffer, rwm_trainer)

    print("\nTraining complete!")
