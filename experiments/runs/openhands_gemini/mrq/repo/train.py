
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import copy
from collections import deque

from config import Config
from model import StateEncoder, StateActionEncoder, ValueNetwork, PolicyNetwork
from data import PrioritizedReplayBuffer, RewardScaler, EnvironmentManager
from layers import LayerNorm # For type hinting or specific checks if needed

class MRQ:
    def __init__(self, config: Config, observation_space_shape, action_dim, discrete_action_space):
        self.config = config
        self.action_dim = action_dim
        self.discrete_action_space = discrete_action_space

        # Activation functions
        self.common_activation = F.elu if config.ACTIVATION_FUNCTION == "ELU" else F.relu
        self.policy_activation = F.relu if config.POLICY_ACTIVATION_FUNCTION == "ReLU" else F.elu # Paper specified ReLU for policy

        # Networks
        self.state_encoder = StateEncoder(observation_space_shape, config.ZS_DIM, self.common_activation).to(config.DEVICE)
        
        # Output dimension for state-action encoder's MDP predictor: zs_dim (for next state embedding) + 1 (reward) + 1 (terminal)
        self.state_action_encoder = StateActionEncoder(
            config.ZS_DIM, action_dim, config.ZA_DIM, config.ZSA_DIM, None, self.common_activation # output_dim is not used anymore
        ).to(config.DEVICE)
        self.target_state_encoder = copy.deepcopy(self.state_encoder).to(config.DEVICE)
        self.target_state_action_encoder = copy.deepcopy(self.state_action_encoder).to(config.DEVICE)

        self.value_net1 = ValueNetwork(config.ZSA_DIM, self.common_activation).to(config.DEVICE)
        self.value_net2 = ValueNetwork(config.ZSA_DIM, self.common_activation).to(config.DEVICE)
        self.target_value_net1 = copy.deepcopy(self.value_net1).to(config.DEVICE)
        self.target_value_net2 = copy.deepcopy(self.value_net2).to(config.DEVICE)

        self.policy_net = PolicyNetwork(config.ZS_DIM, action_dim, discrete_action_space, self.policy_activation).to(config.DEVICE)
        self.target_policy_net = copy.deepcopy(self.policy_net).to(config.DEVICE)

        # Optimizers
        encoder_params = list(self.state_encoder.parameters()) + list(self.state_action_encoder.parameters())
        self.encoder_optimizer = optim.AdamW(encoder_params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.value_optimizer = optim.AdamW(
            list(self.value_net1.parameters()) + list(self.value_net2.parameters()),
            lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
        )
        self.policy_optimizer = optim.AdamW(self.policy_net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

        # Replay Buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            config.REPLAY_BUFFER_CAPACITY, observation_space_shape, action_dim
        )
        
        self.reward_scaler = RewardScaler(reward_range=config.REWARD_RANGE, reward_bins=config.REWARD_BINS).to(config.DEVICE)
        self.average_abs_reward = 1.0 # For reward scaling in value loss

        self.total_timesteps = 0

    def select_action(self, obs, evaluate=False):
        obs_tensor = torch.FloatTensor(obs).to(self.config.DEVICE).unsqueeze(0)
        zs = self.state_encoder(obs_tensor)
        
        with torch.no_grad():
            action_output, _ = self.policy_net(zs)
        
        if self.discrete_action_space:
            if not evaluate:
                # Add exploration noise (Gumbel-Softmax handles exploration during training for discrete)
                # For inference/evaluation, just take argmax
                action = action_output.argmax(dim=-1).cpu().numpy()
            else:
                action = action_output.argmax(dim=-1).cpu().numpy() # Gumbel-Softmax with hard=True gives one-hot, argmax converts
            return action
        else:
            action = action_output.cpu().numpy().flatten()
            if not evaluate:
                noise = np.random.normal(0, self.config.EXPLORATION_NOISE_STD, size=self.action_dim)
                action = np.clip(action + noise, -1, 1) # Assuming action space is normalized to [-1, 1]
            return action

    def _update_target_networks(self):
        # Update target encoders
        for param, target_param in zip(self.state_encoder.parameters(), self.target_state_encoder.parameters()):
            target_param.data.copy_(param.data)
        for param, target_param in zip(self.state_action_encoder.parameters(), self.target_state_action_encoder.parameters()):
            target_param.data.copy_(param.data)

        # Update target value networks
        for param, target_param in zip(self.value_net1.parameters(), self.target_value_net1.parameters()):
            target_param.data.copy_(param.data)
        for param, target_param in zip(self.value_net2.parameters(), self.target_value_net2.parameters()):
            target_param.data.copy_(param.data)

        # Update target policy network
        for param, target_param in zip(self.policy_net.parameters(), self.target_policy_net.parameters()):
            target_param.data.copy_(param.data)
        
        # Update average_abs_reward for value scaling
        # This is simplified; in practice, it's an EMA over rewards in the replay buffer.
        # For now, let's make it the mean of absolute rewards in the current batch.
        rewards_in_buffer = self.replay_buffer.rewards[:self.replay_buffer.size]
        if len(rewards_in_buffer) > 0:
            self.average_abs_reward = np.mean(np.abs(rewards_in_buffer))
            if self.average_abs_reward == 0:
                self.average_abs_reward = 1.0 # Avoid division by zero

    def _compute_encoder_loss(self, obs, actions, rewards, dones, next_obs):
        # Unroll dynamics over H_Enc horizon
        # Unroll dynamics over H_Enc horizon (currently 1-step for simplicity)
        zs_t = self.state_encoder(obs)
        
        with torch.no_grad():
            target_zs_t_plus_1 = self.target_state_encoder(next_obs)
        
        predicted_zs_t_plus_1, predicted_reward_logits, predicted_terminal, _ = \
            self.state_action_encoder(zs_t, actions)

        # Reward Loss (Categorical Cross-Entropy)
        reward_targets_two_hot = self.reward_scaler.two_hot_encode(rewards.squeeze(1))
        reward_loss = F.cross_entropy(predicted_reward_logits, reward_targets_two_hot)

        # Dynamics Loss (MSE)
        dynamics_loss = F.mse_loss(predicted_zs_t_plus_1, target_zs_t_plus_1)

        # Terminal Loss (MSE)
        # Squeeze to match dimensions if `dones` is (batch_size, 1) and `predicted_terminal` is (batch_size, 1)
        terminal_loss = F.mse_loss(predicted_terminal.squeeze(1), dones.squeeze(1))

        # Combine encoder losses
        encoder_loss = (
            self.config.LAMBDA_REWARD * reward_loss +
            self.config.LAMBDA_DYNAMICS * dynamics_loss +
            self.config.LAMBDA_TERMINAL * terminal_loss
        )
        return encoder_loss

    def _compute_value_loss(self, obs, actions, rewards, dones, next_obs, is_weights):
        with torch.no_grad():
            # Target policy action
            zs_next = self.target_state_encoder(next_obs)
            target_actions, _ = self.target_policy_net(zs_next)

            # Add noise to target actions (for continuous)
            if not self.discrete_action_space:
                noise = (torch.randn_like(target_actions) * self.config.TARGET_POLICY_NOISE_STD).clamp(
                    -self.config.TARGET_POLICY_NOISE_CLIP, self.config.TARGET_POLICY_NOISE_CLIP
                )
                target_actions = (target_actions + noise).clamp(-1, 1)

            # Target state-action embedding
            _, _, _, zsa_next = self.target_state_action_encoder(zs_next, target_actions)
            
            # Min of two target Q-values
            target_q1 = self.target_value_net1(zsa_next)
            target_q2 = self.target_value_net2(zsa_next)
            target_q_min = torch.min(target_q1, target_q2)

            # Multi-step returns
            # For simplicity, this is currently 1-step return.
            # Full H_Q implementation requires proper handling of rewards and next_obs for H_Q steps.
            # r_t + gamma * Q(s', a') (1-d_t)
            target_values = rewards + self.config.DISCOUNT_FACTOR * target_q_min * (1 - dones)
            target_values = target_values / self.average_abs_reward # Reward scaling

        # Current Q-values
        zs = self.state_encoder(obs)
        _, _, _, zsa = self.state_action_encoder(zs, actions)
        current_q1 = self.value_net1(zsa)
        current_q2 = self.value_net2(zsa)
        
        # Huber Loss
        value_loss1 = F.huber_loss(current_q1, target_values, reduction='none')
        value_loss2 = F.huber_loss(current_q2, target_values, reduction='none')
        
        # Apply importance sampling weights
        value_loss = (value_loss1 * is_weights).mean() + (value_loss2 * is_weights).mean()
        
        # For updating priorities: TD-error (abs difference between predicted and target)
        td_errors = (torch.abs(current_q1 - target_values) + torch.abs(current_q2 - target_values)) / 2.0
        
        return value_loss, td_errors.detach().cpu().numpy().flatten()


    def _compute_policy_loss(self, obs):
        zs = self.state_encoder(obs)
        actions, pre_activations = self.policy_net(zs)
        
        # The paper uses -0.5 * sum(Q_i(zsa_pi))
        _, _, _, zsa_pi = self.state_action_encoder(zs, actions)
        q1_pi = self.value_net1(zsa_pi)
        q2_pi = self.value_net2(zsa_pi)
        
        policy_loss = -0.5 * (q1_pi + q2_pi).mean()
        
        # Pre-activation regularization
        policy_loss += self.config.LAMBDA_PRE_ACTIV * (pre_activations ** 2).mean()
        
        return policy_loss

    def train_step(self):
        if len(self.replay_buffer) < self.config.MINI_BATCH_SIZE:
            return

        self.total_timesteps += 1

        # Sample batch from replay buffer
        (
            obs, actions, rewards, dones, next_obs, is_weights, buffer_idxs
        ) = self.replay_buffer.sample(self.config.MINI_BATCH_SIZE)

        obs = torch.FloatTensor(obs).to(self.config.DEVICE)
        actions = torch.FloatTensor(actions).to(self.config.DEVICE)
        rewards = torch.FloatTensor(rewards).to(self.config.DEVICE).unsqueeze(1)
        dones = torch.FloatTensor(dones).to(self.config.DEVICE).unsqueeze(1)
        next_obs = torch.FloatTensor(next_obs).to(self.config.DEVICE)
        is_weights = torch.FloatTensor(is_weights).to(self.config.DEVICE).unsqueeze(1)

        # Update encoder
        self.encoder_optimizer.zero_grad()
        encoder_loss = self._compute_encoder_loss(obs, actions, rewards, dones, next_obs)
        encoder_loss.backward()
        # Gradient clip norm for encoder not explicitly mentioned, assume standard behavior.
        self.encoder_optimizer.step()

        # Update value networks
        self.value_optimizer.zero_grad()
        value_loss, td_errors = self._compute_value_loss(obs, actions, rewards, dones, next_obs, is_weights)
        value_loss.backward()
        # Gradient clip norm for value nets not explicitly mentioned.
        self.value_optimizer.step()
        
        # Update priorities in replay buffer
        self.replay_buffer.update_priorities(buffer_idxs, td_errors)

        # Update policy network (delayed update as in TD3)
        if self.total_timesteps % self.config.TARGET_UPDATE_FREQUENCY == 0:
            self.policy_optimizer.zero_grad()
            policy_loss = self._compute_policy_loss(obs)
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.GRADIENT_CLIP_NORM)
            self.policy_optimizer.step()

            # Update target networks
            self._update_target_networks()
        
        # For logging, return losses (optional)
        return {
            "encoder_loss": encoder_loss.item(),
            "value_loss": value_loss.item(),
            # "policy_loss": policy_loss.item() if self.total_timesteps % self.config.TARGET_UPDATE_FREQUENCY == 0 else 0.0,
            "average_abs_reward": self.average_abs_reward
        }

def run_experiment():
    # Setup environment
    env_manager = EnvironmentManager(
        Config.ENV_NAME,
        Config.SEED,
        discrete_action_space=True, # Example, should be dynamic
        is_visual=False # Example, should be dynamic
    )
    # Determine action_dim and observation_space_shape dynamically
    observation_space_shape = env_manager.observation_space_shape
    action_dim = env_manager.action_dim
    discrete_action_space = isinstance(env_manager.env.action_space, gym.spaces.Discrete)

    # Initialize MR.Q agent
    agent = MRQ(Config, observation_space_shape, action_dim, discrete_action_space)

    # Training loop
    obs, _ = env_manager.reset()
    for timestep in range(Config.TOTAL_TIME_STEPS):
        # Select action
        action = agent.select_action(obs)
        
        # Perform action in env
        next_obs, reward, done, _ = env_manager.step(action)

        # Store transition
        agent.replay_buffer.add(obs, action, reward, done, next_obs)
        
        # Update agent
        agent.train_step()

        obs = next_obs
        if done:
            obs, _ = env_manager.reset()
        
        if timestep % Config.EVAL_FREQUENCY == 0:
            print(f"Timestep {timestep}")
            # Evaluation logic here
            # For brevity, skipping full evaluation for now
            # In a real setup, would run agent.select_action(..., evaluate=True) for several episodes.

    env_manager.close()

if __name__ == '__main__':
    run_experiment()
