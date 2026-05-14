import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from models import QNetwork, PolicyNetwork # Assuming these are appropriate for SAC
from utils.replay_buffer import ReplayBuffer

class SACAgent(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, device='cpu'):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha # Entropy regularization coefficient
        self.device = device

        # Actor Network
        self.actor = PolicyNetwork(obs_dim, action_dim * 2, hidden_dim).to(device) # Policy outputs mean and log_std
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Critic Networks (Two Q-networks for SAC)
        self.critic1 = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.critic2 = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.critic1_target = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = QNetwork(obs_dim, action_dim, hidden_dim).to(device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr)

        # Entropy temperature (alpha) is also learned in original SAC
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(device)).item() # Heuristic for target entropy

    def get_action(self, obs, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0) # Add batch dim
        mean, log_std = self.actor(obs).chunk(2, dim=-1)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = normal.sample()

        # Apply tanh squashing to actions to keep them in a bounded range (e.g., -1 to 1)
        # The original paper clips between -1 and 1
        action = torch.tanh(action)
        return action.squeeze(0).cpu().numpy() # Remove batch dim, convert to numpy

    def update_critic(self, state, action, reward, next_state, done):
        with torch.no_grad():
            # Sample action from the current policy for target Q values
            mean, log_std = self.actor(next_state).chunk(2, dim=-1)
            std = log_std.exp()
            normal = Normal(mean, std)
            next_action = normal.sample()
            log_prob_next_action = normal.log_prob(next_action).sum(axis=-1, keepdim=True)
            next_action = torch.tanh(next_action)

            # Compute target Q values
            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            min_target_q = torch.min(target_q1, target_q2) - self.alpha * log_prob_next_action
            target_q_value = reward + self.gamma * (1 - done) * min_target_q

        # Get current Q estimates
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)

        critic1_loss = nn.functional.mse_loss(current_q1, target_q_value)
        critic2_loss = nn.functional.mse_loss(current_q2, target_q_value)
        critic_loss = critic1_loss + critic2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss.item()

    def update_actor_and_alpha(self, state):
        mean, log_std = self.actor(state).chunk(2, dim=-1)
        std = log_std.exp()
        normal = Normal(mean, std)
        action = normal.sample()
        log_prob = normal.log_prob(action).sum(axis=-1, keepdim=True)
        action = torch.tanh(action)

        q1_value = self.critic1(state, action)
        q2_value = self.critic2(state, action)
        min_q_value = torch.min(q1_value, q2_value)

        actor_loss = (self.alpha * log_prob - min_q_value).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha
        alpha_loss = (self.log_alpha * (-log_prob - self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        return actor_loss.item(), alpha_loss.item()

    def update_target_networks(self):
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_parameters(self, replay_buffer: ReplayBuffer, batch_size):
        transitions = replay_buffer.sample(batch_size)

        state = transitions['states'].to(self.device)
        action = transitions['actions'].to(self.device)
        reward = transitions['rewards'].to(self.device)
        next_state = transitions['next_states'].to(self.device)
        done = transitions['dones'].to(self.device)

        critic_loss = self.update_critic(state, action, reward, next_state, done)
        actor_loss, alpha_loss = self.update_actor_and_alpha(state)
        self.update_target_networks()

        return critic_loss, actor_loss, alpha_loss
