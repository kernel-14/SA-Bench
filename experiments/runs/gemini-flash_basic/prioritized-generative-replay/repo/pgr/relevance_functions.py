import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from models.networks import QNetwork, PolicyNetwork, FeatureEncoder, ForwardDynamicsModel # Import new network types

class RelevanceFunction(ABC):
    @abstractmethod
    def calculate(self, transitions):
        pass

    @abstractmethod
    def update(self, real_transitions):
        pass

class ReturnRelevanceFunction(RelevanceFunction):
    def __init__(self, q_network: QNetwork, policy_network: PolicyNetwork, gamma=0.99):
        self.q_network = q_network
        self.policy_network = policy_network
        self.gamma = gamma

    def calculate(self, transitions):
        # F(s, a, s', r) = Q(s, pi(s))
        states = transitions['states']
        with torch.no_grad():
            actions = self.policy_network(states)
            q_values = self.q_network(states, actions)
        return q_values.squeeze(-1) # Return a 1D tensor of relevance scores

    def update(self, real_transitions):
        pass

class TDErrorRelevanceFunction(RelevanceFunction):
    def __init__(self, q_network: QNetwork, target_q_network: QNetwork, policy_network: PolicyNetwork, gamma=0.99):
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.policy_network = policy_network # Added to get actions for next states
        self.gamma = gamma

    def calculate(self, transitions):
        # F(s, a, s', r) = r + gamma * Q_target(s', argmax_a' Q(s', a')) - Q(s, a)
        states = transitions['states']
        actions = transitions['actions']
        rewards = transitions['rewards'].float().squeeze(-1) # Ensure float and 1D
        next_states = transitions['next_states']
        dones = transitions['dones'].float().squeeze(-1) # Ensure float and 1D

        with torch.no_grad():
            current_q_values = self.q_network(states, actions).squeeze(-1)

            # Get next actions from the policy network
            next_actions = self.policy_network(next_states)
            target_next_q_values = self.target_q_network(next_states, next_actions).squeeze(-1)
            target_q = rewards + (1 - dones) * self.gamma * target_next_q_values

        td_error = (target_q - current_q_values).abs()
        return td_error

    def update(self, real_transitions):
        pass

class CuriosityRelevanceFunction(RelevanceFunction):
    def __init__(self, feature_encoder: FeatureEncoder, forward_dynamics_model: ForwardDynamicsModel, optimizer=None, device='cpu'):
        self.feature_encoder = feature_encoder.to(device)
        self.forward_dynamics_model = forward_dynamics_model.to(device)
        self.optimizer = optimizer
        self.loss_fn = nn.MSELoss()
        self.device = device

    def calculate(self, transitions):
        # F(s, a, s', r) = 0.5 * ||g(h(s), a) - h(s')||^2
        states = transitions['states'].to(self.device)
        actions = transitions['actions'].to(self.device)
        next_states = transitions['next_states'].to(self.device)

        with torch.no_grad():
            h_s = self.feature_encoder(states)
            h_s_prime = self.feature_encoder(next_states)
            predicted_h_s_prime = self.forward_dynamics_model(h_s, actions)
            curiosity_score = 0.5 * torch.norm(predicted_h_s_prime - h_s_prime, dim=-1)**2
        return curiosity_score

    def update(self, real_transitions):
        if self.optimizer is None:
            return

        states = real_transitions['states'].to(self.device)
        actions = real_transitions['actions'].to(self.device)
        next_states = real_transitions['next_states'].to(self.device)

        self.optimizer.zero_grad()
        h_s = self.feature_encoder(states)
        h_s_prime = self.feature_encoder(next_states)
        predicted_h_s_prime = self.forward_dynamics_model(h_s, actions)
        loss = self.loss_fn(predicted_h_s_prime, h_s_prime)
        loss.backward()
        self.optimizer.step()
