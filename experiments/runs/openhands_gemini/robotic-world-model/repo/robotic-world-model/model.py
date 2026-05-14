
import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple, List

from config import GlobalConfig
from modules import RWMBase, RWMHeads, PolicyNetwork, ValueNetwork

class RoboticWorldModel(nn.Module):
    """
    The Robotic World Model (RWM) which integrates the GRU base and MLP heads.
    It implements the dual-autoregressive mechanism for long-horizon predictions.
    """
    def __init__(self, cfg: GlobalConfig):
        super().__init__()
        self.cfg = cfg
        self.obs_dim = cfg.rwm_obs_dim
        self.action_dim = cfg.rwm_action_dim
        self.priv_info_dim = cfg.rwm_priv_info_dim
        self.history_horizon_M = cfg.rwm_config.history_horizon_M
        self.forecast_horizon_N = cfg.rwm_config.forecast_horizon_N

        self.rwm_base = RWMBase(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_size=cfg.rwm_config.rwm_gru_hidden_size
        )
        self.rwm_heads = RWMHeads(
            input_size=cfg.rwm_config.rwm_gru_hidden_size,
            obs_dim=self.obs_dim,
            priv_info_dim=self.priv_info_dim,
            hidden_size=cfg.rwm_config.rwm_mlp_head_hidden_size,
            activation=cfg.rwm_config.rwm_mlp_head_activation
        )

    def forward(self,
                observations_history: torch.Tensor, # (batch_size, M, obs_dim)
                actions_history: torch.Tensor,      # (batch_size, M, action_dim)
                actions_forecast: torch.Tensor,     # (batch_size, N, action_dim)
                initial_hidden_state: torch.Tensor = None
        ) -> Tuple[List[Normal], List[Normal], torch.Tensor]:
        """
        Performs a dual-autoregressive rollout for RWM prediction.
        Args:
            observations_history (torch.Tensor): Historical observations (M steps).
            actions_history (torch.Tensor): Historical actions (M steps).
            actions_forecast (torch.Tensor): Future actions to condition the forecast (N steps).
            initial_hidden_state (torch.Tensor, optional): Initial GRU hidden state.
        Returns:
            Tuple[List[Normal], List[Normal], torch.Tensor]:
                - List of predicted observation distributions for N steps.
                - List of predicted privileged information distributions for N steps.
                - Final GRU hidden state after the entire rollout.
        """
        batch_size = observations_history.shape[0]
        device = observations_history.device

        predicted_obs_distributions = []
        predicted_priv_info_distributions = []

        # Inner Autoregression (Context Horizon M)
        # Process M historical steps to get the hidden state for forecasting
        current_hidden_state = initial_hidden_state
        for i in range(self.history_horizon_M):
            obs_t = observations_history[:, i, :]
            action_t = actions_history[:, i, :]
            input_t = torch.cat([obs_t, action_t], dim=-1).unsqueeze(1) # (batch, 1, input_dim)
            _, current_hidden_state = self.rwm_base(input_t, current_hidden_state)

        # Outer Autoregression (Forecast Horizon N)
        # Use the last actual observation from history as the starting point for prediction
        # and feed predictions back into the model.
        predicted_obs_t = observations_history[:, -1, :] # Last observation from history

        for i in range(self.forecast_horizon_N):
            action_t_plus_1 = actions_forecast[:, i, :] # Next action from forecast sequence
            
            # Input for the next prediction: [predicted_obs_t, action_t_plus_1]
            input_for_next_step = torch.cat([predicted_obs_t, action_t_plus_1], dim=-1).unsqueeze(1)

            # Update GRU hidden state using the last predicted observation and next action
            gru_output, current_hidden_state = self.rwm_base(input_for_next_step, current_hidden_state)
            
            # Predict next observation and privileged information distributions
            obs_dist, priv_info_dist = self.rwm_heads(gru_output.squeeze(1)) # Squeeze sequence_length dim

            predicted_obs_distributions.append(obs_dist)
            predicted_priv_info_distributions.append(priv_info_dist)

            # Sample the next observation to feed back into the next step
            predicted_obs_t = obs_dist.sample()

        return predicted_obs_distributions, predicted_priv_info_distributions, current_hidden_state

    def predict_next_step(self,
                          current_obs: torch.Tensor,
                          current_action: torch.Tensor,
                          current_hidden_state: torch.Tensor
        ) -> Tuple[Normal, Normal, torch.Tensor]:
        """
        Predicts the next observation and privileged information for a single step
        given the current observation, action, and GRU hidden state.
        This is used during inner autoregression for history processing
        or during policy rollout in imagination.
        Args:
            current_obs (torch.Tensor): Current observation (batch_size, obs_dim).
            current_action (torch.Tensor): Current action (batch_size, action_dim).
            current_hidden_state (torch.Tensor): Current GRU hidden state (1, batch_size, hidden_size).
        Returns:
            Tuple[Normal, Normal, torch.Tensor]:
                - Predicted observation distribution.
                - Predicted privileged information distribution.
                - Updated GRU hidden state.
        """
        input_t = torch.cat([current_obs, current_action], dim=-1).unsqueeze(1) # (batch, 1, input_dim)
        gru_output, next_hidden_state = self.rwm_base(input_t, current_hidden_state)
        obs_dist, priv_info_dist = self.rwm_heads(gru_output.squeeze(1))
        return obs_dist, priv_info_dist, next_hidden_state


class ActorCritic(nn.Module):
    """
    Combines the PolicyNetwork (Actor) and ValueNetwork (Critic) for MBPO-PPO.
    """
    def __init__(self, cfg: GlobalConfig):
        super().__init__()
        self.cfg = cfg
        self.policy = PolicyNetwork(
            obs_dim=cfg.policy_obs_dim,
            action_dim=cfg.policy_action_dim,
            hidden_sizes=cfg.mbpo_ppo_config.policy_mlp_hidden_shape,
            activation=cfg.mbpo_ppo_config.policy_value_activation
        )
        self.value_function = ValueNetwork(
            obs_dim=cfg.policy_obs_dim,
            hidden_sizes=cfg.mbpo_ppo_config.value_mlp_hidden_shape,
            activation=cfg.mbpo_ppo_config.policy_value_activation
        )

    def forward(self, obs: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        """
        Forward pass for both policy and value function.
        Args:
            obs (torch.Tensor): Observation from the environment.
        Returns:
            Tuple[Normal, torch.Tensor]:
                - Action distribution from the policy.
                - Value estimate from the value function.
        """
        action_dist = self.policy(obs)
        value = self.value_function(obs)
        return action_dist, value

