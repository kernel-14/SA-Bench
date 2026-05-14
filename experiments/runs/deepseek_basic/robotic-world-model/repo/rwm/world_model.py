"""
Robotic World Model (RWM) - Core Implementation

A GRU-based world model with dual-autoregressive mechanism for robust 
long-horizon predictions in robotic control tasks.

Based on: "Robotic World Model: A Neural Network Simulator for Robust 
Policy Optimization in Robotics" by Li et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class GaussianHead(nn.Module):
    """MLP head that predicts mean and log standard deviation of a Gaussian."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * 2),  # mean and log_std
        )
        self.output_dim = output_dim
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        mean = out[..., :self.output_dim]
        log_std = out[..., self.output_dim:]
        log_std = torch.clamp(log_std, min=-10, max=2)
        return mean, log_std


class GRUBase(nn.Module):
    """GRU base network with configurable hidden dimensions."""
    
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...] = (256, 256)):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dims[0],
            num_layers=len(hidden_dims),
            batch_first=True,
        )
        self.hidden_dims = hidden_dims
        
    def forward(
        self, 
        x: torch.Tensor, 
        h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            h: Initial hidden state of shape (num_layers, batch, hidden_dim)
        Returns:
            output: (batch, seq_len, hidden_dim)
            h_n: Final hidden state
        """
        output, h_n = self.gru(x, h)
        return output, h_n
    
    @property
    def hidden_size(self) -> int:
        return self.hidden_dims[0]
    
    @property
    def num_layers(self) -> int:
        return len(self.hidden_dims)


class RoboticWorldModel(nn.Module):
    """
    Robotic World Model (RWM) with dual-autoregressive mechanism.
    
    Architecture:
        - GRU base: (256, 256) hidden dimensions
        - MLP heads: (128) hidden, ReLU activation
        - predicts Gaussian mean and std for observations and privileged info
    
    The dual-autoregressive mechanism:
        1. Inner autoregression: GRU hidden states updated autoregressively 
           after each historical step within the context horizon M.
        2. Outer autoregression: predicted observations from forecast horizon N 
           are fed back into the network for the next prediction.
    
    Args:
        obs_dim: Dimension of observation space
        act_dim: Dimension of action space
        priv_dim: Dimension of privileged information space
        gru_hidden_dims: Hidden dimensions for GRU layers
        head_hidden_dim: Hidden dimension for MLP heads
        history_horizon: M - number of historical steps for context
        forecast_horizon: N - number of future steps to predict
        forecast_decay: alpha - decay factor for multi-step loss
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        priv_dim: int = 0,
        gru_hidden_dims: Tuple[int, ...] = (256, 256),
        head_hidden_dim: int = 128,
        history_horizon: int = 32,
        forecast_horizon: int = 8,
        forecast_decay: float = 1.0,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.priv_dim = priv_dim
        self.history_horizon = history_horizon
        self.forecast_horizon = forecast_horizon
        self.forecast_decay = forecast_decay
        
        # Input to GRU: concatenated observation and action
        input_dim = obs_dim + act_dim
        self.gru = GRUBase(input_dim=input_dim, hidden_dims=gru_hidden_dims)
        gru_output_dim = self.gru.hidden_size
        
        # Observation prediction head
        self.obs_head = GaussianHead(
            input_dim=gru_output_dim,
            output_dim=obs_dim,
            hidden_dim=head_hidden_dim,
        )
        
        # Privileged information prediction head (if applicable)
        if priv_dim > 0:
            self.priv_head = GaussianHead(
                input_dim=gru_output_dim,
                output_dim=priv_dim,
                hidden_dim=head_hidden_dim,
            )
        else:
            self.priv_head = None
            
    def _step(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Single step forward pass through the world model.
        
        Args:
            obs: (batch, obs_dim) current observation
            act: (batch, act_dim) current action
            h: (num_layers, batch, hidden_dim) GRU hidden state
            
        Returns:
            obs_mean: Predicted next observation mean
            obs_std: Predicted next observation standard deviation
            h_next: Next hidden state
            priv_mean, priv_std: Optional privileged info predictions
        """
        # Concatenate observation and action
        x = torch.cat([obs, act], dim=-1).unsqueeze(1)  # (batch, 1, input_dim)
        
        # GRU forward
        output, h_next = self.gru(x, h)
        gru_out = output.squeeze(1)  # (batch, hidden_dim)
        
        # Prediction heads
        obs_mean, obs_log_std = self.obs_head(gru_out)
        obs_std = torch.exp(obs_log_std)
        
        priv_mean, priv_std = None, None
        if self.priv_head is not None:
            priv_mean, priv_log_std = self.priv_head(gru_out)
            priv_std = torch.exp(priv_log_std)
            
        return obs_mean, obs_std, h_next, priv_mean, priv_std
    
    def _get_initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get initial hidden state (zeros)."""
        return torch.zeros(
            self.gru.num_layers, batch_size, self.gru.hidden_size,
            device=device
        )
    
    def forward_history(
        self,
        obs_history: torch.Tensor,
        act_history: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process the history window through the GRU (inner autoregression).
        
        Args:
            obs_history: (batch, M, obs_dim) historical observations
            act_history: (batch, M, act_dim) historical actions
            
        Returns:
            h: Final hidden state after processing history
        """
        batch_size = obs_history.shape[0]
        M = obs_history.shape[1]
        device = obs_history.device
        
        # Concatenate along last dim
        x = torch.cat([obs_history, act_history], dim=-1)  # (batch, M, input_dim)
        
        # Initialize hidden state
        h0 = self._get_initial_hidden(batch_size, device)
        
        # Process full history (GRU handles autoregression internally)
        _, h = self.gru(x, h0)
        
        return h
    
    def autoregressive_forward(
        self,
        obs_history: torch.Tensor,
        act_history: torch.Tensor,
        act_future: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Full autoregressive forward pass: predict N steps into the future.
        
        This implements the dual-autoregressive mechanism:
        1. Inner autoregression: process M historical steps
        2. Outer autoregression: predict N future steps using own predictions
        
        Args:
            obs_history: (batch, M, obs_dim) historical observations
            act_history: (batch, M, act_dim) historical actions  
            act_future: (batch, N, act_dim) future actions for autoregressive rollouts
            
        Returns:
            Dictionary containing:
                - obs_means: (batch, N, obs_dim) predicted observation means
                - obs_stds: (batch, N, obs_dim) predicted observation stds
                - priv_means: (batch, N, priv_dim) predicted privileged info means
                - priv_stds: (batch, N, priv_dim) predicted privileged info stds
        """
        batch_size = obs_history.shape[0]
        N = act_future.shape[1]
        device = obs_history.device
        
        # Step 1: Process history (inner autoregression)
        h = self.forward_history(obs_history, act_history)
        
        # Step 2: Autoregressive prediction (outer autoregression)
        # The last observation from history is used as the starting point
        obs_current = obs_history[:, -1, :]  # (batch, obs_dim)
        
        obs_means_list = []
        obs_stds_list = []
        priv_means_list = []
        priv_stds_list = []
        
        for k in range(N):
            act_k = act_future[:, k, :]  # (batch, act_dim)
            
            obs_mean, obs_std, h, priv_mean, priv_std = self._step(
                obs_current, act_k, h
            )
            
            obs_means_list.append(obs_mean)
            obs_stds_list.append(obs_std)
            
            if priv_mean is not None:
                priv_means_list.append(priv_mean)
                priv_stds_list.append(priv_std)
            
            # Use predicted mean as next observation (reparameterization could be done
            # during training for gradient propagation)
            obs_current = obs_mean
        
        result = {
            'obs_means': torch.stack(obs_means_list, dim=1),  # (batch, N, obs_dim)
            'obs_stds': torch.stack(obs_stds_list, dim=1),
        }
        
        if self.priv_head is not None:
            result['priv_means'] = torch.stack(priv_means_list, dim=1)
            result['priv_stds'] = torch.stack(priv_stds_list, dim=1)
        
        return result
    
    def autoregressive_forward_with_sampling(
        self,
        obs_history: torch.Tensor,
        act_history: torch.Tensor,
        act_future: torch.Tensor,
        use_reparameterization: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Same as autoregressive_forward but with sampling for training
        (reparameterization trick for gradient propagation).
        """
        batch_size = obs_history.shape[0]
        N = act_future.shape[1]
        device = obs_history.device
        
        h = self.forward_history(obs_history, act_history)
        obs_current = obs_history[:, -1, :]
        
        obs_means_list = []
        obs_stds_list = []
        obs_samples_list = []
        priv_means_list = []
        priv_stds_list = []
        priv_samples_list = []
        
        for k in range(N):
            act_k = act_future[:, k, :]
            obs_mean, obs_std, h, priv_mean, priv_std = self._step(
                obs_current, act_k, h
            )
            
            obs_means_list.append(obs_mean)
            obs_stds_list.append(obs_std)
            
            # Sample using reparameterization trick for gradient flow
            if use_reparameterization:
                eps = torch.randn_like(obs_mean)
                obs_sample = obs_mean + obs_std * eps
            else:
                obs_sample = obs_mean + obs_std * torch.randn_like(obs_mean)
            
            obs_samples_list.append(obs_sample)
            
            if priv_mean is not None:
                priv_means_list.append(priv_mean)
                priv_stds_list.append(priv_std)
                if use_reparameterization:
                    eps_p = torch.randn_like(priv_mean)
                    priv_sample = priv_mean + priv_std * eps_p
                else:
                    priv_sample = priv_mean + priv_std * torch.randn_like(priv_mean)
                priv_samples_list.append(priv_sample)
            
            # Feed sampled observation for next step
            obs_current = obs_sample
        
        result = {
            'obs_means': torch.stack(obs_means_list, dim=1),
            'obs_stds': torch.stack(obs_stds_list, dim=1),
            'obs_samples': torch.stack(obs_samples_list, dim=1),
        }
        
        if self.priv_head is not None:
            result['priv_means'] = torch.stack(priv_means_list, dim=1)
            result['priv_stds'] = torch.stack(priv_stds_list, dim=1)
            result['priv_samples'] = torch.stack(priv_samples_list, dim=1)
        
        return result
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Convenience forward pass for training with a batch.
        
        Expected batch keys:
            - obs: (batch, M+N, obs_dim) full observation sequence
            - act: (batch, M+N-1, act_dim) full action sequence
            - priv: (batch, M+N, priv_dim) optional privileged info sequence
        """
        obs = batch['obs']
        act = batch['act']
        
        M = self.history_horizon
        N = self.forecast_horizon
        
        obs_history = obs[:, :M, :]       # First M steps (history)
        act_history = act[:, :M, :]       # First M actions
        act_future = act[:, M-1:M+N-1, :]  # N actions: last hist act + next N-1
        
        return self.autoregressive_forward_with_sampling(
            obs_history, act_history, act_future
        )
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the multi-step prediction loss with forecast decay.
        
        Loss = (1/N) * sum_k alpha^k * [L_o(o'_{t+k}, o_{t+k}) + L_c(c'_{t+k}, c_{t+k})]
        
        Where L_o and L_c are Gaussian negative log-likelihood losses.
        
        Args:
            predictions: Output from autoregressive_forward_with_sampling
            targets: Dictionary containing 'obs' (batch, N, obs_dim) and 
                    optionally 'priv' (batch, N, priv_dim)
                    
        Returns:
            Dictionary with loss components
        """
        N = self.forecast_horizon
        alpha = self.forecast_decay
        device = predictions['obs_means'].device
        
        # Compute per-step weights: alpha^k
        k_values = torch.arange(1, N + 1, device=device, dtype=torch.float32)
        weights = alpha ** k_values  # (N,)
        weights = weights / weights.sum()  # Normalize
        
        # Observation loss: Gaussian NLL
        obs_means = predictions['obs_means']  # (batch, N, obs_dim)
        obs_stds = predictions['obs_stds']    # (batch, N, obs_dim)
        obs_targets = targets['obs']          # (batch, N, obs_dim)
        
        # Gaussian log likelihood per step
        obs_var = obs_stds ** 2
        obs_nll = 0.5 * (
            torch.log(2 * torch.pi * obs_var) + 
            ((obs_targets - obs_means) ** 2) / obs_var
        )  # (batch, N, obs_dim)
        obs_nll = obs_nll.mean(dim=-1)  # (batch, N) - mean over obs dim
        
        # Weighted sum over forecast steps
        obs_loss = (obs_nll * weights.unsqueeze(0)).sum(dim=1).mean()  # scalar
        
        total_loss = obs_loss
        
        result = {'obs_loss': obs_loss}
        
        # Privileged information loss
        if self.priv_head is not None and 'priv' in targets and 'priv_means' in predictions:
            priv_means = predictions['priv_means']
            priv_stds = predictions['priv_stds']
            priv_targets = targets['priv']
            
            priv_var = priv_stds ** 2
            priv_nll = 0.5 * (
                torch.log(2 * torch.pi * priv_var) +
                ((priv_targets - priv_means) ** 2) / priv_var
            )
            priv_nll = priv_nll.mean(dim=-1)
            
            priv_loss = (priv_nll * weights.unsqueeze(0)).sum(dim=1).mean()
            
            total_loss = total_loss + priv_loss
            result['priv_loss'] = priv_loss
        
        result['total_loss'] = total_loss
        return result


def create_rwm_anymal_d() -> RoboticWorldModel:
    """Create RWM for ANYmal D (Table S2, S4)."""
    # Observation: base lin vel(3) + base ang vel(3) + gravity(3) 
    #             + joint pos(12) + joint vel(12) + joint torques(12) = 45
    # Action: joint position targets(12) 
    # Privileged: knee contact(4) + foot contact(4) = 8
    return RoboticWorldModel(
        obs_dim=45,
        act_dim=12,
        priv_dim=8,
        gru_hidden_dims=(256, 256),
        head_hidden_dim=128,
        history_horizon=32,
        forecast_horizon=8,
        forecast_decay=1.0,
    )


def create_rwm_unitree_g1() -> RoboticWorldModel:
    """Create RWM for Unitree G1 (Table S2, S4)."""
    # Observation: base lin vel(3) + base ang vel(3) + gravity(3)
    #             + joint pos(29) + joint vel(29) + joint torques(29) = 96
    # Action: joint position targets(29)
    # Privileged: body contact(26) + foot height(2) + foot velocity(2) = 30
    return RoboticWorldModel(
        obs_dim=96,
        act_dim=29,
        priv_dim=30,
        gru_hidden_dims=(256, 256),
        head_hidden_dim=128,
        history_horizon=32,
        forecast_horizon=8,
        forecast_decay=1.0,
    )
