"""
MR.Q Network Architectures

Based on the paper "Towards General-Purpose Model-Free RL (MR.Q)"
The architectures follow exactly the PyTorch code blocks in Appendix B.2.

All networks use:
- Xavier uniform weight initialization with bias 0
- ELU activations for encoders and value networks
- ReLU activations for policy network
- LayerNorm followed by activation (ln_activ pattern)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math


class StateEncoder(nn.Module):
    """
    State encoder f: s -> z_s
    
    For image inputs: 4 conv layers (32 channels, kernel 3, strides [2,2,2,1]),
    followed by linear + LayerNorm + ELU.
    
    For vector inputs: 3-layer MLP (hidden dim 512),
    LayerNorm + ELU after each layer.
    
    The resulting state embedding z_s has dimension zs_dim (512).
    """
    
    def __init__(self, state_dim, zs_dim=512, image_observations=False, 
                 state_channels=1, image_size=84):
        super().__init__()
        self.zs_dim = zs_dim
        self.image_observations = image_observations
        self.activ = F.elu
        
        if image_observations:
            # 4 conv layers as specified in paper
            self.zs_cnn1 = nn.Conv2d(state_channels, 32, 3, stride=2)
            self.zs_cnn2 = nn.Conv2d(32, 32, 3, stride=2)
            self.zs_cnn3 = nn.Conv2d(32, 32, 3, stride=2)
            self.zs_cnn4 = nn.Conv2d(32, 32, 3, stride=1)
            # Compute flattened size: after 3 strides of 2 on 84x84:
            # 84 -> 42 -> 21 -> 11 (due to stride 2 with no padding on kernel 3)
            # Actually with kernel=3, stride=2, no padding: 
            # floor((84-3)/2+1)=41, floor((41-3)/2+1)=20, floor((20-3)/2+1)=9
            # 9*9*32 = 2592. But paper says 1568 for 84x84.
            # With kernel=3, stride=2: 84->41->20->9, then stride=1: 9->7
            # 7*7*32 = 1568. Correct!
            self.zs_lin = nn.Linear(1568, zs_dim)
        else:
            # 3-layer MLP for vector inputs
            self.zs_mlp1 = nn.Linear(state_dim, 512)
            self.zs_mlp2 = nn.Linear(512, 512)
            self.zs_mlp3 = nn.Linear(512, zs_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier uniform initialization with bias 0."""
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _ln_activ(self, x):
        """LayerNorm followed by ELU activation."""
        x = F.layer_norm(x, (x.shape[-1],))
        return self.activ(x)
    
    def cnn_forward(self, state):
        """Forward pass for image observations."""
        # Normalize: state / 255. - 0.5
        state = state / 255.0 - 0.5
        zs = self.activ(self.zs_cnn1(state))
        zs = self.activ(self.zs_cnn2(zs))
        zs = self.activ(self.zs_cnn3(zs))
        zs = self.activ(self.zs_cnn4(zs))
        batch_size = zs.shape[0]
        zs = zs.reshape(batch_size, 1568)
        return self._ln_activ(self.zs_lin(zs))
    
    def mlp_forward(self, state):
        """Forward pass for vector observations."""
        zs = self._ln_activ(self.zs_mlp1(state))
        zs = self._ln_activ(self.zs_mlp2(zs))
        return self._ln_activ(self.zs_mlp3(zs))
    
    def forward(self, state):
        if self.image_observations:
            return self.cnn_forward(state)
        else:
            return self.mlp_forward(state)


class StateActionEncoder(nn.Module):
    """
    State-action encoder g: (z_s, a) -> z_sa and predictions (r, z_s', d)
    
    Action input is processed by a linear layer + ELU, then concatenated 
    with z_s. Processed by a 3-layer MLP (hidden dim 512) with 
    LayerNorm+ELU after first two layers.
    
    The resulting z_sa is used by a LINEAR layer m to predict:
    - next state embedding (zs_dim)
    - reward distribution (num_reward_bins)  
    - terminal signal (1)
    """
    
    def __init__(self, action_dim, zs_dim=512, za_dim=256, zsa_dim=512, 
                 num_reward_bins=65):
        super().__init__()
        self.zs_dim = zs_dim
        self.za_dim = za_dim
        self.zsa_dim = zsa_dim
        self.num_reward_bins = num_reward_bins
        self.output_dim = zs_dim + num_reward_bins + 1  # z_s', r, d
        
        self.activ = F.elu
        
        # Action embedding
        self.za = nn.Linear(action_dim, za_dim)
        
        # State-action MLP
        self.zsa1 = nn.Linear(zs_dim + za_dim, 512)
        self.zsa2 = nn.Linear(512, 512)
        self.zsa3 = nn.Linear(512, zsa_dim)
        
        # Linear MDP predictor (equation 14 in paper)
        self.model = nn.Linear(zsa_dim, self.output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _ln_activ(self, x):
        x = F.layer_norm(x, (x.shape[-1],))
        return self.activ(x)
    
    def forward(self, zs, action):
        """
        Args:
            zs: state embedding from state encoder [batch, zs_dim]
            action: action [batch, action_dim]
        
        Returns:
            predictions: [batch, output_dim] containing (z_s', r_bins, d)
            zsa: state-action embedding [batch, zsa_dim]
        """
        za = self.activ(self.za(action))
        zsa = torch.cat([zs, za], dim=1)
        zsa = self._ln_activ(self.zsa1(zsa))
        zsa = self._ln_activ(self.zsa2(zsa))
        zsa = self.zsa3(zsa)
        return self.model(zsa), zsa
    
    def unroll(self, zs0, actions, horizon):
        """
        Unroll the dynamics model over a short horizon.
        
        Args:
            zs0: initial state embedding [batch, zs_dim]
            actions: sequence of actions [batch, horizon, action_dim]
            horizon: number of steps to unroll
        
        Returns:
            zs_pred: predicted next state embeddings [batch, horizon, zs_dim]
            r_pred: predicted reward logits [batch, horizon, num_reward_bins]
            d_pred: predicted terminal [batch, horizon, 1]
            zsa_list: state-action embeddings [batch, horizon, zsa_dim]
        """
        batch_size = zs0.shape[0]
        zs_current = zs0
        
        all_zs_pred = []
        all_r_pred = []
        all_d_pred = []
        all_zsa = []
        
        for t in range(horizon):
            pred, zsa = self.forward(zs_current, actions[:, t])
            
            # Split prediction
            zs_pred = pred[:, :self.zs_dim]
            r_pred = pred[:, self.zs_dim:self.zs_dim + self.num_reward_bins]
            d_pred = pred[:, self.zs_dim + self.num_reward_bins:]
            
            all_zs_pred.append(zs_pred)
            all_r_pred.append(r_pred)
            all_d_pred.append(d_pred)
            all_zsa.append(zsa)
            
            zs_current = zs_pred
        
        return (
            torch.stack(all_zs_pred, dim=1),
            torch.stack(all_r_pred, dim=1),
            torch.stack(all_d_pred, dim=1),
            torch.stack(all_zsa, dim=1)
        )


class ValueNetwork(nn.Module):
    """
    Value (Q) network: z_sa -> Q value
    
    4-layer MLP with hidden dim 512, LayerNorm + ELU after first 3 layers.
    Two value networks are used (same architecture, independently initialized).
    """
    
    def __init__(self, zsa_dim=512, hidden_dim=512):
        super().__init__()
        self.activ = F.elu
        
        self.l1 = nn.Linear(zsa_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _ln_activ(self, x):
        x = F.layer_norm(x, (x.shape[-1],))
        return self.activ(x)
    
    def forward(self, zsa):
        q = self._ln_activ(self.l1(zsa))
        q = self._ln_activ(self.l2(q))
        q = self._ln_activ(self.l3(q))
        return self.l4(q)


class PolicyNetwork(nn.Module):
    """
    Policy network: z_s -> action
    
    3-layer MLP with hidden dim 512, LayerNorm + ReLU after first 2 layers.
    
    For discrete actions: final activation is Gumbel Softmax (tau=10).
    For continuous actions: final activation is Tanh.
    """
    
    def __init__(self, zs_dim=512, action_dim=1, discrete_action_space=False, 
                 hidden_dim=512):
        super().__init__()
        self.discrete_action_space = discrete_action_space
        self.action_dim = action_dim
        self.activ = F.relu
        
        self.l1 = nn.Linear(zs_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        
        if discrete_action_space:
            self.final_activ = partial(F.gumbel_softmax, tau=10)
        else:
            self.final_activ = torch.tanh
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _ln_activ(self, x):
        x = F.layer_norm(x, (x.shape[-1],))
        return self.activ(x)
    
    def forward(self, zs, hard=False):
        """
        Args:
            zs: state embedding [batch, zs_dim]
            hard: if True and discrete, use argmax (for evaluation)
        
        Returns:
            action: [batch, action_dim]
            pre_activ: pre-activation z_pi (for regularization)
        """
        a = self._ln_activ(self.l1(zs))
        a = self._ln_activ(self.l2(a))
        pre_activ = self.l3(a)
        
        if self.discrete_action_space:
            if hard:
                action = F.gumbel_softmax(pre_activ, tau=10, hard=True)
            else:
                action = self.final_activ(pre_activ)
        else:
            action = self.final_activ(pre_activ)
        
        return action, pre_activ
