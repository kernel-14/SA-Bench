"""
MR.Q Network Architectures
Based on: Towards General-Purpose Model-Free RL (MR.Q)
Fujimoto et al., 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


# Embedding dimensions (from paper Table 3)
ZS_DIM = 512
ZA_DIM = 256
ZSA_DIM = 512


def xavier_init(module):
    """Xavier uniform initialization with zero bias."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class StateEncoder(nn.Module):
    """
    State encoder f_omega: s -> z_s

    For image inputs: 4 conv layers (32 channels, kernel 3, strides 2,2,2,1)
                      + linear + LayerNorm + ELU
    For vector inputs: 3-layer MLP with LayerNorm + ELU activations
    """

    def __init__(self, state_dim, image_obs=False, state_channels=None):
        super().__init__()
        self.image_obs = image_obs
        self.activ = F.elu

        if image_obs:
            # CNN for 84x84 image observations
            self.zs_cnn1 = nn.Conv2d(state_channels, 32, 3, stride=2)
            self.zs_cnn2 = nn.Conv2d(32, 32, 3, stride=2)
            self.zs_cnn3 = nn.Conv2d(32, 32, 3, stride=2)
            self.zs_cnn4 = nn.Conv2d(32, 32, 3, stride=1)
            # 84x84 -> 41x41 -> 20x20 -> 9x9 -> 7x7, 32 channels = 1568
            self.zs_lin = nn.Linear(1568, ZS_DIM)
        else:
            # MLP for vector observations
            self.zs_mlp1 = nn.Linear(state_dim, 512)
            self.zs_mlp2 = nn.Linear(512, 512)
            self.zs_mlp3 = nn.Linear(512, ZS_DIM)

        self.apply(xavier_init)

    def forward(self, state):
        if self.image_obs:
            return self._cnn_forward(state)
        else:
            return self._mlp_forward(state)

    def _cnn_forward(self, state):
        state = state.float() / 255.0 - 0.5
        zs = self.activ(self.zs_cnn1(state))
        zs = self.activ(self.zs_cnn2(zs))
        zs = self.activ(self.zs_cnn3(zs))
        zs = self.activ(self.zs_cnn4(zs))
        zs = zs.reshape(zs.shape[0], -1)
        zs = self.activ(F.layer_norm(self.zs_lin(zs), (ZS_DIM,)))
        return zs

    def _mlp_forward(self, state):
        zs = self.activ(F.layer_norm(self.zs_mlp1(state), (512,)))
        zs = self.activ(F.layer_norm(self.zs_mlp2(zs), (512,)))
        zs = self.activ(F.layer_norm(self.zs_mlp3(zs), (ZS_DIM,)))
        return zs


class StateActionEncoder(nn.Module):
    """
    State-action encoder g_omega: (z_s, a) -> z_sa
    Plus linear MDP predictor m: z_sa -> (z_s_pred, r_logits, d_pred)

    The model is a single linear layer applied to z_sa (linear MDP predictor).
    """

    def __init__(self, action_dim, reward_bins=65):
        super().__init__()
        self.activ = F.elu
        self.reward_bins = reward_bins

        # Action embedding layer
        self.za = nn.Linear(action_dim, ZA_DIM)

        # State-action MLP
        self.zsa1 = nn.Linear(ZS_DIM + ZA_DIM, 512)
        self.zsa2 = nn.Linear(512, 512)
        self.zsa3 = nn.Linear(512, ZSA_DIM)

        # Linear MDP predictor: z_sa -> (z_s_next, r_logits, d)
        output_dim = ZS_DIM + reward_bins + 1
        self.model = nn.Linear(ZSA_DIM, output_dim)

        self.apply(xavier_init)

    def forward(self, zs, action):
        za = self.activ(self.za(action))
        zsa = torch.cat([zs, za], dim=-1)
        zsa = self.activ(F.layer_norm(self.zsa1(zsa), (512,)))
        zsa = self.activ(F.layer_norm(self.zsa2(zsa), (512,)))
        zsa = self.zsa3(zsa)
        model_out = self.model(zsa)
        return model_out, zsa


class ValueNetwork(nn.Module):
    """
    Value network Q_theta: z_sa -> R
    4-layer MLP, LayerNorm + ELU after first 3 layers.
    """

    def __init__(self):
        super().__init__()
        self.activ = F.elu
        self.l1 = nn.Linear(ZSA_DIM, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 512)
        self.l4 = nn.Linear(512, 1)
        self.apply(xavier_init)

    def forward(self, zsa):
        q = self.activ(F.layer_norm(self.l1(zsa), (512,)))
        q = self.activ(F.layer_norm(self.l2(q), (512,)))
        q = self.activ(F.layer_norm(self.l3(q), (512,)))
        return self.l4(q)


class PolicyNetwork(nn.Module):
    """
    Policy network pi_phi: z_s -> a
    3-layer MLP, LayerNorm + ReLU after first 2 layers.

    Discrete: Gumbel-Softmax (tau=10)
    Continuous: Tanh
    """

    def __init__(self, action_dim, discrete=False, gumbel_tau=10.0):
        super().__init__()
        self.activ = F.relu
        self.discrete = discrete
        self.gumbel_tau = gumbel_tau

        self.l1 = nn.Linear(ZS_DIM, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, action_dim)

        self.apply(xavier_init)

    def forward(self, zs, hard=False):
        """Returns (pre_activation z_pi, activated action a_pi)."""
        a = self.activ(F.layer_norm(self.l1(zs), (512,)))
        a = self.activ(F.layer_norm(self.l2(a), (512,)))
        z_pi = self.l3(a)

        if self.discrete:
            a_pi = F.gumbel_softmax(z_pi, tau=self.gumbel_tau, hard=hard)
        else:
            a_pi = torch.tanh(z_pi)

        return z_pi, a_pi
