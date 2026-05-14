"""Neural network architectures for MR.Q."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Tuple, Optional

from mrq.config import MRQConfig


def ln_activ(x: torch.Tensor, activ: nn.Module) -> torch.Tensor:
    x = F.layer_norm(x, (x.shape[-1],))
    return activ(x)


class LayerNormActiv(nn.Module):
    """LayerNorm followed by activation."""

    def __init__(self, dim: int, activation: str = "ELU"):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        if activation == "ELU":
            self.activ = nn.ELU()
        elif activation == "ReLU":
            self.activ = nn.ReLU()
        else:
            self.activ = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activ(self.norm(x))


class MLPLayer(nn.Module):
    """Linear -> LayerNorm -> Activation."""

    def __init__(self, in_dim: int, out_dim: int, activation: str = "ELU"):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.ln_activ = LayerNormActiv(out_dim, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln_activ(self.linear(x))


class StateEncoder(nn.Module):
    """
    State encoder f_ω: s → z_s.

    Supports vector and image observations.
    """

    def __init__(self, config: MRQConfig, state_dim: int):
        super().__init__()
        self.config = config
        self._image = config.observation_type == "image"

        if self._image:
            c = config.image_channels  # 3 for RGB (DMC), 1 for grayscale stack (Atari)
            self.cnn1 = nn.Conv2d(c, 32, 3, stride=2)
            self.cnn2 = nn.Conv2d(32, 32, 3, stride=2)
            self.cnn3 = nn.Conv2d(32, 32, 3, stride=2)
            self.cnn4 = nn.Conv2d(32, 32, 3, stride=1)
            # Compute flattened size for 84x84 input
            self._cnn_out_dim = 32 * 6 * 6  # after conv layers: 84->41->20->9->7 -> 32*7*7=1568? Let's compute
            # Actually from paper: "Assumes 84×84 input" and flatten is 1568
            # 84/2=42, 42/2=21, 21/2=10, 10/1=10. Hmm.
            # Let me compute: conv1: 84 -> ceil(84/2)=42. conv2: 42->21. conv3: 21->ceil(21/2)=11. conv4: 11->11
            # 32*11*11 = 3872. That's not 1568. 
            # Paper says self.zs_lin = nn.Linear(1568, zs_dim) for 84x84 input.
            # With 3 input channels (DMC visual): first conv takes 3 channels -> 32
            # After 4 convs with strides [2,2,2,1] on 84x84:
            # 84 -> (84-3)/2+1 = 41 -> (41-3)/2+1 = 20 -> (20-3)/2+1 = 9 -> (9-3)/1+1 = 7
            # 32 * 7 * 7 = 1568. Yes!
            self._cnn_out_dim = 1568
            self.zs_lin = nn.Linear(self._cnn_out_dim, config.zs_dim)
        else:
            self.mlp1 = MLPLayer(state_dim, config.hidden_dim, config.encoder_activation)
            self.mlp2 = MLPLayer(config.hidden_dim, config.hidden_dim, config.encoder_activation)
            self.mlp3 = MLPLayer(config.hidden_dim, config.zs_dim, config.encoder_activation)

        self.activ = nn.ELU()

    def _cnn_forward(self, state: torch.Tensor) -> torch.Tensor:
        state = state / 255.0 - 0.5
        zs = self.activ(self.cnn1(state))
        zs = self.activ(self.cnn2(zs))
        zs = self.activ(self.cnn3(zs))
        zs = self.activ(self.cnn4(zs))
        zs = zs.reshape(state.shape[0], self._cnn_out_dim)
        return ln_activ(self.zs_lin(zs), self.activ)

    def _mlp_forward(self, state: torch.Tensor) -> torch.Tensor:
        zs = self.mlp1(state)
        zs = self.mlp2(zs)
        return self.mlp3(zs)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if self._image:
            return self._cnn_forward(state)
        else:
            return self._mlp_forward(state)


class StateActionEncoder(nn.Module):
    """
    State-action encoder g_ω: (s, a) → z_sa.

    The model (MDP predictor) is a linear layer from z_sa to predictions:
    [next_z_s (zs_dim), reward_logits (reward_bins), terminal (1)].
    """

    def __init__(self, config: MRQConfig, action_dim: int):
        super().__init__()
        self.config = config
        self.za = nn.Linear(action_dim, config.za_dim)
        self.zsa1 = nn.Linear(config.zs_dim + config.za_dim, config.hidden_dim)
        self.zsa2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.zsa3 = nn.Linear(config.hidden_dim, config.zsa_dim)
        self.output_dim = config.zs_dim + config.reward_bins + 1
        self.model = nn.Linear(config.zsa_dim, self.output_dim)
        self.activ = nn.ELU()

    def forward(self, zs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        za = self.activ(self.za(action))
        zsa = torch.cat([zs, za], dim=1)
        zsa = ln_activ(self.zsa1(zsa), self.activ)
        zsa = ln_activ(self.zsa2(zsa), self.activ)
        zsa = self.zsa3(zsa)
        preds = self.model(zsa)
        return preds, zsa


class ValueNetwork(nn.Module):
    """Q-value network: 4-layer MLP with LayerNorm+ELU."""

    def __init__(self, config: MRQConfig):
        super().__init__()
        dim = config.hidden_dim
        self.l1 = nn.Linear(config.zsa_dim, dim)
        self.l2 = nn.Linear(dim, dim)
        self.l3 = nn.Linear(dim, dim)
        self.l4 = nn.Linear(dim, 1)
        self.activ = nn.ELU()

    def forward(self, zsa: torch.Tensor) -> torch.Tensor:
        q = ln_activ(self.l1(zsa), self.activ)
        q = ln_activ(self.l2(q), self.activ)
        q = ln_activ(self.l3(q), self.activ)
        return self.l4(q)


class PolicyNetwork(nn.Module):
    """
    Policy network π_φ: z_s → a.

    Supports continuous (Tanh) and discrete (Gumbel-Softmax) actions.
    """

    def __init__(self, config: MRQConfig, action_dim: int):
        super().__init__()
        self.config = config
        self.action_dim = action_dim

        self.l1 = nn.Linear(config.zs_dim, config.hidden_dim)
        self.l2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.l3 = nn.Linear(config.hidden_dim, action_dim)
        self.activ = nn.ReLU()

        if config.discrete_actions:
            self.final_activ = lambda x: F.gumbel_softmax(x, tau=config.gumbel_softmax_tau, hard=False)
        else:
            self.final_activ = torch.tanh

    def forward(self, zs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            action: sampled or deterministic action
            z_pi: pre-activation (for regularization loss)
        """
        a = ln_activ(self.l1(zs), self.activ)
        a = ln_activ(self.l2(a), self.activ)
        z_pi = self.l3(a)

        if self.config.discrete_actions:
            if deterministic:
                action = F.one_hot(z_pi.argmax(dim=-1), num_classes=self.action_dim).float()
            else:
                action = self.final_activ(z_pi)
        else:
            action = self.final_activ(z_pi)
        return action, z_pi
