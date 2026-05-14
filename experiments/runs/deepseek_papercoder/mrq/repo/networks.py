# networks.py

"""
Neural network architectures for the MR.Q algorithm.

Defines the following classes:
    - StateEncoder          : maps raw observations to state embeddings z_s.
    - StateActionEncoder    : maps (z_s, action) to state-action embeddings z_sa.
    - LinearPredictor       : predicts next state embedding, reward logits, and terminal signal.
    - QNetwork              : value function from z_sa to scalar Q-value.
    - PolicyNetwork         : maps z_s to an action (continuous Tanh or discrete Gumbel-Softmax).
    - Encoder               : container combining StateEncoder and StateActionEncoder.

All architectures follow exactly the paper's Appendix B.2. Hyperparameters (dimensions, etc.)
are taken from a `Config` object that mirrors `config.yaml`.

Utility functions:
    - init_weights          : Xavier uniform initialisation for Linear/Conv2d, bias=0.
    - copy_network_parameters : hard copy of parameters from one network to another.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Import the configuration class (avoid circular imports because config.py does not import this module)
from config import Config


# ------------------------------------------------------------------------------
# Weight initialisation
# ------------------------------------------------------------------------------

def init_weights(m: nn.Module) -> None:
    """
    Recursively initialise the weights of a module.

    Linear and Conv2d layers receive Xavier uniform weights and bias=0.
    LayerNorm layers are left at their default PyTorch initialisation.
    """
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        # Weight and bias of LayerNorm are already initialised to 1 and 0 by default.
        pass


# ------------------------------------------------------------------------------
# Parameter copying utility
# ------------------------------------------------------------------------------

def copy_network_parameters(source_net: nn.Module, target_net: nn.Module) -> None:
    """
    Hard copy all parameters from source_net to target_net.

    Parameters
    ----------
    source_net : nn.Module
        Network whose parameters are to be copied.
    target_net : nn.Module
        Network that receives the parameters.
    """
    target_net.load_state_dict(source_net.state_dict())


# ==============================================================================
# State Encoder – image or vector input → z_s (512‑dim)
# ==============================================================================

class StateEncoder(nn.Module):
    """
    Produces a state embedding z_s from raw observations.

    Two mutually exclusive pathways are supported:
      - Image pathway (for Atari, DM Control visual): a stack of convolutional layers
        followed by a linear projection, LayerNorm, and ELU activation.
      - Vector pathway (for Gym locomotion, DM Control proprioceptive): a 3‑layer MLP
        with LayerNorm and ELU after each layer.

    Parameters
    ----------
    cfg : Config
        Global configuration object. Used fields:
            - common.zs_dim (int)        : dimension of z_s (default 512).
            - common.hidden_dim (int)    : MLP hidden units (default 512).
    is_image : bool
        Whether the observation is image (True) or flat vector (False).
    channels : int, optional
        Number of input channels for the image branch. Required when is_image=True.
    state_dim : int, optional
        Dimensionality of the flat state vector. Required when is_image=False.
    """

    def __init__(
        self,
        cfg: Config,
        is_image: bool,
        channels: int = None,
        state_dim: int = None,
    ):
        super().__init__()
        self.is_image = is_image
        self.zs_dim = cfg.zs_dim

        if self.is_image:
            if channels is None:
                raise ValueError("channels must be provided for image input")
            # Convolutional stack exactly as in Appendix B.2
            self.cnn1 = nn.Conv2d(channels, 32, kernel_size=3, stride=2)  # -> 41x41
            self.cnn2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)        # -> 20x20
            self.cnn3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)        # -> 9x9
            self.cnn4 = nn.Conv2d(32, 32, kernel_size=3, stride=1)        # -> 7x7
            # Flatten dimension: 7 * 7 * 32 = 1568
            self.linear = nn.Linear(1568, self.zs_dim)
            self.ln = nn.LayerNorm(self.zs_dim)
            self.activ = nn.ELU()
        else:
            if state_dim is None:
                raise ValueError("state_dim must be provided for vector input")
            hidden_dim = cfg.hidden_dim
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, self.zs_dim)
            self.ln3 = nn.LayerNorm(self.zs_dim)
            self.activ = nn.ELU()

        # Apply Xavier uniform init with bias=0 to all linear/conv layers
        self.apply(init_weights)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state : torch.Tensor
            - Image: shape (B, C, H, W), dtype float, values in [0, 255].
            - Vector: shape (B, state_dim), dtype float.

        Returns
        -------
        z_s : torch.Tensor, shape (B, zs_dim)
        """
        if self.is_image:
            # Scale pixel values to [0, 1] and subtract 0.5 (as in the code block)
            x = state.float() / 255.0 - 0.5
            x = self.activ(self.cnn1(x))
            x = self.activ(self.cnn2(x))
            x = self.activ(self.cnn3(x))
            x = self.activ(self.cnn4(x))
            x = x.view(x.size(0), -1)          # flatten to (B, 1568)
            x = self.linear(x)
            x = self.ln(x)
            x = self.activ(x)
            return x
        else:
            x = self.fc1(state)
            x = self.ln1(x)
            x = self.activ(x)
            x = self.fc2(x)
            x = self.ln2(x)
            x = self.activ(x)
            x = self.fc3(x)
            x = self.ln3(x)
            x = self.activ(x)
            return x


# ==============================================================================
# State‑Action Encoder – (z_s, action) → z_sa (512‑dim)
# ==============================================================================

class StateActionEncoder(nn.Module):
    """
    Maps a state embedding and an action to a state‑action embedding z_sa.

    Architecture:
        - action embedding: Linear(action_dim, za_dim) -> ELU
        - concatenate [z_s ; za]
        - 3‑layer MLP (hidden_dim each, LayerNorm + ELU after first two)
        - final linear layer (no activation, no LayerNorm) → z_sa

    Parameters
    ----------
    cfg : Config
        Global configuration. Used fields:
            - common.za_dim (int)      : action embedding dimension (default 256).
            - common.hidden_dim (int)  : hidden units (default 512).
            - common.zsa_dim (int)     : output dimension (default 512).
    action_dim : int
        Dimensionality of the environment's action space.
    """

    def __init__(self, cfg: Config, action_dim: int):
        super().__init__()
        za_dim = cfg.za_dim
        hidden_dim = cfg.hidden_dim
        zsa_dim = cfg.zsa_dim

        self.action_embed = nn.Linear(action_dim, za_dim)
        self.fc1 = nn.Linear(cfg.zs_dim + za_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, zsa_dim)        # no LN/activation after this
        self.activ = nn.ELU()

        self.apply(init_weights)

    def forward(self, z_s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z_s : torch.Tensor, shape (B, zs_dim)
        action : torch.Tensor, shape (B, action_dim)

        Returns
        -------
        z_sa : torch.Tensor, shape (B, zsa_dim)
        """
        za = self.activ(self.action_embed(action))          # (B, za_dim)
        x = torch.cat([z_s, za], dim=1)                     # (B, zs_dim+za_dim)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.activ(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.activ(x)
        z_sa = self.fc3(x)                                  # no activation/LN
        return z_sa


# ==============================================================================
# Linear Predictor – from z_sa to next state embedding, reward logits, terminal
# ==============================================================================

class LinearPredictor(nn.Module):
    """
    A single linear layer that maps z_sa into three predictions:
        - next state embedding z_next (zsa_dim dimensions)
        - reward logits (65 bins for two‑hot categorical loss)
        - terminal probability (scalar)

    Parameters
    ----------
    cfg : Config
        Global configuration. Used fields:
            - common.zsa_dim (int)       : dimension of z_sa (default 512).
            - common.reward_bins (int)   : number of reward bins (default 65).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        output_dim = cfg.zsa_dim + cfg.reward_bins + 1       # 512 + 65 + 1 = 578
        self.model = nn.Linear(cfg.zsa_dim, output_dim)
        self.zsa_dim = cfg.zsa_dim
        self.reward_bins = cfg.reward_bins
        self.apply(init_weights)                              # bias set to 0

    def forward(
        self, z_sa: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        z_sa : torch.Tensor, shape (B, zsa_dim)

        Returns
        -------
        z_next : torch.Tensor, shape (B, zsa_dim)
        reward_logits : torch.Tensor, shape (B, reward_bins)
        terminal : torch.Tensor, shape (B, 1)
        """
        out = self.model(z_sa)                                # (B, 578)
        z_next = out[:, : self.zsa_dim]                       # (B, 512)
        reward_logits = out[:, self.zsa_dim : self.zsa_dim + self.reward_bins]  # (B, 65)
        terminal = out[:, -1:]                                # (B, 1)
        return z_next, reward_logits, terminal


# ==============================================================================
# Value Network (Q‑function) – z_sa → scalar Q‑value
# ==============================================================================

class QNetwork(nn.Module):
    """
    Multi‑layer perceptron mapping z_sa to a single Q‑value.

    Architecture (four layers):
        - Linear(zsa_dim, hidden) → LayerNorm → ELU
        - Linear(hidden, hidden)   → LayerNorm → ELU
        - Linear(hidden, hidden)   → LayerNorm → ELU
        - Linear(hidden, 1)        (no activation/norm)

    Two independent instances are used (twin critics).

    Parameters
    ----------
    cfg : Config
        Global configuration. Used fields:
            - common.zsa_dim (int)       : input dimension (default 512).
            - common.hidden_dim (int)    : hidden layer size (default 512).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        hidden_dim = cfg.hidden_dim
        self.fc1 = nn.Linear(cfg.zsa_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.activ = nn.ELU()
        self.apply(init_weights)

    def forward(self, z_sa: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z_sa : torch.Tensor, shape (B, zsa_dim)

        Returns
        -------
        value : torch.Tensor, shape (B, 1)
        """
        x = self.fc1(z_sa)
        x = self.ln1(x)
        x = self.activ(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.activ(x)
        x = self.fc3(x)
        x = self.ln3(x)
        x = self.activ(x)
        value = self.fc4(x)
        return value


# ==============================================================================
# Policy Network – z_s → action (continuous or discrete)
# ==============================================================================

class PolicyNetwork(nn.Module):
    """
    Deterministic policy mapping state embedding z_s to an action.

    For continuous actions the final activation is Tanh.
    For discrete actions the final activation is Gumbel‑Softmax
    (providing a differentiable one‑hot approximation).

    The function also returns the pre‑activation logits, which are used
    for the pre‑activation regularisation loss.

    Parameters
    ----------
    cfg : Config
        Global configuration. Used fields:
            - common.zs_dim (int)        : input dimension (default 512).
            - common.hidden_dim (int)    : hidden layer size (default 512).
            - policy.gumbel_softmax_tau (float) : τ for Gumbel‑Softmax (default 10).
    action_dim : int
        Dimensionality of the action space.
    discrete : bool
        Whether the action space is discrete (True) or continuous (False).
    """

    def __init__(self, cfg: Config, action_dim: int, discrete: bool):
        super().__init__()
        hidden_dim = cfg.hidden_dim
        self.discrete = discrete
        self.action_dim = action_dim

        self.fc1 = nn.Linear(cfg.zs_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)     # pre‑activation logits
        self.activ = nn.ReLU()

        if self.discrete:
            self.tau = cfg.gumbel_softmax_tau            # from config
        else:
            self.tau = None

        self.apply(init_weights)

    def forward(
        self, z_s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        z_s : torch.Tensor, shape (B, zs_dim)

        Returns
        -------
        action : torch.Tensor
            Continuous: shape (B, action_dim) in [-1, 1].
            Discrete:  shape (B, action_dim), differentiable one‑hot vectors.
        pre_activ : torch.Tensor
            The raw logits before the final activation (for regularization).
        """
        x = self.fc1(z_s)
        x = self.ln1(x)
        x = self.activ(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.activ(x)
        pre_activ = self.fc3(x)                         # (B, action_dim)

        if self.discrete:
            # Use Gumbel-Softmax with hard=False (returns soft one‑hot)
            action = F.gumbel_softmax(
                pre_activ, tau=self.tau, hard=False, dim=-1
            )
        else:
            action = torch.tanh(pre_activ)
        return action, pre_activ


# ==============================================================================
# Encoder (container) – combines StateEncoder and StateActionEncoder
# ==============================================================================

class Encoder(nn.Module):
    """
    Container for the full state and state‑action encoders.

    Provides convenience methods:
        - forward(state, action) → (z_s, z_sa)
        - forward_state(state)  → z_s
        - forward_sa(z_s, action) → z_sa

    The encoder itself holds no trainable parameters; those belong to
    the underlying StateEncoder and StateActionEncoder modules.

    Parameters
    ----------
    cfg : Config
        Global configuration.
    observation_space : gym.spaces.Box
        Used to detect observation type (image vs vector) and dimensions.
    action_dim : int
        Dimensionality of the action space.
    """

    def __init__(self, cfg: Config, observation_space, action_dim: int):
        super().__init__()
        # Determine observation modality from the Gym space
        if len(observation_space.shape) == 3:
            # Image observation: (C, H, W)
            is_image = True
            channels = observation_space.shape[0]
            state_dim = None
        elif len(observation_space.shape) == 1:
            # Flat vector observation
            is_image = False
            channels = None
            state_dim = observation_space.shape[0]
        else:
            raise ValueError(
                f"Unsupported observation space shape {observation_space.shape}"
            )

        self.state_enc = StateEncoder(
            cfg, is_image=is_image, channels=channels, state_dim=state_dim
        )
        self.sa_enc = StateActionEncoder(cfg, action_dim)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both z_s and z_sa from raw state and action.

        Returns
        -------
        z_s : torch.Tensor, shape (B, zs_dim)
        z_sa : torch.Tensor, shape (B, zsa_dim)
        """
        z_s = self.state_enc(state)
        z_sa = self.sa_enc(z_s, action)
        return z_s, z_sa

    def forward_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode only the state into z_s.
        """
        return self.state_enc(state)

    def forward_sa(
        self, z_s: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode a pre‑computed state embedding and an action into z_sa.
        """
        return self.sa_enc(z_s, action)
