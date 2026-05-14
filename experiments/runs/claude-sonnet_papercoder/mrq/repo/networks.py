```python
## networks.py
"""Neural network modules for MR.Q (Model-based Representations for Q-learning).

This module defines all network architectures described in Appendix B.2 of the
paper, plus the MRQNetworks container class that manages all networks, their
target copies, and optimizers.

Key design principles:
  - LayerNorm-before-activation (ln_activ pattern) throughout
  - Xavier uniform init for all linear/conv layers, zero bias everywhere
  - ELU for encoder and value networks, ReLU for policy network
  - Decoupled training: encoder, value, and policy have separate AdamW optimizers
  - Hard target network copies (not EMA), synced every T_target=250 steps
  - No activation on the zsa3 output (raw embedding approximately linear with value)

Ablation variants supported via Config flags:
  - cfg.value_linear: Replace non-linear value network with linear z_sa^T w
  - cfg.nonlinear_model: Replace linear MDP predictor with separate MLPs
"""

import copy
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


# ---------------------------------------------------------------------------
# Weight initialization helper
# ---------------------------------------------------------------------------


def _init_weights(module: nn.Module) -> None:
    """Apply Xavier uniform initialization to linear and conv layers.

    Called via module.apply(_init_weights) at the end of each network's
    __init__. Sets all biases to zero and applies Xavier uniform to all
    weight matrices, as specified in Table 3 (Appendix B.1) of the paper.

    Args:
        module: A single nn.Module instance. Only nn.Linear and nn.Conv2d
            layers are modified; all other module types are left unchanged.
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


# ---------------------------------------------------------------------------
# State Encoder: CNN variant (image observations)
# ---------------------------------------------------------------------------


class StateEncoderCNN(nn.Module):
    """Encodes image observations into a state embedding z_s.

    Used for DMC-Visual (9-channel 84×84 RGB stacks) and Atari (4-channel
    84×84 grayscale stacks). Architecture from Appendix B.2:

        Conv2d(in_channels, 32, 3, stride=2) → ELU
        Conv2d(32, 32, 3, stride=2)          → ELU
        Conv2d(32, 32, 3, stride=2)          → ELU
        Conv2d(32, 32, 3, stride=1)          → ELU
        Flatten → Linear(1568, zs_dim) → LayerNorm(zs_dim) → ELU

    Spatial dimension trace for 84×84 input:
        84 → 41 → 20 → 9 → 7  (after 4 conv layers)
        Flattened: 32 × 7 × 7 = 1568

    Input normalization (state / 255.0 - 0.5) is applied inside forward()
    to map uint8 [0, 255] pixel values to float32 [-0.5, 0.5].

    Attributes:
        conv1: First convolutional layer, stride=2.
        conv2: Second convolutional layer, stride=2.
        conv3: Third convolutional layer, stride=2.
        conv4: Fourth convolutional layer, stride=1.
        linear: Linear projection from flattened conv output to zs_dim.
        layer_norm: LayerNorm applied after the linear projection.
    """

    # Flattened size after 4 conv layers on 84×84 input: 32 × 7 × 7 = 1568
    _FLAT_SIZE: int = 1568

    def __init__(self, in_channels: int, zs_dim: int) -> None:
        """Initialise the CNN state encoder.

        Args:
            in_channels: Number of input channels. 9 for DMC-Visual
                (3 RGB frames × 3 channels), 4 for Atari (4 grayscale frames).
            zs_dim: Output state embedding dimension (512 per config.yaml).

        Raises:
            ValueError: If in_channels < 1 or zs_dim < 1.
        """
        super().__init__()

        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}.")
        if zs_dim < 1:
            raise ValueError(f"zs_dim must be >= 1, got {zs_dim}.")

        self.conv1: nn.Conv2d = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2)
        self.conv2: nn.Conv2d = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.conv3: nn.Conv2d = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.conv4: nn.Conv2d = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.linear: nn.Linear = nn.Linear(self._FLAT_SIZE, zs_dim)
        self.layer_norm: nn.LayerNorm = nn.LayerNorm(zs_dim)

        # Apply Xavier uniform init with zero bias to all linear/conv layers.
        self.apply(_init_weights)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Encode an image observation into a state embedding.

        Args:
            state: Image tensor of shape (batch, in_channels, 84, 84),
                dtype float32. Pixel values should be in [0, 255].

        Returns:
            State embedding of shape (batch, zs_dim), dtype float32.
        """
        # Normalize pixels from [0, 255] to [-0.5, 0.5].
        x: torch.Tensor = state / 255.0 - 0.5

        # Four convolutional layers with ELU activations.
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        # Flatten spatial dimensions: (batch, 32, 7, 7) → (batch, 1568)
        x = x.reshape(x.shape[0], -1)

        # Linear projection followed by LayerNorm then ELU (ln_activ pattern).
        x = self.linear(x)
        x = F.elu(self.layer_norm(x))

        return x  # shape: (batch, zs_dim)


# ---------------------------------------------------------------------------
# State Encoder: MLP variant (vector observations)
# ---------------------------------------------------------------------------


class StateEncoderMLP(nn.Module):
    """Encodes vector observations into a state embedding z_s.

    Used for Gym locomotion and DMC-Proprioceptive benchmarks. Architecture
    from Appendix B.2:

        Linear(state_dim, hidden_dim) → LayerNorm(hidden_dim) → ELU
        Linear(hidden_dim, hidden_dim) → LayerNorm(hidden_dim) → ELU
        Linear(hidden_dim, zs_dim)    → LayerNorm(zs_dim)    → ELU

    All three layers use the ln_activ pattern (LayerNorm before activation).
    With default config (hidden_dim=512, zs_dim=512), all layers have the
    same width.

    Attributes:
        mlp1: First linear layer mapping state_dim → hidden_dim.
        mlp2: Second linear layer mapping hidden_dim → hidden_dim.
        mlp3: Third linear layer mapping hidden_dim → zs_dim.
        ln1: LayerNorm for the first layer output.
        ln2: LayerNorm for the second layer output.
        ln3: LayerNorm for the third layer output.
    """

    def __init__(
        self, state_dim: int, zs_dim: int, hidden_dim: int = 512
    ) -> None:
        """Initialise the MLP state encoder.

        Args:
            state_dim: Dimensionality of the input observation vector.
            zs_dim: Output state embedding dimension (512 per config.yaml).
            hidden_dim: Hidden layer width (512 per config.yaml).

        Raises:
            ValueError: If state_dim < 1, zs_dim < 1, or hidden_dim < 1.
        """
        super().__init__()

        if state_dim < 1:
            raise ValueError(f"state_dim must be >= 1, got {state_dim}.")
        if zs_dim < 1:
            raise ValueError(f"zs_dim must be >= 1, got {zs_dim}.")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}.")

        self.mlp1: nn.Linear = nn.Linear(state_dim, hidden_dim)
        self.mlp2: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.mlp3: nn.Linear = nn.Linear(hidden_dim, zs_dim)

        self.ln1: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.ln2: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.ln3: nn.LayerNorm = nn.LayerNorm(zs_dim)

        # Apply Xavier uniform init with zero bias.
        self.apply(_init_weights)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Encode a vector observation into a state embedding.

        Args:
            state: Observation tensor of shape (batch, state_dim), float32.

        Returns:
            State embedding of shape (batch, zs_dim), dtype float32.
        """
        # Three layers each with LayerNorm then ELU (ln_activ pattern).
        x: torch.Tensor = F.elu(self.ln1(self.mlp1(state)))
        x = F.elu(self.ln2(self.mlp2(x)))
        x = F.elu(self.ln3(self.mlp3(x)))
        return x  # shape: (batch, zs_dim)


# ---------------------------------------------------------------------------
# State-Action Encoder with linear MDP predictor
# ---------------------------------------------------------------------------


class _NonlinearMDPPredictor(nn.Module):
    """Non-linear MDP predictor for the 'nonlinear_model' ablation.

    Replaces the single linear layer model = Linear(zsa_dim, output_dim)
    with three separate small MLPs predicting each output component
    independently from zsa. This tests whether the linear constraint on
    the MDP predictor matters for performance.

    Architecture (one MLP per output component):
        zs_next: Linear(zsa_dim, hidden_dim) → ELU → Linear(hidden_dim, zs_dim)
        r_logits: Linear(zsa_dim, hidden_dim) → ELU → Linear(hidden_dim, reward_bins)
        d_pred: Linear(zsa_dim, hidden_dim) → ELU → Linear(hidden_dim, 1)

    Attributes:
        zs_dim: Dimension of the predicted next state embedding.
        reward_bins: Number of reward bins for categorical prediction.
        zs_net: MLP predicting next state embedding.
        r_net: MLP predicting reward logits.
        d_net: MLP predicting terminal signal.
    """

    def __init__(
        self,
        zsa_dim: int,
        zs_dim: int,
        reward_bins: int,
        hidden_dim: int = 256,
    ) -> None:
        """Initialise the non-linear MDP predictor.

        Args:
            zsa_dim: Input state-action embedding dimension.
            zs_dim: Output dimension for next state prediction.
            reward_bins: Number of bins for categorical reward prediction.
            hidden_dim: Hidden layer width for each sub-MLP.
        """
        super().__init__()

        self.zs_dim: int = zs_dim
        self.reward_bins: int = reward_bins

        # Sub-MLP for next state embedding prediction.
        self.zs_net: nn.Sequential = nn.Sequential(
            nn.Linear(zsa_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, zs_dim),
        )

        # Sub-MLP for reward logits prediction.
        self.r_net: nn.Sequential = nn.Sequential(
            nn.Linear(zsa_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, reward_bins),
        )

        # Sub-MLP for terminal signal prediction.
        self.d_net: nn.Sequential = nn.Sequential(
            nn.Linear(zsa_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(_init_weights)

    def forward(self, zsa: torch.Tensor) -> torch.Tensor:
        """Predict MDP components from state-action embedding.

        Args:
            zsa: State-action embedding of shape (batch, zsa_dim).

        Returns:
            Concatenated output of shape (batch, zs_dim + reward_bins + 1).
            Components: [zs_next_pred | r_logits | d_pred].
        """
        zs_next: torch.Tensor = self.zs_net(zsa)
        r_logits: torch.Tensor = self.r_net(zsa)
        d_pred: torch.Tensor = self.d_net(zsa)
        return torch.cat([zs_next, r_logits, d_pred], dim=-1)


class StateActionEncoder(nn.Module):
    """Encodes (z_s, action) pairs into state-action embeddings z_sa.

    Also applies the linear MDP predictor m to produce predictions of the
    next state embedding, reward logits, and terminal signal. These
    predictions are used in the encoder loss (Section 4.2.1).

    Architecture from Appendix B.2:
        action → Linear(action_dim, za_dim) → ELU → z_a
        [z_s, z_a] → concat → shape (batch, zs_dim + za_dim)
        Linear(zs_dim + za_dim, hidden_dim) → LayerNorm → ELU
        Linear(hidden_dim, hidden_dim)       → LayerNorm → ELU
        Linear(hidden_dim, zsa_dim)          → (no activation) → z_sa
        Linear(zsa_dim, output_dim)          → model_output

    The model_output is split in agent.py as:
        zs_next_pred = model_output[:, :zs_dim]
        r_logits     = model_output[:, zs_dim:zs_dim+reward_bins]
        d_pred       = model_output[:, -1:]

    Critical: z_sa has NO activation after zsa3. This preserves the
    approximately linear relationship with the value function.

    Attributes:
        za_linear: Linear layer mapping action to action embedding z_a.
        zsa1: First MLP layer taking concatenated [z_s, z_a].
        zsa2: Second MLP layer.
        zsa3: Third MLP layer producing z_sa (no activation).
        model: Linear MDP predictor (or non-linear variant for ablation).
        ln1: LayerNorm for zsa1 output.
        ln2: LayerNorm for zsa2 output.
        nonlinear_model: Whether to use non-linear MDP predictor (ablation).
    """

    def __init__(
        self,
        action_dim: int,
        zs_dim: int,
        za_dim: int,
        zsa_dim: int,
        hidden_dim: int,
        output_dim: int,
        nonlinear_model: bool = False,
    ) -> None:
        """Initialise the state-action encoder.

        Args:
            action_dim: Dimensionality of the action vector (or number of
                discrete actions for one-hot encoding).
            zs_dim: State embedding dimension (512 per config.yaml).
            za_dim: Action embedding dimension (256 per config.yaml).
            zsa_dim: State-action embedding dimension (512 per config.yaml).
            hidden_dim: Hidden layer width (512 per config.yaml).
            output_dim: Output dimension of the linear MDP predictor.
                Should be zs_dim + reward_bins + 1 = 578 per config.yaml.
            nonlinear_model: If True, use separate MLPs for each MDP
                component instead of a single linear layer (ablation).

        Raises:
            ValueError: If any dimension is < 1.
        """
        super().__init__()

        for name, val in [
            ("action_dim", action_dim),
            ("zs_dim", zs_dim),
            ("za_dim", za_dim),
            ("zsa_dim", zsa_dim),
            ("hidden_dim", hidden_dim),
            ("output_dim", output_dim),
        ]:
            if val < 1:
                raise ValueError(f"{name} must be >= 1, got {val}.")

        self.nonlinear_model: bool = nonlinear_model
        self._zs_dim: int = zs_dim
        self._output_dim: int = output_dim

        # Action embedding: action → z_a
        self.za_linear: nn.Linear = nn.Linear(action_dim, za_dim)

        # State-action MLP: [z_s, z_a] → z_sa
        self.zsa1: nn.Linear = nn.Linear(zs_dim + za_dim, hidden_dim)
        self.zsa2: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.zsa3: nn.Linear = nn.Linear(hidden_dim, zsa_dim)

        # LayerNorm for the first two MLP layers (ln_activ pattern).
        self.ln1: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.ln2: nn.LayerNorm = nn.LayerNorm(hidden_dim)

        # Linear MDP predictor m: z_sa → (z_s', r_logits, d)
        # For the nonlinear_model ablation, use separate MLPs instead.
        if nonlinear_model:
            # Derive component sizes from output_dim.
            # output_dim = zs_dim + reward_bins + 1
            reward_bins: int = output_dim - zs_dim - 1
            self.model: nn.Module = _NonlinearMDPPredictor(
                zsa_dim=zsa_dim,
                zs_dim=zs_dim,
                reward_bins=reward_bins,
                hidden_dim=hidden_dim // 2,  # smaller sub-MLPs
            )
        else:
            self.model = nn.Linear(zsa_dim, output_dim)

        # Apply Xavier uniform init with zero bias to all linear/conv layers.
        # Note: _NonlinearMDPPredictor applies its own _init_weights internally.
        self.apply(_init_weights)

    def forward(
        self, zs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode (z_s, action) into z_sa and MDP predictions.

        Args:
            zs: State embedding of shape (batch, zs_dim), float32.
            action: Action tensor of shape (batch, action_dim), float32.
                For discrete environments: soft one-hot from Gumbel-Softmax.
                For continuous environments: values in [-1, 1].

        Returns:
            Tuple of:
                model_output (torch.Tensor): MDP predictor output of shape
                    (batch, output_dim). Split in agent.py as:
                    [:, :zs_dim] → predicted next state embedding
                    [:, zs_dim:zs_dim+reward_bins] → reward logits
                    [:, -1:] → predicted terminal signal
                zsa (torch.Tensor): State-action embedding of shape
                    (batch, zsa_dim). No activation applied — raw embedding
                    approximately linear with the value function.
        """
        # Action embedding: (batch, action_dim) → (batch, za_dim)
        za: torch.Tensor = F.elu(self.za_linear(action))

        # Concatenate state and action embeddings.
        # (batch, zs_dim) + (batch, za_dim) → (batch, zs_dim + za_dim)
        zsa: torch.Tensor = torch.cat([zs, za], dim=1)

        # Two MLP layers with LayerNorm then ELU (ln_activ pattern).
        zsa = F.elu(self.ln1(self.zsa1(zsa)))
        zsa = F.elu(self.ln2(self.zsa2(zsa)))

        # Final projection to z_sa — NO activation (critical design choice).
        zsa = self.zsa3(zsa)  # shape: (batch, zsa_dim)

        # Linear MDP predictor: z_sa → (z_s', r_logits, d)
        model_output: torch.Tensor = self.model(zsa)  # shape: (batch, output_dim)

        return model_output, zsa


# ---------------------------------------------------------------------------
# Value Network
# ---------------------------------------------------------------------------


class _LinearValueNetwork(nn.Module):
    """Linear value function for the 'linear_value' ablation.

    Replaces the non-linear 4-layer MLP with a single linear layer:
        Q(z_sa) = z_sa^T w

    This tests whether the non-linear value function is necessary, as
    discussed in Section 4.1 (Theorem 1 shows model-free and model-based
    objectives converge to the same solution in the linear case).

    Attributes:
        linear: Single linear layer mapping zsa_dim → 1.
    """

    def __init__(self, zsa_dim: int) -> None:
        """Initialise the linear value network.

        Args:
            zsa_dim: State-action embedding dimension.
        """
        super().__init__()
        self.linear: nn.Linear = nn.Linear(zsa_dim, 1)
        self.apply(_init_weights)

    def forward(self, zsa: torch.Tensor) -> torch.Tensor:
        """Compute linear Q-value estimate.

        Args:
            zsa: State-action embedding of shape (batch, zsa_dim).

        Returns:
            Q-value of shape (batch, 1).
        """
        return self.linear(zsa)


class ValueNetwork(nn.Module):
    """Maps state-action embeddings to scalar Q-values.

    Two identical instances are created (value1, value2) for double
    Q-learning (TD3-style). Architecture from Appendix B.2:

        Linear(zsa_dim, hidden_dim) → LayerNorm → ELU
        Linear(hidden_dim, hidden_dim) → LayerNorm → ELU
        Linear(hidden_dim, hidden_dim) → LayerNorm → ELU
        Linear(hidden_dim, 1)

    The value network receives z_sa that has been detached from the encoder
    computation graph in agent.py. Gradients do not flow back into the encoder.

    Attributes:
        l1: First linear layer.
        l2: Second linear layer.
        l3: Third linear layer.
        l4: Output linear layer mapping to scalar Q-value.
        ln1: LayerNorm for l1 output.
        ln2: LayerNorm for l2 output.
        ln3: LayerNorm for l3 output.
    """

    def __init__(self, zsa_dim: int, hidden_dim: int = 512) -> None:
        """Initialise the value network.

        Args:
            zsa_dim: State-action embedding dimension (512 per config.yaml).
            hidden_dim: Hidden layer width (512 per config.yaml).

        Raises:
            ValueError: If zsa_dim < 1 or hidden_dim < 1.
        """
        super().__init__()

        if zsa_dim < 1:
            raise ValueError(f"zsa_dim must be >= 1, got {zsa_dim}.")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}.")

        self.l1: nn.Linear = nn.Linear(zsa_dim, hidden_dim)
        self.l2: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.l3: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.l4: nn.Linear = nn.Linear(hidden_dim, 1)

        self.ln1: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.ln2: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.ln3: nn.LayerNorm = nn.LayerNorm(hidden_dim)

        self.apply(_init_weights)

    def forward(self, zsa: torch.Tensor) -> torch.Tensor:
        """Compute Q-value from state-action embedding.

        Args:
            zsa: State-action embedding of shape (batch, zsa_dim), float32.

        Returns:
            Q-value of shape (batch, 1), float32.
        """
        q: torch.Tensor = F.elu(self.ln1(self.l1(zsa)))
        q = F.elu(self.ln2(self.l2(q)))
        q = F.elu(self.ln3(self.l3(q)))
        q = self.l4(q)
        return q  # shape: (batch, 1)


# ---------------------------------------------------------------------------
# Policy Network
# ---------------------------------------------------------------------------


class PolicyNetwork(nn.Module):
    """Maps state embeddings to actions via deterministic policy gradient.

    Returns both the pre-activation output z_pi (for regularization) and
    the activated action a_pi (for Q-value computation). Architecture from
    Appendix B.2:

        Linear(zs_dim, hidden_dim) → LayerNorm → ReLU   ← ReLU, not ELU
        Linear(hidden_dim, hidden_dim) → LayerNorm → ReLU
        Linear(hidden_dim, action_dim)                   ← z_pi (pre-activation)
        Tanh(z_pi)  or  GumbelSoftmax(z_pi, tau=10)     ← a_pi

    Note: The policy uses ReLU activations, unlike the encoder and value
    networks which use ELU. This is explicitly specified in Table 3.

    For discrete actions, Gumbel-Softmax (Jang et al., 2017) with tau=10
    produces a differentiable soft one-hot, enabling the deterministic
    policy gradient to work with discrete action spaces.

    Attributes:
        l1: First linear layer.
        l2: Second linear layer.
        l3: Output linear layer mapping to action_dim (pre-activation).
        ln1: LayerNorm for l1 output.
        ln2: LayerNorm for l2 output.
        discrete: Whether the action space is discrete.
        gumbel_tau: Gumbel-Softmax temperature (10.0 per config.yaml).
    """

    def __init__(
        self,
        zs_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        discrete: bool = False,
        gumbel_tau: float = 10.0,
    ) -> None:
        """Initialise the policy network.

        Args:
            zs_dim: State embedding dimension (512 per config.yaml).
            action_dim: Number of action dimensions (continuous) or number
                of discrete actions (Atari).
            hidden_dim: Hidden layer width (512 per config.yaml).
            discrete: If True, use Gumbel-Softmax final activation.
                If False, use Tanh.
            gumbel_tau: Gumbel-Softmax temperature (10.0 per config.yaml).
                Higher values produce softer distributions.

        Raises:
            ValueError: If zs_dim < 1, action_dim < 1, hidden_dim < 1,
                or gumbel_tau <= 0.
        """
        super().__init__()

        if zs_dim < 1:
            raise ValueError(f"zs_dim must be >= 1, got {zs_dim}.")
        if action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {action_dim}.")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}.")
        if gumbel_tau <= 0.0:
            raise ValueError(f"gumbel_tau must be > 0, got {gumbel_tau}.")

        self.discrete: bool = discrete
        self.gumbel_tau: float = gumbel_tau

        self.l1: nn.Linear = nn.Linear(zs_dim, hidden_dim)
        self.l2: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.l3: nn.Linear = nn.Linear(hidden_dim, action_dim)

        # LayerNorm for the first two layers (ln_activ pattern with ReLU).
        self.ln1: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.ln2: nn.LayerNorm = nn.LayerNorm(hidden_dim)

        self.apply(_init_weights)

    def forward(
        self, zs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute policy action from state embedding.

        Args:
            zs: State embedding of shape (batch, zs_dim), float32.
                Should be detached from the encoder computation graph
                when used for policy gradient updates.

        Returns:
            Tuple of:
                z_pi (torch.Tensor): Pre-activation output of shape
                    (batch, action_dim). Used in the policy loss for
                    pre-activation regularization: λ_pre_activ * ||z_pi||².
                a_pi (torch.Tensor): Activated action of shape
                    (batch, action_dim). For continuous: Tanh output in
                    [-1, 1]. For discrete: Gumbel-Softmax soft one-hot.
        """
        # Two hidden layers with LayerNorm then ReLU (ln_activ with ReLU).
        a: torch.Tensor = F.relu(self.ln1(self.l1(zs)))
        a = F.relu(self.ln2(self.l2(a)))

        # Pre-activation output (no activation applied yet).
        z_pi: torch.Tensor = self.l3(a)  # shape: (batch, action_dim)

        # Apply final activation based on action space type.
        if self.discrete:
            # Gumbel-Softmax: differentiable discrete action selection.
            # hard=False produces a soft one-hot for gradient flow.
            # tau=10.0 makes the distribution relatively sharp.
            a_pi: torch.Tensor = F.gumbel_softmax(
                z_pi, tau=self.gumbel_tau, hard=False
            )
        else:
            # Tanh: maps pre-activation to [-1, 1] for continuous actions.
            a_pi = torch.tanh(z_pi)

        return z_pi, a_pi  # (pre-activation, activated action)


# ---------------------------------------------------------------------------
# MRQNetworks: Container for all networks, targets, and optimizers
# ---------------------------------------------------------------------------


class MRQNetworks(nn.Module):
    """Container managing all MR.