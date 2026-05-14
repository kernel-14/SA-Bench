## models/policy.py
"""Policy and value function networks for MBPO-PPO in the RWM project.

This module implements the policy network (``PolicyNetwork``) and value
function network (``ValueNetwork``) used in the MBPO-PPO policy optimization
framework described in Section 3.3 and Table S9 of the paper.

Both networks are simple 3-hidden-layer MLPs with ELU activations, operating
on the **policy observation** space (Table S5), which differs from the world
model observation space (Table S2):

  - Policy obs (ANYmal D, 48-dim): base_lin_vel, base_ang_vel, gravity,
    velocity_command, joint_pos, joint_vel, last_actions
  - Policy obs (Unitree G1, 99-dim): same structure with G1 joint counts

The policy observation is constructed externally by
``MBPOPPOTrainer.imagine_trajectories`` via ``BaseEnv.construct_policy_obs``,
which extracts the relevant fields from the world model observation and
appends the velocity command and last action. The networks themselves are
agnostic to this construction — they receive a flat ``[B, obs_dim]`` tensor.

Architecture (Table S9):
  - Policy: MLP, hidden [128, 128, 128], ELU, Gaussian output
  - Value function: MLP, hidden [128, 128, 128], ELU, scalar output

Training parameters (Table S11):
  - Learning rate: 0.001 (``mbpo_ppo.learning_rate``)
  - Clip range ε: 0.2 (``mbpo_ppo.clip_range``)
  - Entropy coefficient: 0.005 (``mbpo_ppo.entropy_coefficient``)
  - KL divergence target: 0.01 (``mbpo_ppo.kl_divergence_target``)

Usage:
    policy = PolicyNetwork(obs_dim=48, action_dim=12, hidden_sizes=[128, 128, 128])
    value_fn = ValueNetwork(obs_dim=48, hidden_sizes=[128, 128, 128])

    # During imagination rollouts:
    action, log_prob = policy.get_action(policy_obs)
    value = value_fn(policy_obs)

    # During PPO update:
    new_log_probs, entropy = policy.evaluate_actions(obs_batch, action_batch)
    new_values = value_fn(obs_batch)
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Log-std clamping bounds for numerical stability.
# Applied to self.log_std before computing exp(log_std) to prevent:
#   - std collapsing to zero (logstd << -5 → std ≈ 0 → NaN in log_prob)
#   - std exploding (logstd >> 2 → std >> 7 → policy outputs random noise)
# Range [-5, 2] corresponds to std in [exp(-5), exp(2)] ≈ [0.007, 7.4],
# which covers all physically meaningful action ranges for joint position
# targets in locomotion tasks.
# ---------------------------------------------------------------------------
_LOG_STD_MIN: float = -5.0
_LOG_STD_MAX: float = 2.0

# Small epsilon added to std for numerical safety in Normal distribution.
# Prevents std from being exactly zero even after clamping, which would
# cause division-by-zero in log_prob computation.
_STD_EPS: float = 1e-8


def _build_mlp_backbone(
    input_size: int,
    hidden_sizes: List[int],
    activation: str = "elu",
) -> nn.Sequential:
    """Build a multi-layer MLP backbone with the specified activation.

    Constructs a sequence of Linear → Activation layers for each hidden
    size in ``hidden_sizes``. The output of the backbone is the last hidden
    layer's activation — no output head is included (heads are added
    separately in each network class).

    Architecture for hidden_sizes=[128, 128, 128]:
        Linear(input_size → 128) → ELU
        Linear(128 → 128) → ELU
        Linear(128 → 128) → ELU
    Output: 128-dim feature vector

    This matches Table S9: "policy, MLP, 128 128 128, ELU" and
    "value function, MLP, 128 128 128, ELU".

    Args:
        input_size: Input feature dimension. For policy and value function,
            this is ``policy_obs_dim`` (48 for ANYmal D, 99 for Unitree G1,
            from ``config.anymal_d.policy_obs_dim`` or
            ``config.unitree_g1.policy_obs_dim``).
        hidden_sizes: List of hidden layer sizes. From ``config.policy.hidden_sizes``
            = [128, 128, 128] and ``config.value_function.hidden_sizes`` = [128, 128, 128]
            (Table S9). Must be non-empty.
        activation: Activation function name. From ``config.policy.activation``
            = "elu" (Table S9). Supported values: "elu", "relu", "tanh".
            Default: "elu".

    Returns:
        An ``nn.Sequential`` module implementing the MLP backbone.
        Output dimension equals ``hidden_sizes[-1]`` (128 for the paper's
        default configuration).

    Raises:
        ValueError: If ``hidden_sizes`` is empty.
        ValueError: If ``activation`` is not a supported activation name.
    """
    if not hidden_sizes:
        raise ValueError(
            "hidden_sizes must be non-empty. "
            "Check config.policy.hidden_sizes or config.value_function.hidden_sizes "
            "in config.yaml. The paper uses [128, 128, 128] (Table S9)."
        )

    # Resolve activation function from string name.
    # Using module instances (not classes) so they can be added to Sequential.
    _activation_map = {
        "elu": nn.ELU(),
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
    }
    activation_lower: str = activation.lower()
    if activation_lower not in _activation_map:
        raise ValueError(
            f"Unsupported activation '{activation}'. "
            f"Supported activations: {list(_activation_map.keys())}. "
            "Check config.policy.activation in config.yaml. "
            "The paper uses 'elu' (Table S9)."
        )

    # Build layers: Linear → Activation for each hidden size.
    layers: List[nn.Module] = []
    in_dim: int = input_size
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(in_dim, hidden_size))
        # Create a fresh activation instance for each layer to avoid
        # sharing state between layers (important for stateful activations,
        # though ELU/ReLU/Tanh are stateless).
        layers.append(type(_activation_map[activation_lower])())
        in_dim = hidden_size

    return nn.Sequential(*layers)


class PolicyNetwork(nn.Module):
    """Gaussian policy network for MBPO-PPO continuous control.

    Implements the policy network from Table S9 of the paper. The network
    maps policy observations to a Gaussian action distribution with:
      - State-dependent mean: ``mean = mean_head(backbone(obs))``
      - State-independent log std: ``log_std`` (learned parameter, shape
        ``[action_dim]``, initialized to zeros → initial std = 1.0)

    The state-independent log std is the standard PPO choice for continuous
    control. It provides a global exploration level that adapts during
    training without depending on the current observation, which stabilizes
    early training when the policy is far from optimal.

    The policy operates on the **policy observation** (Table S5), not the
    world model observation (Table S2). The policy obs is constructed by
    ``BaseEnv.construct_policy_obs`` in ``MBPOPPOTrainer.imagine_trajectories``
    and ``MBPOPPOTrainer.collect_real_data``.

    **No tanh squashing** is applied to actions. For joint position targets
    in locomotion, the action space is bounded by the PD controller limits
    (enforced in the environment), not by a tanh transformation. This
    simplifies the log_prob computation (no change-of-variables correction).

    Attributes:
        obs_dim: Policy observation dimension. 48 for ANYmal D
            (``config.anymal_d.policy_obs_dim``), 99 for Unitree G1
            (``config.unitree_g1.policy_obs_dim``).
        action_dim: Action space dimension. 12 for ANYmal D
            (``config.anymal_d.action_dim``), 29 for Unitree G1
            (``config.unitree_g1.action_dim``).
        hidden_sizes: List of hidden layer sizes. [128, 128, 128] (Table S9).
        net: 3-hidden-layer MLP backbone with ELU activations.
            Input: obs_dim, output: hidden_sizes[-1] = 128.
        mean_head: Linear layer mapping backbone output to action mean.
            Input: 128, output: action_dim.
        log_std: Learned state-independent log standard deviation.
            Shape: ``[action_dim]``. Initialized to zeros (std = 1.0).
            Clamped to ``[_LOG_STD_MIN, _LOG_STD_MAX]`` before use.
    """

    def __init__(
        self,
        obs_dim: int = 48,
        action_dim: int = 12,
        hidden_sizes: List[int] = None,
        activation: str = "elu",
    ) -> None:
        """Initialize the policy network.

        Builds the MLP backbone, mean head, and log_std parameter. The
        backbone and mean head are initialized with PyTorch's default
        Kaiming uniform initialization (appropriate for ELU activations).
        The log_std parameter is initialized to zeros.

        Args:
            obs_dim: Policy observation dimension. From
                ``config.anymal_d.policy_obs_dim`` (48) or
                ``config.unitree_g1.policy_obs_dim`` (99) in config.yaml.
                Must be positive. Default: 48 (ANYmal D).
            action_dim: Action space dimension. From
                ``config.anymal_d.action_dim`` (12) or
                ``config.unitree_g1.action_dim`` (29) in config.yaml.
                Must be positive. Default: 12 (ANYmal D).
            hidden_sizes: List of hidden layer sizes for the MLP backbone.
                From ``config.policy.hidden_sizes`` = [128, 128, 128]
                (Table S9). If None, defaults to [128, 128, 128]. Default: None.
            activation: Activation function name. From
                ``config.policy.activation`` = "elu" (Table S9).
                Supported: "elu", "relu", "tanh". Default: "elu".

        Raises:
            ValueError: If ``obs_dim`` <= 0 or ``action_dim`` <= 0.
            ValueError: If ``hidden_sizes`` is empty.
            ValueError: If ``activation`` is not supported.
        """
        super().__init__()

        # Apply default for mutable default argument
        if hidden_sizes is None:
            hidden_sizes = [128, 128, 128]

        # ----------------------------------------------------------------
        # Validate dimensions
        # ----------------------------------------------------------------
        if obs_dim <= 0:
            raise ValueError(
                f"obs_dim must be positive, got {obs_dim}. "
                "Check config.anymal_d.policy_obs_dim or "
                "config.unitree_g1.policy_obs_dim in config.yaml. "
                "ANYmal D: policy_obs_dim=48, Unitree G1: policy_obs_dim=99 (Table S5)."
            )
        if action_dim <= 0:
            raise ValueError(
                f"action_dim must be positive, got {action_dim}. "
                "Check config.anymal_d.action_dim or "
                "config.unitree_g1.action_dim in config.yaml. "
                "ANYmal D: action_dim=12, Unitree G1: action_dim=29 (Table S4)."
            )

        # ----------------------------------------------------------------
        # Store configuration
        # ----------------------------------------------------------------
        self.obs_dim: int = int(obs_dim)
        self.action_dim: int = int(action_dim)
        self.hidden_sizes: List[int] = list(hidden_sizes)

        # ----------------------------------------------------------------
        # Build MLP backbone (Table S9: "MLP, 128 128 128, ELU")
        # ----------------------------------------------------------------
        # Output dimension: hidden_sizes[-1] = 128
        self.net: nn.Sequential = _build_mlp_backbone(
            input_size=self.obs_dim,
            hidden_sizes=self.hidden_sizes,
            activation=activation,
        )

        # ----------------------------------------------------------------
        # Build action mean head
        # ----------------------------------------------------------------
        # Maps the 128-dim backbone output to action_dim mean values.
        # No activation — raw linear output is the action mean.
        self.mean_head: nn.Linear = nn.Linear(
            self.hidden_sizes[-1],
            self.action_dim,
        )

        # ----------------------------------------------------------------
        # Learned state-independent log standard deviation
        # ----------------------------------------------------------------
        # Shape: [action_dim] — one log_std per action dimension.
        # Initialized to zeros → initial std = exp(0) = 1.0.
        # This provides a reasonable initial exploration range for joint
        # position targets in locomotion tasks (±1 radian).
        #
        # nn.Parameter ensures:
        #   - Included in model.parameters() for optimizer updates
        #   - Moves with .to(device) automatically
        #   - Saved/loaded with state_dict
        self.log_std: nn.Parameter = nn.Parameter(
            torch.zeros(self.action_dim)
        )

    def forward(
        self,
        obs: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute the Gaussian action distribution parameters from observations.

        Passes the policy observation through the MLP backbone and mean head
        to produce the action mean. Computes the action standard deviation
        from the clamped log_std parameter.

        This method is the building block for both ``get_action`` (sampling)
        and ``evaluate_actions`` (log prob computation). It is separated from
        sampling to allow reuse in both contexts.

        Args:
            obs: Policy observation tensor of shape ``[B, obs_dim]``.
                For ANYmal D: ``[B, 48]``. For Unitree G1: ``[B, 99]``.
                Must be on the same device as the model parameters.

        Returns:
            A tuple ``(mean, std)`` where:
              - ``mean``: Action mean tensor of shape ``[B, action_dim]``.
                For ANYmal D: ``[B, 12]``. For Unitree G1: ``[B, 29]``.
              - ``std``: Action standard deviation tensor of shape
                ``[B, action_dim]``. Same std for all observations in the
                batch (state-independent). Values in
                ``[exp(_LOG_STD_MIN) + eps, exp(_LOG_STD_MAX) + eps]``
                ≈ ``[0.007, 7.4]``.
        """
        # ----------------------------------------------------------------
        # 1. Pass observation through MLP backbone.
        # ----------------------------------------------------------------
        # features: [B, hidden_sizes[-1]] = [B, 128]
        features: Tensor = self.net(obs)

        # ----------------------------------------------------------------
        # 2. Compute action mean from backbone features.
        # ----------------------------------------------------------------
        # mean: [B, action_dim]
        mean: Tensor = self.mean_head(features)

        # ----------------------------------------------------------------
        # 3. Compute action std from clamped log_std parameter.
        # ----------------------------------------------------------------
        # Clamp log_std to [_LOG_STD_MIN, _LOG_STD_MAX] = [-5, 2] to prevent:
        #   - std → 0 (logstd << -5): causes NaN in log_prob and entropy
        #   - std → ∞ (logstd >> 2): policy outputs random noise, unstable training
        #
        # self.log_std: [action_dim] — broadcast to [B, action_dim] via expand_as
        log_std_clamped: Tensor = torch.clamp(
            self.log_std,
            _LOG_STD_MIN,
            _LOG_STD_MAX,
        )  # shape: [action_dim]

        # Compute std = exp(log_std) + eps for numerical safety.
        # expand_as broadcasts [action_dim] to [B, action_dim] to match mean.
        std: Tensor = torch.exp(log_std_clamped).expand_as(mean) + _STD_EPS
        # shape: [B, action_dim]

        return mean, std

    def get_action(
        self,
        obs: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Sample an action from the policy distribution and compute its log prob.

        Used during:
          - ``MBPOPPOTrainer.imagine_trajectories``: samples actions for
            T=100 imagination steps across 4096 parallel environments.
          - ``MBPOPPOTrainer.collect_real_data``: samples actions for
            real environment interaction (single environment).

        Uses ``rsample()`` (reparameterized sampling) instead of ``sample()``
        to allow gradient flow through the action if needed. In standard PPO,
        gradients do not flow through the sampled action (the PPO objective
        uses importance sampling ratios, not direct policy gradients through
        actions). However, ``rsample()`` is used for consistency with the
        reparameterization-based training in the world model.

        The log probability is computed as the sum of independent Gaussian
        log probs over all action dimensions:
            log π(a|s) = Σ_i log N(a_i | μ_i, σ_i)

        This is equivalent to the log prob of a multivariate Gaussian with
        diagonal covariance matrix diag(σ_1², ..., σ_n²).

        Args:
            obs: Policy observation tensor of shape ``[B, obs_dim]``.
                For ANYmal D: ``[B, 48]``. For Unitree G1: ``[B, 99]``.

        Returns:
            A tuple ``(action, log_prob)`` where:
              - ``action``: Sampled action tensor of shape ``[B, action_dim]``.
                For ANYmal D: ``[B, 12]``. For Unitree G1: ``[B, 29]``.
                Values are unbounded (no tanh squashing).
              - ``log_prob``: Log probability of the sampled action under the
                current policy, shape ``[B]``. One scalar per environment.
                Used as ``old_log_probs`` in the PPO importance sampling ratio.
        """
        # ----------------------------------------------------------------
        # 1. Compute distribution parameters.
        # ----------------------------------------------------------------
        mean: Tensor
        std: Tensor
        mean, std = self.forward(obs)

        # ----------------------------------------------------------------
        # 2. Construct Gaussian distribution.
        # ----------------------------------------------------------------
        # torch.distributions.Normal handles the log_prob and entropy
        # computations correctly, including the normalization constant.
        dist: Normal = Normal(mean, std)

        # ----------------------------------------------------------------
        # 3. Sample action via reparameterized sampling.
        # ----------------------------------------------------------------
        # rsample() = mean + eps * std, where eps ~ N(0, I)
        # Gradients flow through mean and std (not through eps).
        # action: [B, action_dim]
        action: Tensor = dist.rsample()

        # ----------------------------------------------------------------
        # 4. Compute log probability of the sampled action.
        # ----------------------------------------------------------------
        # dist.log_prob(action): [B, action_dim] — per-dimension log probs
        # .sum(dim=-1): [B] — sum over action dimensions for joint log prob
        # This is correct for independent Gaussian dimensions (diagonal covariance).
        log_prob: Tensor = dist.log_prob(action).sum(dim=-1)
        # shape: [B]

        return action, log_prob

    def evaluate_actions(
        self,
        obs: Tensor,
        actions: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Recompute log probs and entropy for given (obs, action) pairs.

        Used during ``MBPOPPOTrainer.ppo_update`` to compute the PPO
        importance sampling ratio and entropy bonus. This method evaluates
        the current policy's probability of the actions that were taken
        during the imagination rollout (stored in the rollout buffer).

        The PPO clip objective requires:
            ratio = exp(new_log_probs - old_log_probs)
            clip_loss = -min(ratio * advantages, clip(ratio, 1-ε, 1+ε) * advantages)

        where ``new_log_probs`` comes from this method and ``old_log_probs``
        were stored during ``imagine_trajectories``.

        The entropy bonus encourages exploration:
            entropy_loss = -entropy_coefficient * entropy.mean()

        where ``entropy`` comes from this method.

        Args:
            obs: Policy observation tensor of shape ``[B, obs_dim]``.
                These are the observations from the imagination rollout,
                stored in the rollout buffer during ``imagine_trajectories``.
                For ANYmal D: ``[B, 48]``. For Unitree G1: ``[B, 99]``.
            actions: Actions taken during the imagination rollout, shape
                ``[B, action_dim]``. These are the actions stored in the
                rollout buffer, sampled by ``get_action`` during rollout.
                For ANYmal D: ``[B, 12]``. For Unitree G1: ``[B, 29]``.

        Returns:
            A tuple ``(log_probs, entropy)`` where:
              - ``log_probs``: Log probability of ``actions`` under the
                current policy, shape ``[B]``. Used to compute the PPO
                importance sampling ratio.
              - ``entropy``: Per-sample entropy of the current policy
                distribution, shape ``[B]``. Used for the entropy bonus
                in the PPO loss. Entropy of a diagonal Gaussian:
                Σ_i (0.5 * log(2πe * σ_i²)) = Σ_i (log(σ_i) + 0.5 * log(2πe))
        """
        # ----------------------------------------------------------------
        # 1. Compute distribution parameters for the current policy.
        # ----------------------------------------------------------------
        # Note: This uses the CURRENT policy parameters (after any gradient
        # updates in previous PPO epochs), not the parameters at rollout time.
        # This is correct — PPO evaluates the current policy's probability
        # of the stored actions to compute the importance sampling ratio.
        mean: Tensor
        std: Tensor
        mean, std = self.forward(obs)

        # ----------------------------------------------------------------
        # 2. Construct Gaussian distribution.
        # ----------------------------------------------------------------
        dist: Normal = Normal(mean, std)

        # ----------------------------------------------------------------
        # 3. Compute log probability of the stored actions.
        # ----------------------------------------------------------------
        # dist.log_prob(actions): [B, action_dim] — per-dimension log probs
        # .sum(dim=-1): [B] — joint log prob (sum for independent dimensions)
        log_probs: Tensor = dist.log_prob(actions).sum(dim=-1)
        # shape: [B]

        # ----------------------------------------------------------------
        # 4. Compute per-sample entropy.
        # ----------------------------------------------------------------
        # dist.entropy(): [B, action_dim] — per-dimension entropy
        # .sum(dim=-1): [B] — total entropy (sum for independent dimensions)
        # Entropy of N(μ, σ): 0.5 * log(2πe * σ²) = log(σ) + 0.5 * log(2πe)
        entropy: Tensor = dist.entropy().sum(dim=-1)
        # shape: [B]

        return log_probs, entropy


class ValueNetwork(nn.Module):
    """Value function network for MBPO-PPO advantage estimation.

    Implements the value function from Table S9 of the paper. The network
    maps policy observations to scalar state value estimates V(s), used for
    Generalized Advantage Estimation (GAE) in the PPO update.

    The value function takes the same **policy observation** as the policy
    network (Table S5), not the world model observation (Table S2). This is
    important — during imagination rollouts, the value function is queried
    with the same constructed policy obs that the policy receives.

    Architecture (Table S9):
      - Type: MLP
      - Hidden shape: [128, 128, 128]
      - Activation: ELU
      - Output: scalar value estimate V(s)

    The value function is trained with the PPO value loss:
        value_loss = 0.5 * MSE(V(s), returns)

    where ``returns`` are the GAE-computed returns from the imagination
    rollout. The 0.5 coefficient is standard in PPO implementations.

    Attributes:
        obs_dim: Policy observation dimension. 48 for ANYmal D
            (``config.anymal_d.policy_obs_dim``), 99 for Unitree G1
            (``config.unitree_g1.policy_obs_dim``).
        hidden_sizes: List of hidden layer sizes. [128, 128, 128] (Table S9).
        net: 3-hidden-layer MLP backbone with ELU activations.
            Input: obs_dim, output: hidden_sizes[-1] = 128.
        value_head: Linear layer mapping backbone output to scalar value.
            Input: 128, output: 1.
    """

    def __init__(
        self,
        obs_dim: int = 48,
        hidden_sizes: List[int] = None,
        activation: str = "elu",
    ) -> None:
        """Initialize the value function network.

        Builds the MLP backbone and scalar value head. The backbone uses
        the same architecture as ``PolicyNetwork`` (Table S9: same hidden
        sizes and activation). The value head is a single linear layer
        mapping to a scalar.

        Args:
            obs_dim: Policy observation dimension. From
                ``config.anymal_d.policy_obs_dim`` (48) or
                ``config.unitree_g1.policy_obs_dim`` (99) in config.yaml.
                Must be positive. Default: 48 (ANYmal D).
            hidden_sizes: List of hidden layer sizes for the MLP backbone.
                From ``config.value_function.hidden_sizes`` = [128, 128, 128]
                (Table S9). If None, defaults to [128, 128, 128]. Default: None.
            activation: Activation function name. From
                ``config.value_function.activation`` = "elu" (Table S9).
                Supported: "elu", "relu", "tanh". Default: "elu".

        Raises:
            ValueError: If ``obs_dim`` <= 0.
            ValueError: If ``hidden_sizes`` is empty.
            ValueError: If ``activation`` is not supported.
        """
        super().__init__()

        # Apply default for mutable default argument
        if hidden_sizes is None:
            hidden_sizes = [128, 128, 128]

        # ----------------------------------------------------------------
        # Validate dimensions
        # ----------------------------------------------------------------
        if obs_dim <= 0:
            raise ValueError(
                f"obs_dim must be positive, got {obs_dim}. "
                "Check config.anymal_d.policy_obs_dim or "
                "config.unitree_g1.policy_obs_dim in config.yaml. "
                "ANYmal D: policy_obs_dim=48, Unitree G1: policy_obs_dim=99 (Table S5)."
            )

        # ----------------------------------------------------------------
        # Store configuration
        # ----------------------------------------------------------------
        self.obs_dim: int = int(obs_dim)
        self.hidden_sizes: List[int] = list(hidden_sizes)

        # ----------------------------------------------------------------
        # Build MLP backbone (Table S9: "MLP, 128 128 128, ELU")
        # ----------------------------------------------------------------
        # Identical structure to PolicyNetwork's backbone.
        # Output dimension: hidden_sizes[-1] = 128
        self.net: nn.Sequential = _build_mlp_backbone(
            input_size=self.obs_dim,
            hidden_sizes=self.hidden_sizes,
            activation=activation,
        )

        # ----------------------------------------------------------------
        # Build scalar value head
        # ----------------------------------------------------------------
        # Maps the 128-dim backbone output to a single scalar value estimate.
        # No activation — raw linear output is the value estimate V(s).
        # Output shape: [B, 1] (not [B] — the extra dimension is squeezed
        # by the caller if needed, or used directly in MSE loss computation).
        self.value_head: nn.Linear = nn.Linear(
            self.hidden_sizes[-1],
            1,
        )

    def forward(
        self,
        obs: Tensor,
    ) -> Tensor:
        """Compute the scalar state value estimate from the policy observation.

        Passes the policy observation through the MLP backbone and value head
        to produce a scalar value estimate V(s) for each environment in the
        batch.

        Called during:
          - ``MBPOPPOTrainer.imagine_trajectories``: computes V(s_t) for
            each imagination step to store in the rollout buffer for GAE.
          - ``MBPOPPOTrainer.ppo_update``: recomputes V(s_t) with the
            current value function parameters for the PPO value loss.

        The value estimate is used in GAE (Generalized Advantage Estimation):
            δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t = Σ_{k=0}^{T-t-1} (γλ)^k * δ_{t+k}
            returns_t = A_t + V(s_t)

        where ``γ=0.99`` (``mbpo_ppo.discount_factor``) and ``λ=0.95``
        (``mbpo_ppo.gae_lambda``) from config.yaml.

        Args:
            obs: Policy observation tensor of shape ``[B, obs_dim]``.
                For ANYmal D: ``[B, 48]``. For Unitree G1: ``[B, 99]``.
                Must be on the same device as the model parameters.

        Returns:
            Value estimate tensor of shape ``[B, 1]``. One scalar value
            per environment in the batch. The extra dimension (size 1) is
            kept for compatibility with MSE loss computation:
                ``F.mse_loss(values, returns.unsqueeze(-1))``
            Callers can squeeze with ``.squeeze(-1)`` to get shape ``[B]``
            if needed for GAE computation.
        """
        # ----------------------------------------------------------------
        # 1. Pass observation through MLP backbone.
        # ----------------------------------------------------------------
        # features: [B, hidden_sizes[-1]] = [B, 128]
        features: Tensor = self.net(obs)

        # ----------------------------------------------------------------
        # 2. Compute scalar value estimate from backbone features.
        # ----------------------------------------------------------------
        # value: [B, 1] — one scalar per environment
        # No activation — raw linear output is the value estimate.
        # The value can be any real number (unbounded), which is appropriate
        # for estimating cumulative discounted returns.
        value: Tensor = self.value_head(features)
        # shape: [B, 1]

        return value
