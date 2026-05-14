## relevance/base.py
"""Abstract base class for relevance functions in Prioritized Generative Replay (PGR).

Defines the interface contract that all relevance functions must fulfill.
Concrete implementations (ICMRelevance, RNDRelevance, etc.) inherit from
this class and are consumed by PGRTrainer exclusively through this interface,
enabling plug-and-play swapping of relevance functions without modifying the
training loop.

The relevance function F(s, a, s', r) = c assigns a scalar priority score c
to each transition τ = (s, a, s', r). This score is used as the conditioning
signal for the conditional diffusion model G(τ | c), guiding generation
towards more learning-relevant regions of transition space (Section 4.2).

Supported relevance function variants (paper Section 4.2 and Appendix A):
    - Reward:    F(s, a, s', r) = r
    - Return:    F(s, a, s', r) = Q(s, π(s))
    - TD Error:  F(s, a, s', r) = r + γ·Q_target(s', argmax Q(s', a')) - Q(s, a)
    - Curiosity: F(s, a, s', r) = 0.5 · ||g(h(s), a) - h(s')||²  [ICM, default]
    - RND:       F(s, a, s', r) = 0.5 · ||f̂_θ(s') - f(s')||²
"""

from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn


class BaseRelevance(nn.Module, ABC):
    """Abstract base class for all PGR relevance functions.

    Inherits from both ``torch.nn.Module`` and ``abc.ABC`` to provide:
    1. Standard PyTorch module behavior (``.to(device)``, ``.parameters()``,
       ``.train()``, ``.eval()``) for concrete subclasses with learnable
       parameters (ICMRelevance, RNDRelevance).
    2. ABC enforcement ensuring all subclasses implement ``score()`` and
       ``update()`` — the two methods consumed by ``PGRTrainer``.

    Concrete subclasses that have no learnable parameters (e.g. reward-based
    or return-based relevance) should implement ``update()`` as a no-op
    returning ``0.0``.

    The universal transition dict format shared across all PGR modules
    (per Shared Knowledge in the design document) uses the following keys:

        {
            'observations':      Tensor (B, obs_dim),
            'actions':           Tensor (B, action_dim),
            'next_observations': Tensor (B, obs_dim),
            'rewards':           Tensor (B, 1),
            'dones':             Tensor (B, 1),
        }

    Attributes:
        obs_dim: Flat observation dimension. Used by subclasses to build
            encoder input layers. For pixel-based tasks this is the CNN
            latent dimension (e.g. drqv2.feature_dim = 50), not the raw
            pixel dimension.
        action_dim: Action dimension. Used by ICMRelevance for the forward
            dynamics model input (concatenation of latent state and action).
        device: PyTorch device string (e.g. ``"cuda"`` or ``"cpu"``).
            Subclasses use this to move their networks and input tensors to
            the correct device. Corresponds to ``hardware.device`` in
            config.yaml (default ``"cuda"``).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: str = "cuda",
    ) -> None:
        """Initialises the base relevance function.

        Calls ``nn.Module.__init__()`` via ``super()`` and stores the three
        constructor arguments as instance attributes. Does NOT instantiate
        any networks — that is the exclusive responsibility of subclasses.

        Args:
            obs_dim: Flat observation dimension. For state-based DMC tasks
                this is the concatenated dm_control observation vector size
                (e.g. 67 for quadruped-walk, 17 for cheetah-run). For
                pixel-based tasks this is the CNN encoder output dimension
                (feature_dim = 50 for DRQv2). Corresponds to the obs_dim
                inferred from the environment wrapper at PGRTrainer init time.
            action_dim: Number of action dimensions. Corresponds to
                ``env.action_space_dim()`` (e.g. 12 for quadruped-walk,
                6 for cheetah-run, 6 for Walker2d-v2).
            device: PyTorch device string. Corresponds to
                ``hardware.device`` in config.yaml (default ``"cuda"``).
                Subclasses should call ``self.to(self.device)`` at the end
                of their own ``__init__`` to move all registered parameters
                and buffers to this device.
        """
        # Initialize nn.Module — required before any parameter registration.
        super().__init__()

        self.obs_dim: int = obs_dim
        self.action_dim: int = action_dim
        self.device: str = device

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def score(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        reward: torch.Tensor,
    ) -> torch.Tensor:
        """Computes per-transition relevance scores for a batch of transitions.

        Implements F(s, a, s', r) = c, mapping each transition tuple to a
        scalar priority score. The score is used as the conditioning signal
        for the conditional diffusion model (Section 4.2 of the paper).

        **Contract for all implementations:**

        1. **Input shapes:**
               - ``obs``:      ``(B, obs_dim)``   float32
               - ``action``:   ``(B, action_dim)`` float32
               - ``next_obs``: ``(B, obs_dim)``   float32
               - ``reward``:   ``(B, 1)`` or ``(B,)`` float32

        2. **Output shape:** ``(B, 1)`` float32 tensor of per-transition
           relevance scores. The ``(B, 1)`` shape (not ``(B,)``) is required
           for consistent broadcasting in ``ConditionalDiffusion.train_step()``
           and for storage in ``ReplayBuffer.relevance_scores`` (shape
           ``(capacity, 1)``).

        3. **Detached output:** The returned tensor MUST NOT carry gradients
           into the policy update. Implementations must either wrap the
           forward pass in ``torch.no_grad()`` or call ``.detach()`` before
           returning. This is critical because ``score()`` is called:
               - In ``PGRTrainer._update_relevance_scores()`` to label all
                 transitions in D_real (no gradient needed).
               - In ``PGRTrainer._train_diffusion()`` to compute conditions
                 for each diffusion training batch (gradients through the
                 relevance function would incorrectly couple ICM and diffusion
                 training).

        4. **Raw unnormalized scores:** Normalization to [0, 1] is the
           responsibility of ``PGRTrainer``, not this method. The trainer
           normalizes scores per inner loop call using the min/max of all
           scores in D_real before passing them as conditions to the diffusion
           model (Shared Knowledge point 3 in the design document).

        5. **Why ``reward`` is included:** The paper defines F as a function
           of the full transition tuple F(s, a, s', r) for all variants.
           The reward-based variant (F = r) and TD-error variant both require
           the reward argument. ICM and RND implementations may ignore it.

        Args:
            obs: Current observations, float32 tensor of shape
                ``(B, obs_dim)``. Must be on ``self.device``.
            action: Actions taken, float32 tensor of shape
                ``(B, action_dim)``. Must be on ``self.device``.
            next_obs: Next observations, float32 tensor of shape
                ``(B, obs_dim)``. Must be on ``self.device``.
            reward: Rewards received, float32 tensor of shape ``(B, 1)``
                or ``(B,)``. Must be on ``self.device``.

        Returns:
            Float32 tensor of shape ``(B, 1)`` containing per-transition
            relevance scores. Detached from the computation graph (no
            gradients). Values are raw (unnormalized) — normalization to
            [0, 1] is performed by the caller (PGRTrainer).
        """
        ...

    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> float:
        """Performs one gradient step to update the relevance function parameters.

        Called by ``PGRTrainer._update_relevance_scores()`` every
        ``relevance.update_freq = 20`` policy gradient steps, corresponding
        to 5% of all policy steps as specified in Section 5 of the paper.

        **Contract for all implementations:**

        1. **Input format:** The ``batch`` dict follows the universal
           transition format shared across all PGR modules:

               {
                   'observations':      Tensor (B, obs_dim),
                   'actions':           Tensor (B, action_dim),
                   'next_observations': Tensor (B, obs_dim),
                   'rewards':           Tensor (B, 1),
                   'dones':             Tensor (B, 1),
               }

           All tensors are float32 on ``self.device`` (device placement is
           handled by ``ReplayBuffer.sample()`` before the batch reaches here).

        2. **Exactly one gradient step:** The method must perform exactly one
           forward pass, loss computation, backward pass, and optimizer step.
           It must NOT loop internally — the caller controls the update
           frequency.

        3. **Return value:** Returns the scalar training loss as a Python
           ``float`` (via ``loss.item()``), not a tensor. This value is
           logged by ``PGRTrainer`` for monitoring ICM/RND training stability.

        4. **No-op for parameter-free variants:** Relevance functions with
           no learnable parameters (reward-based, return-based) should
           implement this method as a no-op returning ``0.0``. The TD-error
           variant also returns ``0.0`` since it uses already-trained
           Q-networks from the policy.

        5. **Gradient isolation:** The optimizer step must only update the
           relevance function's own parameters — it must NOT affect the
           policy networks (actor, critics) or the diffusion model. Each
           concrete subclass maintains its own optimizer instance.

        Args:
            batch: Transition dict sampled from D_real by PGRTrainer.
                Contains float32 tensors on ``self.device`` with keys
                'observations', 'actions', 'next_observations', 'rewards',
                'dones'. Batch size corresponds to
                ``config.sampling.batch_size`` (default 256).

        Returns:
            Scalar training loss as a Python float. Used by PGRTrainer for
            logging (e.g. ``"relevance/icm_loss"`` in the metrics dict).
            Returns ``0.0`` for parameter-free relevance variants.
        """
        ...

    # ── Concrete utility methods ──────────────────────────────────────────────

    def to_device(self, *tensors: torch.Tensor) -> tuple:
        """Moves a sequence of tensors to self.device with float32 dtype.

        Convenience helper for concrete subclasses to ensure all input
        tensors are on the correct device before forward passes. Avoids
        repetitive ``.to(device=self.device, dtype=torch.float32)`` calls
        in ``score()`` and ``update()`` implementations.

        Usage in subclasses::

            obs, action, next_obs = self.to_device(obs, action, next_obs)

        Args:
            *tensors: Variable number of torch.Tensor instances to move.

        Returns:
            Tuple of tensors moved to ``self.device`` with dtype float32,
            in the same order as the input arguments.
        """
        return tuple(
            t.to(device=self.device, dtype=torch.float32) for t in tensors
        )

    def __repr__(self) -> str:
        """Returns a concise string representation of the relevance function.

        Subclasses may override this for more informative representations,
        but the base implementation provides a useful default showing the
        class name and key dimensions.

        Returns:
            String of the form
            ``"ClassName(obs_dim=D, action_dim=A, device='cuda')"``
        """
        return (
            f"{self.__class__.__name__}("
            f"obs_dim={self.obs_dim}, "
            f"action_dim={self.action_dim}, "
            f"device='{self.device}')"
        )
