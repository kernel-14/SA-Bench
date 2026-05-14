## buffers/replay_buffer.py
"""Replay buffer for Prioritized Generative Replay (PGR).

Provides a circular, numpy-backed replay buffer used for both D_real (real
environment transitions) and D_syn (synthetically generated transitions).
All storage lives on CPU for memory efficiency; GPU-ready tensors are
produced only at sample time.

Consumed by:
    - MixedSampler: calls sample() on both buffers
    - PGRTrainer: add(), sample_top_k(), update_relevance_scores(), clear()
    - ConditionalDiffusion: receives batches from sample() for diffusion training
    - ICMRelevance: receives batches from sample() for ICM gradient updates
    - Evaluator: sample() and get_all_as_tensor() for analysis experiments
"""

from typing import Dict, Optional

import numpy as np
import torch


class ReplayBuffer:
    """Circular numpy-backed replay buffer for online RL transitions.

    Stores transitions as pre-allocated float32 numpy arrays on CPU and
    returns GPU-ready torch.Tensor dicts at sample time. Supports both
    uniform sampling (for policy/diffusion training) and top-k sampling
    by relevance score (for the PGR prompting strategy).

    The universal transition dict format returned by sample() and
    sample_top_k() follows the shared convention across all PGR modules:

        {
            'observations':      Tensor (B, obs_dim),
            'actions':           Tensor (B, action_dim),
            'next_observations': Tensor (B, obs_dim),
            'rewards':           Tensor (B, 1),
            'dones':             Tensor (B, 1),
        }

    sample_top_k() additionally includes:

        {
            ...,
            'relevance_scores':  Tensor (k, 1),
        }

    Attributes:
        capacity: Maximum number of transitions the buffer can hold.
        obs_dim: Flat observation dimension.
        action_dim: Action dimension.
        device: PyTorch device string for returned tensors (e.g. "cuda").
        ptr: Write pointer; next slot to be overwritten (modulo capacity).
        size: Number of valid transitions currently stored (capped at capacity).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        capacity: int = 1_000_000,
        device: str = "cuda",
    ) -> None:
        """Initialises the replay buffer and pre-allocates storage arrays.

        All six numpy arrays (observations, actions, next_observations,
        rewards, dones, relevance_scores) are allocated at construction time
        to avoid dynamic resizing during training.

        Args:
            obs_dim: Flat observation dimension. For pixel-based tasks this
                is the CNN latent dimension (e.g. drqv2.feature_dim = 50),
                not the raw pixel dimension — the DRQv2 policy encodes pixels
                before storing transitions.
            action_dim: Action dimension.
            capacity: Maximum number of transitions. Corresponds to
                buffer.real_capacity (1_000_000) for D_real and
                buffer.syn_capacity (1_000_000 or 2_000_000 for the combined
                scaling experiment) for D_syn, as specified in config.yaml.
            device: PyTorch device string. Tensors returned by sample() and
                related methods are moved to this device. Corresponds to
                hardware.device in config.yaml (default "cuda").
        """
        self.capacity: int = capacity
        self.obs_dim: int = obs_dim
        self.action_dim: int = action_dim
        self.device: str = device

        # ── Pre-allocate CPU numpy arrays ─────────────────────────────────────
        # All arrays use float32 for consistency with PyTorch default dtype.
        # rewards and dones are stored as (capacity, 1) to match the (B, 1)
        # shape required by the universal transition dict format.
        self.observations: np.ndarray = np.zeros(
            (capacity, obs_dim), dtype=np.float32
        )
        self.actions: np.ndarray = np.zeros(
            (capacity, action_dim), dtype=np.float32
        )
        self.next_observations: np.ndarray = np.zeros(
            (capacity, obs_dim), dtype=np.float32
        )
        self.rewards: np.ndarray = np.zeros((capacity, 1), dtype=np.float32)
        self.dones: np.ndarray = np.zeros((capacity, 1), dtype=np.float32)

        # Relevance scores are initialised to 0.0. Before the first ICM update
        # (at step icm_update_freq=20), all scores are 0, so sample_top_k()
        # degrades gracefully to uniform sampling — acceptable during warmup.
        self.relevance_scores: np.ndarray = np.zeros(
            (capacity, 1), dtype=np.float32
        )

        # ── Circular buffer state ─────────────────────────────────────────────
        self.ptr: int = 0   # Next write slot (wraps modulo capacity).
        self.size: int = 0  # Number of valid entries (capped at capacity).

    # ── Public API ────────────────────────────────────────────────────────────

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        """Stores a single transition in the buffer.

        Writes at the current pointer position and advances the pointer
        modulo capacity, implementing circular (ring) buffer semantics.
        Old transitions are silently overwritten once the buffer is full.

        The relevance score at the overwritten slot is intentionally NOT
        reset to 0 — stale scores are preferable to zero scores for the
        prompting strategy, since zero would incorrectly deprioritize
        recently added transitions until the next ICM scoring pass.

        Args:
            obs: Float32 numpy array of shape (obs_dim,) — current observation.
            action: Float32 numpy array of shape (action_dim,) — action taken.
            next_obs: Float32 numpy array of shape (obs_dim,) — next observation.
            reward: Scalar float reward received.
            done: Boolean episode termination flag (True if episode ended).
        """
        self.observations[self.ptr] = np.asarray(obs, dtype=np.float32)
        self.actions[self.ptr] = np.asarray(action, dtype=np.float32)
        self.next_observations[self.ptr] = np.asarray(next_obs, dtype=np.float32)
        self.rewards[self.ptr, 0] = float(reward)
        self.dones[self.ptr, 0] = float(done)
        # relevance_scores[ptr] is intentionally left unchanged on overwrite.

        # Advance circular pointer.
        self.ptr = (self.ptr + 1) % self.capacity
        # Cap size at capacity once the buffer has wrapped.
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Samples a batch of transitions uniformly without replacement.

        Samples from the valid region [0, size) of the buffer. If size is
        smaller than batch_size (early training before warmup completes),
        falls back to sampling with replacement to avoid crashing — the
        caller (PGRTrainer) should ideally ensure a warmup period before
        calling sample(), but this fallback provides robustness.

        Args:
            batch_size: Number of transitions to sample. Corresponds to
                sampling.batch_size in config.yaml (default 256).

        Returns:
            Dict with keys 'observations', 'actions', 'next_observations',
            'rewards', 'dones'. All values are float32 tensors on self.device
            with batch dimension batch_size.
        """
        if self.size == 0:
            raise RuntimeError(
                "ReplayBuffer.sample() called on an empty buffer. "
                "Ensure at least one transition has been added before sampling."
            )

        # Fall back to sampling with replacement if buffer is smaller than
        # the requested batch size (rare edge case during early warmup).
        replace: bool = self.size < batch_size
        indices: np.ndarray = np.random.choice(
            self.size, size=batch_size, replace=replace
        )

        return self._gather_indices(indices, include_relevance=False)

    def sample_top_k(self, k_fraction: float = 0.1) -> Dict[str, torch.Tensor]:
        """Returns the top-k transitions by relevance score.

        Implements the PGR prompting strategy (Section 4.3 of the paper,
        inspired by Peebles et al., 2022): selects the top k_fraction of
        transitions from D_real by their stored relevance score, then
        returns them so that PGRTrainer can randomly sample their score
        values as conditions for the conditional diffusion model.

        The returned dict includes a 'relevance_scores' key (unlike the
        standard sample() output) so that PGRTrainer._generate_synthetic_data()
        can access the score values for conditioning.

        Args:
            k_fraction: Fraction of valid transitions to select as top-k.
                Corresponds to diffusion.top_k_fraction in config.yaml
                (default 0.1 = top 10% of D_real).

        Returns:
            Dict with keys 'observations', 'actions', 'next_observations',
            'rewards', 'dones', 'relevance_scores'. All values are float32
            tensors on self.device with batch dimension k.
        """
        if self.size == 0:
            raise RuntimeError(
                "ReplayBuffer.sample_top_k() called on an empty buffer."
            )

        # Compute k — at least 1 transition must be selected.
        k: int = max(1, int(k_fraction * self.size))
        k = min(k, self.size)  # Cannot exceed the number of valid entries.

        # Get relevance scores for valid entries only.
        valid_scores: np.ndarray = self.relevance_scores[:self.size, 0]

        # argsort returns indices in ascending order; take the last k (highest).
        sorted_indices: np.ndarray = np.argsort(valid_scores)
        top_k_indices: np.ndarray = sorted_indices[-k:]

        return self._gather_indices(top_k_indices, include_relevance=True)

    def update_relevance_scores(
        self,
        indices: np.ndarray,
        scores: np.ndarray,
    ) -> None:
        """Writes ICM relevance scores back into the buffer at given indices.

        Called by PGRTrainer._update_relevance_scores() after the ICM has
        scored a batch of transitions from D_real. Normalization to [0, 1]
        is the caller's responsibility (performed in PGRTrainer before
        passing to the diffusion model as conditions).

        Args:
            indices: Integer numpy array of shape (N,) containing buffer
                indices to update. Must be in [0, size).
            scores: Float32 numpy array of shape (N,) or (N, 1) containing
                the raw ICM prediction errors for each indexed transition.
        """
        scores_flat: np.ndarray = np.asarray(scores, dtype=np.float32).flatten()
        self.relevance_scores[indices, 0] = scores_flat

    def get_all_as_tensor(self) -> Dict[str, torch.Tensor]:
        """Returns all valid transitions as GPU-ready tensors.

        Used by ConditionalDiffusion.fit_normalizer() on the first inner
        loop call to fit the transition normalizer on the full D_real dataset.

        For D_real at capacity (1M transitions) with obs_dim=21 (quadruped-walk),
        this transfers ~224 MB to GPU — manageable on modern hardware.

        Returns:
            Dict with keys 'observations', 'actions', 'next_observations',
            'rewards', 'dones'. All values are float32 tensors on self.device
            with batch dimension self.size.
        """
        if self.size == 0:
            raise RuntimeError(
                "ReplayBuffer.get_all_as_tensor() called on an empty buffer."
            )

        indices: np.ndarray = np.arange(self.size)
        return self._gather_indices(indices, include_relevance=False)

    def __len__(self) -> int:
        """Returns the number of valid transitions currently stored.

        Returns:
            Integer in [0, capacity].
        """
        return self.size

    def is_full(self) -> bool:
        """Returns True if the buffer has reached its capacity.

        Used by PGRTrainer._generate_synthetic_data() to determine when
        D_syn has been fully populated and generation can stop.

        Returns:
            True if self.size >= self.capacity, False otherwise.
        """
        return self.size >= self.capacity

    def clear(self) -> None:
        """Resets the buffer to an empty state without zeroing storage arrays.

        Resets ptr and size to 0. The underlying numpy arrays are NOT zeroed
        for efficiency — the size field ensures stale data beyond index size
        is never sampled. Called on D_syn at the start of each inner loop
        before regeneration. Never called on D_real.
        """
        self.ptr = 0
        self.size = 0

    # ── Private helpers ───────────────────────────────────────────────────────

    def _gather_indices(
        self,
        indices: np.ndarray,
        include_relevance: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Gathers transitions at the given indices and converts to tensors.

        Centralises the numpy-to-tensor conversion logic used by sample(),
        sample_top_k(), and get_all_as_tensor(). All tensors are moved to
        self.device with dtype=torch.float32.

        Args:
            indices: Integer numpy array of buffer indices to gather.
            include_relevance: If True, includes 'relevance_scores' in the
                returned dict (used by sample_top_k() only).

        Returns:
            Transition dict with float32 tensors on self.device.
        """
        # Gather numpy slices — advanced indexing returns copies, not views.
        obs_np: np.ndarray = self.observations[indices]
        actions_np: np.ndarray = self.actions[indices]
        next_obs_np: np.ndarray = self.next_observations[indices]
        rewards_np: np.ndarray = self.rewards[indices]
        dones_np: np.ndarray = self.dones[indices]

        # Convert to float32 tensors on the target device.
        batch: Dict[str, torch.Tensor] = {
            "observations": torch.tensor(
                obs_np, dtype=torch.float32, device=self.device
            ),
            "actions": torch.tensor(
                actions_np, dtype=torch.float32, device=self.device
            ),
            "next_observations": torch.tensor(
                next_obs_np, dtype=torch.float32, device=self.device
            ),
            "rewards": torch.tensor(
                rewards_np, dtype=torch.float32, device=self.device
            ),
            "dones": torch.tensor(
                dones_np, dtype=torch.float32, device=self.device
            ),
        }

        if include_relevance:
            scores_np: np.ndarray = self.relevance_scores[indices]
            batch["relevance_scores"] = torch.tensor(
                scores_np, dtype=torch.float32, device=self.device
            )

        return batch

    def __repr__(self) -> str:
        """Returns a concise string representation of the buffer state."""
        return (
            f"ReplayBuffer("
            f"size={self.size}/{self.capacity}, "
            f"obs_dim={self.obs_dim}, "
            f"action_dim={self.action_dim}, "
            f"device='{self.device}')"
        )
