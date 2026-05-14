"""
dataset.py

Implements the TrajectoryBuffer class, which stores trajectories from interactions
with the environment and provides sampling methods for autoregressive world‑model
training, imagination initialization, and serialisation.

The buffer handles both unlimited pretraining datasets and capacity‑limited online
replay buffers for MBPO‑PPO.  All tensor operations are performed on the device
specified at construction (usually CPU to save GPU memory).
"""

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class TrajectoryBuffer:
    """
    A collection of trajectories (episodes) of robot interactions.

    Stores sequences of observations, actions, privileged information, rewards,
    and termination flags.  The buffer supports:
    - Pushing single transitions that are automatically grouped into episodes.
    - Sampling contiguous windows of fixed length (history + forecast) for
      autoregressive world‑model training.
    - Sampling individual observations to start imagination rollouts.
    - Computing dataset‑wide normalisation statistics for observations and actions.
    - Saving / loading to / from disk.
    """

    def __init__(
        self,
        capacity_transitions: Optional[int],
        obs_dim: int,
        act_dim: int,
        priv_dim: int,
        device: str = "cpu",
    ):
        """
        Args:
            capacity_transitions: Maximum number of transitions to keep (FIFO).
                If None, the buffer has unlimited size (used for pretraining).
            obs_dim: Dimension of the world‑model observation (full state).
            act_dim: Dimension of the action vector.
            priv_dim: Dimension of the privileged information vector.
            device: Device on which all internal tensors are stored (usually 'cpu').
        """
        self.capacity = capacity_transitions
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.priv_dim = priv_dim
        self.device = torch.device(device)

        # Internal storage – list of complete episodes.
        # Each episode is a dict:
        #   'obs': (T, obs_dim)
        #   'act': (T, act_dim)
        #   'priv': (T, priv_dim)
        #   'rew': (T,)
        #   'done': (T,)   bool
        self.episodes: List[Dict[str, torch.Tensor]] = []
        self._total_transitions = 0

        # Buffers for the current (unfinished) episode.
        self._cur_ep_buffer = {
            "obs": [],
            "act": [],
            "priv": [],
            "rew": [],
            "done": [],
        }

        # Normalisation statistics – computed once, then reused.
        self._stats: Optional[Dict[str, torch.Tensor]] = None

    def push(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        rew: float,
        next_obs: torch.Tensor,
        priv: torch.Tensor,
        done: bool,
    ) -> None:
        """
        Store a single transition.

        The transition is appended to the current episode.  When `done` is True,
        the episode is finalised and added to the main storage.  If the buffer
        has a capacity limit, the oldest episode(s) are removed to keep the
        total number of transitions within the limit.

        Args:
            obs: Observation at time t.  Shape: (obs_dim,)
            act: Action taken at time t.  Shape: (act_dim,)
            rew: Reward received.
            next_obs: Next observation (not stored internally; only obs sequences are kept).
            priv: Privileged information at time t.  Shape: (priv_dim,)
            done: Whether the episode terminated after this step.
        """
        # Move inputs to the internal device and ensure correct dtypes.
        obs = obs.to(device=self.device, dtype=torch.float32)
        act = act.to(device=self.device, dtype=torch.float32)
        priv = priv.to(device=self.device, dtype=torch.float32)
        # next_obs is not stored; only the consecutive obs sequence matters.

        # Append to current episode buffers.
        self._cur_ep_buffer["obs"].append(obs)
        self._cur_ep_buffer["act"].append(act)
        self._cur_ep_buffer["priv"].append(priv)
        self._cur_ep_buffer["rew"].append(rew)
        self._cur_ep_buffer["done"].append(done)

        if done:
            # Finalise the episode.
            T = len(self._cur_ep_buffer["obs"])
            if T == 0:
                # Should not happen, but just in case.
                self._cur_ep_buffer = {k: [] for k in self._cur_ep_buffer}
                return

            ep = {
                "obs": torch.stack(self._cur_ep_buffer["obs"], dim=0),  # (T, obs_dim)
                "act": torch.stack(self._cur_ep_buffer["act"], dim=0),  # (T, act_dim)
                "priv": torch.stack(self._cur_ep_buffer["priv"], dim=0),  # (T, priv_dim)
                "rew": torch.tensor(self._cur_ep_buffer["rew"], device=self.device, dtype=torch.float32),  # (T,)
                "done": torch.tensor(self._cur_ep_buffer["done"], device=self.device, dtype=torch.bool),  # (T,)
            }
            self.episodes.append(ep)
            self._total_transitions += T

            # Enforce capacity by removing the oldest episodes.
            if self.capacity is not None:
                while self.episodes and self._total_transitions > self.capacity:
                    removed = self.episodes.pop(0)
                    self._total_transitions -= removed["obs"].shape[0]

            # Reset the current‑episode buffer.
            self._cur_ep_buffer = {k: [] for k in self._cur_ep_buffer}

    def sample_batch(
        self, batch_size: int, history_len: int, forecast_len: int
    ) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of contiguous windows for autoregressive training.

        Each window consists of `history_len + forecast_len` consecutive steps
        (observations, actions, privileged information).  The buffer guarantees
        that windows never cross episode boundaries.

        Args:
            batch_size: Number of windows to sample.
            history_len: Number of historical steps (M in the paper).
            forecast_len: Number of future steps to predict (N in the paper).

        Returns:
            A dictionary with keys:
              - 'obs_seq':  shape (batch_size, L, obs_dim)
              - 'act_seq':  shape (batch_size, L, act_dim)
              - 'priv_seq': shape (batch_size, L, priv_dim)
            where L = history_len + forecast_len.

        Raises:
            RuntimeError: If there are no episodes long enough to provide the
                required window length.
        """
        L = history_len + forecast_len
        # Collect indices of episodes that are long enough.
        valid_eps = [i for i, ep in enumerate(self.episodes) if ep["obs"].shape[0] >= L]
        if not valid_eps:
            raise RuntimeError(
                f"No episodes of length >= {L}. Current episodes lengths: "
                f"{[ep['obs'].shape[0] for ep in self.episodes]}"
            )

        # Sample episode indices (with replacement if necessary).
        ep_indices = torch.randint(
            0, len(valid_eps), (batch_size,), device=self.device
        )

        obs_seq_list = []
        act_seq_list = []
        priv_seq_list = []

        for idx in ep_indices:
            ep_idx = valid_eps[idx]
            ep = self.episodes[ep_idx]
            T = ep["obs"].shape[0]
            # Random start index within the episode.
            start = torch.randint(0, T - L + 1, (1,), device=self.device).item()
            sl = slice(start, start + L)
            obs_seq_list.append(ep["obs"][sl])
            act_seq_list.append(ep["act"][sl])
            priv_seq_list.append(ep["priv"][sl])

        # Stack into batch tensors.
        obs_seq = torch.stack(obs_seq_list, dim=0)   # (B, L, obs_dim)
        act_seq = torch.stack(act_seq_list, dim=0)   # (B, L, act_dim)
        priv_seq = torch.stack(priv_seq_list, dim=0) # (B, L, priv_dim)

        return {"obs_seq": obs_seq, "act_seq": act_seq, "priv_seq": priv_seq}

    def sample_initial_obs(self, batch_size: int) -> torch.Tensor:
        """
        Sample individual observation vectors uniformly from all stored transitions.

        This is used to initialise imagination agents in the MBRL loop.

        Args:
            batch_size: Number of observations to sample.

        Returns:
            Tensor of shape (batch_size, obs_dim).

        Raises:
            RuntimeError: If the buffer contains no transitions.
        """
        total_trans = self._total_transitions
        if total_trans == 0:
            raise RuntimeError("Cannot sample from an empty buffer.")

        # Build cumulative transition counts to efficiently sample flat indices.
        cum_lens = [0]  # cumulative, where cum_lens[i] = total transitions in episodes 0..i-1
        for ep in self.episodes:
            cum_lens.append(cum_lens[-1] + ep["obs"].shape[0])
        cum_lens = torch.tensor(cum_lens, device=self.device)

        # Sample flat indices.
        flat_indices = torch.randint(0, total_trans, (batch_size,), device=self.device)

        # Map each flat index to (episode index, step index).
        ep_idx = torch.searchsorted(cum_lens, flat_indices, right=True) - 1
        step_idx = flat_indices - cum_lens[ep_idx]

        # Retrieve the observations.
        obs_list = []
        for b in range(batch_size):
            ep = self.episodes[ep_idx[b].item()]
            obs_list.append(ep["obs"][step_idx[b].item()])

        return torch.stack(obs_list, dim=0).to(self.device)

    def compute_normalization_stats(self) -> Dict[str, torch.Tensor]:
        """
        Compute dataset‑wide per‑dimension mean and standard deviation
        for observations and actions.

        Statistics are stored internally and returned.  During all subsequent
        operations, the buffer returns raw data; normalisation is applied by
        the world model if needed.

        Returns:
            Dictionary with keys:
                - 'mean_obs': shape (obs_dim,)
                - 'std_obs':  shape (obs_dim,)
                - 'mean_act': shape (act_dim,)
                - 'std_act':  shape (act_dim,)
        """
        # Concatenate all observations and actions.
        all_obs = torch.cat([ep["obs"] for ep in self.episodes], dim=0)
        all_act = torch.cat([ep["act"] for ep in self.episodes], dim=0)

        mean_obs = all_obs.mean(dim=0)
        std_obs = all_obs.std(dim=0, unbiased=False)
        # Add a small epsilon to avoid division by zero when std is zero.
        std_obs = torch.clamp(std_obs, min=1e-6)

        mean_act = all_act.mean(dim=0)
        std_act = all_act.std(dim=0, unbiased=False)
        std_act = torch.clamp(std_act, min=1e-6)

        self._stats = {
            "mean_obs": mean_obs,
            "std_obs": std_obs,
            "mean_act": mean_act,
            "std_act": std_act,
        }
        return self._stats

    @property
    def stats(self) -> Optional[Dict[str, torch.Tensor]]:
        return self._stats

    def save(self, path: str) -> None:
        """
        Save the entire buffer (all episodes and normalisation stats) to disk.

        Args:
            path: File path (e.g., 'data/buffer.pt').
        """
        save_dict = {
            "episodes": self.episodes,
            "stats": self._stats,
            "capacity": self.capacity,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "priv_dim": self.priv_dim,
            "device": str(self.device),
            "total_transitions": self._total_transitions,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(save_dict, path)

    def load(self, path: str) -> None:
        """
        Restore a previously saved buffer from disk.

        Args:
            path: File path to the saved state.
        """
        data = torch.load(path, map_location=self.device)
        self.episodes = data["episodes"]
        self._stats = data.get("stats", None)
        self.capacity = data["capacity"]
        self.obs_dim = data["obs_dim"]
        self.act_dim = data["act_dim"]
        self.priv_dim = data["priv_dim"]
        self._total_transitions = data.get("total_transitions", sum(ep["obs"].shape[0] for ep in self.episodes))
        self._cur_ep_buffer = {k: [] for k in ["obs", "act", "priv", "rew", "done"]}

    def __len__(self) -> int:
        """Return the total number of stored transitions across all episodes."""
        return self._total_transitions
