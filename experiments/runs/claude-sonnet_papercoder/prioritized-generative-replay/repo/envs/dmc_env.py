## envs/dmc_env.py
"""DeepMind Control Suite environment wrapper for Prioritized Generative Replay (PGR).

Provides a unified interface for both state-based and pixel-based DMC tasks,
handling observation flattening, pixel rendering, frame stacking, and episode
management. Consumed by PGRTrainer for data collection and Evaluator for
policy evaluation.
"""

import collections
from typing import Deque, Dict, Optional, Tuple

import numpy as np
from dm_control import suite


class DMCEnv:
    """Wrapper around dm_control environments for state-based and pixel-based RL.

    Supports all DMC tasks used in the PGR paper:
        - quadruped-walk, cheetah-run, reacher-hard, finger-turn-hard (state/pixel)
        - walker-walk, hopper-hop (state/pixel)

    For state-based tasks, observations are flattened concatenations of all
    dm_control observation arrays. For pixel-based tasks, observations are
    stacked RGB frames of shape ``(frame_stack * 3, image_size, image_size)``
    stored as uint8 arrays matching the DRQ-v2 convention.

    Attributes:
        env_name: Original environment name string (e.g. ``"quadruped-walk"``).
        pixel_obs: Whether pixel observations are used.
        image_size: Height and width of rendered pixel frames.
        frame_stack: Number of consecutive frames stacked for pixel observations.
    """

    def __init__(
        self,
        env_name: str = "quadruped-walk",
        pixel_obs: bool = False,
        image_size: int = 84,
        frame_stack: int = 3,
        seed: int = 0,
    ) -> None:
        """Initialises the DMC environment wrapper.

        Parses ``env_name`` as ``"domain-task"`` (splitting on the first hyphen),
        loads the dm_control environment with the given seed, and pre-computes
        observation and action space dimensions.

        Args:
            env_name: DMC environment identifier in ``"domain-task"`` format.
                Multi-hyphen task names (e.g. ``"finger-turn-hard"``) are handled
                by splitting on the first hyphen only and replacing remaining
                hyphens with underscores in the task portion.
            pixel_obs: If ``True``, observations are rendered RGB frames stacked
                over ``frame_stack`` timesteps. If ``False``, observations are
                flattened state vectors.
            image_size: Height and width (in pixels) of rendered frames.
                Corresponds to ``env.image_size`` in config.yaml (default 84).
            frame_stack: Number of consecutive frames to stack for pixel
                observations. Corresponds to ``env.frame_stack`` in config.yaml
                (default 3).
            seed: Random seed passed to dm_control via ``task_kwargs``.
                Corresponds to ``env.seed`` in config.yaml.
        """
        self.env_name: str = env_name
        self.pixel_obs: bool = pixel_obs
        self.image_size: int = image_size
        self.frame_stack: int = frame_stack

        # ── Parse domain and task from env_name ──────────────────────────────
        # Split on the first hyphen only; replace remaining hyphens in the
        # task portion with underscores (e.g. "finger-turn-hard" → "turn_hard").
        parts = env_name.split("-", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(
                f"env_name '{env_name}' must follow 'domain-task' format "
                f"(e.g. 'quadruped-walk', 'finger-turn-hard')."
            )
        domain_name: str = parts[0]
        task_name: str = parts[1].replace("-", "_")

        # ── Load dm_control environment ───────────────────────────────────────
        # task_kwargs={'random': seed} is the correct dm_control seeding API.
        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs={"random": seed},
        )

        # ── Action space ──────────────────────────────────────────────────────
        action_spec = self._env.action_spec()
        self._action_dim: int = int(action_spec.shape[0])
        self._action_min: np.ndarray = action_spec.minimum.astype(np.float32)
        self._action_max: np.ndarray = action_spec.maximum.astype(np.float32)

        # ── Frame stacking deque (pixel mode only) ────────────────────────────
        # maxlen enforces automatic eviction of oldest frames.
        self._frames: Deque[np.ndarray] = collections.deque(maxlen=self.frame_stack)

        # ── Observation dimension ─────────────────────────────────────────────
        self._obs_dim: int = self._compute_obs_dim()

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Resets the environment and returns the initial observation.

        For pixel mode, the frame deque is filled with ``frame_stack`` copies
        of the initial rendered frame to avoid zero-padding artefacts.

        Returns:
            For state mode: float32 numpy array of shape ``(obs_dim,)``.
            For pixel mode: uint8 numpy array of shape
                ``(frame_stack * 3, image_size, image_size)``.
        """
        time_step = self._env.reset()

        if self.pixel_obs:
            frame: np.ndarray = self._render_frame()
            # Fill all deque slots with the same initial frame.
            for _ in range(self.frame_stack):
                self._frames.append(frame)
            return self._get_stacked_frames()
        else:
            return self._flatten_obs(time_step.observation)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Advances the environment by one step.

        Actions are clipped to the valid range defined by the dm_control
        action spec before being passed to the environment.

        Args:
            action: Float numpy array of shape ``(action_dim,)``.

        Returns:
            A tuple ``(next_obs, reward, done, info)`` where:
                - ``next_obs``: float32 state vector or uint8 pixel stack.
                - ``reward``: Scalar float reward (0.0 if None).
                - ``done``: ``True`` if the episode has ended.
                - ``info``: Empty dict (dm_control provides no extra info).
        """
        # Clip to valid action range — defensive but consistent with prior work.
        clipped_action: np.ndarray = np.clip(
            action, self._action_min, self._action_max
        ).astype(np.float32)

        time_step = self._env.step(clipped_action)

        # dm_control reward is None only on the first TimeStep (after reset),
        # which is never returned by step(). Guard with 0.0 for safety.
        reward: float = float(time_step.reward) if time_step.reward is not None else 0.0

        # time_step.last() is True at the final step of the episode (StepType.LAST).
        done: bool = bool(time_step.last())

        if self.pixel_obs:
            frame = self._render_frame()
            self._frames.append(frame)
            next_obs: np.ndarray = self._get_stacked_frames()
        else:
            next_obs = self._flatten_obs(time_step.observation)

        return next_obs, reward, done, {}

    def observation_space_dim(self) -> int:
        """Returns the flat observation dimension for replay buffer allocation.

        For state mode: number of elements in the flattened state vector.
        For pixel mode: ``frame_stack * 3 * image_size * image_size`` — the
        flat dimension used by ``ReplayBuffer`` to allocate storage arrays.
        Note that ``reset()`` and ``step()`` return 3D arrays for pixel mode;
        flattening/reshaping at the buffer boundary is the caller's responsibility.

        Returns:
            Integer flat observation dimension.
        """
        return self._obs_dim

    def action_space_dim(self) -> int:
        """Returns the number of action dimensions.

        Returns:
            Integer action dimension.
        """
        return self._action_dim

    def action_range(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the valid action range as (min, max) numpy arrays.

        Returns:
            Tuple of ``(action_min, action_max)`` float32 arrays of shape
            ``(action_dim,)``.
        """
        return self._action_min, self._action_max

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_obs_dim(self) -> int:
        """Computes the flat observation dimension by probing the environment.

        For state mode, performs a reset and sums the sizes of all observation
        arrays. For pixel mode, computes the dimension analytically.

        Returns:
            Integer flat observation dimension.
        """
        if self.pixel_obs:
            return self.frame_stack * 3 * self.image_size * self.image_size

        # State mode: probe by resetting and measuring the flattened obs size.
        time_step = self._env.reset()
        flat_obs: np.ndarray = self._flatten_obs(time_step.observation)
        return int(flat_obs.shape[0])

    def _flatten_obs(self, observation: Dict) -> np.ndarray:
        """Flattens an OrderedDict of dm_control observation arrays.

        Concatenates all observation values (each potentially multi-dimensional)
        into a single 1D float32 numpy array.

        Args:
            observation: OrderedDict mapping observation names to numpy arrays,
                as returned by dm_control's ``TimeStep.observation``.

        Returns:
            Float32 numpy array of shape ``(obs_dim,)``.
        """
        arrays = [np.asarray(v, dtype=np.float32).flatten() for v in observation.values()]
        return np.concatenate(arrays, axis=0)

    def _render_frame(self) -> np.ndarray:
        """Renders the current environment state as a pixel frame.

        Uses ``camera_id=0`` (the default camera) consistent with DRQ-v2 and
        prior pixel-based DMC work. Returns a uint8 array in channel-first
        format ``(3, image_size, image_size)``.

        Returns:
            uint8 numpy array of shape ``(3, image_size, image_size)`` with
            pixel values in ``[0, 255]``.
        """
        # physics.render returns (H, W, 3) uint8.
        frame_hwc: np.ndarray = self._env.physics.render(
            height=self.image_size,
            width=self.image_size,
            camera_id=0,
        )
        # Transpose to channel-first (3, H, W) for PyTorch convention.
        frame_chw: np.ndarray = np.transpose(frame_hwc, (2, 0, 1))
        return frame_chw.astype(np.uint8)

    def _get_stacked_frames(self) -> np.ndarray:
        """Concatenates all frames in the deque along the channel axis.

        Returns:
            uint8 numpy array of shape ``(frame_stack * 3, image_size, image_size)``.
        """
        # Each frame is (3, H, W); concatenate along axis 0 → (frame_stack*3, H, W).
        return np.concatenate(list(self._frames), axis=0)

    def __repr__(self) -> str:
        """Returns a concise string representation of the environment wrapper."""
        mode: str = "pixel" if self.pixel_obs else "state"
        return (
            f"DMCEnv(env_name='{self.env_name}', mode={mode}, "
            f"obs_dim={self._obs_dim}, action_dim={self._action_dim})"
        )
