## utils/checkpoint.py
"""Checkpoint management for the emergent planning interpretability pipeline.

This module provides the CheckpointManager class, which handles saving and
loading agent and optimizer state during training and analysis. It serves two
distinct roles in the pipeline:

1. **Training persistence** (used by IMPALATrainer): saves agent and optimizer
   state every ``config.training.checkpoint_every`` steps (default 1,000,000),
   enabling training resumption and providing the 50 checkpoints needed for the
   emergence analysis (Section 6.2, Appendix C).

2. **Analysis retrieval** (used by TrainingEmergenceAnalyzer): retrieves specific
   checkpoints by step number to reconstruct agent states at different points in
   training for the emergence correlation study.

Checkpoint naming convention: ``ckpt_{step:09d}.pt``

The 9-digit zero-padded format (e.g., ``ckpt_001000000.pt``) ensures that
lexicographic sorting of filenames equals chronological ordering. With 250M
total steps and checkpoints every 1M, the maximum step is 250,000,000 (9
digits), so 9-digit padding is exactly sufficient.

This module has zero project-level dependencies to prevent circular imports.
Only ``torch``, ``os``, and ``pathlib`` are used.

Example:
    >>> manager = CheckpointManager("checkpoints/drc33")
    >>> path = manager.save(agent, optimizer, step=1_000_000, metrics={"solve_rate": 0.5})
    >>> step, metrics = manager.load(path, agent, optimizer)
    >>> checkpoints = manager.list_checkpoints()
    >>> closest = manager.get_checkpoint_at_step(5_000_000)
"""

from __future__ import annotations

import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import torch


# Checkpoint filename prefix and suffix used for glob matching and parsing.
_CKPT_PREFIX: str = "ckpt_"
_CKPT_SUFFIX: str = ".pt"
_CKPT_STEP_WIDTH: int = 9  # Zero-padded to 9 digits: max step 250_000_000


class CheckpointManager:
    """Manages saving and loading of agent checkpoints during training and analysis.

    Checkpoints are stored as PyTorch ``.pt`` files containing the agent state
    dict, optimizer state dict, training step, and a metrics dict. The naming
    convention ``ckpt_{step:09d}.pt`` ensures lexicographic sorting equals
    chronological ordering.

    Attributes:
        checkpoint_dir: Absolute or relative path to the directory where
            checkpoint files are stored. Created on construction if absent.

    Example:
        >>> manager = CheckpointManager("checkpoints/drc33")
        >>> # During training:
        >>> path = manager.save(agent, optimizer, step=1_000_000,
        ...                     metrics={"solve_rate": 0.85, "loss": 0.12})
        >>> # During analysis:
        >>> step, metrics = manager.load(path, agent, optimizer=None)
        >>> all_paths = manager.list_checkpoints()
        >>> ckpt_5m = manager.get_checkpoint_at_step(5_000_000)
    """

    def __init__(self, checkpoint_dir: str) -> None:
        """Initialize the CheckpointManager and create the checkpoint directory.

        Creates the full directory tree if it does not already exist. The
        operation is idempotent: calling the constructor multiple times with
        the same path (e.g., when resuming training) is safe.

        Args:
            checkpoint_dir: Path to the directory for storing checkpoint files.
                May be relative (e.g., ``"checkpoints/drc33"``) or absolute.
                Parent directories are created as needed.
        """
        self.checkpoint_dir: str = checkpoint_dir
        pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def save(
        self,
        agent: Any,
        optimizer: Any,
        step: int,
        metrics: Dict[str, Any],
    ) -> str:
        """Save agent and optimizer state to a checkpoint file.

        The checkpoint payload is a single dict containing:
        - ``'step'``: the current training step (int), used for retrieval.
        - ``'agent_state_dict'``: ``agent.state_dict()``, needed for all
          downstream analysis (probing, interventions, emergence analysis).
        - ``'optimizer_state_dict'``: ``optimizer.state_dict()``, needed for
          training resumption.
        - ``'metrics'``: caller-provided dict of training metrics (e.g.,
          ``{'solve_rate': 0.85, 'policy_loss': 0.12}``).

        The filename is ``ckpt_{step:09d}.pt``. The 9-digit zero-padded format
        ensures lexicographic sorting equals chronological ordering across all
        250M training steps.

        Args:
            agent: A PyTorch ``nn.Module`` with a ``state_dict()`` method.
                Typically a ``DRCAgent`` instance.
            optimizer: A PyTorch optimizer with a ``state_dict()`` method.
                Typically ``torch.optim.Adam``. Must not be ``None`` here
                (use ``load`` with ``optimizer=None`` for analysis-only loading).
            step: Current training step (number of environment transitions
                processed so far). Used in the filename and stored in the
                checkpoint for retrieval.
            metrics: Dict of scalar training metrics to store alongside the
                state dicts. Keys and values are caller-defined. Common keys:
                ``'solve_rate'``, ``'policy_loss'``, ``'value_loss'``,
                ``'entropy'``, ``'mean_reward'``.

        Returns:
            Full path string of the saved checkpoint file, e.g.,
            ``"checkpoints/drc33/ckpt_001000000.pt"``. The caller (IMPALATrainer)
            can log this path to confirm the save.

        Example:
            >>> path = manager.save(
            ...     agent, optimizer, step=1_000_000,
            ...     metrics={"solve_rate": 0.85, "loss": 0.12}
            ... )
            >>> print(path)
            checkpoints/drc33/ckpt_001000000.pt
        """
        filename = f"{_CKPT_PREFIX}{step:0{_CKPT_STEP_WIDTH}d}{_CKPT_SUFFIX}"
        full_path = os.path.join(self.checkpoint_dir, filename)

        payload: Dict[str, Any] = {
            "step": step,
            "agent_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }

        torch.save(payload, full_path)
        return full_path

    def load(
        self,
        path: str,
        agent: Any,
        optimizer: Optional[Any] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Load agent (and optionally optimizer) state from a checkpoint file.

        Loads state dicts **in-place** into the provided ``agent`` and
        ``optimizer`` objects. The caller is responsible for moving the agent
        to the appropriate device after loading.

        Using ``map_location='cpu'`` as the default ensures checkpoints saved
        on GPU can be loaded on CPU (and vice versa). The caller can then call
        ``agent.to(device)`` to move to the target device.

        When ``optimizer`` is ``None`` (the common case in analysis contexts
        such as ``TrainingEmergenceAnalyzer``), the optimizer state is silently
        skipped. This avoids requiring a dummy optimizer object just to load
        an agent for probing or intervention experiments.

        Args:
            path: Full path to the checkpoint file to load. Typically obtained
                from ``list_checkpoints()`` or ``get_checkpoint_at_step()``.
            agent: A PyTorch ``nn.Module`` whose state dict will be updated
                in-place from the checkpoint. Must have the same architecture
                as the agent that was saved.
            optimizer: Optional PyTorch optimizer whose state dict will be
                updated in-place. Pass ``None`` when only the agent state is
                needed (e.g., during analysis). Defaults to ``None``.

        Returns:
            A tuple ``(step, metrics)`` where:
            - ``step`` (int): The training step at which this checkpoint was
              saved. Used by ``TrainingEmergenceAnalyzer`` to confirm which
              checkpoint was loaded.
            - ``metrics`` (dict): The metrics dict stored at save time.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            RuntimeError: If the checkpoint file is corrupted or incompatible
                with the provided agent architecture (raised by PyTorch).

        Example:
            >>> # Training resumption (with optimizer):
            >>> step, metrics = manager.load("checkpoints/drc33/ckpt_005000000.pt",
            ...                              agent, optimizer)
            >>> # Analysis only (without optimizer):
            >>> step, metrics = manager.load("checkpoints/drc33/ckpt_005000000.pt",
            ...                              agent, optimizer=None)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        # map_location='cpu' is device-agnostic: works regardless of whether
        # the checkpoint was saved on CPU or GPU.
        checkpoint: Dict[str, Any] = torch.load(path, map_location="cpu")

        agent.load_state_dict(checkpoint["agent_state_dict"])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        step: int = int(checkpoint["step"])
        metrics: Dict[str, Any] = checkpoint.get("metrics", {})

        return step, metrics

    def list_checkpoints(self) -> List[str]:
        """Return a sorted list of all checkpoint file paths in the directory.

        Finds all files matching the pattern ``ckpt_*.pt`` in
        ``self.checkpoint_dir``. Because of the zero-padded naming convention
        (``ckpt_{step:09d}.pt``), lexicographic sorting equals chronological
        ordering — this is the key reason for the 9-digit padding.

        Returns:
            Sorted list of full path strings for all checkpoint files. Returns
            an empty list if the directory contains no matching files. The list
            is sorted in ascending chronological order (earliest checkpoint
            first).

        Example:
            >>> checkpoints = manager.list_checkpoints()
            >>> print(checkpoints[:3])
            ['checkpoints/drc33/ckpt_001000000.pt',
             'checkpoints/drc33/ckpt_002000000.pt',
             'checkpoints/drc33/ckpt_003000000.pt']
        """
        ckpt_dir = pathlib.Path(self.checkpoint_dir)
        pattern = f"{_CKPT_PREFIX}*{_CKPT_SUFFIX}"

        # glob returns an unordered iterator; sort for deterministic ordering.
        paths: List[str] = sorted(
            str(p) for p in ckpt_dir.glob(pattern)
        )
        return paths

    def get_checkpoint_at_step(self, step: int) -> Optional[str]:
        """Return the path of the checkpoint closest to the requested step.

        Used by ``TrainingEmergenceAnalyzer`` to retrieve checkpoints at
        specific training steps (e.g., every 1M steps for the first 50M
        transitions). Uses closest-match rather than exact-match to handle
        potential off-by-one issues in step counting during training.

        Args:
            step: Target training step. The checkpoint whose saved step is
                numerically closest to this value will be returned.

        Returns:
            Full path string of the closest checkpoint, or ``None`` if no
            checkpoints exist in the directory.

        Example:
            >>> # Retrieve the checkpoint closest to 5M transitions:
            >>> path = manager.get_checkpoint_at_step(5_000_000)
            >>> print(path)
            checkpoints/drc33/ckpt_005000000.pt
            >>> # Returns None if no checkpoints exist:
            >>> empty_manager = CheckpointManager("checkpoints/empty")
            >>> empty_manager.get_checkpoint_at_step(1_000_000) is None
            True
        """
        checkpoints: List[str] = self.list_checkpoints()

        if not checkpoints:
            return None

        # Find the checkpoint whose step is numerically closest to the target.
        closest: str = min(
            checkpoints,
            key=lambda p: abs(self._parse_step(p) - step),
        )
        return closest

    def _parse_step(self, path: str) -> int:
        """Extract the training step integer from a checkpoint filename.

        Parses the step number from filenames of the form
        ``ckpt_{step:09d}.pt``. This is a private helper used by
        ``get_checkpoint_at_step`` to enable closest-match retrieval.

        Args:
            path: Full or relative path to a checkpoint file. Only the
                filename component (basename) is used for parsing.

        Returns:
            The integer step number encoded in the filename.

        Raises:
            ValueError: If the filename does not match the expected pattern
                ``ckpt_{digits}.pt``.

        Example:
            >>> manager._parse_step("checkpoints/drc33/ckpt_005000000.pt")
            5000000
            >>> manager._parse_step("ckpt_001000000.pt")
            1000000
        """
        basename: str = os.path.basename(path)

        # Strip prefix and suffix to isolate the step digits.
        if not basename.startswith(_CKPT_PREFIX) or not basename.endswith(_CKPT_SUFFIX):
            raise ValueError(
                f"Checkpoint filename '{basename}' does not match expected pattern "
                f"'{_CKPT_PREFIX}{{step:09d}}{_CKPT_SUFFIX}'. "
                f"Expected format: ckpt_000000000.pt"
            )

        step_str: str = basename[len(_CKPT_PREFIX): -len(_CKPT_SUFFIX)]

        try:
            return int(step_str)
        except ValueError as exc:
            raise ValueError(
                f"Could not parse step integer from checkpoint filename '{basename}'. "
                f"Extracted step string: '{step_str}'"
            ) from exc
