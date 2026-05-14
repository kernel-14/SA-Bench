## utils.py

"""
Common utilities for the PGR (Prioritized Generative Replay) reproduction.

Provides:
- `Transition` namedtuple to represent experience tuples.
- Device management (`get_device`).
- Logging setup (`setup_logging`), including optional `wandb` integration.
- Evaluation metric tracking (`RunningAverage`).
- Optional state normalisation (`Normalizer`).
- Random seed setting and tensor conversion helpers.
"""

import logging
import os
from collections import namedtuple
from typing import Optional, Any

import numpy as np
import torch

# Import Config only for type annotations; it is a top-level class that does not
# import utils, so no circular dependency.
try:
    from config import Config
except ImportError:
    Config = Any  # fallback for environments where config is not importable


# ------------------------------------------------------------------------------
# 1. Transition data structure
# ------------------------------------------------------------------------------

Transition = namedtuple(
    'Transition',
    ['state', 'action', 'reward', 'next_state', 'done']
)
"""A single experience tuple: (s, a, r, s', done)."""


# ------------------------------------------------------------------------------
# 2. Device management
# ------------------------------------------------------------------------------

_DEVICE: Optional[torch.device] = None


def get_device() -> torch.device:
    """Return the current device (CUDA if available, otherwise CPU)."""
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _DEVICE


# ------------------------------------------------------------------------------
# 3. Logging setup
# ------------------------------------------------------------------------------

def setup_logging(config: Config) -> logging.Logger:
    """
    Initialise logging for the PGR experiment.

    Creates a file handler in `config.logging.checkpoint_dir` and a stream
    handler. Optionally starts a `wandb` run according to `config.logging`.

    Parameters
    ----------
    config : Config
        Master configuration object.

    Returns
    -------
    logging.Logger
        The logger instance used by the whole experiment.
    """
    log_dir = config.logging.checkpoint_dir
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger('PGR')
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Optional wandb integration
    if config.logging.use_wandb:
        import wandb
        wandb.init(
            project=config.logging.project_name,
            name=config.logging.run_name,
            config=dataclasses.asdict(config) if hasattr(config, '__dataclass_fields__') else vars(config)
        )
        logger.info("Weights & Biases logging enabled.")
    else:
        logger.info("Weights & Biases logging not requested.")

    return logger


# ------------------------------------------------------------------------------
# 4. Running average tracker for evaluation metrics
# ------------------------------------------------------------------------------

class RunningAverage:
    """Online computation of mean and standard deviation using Welford's algorithm."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def var(self) -> float:
        return self.M2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return self.var ** 0.5

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0


# ------------------------------------------------------------------------------
# 5. State normalisation (optional)
# ------------------------------------------------------------------------------

class Normalizer:
    """
    Running mean and standard deviation normaliser for state vectors.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the state space.
    eps : float, optional
        Small value added to standard deviation for numerical stability.
    """

    def __init__(self, state_dim: int, eps: float = 1e-8) -> None:
        self.mean = np.zeros(state_dim, dtype=np.float32)
        self.std = np.ones(state_dim, dtype=np.float32)
        self.eps = eps
        self.n = 0
        self.sum_sq = np.zeros(state_dim, dtype=np.float32)

    def update(self, states: np.ndarray) -> None:
        """
        Update the running statistics with a batch of states.

        Parameters
        ----------
        states : np.ndarray, shape [batch_size, state_dim]
        """
        batch_size = states.shape[0]
        self.n += batch_size
        delta = states - self.mean
        self.mean += delta.sum(axis=0) / self.n
        delta2 = states - self.mean
        self.sum_sq += np.sum(delta * delta2, axis=0)

    def normalize(self, state: np.ndarray) -> np.ndarray:
        """
        Normalise the given state(s) using the current statistics.

        Parameters
        ----------
        state : np.ndarray
            Single state (shape [state_dim]) or batch of states (shape [N, state_dim]).

        Returns
        -------
        np.ndarray
            Normalised state(s), same shape as input.
        """
        # update std (clipped) for current statistics
        var = self.sum_sq / (self.n - 1) if self.n > 1 else np.ones_like(self.sum_sq)
        self.std = np.maximum(np.sqrt(var), self.eps)
        return (state - self.mean) / self.std

    def denormalize(self, state: np.ndarray) -> np.ndarray:
        """Reverse the normalisation."""
        return state * self.std + self.mean


# ------------------------------------------------------------------------------
# 6. Helper functions
# ------------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def to_tensor(data, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Convert a numpy array to a torch tensor and move to the specified device.

    Parameters
    ----------
    data : array-like
        Data to convert. Accepts scalar, list, numpy array, or already a tensor.
    device : torch.device, optional
        Target device. If None, uses the default device from `get_device()`.

    Returns
    -------
    torch.Tensor
        Tensor on the requested device.
    """
    if device is None:
        device = get_device()
    # If already a tensor, just move (and convert float if needed)
    if isinstance(data, torch.Tensor):
        return data.float().to(device)
    return torch.as_tensor(np.asarray(data), dtype=torch.float32, device=device)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a numpy array on CPU."""
    return tensor.detach().cpu().numpy()
