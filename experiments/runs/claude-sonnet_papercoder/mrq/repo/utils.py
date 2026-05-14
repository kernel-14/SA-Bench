## utils.py
"""Shared utility functions and classes for MR.Q.

This module provides:
  - TwoHotEncoder: Encodes scalar rewards into soft categorical distributions
    over symlog-spaced bins, as described in Section 4.2.1 of the paper.
  - set_seed: Reproducible seeding for all random number generators.
  - compute_td3_normalized / compute_human_normalized: Score normalization
    utilities for Gym and Atari benchmarks respectively.
  - compute_aggregate_metrics: Mean, Median, IQM with 95% bootstrap CIs.
  - bootstrap_ci: Generic stratified bootstrap confidence interval.
  - huber_loss: Element-wise Huber loss for value network updates.

No dependencies on other project files — only external libraries.
"""

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# TwoHotEncoder
# ---------------------------------------------------------------------------


class TwoHotEncoder:
    """Encodes scalar rewards into two-hot distributions over symlog-spaced bins.

    The bins are linearly spaced in symlog space over [low, high], which
    corresponds to non-uniform spacing in the original reward space. This
    matches the approach in Hafner et al. (2023) and the MR.Q paper
    (Section 4.2.1).

    From config.yaml:
        reward_bins: 65
        reward_range_low: -10.0   (symlog space)
        reward_range_high: 10.0   (symlog space)
        Effective raw range: symexp(±10) ≈ ±22026

    Attributes:
        n_bins: Number of discrete bins.
        bins: 1-D float tensor of shape (n_bins,) containing bin centres in
            symlog space, linearly spaced from low to high.
        device: Torch device on which all tensors reside.
    """

    def __init__(
        self,
        n_bins: int,
        low: float,
        high: float,
        device: torch.device,
    ) -> None:
        """Initialise the encoder.

        Args:
            n_bins: Number of bins (65 per config.yaml).
            low: Lower bound in symlog space (-10.0 per config.yaml).
            high: Upper bound in symlog space (10.0 per config.yaml).
            device: Torch device for all internal tensors.

        Raises:
            ValueError: If n_bins < 2 or low >= high.
        """
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}.")
        if low >= high:
            raise ValueError(
                f"low must be strictly less than high, got low={low}, high={high}."
            )

        self.n_bins: int = n_bins
        self.device: torch.device = device

        # Bins are linearly spaced in symlog space.
        # Shape: (n_bins,)
        self.bins: torch.Tensor = torch.linspace(
            low, high, n_bins, dtype=torch.float32, device=device
        )

    # ------------------------------------------------------------------
    # Symlog / symexp transformations
    # ------------------------------------------------------------------

    def symlog(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the symlog transformation: sign(x) * log(|x| + 1).

        This is a monotonic compression that maps large reward magnitudes
        into a compact range while preserving the sign and zero.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of the same shape as x with symlog applied element-wise.
        """
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

    def symexp(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the symexp transformation: sign(x) * (exp(|x|) - 1).

        This is the exact inverse of symlog.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of the same shape as x with symexp applied element-wise.
        """
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

    # ------------------------------------------------------------------
    # Encode: reward scalar → two-hot distribution
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode scalar rewards into two-hot distributions over bins.

        Each scalar reward is mapped to a probability distribution over
        n_bins bins with exactly two non-zero entries (or one if the value
        falls exactly on a bin boundary). The two non-zero entries are the
        two nearest bins in symlog space, weighted by linear interpolation.

        Args:
            x: Reward tensor of shape (batch,) or (batch, 1).

        Returns:
            Float tensor of shape (batch, n_bins) where each row is a valid
            probability distribution (non-negative, sums to 1.0).
        """
        # Ensure shape (batch,)
        x = x.to(self.device)
        if x.dim() == 2:
            x = x.squeeze(-1)
        batch_size: int = x.shape[0]

        # Step 1: Map rewards to symlog space.
        x_symlog: torch.Tensor = self.symlog(x)

        # Step 2: Clamp to the bin range to handle out-of-range values.
        x_symlog = torch.clamp(x_symlog, self.bins[0].item(), self.bins[-1].item())

        # Step 3: Find the index of the first bin strictly greater than x_symlog.
        # searchsorted returns the insertion index; subtract 1 for the lower bin.
        # Shape: (batch,)
        upper_idx: torch.Tensor = torch.searchsorted(
            self.bins.contiguous(), x_symlog.contiguous()
        )
        # Clamp so that lower_idx is in [0, n_bins - 2] and upper_idx in [1, n_bins - 1].
        upper_idx = torch.clamp(upper_idx, 1, self.n_bins - 1)
        lower_idx: torch.Tensor = upper_idx - 1  # shape (batch,)

        # Step 4: Compute interpolation weights.
        lower_val: torch.Tensor = self.bins[lower_idx]   # shape (batch,)
        upper_val: torch.Tensor = self.bins[upper_idx]   # shape (batch,)

        # Fraction of the way from lower bin to upper bin.
        # Bin spacing is uniform in symlog space so denominator is constant,
        # but we compute it explicitly for correctness.
        bin_width: torch.Tensor = upper_val - lower_val  # shape (batch,)
        upper_weight: torch.Tensor = (x_symlog - lower_val) / bin_width  # in [0, 1]
        lower_weight: torch.Tensor = 1.0 - upper_weight                  # in [0, 1]

        # Step 5: Scatter weights into a (batch, n_bins) tensor.
        result: torch.Tensor = torch.zeros(
            batch_size, self.n_bins, dtype=torch.float32, device=self.device
        )
        result.scatter_(
            dim=1,
            index=lower_idx.unsqueeze(1),
            src=lower_weight.unsqueeze(1),
        )
        result.scatter_(
            dim=1,
            index=upper_idx.unsqueeze(1),
            src=upper_weight.unsqueeze(1),
        )

        return result  # shape (batch, n_bins), each row sums to 1.0

    # ------------------------------------------------------------------
    # Decode: predicted logits → scalar reward estimate
    # ------------------------------------------------------------------

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Decode predicted bin logits into scalar reward estimates.

        Applies softmax to convert logits to a probability distribution,
        computes the expected bin value in symlog space, then maps back to
        the original reward space via symexp.

        This is used for monitoring and debugging, not in the training loss.
        The training loss uses cross-entropy between logits and two-hot targets.

        Args:
            logits: Raw network output of shape (batch, n_bins).

        Returns:
            Scalar reward estimates of shape (batch,) in the original
            (non-symlog) reward space.
        """
        logits = logits.to(self.device)

        # Convert logits to probabilities.
        probs: torch.Tensor = F.softmax(logits, dim=-1)  # shape (batch, n_bins)

        # Expected bin value in symlog space.
        # bins shape: (n_bins,) → broadcast to (batch, n_bins)
        expected_symlog: torch.Tensor = (probs * self.bins.unsqueeze(0)).sum(dim=-1)

        # Map back to original reward space.
        return self.symexp(expected_symlog)  # shape (batch,)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set all random number generators for full reproducibility.

    Sets seeds for Python's built-in random module, NumPy, and PyTorch
    (both CPU and all CUDA devices). Also configures cuDNN for deterministic
    operation at the cost of some performance.

    Args:
        seed: Integer seed value. Should be in [0, 2^32 - 1].
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Enable deterministic algorithms for full reproducibility.
    # This may reduce performance but ensures identical results across runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------


def compute_td3_normalized(
    scores: Dict[str, float],
    random_scores: Dict[str, float],
    td3_scores: Dict[str, float],
) -> Dict[str, float]:
    """Normalize Gym locomotion scores using TD3 as the reference.

    Implements the formula from Appendix B.3 of the paper:
        TD3-Normalized(x) = (x - random_score) / (TD3_score - random_score)

    Reference scores come from config.yaml under
    benchmarks.gym.normalization. Normalized scores can exceed 1.0 when
    the algorithm outperforms TD3 (e.g., MR.Q on Humanoid).

    Args:
        scores: Dict mapping environment name to raw episode return.
        random_scores: Dict mapping environment name to random-policy score.
        td3_scores: Dict mapping environment name to TD3 reference score.

    Returns:
        Dict mapping environment name to TD3-normalized score.

    Raises:
        KeyError: If an environment in scores is missing from reference dicts.
        ZeroDivisionError: If TD3 score equals random score for any environment.
    """
    normalized: Dict[str, float] = {}
    for env_name, score in scores.items():
        rand_score: float = random_scores[env_name]
        td3_score: float = td3_scores[env_name]
        denom: float = td3_score - rand_score
        if denom == 0.0:
            raise ZeroDivisionError(
                f"TD3 score equals random score for '{env_name}': "
                f"td3={td3_score}, random={rand_score}."
            )
        normalized[env_name] = (score - rand_score) / denom
    return normalized


def compute_human_normalized(
    scores: Dict[str, float],
    random_scores: Dict[str, float],
    human_scores: Dict[str, float],
) -> Dict[str, float]:
    """Normalize Atari scores using human performance as the reference.

    Implements the formula from Appendix B.3 of the paper:
        Human-Normalized(x) = (x - random_score) / (human_score - random_score)

    Reference scores come from config.yaml under
    benchmarks.atari.normalization.

    Args:
        scores: Dict mapping game name to raw episode return.
        random_scores: Dict mapping game name to random-policy score.
        human_scores: Dict mapping game name to human reference score.

    Returns:
        Dict mapping game name to human-normalized score.

    Raises:
        KeyError: If a game in scores is missing from reference dicts.
        ZeroDivisionError: If human score equals random score for any game.
    """
    normalized: Dict[str, float] = {}
    for game_name, score in scores.items():
        rand_score: float = random_scores[game_name]
        human_score: float = human_scores[game_name]
        denom: float = human_score - rand_score
        if denom == 0.0:
            raise ZeroDivisionError(
                f"Human score equals random score for '{game_name}': "
                f"human={human_score}, random={rand_score}."
            )
        normalized[game_name] = (score - rand_score) / denom
    return normalized


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def bootstrap_ci(
    data: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """Compute a bootstrap confidence interval for a given statistic.

    Performs non-parametric bootstrap resampling to estimate the confidence
    interval of a statistic computed over the input data. This matches the
    "95% stratified bootstrap confidence interval" reported in the paper.

    Args:
        data: 1-D array of observed values (e.g., normalized scores).
        statistic_fn: Callable that takes a 1-D array and returns a scalar
            statistic (e.g., np.mean, np.median, or a custom IQM function).
        n_bootstrap: Number of bootstrap resamples. Default 10,000 gives
            stable CI estimates.
        ci: Confidence level in (0, 1). Default 0.95 for 95% CI.

    Returns:
        Tuple (lower_bound, upper_bound) of the confidence interval.

    Raises:
        ValueError: If data is empty or ci is not in (0, 1).
    """
    if len(data) == 0:
        raise ValueError("data must be non-empty.")
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0, 1), got {ci}.")

    data_arr: np.ndarray = np.asarray(data, dtype=np.float64)
    n: int = len(data_arr)
    alpha: float = (1.0 - ci) / 2.0

    bootstrap_stats: np.ndarray = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        resample: np.ndarray = data_arr[
            np.random.randint(0, n, size=n)
        ]
        bootstrap_stats[i] = statistic_fn(resample)

    lower: float = float(np.percentile(bootstrap_stats, 100.0 * alpha))
    upper: float = float(np.percentile(bootstrap_stats, 100.0 * (1.0 - alpha)))
    return lower, upper


def _iqm(data: np.ndarray) -> float:
    """Compute the Interquartile Mean (IQM) of a 1-D array.

    Discards the bottom 25% and top 25% of values, then returns the mean
    of the remaining 50%. This is the aggregate metric used in the paper
    alongside mean and median.

    Args:
        data: 1-D array of values.

    Returns:
        IQM as a float. Returns NaN if the trimmed slice is empty.
    """
    sorted_data: np.ndarray = np.sort(data)
    n: int = len(sorted_data)
    lower_idx: int = int(np.floor(0.25 * n))
    upper_idx: int = int(np.ceil(0.75 * n))
    trimmed: np.ndarray = sorted_data[lower_idx:upper_idx]
    if len(trimmed) == 0:
        return float("nan")
    return float(np.mean(trimmed))


def compute_aggregate_metrics(
    normalized_scores: List[float],
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
) -> Dict[str, Any]:
    """Compute mean, median, and IQM with bootstrap confidence intervals.

    These are the aggregate metrics reported in Tables 4–7 of the paper.
    The input is a flat list of normalized scores, one per environment,
    aggregated across a benchmark (e.g., all 28 DMC tasks).

    Args:
        normalized_scores: List of normalized scores (one per environment).
            For Gym: TD3-normalized. For Atari: human-normalized.
            For DMC: raw reward / 1000.
        n_bootstrap: Number of bootstrap resamples for CI estimation.
        ci: Confidence level for bootstrap CIs. Default 0.95.

    Returns:
        Dictionary with the following keys:
            'mean'       (float): Mean of normalized scores.
            'median'     (float): Median of normalized scores.
            'iqm'        (float): Interquartile mean of normalized scores.
            'mean_ci'    (Tuple[float, float]): 95% bootstrap CI for mean.
            'median_ci'  (Tuple[float, float]): 95% bootstrap CI for median.
            'iqm_ci'     (Tuple[float, float]): 95% bootstrap CI for IQM.

    Raises:
        ValueError: If normalized_scores is empty.
    """
    if len(normalized_scores) == 0:
        raise ValueError("normalized_scores must be non-empty.")

    data: np.ndarray = np.asarray(normalized_scores, dtype=np.float64)

    mean_val: float = float(np.mean(data))
    median_val: float = float(np.median(data))
    iqm_val: float = _iqm(data)

    mean_ci: Tuple[float, float] = bootstrap_ci(
        data, statistic_fn=np.mean, n_bootstrap=n_bootstrap, ci=ci
    )
    median_ci: Tuple[float, float] = bootstrap_ci(
        data, statistic_fn=np.median, n_bootstrap=n_bootstrap, ci=ci
    )
    iqm_ci: Tuple[float, float] = bootstrap_ci(
        data, statistic_fn=_iqm, n_bootstrap=n_bootstrap, ci=ci
    )

    return {
        "mean": mean_val,
        "median": median_val,
        "iqm": iqm_val,
        "mean_ci": mean_ci,
        "median_ci": median_ci,
        "iqm_ci": iqm_ci,
    }


# ---------------------------------------------------------------------------
# Huber loss
# ---------------------------------------------------------------------------


def huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Compute element-wise Huber loss between predictions and targets.

    Used in the value network update (Section 4.2.2) instead of MSE to
    eliminate bias from LAP prioritized sampling (Fujimoto et al., 2020).

    The Huber loss is defined as:
        L(e) = 0.5 * e^2                    if |e| <= delta
        L(e) = delta * (|e| - 0.5 * delta)  if |e| >  delta

    where e = pred - target.

    No reduction is applied — the caller is responsible for aggregating
    (e.g., via .mean() or weighted sum for prioritized replay).

    Args:
        pred: Predicted values tensor of arbitrary shape.
        target: Target values tensor of the same shape as pred.
        delta: Threshold at which the loss transitions from quadratic to
            linear. Default 1.0 (standard Huber loss).

    Returns:
        Element-wise Huber loss tensor of the same shape as pred and target.

    Raises:
        ValueError: If delta <= 0.
    """
    if delta <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}.")

    # Delegate to PyTorch's optimised implementation with no reduction.
    return F.huber_loss(pred, target, reduction="none", delta=delta)
