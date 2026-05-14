## utils.py
"""Shared utility functions for reproducing 'Conformal Prediction as Bayesian Quadrature'.

This module provides four foundational utilities used throughout the project:
  - get_rng: creates a top-level reproducible RNG
  - make_trial_rng: creates per-trial independently seeded RNGs for parallel execution
  - safe_inf_search: implements the infimum search over a lambda grid
  - format_ci: formats Clopper-Pearson confidence intervals for table display

This file has zero dependencies on other project files.
"""

import numpy as np


def get_rng(seed: int) -> np.random.Generator:
    """Create a reproducible NumPy Generator from a fixed seed.

    Uses the new-style PCG64-backed Generator API rather than the legacy
    ``np.random.seed()`` interface. This is the single entry point for
    creating top-level RNGs (e.g., for the overall experiment or for the
    fixed calibration sets used in Figure 4 density plots).

    Args:
        seed: Integer seed for the Generator. Must be a non-negative integer
            or any value accepted by ``np.random.default_rng``.

    Returns:
        A fresh ``np.random.Generator`` instance seeded with ``seed``.

    Example:
        >>> rng = get_rng(42)
        >>> rng.uniform(0, 1, size=3)
        array([0.77395605, 0.43887844, 0.85859792])
    """
    return np.random.default_rng(seed)


def make_trial_rng(base_seed: int, trial_idx: int) -> np.random.Generator:
    """Create an independently seeded RNG for a specific trial.

    Each trial in the 10,000-trial Monte Carlo loop requires its own
    independent, reproducible RNG. This function achieves that by computing
    ``base_seed + trial_idx`` as the seed, ensuring:

    - **Independence**: distinct seeds for each trial index.
    - **Reproducibility**: re-running trial ``i`` with the same ``base_seed``
      always produces identical results.
    - **Parallelism safety**: joblib workers run in separate processes and
      reconstruct their RNG solely from ``trial_idx``, with no shared state.

    With ``base_seed=42`` (from ``config.yaml``) and ``trial_idx`` in
    ``[0, 9999]``, seeds range from 42 to 10041 — all distinct and valid.

    Args:
        base_seed: The experiment-level base seed (``config.seed``, default 42
            per ``config.yaml``).
        trial_idx: Zero-based trial index in ``[0, M-1]`` where ``M = 10000``.

    Returns:
        A fresh ``np.random.Generator`` instance seeded with
        ``base_seed + trial_idx``.

    Example:
        >>> rng_0 = make_trial_rng(42, 0)   # seed = 42
        >>> rng_1 = make_trial_rng(42, 1)   # seed = 43
        >>> rng_0.uniform() != rng_1.uniform()
        True
    """
    return np.random.default_rng(base_seed + trial_idx)


def safe_inf_search(values: np.ndarray, condition: np.ndarray) -> float:
    """Return the first value where condition is True, or np.inf if none found.

    Implements the infimum search used by all three decision rules:

        λ* = inf{λ ∈ lambda_grid : criterion(λ) satisfied}

    All three rules (CRC, RCPS, CBQ-HPD) reduce to finding the first index
    in a sorted ``lambda_grid`` where a boolean criterion flips to ``True``.
    If no lambda satisfies the criterion, ``np.inf`` is returned — a valid
    outcome when the calibration set is too small to certify any risk level.

    Edge cases handled explicitly:
    - All ``False`` → ``np.inf`` (method cannot achieve target risk).
    - First element ``True`` → ``values[0]`` (smallest lambda satisfies criterion).
    - All ``True`` → ``values[0]`` (infimum is the smallest lambda).

    Note:
        ``np.argmax`` returns 0 both when the first element is ``True`` and
        when *no* element is ``True`` (all ``False``). The ``np.any(condition)``
        guard distinguishes these two cases.

    Args:
        values: 1-D array of lambda values, sorted in ascending order
            (e.g., ``np.linspace(0, 1, 500)``). Shape ``(G,)``.
        condition: Boolean array of the same length as ``values``. Element
            ``condition[i]`` is ``True`` when ``values[i]`` satisfies the
            decision rule criterion. Shape ``(G,)``.

    Returns:
        The first ``values[i]`` where ``condition[i]`` is ``True``, or
        ``np.inf`` if no such index exists.

    Raises:
        ValueError: If ``values`` and ``condition`` have different lengths.

    Example:
        >>> grid = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        >>> cond = np.array([False, False, True, True, True])
        >>> safe_inf_search(grid, cond)
        0.3
        >>> safe_inf_search(grid, np.zeros(5, dtype=bool))
        inf
    """
    if len(values) != len(condition):
        raise ValueError(
            f"values and condition must have the same length, "
            f"got {len(values)} and {len(condition)}."
        )

    if not np.any(condition):
        return np.inf

    first_true_idx: int = int(np.argmax(condition))
    return float(values[first_true_idx])


def format_ci(low: float, high: float, decimals: int = 2) -> str:
    """Format a confidence interval as a percentage string for table display.

    Converts proportions in ``[0, 1]`` to percentage strings matching the
    format used in the paper's Tables 1, 2, and 3. For example, the paper
    reports ``[20.40%, 22.01%]`` for the CRC 95% CI in Table 1.

    Args:
        low: Lower bound of the confidence interval as a proportion in
            ``[0, 1]`` (output of ``clopper_pearson_ci``).
        high: Upper bound of the confidence interval as a proportion in
            ``[0, 1]`` (output of ``clopper_pearson_ci``).
        decimals: Number of decimal places for the percentage values.
            Default is 2, matching the paper's reported precision.

    Returns:
        A formatted string of the form ``"[XX.XX%, YY.YY%]"``.

    Example:
        >>> format_ci(0.2040, 0.2201)
        '[20.40%, 22.01%]'
        >>> format_ci(0.0, 0.0004, decimals=2)
        '[0.00%, 0.04%]'
        >>> format_ci(0.0, 0.0, decimals=2)
        '[0.00%, 0.00%]'
    """
    low_pct: float = low * 100.0
    high_pct: float = high * 100.0
    fmt: str = f"{{:.{decimals}f}}"
    return f"[{fmt.format(low_pct)}%, {fmt.format(high_pct)}%]"
