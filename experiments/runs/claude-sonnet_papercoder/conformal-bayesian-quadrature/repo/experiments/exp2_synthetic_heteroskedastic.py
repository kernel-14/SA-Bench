## experiments/exp2_synthetic_heteroskedastic.py
"""Experiment 2: Synthetic Heteroskedastic Data (Section 5.2 of the paper).

This module implements the second experiment from "Conformal Prediction as
Bayesian Quadrature". The experiment uses synthetic heteroskedastic regression
data where the true risk can be computed analytically via numerical integration,
enabling exact measurement of how often each method's chosen λ exceeds the
target risk α = 0.1.

Setup (from config.yaml / HeteroskedasticConfig):
  - M = 10,000 random trials
  - n_cal = 200 calibration samples per trial
  - X ~ Uniform[0, 4]
  - Y | X ~ N(0, X²)  (std = X)
  - Prediction intervals: [-λ, λ] (symmetric, centered at 0)
  - Loss: miscoverage loss = 1{|Y| > λ}
  - α = 0.1 (target 90% coverage)
  - β = 0.95 (confidence level for CBQ-HPD)
  - B = 1.0 (upper bound on 0-1 miscoverage loss)
  - λ_grid = np.linspace(0, 20, 1000)

True risk: E_X[Pr(|Y| > λ | X)] = E_X[2·Φ(-λ/X)] over X ~ U[0, 4]
           computed via numerical integration in loss_functions.py.

Expected results (Table 2):
  SCP/CRC:        ~46.19% failure rate, mean PI length ~7.99
  RCPS:            ~0.00% failure rate, mean PI length ~14.29
  CBQ-HPD β=0.95:  ~3.42% failure rate, mean PI length ~9.50

The high SCP/CRC failure rate (46.19%) arises because the marginal guarantee
averages over many calibration draws. When the calibration set happens to
contain mostly low-variance samples (small X), the chosen λ is too small to
cover high-variance test samples. RCPS controls risk but at the cost of very
wide intervals. CBQ-HPD achieves a balance.

References:
    Paper Section 5.2: Synthetic Heteroskedastic Data experiment.
    Paper Table 2: Expected numerical results.
    config.yaml: exp2_synthetic_heteroskedastic.* settings.
    HeteroskedasticConfig in config.py: all hyperparameters.
"""

from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup: allow imports from the project root when this file is run
# from the experiments/ subdirectory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import HeteroskedasticConfig
from decision_rules import (
    compute_losses_grid,
    find_lambda_cbq_hpd,
    find_lambda_crc,
    find_lambda_rcps,
)
from evaluation import TrialResult
from loss_functions import miscoverage_loss, true_heteroskedastic_risk
from utils import make_trial_rng


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_heteroskedastic_cal_data(
    n: int,
    x_low: float,
    x_high: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate calibration data for the synthetic heteroskedastic experiment.

    Implements the data generating process from Section 5.2 of the paper:
      - X ~ Uniform[x_low, x_high]
      - Y | X ~ N(0, X²)  →  std(Y | X) = X

    The resulting array has shape (n, 2) with column 0 = X and column 1 = Y.
    This format is consumed by loss_functions.miscoverage_loss, which uses
    only the Y column (column 1) to compute 1{|Y| > λ}.

    Edge case at X = 0: rng.normal(loc=0.0, scale=0.0) returns 0.0 in NumPy,
    which is the correct degenerate distribution N(0, 0). No special handling
    is needed.

    Args:
        n: Number of calibration samples. Per config.yaml
            (exp2_synthetic_heteroskedastic.n_cal = 200).
        x_low: Lower bound of X ~ Uniform[x_low, x_high]. Per config.yaml
            (exp2_synthetic_heteroskedastic.x_low = 0.0).
        x_high: Upper bound of X ~ Uniform[x_low, x_high]. Per config.yaml
            (exp2_synthetic_heteroskedastic.x_high = 4.0).
        rng: NumPy Generator for reproducible sampling. Should be a
            trial-specific RNG created via utils.make_trial_rng to ensure
            independence across parallel trials.

    Returns:
        Array of shape (n, 2) where:
          - Column 0: X values drawn from Uniform[x_low, x_high].
          - Column 1: Y values drawn from N(0, X²) (std = X per sample).
        Both columns are float64.

    Example:
        >>> rng = np.random.default_rng(42)
        >>> data = generate_heteroskedastic_cal_data(n=5, x_low=0.0, x_high=4.0, rng=rng)
        >>> data.shape
        (5, 2)
        >>> np.all(data[:, 0] >= 0.0) and np.all(data[:, 0] <= 4.0)
        True
        >>> # Y values can be any real number (unbounded normal)
        >>> data.dtype == np.float64
        True
    """
    # Step 1: Draw X ~ Uniform[x_low, x_high] for n samples.
    # Shape: (n,)
    X: np.ndarray = rng.uniform(low=x_low, high=x_high, size=n)

    # Step 2: Draw Y | X ~ N(0, X²) for each sample.
    # Since std(Y | X) = X, we pass scale=X to rng.normal.
    # rng.normal with scale as an array draws one sample per element.
    # Shape: (n,)
    Y: np.ndarray = rng.normal(loc=0.0, scale=X, size=n)

    # Step 3: Stack X and Y into a (n, 2) array.
    # Column 0 = X, Column 1 = Y — matches the convention in miscoverage_loss.
    cal_data: np.ndarray = np.column_stack([X, Y])

    return cal_data


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------


def _safe_true_risk_heteroskedastic(
    lam: float,
    x_low: float,
    x_high: float,
) -> float:
    """Compute true heteroskedastic risk with np.inf handling.

    If lam = np.inf (no valid λ found in the grid), the prediction interval
    [-∞, ∞] covers everything, incurring zero miscoverage risk. We return
    0.0 in this case rather than calling true_heteroskedastic_risk(np.inf, ...)
    which already handles this via the special case in loss_functions.py.

    Args:
        lam: Chosen threshold λ. May be np.inf.
        x_low: Lower bound of X distribution.
        x_high: Upper bound of X distribution.

    Returns:
        True miscoverage risk E_X[2·Φ(-λ/X)], or 0.0 if lam = np.inf.
    """
    if lam == np.inf:
        # λ = ∞ means the prediction interval [-∞, ∞] covers everything.
        # This incurs zero miscoverage risk.
        return 0.0
    # Delegate to the numerical integration in loss_functions.py.
    # true_heteroskedastic_risk already handles lam=np.inf internally,
    # but we guard here for clarity and to avoid unnecessary integration calls.
    return true_heteroskedastic_risk(lam, x_low, x_high)


def run_single_trial_heteroskedastic(
    trial_idx: int,
    config: HeteroskedasticConfig,
) -> TrialResult:
    """Run one complete trial of the synthetic heteroskedastic experiment.

    This function is the unit of parallelism — it is called M=10,000 times
    via joblib.Parallel. Each call is fully independent: it creates its own
    RNG, generates its own calibration data, runs all three decision rules,
    and evaluates the true risk via numerical integration.

    Steps:
      1. Create trial-specific RNG via make_trial_rng(config.seed, trial_idx).
      2. Generate calibration data of shape (n_cal, 2) with columns [X, Y].
      3. Define loss closure: loss_fn(lam) = miscoverage_loss(cal_data, lam).
      4. Pre-compute losses over the entire lambda grid (shape: G × n_cal).
      5. Find λ_crc via Conformal Risk Control (Proposition 3.2).
      6. Find λ_rcps via RCPS with Hoeffding UCB (Bates et al., 2021).
      7. Find λ_hpd via CBQ-HPD at β=0.95 (Corollary 4.4).
      8. Compute true risk via numerical integration for each method.
      9. Compute prediction interval lengths (2λ for each method).
      10. Return TrialResult with all values.

    Args:
        trial_idx: Zero-based trial index in [0, M-1]. Used to seed the
            trial-specific RNG via make_trial_rng(config.seed, trial_idx).
            With config.seed=42 and trial_idx in [0, 9999], seeds range
            from 42 to 10041 — all distinct and valid.
        config: HeteroskedasticConfig instance with all hyperparameters.
            Key values from config.yaml: n_cal=200, x_low=0.0, x_high=4.0,
            alpha=0.1, beta=0.95, B=1.0, n_mc_samples=1000,
            lambda_grid=np.linspace(0, 20, 1000).

    Returns:
        TrialResult with:
          - lambda_crc: λ chosen by CRC (or np.inf if no valid λ found).
          - lambda_rcps: λ chosen by RCPS (or np.inf).
          - lambda_hpd: λ chosen by CBQ-HPD (or np.inf).
          - risk_crc: True miscoverage risk at lambda_crc (0.0 if np.inf).
          - risk_rcps: True miscoverage risk at lambda_rcps (0.0 if np.inf).
          - risk_hpd: True miscoverage risk at lambda_hpd (0.0 if np.inf).
          - extra: {
              'interval_length_crc':  2 * lambda_crc,
              'interval_length_rcps': 2 * lambda_rcps,
              'interval_length_hpd':  2 * lambda_hpd,
            }
            Interval length = 2λ since the prediction interval is [-λ, λ].
            np.inf entries are preserved (handled by evaluation.py's
            _lambda_stats which filters finite values only).

    Example:
        >>> config = HeteroskedasticConfig()
        >>> config.M = 5  # Small for testing
        >>> config.n_mc_samples = 100  # Fast for testing
        >>> result = run_single_trial_heteroskedastic(0, config)
        >>> isinstance(result, TrialResult)
        True
        >>> 0.0 <= result.risk_crc <= 1.0
        True
        >>> result.extra['interval_length_crc'] >= 0.0
        True
    """
    # ------------------------------------------------------------------
    # Step 1: Create trial-specific RNG.
    # Each trial gets a unique seed = config.seed + trial_idx.
    # This ensures independence across parallel workers and reproducibility
    # when re-running a specific trial.
    # ------------------------------------------------------------------
    rng: np.random.Generator = make_trial_rng(config.seed, trial_idx)

    # ------------------------------------------------------------------
    # Step 2: Generate calibration data.
    # Shape: (n_cal, 2) = (200, 2) per config.yaml.
    # Column 0: X ~ Uniform[0, 4], Column 1: Y | X ~ N(0, X²).
    # ------------------------------------------------------------------
    cal_data: np.ndarray = generate_heteroskedastic_cal_data(
        n=config.n_cal,
        x_low=config.x_low,
        x_high=config.x_high,
        rng=rng,
    )

    # ------------------------------------------------------------------
    # Step 3: Define the loss function closure.
    # Captures cal_data from the enclosing scope.
    # Signature: loss_fn(lam: float) -> np.ndarray of shape (n_cal,).
    # Values in {0.0, 1.0} (binary miscoverage indicator).
    # Monotonically non-increasing in lam: larger interval → fewer misses.
    # ------------------------------------------------------------------
    def loss_fn(lam: float) -> np.ndarray:
        """Per-sample miscoverage loss 1{|Y_i| > lam} at threshold lam."""
        return miscoverage_loss(cal_data, lam)

    # ------------------------------------------------------------------
    # Step 4: Pre-compute losses over the entire lambda grid.
    # Shape: (G, n_cal) = (1000, 200) for lambda_grid=np.linspace(0,20,1000).
    # This avoids redundant loss evaluations across the three decision rules.
    # Each row j contains the per-sample losses at lambda_grid[j].
    # ------------------------------------------------------------------
    losses_per_lambda: np.ndarray = compute_losses_grid(
        loss_fn=loss_fn,
        lambda_grid=config.lambda_grid,
    )

    # ------------------------------------------------------------------
    # Step 5: Find λ_crc via Conformal Risk Control (Proposition 3.2).
    # Criterion: (1/(n+1)) * (sum(losses) + B) ≤ α
    # With n=200, B=1.0, α=0.1: needs (Σℓᵢ + 1) / 201 ≤ 0.1
    # i.e., Σℓᵢ ≤ 19.1 (sum of 200 binary losses ≤ 19.1 → at most 19 misses).
    # ------------------------------------------------------------------
    lambda_crc: float = find_lambda_crc(
        losses_per_lambda=losses_per_lambda,
        lambda_grid=config.lambda_grid,
        B=config.B,
        alpha=config.alpha,
    )

    # ------------------------------------------------------------------
    # Step 6: Find λ_rcps via RCPS with Hoeffding UCB (Bates et al., 2021).
    # delta = 1 - beta = 0.05 (config.yaml: rcps.delta = 0.05).
    # Hoeffding UCB: R̂(λ) + sqrt(log(1/0.05) / (2*200)) ≤ 0.1
    # sqrt(log(20)/400) ≈ sqrt(2.996/400) ≈ 0.0865
    # So needs R̂(λ) ≤ 0.1 - 0.0865 ≈ 0.0135 — very conservative.
    # This explains the wide intervals (mean length ~14.29 in Table 2).
    # ------------------------------------------------------------------
    delta: float = 1.0 - config.beta  # = 0.05 per config.yaml (rcps.delta)
    lambda_rcps: float = find_lambda_rcps(
        losses_per_lambda=losses_per_lambda,
        lambda_grid=config.lambda_grid,
        alpha=config.alpha,
        delta=delta,
    )

    # ------------------------------------------------------------------
    # Step 7: Find λ_hpd via CBQ-HPD at β=0.95 (Corollary 4.4).
    # Criterion: Pr(L⁺ ≤ α | ℓ_{1:n}) ≥ β = 0.95
    # Uses n_mc_samples=1000 Dirichlet samples per lambda evaluation.
    # The same rng is passed — it has been advanced by steps 2-4 but
    # remains independent across trials due to per-trial seeding.
    # ------------------------------------------------------------------
    lambda_hpd: float = find_lambda_cbq_hpd(
        losses_per_lambda=losses_per_lambda,
        lambda_grid=config.lambda_grid,
        B=config.B,
        alpha=config.alpha,
        beta=config.beta,
        num_samples=config.n_mc_samples,
        rng=rng,
    )

    # ------------------------------------------------------------------
    # Step 8: Compute true risks via numerical integration.
    # true_heteroskedastic_risk(λ) = E_X[2·Φ(-λ/X)] over X ~ U[0, 4].
    # Risk exceeds α=0.1 iff this value > 0.1.
    # np.inf handling: _safe_true_risk_heteroskedastic returns 0.0 for np.inf.
    # ------------------------------------------------------------------
    risk_crc: float = _safe_true_risk_heteroskedastic(
        lambda_crc, config.x_low, config.x_high
    )
    risk_rcps: float = _safe_true_risk_heteroskedastic(
        lambda_rcps, config.x_low, config.x_high
    )
    risk_hpd: float = _safe_true_risk_heteroskedastic(
        lambda_hpd, config.x_low, config.x_high
    )

    # ------------------------------------------------------------------
    # Step 9: Compute prediction interval lengths.
    # The prediction interval is [-λ, λ], so length = 2λ.
    # np.inf entries are preserved — evaluation.py's _lambda_stats filters
    # finite values when computing mean interval length.
    # ------------------------------------------------------------------
    interval_length_crc: float = 2.0 * lambda_crc
    interval_length_rcps: float = 2.0 * lambda_rcps
    interval_length_hpd: float = 2.0 * lambda_hpd

    # ------------------------------------------------------------------
    # Step 10: Construct and return TrialResult.
    # extra dict stores interval lengths for Table 2's "Mean Prediction
    # Interval Length" column. Keys follow the convention expected by
    # evaluation._aggregate_extra: "interval_length_{method_suffix}".
    # ------------------------------------------------------------------
    return TrialResult(
        lambda_crc=lambda_crc,
        lambda_rcps=lambda_rcps,
        lambda_hpd=lambda_hpd,
        risk_crc=risk_crc,
        risk_rcps=risk_rcps,
        risk_hpd=risk_hpd,
        extra={
            "interval_length_crc": interval_length_crc,
            "interval_length_rcps": interval_length_rcps,
            "interval_length_hpd": interval_length_hpd,
        },
    )


# ---------------------------------------------------------------------------
# Parallel experiment runner
# ---------------------------------------------------------------------------


def run_experiment_heteroskedastic(
    config: HeteroskedasticConfig,
) -> List[TrialResult]:
    """Run all M=10,000 trials of the synthetic heteroskedastic experiment.

    Uses joblib.Parallel with the loky backend (default) to distribute trials
    across available CPU cores. Each trial is fully independent — no shared
    mutable state between workers. The HeteroskedasticConfig dataclass and
    its lambda_grid numpy array of shape (1000,) are safely serializable
    by joblib.

    The return order from joblib.Parallel preserves the input order: result[i]
    corresponds to trial i. This is important for reproducibility analysis.

    Performance note:
        Each trial involves:
          - Generating 200 calibration samples (fast)
          - Computing losses for 1000 lambda values × 200 samples (fast)
          - Running CRC and RCPS (vectorized, fast)
          - Running CBQ-HPD: 1000 lambda values × 1000 Dirichlet samples each
            = 1,000,000 Dirichlet samples per trial (moderate cost)
          - Computing 3 numerical integrals for true risk (fast with scipy.quad)
        With n_jobs=-1 and 8 cores, 10,000 trials should complete in ~10-30 min.

    Args:
        config: HeteroskedasticConfig instance with all hyperparameters.
            Key values from config.yaml: M=10000, n_jobs=-1 (all cores),
            seed=42, n_cal=200, alpha=0.1, beta=0.95, B=1.0,
            n_mc_samples=1000, lambda_grid=np.linspace(0, 20, 1000).

    Returns:
        List of M TrialResult objects, one per trial, in order of trial_idx.
        Each TrialResult contains the chosen λ, true risk, and interval
        length for all three methods (CRC, RCPS, CBQ-HPD).

    Example:
        >>> config = HeteroskedasticConfig()
        >>> config.M = 5  # Small for testing
        >>> config.n_jobs = 1  # Sequential for debugging
        >>> config.n_mc_samples = 100  # Fast for testing
        >>> results = run_experiment_heteroskedastic(config)
        >>> len(results)
        5
        >>> all(isinstance(r, TrialResult) for r in results)
        True
        >>> all('interval_length_crc' in r.extra for r in results)
        True
    """
    # joblib.Parallel with n_jobs=-1 uses all available CPU cores.
    # joblib.delayed wraps run_single_trial_heteroskedastic for lazy evaluation.
    # tqdm wraps the range generator to display a progress bar.
    #
    # Note: tqdm progress updates may be irregular with parallel execution
    # (loky backend) — this is cosmetic only and does not affect correctness.
    trial_results: List[TrialResult] = Parallel(n_jobs=config.n_jobs)(
        delayed(run_single_trial_heteroskedastic)(i, config)
        for i in tqdm(
            range(config.M),
            desc="Exp2: Synthetic Heteroskedastic",
            unit="trial",
        )
    )

    return trial_results
