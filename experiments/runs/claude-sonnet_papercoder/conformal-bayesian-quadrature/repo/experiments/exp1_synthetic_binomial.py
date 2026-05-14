## experiments/exp1_synthetic_binomial.py
"""Experiment 1: Synthetic Binomial Data (Section 5.1 of the paper).

This module implements the first experiment from "Conformal Prediction as
Bayesian Quadrature". The experiment uses a synthetic binomial loss where
the true expected loss is known analytically as 1 - λ, enabling direct
verification of risk control guarantees.

Setup (from config.yaml / BinomialConfig):
  - M = 10,000 random trials
  - n_cal = 10 calibration samples per trial
  - K = 4 (binomial averaging parameter)
  - α = 0.4 (target risk level)
  - β = 0.95 (confidence level for CBQ-HPD)
  - B = 1.0 (upper bound on losses)
  - λ_grid = np.linspace(0, 1, 500)

Loss function (Section 5.1):
  ℓ(zᵢ, λ) = (1/K) · Σₖ₌₁ᴷ 1{Vᵢₖ > λ}   where Vᵢₖ ~ Uniform(0, 1)

True expected loss: E[ℓ(z, λ)] = 1 - λ
Risk threshold: λ < 0.6 ↔ risk > α = 0.4

Expected results (Table 1):
  CRC:          ~21.20% failure rate  [20.40%, 22.01%]
  RCPS:          ~0.00% failure rate  [0.00%, 0.04%]
  CBQ-HPD β=0.95: ~0.03% failure rate  [0.01%, 0.09%]

References:
    Paper Section 5.1: Synthetic Binomial Data experiment.
    config.yaml: exp1_synthetic_binomial.* settings.
    BinomialConfig in config.py: all hyperparameters.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

import cbq_core
from config import BinomialConfig
from decision_rules import (
    compute_losses_grid,
    find_lambda_cbq_hpd,
    find_lambda_crc,
    find_lambda_rcps,
)
from evaluation import TrialResult
from loss_functions import binomial_loss, true_binomial_risk
from utils import make_trial_rng


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_binomial_cal_data(
    n: int,
    K: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate calibration data for the synthetic binomial experiment.

    Each calibration point i consists of K independent Uniform(0, 1) draws
    V_{i1}, ..., V_{iK}. The per-sample loss is computed later from this
    data via loss_functions.binomial_loss:

        ℓ(zᵢ, λ) = (1/K) · Σₖ 1{Vᵢₖ > λ}

    Args:
        n: Number of calibration samples. Per config.yaml
            (exp1_synthetic_binomial.n_cal = 10).
        K: Binomial averaging parameter. Per config.yaml
            (exp1_synthetic_binomial.K = 4). Must match the number of
            columns expected by binomial_loss.
        rng: NumPy Generator for reproducible sampling. Should be a
            trial-specific RNG created via utils.make_trial_rng to ensure
            independence across parallel trials.

    Returns:
        Array of shape (n, K) where each entry Vᵢₖ ~ Uniform(0, 1).
        Row i contains [V_{i1}, V_{i2}, ..., V_{iK}] for calibration
        sample i.

    Example:
        >>> rng = np.random.default_rng(42)
        >>> data = generate_binomial_cal_data(n=10, K=4, rng=rng)
        >>> data.shape
        (10, 4)
        >>> np.all((data >= 0.0) & (data <= 1.0))
        True
    """
    # Draw n * K independent Uniform(0, 1) samples and reshape to (n, K).
    # rng.uniform is the new-style API — never use np.random.uniform here.
    cal_data: np.ndarray = rng.uniform(0.0, 1.0, size=(n, K))
    return cal_data


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------


def run_single_trial_binomial(
    trial_idx: int,
    config: BinomialConfig,
) -> TrialResult:
    """Run one complete trial of the synthetic binomial experiment.

    This function is the unit of parallelism — it is called M=10,000 times
    via joblib.Parallel. Each call is fully independent: it creates its own
    RNG, generates its own calibration data, runs all three decision rules,
    and evaluates the true risk analytically.

    Steps:
      1. Create trial-specific RNG via make_trial_rng(config.seed, trial_idx).
      2. Generate calibration data of shape (n_cal, K).
      3. Define loss closure: loss_fn(lam) = binomial_loss(cal_data, lam, K).
      4. Pre-compute losses over the entire lambda grid (shape: G × n_cal).
      5. Find λ_crc via Conformal Risk Control (Proposition 3.2).
      6. Find λ_rcps via RCPS with Hoeffding UCB (Bates et al., 2021).
      7. Find λ_hpd via CBQ-HPD at β=0.95 (Corollary 4.4).
      8. Compute true risk = 1 - λ for each method (analytical).
      9. Return TrialResult with all six values.

    Args:
        trial_idx: Zero-based trial index in [0, M-1]. Used to seed the
            trial-specific RNG via make_trial_rng(config.seed, trial_idx).
            With config.seed=42 and trial_idx in [0, 9999], seeds range
            from 42 to 10041 — all distinct and valid.
        config: BinomialConfig instance with all hyperparameters. Key values
            from config.yaml: n_cal=10, K=4, alpha=0.4, beta=0.95, B=1.0,
            n_mc_samples=1000, lambda_grid=np.linspace(0,1,500).

    Returns:
        TrialResult with:
          - lambda_crc: λ chosen by CRC (or np.inf if no valid λ found).
          - lambda_rcps: λ chosen by RCPS (or np.inf).
          - lambda_hpd: λ chosen by CBQ-HPD (or np.inf).
          - risk_crc: True risk = 1 - lambda_crc (0.0 if lambda_crc=np.inf).
          - risk_rcps: True risk = 1 - lambda_rcps (0.0 if lambda_rcps=np.inf).
          - risk_hpd: True risk = 1 - lambda_hpd (0.0 if lambda_hpd=np.inf).
          - extra: {} (empty for Experiment 1).

    Example:
        >>> config = BinomialConfig()
        >>> config.M = 5  # Small for testing
        >>> result = run_single_trial_binomial(0, config)
        >>> isinstance(result, TrialResult)
        True
        >>> 0.0 <= result.risk_crc <= 1.0 or result.risk_crc == 0.0
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
    # Shape: (n_cal, K) = (10, 4) per config.yaml.
    # ------------------------------------------------------------------
    cal_data: np.ndarray = generate_binomial_cal_data(
        n=config.n_cal,
        K=config.K,
        rng=rng,
    )

    # ------------------------------------------------------------------
    # Step 3: Define the loss function closure.
    # Captures cal_data and K from the enclosing scope.
    # Signature: loss_fn(lam: float) -> np.ndarray of shape (n_cal,).
    # Values in {0, 1/K, 2/K, ..., 1} = {0, 0.25, 0.5, 0.75, 1.0} for K=4.
    # ------------------------------------------------------------------
    def loss_fn(lam: float) -> np.ndarray:
        """Per-sample binomial loss at threshold lam."""
        return binomial_loss(cal_data, lam, config.K)

    # ------------------------------------------------------------------
    # Step 4: Pre-compute losses over the entire lambda grid.
    # Shape: (G, n_cal) = (500, 10) for lambda_grid=np.linspace(0,1,500).
    # This avoids redundant loss evaluations across the three decision rules.
    # ------------------------------------------------------------------
    losses_per_lambda: np.ndarray = compute_losses_grid(
        loss_fn=loss_fn,
        lambda_grid=config.lambda_grid,
    )

    # ------------------------------------------------------------------
    # Step 5: Find λ_crc via Conformal Risk Control (Proposition 3.2).
    # Criterion: (1/(n+1)) * (sum(losses) + B) ≤ α
    # With n=10, B=1.0, α=0.4: sum(losses) ≤ 3.4.
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
    # Criterion: R̂_n(λ) + sqrt(log(1/δ) / (2n)) ≤ α
    # With n=10, δ=0.05: UCB addend ≈ 0.387 → very conservative.
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
    # Criterion: Pr(L⁺ ≤ α | ℓ_{1:n}) ≥ β
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
    # Step 8: Compute true risks analytically.
    # true_binomial_risk(λ) = 1 - λ.
    # Special case: if λ = np.inf (no valid λ found), the method defaults
    # to the most conservative choice (prediction set covers everything),
    # which incurs zero risk. true_binomial_risk handles np.inf by returning
    # -inf, but we clamp to 0.0 to correctly mark as non-failure.
    # ------------------------------------------------------------------
    risk_crc: float = _safe_true_risk(lambda_crc)
    risk_rcps: float = _safe_true_risk(lambda_rcps)
    risk_hpd: float = _safe_true_risk(lambda_hpd)

    # ------------------------------------------------------------------
    # Step 9: Construct and return TrialResult.
    # extra={} for Experiment 1 — no additional metrics needed.
    # ------------------------------------------------------------------
    return TrialResult(
        lambda_crc=lambda_crc,
        lambda_rcps=lambda_rcps,
        lambda_hpd=lambda_hpd,
        risk_crc=risk_crc,
        risk_rcps=risk_rcps,
        risk_hpd=risk_hpd,
        extra={},
    )


def _safe_true_risk(lam: float) -> float:
    """Compute true binomial risk with np.inf handling.

    If lam = np.inf (no valid λ found in the grid), the method cannot
    certify any finite threshold, which in practice means it would use
    λ = ∞ (prediction set covers everything → zero risk). We return 0.0
    in this case rather than -inf from 1 - inf.

    Args:
        lam: Chosen threshold λ. May be np.inf.

    Returns:
        True risk = 1 - lam, clamped to 0.0 for lam = np.inf.
    """
    if lam == np.inf:
        # λ = ∞ means the prediction set covers the entire output space.
        # This incurs zero miscoverage risk.
        return 0.0
    return true_binomial_risk(lam)


# ---------------------------------------------------------------------------
# Parallel experiment runner
# ---------------------------------------------------------------------------


def run_experiment_binomial(
    config: BinomialConfig,
) -> List[TrialResult]:
    """Run all M=10,000 trials of the synthetic binomial experiment in parallel.

    Uses joblib.Parallel with the loky backend (default) to distribute trials
    across available CPU cores. Each trial is fully independent — no shared
    mutable state between workers. The config dataclass and lambda_grid numpy
    array are safely serializable by joblib.

    The return order from joblib.Parallel preserves the input order: result[i]
    corresponds to trial i. This is important for reproducibility analysis.

    Args:
        config: BinomialConfig instance with all hyperparameters. Key values
            from config.yaml: M=10000, n_jobs=-1 (all cores), seed=42.

    Returns:
        List of M TrialResult objects, one per trial, in order of trial_idx.
        Each TrialResult contains the chosen λ and true risk for all three
        methods (CRC, RCPS, CBQ-HPD).

    Example:
        >>> config = BinomialConfig()
        >>> config.M = 10  # Small for testing
        >>> config.n_jobs = 1  # Sequential for debugging
        >>> results = run_experiment_binomial(config)
        >>> len(results)
        10
        >>> all(isinstance(r, TrialResult) for r in results)
        True
    """
    # joblib.Parallel with n_jobs=-1 uses all available CPU cores.
    # joblib.delayed wraps run_single_trial_binomial for lazy evaluation.
    # tqdm wraps the range generator to display a progress bar.
    #
    # Note: tqdm progress updates may be irregular with parallel execution
    # (loky backend) — this is cosmetic only and does not affect correctness.
    trial_results: List[TrialResult] = Parallel(n_jobs=config.n_jobs)(
        delayed(run_single_trial_binomial)(i, config)
        for i in tqdm(
            range(config.M),
            desc="Exp1: Synthetic Binomial",
            unit="trial",
        )
    )

    return trial_results


# ---------------------------------------------------------------------------
# Figure 4 data generation
# ---------------------------------------------------------------------------


def compute_L_plus_for_figure4(
    config: BinomialConfig,
) -> Dict[float, np.ndarray]:
    """Generate L⁺ samples for Figure 4 density plots.

    Reproduces Figure 4 from the paper: "Probability density for L⁺ with
    λ ∈ {0.7, 0.8, 0.9} estimated using 100,000 Dirichlet samples."

    For each λ in config.figure4_lambdas = [0.7, 0.8, 0.9] (from config.yaml:
    exp1_synthetic_binomial.figure4_lambdas), this function:
      1. Generates a fixed calibration set using config.seed.
      2. Computes per-sample losses at that λ.
      3. Draws n_mc_figure=100,000 L⁺ samples via Dirichlet Monte Carlo.

    The 100,000 samples (config.yaml: cbq.n_mc_figure = 100000) provide
    sufficient resolution for smooth KDE density estimation in plotting.py.

    Why these λ values:
      - λ=0.7: true risk = 0.3 < α=0.4 (safe, but L⁺ has mass above α)
      - λ=0.8: true risk = 0.2 < α=0.4 (safer, L⁺ more concentrated below α)
      - λ=0.9: true risk = 0.1 < α=0.4 (very safe, L⁺ mostly below α)

    The figure illustrates how the L⁺ distribution shifts left as λ increases,
    showing that Pr(L⁺ ≤ α) increases with λ — the core intuition behind the
    CBQ-HPD decision rule.

    Args:
        config: BinomialConfig instance. Key values from config.yaml:
          - exp1_synthetic_binomial.figure4_lambdas = [0.7, 0.8, 0.9]
          - cbq.n_mc_figure = 100000
          - exp1_synthetic_binomial.n_cal = 10
          - exp1_synthetic_binomial.K = 4
          - exp1_synthetic_binomial.B = 1.0
          - experiment.seed = 42

    Returns:
        Dictionary mapping each λ value to an array of L⁺ samples of shape
        (n_mc_figure,) = (100000,). Keys are the float λ values from
        config.figure4_lambdas. Passed directly to plotting.plot_L_plus_density.

    Example:
        >>> config = BinomialConfig()
        >>> config.n_mc_figure = 1000  # Small for testing
        >>> result = compute_L_plus_for_figure4(config)
        >>> sorted(result.keys())
        [0.7, 0.8, 0.9]
        >>> result[0.7].shape
        (1000,)
        >>> np.all(result[0.7] >= 0.0) and np.all(result[0.7] <= 1.0)
        True
    """
    # Use config.seed directly for a fixed, reproducible calibration set.
    # This is a one-off visualization computation, not a trial, so we use
    # get_rng rather than make_trial_rng.
    rng: np.random.Generator = np.random.default_rng(config.seed)

    # Generate a fixed calibration set representative of the experiment.
    # Shape: (n_cal, K) = (10, 4) per config.yaml.
    cal_data: np.ndarray = generate_binomial_cal_data(
        n=config.n_cal,
        K=config.K,
        rng=rng,
    )

    # Build the result dictionary: λ → L⁺ samples.
    result_dict: Dict[float, np.ndarray] = {}

    # figure4_lambdas = [0.7, 0.8, 0.9] from config.yaml
    # (exp1_synthetic_binomial.figure4_lambdas).
    for lam in config.figure4_lambdas:
        lam_float: float = float(lam)

        # Compute per-sample losses at this λ.
        # Shape: (n_cal,) = (10,) with values in {0, 0.25, 0.5, 0.75, 1.0}.
        losses: np.ndarray = binomial_loss(cal_data, lam_float, config.K)

        # Draw n_mc_figure=100,000 L⁺ samples via Dirichlet Monte Carlo.
        # This is the same computation as in the decision rule but with
        # 100x more samples for smooth density estimation.
        # Shape: (n_mc_figure,) = (100000,).
        L_plus_samples: np.ndarray = cbq_core.compute_L_plus_samples(
            losses=losses,
            B=config.B,
            num_samples=config.n_mc_figure,
            rng=rng,
        )

        result_dict[lam_float] = L_plus_samples

    return result_dict
