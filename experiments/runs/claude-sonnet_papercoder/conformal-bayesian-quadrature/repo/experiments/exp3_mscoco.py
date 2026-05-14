## experiments/exp3_mscoco.py
"""Experiment 3: False Negative Rate on MS-COCO (Section 5.3 of the paper).

This module implements the third experiment from "Conformal Prediction as
Bayesian Quadrature". The experiment controls the false negative rate (FNR)
of multilabel classification on MS-COCO, mirroring the setup from Angelopoulos
& Bates (2023, Section 5.1).

Setup (from config.yaml / MSCOCOConfig):
  - M = 10,000 random trials
  - n_cal = 1000 calibration examples per split
  - n_test = 3952 test examples per split
  - α = 0.1 (target FNR level)
  - β = 0.95 (confidence level for CBQ-HPD)
  - B = 1.0 (upper bound on FNR loss, which is in [0, 1])
  - λ_grid = np.linspace(0, 1, 500)

Loss function (Section 5.3, mirrors Angelopoulos & Bates 2023 §5.1):
  The control parameter λ ∈ [0, 1] is a tolerance parameter where the
  prediction set is C_λ(x) = {c : score_c ≥ 1 - λ}. This ensures the
  FNR loss is monotonically non-increasing in λ (larger λ → larger
  prediction sets → fewer false negatives → lower FNR), satisfying the
  CRC monotonicity assumption (Proposition 3.2).

  FNR_i(λ) = Σ_c [labels_ic · 1{score_ic < 1 - λ}] / Σ_c [labels_ic]

True risk: empirical FNR on the 3952 held-out test examples.
Risk threshold: empirical_fnr_risk(test_scores, test_labels, λ) > α = 0.1

Expected results (Table 3):
  CRC:          ~45.05% failure rate, pred set size ~2.92
  RCPS:          ~0.00% failure rate, pred set size ~3.57
  CBQ-HPD β=0.95: ~5.43% failure rate, pred set size ~3.04

The high CRC failure rate (45.05%) arises because the marginal guarantee
averages over many calibration draws. CBQ-HPD achieves a balance between
RCPS's conservatism and CRC's aggressiveness.

References:
    Paper Section 5.3: False Negative Rate on MS-COCO.
    Paper Table 3: Expected numerical results.
    Angelopoulos & Bates (2023, Section 5.1): experimental setup mirrored here.
    config.yaml: exp3_mscoco.* settings.
    MSCOCOConfig in config.py: all hyperparameters.
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

from config import MSCOCOConfig
from data.mscoco_loader import MSCOCOLoader
from decision_rules import (
    compute_losses_grid,
    find_lambda_cbq_hpd,
    find_lambda_crc,
    find_lambda_rcps,
)
from evaluation import TrialResult
from loss_functions import empirical_fnr_risk, fnr_loss, pred_set_size
from utils import make_trial_rng


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------


def run_single_trial_mscoco(
    trial_idx: int,
    config: MSCOCOConfig,
    scores: np.ndarray,
    labels: np.ndarray,
) -> TrialResult:
    """Run one complete trial of the MS-COCO FNR experiment.

    This function is the unit of parallelism — it is called M=10,000 times
    via joblib.Parallel. Each call is fully independent: it creates its own
    RNG, draws a random calibration/test split, runs all three decision rules,
    and evaluates the empirical FNR on the held-out test set.

    Parameterization note:
        The control parameter λ ∈ [0, 1] is a tolerance parameter where the
        prediction set is C_λ(x) = {c : score_c ≥ 1 - λ}. This ensures the
        FNR loss is monotonically non-increasing in λ (larger λ → larger
        prediction sets → fewer false negatives → lower FNR), satisfying the
        CRC monotonicity assumption required by Proposition 3.2.

        Concretely, fnr_loss(scores, labels, lam) computes FNR with score
        threshold = 1 - lam. The lambda grid [0, 1] maps to score thresholds
        [1, 0]. The infimum search finds the smallest λ (most conservative,
        largest prediction sets) satisfying the risk criterion.

    Steps:
      1. Create trial-specific RNG via make_trial_rng(config.seed, trial_idx).
      2. Draw random calibration/test split (1000 cal + 3952 test).
      3. Define loss closure: loss_fn(lam) = fnr_loss(cal_scores, cal_labels, lam).
      4. Pre-compute losses over the entire lambda grid (shape: 500 × 1000).
      5. Find λ_crc via Conformal Risk Control (Proposition 3.2).
      6. Find λ_rcps via RCPS with Hoeffding UCB (Bates et al., 2021).
      7. Find λ_hpd via CBQ-HPD at β=0.95 (Corollary 4.4).
      8. Evaluate empirical FNR on test set for each method.
      9. Compute mean prediction set sizes on test set.
      10. Return TrialResult with all values.

    Args:
        trial_idx: Zero-based trial index in [0, M-1]. Used to seed the
            trial-specific RNG via make_trial_rng(config.seed, trial_idx).
            With config.seed=42 and trial_idx in [0, 9999], seeds range
            from 42 to 10041 — all distinct and valid.
        config: MSCOCOConfig instance with all hyperparameters. Key values
            from config.yaml: n_cal=1000, n_test=3952, alpha=0.1, beta=0.95,
            B=1.0, n_mc_samples=1000, lambda_grid=np.linspace(0, 1, 500).
        scores: Precomputed softmax scores of shape (N, C) with values in
            [0, 1]. Loaded once in run_experiment_mscoco and passed to all
            parallel workers to avoid repeated disk I/O.
        labels: Binary ground-truth labels of shape (N, C) with values in
            {0, 1}. Same shape as scores.

    Returns:
        TrialResult with:
          - lambda_crc: λ chosen by CRC (or np.inf if no valid λ found).
          - lambda_rcps: λ chosen by RCPS (or np.inf).
          - lambda_hpd: λ chosen by CBQ-HPD (or np.inf).
          - risk_crc: Empirical FNR on test set at lambda_crc (0.0 if np.inf).
          - risk_rcps: Empirical FNR on test set at lambda_rcps (0.0 if np.inf).
          - risk_hpd: Empirical FNR on test set at lambda_hpd (0.0 if np.inf).
          - extra: {
              'pred_set_size_crc':  mean prediction set size at lambda_crc,
              'pred_set_size_rcps': mean prediction set size at lambda_rcps,
              'pred_set_size_hpd':  mean prediction set size at lambda_hpd,
            }

    Example:
        >>> import numpy as np
        >>> config = MSCOCOConfig()
        >>> config.M = 5
        >>> config.n_mc_samples = 100
        >>> # Synthetic data for testing (not real MS-COCO)
        >>> rng = np.random.default_rng(0)
        >>> scores = rng.uniform(0, 1, size=(5000, 80)).astype(np.float32)
        >>> labels = rng.integers(0, 2, size=(5000, 80)).astype(np.int32)
        >>> result = run_single_trial_mscoco(0, config, scores, labels)
        >>> isinstance(result, TrialResult)
        True
        >>> 0.0 <= result.risk_crc <= 1.0
        True
        >>> result.extra['pred_set_size_crc'] >= 0.0
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
    # Step 2: Draw random calibration/test split without replacement.
    # Total required: n_cal + n_test = 1000 + 3952 = 4952 examples.
    # The first n_cal indices form the calibration set; the rest form the
    # test set. This mirrors MSCOCOLoader.get_random_split but is inlined
    # here to avoid re-instantiating the loader in each parallel worker.
    # ------------------------------------------------------------------
    n_total_pool: int = scores.shape[0]
    n_required: int = config.n_cal + config.n_test

    # Sample n_required unique indices from [0, N) without replacement.
    # rng.choice with replace=False guarantees no duplicate indices.
    # Shape: (n_required,) = (4952,)
    all_indices: np.ndarray = rng.choice(
        n_total_pool, size=n_required, replace=False
    )

    # Partition: first n_cal → calibration, remaining n_test → test.
    cal_idx: np.ndarray = all_indices[: config.n_cal]    # shape (1000,)
    test_idx: np.ndarray = all_indices[config.n_cal :]   # shape (3952,)

    # ------------------------------------------------------------------
    # Step 3: Extract calibration and test subsets via fancy indexing.
    # Fancy indexing produces copies (not views), which is safe for
    # parallel workers that may modify arrays independently.
    # ------------------------------------------------------------------
    cal_scores: np.ndarray = scores[cal_idx]    # shape (1000, C)
    cal_labels: np.ndarray = labels[cal_idx]    # shape (1000, C)
    test_scores: np.ndarray = scores[test_idx]  # shape (3952, C)
    test_labels: np.ndarray = labels[test_idx]  # shape (3952, C)

    # ------------------------------------------------------------------
    # Step 4: Define the loss function closure.
    # Captures cal_scores and cal_labels from the enclosing scope.
    # Signature: loss_fn(lam: float) -> np.ndarray of shape (n_cal,).
    #
    # Parameterization: lam ∈ [0, 1] is a tolerance parameter where the
    # prediction set is C_λ(x) = {c : score_c ≥ 1 - λ}. This ensures
    # FNR is monotonically non-increasing in λ (larger λ → larger sets
    # → fewer false negatives → lower FNR), satisfying the CRC assumption.
    #
    # fnr_loss(scores, labels, lam) uses score threshold = lam directly,
    # so we pass (1 - lam) as the threshold to achieve the correct
    # monotonicity direction.
    # ------------------------------------------------------------------
    def loss_fn(lam: float) -> np.ndarray:
        """Per-sample FNR loss at tolerance parameter lam.

        Prediction set: C_λ(x) = {c : score_c ≥ 1 - λ}
        FNR_i(λ) = Σ_c [labels_ic · 1{score_ic < 1 - λ}] / Σ_c [labels_ic]

        This is non-increasing in λ: larger λ → lower score threshold
        → more labels included → fewer false negatives → lower FNR.
        """
        # Convert tolerance parameter λ to score threshold (1 - λ).
        # fnr_loss uses prediction set {c : score_c >= score_threshold}.
        score_threshold: float = 1.0 - lam
        return fnr_loss(cal_scores, cal_labels, score_threshold)

    # ------------------------------------------------------------------
    # Step 5: Pre-compute losses over the entire lambda grid.
    # Shape: (G, n_cal) = (500, 1000) for lambda_grid=np.linspace(0,1,500).
    # This avoids redundant loss evaluations across the three decision rules.
    # Each row j contains the per-sample FNR losses at lambda_grid[j].
    # ------------------------------------------------------------------
    losses_per_lambda: np.ndarray = compute_losses_grid(
        loss_fn=loss_fn,
        lambda_grid=config.lambda_grid,
    )

    # ------------------------------------------------------------------
    # Step 6: Find λ_crc via Conformal Risk Control (Proposition 3.2).
    # Criterion: (1/(n+1)) * (sum(losses) + B) ≤ α
    # With n=1000, B=1.0, α=0.1:
    #   (Σ FNR_i(λ) + 1) / 1001 ≤ 0.1
    #   Σ FNR_i(λ) ≤ 99.1
    # Since FNR_i ∈ [0, 1], this requires at most ~9.9% average FNR.
    # ------------------------------------------------------------------
    lambda_crc: float = find_lambda_crc(
        losses_per_lambda=losses_per_lambda,
        lambda_grid=config.lambda_grid,
        B=config.B,
        alpha=config.alpha,
    )

    # ------------------------------------------------------------------
    # Step 7: Find λ_rcps via RCPS with Hoeffding UCB (Bates et al., 2021).
    # delta = 1 - beta = 0.05 (config.yaml: rcps.delta = 0.05).
    # Hoeffding UCB: R̂(λ) + sqrt(log(1/0.05) / (2·1000)) ≤ 0.1
    # sqrt(log(20)/2000) ≈ sqrt(2.996/2000) ≈ 0.0387
    # So needs R̂(λ) ≤ 0.1 - 0.0387 ≈ 0.0613 — more conservative than CRC.
    # This explains the larger prediction sets (mean size ~3.57 in Table 3).
    # ------------------------------------------------------------------
    delta: float = 1.0 - config.beta  # = 0.05 per config.yaml (rcps.delta)
    lambda_rcps: float = find_lambda_rcps(
        losses_per_lambda=losses_per_lambda,
        lambda_grid=config.lambda_grid,
        alpha=config.alpha,
        delta=delta,
    )

    # ------------------------------------------------------------------
    # Step 8: Find λ_hpd via CBQ-HPD at β=0.95 (Corollary 4.4).
    # Criterion: Pr(L⁺ ≤ α | ℓ_{1:n}) ≥ β = 0.95
    # Uses n_mc_samples=1000 Dirichlet samples per lambda evaluation.
    # The same rng is passed — it has been advanced by steps 2-5 but
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
    # Step 9: Evaluate empirical FNR on the held-out test set.
    # empirical_fnr_risk computes mean(fnr_loss(test_scores, test_labels, threshold))
    # over the 3952 test examples. Risk exceeds α=0.1 if this value > 0.1.
    #
    # We apply the same tolerance-to-threshold conversion: score_threshold = 1 - λ.
    # np.inf handling: if λ = np.inf, the score threshold = 1 - inf = -inf,
    # meaning all labels are included → FNR = 0. We handle this explicitly.
    # ------------------------------------------------------------------
    risk_crc: float = _safe_empirical_fnr_risk(
        test_scores, test_labels, lambda_crc
    )
    risk_rcps: float = _safe_empirical_fnr_risk(
        test_scores, test_labels, lambda_rcps
    )
    risk_hpd: float = _safe_empirical_fnr_risk(
        test_scores, test_labels, lambda_hpd
    )

    # ------------------------------------------------------------------
    # Step 10: Compute mean prediction set sizes on the test set.
    # pred_set_size(scores, threshold) = mean(sum(scores >= threshold, axis=1))
    # We apply the tolerance-to-threshold conversion: threshold = 1 - λ.
    # np.inf handling: if λ = np.inf, threshold = -inf → all labels included.
    # ------------------------------------------------------------------
    pss_crc: float = _safe_pred_set_size(test_scores, lambda_crc)
    pss_rcps: float = _safe_pred_set_size(test_scores, lambda_rcps)
    pss_hpd: float = _safe_pred_set_size(test_scores, lambda_hpd)

    # ------------------------------------------------------------------
    # Step 11: Construct and return TrialResult.
    # extra dict stores prediction set sizes for Table 3's "Pred. Set Size"
    # column. Keys follow the convention expected by evaluation._aggregate_extra:
    # "pred_set_size_{method_suffix}".
    # ------------------------------------------------------------------
    return TrialResult(
        lambda_crc=lambda_crc,
        lambda_rcps=lambda_rcps,
        lambda_hpd=lambda_hpd,
        risk_crc=risk_crc,
        risk_rcps=risk_rcps,
        risk_hpd=risk_hpd,
        extra={
            "pred_set_size_crc": pss_crc,
            "pred_set_size_rcps": pss_rcps,
            "pred_set_size_hpd": pss_hpd,
        },
    )


# ---------------------------------------------------------------------------
# Private helpers for np.inf handling
# ---------------------------------------------------------------------------


def _safe_empirical_fnr_risk(
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    lam: float,
) -> float:
    """Compute empirical FNR risk with np.inf handling.

    If lam = np.inf (no valid λ found in the grid), the tolerance parameter
    is infinite, meaning the score threshold = 1 - inf = -inf, so all labels
    are included in the prediction set. This incurs zero FNR risk.

    Args:
        test_scores: Test set softmax scores of shape (n_test, C).
        test_labels: Test set binary labels of shape (n_test, C).
        lam: Tolerance parameter λ. May be np.inf.

    Returns:
        Empirical mean FNR on the test set, or 0.0 if lam = np.inf.
    """
    if lam == np.inf:
        # λ = ∞ → score threshold = 1 - ∞ = -∞ → all labels included → FNR = 0.
        return 0.0
    # Convert tolerance parameter λ to score threshold (1 - λ).
    score_threshold: float = 1.0 - lam
    return empirical_fnr_risk(test_scores, test_labels, score_threshold)


def _safe_pred_set_size(
    test_scores: np.ndarray,
    lam: float,
) -> float:
    """Compute mean prediction set size with np.inf handling.

    If lam = np.inf, the score threshold = 1 - inf = -inf, so all labels
    are included. The prediction set size equals the total number of classes C.

    Args:
        test_scores: Test set softmax scores of shape (n_test, C).
        lam: Tolerance parameter λ. May be np.inf.

    Returns:
        Mean prediction set size across test examples, or C (all classes)
        if lam = np.inf.
    """
    if lam == np.inf:
        # λ = ∞ → all labels included → set size = C (number of classes).
        C: int = test_scores.shape[1]
        return float(C)
    # Convert tolerance parameter λ to score threshold (1 - λ).
    score_threshold: float = 1.0 - lam
    return pred_set_size(test_scores, score_threshold)


# ---------------------------------------------------------------------------
# Parallel experiment runner
# ---------------------------------------------------------------------------


def run_experiment_mscoco(
    config: MSCOCOConfig,
    loader: MSCOCOLoader,
) -> List[TrialResult]:
    """Run all M=10,000 trials of the MS-COCO FNR experiment in parallel.

    Loads the precomputed MS-COCO scores and labels once in the main process,
    then distributes the 10,000 trials across available CPU cores via
    joblib.Parallel. Each trial receives the full scores and labels arrays
    as arguments — joblib's loky backend serializes these once per worker
    process (not once per trial), making this efficient for large arrays.

    Data loading:
        Calls loader.load() to read scores.npy and labels.npy from disk.
        If the files are not found, loader.load() raises FileNotFoundError
        with instructions for downloading from the conformal-risk-control
        GitHub repository.

    Validation:
        Checks that the total pool size N >= n_cal + n_test = 4952 before
        starting the parallel loop. This prevents cryptic errors mid-run.

    Parallelism note:
        With n_jobs=-1 (all cores) and 10,000 trials, each trial involves:
          - 1 random split (fast)
          - 500 × fnr_loss evaluations over 1000 calibration samples (moderate)
          - CRC and RCPS (vectorized, fast)
          - CBQ-HPD: 500 lambda values × 1000 Dirichlet samples (dominant cost)
          - 2 test set evaluations (fast)
        Total runtime depends on hardware; expect 30-120 minutes on a modern
        multi-core machine.

    Args:
        config: MSCOCOConfig instance with all hyperparameters. Key values
            from config.yaml: M=10000, n_jobs=-1 (all cores), seed=42,
            n_cal=1000, n_test=3952, alpha=0.1, beta=0.95, B=1.0,
            n_mc_samples=1000, lambda_grid=np.linspace(0, 1, 500),
            scores_path="data/mscoco/scores.npy",
            labels_path="data/mscoco/labels.npy".
        loader: MSCOCOLoader instance initialized with the paths from config.
            load() will be called here if not already called. The loader's
            scores and labels arrays are extracted and passed to workers.

    Returns:
        List of M TrialResult objects, one per trial, in order of trial_idx.
        Each TrialResult contains the chosen λ, empirical FNR risk, and
        mean prediction set size for all three methods (CRC, RCPS, CBQ-HPD).

    Raises:
        FileNotFoundError: If the MS-COCO data files are not found at the
            paths specified in config. See MSCOCOLoader.load() for details.
        ValueError: If the total pool size N < n_cal + n_test = 4952.

    Example:
        >>> import numpy as np
        >>> config = MSCOCOConfig()
        >>> config.M = 5  # Small for testing
        >>> config.n_jobs = 1  # Sequential for debugging
        >>> config.n_mc_samples = 100  # Fast for testing
        >>> # Use synthetic data for testing (not real MS-COCO)
        >>> loader = MSCOCOLoader.__new__(MSCOCOLoader)
        >>> loader.scores_path = "dummy"
        >>> loader.labels_path = "dummy"
        >>> rng = np.random.default_rng(0)
        >>> loader.scores = rng.uniform(0, 1, size=(5000, 80)).astype(np.float32)
        >>> loader.labels = rng.integers(0, 2, size=(5000, 80)).astype(np.int32)
        >>> results = run_experiment_mscoco(config, loader)
        >>> len(results)
        5
        >>> all(isinstance(r, TrialResult) for r in results)
        True
    """
    # ------------------------------------------------------------------
    # Step 1: Load data if not already loaded.
    # loader.load() is idempotent in the sense that it re-loads from disk
    # each time it is called. We check if data is already present to avoid
    # redundant I/O when the caller has pre-loaded the data.
    # ------------------------------------------------------------------
    if loader.scores is None or loader.labels is None:
        loader.load()

    # Extract arrays from the loader for direct use in parallel workers.
    # Assigning to local variables avoids repeated attribute lookups and
    # makes the type checker happy (loader.scores could be None before load()).
    scores: np.ndarray = loader.scores  # type: ignore[assignment]
    labels: np.ndarray = loader.labels  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Step 2: Validate pool size before starting the parallel loop.
    # This prevents cryptic errors mid-run if the dataset is too small.
    # ------------------------------------------------------------------
    n_total_pool: int = scores.shape[0]
    n_required: int = config.n_cal + config.n_test  # 1000 + 3952 = 4952

    if n_total_pool < n_required:
        raise ValueError(
            f"Total pool size N={n_total_pool} is smaller than the minimum "
            f"required for one split (n_cal + n_test = {n_required}). "
            f"The MS-COCO dataset must have at least {n_required} examples. "
            f"Check the data files at:\n"
            f"  scores: {config.scores_path}\n"
            f"  labels: {config.labels_path}\n"
            f"Download from: https://github.com/aangelopoulos/conformal-risk-control"
        )

    # Log dataset info for transparency.
    n_classes: int = scores.shape[1]
    print(
        f"[Exp3 MS-COCO] Dataset: N={n_total_pool} examples, C={n_classes} classes. "
        f"Running M={config.M} trials with n_cal={config.n_cal}, "
        f"n_test={config.n_test}, α={config.alpha}, β={config.beta}."
    )

    # ------------------------------------------------------------------
    # Step 3: Run all M trials in parallel via joblib.
    # joblib.Parallel with n_jobs=-1 uses all available CPU cores.
    # joblib.delayed wraps run_single_trial_mscoco for lazy evaluation.
    # tqdm wraps the range generator to display a progress bar.
    #
    # Serialization note: joblib's loky backend serializes scores and labels
    # once per worker process (not once per task), so the overhead is
    # proportional to n_jobs, not M. For scores of shape ~(5000, 80) as
    # float32 (~1.6 MB), this is negligible.
    #
    # The return order from joblib.Parallel preserves the input order:
    # result[i] corresponds to trial i. This is important for reproducibility.
    # ------------------------------------------------------------------
    trial_results: List[TrialResult] = Parallel(n_jobs=config.n_jobs)(
        delayed(run_single_trial_mscoco)(i, config, scores, labels)
        for i in tqdm(
            range(config.M),
            desc="Exp3: MS-COCO FNR",
            unit="trial",
        )
    )

    return trial_results
