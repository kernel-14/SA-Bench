## decision_rules.py
"""Decision rules for Conformal Bayesian Quadrature experiments.

This module implements the three decision rules compared in the paper's
experiments (Section 5):

  1. CRC  — Conformal Risk Control (Angelopoulos et al., 2024), recovered as
             the infimum λ where E[L⁺] ≤ α (Proposition 3.2, Section 4.6).
  2. RCPS — Risk-Controlling Prediction Sets (Bates et al., 2021) with a
             Hoeffding upper confidence bound (config.yaml: rcps.ucb_type).
  3. CBQ-HPD — Our proposed method: infimum λ where Pr(L⁺ ≤ α) ≥ β
               (Section 5, Corollary 4.4, equation 29).

All three decision rules share the same interface:
  - They receive ``losses_per_lambda`` of shape (G, n_cal) pre-computed by
    ``compute_losses_grid``, where G = len(lambda_grid).
  - They return a single float: the chosen λ (or np.inf if no λ satisfies
    the criterion).

The ``compute_losses_grid`` helper evaluates a loss closure over the entire
lambda grid, producing the shared input consumed by all three rules.

References:
    Paper Section 5: Experimental setup and decision rules.
    Paper Proposition 3.2 / Section 4.6: CRC recovery from E[L⁺].
    Paper Corollary 4.4: CBQ-HPD via the β-quantile of L⁺.
    Bates et al. (2021): RCPS with Hoeffding UCB.
    config.yaml: rcps.delta=0.05, rcps.ucb_type="hoeffding",
                 cbq.n_mc_samples=1000, exp*.alpha, exp*.beta, exp*.B.
"""

import math
from typing import Callable

import numpy as np

import cbq_core
from utils import safe_inf_search


# ---------------------------------------------------------------------------
# Helper: pre-compute losses over the entire lambda grid
# ---------------------------------------------------------------------------


def compute_losses_grid(
    loss_fn: Callable[[float], np.ndarray],
    lambda_grid: np.ndarray,
) -> np.ndarray:
    """Evaluate a loss function over every point in the lambda grid.

    This helper is called once per trial before invoking any of the three
    decision rules. By pre-computing all losses up front, we avoid redundant
    evaluations: each of the three rules (CRC, RCPS, CBQ-HPD) can then
    operate on the same ``losses_per_lambda`` array without re-calling the
    loss function.

    The ``loss_fn`` is a closure over the calibration data for the current
    trial. It accepts a single float ``lam`` and returns a 1-D array of
    shape ``(n_cal,)`` containing the per-sample losses at that threshold.

    Args:
        loss_fn: A callable with signature ``loss_fn(lam: float) ->
            np.ndarray`` of shape ``(n_cal,)``. Typically a closure over
            calibration data, e.g.::

                loss_fn = lambda lam: binomial_loss(cal_data, lam, K=4)

            The function must be deterministic (same output for same input)
            since it is called once per grid point.
        lambda_grid: 1-D array of lambda values sorted in ascending order,
            shape ``(G,)``. Created by ``np.linspace`` in config.py, so
            monotonicity is guaranteed. Examples from config.yaml:
              - Exp1: np.linspace(0.0, 1.0, 500)
              - Exp2: np.linspace(0.0, 20.0, 1000)
              - Exp3: np.linspace(0.0, 1.0, 500)

    Returns:
        2-D array of shape ``(G, n_cal)`` where ``result[j, i]`` is the
        loss for calibration sample ``i`` at threshold ``lambda_grid[j]``.
        Row ``j`` corresponds to ``lambda_grid[j]``.

    Example:
        >>> import numpy as np
        >>> cal_data = np.array([[0.3, 0.7, 0.5, 0.9]])  # shape (1, 4)
        >>> loss_fn = lambda lam: np.mean(cal_data > lam, axis=1)
        >>> grid = np.array([0.0, 0.5, 1.0])
        >>> losses = compute_losses_grid(loss_fn, grid)
        >>> losses.shape
        (3, 1)
        >>> losses[0]   # lam=0.0: all 4 entries > 0 → loss = 1.0
        array([1.])
        >>> losses[2]   # lam=1.0: no entries > 1 → loss = 0.0
        array([0.])
    """
    # Evaluate loss_fn at each grid point and collect results into a list.
    # Each element has shape (n_cal,); stacking gives shape (G, n_cal).
    losses_list: list[np.ndarray] = [loss_fn(float(lam)) for lam in lambda_grid]

    # np.array on a list of equal-length 1-D arrays produces shape (G, n_cal).
    losses_per_lambda: np.ndarray = np.array(losses_list, dtype=float)

    return losses_per_lambda


# ---------------------------------------------------------------------------
# Private helper: Hoeffding UCB scalar computation
# ---------------------------------------------------------------------------


def _hoeffding_ucb(
    empirical_risk: float,
    n: int,
    delta: float,
) -> float:
    """Compute the Hoeffding upper confidence bound for a single lambda.

    Implements the Hoeffding UCB from Bates et al. (2021) and config.yaml
    (rcps.ucb_type = "hoeffding", rcps.delta = 0.05):

        UCB(λ) = R̂ₙ(λ) + sqrt(log(1/δ) / (2n))

    where R̂ₙ(λ) is the empirical risk and δ is the failure probability.
    The UCB term sqrt(log(1/δ) / (2n)) is constant across all λ for a
    fixed calibration set size n and failure probability δ.

    This scalar version is provided for clarity and unit testing. The
    vectorized equivalent is used inline in ``find_lambda_rcps`` for
    performance.

    Args:
        empirical_risk: Empirical mean loss R̂ₙ(λ) = (1/n) Σᵢ ℓ(zᵢ, λ).
            Should be in [0, B] where B = 1.0 for all experiments.
        n: Number of calibration samples. Must be > 0.
            Per config.yaml: n_cal = 10 (Exp1), 200 (Exp2), 1000 (Exp3).
        delta: Failure probability for the Hoeffding bound. Per config.yaml
            (rcps.delta = 0.05), this equals 1 - beta = 1 - 0.95 = 0.05.

    Returns:
        Scalar float equal to empirical_risk + sqrt(log(1/delta) / (2*n)).
        This is the upper confidence bound on the true risk.

    Example:
        >>> _hoeffding_ucb(0.3, n=10, delta=0.05)  # doctest: +ELLIPSIS
        0.6...
        >>> # Hoeffding term for n=10, delta=0.05:
        >>> import math
        >>> math.sqrt(math.log(1/0.05) / (2*10))  # doctest: +ELLIPSIS
        0.38...
        >>> # UCB = 0.3 + 0.38... ≈ 0.68...
        >>> _hoeffding_ucb(0.0, n=200, delta=0.05)  # doctest: +ELLIPSIS
        0.12...
    """
    # Hoeffding correction: sqrt(log(1/δ) / (2n))
    # math.log is used for scalar computation (slightly faster than np.log
    # for scalars and avoids numpy overhead).
    hoeffding_term: float = math.sqrt(math.log(1.0 / delta) / (2.0 * n))
    return empirical_risk + hoeffding_term


# ---------------------------------------------------------------------------
# Decision Rule 1: Conformal Risk Control (CRC)
# ---------------------------------------------------------------------------


def find_lambda_crc(
    losses_per_lambda: np.ndarray,
    lambda_grid: np.ndarray,
    B: float,
    alpha: float,
) -> float:
    """Find λ_crc via the Conformal Risk Control criterion (Proposition 3.2).

    Implements the CRC decision rule from Proposition 3.2 and Section 4.6
    of the paper. This is the infimum λ where the analytical mean of L⁺
    does not exceed α:

        λ_crc = inf{λ : (1/(n+1)) · (Σᵢ₌₁ⁿ ℓ(zᵢ, λ) + B) ≤ α}

    This is equivalent to the Conformal Risk Control guarantee from
    Angelopoulos et al. (2024, equation 4) and is recovered from our
    framework by taking E[L⁺] = (1/(n+1))(Σℓᵢ + B) (Section 4.6).

    The criterion is vectorized: all G row sums are computed simultaneously
    with ``np.sum(losses_per_lambda, axis=1)``, making this O(G × n_cal)
    with no Python loops.

    Note on monotonicity: For monotone non-increasing loss functions (as
    assumed by CRC), the criterion (1/(n+1))(Σℓᵢ + B) is non-increasing
    in λ. Therefore, the condition array transitions from False to True at
    most once, and ``safe_inf_search`` correctly finds the infimum.

    Args:
        losses_per_lambda: Pre-computed losses of shape (G, n_cal) where
            ``losses_per_lambda[j, i]`` is the loss for calibration sample
            ``i`` at ``lambda_grid[j]``. Produced by ``compute_losses_grid``.
            Values should be in [0, B].
        lambda_grid: 1-D array of lambda values sorted ascending, shape (G,).
            Must have the same length as ``losses_per_lambda.shape[0]``.
        B: Upper bound on individual losses. Set to 1.0 for all experiments
            per config.yaml (exp1_synthetic_binomial.B = 1.0,
            exp2_synthetic_heteroskedastic.B = 1.0, exp3_mscoco.B = 1.0).
            Represents ℓ₍ₙ₊₁₎ = B in the L⁺ construction.
        alpha: Target risk level. Per config.yaml:
            - exp1_synthetic_binomial.alpha = 0.4
            - exp2_synthetic_heteroskedastic.alpha = 0.1
            - exp3_mscoco.alpha = 0.1

    Returns:
        The smallest λ in ``lambda_grid`` where the CRC criterion is
        satisfied, or ``np.inf`` if no such λ exists (method cannot achieve
        the target risk with the given calibration data).

    Example:
        >>> import numpy as np
        >>> # n_cal=3, B=1.0, alpha=0.4
        >>> # losses at lam=0.5: [0.5, 0.5, 0.5] → criterion = (1.5+1)/4 = 0.625 > 0.4
        >>> # losses at lam=0.8: [0.2, 0.2, 0.2] → criterion = (0.6+1)/4 = 0.4 ≤ 0.4
        >>> losses_grid = np.array([[0.5, 0.5, 0.5], [0.2, 0.2, 0.2]])
        >>> grid = np.array([0.5, 0.8])
        >>> find_lambda_crc(losses_grid, grid, B=1.0, alpha=0.4)
        0.8
        >>> # No lambda satisfies criterion
        >>> losses_high = np.array([[0.9, 0.9, 0.9], [0.8, 0.8, 0.8]])
        >>> find_lambda_crc(losses_high, grid, B=1.0, alpha=0.4)
        inf
    """
    # Number of calibration samples (n in the paper).
    n_cal: int = losses_per_lambda.shape[1]

    # Vectorized computation of the CRC criterion for all G lambda values.
    # Step 1: Sum losses across calibration samples for each lambda.
    #         np.sum(axis=1) gives shape (G,): Σᵢ ℓ(zᵢ, λⱼ) for each j.
    row_sums: np.ndarray = np.sum(losses_per_lambda, axis=1)  # shape (G,)

    # Step 2: Compute CRC criterion = (Σℓᵢ + B) / (n + 1) for each lambda.
    #         This is E[L⁺] from Section 4.6 of the paper.
    criteria: np.ndarray = (row_sums + B) / (n_cal + 1)  # shape (G,)

    # Step 3: Boolean condition: criterion ≤ α.
    condition: np.ndarray = criteria <= alpha  # shape (G,), dtype bool

    # Step 4: Return the infimum lambda satisfying the condition.
    return safe_inf_search(lambda_grid, condition)


# ---------------------------------------------------------------------------
# Decision Rule 2: RCPS with Hoeffding UCB
# ---------------------------------------------------------------------------


def find_lambda_rcps(
    losses_per_lambda: np.ndarray,
    lambda_grid: np.ndarray,
    alpha: float,
    delta: float = 0.05,
) -> float:
    """Find λ_rcps via the RCPS Hoeffding upper confidence bound.

    Implements the Risk-Controlling Prediction Sets (RCPS) baseline from
    Bates et al. (2021) with a Hoeffding UCB. From config.yaml:
    ``rcps.ucb_type = "hoeffding"`` and ``rcps.delta = 0.05``.

    The decision rule is:

        λ_rcps = inf{λ : R̂ₙ(λ) + sqrt(log(1/δ) / (2n)) ≤ α}

    where:
      - R̂ₙ(λ) = (1/n) Σᵢ ℓ(zᵢ, λ) is the empirical risk
      - δ = 1 − β = 0.05 is the failure probability (config.yaml: rcps.delta)
      - n = n_cal is the calibration set size

    The Hoeffding correction term sqrt(log(1/δ) / (2n)) is constant across
    all λ for a fixed calibration set, so it is computed once and broadcast.

    RCPS is more conservative than CRC because the Hoeffding bound provides
    a high-probability guarantee (not just a marginal one). This explains
    why RCPS achieves 0% failure rate in Table 1 but selects larger λ
    (larger prediction sets) than our CBQ-HPD method.

    Args:
        losses_per_lambda: Pre-computed losses of shape (G, n_cal).
            Produced by ``compute_losses_grid``. Values in [0, B].
        lambda_grid: 1-D array of lambda values sorted ascending, shape (G,).
        alpha: Target risk level. Per config.yaml:
            - exp1_synthetic_binomial.alpha = 0.4
            - exp2_synthetic_heteroskedastic.alpha = 0.1
            - exp3_mscoco.alpha = 0.1
        delta: Failure probability for the Hoeffding bound. Default 0.05
            per config.yaml (rcps.delta = 0.05), equal to 1 − β = 1 − 0.95.
            Must be in (0, 1).

    Returns:
        The smallest λ in ``lambda_grid`` where the Hoeffding UCB does not
        exceed α, or ``np.inf`` if no such λ exists.

    Example:
        >>> import numpy as np
        >>> # n_cal=200, delta=0.05: Hoeffding term = sqrt(log(20)/(400)) ≈ 0.0869
        >>> # For losses all = 0.0: UCB = 0.0 + 0.0869 = 0.0869 ≤ 0.1 → satisfied
        >>> losses_zero = np.zeros((3, 200))
        >>> grid = np.array([0.0, 0.5, 1.0])
        >>> result = find_lambda_rcps(losses_zero, grid, alpha=0.1, delta=0.05)
        >>> result == 0.0  # First lambda satisfies criterion
        True
        >>> # High losses: UCB > alpha for all lambda
        >>> losses_high = np.ones((3, 200)) * 0.5
        >>> find_lambda_rcps(losses_high, grid, alpha=0.1, delta=0.05)
        inf
    """
    # Number of calibration samples.
    n_cal: int = losses_per_lambda.shape[1]

    # Compute the Hoeffding correction term once — it is constant for all λ.
    # From config.yaml: rcps.ucb_type = "hoeffding", rcps.delta = 0.05.
    # Term = sqrt(log(1/δ) / (2n))
    hoeffding_term: float = math.sqrt(math.log(1.0 / delta) / (2.0 * n_cal))

    # Vectorized empirical risk computation for all G lambda values.
    # np.mean(axis=1) gives shape (G,): R̂ₙ(λⱼ) = (1/n) Σᵢ ℓ(zᵢ, λⱼ).
    empirical_risks: np.ndarray = np.mean(losses_per_lambda, axis=1)  # shape (G,)

    # Compute UCBs: R̂ₙ(λ) + Hoeffding term for each lambda.
    # Broadcasting: hoeffding_term is a scalar added to each element.
    ucbs: np.ndarray = empirical_risks + hoeffding_term  # shape (G,)

    # Boolean condition: UCB ≤ α.
    condition: np.ndarray = ucbs <= alpha  # shape (G,), dtype bool

    # Return the infimum lambda satisfying the condition.
    return safe_inf_search(lambda_grid, condition)


# ---------------------------------------------------------------------------
# Decision Rule 3: CBQ-HPD (our proposed method)
# ---------------------------------------------------------------------------


def find_lambda_cbq_hpd(
    losses_per_lambda: np.ndarray,
    lambda_grid: np.ndarray,
    B: float,
    alpha: float,
    beta: float,
    num_samples: int,
    rng: np.random.Generator,
) -> float:
    """Find λ_hpd^β via the CBQ highest posterior density criterion.

    Implements the paper's proposed decision rule from Section 5 and
    Corollary 4.4 (equation 29):

        λ_hpd^β = inf{λ : Pr(L⁺(λ) ≤ α | ℓ₁:ₙ) ≥ β}

    For each λ in the grid, this function estimates Pr(L⁺ ≤ α) via Monte
    Carlo simulation of Dirichlet random variates (1000 samples per
    config.yaml: cbq.n_mc_samples). The decision rule selects the smallest
    λ where this probability is at least β = 0.95.

    This is strictly more conservative than CRC (which uses only E[L⁺])
    because it requires the β-quantile of L⁺ to be below α, not just the
    mean. This explains the near-zero failure rates in Tables 1–3 while
    maintaining smaller prediction sets than RCPS.

    Relationship to CRC (Section 4.6):
        CRC corresponds to β → 0.5 (median of L⁺ ≤ α), while our method
        uses β = 0.95 (95th percentile of L⁺ ≤ α). The CRC criterion
        E[L⁺] ≤ α is recovered by taking the expectation rather than a
        quantile of L⁺.

    Performance note:
        This function calls ``cbq_core.prob_L_plus_leq_alpha`` once per
        grid point (G calls total). Each call draws ``num_samples`` Dirichlet
        samples. With G = 500 and num_samples = 1000, this is 500,000
        Dirichlet samples per trial. Parallelism across the M = 10,000
        trials (via joblib in the experiment files) makes this tractable.

    RNG state:
        The same ``rng`` object is passed for all G lambda evaluations.
        Since ``rng`` is stateful, each call to ``prob_L_plus_leq_alpha``
        advances the RNG state. This is correct — samples remain i.i.d.
        across calls. The per-trial RNG is created via
        ``utils.make_trial_rng(base_seed, trial_idx)``.

    Args:
        losses_per_lambda: Pre-computed losses of shape (G, n_cal).
            Produced by ``compute_losses_grid``. Values in [0, B].
        lambda_grid: 1-D array of lambda values sorted ascending, shape (G,).
        B: Upper bound on individual losses. Set to 1.0 for all experiments
            per config.yaml (exp1_synthetic_binomial.B = 1.0,
            exp2_synthetic_heteroskedastic.B = 1.0, exp3_mscoco.B = 1.0).
        alpha: Target risk level. Per config.yaml:
            - exp1_synthetic_binomial.alpha = 0.4
            - exp2_synthetic_heteroskedastic.alpha = 0.1
            - exp3_mscoco.alpha = 0.1
        beta: Confidence level for the HPD criterion. Per config.yaml:
            - exp1_synthetic_binomial.beta = 0.95
            - exp2_synthetic_heteroskedastic.beta = 0.95
            - exp3_mscoco.beta = 0.95
            The method requires Pr(L⁺ ≤ α) ≥ β to select a lambda.
        num_samples: Number of Monte Carlo Dirichlet samples per lambda
            evaluation. Per config.yaml (cbq.n_mc_samples = 1000).
            Higher values reduce Monte Carlo variance but increase runtime.
        rng: Per-trial NumPy Generator created via
            ``utils.make_trial_rng(base_seed, trial_idx)``. Shared across
            all G lambda evaluations within a single trial.

    Returns:
        The smallest λ in ``lambda_grid`` where Pr(L⁺ ≤ α) ≥ β, or
        ``np.inf`` if no such λ exists (method cannot achieve the target
        confidence level with the given calibration data).

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> # n_cal=10, all losses=0: L+ = U_{n+1}*B ~ Beta(1,10), mean=1/11≈0.09
        >>> # Pr(L+ ≤ 0.4) should be very high → first lambda satisfies criterion
        >>> losses_zero = np.zeros((3, 10))
        >>> grid = np.array([0.0, 0.5, 1.0])
        >>> result = find_lambda_cbq_hpd(
        ...     losses_zero, grid, B=1.0, alpha=0.4, beta=0.95,
        ...     num_samples=10000, rng=rng
        ... )
        >>> result == 0.0  # First lambda satisfies Pr(L+ ≤ 0.4) ≥ 0.95
        True
        >>> # All losses = B = 1.0: L+ = 1.0 deterministically, Pr(L+ ≤ 0.4) = 0
        >>> rng2 = np.random.default_rng(0)
        >>> losses_max = np.ones((3, 10))
        >>> find_lambda_cbq_hpd(
        ...     losses_max, grid, B=1.0, alpha=0.4, beta=0.95,
        ...     num_samples=1000, rng=rng2
        ... )
        inf
    """
    # Number of grid points.
    G: int = len(lambda_grid)

    # Build the boolean condition array by iterating over all G lambda values.
    # For each lambda index j, estimate Pr(L⁺(λⱼ) ≤ α) via Monte Carlo and
    # check if it meets the β threshold.
    #
    # We use a Python list comprehension here because each call to
    # prob_L_plus_leq_alpha is independent and advances the shared rng state.
    # Vectorization across lambda is not straightforward since each lambda
    # produces a different loss vector (different row of losses_per_lambda).
    condition: np.ndarray = np.array(
        [
            cbq_core.prob_L_plus_leq_alpha(
                losses=losses_per_lambda[j],  # shape (n_cal,)
                B=B,
                alpha=alpha,
                num_samples=num_samples,
                rng=rng,
            )
            >= beta
            for j in range(G)
        ],
        dtype=bool,
    )  # shape (G,)

    # Return the infimum lambda satisfying Pr(L⁺ ≤ α) ≥ β.
    return safe_inf_search(lambda_grid, condition)
