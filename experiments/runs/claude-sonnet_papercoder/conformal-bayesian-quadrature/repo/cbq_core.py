## cbq_core.py
"""Core Conformal Bayesian Quadrature algorithm from Section 4 of the paper.

This module implements the mathematical machinery of Theorem 4.3 and its
supporting results. All functions are stateless and operate on NumPy arrays.
The file has zero dependencies on other project files — it only imports numpy.

Key mathematical objects implemented:
  - Dir(1,...,1) sampling via Gamma normalization (Lemma 4.2)
  - L⁺ = Σᵢ Uᵢ · ℓ₍ᵢ₎ Monte Carlo samples (Theorem 4.3)
  - Pr(L⁺ ≤ α) estimation for the CBQ-HPD decision rule (Corollary 4.4)
  - E[L⁺] = (1/(n+1)) · (Σ ℓᵢ + B) analytical formula (Section 4.6 / CRC recovery)

References:
    Theorem 4.3: L⁺ stochastically dominates the posterior expected loss.
    Lemma 4.2: Quantile spacings follow Dir(1,...,1).
    Section 4.6: E[L⁺] recovers the Conformal Risk Control criterion.
    Corollary 4.4: b*_β = inf{b : Pr(L⁺ ≤ b) ≥ β} gives the HPD threshold.
"""

import numpy as np


def sample_dirichlet_spacings(
    n: int,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from Dir(1, 1, ..., 1) with n+1 components.

    Implements Lemma 4.2: the quantile spacings (U₁, ..., Uₙ₊₁) between
    consecutive order statistics of n i.i.d. uniform samples follow a
    symmetric Dirichlet distribution Dir(1, ..., 1) with n+1 components.

    The sampling uses the Gamma normalization trick: if Gᵢ ~ Gamma(αᵢ, 1)
    independently, then (G₁/S, ..., Gₖ/S) ~ Dir(α₁, ..., αₖ) where
    S = Σ Gᵢ. For Dir(1,...,1), each Gᵢ ~ Gamma(1,1) = Exponential(1).

    Args:
        n: Number of calibration samples. The Dirichlet vector has n+1
            components corresponding to the n+1 spacings between the n
            order statistics and the boundary points 0 and 1.
        num_samples: Number of independent Dirichlet vectors to draw.
            Use config.cbq.n_mc_samples = 1000 for decision rules and
            config.cbq.n_mc_figure = 100000 for Figure 4 density plots.
        rng: NumPy Generator for reproducible sampling. Should be a
            trial-specific RNG created via utils.make_trial_rng to ensure
            independence across parallel trials.

    Returns:
        Array of shape (num_samples, n+1) where each row is an independent
        sample from Dir(1, ..., 1) with n+1 components. Each row sums to
        exactly 1.0 and all entries are non-negative.

    Example:
        >>> rng = np.random.default_rng(42)
        >>> U = sample_dirichlet_spacings(n=3, num_samples=5, rng=rng)
        >>> U.shape
        (5, 4)
        >>> np.allclose(U.sum(axis=1), 1.0)
        True
        >>> np.all(U >= 0)
        True
        >>> # Column means should be approximately 1/(n+1) = 0.25
        >>> np.abs(U.mean(axis=0) - 0.25).max() < 0.05  # with 5 samples
        True
    """
    # Draw (num_samples, n+1) independent Exponential(1) = Gamma(1,1) samples.
    # This is the standard Gamma normalization trick for Dirichlet sampling.
    raw: np.ndarray = rng.exponential(scale=1.0, size=(num_samples, n + 1))

    # Normalize each row by its sum to project onto the (n+1)-simplex.
    # keepdims=True ensures broadcasting works correctly: shape (num_samples, 1).
    row_sums: np.ndarray = raw.sum(axis=1, keepdims=True)
    spacings: np.ndarray = raw / row_sums

    return spacings


def compute_L_plus_samples(
    losses: np.ndarray,
    B: float,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate Monte Carlo samples of L⁺ = Σᵢ Uᵢ · ℓ₍ᵢ₎ (Theorem 4.3).

    Implements the L⁺ random variable from Theorem 4.3 of the paper:

        U₁, ..., Uₙ₊₁ ~ Dir(1, ..., 1)
        L⁺ = Σᵢ₌₁ⁿ⁺¹ Uᵢ · ℓ₍ᵢ₎

    where ℓ₍₁₎ ≤ ... ≤ ℓ₍ₙ₎ are the order statistics of the calibration
    losses and ℓ₍ₙ₊₁₎ = B is the upper bound appended as the (n+1)-th value.

    Theorem 4.3 guarantees that L⁺ stochastically dominates the posterior
    expected loss for any prior π:

        inf_π Pr(L ≤ b | ℓ₁:ₙ) ≥ Pr(L⁺ ≤ b)   for any b ∈ (-∞, B]

    Args:
        losses: Per-sample calibration losses for a fixed λ, shape (n,).
            Values should be in [0, B] by assumption. These are the outputs
            of the loss function ℓ(zᵢ, λ) for i = 1, ..., n.
        B: Upper bound on individual losses. Set to 1.0 for all experiments
            per config.yaml (exp1_synthetic_binomial.B, exp2_synthetic_
            heteroskedastic.B, exp3_mscoco.B). The appended ℓ₍ₙ₊₁₎ = B
            represents the worst-case unobserved future loss.
        num_samples: Number of L⁺ samples to generate. Use
            config.cbq.n_mc_samples = 1000 for decision rules (fast) and
            config.cbq.n_mc_figure = 100000 for Figure 4 density plots
            (high-quality visualization).
        rng: NumPy Generator for reproducible sampling.

    Returns:
        Array of shape (num_samples,) containing independent Monte Carlo
        samples of L⁺. Each sample is a weighted sum of the extended order
        statistics with Dirichlet-distributed weights.

    Example:
        >>> rng = np.random.default_rng(42)
        >>> losses = np.array([0.2, 0.5, 0.8])
        >>> L_plus = compute_L_plus_samples(losses, B=1.0, num_samples=10000, rng=rng)
        >>> L_plus.shape
        (10000,)
        >>> # E[L+] = (0.2 + 0.5 + 0.8 + 1.0) / 4 = 0.625
        >>> abs(L_plus.mean() - 0.625) < 0.02
        True
        >>> # L+ is bounded in [min(losses), B] = [0.2, 1.0]
        >>> L_plus.min() >= 0.2 - 1e-10
        True
        >>> L_plus.max() <= 1.0 + 1e-10
        True
    """
    n: int = len(losses)

    # Step 1: Sort losses to get order statistics ℓ₍₁₎ ≤ ... ≤ ℓ₍ₙ₎.
    # Shape: (n,)
    ell_sorted: np.ndarray = np.sort(losses)

    # Step 2: Append B as the (n+1)-th order statistic ℓ₍ₙ₊₁₎ = B.
    # This represents the worst-case unobserved future loss (Theorem 4.3).
    # Shape: (n+1,)
    ell_extended: np.ndarray = np.append(ell_sorted, B)

    # Step 3: Sample Dirichlet spacings U ~ Dir(1,...,1) with n+1 components.
    # Shape: (num_samples, n+1)
    U: np.ndarray = sample_dirichlet_spacings(n, num_samples, rng)

    # Step 4: Compute L⁺ = U @ ell_extended via matrix-vector product.
    # Each row of U is dotted with ell_extended, giving one sample of L⁺.
    # Equivalent to np.sum(U * ell_extended[np.newaxis, :], axis=1) but
    # the matrix form is more concise and equally efficient.
    # Shape: (num_samples,)
    L_plus: np.ndarray = U @ ell_extended

    return L_plus


def prob_L_plus_leq_alpha(
    losses: np.ndarray,
    B: float,
    alpha: float,
    num_samples: int,
    rng: np.random.Generator,
) -> float:
    """Estimate Pr(L⁺ ≤ α) via Monte Carlo (Corollary 4.4).

    This is the core quantity used by the CBQ-HPD decision rule. The
    decision rule λ_hpd^β (Section 5 of the paper) finds the infimum λ
    where this probability is ≥ β:

        λ_hpd^β = inf{λ : Pr(L⁺ ≤ α | ℓ₁:ₙ) ≥ β}

    This function computes the left-hand side for a fixed λ (which
    determines the calibration losses passed in).

    Monotonicity property: For monotone non-increasing loss functions,
    as λ increases the losses ℓᵢ(λ) decrease, so the distribution of L⁺
    shifts left and Pr(L⁺ ≤ α) increases. This ensures the infimum search
    in decision_rules.find_lambda_cbq_hpd is well-defined.

    Variance of the estimator: With num_samples=1000, the standard error
    of the Monte Carlo estimate is at most sqrt(0.25/1000) ≈ 0.016, which
    is acceptable for the β=0.95 threshold.

    Args:
        losses: Per-sample calibration losses for a fixed λ, shape (n,).
            Values should be in [0, B].
        B: Upper bound on individual losses (1.0 for all experiments per
            config.yaml).
        alpha: Target risk level. The probability that L⁺ does not exceed
            this threshold is what we estimate. Values per config.yaml:
            0.4 for exp1_synthetic_binomial, 0.1 for exp2_synthetic_
            heteroskedastic and exp3_mscoco.
        num_samples: Number of Monte Carlo samples. Use
            config.cbq.n_mc_samples = 1000 per config.yaml.
        rng: NumPy Generator for reproducible sampling.

    Returns:
        Scalar float in [0, 1] estimating Pr(L⁺ ≤ α). This is the
        empirical CDF of L⁺ evaluated at α. When this value is ≥ β = 0.95,
        the method has sufficient confidence that the expected loss is
        controlled at level α.

    Example:
        >>> rng = np.random.default_rng(42)
        >>> # All losses = 0, B = 1.0, alpha = 0.4
        >>> # L+ = U_{n+1} * 1.0 ~ Beta(1, n), mean = 1/(n+1)
        >>> # For n=10, mean = 1/11 ≈ 0.091, so Pr(L+ <= 0.4) should be high
        >>> losses = np.zeros(10)
        >>> p = prob_L_plus_leq_alpha(losses, B=1.0, alpha=0.4, num_samples=10000, rng=rng)
        >>> p > 0.9  # Most L+ samples should be well below 0.4
        True
        >>> # All losses = B = 1.0: L+ = 1.0 deterministically, Pr(L+ <= 0.4) = 0
        >>> losses_max = np.ones(10)
        >>> p_max = prob_L_plus_leq_alpha(losses_max, B=1.0, alpha=0.4, num_samples=1000, rng=rng)
        >>> p_max == 0.0
        True
    """
    # Generate Monte Carlo samples of L⁺.
    # Shape: (num_samples,)
    L_plus_samples: np.ndarray = compute_L_plus_samples(
        losses=losses,
        B=B,
        num_samples=num_samples,
        rng=rng,
    )

    # Estimate Pr(L⁺ ≤ α) as the fraction of samples not exceeding α.
    # np.mean of a boolean array gives the proportion of True values.
    probability: float = float(np.mean(L_plus_samples <= alpha))

    return probability


def expected_L_plus(losses: np.ndarray, B: float) -> float:
    """Compute E[L⁺] analytically — the Conformal Risk Control criterion.

    Implements the analytical expectation from Section 4.6 of the paper.
    Since E[Uᵢ] = 1/(n+1) for all i under Dir(1,...,1):

        E[L⁺] = Σᵢ₌₁ⁿ⁺¹ E[Uᵢ] · ℓ₍ᵢ₎
               = (1/(n+1)) · Σᵢ₌₁ⁿ⁺¹ ℓ₍ᵢ₎
               = (1/(n+1)) · (Σᵢ₌₁ⁿ ℓᵢ + B)

    Note that the sum of order statistics equals the sum of the original
    values, so sorting is unnecessary here. This is a useful efficiency
    gain since find_lambda_crc calls this for every λ in the grid.

    Connection to CRC (Section 4.6): The Conformal Risk Control decision
    rule (Proposition 3.2) finds the infimum λ where E[L⁺] ≤ α, which is
    exactly the CRC criterion from Angelopoulos et al. (2024):

        λ_crc = inf{λ : (1/(n+1)) · (Σᵢ ℓᵢ(λ) + B) ≤ α}

    This function computes the left-hand side for a fixed λ.

    Args:
        losses: Per-sample calibration losses for a fixed λ, shape (n,).
            Values should be in [0, B]. No sorting is required.
        B: Upper bound on individual losses (1.0 for all experiments per
            config.yaml). Represents the (n+1)-th order statistic ℓ₍ₙ₊₁₎.

    Returns:
        Scalar float equal to (sum(losses) + B) / (n + 1). This is the
        analytical mean of L⁺ and equals the CRC empirical risk criterion.

    Example:
        >>> # n=1, losses=[0.5], B=1.0: E[L+] = (0.5 + 1.0) / 2 = 0.75
        >>> expected_L_plus(np.array([0.5]), B=1.0)
        0.75
        >>> # n=2, losses=[0.3, 0.7], B=1.0: E[L+] = (0.3 + 0.7 + 1.0) / 3 = 2/3
        >>> abs(expected_L_plus(np.array([0.3, 0.7]), B=1.0) - 2.0/3) < 1e-10
        True
        >>> # n=10, all losses=0, B=1.0: E[L+] = 1.0 / 11 ≈ 0.0909
        >>> abs(expected_L_plus(np.zeros(10), B=1.0) - 1.0/11) < 1e-10
        True
        >>> # Consistency with Monte Carlo: mean of L+ samples ≈ expected_L_plus
        >>> rng = np.random.default_rng(0)
        >>> losses = np.array([0.1, 0.4, 0.6])
        >>> mc_mean = compute_L_plus_samples(losses, B=1.0, num_samples=100000, rng=rng).mean()
        >>> analytical = expected_L_plus(losses, B=1.0)
        >>> abs(mc_mean - analytical) < 0.01
        True
    """
    n: int = len(losses)

    # E[L⁺] = (Σᵢ₌₁ⁿ ℓᵢ + B) / (n + 1)
    # No sorting needed: sum is order-invariant.
    expectation: float = float((np.sum(losses) + B) / (n + 1))

    return expectation
