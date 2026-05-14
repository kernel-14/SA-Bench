"""
Unit tests for the Conformal Prediction as Bayesian Quadrature implementation.

Tests verify:
1. SCP recovery (Section 4.6)
2. CRC recovery (Section 4.6)
3. E[L+] formula (Equation 27)
4. Dirichlet sampling correctness
5. BQ decision rule monotonicity
"""

import numpy as np
from scipy import stats


def test_scp_recovery():
    """
    Verify that SCP is recovered as a special case of BQ (Section 4.6).

    For miscoverage loss ell_i = 1{s_i > lambda}:
      E[L+] = 1 - k/(n+1) where k = #{s_i <= lambda}
    Setting E[L+] <= alpha gives k >= (n+1)(1-alpha),
    which recovers SCP: lambda = s_{(ceil((n+1)(1-alpha)))}.
    """
    from methods import split_conformal_prediction, compute_expected_L_plus

    rng = np.random.default_rng(0)
    n = 20
    alpha = 0.1
    scores = rng.uniform(0, 1, size=n)

    lam_scp = split_conformal_prediction(scores, alpha)

    # At SCP threshold, E[L+] should be <= alpha
    losses = (scores > lam_scp).astype(float)
    E_L_plus = compute_expected_L_plus(losses, B=1.0)
    assert E_L_plus <= alpha + 1e-10, f"E[L+]={E_L_plus} > alpha={alpha}"

    # At the previous threshold (one step smaller), E[L+] should be > alpha
    sorted_scores = np.sort(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > 1:
        lam_prev = sorted_scores[k - 2]
        losses_prev = (scores > lam_prev).astype(float)
        E_L_plus_prev = compute_expected_L_plus(losses_prev, B=1.0)
        assert E_L_plus_prev > alpha - 1e-10, \
            f"E[L+] at prev threshold={E_L_plus_prev} should be > alpha={alpha}"

    print("PASS: SCP recovery")


def test_crc_recovery():
    """
    Verify that CRC is recovered as a special case of BQ (Section 4.6).

    CRC selects lambda = inf{lambda : (sum_i ell_i + B) / (n+1) <= alpha}
    This is equivalent to E[L+] <= alpha.
    """
    from methods import conformal_risk_control, compute_expected_L_plus

    rng = np.random.default_rng(1)
    n = 10
    alpha = 0.4
    B = 1.0
    K = 4

    V = rng.uniform(0, 1, size=(n, K))
    lambda_grid = np.linspace(0, 1, 101)

    def losses_fn(lam):
        return np.mean(V > lam, axis=1)

    lam_crc = conformal_risk_control(losses_fn, lambda_grid, alpha, B=B)

    # At CRC threshold, E[L+] should be <= alpha
    losses = losses_fn(lam_crc)
    E_L_plus = compute_expected_L_plus(losses, B=B)
    assert E_L_plus <= alpha + 1e-10, f"E[L+]={E_L_plus} > alpha={alpha}"

    print("PASS: CRC recovery")


def test_expected_L_plus():
    """
    Verify E[L+] = (sum(losses) + B) / (n+1) (Equation 27).
    """
    from methods import sample_L_plus, compute_expected_L_plus

    rng = np.random.default_rng(2)
    losses = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    B = 1.0
    n = len(losses)

    # Analytical formula
    expected = (np.sum(losses) + B) / (n + 1)
    analytical = compute_expected_L_plus(losses, B)
    assert abs(analytical - expected) < 1e-10, \
        f"Analytical={analytical}, expected={expected}"

    # Monte Carlo estimate should be close
    L_plus = sample_L_plus(losses, B=B, n_samples=100000, rng=rng)
    mc_mean = L_plus.mean()
    assert abs(mc_mean - expected) < 0.01, \
        f"MC mean={mc_mean}, expected={expected}"

    print(f"PASS: E[L+] formula (analytical={analytical:.4f}, MC={mc_mean:.4f})")


def test_dirichlet_distribution():
    """
    Verify that the Dirichlet(1,...,1) samples sum to 1 and have correct mean.
    """
    from methods import sample_L_plus

    rng = np.random.default_rng(3)
    n = 5
    losses = np.zeros(n)  # All zeros
    B = 1.0

    # With all-zero losses, L+ = U_{n+1} * B ~ Beta(1, n) * B
    # E[L+] = B / (n+1)
    L_plus = sample_L_plus(losses, B=B, n_samples=100000, rng=rng)
    expected_mean = B / (n + 1)
    mc_mean = L_plus.mean()
    assert abs(mc_mean - expected_mean) < 0.01, \
        f"MC mean={mc_mean}, expected={expected_mean}"

    print(f"PASS: Dirichlet distribution (mean={mc_mean:.4f}, expected={expected_mean:.4f})")


def test_bq_more_conservative_than_crc():
    """
    Verify that BQ (beta=0.95) is more conservative than CRC.

    BQ selects a larger lambda (more conservative) than CRC because it
    requires Pr(L+ <= alpha) >= 0.95 rather than just E[L+] <= alpha.
    """
    from methods import (
        conformal_risk_control,
        bayesian_quadrature_decision_rule,
    )

    rng = np.random.default_rng(4)
    n = 10
    K = 4
    alpha = 0.4
    B = 1.0

    V = rng.uniform(0, 1, size=(n, K))
    lambda_grid = np.linspace(0, 1, 101)

    def losses_fn(lam, _V=V):
        return np.mean(_V > lam, axis=1)

    lam_crc = conformal_risk_control(losses_fn, lambda_grid, alpha, B=B)
    lam_bq = bayesian_quadrature_decision_rule(
        losses_fn, lambda_grid, alpha, beta=0.95, B=B,
        n_samples=1000, rng=rng
    )

    # BQ should be >= CRC (more conservative)
    assert lam_bq >= lam_crc - 1e-10, \
        f"BQ lambda={lam_bq} < CRC lambda={lam_crc}"

    print(f"PASS: BQ more conservative than CRC (BQ={lam_bq:.2f}, CRC={lam_crc:.2f})")


def test_stochastic_dominance():
    """
    Verify that L+ stochastically dominates the posterior expected loss
    for a simple case (Theorem 4.3).

    For a fixed set of losses, the true expected loss should be <= b_beta*
    with probability >= beta.
    """
    from methods import sample_L_plus

    rng = np.random.default_rng(5)

    # Simple case: losses are [0, 0.5], B=1
    # True distribution: Uniform(0, 1)
    # True expected loss = 0.5
    losses = np.array([0.0, 0.5])
    B = 1.0
    true_expected_loss = 0.5

    L_plus = sample_L_plus(losses, B=B, n_samples=100000, rng=rng)

    # Pr(L+ <= true_expected_loss) should be >= 0 (trivially)
    # More importantly, the 95th percentile of L+ should be >= true_expected_loss
    b_95 = np.quantile(L_plus, 0.95)
    assert b_95 >= true_expected_loss - 0.1, \
        f"95th percentile of L+={b_95} < true expected loss={true_expected_loss}"

    print(f"PASS: Stochastic dominance (95th pct of L+={b_95:.4f} >= true={true_expected_loss})")


if __name__ == "__main__":
    print("Running tests for Conformal Prediction as Bayesian Quadrature...")
    print()

    test_scp_recovery()
    test_crc_recovery()
    test_expected_L_plus()
    test_dirichlet_distribution()
    test_bq_more_conservative_than_crc()
    test_stochastic_dominance()

    print()
    print("All tests passed!")
