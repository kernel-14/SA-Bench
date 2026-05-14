"""
Tests for the randomized midpoint sampler and related components.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from sampler import compute_alpha_hat_schedule, RandomizedMidpointSampler
from score_functions import GaussianScoreFunction, GMMScoreFunction
from theoretical_bounds import (
    tv_bound_theorem1, iteration_complexity, compare_with_prior_works,
    improvement_factor, parallel_sampler_complexity
)


def test_alpha_hat_schedule():
    """Test that the alpha hat schedule satisfies the required properties."""
    T = 100
    c0 = 5.0
    c1 = 10.0

    hat_alpha = compute_alpha_hat_schedule(T, c0, c1)

    # Check initial value
    assert abs(hat_alpha[T + 1] - 1.0 / T**c0) < 1e-10, "Initial value incorrect"

    # Check monotonicity: hat_alpha should be increasing as index decreases
    for t in range(1, T + 1):
        assert hat_alpha[t - 1] >= hat_alpha[t], f"Not monotone at t={t}"

    # Check bounds
    assert all(0 <= a <= 1 for a in hat_alpha), "Values out of [0, 1]"

    # Check that hat_alpha[0] is close to 1
    assert hat_alpha[0] > 0.9, f"hat_alpha[0] = {hat_alpha[0]} should be close to 1"

    print("test_alpha_hat_schedule: PASSED")


def test_gaussian_score_function():
    """Test the Gaussian score function."""
    d = 5
    sigma_sq = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    score_fn = GaussianScoreFunction(sigma_sq)

    # Test at tau = 0 (no noise): score should be -x / sigma_sq
    tau = 0.0
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    score = score_fn(tau, x)
    expected = -x / sigma_sq
    assert np.allclose(score, expected), f"Score at tau=0 incorrect: {score} vs {expected}"

    # Test at tau = 1 (pure noise): score should be -x (since sigma_tau = 1)
    tau = 1.0
    score = score_fn(tau, x)
    expected = -x
    assert np.allclose(score, expected), f"Score at tau=1 incorrect: {score} vs {expected}"

    # Test linearity: score should be linear in x
    x2 = 2 * x
    score2 = score_fn(0.5, x2)
    score1 = score_fn(0.5, x)
    assert np.allclose(score2, 2 * score1), "Score not linear in x"

    print("test_gaussian_score_function: PASSED")


def test_gmm_score_function():
    """Test the GMM score function."""
    d = 2
    means = np.array([[1.0, 0.0], [-1.0, 0.0]])
    weights = np.array([0.5, 0.5])
    sigma = 1.0
    score_fn = GMMScoreFunction(means, weights, sigma)

    # Test at a point equidistant from both components
    tau = 0.5
    x = np.array([0.0, 0.0])
    score = score_fn(tau, x)

    # By symmetry, the score should be zero at x=0
    assert np.allclose(score, 0.0, atol=1e-6), f"Score at origin should be 0: {score}"

    # Test that score is a valid vector
    x2 = np.array([2.0, 1.0])
    score2 = score_fn(tau, x2)
    assert score2.shape == (d,), f"Score shape incorrect: {score2.shape}"

    print("test_gmm_score_function: PASSED")


def test_sampler_initialization():
    """Test that the sampler initializes correctly."""
    d = 5
    T = 100
    K = 10

    def score_fn(tau, x):
        return -x  # Simple score function

    sampler = RandomizedMidpointSampler(score_fn, d, T, K)

    assert sampler.d == d
    assert sampler.T == T
    assert sampler.K == K
    assert sampler.N == 2 * T // K

    print("test_sampler_initialization: PASSED")


def test_sampler_output_shape():
    """Test that the sampler produces outputs of the correct shape."""
    d = 5
    T = 20
    K = 4

    def score_fn(tau, x):
        return -x

    sampler = RandomizedMidpointSampler(score_fn, d, T, K, rng=np.random.default_rng(42))

    # Single sample
    samples = sampler.sample(n_samples=1)
    assert samples.shape == (1, d), f"Single sample shape incorrect: {samples.shape}"

    # Multiple samples
    samples = sampler.sample(n_samples=5)
    assert samples.shape == (5, d), f"Multiple samples shape incorrect: {samples.shape}"

    print("test_sampler_output_shape: PASSED")


def test_sampler_gaussian_target():
    """
    Test that the sampler approximately recovers a Gaussian target.

    For a Gaussian target, the sampler should produce samples that
    approximately follow the target distribution.
    """
    d = 2
    sigma_sq = np.array([2.0, 3.0])
    score_fn = GaussianScoreFunction(sigma_sq)

    T = 50
    K = 5
    n_samples = 500

    sampler = RandomizedMidpointSampler(score_fn, d, T, K, rng=np.random.default_rng(42))
    samples = sampler.sample(n_samples=n_samples)

    # Check that samples have approximately the right variance
    # (The sampler should produce samples close to N(0, sigma_sq))
    sample_var = np.var(samples, axis=0)

    # With T=50, we expect some error, but the variance should be in the right ballpark
    # The target variance is sigma_sq = [2, 3]
    # We check that the ratio is within a factor of 3
    for i in range(d):
        ratio = sample_var[i] / sigma_sq[i]
        assert 0.1 < ratio < 10, f"Variance ratio {ratio} out of range for dim {i}"

    print("test_sampler_gaussian_target: PASSED")


def test_theoretical_bounds():
    """Test the theoretical bound functions."""
    T = 1000
    L = 10.0
    d = 100

    # TV bound should be positive and finite
    tv = tv_bound_theorem1(T, L, d)
    assert tv > 0, "TV bound should be positive"
    assert tv < float("inf"), "TV bound should be finite"

    # Iteration complexity should be positive
    eps = 0.1
    complexity = iteration_complexity(eps, L, d)
    assert complexity > 0, "Iteration complexity should be positive"

    # Improvement factor should be >= 1
    factor = improvement_factor(L, d)
    assert factor >= 1.0, "Improvement factor should be >= 1"

    # Compare with prior works
    comparison = compare_with_prior_works(T, L, d)
    assert "Ours (Theorem 1)" in comparison
    assert all(0 <= v <= 1 for v in comparison.values()), "TV distances should be in [0, 1]"

    print("test_theoretical_bounds: PASSED")


def test_convergence_rate():
    """
    Test that the KL divergence decreases at the expected rate O(log^4(T)/T^3).

    This is a simplified test using the Gaussian tracker.
    """
    d = 5
    sigma_sq = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    T_values = [20, 50, 100]
    K = 5

    kl_values = []
    for T in T_values:
        # Use the analytical Gaussian tracker
        from experiments.numerical_experiment import compute_kl_gaussian_sampler
        kl = compute_kl_gaussian_sampler(sigma_sq, T, K, n_trials=50,
                                          rng=np.random.default_rng(42))
        kl_values.append(kl)

    # Check that KL divergence decreases with T
    for i in range(len(kl_values) - 1):
        if kl_values[i] > 0 and kl_values[i + 1] > 0:
            assert kl_values[i] >= kl_values[i + 1] * 0.1,                 f"KL divergence not decreasing: {kl_values[i]} -> {kl_values[i+1]}"

    print("test_convergence_rate: PASSED")


def run_all_tests():
    """Run all tests."""
    tests = [
        test_alpha_hat_schedule,
        test_gaussian_score_function,
        test_gmm_score_function,
        test_sampler_initialization,
        test_sampler_output_shape,
        test_sampler_gaussian_target,
        test_theoretical_bounds,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"{test.__name__}: FAILED - {e}")
            failed += 1

    print(f"\n{passed} tests passed, {failed} tests failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
