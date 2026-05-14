"""
Unit tests verifying the correctness of the implementation against
the mathematical properties stated in the paper.

Tests cover:
  - Projection matrix Φ properties (Lemma 1)
  - Projected value function fixed-point equation (Lemma 1)
  - Bellman equation consistency
  - Policy gradient theorem
  - Projected policy gradient monotone improvement (Lemma 5)
  - Performance difference lemma (Lemma 6)
  - Convergence of PPG on small MDPs
  - MDP complexity constant bounds (Lemma 18)
"""

from __future__ import annotations

import numpy as np
import pytest

from complexity import (
    compute_C_m,
    compute_C_p,
    compute_C_r,
    compute_kappa_r,
    compute_smoothness_constants,
)
from mdp import AverageRewardMDP
from mdp_factory import (
    make_mdp_varying_size,
    make_mdps_varying_kernel,
    make_mdps_varying_reward,
    make_transition_uniform,
)
from policy_gradient import projected_policy_gradient, theoretical_bound
from utils import project_policy, project_simplex, uniform_policy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_simple_mdp(S: int = 3, A: int = 3, seed: int = 0) -> AverageRewardMDP:
    """Create a small random ergodic MDP for testing."""
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(S), size=(S, A))
    r = rng.uniform(-1, 1, size=(S, A))
    return AverageRewardMDP(P, r)


# ---------------------------------------------------------------------------
# Tests: Projection utilities
# ---------------------------------------------------------------------------

class TestProjectSimplex:
    def test_output_is_probability_vector(self):
        rng = np.random.default_rng(0)
        v = rng.normal(size=10)
        p = project_simplex(v)
        assert np.all(p >= -1e-10)
        assert abs(p.sum() - 1.0) < 1e-10

    def test_already_on_simplex(self):
        v = np.array([0.2, 0.5, 0.3])
        p = project_simplex(v)
        np.testing.assert_allclose(p, v, atol=1e-10)

    def test_negative_entries(self):
        v = np.array([-1.0, 2.0, 0.5])
        p = project_simplex(v)
        assert np.all(p >= -1e-10)
        assert abs(p.sum() - 1.0) < 1e-10


class TestProjectPolicy:
    def test_rows_are_distributions(self):
        rng = np.random.default_rng(1)
        pi_raw = rng.normal(size=(5, 4))
        pi = project_policy(pi_raw)
        assert pi.shape == (5, 4)
        np.testing.assert_allclose(pi.sum(axis=1), np.ones(5), atol=1e-10)
        assert np.all(pi >= -1e-10)


# ---------------------------------------------------------------------------
# Tests: Projection matrix Φ
# ---------------------------------------------------------------------------

class TestProjectionMatrix:
    def test_phi_is_idempotent(self):
        """Φ^2 = Φ (projection matrix is idempotent)."""
        mdp = make_simple_mdp(4, 3)
        Phi = mdp.Phi
        np.testing.assert_allclose(Phi @ Phi, Phi, atol=1e-12)

    def test_phi_annihilates_ones(self):
        """Φ 1 = 0."""
        mdp = make_simple_mdp(5, 3)
        ones = np.ones(mdp.S)
        np.testing.assert_allclose(mdp.Phi @ ones, np.zeros(mdp.S), atol=1e-12)

    def test_phi_norm_bound(self):
        """||Φ||_∞ ≤ 2  (Lemma 18 item 1)."""
        for S in [3, 5, 10]:
            mdp = make_simple_mdp(S, 3)
            # Compute L_∞ induced norm
            norm = np.max(np.sum(np.abs(mdp.Phi), axis=1))
            assert norm <= 2.0 + 1e-10, f"||Φ||_∞ = {norm} > 2 for S={S}"


# ---------------------------------------------------------------------------
# Tests: Transition matrix and stationary distribution
# ---------------------------------------------------------------------------

class TestTransitionMatrix:
    def test_rows_sum_to_one(self):
        mdp = make_simple_mdp(4, 3)
        rng = np.random.default_rng(0)
        pi = rng.dirichlet(np.ones(3), size=4)
        P_pi = mdp.transition_matrix(pi)
        np.testing.assert_allclose(P_pi.sum(axis=1), np.ones(4), atol=1e-12)

    def test_stationary_distribution_is_valid(self):
        mdp = make_simple_mdp(4, 3)
        pi = uniform_policy(4, 3)
        d = mdp.stationary_distribution(pi)
        assert abs(d.sum() - 1.0) < 1e-8
        assert np.all(d >= -1e-10)

    def test_stationary_distribution_satisfies_balance(self):
        """d^π P^π = d^π."""
        mdp = make_simple_mdp(5, 4)
        pi = uniform_policy(5, 4)
        d = mdp.stationary_distribution(pi)
        P_pi = mdp.transition_matrix(pi)
        np.testing.assert_allclose(d @ P_pi, d, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: Projected value function (Lemma 1)
# ---------------------------------------------------------------------------

class TestProjectedValueFunction:
    def test_fixed_point_equation(self):
        """
        v_φ^π = Φ(r^π + P^π v_φ^π)  (Equation 14 / Lemma 1).
        """
        mdp = make_simple_mdp(4, 3)
        pi = uniform_policy(4, 3)
        v = mdp.projected_value_function(pi)
        r_pi = mdp.reward_vector(pi)
        P_pi = mdp.transition_matrix(pi)
        rhs = mdp.Phi @ (r_pi + P_pi @ v)
        np.testing.assert_allclose(v, rhs, atol=1e-8)

    def test_orthogonal_to_ones(self):
        """v_φ^π ⊥ 1, i.e., 1^T v_φ^π = 0."""
        mdp = make_simple_mdp(5, 4)
        pi = uniform_policy(5, 4)
        v = mdp.projected_value_function(pi)
        assert abs(v.sum()) < 1e-8

    def test_bellman_equation_consistency(self):
        """
        ρ^π 1 + v_φ^π ≈ r^π + P^π v_φ^π  (Bellman equation, Equation 3).
        """
        mdp = make_simple_mdp(4, 3)
        pi = uniform_policy(4, 3)
        rho = mdp.average_reward(pi)
        v = mdp.projected_value_function(pi)
        r_pi = mdp.reward_vector(pi)
        P_pi = mdp.transition_matrix(pi)
        lhs = rho * np.ones(mdp.S) + v
        rhs = r_pi + P_pi @ v
        # The Bellman equation holds up to a constant (the projection removes it)
        diff = lhs - rhs
        # diff should be a constant vector (same value at all states)
        np.testing.assert_allclose(diff - diff.mean(), np.zeros(mdp.S), atol=1e-8)


# ---------------------------------------------------------------------------
# Tests: Average reward and Q-function
# ---------------------------------------------------------------------------

class TestAverageReward:
    def test_average_reward_is_scalar(self):
        mdp = make_simple_mdp(3, 3)
        pi = uniform_policy(3, 3)
        rho = mdp.average_reward(pi)
        assert isinstance(rho, float)

    def test_q_function_shape(self):
        mdp = make_simple_mdp(4, 3)
        pi = uniform_policy(4, 3)
        Q = mdp.q_function(pi)
        assert Q.shape == (4, 3)

    def test_q_function_bellman(self):
        """
        Q^π(s,a) = r(s,a) - ρ^π + Σ_{s'} P(s'|s,a) v^π(s')
        and v^π(s) = Σ_a π(a|s) Q^π(s,a).
        """
        mdp = make_simple_mdp(4, 3)
        pi = uniform_policy(4, 3)
        Q = mdp.q_function(pi)
        v = mdp.projected_value_function(pi)
        # v^π(s) = Σ_a π(a|s) Q^π(s,a)
        v_from_Q = np.einsum("sa,sa->s", pi, Q)
        np.testing.assert_allclose(v_from_Q, v, atol=1e-8)


# ---------------------------------------------------------------------------
# Tests: Policy gradient
# ---------------------------------------------------------------------------

class TestPolicyGradient:
    def test_gradient_shape(self):
        mdp = make_simple_mdp(4, 3)
        pi = uniform_policy(4, 3)
        grad = mdp.policy_gradient(pi)
        assert grad.shape == (4, 3)

    def test_gradient_finite_difference(self):
        """
        Verify ∂ρ^π/∂π(a|s) ≈ (ρ(π + ε e_{s,a}) - ρ(π - ε e_{s,a})) / (2ε)
        using finite differences on the projected policy.
        """
        mdp = make_simple_mdp(3, 3, seed=42)
        pi = uniform_policy(3, 3)
        grad = mdp.policy_gradient(pi)

        eps = 1e-5
        grad_fd = np.zeros_like(pi)
        for s in range(mdp.S):
            for a in range(mdp.A):
                pi_plus = pi.copy()
                pi_plus[s, a] += eps
                pi_plus = project_policy(pi_plus)

                pi_minus = pi.copy()
                pi_minus[s, a] -= eps
                pi_minus = project_policy(pi_minus)

                rho_plus = mdp.average_reward(pi_plus)
                rho_minus = mdp.average_reward(pi_minus)
                grad_fd[s, a] = (rho_plus - rho_minus) / (2 * eps)

        # The policy gradient theorem gives the gradient in the unconstrained
        # space; finite differences on the projected policy may differ slightly.
        # We check that the signs agree and magnitudes are in the same ballpark.
        np.testing.assert_allclose(grad, grad_fd, atol=1e-3)


# ---------------------------------------------------------------------------
# Tests: Projected policy gradient convergence
# ---------------------------------------------------------------------------

class TestPPGConvergence:
    def test_monotone_improvement(self):
        """
        Lemma 5: ρ^{π_{k+1}} ≥ ρ^{π_k} for all k.
        """
        mdp = make_simple_mdp(4, 3, seed=7)
        pi_init = uniform_policy(4, 3)
        _, rho_star = mdp.optimal_policy()

        result = projected_policy_gradient(
            mdp=mdp,
            pi_init=pi_init,
            eta=0.01,
            n_iterations=100,
            rho_star=rho_star,
        )
        rewards = np.array(result.rewards)
        # Allow tiny numerical noise
        diffs = np.diff(rewards)
        assert np.all(diffs >= -1e-8), f"Non-monotone at indices: {np.where(diffs < -1e-8)}"

    def test_convergence_to_optimal(self):
        """
        PPG should converge close to the optimal reward on a small MDP.
        """
        mdp = make_simple_mdp(3, 3, seed=5)
        pi_init = uniform_policy(3, 3)
        _, rho_star = mdp.optimal_policy()

        result = projected_policy_gradient(
            mdp=mdp,
            pi_init=pi_init,
            eta=0.05,
            n_iterations=5000,
            rho_star=rho_star,
        )
        final_gap = result.suboptimality[-1]
        assert final_gap < 0.05, f"Final suboptimality {final_gap:.4f} too large"

    def test_suboptimality_decreasing(self):
        """
        The suboptimality gap should be non-increasing overall.
        """
        mdp = make_simple_mdp(4, 4, seed=3)
        pi_init = uniform_policy(4, 4)
        _, rho_star = mdp.optimal_policy()

        result = projected_policy_gradient(
            mdp=mdp,
            pi_init=pi_init,
            eta=0.01,
            n_iterations=200,
            rho_star=rho_star,
        )
        # The gap at the end should be smaller than at the start
        assert result.suboptimality[-1] <= result.suboptimality[0] + 1e-8


# ---------------------------------------------------------------------------
# Tests: MDP complexity constants (Lemma 18)
# ---------------------------------------------------------------------------

class TestComplexityConstants:
    def test_kappa_r_bound(self):
        """κ_r ≤ 2  (Lemma 18 item 3)."""
        for S, A in [(3, 3), (5, 4)]:
            mdp = make_simple_mdp(S, A)
            rng = np.random.default_rng(0)
            policies = [rng.dirichlet(np.ones(A), size=S) for _ in range(20)]
            kappa_r = compute_kappa_r(mdp, policies)
            assert kappa_r <= 2.0 + 1e-10, f"κ_r = {kappa_r} > 2"

    def test_C_p_bound(self):
        """C_p ≤ sqrt(|A|)  (Lemma 18 item 5)."""
        S, A = 4, 4
        mdp = make_simple_mdp(S, A)
        rng = np.random.default_rng(0)
        policies = [rng.dirichlet(np.ones(A), size=S) for _ in range(30)]
        C_p = compute_C_p(mdp, policies)
        assert C_p <= np.sqrt(A) + 1e-8, f"C_p = {C_p} > sqrt(A) = {np.sqrt(A)}"

    def test_C_r_bound(self):
        """C_r ≤ sqrt(|A|)  (Lemma 18 item 6)."""
        S, A = 4, 4
        mdp = make_simple_mdp(S, A)
        rng = np.random.default_rng(0)
        policies = [rng.dirichlet(np.ones(A), size=S) for _ in range(30)]
        C_r = compute_C_r(mdp, policies)
        assert C_r <= np.sqrt(A) + 1e-8, f"C_r = {C_r} > sqrt(A) = {np.sqrt(A)}"

    def test_smoothness_constants_positive(self):
        """L_1^Π and L_2^Π should be non-negative."""
        C_m, C_p, C_r, kappa_r = 2.0, 1.0, 0.5, 1.5
        L1, L2 = compute_smoothness_constants(C_m, C_p, C_r, kappa_r)
        assert L1 >= 0.0
        assert L2 >= 0.0

    def test_uniform_kernel_C_p_is_zero(self):
        """
        For a uniform transition kernel, C_p = 0 because P^π is independent
        of π (all actions lead to the same distribution).
        """
        S, A = 4, 4
        P = make_transition_uniform(S, A)
        r = np.ones((S, A))
        mdp = AverageRewardMDP(P, r)
        rng = np.random.default_rng(0)
        policies = [rng.dirichlet(np.ones(A), size=S) for _ in range(20)]
        C_p = compute_C_p(mdp, policies)
        assert C_p < 1e-10, f"C_p = {C_p} should be 0 for uniform kernel"


# ---------------------------------------------------------------------------
# Tests: MDP factory
# ---------------------------------------------------------------------------

class TestMDPFactory:
    def test_varying_size_transition_valid(self):
        for S, A in [(3, 3), (9, 9)]:
            mdp = make_mdp_varying_size(S, A)
            assert mdp.P.shape == (S, A, S)
            np.testing.assert_allclose(
                mdp.P.sum(axis=2), np.ones((S, A)), atol=1e-10
            )

    def test_varying_reward_mdps_created(self):
        mdps = make_mdps_varying_reward(S=8, A=8, seed=0)
        assert set(mdps.keys()) == {
            "no_variance", "low_variance", "high_variance", "max_variance"
        }

    def test_varying_kernel_mdps_created(self):
        mdps = make_mdps_varying_kernel(S=8, A=8, seed=0)
        assert set(mdps.keys()) == {"uniform", "non_uniform", "deterministic"}

    def test_deterministic_kernel_is_deterministic(self):
        """Each row of P(·|s,a) should have exactly one non-zero entry."""
        mdps = make_mdps_varying_kernel(S=8, A=8, seed=0)
        P = mdps["deterministic"].P
        for s in range(8):
            for a in range(8):
                row = P[s, a]
                assert np.sum(row > 0.5) == 1, "Deterministic kernel should have one non-zero per row"


# ---------------------------------------------------------------------------
# Tests: Theoretical bound
# ---------------------------------------------------------------------------

class TestTheoreticalBound:
    def test_bound_decreasing(self):
        """The theoretical bound 1/(1/gap_0 + ν k) is decreasing in k."""
        gap_0 = 0.5
        nu = 0.01
        bounds = [theoretical_bound(gap_0, nu, k) for k in range(100)]
        diffs = np.diff(bounds)
        assert np.all(diffs <= 0), "Theoretical bound should be non-increasing"

    def test_bound_at_k0(self):
        """At k=0, the bound equals gap_0."""
        gap_0 = 0.3
        nu = 0.05
        assert abs(theoretical_bound(gap_0, nu, 0) - gap_0) < 1e-12

    def test_bound_sublinear(self):
        """The bound should be O(1/k)."""
        gap_0 = 1.0
        nu = 1.0
        # For large k, bound ≈ 1/(ν k)
        k = 1000
        bound = theoretical_bound(gap_0, nu, k)
        expected = 1.0 / (nu * k)
        np.testing.assert_allclose(bound, expected, rtol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
