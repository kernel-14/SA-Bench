import numpy as np
from typing import Tuple, List, Optional
from mdp import TabularMDP


def project_onto_simplex(v: np.ndarray) -> np.ndarray:
    """Project vector v onto the probability simplex using Euclidean projection.
    Algorithm from Duchi et al. 2008 (efficient projection onto L1 ball).
    v: vector of shape (D,)
    Returns: projected vector of shape (D,)
    """
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.searchsorted((u * np.arange(1, len(u) + 1) > (cssv - 1.0)), True)
    if rho == 0:
        theta = 0.0
    else:
        theta = (cssv[rho - 1] - 1.0) / rho
    return np.maximum(v - theta, 0.0)


def project_policy(pi: np.ndarray) -> np.ndarray:
    """Project each row of pi onto the probability simplex (i.e., enforce pi(s,:) in Delta(A)).
    pi: (S, A) policy matrix.
    Returns: (S, A) projected policy.
    """
    S, A = pi.shape
    proj = np.zeros_like(pi)
    for s in range(S):
        proj[s, :] = project_onto_simplex(pi[s, :])
    return proj


def policy_gradient_update(
    mdp: TabularMDP,
    pi: np.ndarray,
    eta: float,
) -> np.ndarray:
    """Single step of projected policy gradient.
    pi_{k+1} = Proj_Pi[pi_k + eta * grad_rho(pi_k)]

    grad_rho(s,a) = d^{pi}(s) * Q^{pi}(s,a)
    """
    d = mdp.stationary_distribution(pi)
    Q = mdp.compute_Q(pi)
    grad = d[:, np.newaxis] * Q
    pi_new = pi + eta * grad
    pi_new = project_policy(pi_new)
    return pi_new


def run_ppg(
    mdp: TabularMDP,
    pi0: np.ndarray,
    eta: float,
    num_iters: int,
    track_every: int = 1,
) -> Tuple[List[float], List[np.ndarray], List[float]]:
    """Run projected policy gradient for average reward MDP.

    Args:
        mdp: Tabular MDP instance
        pi0: Initial policy (S, A), rows sum to 1
        eta: Step size, should be < 1/L2^Pi
        num_iters: Number of PG iterations
        track_every: Track metrics every N iterations

    Returns:
        rho_history: Average reward at each tracked iteration
        pi_history: Policy at each tracked iteration
        opt_gap_history: Optimality gap (rho* - rho_k) at each tracked iter
    """
    S, A = mdp.S, mdp.A
    pi = pi0.copy()
    rho_history = []
    pi_history = []
    opt_gap_history = []

    rho_initial = mdp.average_reward(pi)
    rho_history.append(rho_initial)
    pi_history.append(pi.copy())
    opt_gap_history.append(None)

    for k in range(num_iters):
        pi = policy_gradient_update(mdp, pi, eta)
        if (k + 1) % track_every == 0:
            rho = mdp.average_reward(pi)
            rho_history.append(rho)
            pi_history.append(pi.copy())
            opt_gap_history.append(None)

    return rho_history, pi_history, opt_gap_history


def compute_exact_optimal_policy(mdp: TabularMDP, n_restarts: int = 20) -> Tuple[np.ndarray, float]:
    """Compute empirical optimal policy via random search + PPG refinement.
    Not guaranteed to find exact optimum but works for small MDPs.
    """
    S, A = mdp.S, mdp.A
    best_rho = -float('inf')
    best_pi = None
    rng = np.random.RandomState(42)

    for _ in range(n_restarts):
        pi = rng.rand(S, A)
        pi = pi / pi.sum(axis=1, keepdims=True)
        rho_hist, pi_hist, _ = run_ppg(mdp, pi, eta=0.01, num_iters=500, track_every=50)
        rho = rho_hist[-1]
        if rho > best_rho:
            best_rho = rho
            best_pi = pi_hist[-1].copy()

    return best_pi, best_rho


def set_opt_gaps(mdp: TabularMDP, rho_star: float, opt_gap_history: List[Optional[float]],
                  rho_history: List[float]):
    """Fill in the None entries in opt_gap_history."""
    for i in range(len(rho_history)):
        opt_gap_history[i] = rho_star - rho_history[i]
