import numpy as np
from typing import Dict
from mdp import TabularMDP


def compute_operator_norm_inf(matrix: np.ndarray) -> float:
    """Compute L_infinity operator norm: max_i sum_j |A_{ij}|."""
    return float(np.max(np.sum(np.abs(matrix), axis=1)))


def compute_Cm(mdp: TabularMDP) -> float:
    """Compute Cm = max_pi ||(I - Phi * P^pi)^{-1}||_inf.

    Bounded by 2 * Ce * |S| / (1 - lambda).
    We approximate by sampling policies.
    """
    S, A = mdp.S, mdp.A
    rng = np.random.RandomState(42)
    cm_max = 0.0
    cm_values = []
    for _ in range(100):
        pi = rng.rand(S, A)
        pi = pi / pi.sum(axis=1, keepdims=True)
        try:
            Phi = mdp.projection_matrix()
            P_pi = mdp.transition_matrix(pi)
            M_inv = np.linalg.inv(np.eye(S) - Phi @ P_pi)
            op_norm = compute_operator_norm_inf(M_inv)
            cm_values.append(op_norm)
        except np.linalg.LinAlgError:
            pass
    if cm_values:
        cm_max = max(cm_max, max(cm_values))
    return cm_max


def compute_Cp(mdp: TabularMDP, num_samples: int = 100) -> float:
    """Compute Cp = max_{pi, pi'} ||P^{pi'} - P^{pi}||_inf / ||pi' - pi||_2.

    Bounded by sqrt(A).
    """
    S, A = mdp.S, mdp.A
    cp_max = 0.0
    rng = np.random.RandomState(7)
    for _ in range(num_samples):
        pi1 = rng.rand(S, A)
        pi1 = pi1 / pi1.sum(axis=1, keepdims=True)
        pi2 = rng.rand(S, A)
        pi2 = pi2 / pi2.sum(axis=1, keepdims=True)
        diff = pi2 - pi1
        diff_norm = np.sqrt(np.sum(diff**2))
        if diff_norm < 1e-12:
            continue
        P1 = mdp.transition_matrix(pi1)
        P2 = mdp.transition_matrix(pi2)
        P_diff_norm = compute_operator_norm_inf(P2 - P1)
        cp_max = max(cp_max, P_diff_norm / diff_norm)
    return cp_max


def compute_Cr(mdp: TabularMDP, num_samples: int = 100) -> float:
    """Compute Cr = max_{pi, pi'} ||r^{pi'} - r^{pi}||_inf / ||pi' - pi||_2.

    Bounded by sqrt(A).
    """
    S, A = mdp.S, mdp.A
    cr_max = 0.0
    rng = np.random.RandomState(7)
    for _ in range(num_samples):
        pi1 = rng.rand(S, A)
        pi1 = pi1 / pi1.sum(axis=1, keepdims=True)
        pi2 = rng.rand(S, A)
        pi2 = pi2 / pi2.sum(axis=1, keepdims=True)
        diff = pi2 - pi1
        diff_norm = np.sqrt(np.sum(diff**2))
        if diff_norm < 1e-12:
            continue
        r1 = mdp.reward_vector(pi1)
        r2 = mdp.reward_vector(pi2)
        r_diff_norm = np.max(np.abs(r2 - r1))
        cr_max = max(cr_max, r_diff_norm / diff_norm)
    return cr_max


def compute_kappa_r(mdp: TabularMDP, num_samples: int = 100) -> float:
    """Compute kappa_r = max_pi ||Phi * r^pi||_inf. Bounded by 2."""
    S, A = mdp.S, mdp.A
    Phi = mdp.projection_matrix()
    kappa_max = 0.0
    rng = np.random.RandomState(7)
    for _ in range(num_samples):
        pi = rng.rand(S, A)
        pi = pi / pi.sum(axis=1, keepdims=True)
        r_pi = mdp.reward_vector(pi)
        norm_val = np.max(np.abs(Phi @ r_pi))
        kappa_max = max(kappa_max, norm_val)
    return kappa_max


def compute_L1_pi(Cm: float, Cp: float, Cr: float, kappa_r: float) -> float:
    """Compute restricted Lipschitz constant L1^Pi.

    L1^Pi = 2 * (Cr + Cp * Cm * kappa_r + 2 * (Cm^2 * Cp * kappa_r + Cm * Cr))
    """
    return 2.0 * (Cr + Cp * Cm * kappa_r + 2.0 * (Cm**2 * Cp * kappa_r + Cm * Cr))


def compute_L2_pi(Cm: float, Cp: float, Cr: float, kappa_r: float) -> float:
    """Compute restricted smoothness constant L2^Pi.

    L2^Pi = 4 * (Cp^2 * Cm^2 * kappa_r + Cp * Cm * Cr
                 + (Cp + 1) * (Cm^2 * Cp * kappa_r + Cm * Cr)
                 + 4 * (Cm^3 * Cp^2 * kappa_r + Cm^2 * Cp * Cr))
    """
    term1 = Cp**2 * Cm**2 * kappa_r
    term2 = Cp * Cm * Cr
    term3 = (Cp + 1) * (Cm**2 * Cp * kappa_r + Cm * Cr)
    term4 = 4.0 * (Cm**3 * Cp**2 * kappa_r + Cm**2 * Cp * Cr)
    return 4.0 * (term1 + term2 + term3 + term4)


def compute_CPL(mdp: TabularMDP, pi_star: np.ndarray, num_samples: int = 200) -> float:
    """Compute C_PL = max_{pi, s} d^{pi_star}(s) / d^{pi}(s)."""
    S, A = mdp.S, mdp.A
    d_star = mdp.stationary_distribution(pi_star)
    cpl_max = 0.0
    rng = np.random.RandomState(7)
    for _ in range(num_samples):
        pi = rng.rand(S, A)
        pi = pi / pi.sum(axis=1, keepdims=True)
        d_pi = mdp.stationary_distribution(pi)
        for s in range(S):
            if d_pi[s] > 1e-12:
                ratio = d_star[s] / d_pi[s]
                cpl_max = max(cpl_max, ratio)
    return cpl_max


def compute_all_constants(mdp: TabularMDP, pi_star: np.ndarray) -> Dict[str, float]:
    """Compute all MDP complexity constants."""
    Cm = compute_Cm(mdp)
    Cp = compute_Cp(mdp)
    Cr = compute_Cr(mdp)
    kappa = compute_kappa_r(mdp)
    L1 = compute_L1_pi(Cm, Cp, Cr, kappa)
    L2 = compute_L2_pi(Cm, Cp, Cr, kappa)
    CPL = compute_CPL(mdp, pi_star)
    return {
        'Cm': Cm,
        'Cp': Cp,
        'Cr': Cr,
        'kappa_r': kappa,
        'L1_Pi': L1,
        'L2_Pi': L2,
        'C_PL': CPL,
    }
