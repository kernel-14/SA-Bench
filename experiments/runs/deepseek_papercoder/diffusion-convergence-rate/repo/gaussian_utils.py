## gaussian_utils.py

"""
Exact closed‑form operations for the Gaussian setting used in the numerical
validation of the convergence rate (Appendix A). All functions are pure and
deterministic – they depend only on the input `tau` and the diagonal covariance
of the data distribution.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def compute_score_matrix(tau: float, sigma_diag: np.ndarray) -> np.ndarray:
    """
    Return the diagonal entries of the linear score matrix M_tau such that
        s_tau^*(x) = M_tau @ x   (here M_tau is diagonal).

    Args:
        tau: noise level in (0,1)
        sigma_diag: 1D array of variances for the target Gaussian data.

    Returns:
        1D array of length d containing the diagonal of -((1-tau)*Sigma + tau*I)^{-1}.
    """
    return -1.0 / ((1.0 - tau) * sigma_diag + tau)


def target_covariance(tau: float, sigma_diag: np.ndarray) -> np.ndarray:
    """
    Compute the d x d covariance matrix of the forward process X_tau.

    For X_0 ~ N(0, diag(sigma_diag)) and X_tau = sqrt(1-tau) X_0 + sqrt(tau) Z,
    the marginal distribution is N(0, (1-tau)*Sigma + tau*I_d).

    Args:
        tau: noise level in (0,1)
        sigma_diag: 1D array of length d.

    Returns:
        Diagonal covariance matrix of shape (d, d).
    """
    diag_vals = (1.0 - tau) * sigma_diag + tau
    return np.diag(diag_vals)


class GaussianKL:
    """
    Compute the Kullback–Leibler divergence between two zero‑mean Gaussian
    distributions with full covariance matrices A and B.
    """

    @staticmethod
    def compute(A: np.ndarray, B: np.ndarray) -> float:
        """
        KL divergence KL( N(0,A) || N(0,B) ).

        Formula:
            KL = 0.5 * ( tr(B^{-1} A) - d + ln(det B / det A) )

        Both A and B must be symmetric positive definite.

        Args:
            A: covariance matrix of the first distribution (d, d)
            B: covariance matrix of the second distribution (d, d)

        Returns:
            Non‑negative KL divergence (float).
        """
        d = A.shape[0]
        # Cholesky factor of B (B = L L^T)
        chol_B, lower = cho_factor(B, lower=True)

        # solve B * X = A  ->  X = B^{-1} A   using Cholesky factor
        # cho_solve expects (factor, lower) and returns solution to L L^T X = A
        X = cho_solve((chol_B, lower), A)

        trace_term = np.trace(X)

        # log determinants – slogdet is stable for positive definite matrices
        sign_A, logdet_A = np.linalg.slogdet(A)
        sign_B, logdet_B = np.linalg.slogdet(B)
        # both determinants are positive, so sign should be 1.0; we ignore it.

        kl = 0.5 * (trace_term - d + logdet_B - logdet_A)
        return kl
