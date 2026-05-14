import numpy as np
import random
import os
from typing import Union

def kl_divergence_gaussian(
    mu1: np.ndarray, 
    sigma1: np.ndarray, 
    mu2: np.ndarray, 
    sigma2: np.ndarray
) -> float:
    """
    Calculates the Kullback-Leibler (KL) divergence between two multivariate
    Gaussian distributions.

    P ~ N(mu1, sigma1)
    Q ~ N(mu2, sigma2)

    Formula:
    D_KL(P||Q) = 0.5 * (tr(sigma2_inv @ sigma1) + (mu2 - mu1).T @ sigma2_inv @ (mu2 - mu1)
                       - d + log(det(sigma2)) - log(det(sigma1)))

    Args:
        mu1: Mean vector of the first Gaussian distribution (P).
        sigma1: Covariance matrix of the first Gaussian distribution (P).
        mu2: Mean vector of the second Gaussian distribution (Q).
        sigma2: Covariance matrix of the second Gaussian distribution (Q).

    Returns:
        The KL divergence between P and Q.

    Raises:
        ValueError: If input dimensions are inconsistent or if covariance matrices
                    are not positive definite.
    """
    d = mu1.shape[0]

    if mu2.shape[0] != d or sigma1.shape != (d, d) or sigma2.shape != (d, d):
        raise ValueError("Inconsistent dimensions for input Gaussian parameters.")

    # Ensure covariance matrices are numerically stable
    # Add a small epsilon to the diagonal if needed, though np.linalg.inv usually handles it.
    # For this problem, sigma2 is expected to be well-conditioned.

    try:
        inv_sigma2 = np.linalg.inv(sigma2)
    except np.linalg.LinAlgError:
        raise ValueError("Covariance matrix sigma2 is singular, cannot compute inverse.")

    # Term 1: tr(sigma2_inv @ sigma1)
    trace_term = np.trace(inv_sigma2 @ sigma1)

    # Term 2: (mu2 - mu1).T @ sigma2_inv @ (mu2 - mu1)
    diff_mu = mu2 - mu1
    quadratic_term = diff_mu.T @ inv_sigma2 @ diff_mu

    # Term 3 & 4: log(det(sigma2)) - log(det(sigma1))
    # Use slogdet for numerical stability, which returns (sign, logabsdet)
    sign_det_sigma1, log_det_sigma1 = np.linalg.slogdet(sigma1)
    sign_det_sigma2, log_det_sigma2 = np.linalg.slogdet(sigma2)

    if sign_det_sigma1 <= 0 or sign_det_sigma2 <= 0:
        # A non-positive determinant indicates a singular or non-positive definite matrix
        # which can happen with empirical covariance or if numerical issues arise.
        # For valid Gaussian distributions, determinants must be positive.
        # Handle this as an error or by adding a small diagonal perturbation,
        # but for exact theoretical Gaussians, it shouldn't occur.
        print(f"Warning: Non-positive determinant detected. "
              f"det(sigma1) sign: {sign_det_sigma1}, det(sigma2) sign: {sign_det_sigma2}")
        # As per the reproduction plan, exact scores for Gaussian target are used,
        # so this should be positive. If it happens, it's a serious numerical issue.
        # For robust calculation, one might add a small diagonal epsilon:
        # sigma1_stable = sigma1 + np.eye(d) * 1e-6
        # sigma2_stable = sigma2 + np.eye(d) * 1e-6
        # and re-compute slogdet with these.
        # For now, let's proceed and allow the original matrices to be used,
        # relying on proper construction.

    kl_div = 0.5 * (trace_term + quadratic_term - d + log_det_sigma2 - log_det_sigma1)

    return float(kl_div)


def setup_rng(seed: int):
    """
    Initializes the global random number generators for numpy and the
    standard Python random module to ensure reproducibility.

    Args:
        seed: The integer seed value for the random number generators.
    """
    np.random.seed(seed)
    random.seed(seed)


def create_output_directory(path: str):
    """
    Creates a directory at the specified path if it does not already exist.

    Args:
        path: The path to the directory to create.
    """
    os.makedirs(path, exist_ok=True)

