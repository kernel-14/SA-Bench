import numpy as np
from typing import Union, Dict, Any

class GaussianDiffusionProcess:
    """
    Encapsulates the mathematical definition of the forward diffusion process
    and the exact score function for a Gaussian target distribution.

    This class handles the generation of the initial data distribution's
    covariance matrix (Sigma_0), sampling from this distribution, and
    providing the exact score function for any given time tau and data point x.
    It also computes the covariance for the reference distribution q_K.
    """

    def __init__(self, d: int, k: int, rng_seed: int = None):
        """
        Initializes the GaussianDiffusionProcess with problem dimensions and
        sets up the initial covariance matrix.

        Args:
            d: The total dimension of the data.
            k: The number of active (non-zero variance) dimensions for the
               initial data X_0.
            rng_seed: A seed for numpy's random number generator to ensure
                      reproducibility of Sigma_0 generation and X_0 sampling.
                      Defaults to None for non-reproducible randomness if not set.
        """
        if not isinstance(d, int) or d <= 0:
            raise ValueError("Dimension 'd' must be a positive integer.")
        if not isinstance(k, int) or k < 0 or k > d:
            raise ValueError("Number of active dimensions 'k' must be a non-negative "
                             "integer less than or equal to 'd'.")

        self.d: int = d
        self.k: int = k
        self.rng: np.random.Generator = np.random.default_rng(rng_seed)
        self.sigma0: np.ndarray = self._generate_sigma0()

    def _generate_sigma0(self) -> np.ndarray:
        """
        Creates the diagonal covariance matrix Sigma_0 for the initial data X_0.
        The first 'k' diagonal entries are uniformly distributed in [0, 10],
        and the remaining 'd-k' entries are zero.

        Returns:
            A d x d NumPy array representing the diagonal covariance matrix Sigma_0.
        """
        sigma0_diag_elements = np.zeros(self.d, dtype=np.float64)
        if self.k > 0:
            # Generate k random values for active dimensions from [0, 10]
            active_variances = self.rng.uniform(0.0, 10.0, self.k)
            sigma0_diag_elements[:self.k] = active_variances
        
        # Construct a diagonal matrix from these elements
        sigma0 = np.diag(sigma0_diag_elements)
        return sigma0

    def sample_x0(self, num_samples: int) -> np.ndarray:
        """
        Samples initial data points X_0 from the target data distribution
        N(0, Sigma_0).

        Args:
            num_samples: The number of X_0 samples to generate.

        Returns:
            An array of shape (num_samples, d), where each row is a sample from X_0.
        """
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("Number of samples must be a positive integer.")

        mean_vector = np.zeros(self.d, dtype=np.float64)
        return self.rng.multivariate_normal(mean_vector, self.sigma0, size=num_samples)

    def get_exact_score(self, x: np.ndarray, current_tau: float) -> np.ndarray:
        """
        Computes the exact score function s_tau^*(x) = -((1-tau)Sigma_0 + tau I_d)^(-1) x.
        This calculation is optimized for diagonal Sigma_0.

        Args:
            x: The data point(s) at which to compute the score. Can be a single
               vector of shape (d,) or a batch of vectors (num_points, d).
            current_tau: The current time step tau. Must be in (0, 1).

        Returns:
            The computed score vector(s). Its shape will match the input x.

        Raises:
            ValueError: If current_tau is not within (0, 1).
        """
        if not isinstance(current_tau, (float, np.float32, np.float64)) or not (0.0 < current_tau < 1.0):
            raise ValueError("current_tau must be a float in (0, 1) for score function calculation.")
        if not isinstance(x, np.ndarray) or x.shape[-1] != self.d:
            raise ValueError(f"Input 'x' must be a numpy array with last dimension {self.d}.")

        # Sigma_0 is diagonal, so ((1-tau)Sigma_0 + tau I_d) is also diagonal.
        # Its inverse is found by inverting the diagonal elements.
        diag_sigma0_elements = np.diag(self.sigma0)
        
        # Calculate the diagonal elements of the current covariance matrix
        diag_current_cov_elements = (1.0 - current_tau) * diag_sigma0_elements + current_tau
        
        # Compute the reciprocal of these elements for the inverse
        # All diag_current_cov_elements should be > 0 if current_tau > 0 and diag_sigma0_elements >= 0.
        # Add a small epsilon to avoid division by zero if current_tau is extremely close to 0 and k=0.
        # However, the constraint 0 < current_tau ensures this is positive.
        inv_diag_current_cov_elements = 1.0 / diag_current_cov_elements
        
        # Compute the score: - (inv_diag_current_cov_elements * x)
        # np.atleast_2d(x) ensures x is treated as a batch of vectors,
        # and then we slice to match the original dimension if it was a single vector.
        x_reshaped = np.atleast_2d(x)
        score = -x_reshaped * inv_diag_current_cov_elements
        
        # If original x was 1D, return 1D score
        if x.ndim == 1:
            return score.flatten()
        return score

    def get_qK_covariance(self, last_tau_k0: float) -> np.ndarray:
        """
        Calculates the covariance matrix of X_tau_K,0, which represents the
        distribution q_K used for evaluation. This is given by
        (1 - tau_K,0)Sigma_0 + tau_K,0 * I_d.

        Args:
            last_tau_k0: The specific tau value for the start of the last round
                         (corresponding to tau_K,0 in the paper's notation).
                         Must be in (0, 1).

        Returns:
            The d x d NumPy array representing the covariance matrix of q_K.

        Raises:
            ValueError: If last_tau_k0 is not within (0, 1).
        """
        if not isinstance(last_tau_k0, (float, np.float32, np.float64)) or not (0.0 < last_tau_k0 < 1.0):
            raise ValueError("last_tau_k0 must be a float in (0, 1).")

        identity_matrix = np.eye(self.d, dtype=np.float64)
        qK_covariance = (1.0 - last_tau_k0) * self.sigma0 + last_tau_k0 * identity_matrix
        return qK_covariance

    def get_sigma0(self) -> np.ndarray:
        """
        Returns the initial covariance matrix Sigma_0 of the target data distribution.

        Returns:
            The d x d NumPy array representing Sigma_0.
        """
        return self.sigma0

