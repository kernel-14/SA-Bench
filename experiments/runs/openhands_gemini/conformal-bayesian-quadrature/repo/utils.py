import numpy as np
from scipy.stats import dirichlet, binom
from typing import Tuple

class Utils:
    """
    Utility functions for statistical calculations, e.g., Dirichlet sampling
    and confidence interval computation.
    """

    @staticmethod
    def sample_dirichlet(num_samples: int, alpha_params: np.ndarray) -> np.ndarray:
        """
        Samples from a Dirichlet distribution.

        Args:
            num_samples (int): The number of samples to draw.
            alpha_params (np.ndarray): Array of concentration parameters (alpha_1, ..., alpha_k).

        Returns:
            np.ndarray: Samples from the Dirichlet distribution, shape (num_samples, len(alpha_params)).
        """
        return dirichlet.rvs(alpha_params, size=num_samples)

    @staticmethod
    def compute_clopper_pearson_ci(
        num_successes: int, num_trials: int, confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Computes the Clopper-Pearson exact confidence interval for a binomial proportion.

        Args:
            num_successes (int): Number of successes.
            num_trials (int): Number of trials.
            confidence_level (float): Desired confidence level (e.g., 0.95 for 95% CI).

        Returns:
            Tuple[float, float]: A tuple (lower_bound, upper_bound) of the confidence interval.
        """
        if num_trials == 0:
            return 0.0, 1.0 # Or raise an error, depending on desired behavior for 0 trials

        # Using scipy.stats.beta for the calculation, as Clopper-Pearson is based on Beta distribution quantiles
        # Lower bound
        low = binom.interval(confidence_level, n=num_trials, p=num_successes / num_trials)[0] / num_trials if num_successes > 0 else 0.0
        # Upper bound
        high = binom.interval(confidence_level, n=num_trials, p=num_successes / num_trials)[1] / num_trials if num_successes < num_trials else 1.0

        # Adjust for edge cases if necessary, e.g., for very small success/failure rates
        # The above interval(p) method might not be directly giving Clopper-Pearson.
        # A more robust way using beta.ppf:
        from scipy.stats import beta
        if num_successes == 0:
            lower_bound = 0.0
        else:
            lower_bound = beta.ppf((1 - confidence_level) / 2, num_successes, num_trials - num_successes + 1)

        if num_successes == num_trials:
            upper_bound = 1.0
        else:
            upper_bound = beta.ppf(1 - (1 - confidence_level) / 2, num_successes + 1, num_trials - num_successes)
            
        return lower_bound, upper_bound
