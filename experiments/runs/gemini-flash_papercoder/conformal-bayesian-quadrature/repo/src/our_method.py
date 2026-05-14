import numpy as np
from scipy.optimize import brentq
# scipy.stats.dirichlet is not strictly needed if using np.random.Generator.dirichlet
from typing import List, Tuple, Callable

# Assume LossFunction and Method are imported from their respective modules
# Explicitly import for clarity based on task instructions
from src.baselines import Method
from src.loss_functions import LossFunction


class OurMethod(Method):
    """
    Implements the core Bayesian Quadrature-based method for Conformal Prediction
    to find the decision rule lambda_hpd^beta.
    """
    def __init__(self,
                 alpha: float,
                 beta: float,
                 B: float,
                 num_dirichlet_samples: int,
                 lambda_search_range: Tuple[float, float],
                 random_state: np.random.Generator):
        """
        Initializes OurMethod with parameters from the configuration.

        Args:
            alpha (float): Target risk threshold (e.g., from config.experiment.alpha or experiment-specific alpha).
            beta (float): Confidence level for the conditional guarantee (e.g., config.experiment.beta).
            B (float): Maximum possible loss (config.experiment.B).
            num_dirichlet_samples (int): Number of Monte Carlo samples for Dirichlet distribution
                                         (config.experiment.num_dirichlet_samples).
            lambda_search_range (Tuple[float, float]): The lower and upper bounds for the lambda search
                                                       (config.method_params.lambda_search_range).
            random_state (np.random.Generator): NumPy random number generator for reproducibility.
        """
        # Validate inputs that are not already validated by the base class.
        if not (0.0 <= beta <= 1.0):
            raise ValueError("beta must be between 0.0 and 1.0.")
        if not isinstance(num_dirichlet_samples, int) or num_dirichlet_samples <= 0:
            raise ValueError("num_dirichlet_samples must be a positive integer.")
        if not isinstance(random_state, np.random.Generator):
            raise TypeError("random_state must be an instance of np.random.Generator.")

        super().__init__(alpha, B, lambda_search_range)
        self.beta = beta
        self.num_dirichlet_samples = num_dirichlet_samples
        self.random_state = random_state

    def compute_lambda(self, cal_data: List[Tuple], loss_fn: LossFunction) -> float:
        """
        Finds lambda_hpd^beta by solving the root-finding problem:
        Pr(L^+ <= alpha | calibration_losses) - beta = 0.
        This method uses scipy.optimize.brentq to find the smallest lambda that satisfies the condition.

        Args:
            cal_data (List[Tuple]): List of calibration data points (z_i).
            loss_fn (LossFunction): The specific loss function instance for the experiment.

        Returns:
            float: The determined lambda_hpd^beta value.
        """
        # The objective function `g(lambda) = Pr(L^+ <= alpha | ℓ_1:n) - beta`
        # is monotonically non-decreasing with `lambda_val`.
        # We are looking for `inf {lambda : g(lambda) >= 0}`.

        lambda_low, lambda_high = self.lambda_search_range

        # Wrap the objective function to fix cal_data and loss_fn for brentq.
        # This creates a callable that only depends on lambda_val, which brentq expects.
        objective_wrapper: Callable[[float], float] = lambda l: self._objective_fn(l, cal_data, loss_fn)

        val_low = objective_wrapper(lambda_low)
        val_high = objective_wrapper(lambda_high)

        if val_low >= 0:
            # Case 1: The condition `Pr(L^+ <= alpha) >= beta` is already met
            # at the lower bound of the search range. This means the infimum
            # lambda is `lambda_low` or smaller. Since `lambda_low` is our minimum
            # search boundary, we return it.
            return lambda_low
        elif val_high < 0:
            # Case 2: The condition `Pr(L^+ <= alpha) >= beta` is NOT met
            # even at the upper bound of the search range. This implies that
            # no lambda within the given range `[lambda_low, lambda_high]` satisfies
            # the condition for the target `beta`. In this scenario, the
            # `lambda_hpd^beta` would effectively be greater than `lambda_high`
            # (conceptually, "infinity"). As a practical choice within the defined
            # limits, we return `lambda_high`, representing the most permissive `lambda`
            # we can choose without satisfying the guarantee within the bounds.
            # A warning could be logged here if desired for debugging.
            return lambda_high
        else:
            # Case 3: `val_low < 0` and `val_high >= 0`. This means the root
            # (where `g(lambda) = 0`) is bracketed within the interval
            # `(lambda_low, lambda_high)`. `brentq` can be reliably used here
            # to find this root, which corresponds to the infimum `lambda`.
            try:
                lambda_hpd = brentq(objective_wrapper, lambda_low, lambda_high)
                return lambda_hpd
            except ValueError as e:
                # This should ideally not happen if the earlier checks for val_low and val_high
                # correctly bracket the root. It might indicate numerical issues or an
                # unexpected behavior of the objective function.
                print(f"Warning: scipy.optimize.brentq failed with error: {e}. "
                      f"Objective values: low={val_low}, high={val_high}. "
                      f"Returning lambda_high as fallback.")
                return lambda_high # Fallback to the upper bound if root finding fails unexpectedly

    def _objective_fn(self, lambda_val: float, cal_data: List[Tuple], loss_fn: LossFunction) -> float:
        """
        Calculates the objective function for the root-finding problem:
        Pr(L^+ <= alpha | calibration_losses) - beta.
        This function is monotonically non-decreasing in lambda_val.

        Args:
            lambda_val (float): The current lambda value to evaluate.
            cal_data (List[Tuple]): List of calibration data points (z_i).
            loss_fn (LossFunction): The specific loss function instance.

        Returns:
            float: The value Pr(L^+ <= alpha | calibration_losses) - beta.
        """
        # If there's no calibration data, we can't estimate L+.
        # This case should ideally not happen in experimental settings where n > 0.
        if not cal_data:
            # If no data, Pr(L^+ <= alpha) is hard to define without a specific prior.
            # Returning a value that ensures a decision is made or indicates an issue.
            # For robustness, let's assume we cannot satisfy the condition if n=0.
            return -1.0 - self.beta # Ensures Pr - beta is negative, leading to lambda_high
            
        # 1. Compute individual losses for the current lambda_val
        calibration_losses = [loss_fn.calculate_individual_loss(z, lambda_val) for z in cal_data]

        # 2. Calculate Pr(L^+ <= alpha | ℓ_1:n) via Monte Carlo
        pr_lplus_le_alpha = self._calculate_pr_lplus_le_alpha(calibration_losses)

        # 3. Return the difference from beta, which is the value that brentq will try to zero.
        return pr_lplus_le_alpha - self.beta

    def _calculate_pr_lplus_le_alpha(self, current_lambda_losses: List[float]) -> float:
        """
        Estimates Pr(L^+ <= alpha | ℓ_1:n) using Monte Carlo simulation
        with Dirichlet-distributed quantile spacings.

        Args:
            current_lambda_losses (List[float]): The list of individual loss values (ℓ_i)
                                                 computed for a specific lambda_val.

        Returns:
            float: The estimated probability P(L^+ <= alpha).
        """
        n: int = len(current_lambda_losses)

        # 1. Sort observed losses to get order statistics ℓ_(i)
        # Convert to numpy array for efficient sorting and operations.
        # Ensure losses are within the defined bound B for safety, although the
        # problem statement implies they are.
        sorted_losses: np.ndarray = np.sort(np.array(current_lambda_losses, dtype=float))

        # 2. Append B as ℓ_(n+1)
        # This creates the vector (ℓ_(1), ..., ℓ_(n), B) of length n+1.
        ell_ordered_stats: np.ndarray = np.append(sorted_losses, self.B)

        # 3. Generate num_dirichlet_samples from Dirichlet(1, ..., 1)
        # The concentration parameter `alpha_vec` has shape (n+1,) with all values being 1.
        alpha_vec: np.ndarray = np.ones(n + 1, dtype=float)
        
        # self.random_state.dirichlet returns samples of shape (size, n+1).
        U_samples: np.ndarray = self.random_state.dirichlet(alpha_vec, size=self.num_dirichlet_samples)

        # 4. Compute L^+ = Σ U_j ℓ_(j) for each sample
        # This is a matrix-vector product.
        # L_plus_values will be an array of shape (num_dirichlet_samples,).
        L_plus_values: np.ndarray = np.dot(U_samples, ell_ordered_stats)

        # 5. Count how many L^+ values are <= alpha (self.alpha)
        count_le_alpha: int = np.sum(L_plus_values <= self.alpha)

        # 6. Return proportion
        return float(count_le_alpha) / self.num_dirichlet_samples

