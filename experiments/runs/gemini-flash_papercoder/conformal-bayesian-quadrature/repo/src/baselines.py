import abc
import numpy as np
from scipy import optimize
from typing import Any, List, Tuple

# Assuming LossFunction is available from src.loss_functions
# To avoid circular import or direct dependency in this file for type hinting,
# we can use a forward reference or ensure it's imported at the top level
# when baselines.py is used. For now, let's assume it's imported or will be resolved.
from src.loss_functions import LossFunction


class Method(abc.ABC):
    """
    Abstract base class for all decision rule computation methods.

    Establishes a standardized interface for methods to compute a control
    parameter lambda based on calibration data and a specific loss function.
    """

    def __init__(self, alpha: float, B: float, lambda_search_range: Tuple[float, float]):
        """
        Initializes the base method with common parameters.

        Args:
            alpha: The target risk threshold.
            B: The maximum possible loss.
            lambda_search_range: A tuple (min_lambda, max_lambda) defining the
                                 bounds for searching the control parameter lambda.
        """
        if not (0 <= alpha <= 1):
            raise ValueError("alpha must be between 0 and 1.")
        if not (B > 0):
            raise ValueError("B (maximum loss) must be positive.")
        if not (len(lambda_search_range) == 2 and lambda_search_range[0] <= lambda_search_range[1]):
            raise ValueError("lambda_search_range must be a tuple (min, max) with min <= max.")

        self.alpha = alpha
        self.B = B
        self.lambda_search_range = lambda_search_range

    @abc.abstractmethod
    def compute_lambda(self, cal_data: List[Tuple], loss_fn: LossFunction) -> float:
        """
        Abstract method to compute the decision rule (control parameter) lambda.

        This method must be implemented by all concrete subclasses.

        Args:
            cal_data: A list of calibration data samples, where each sample is a tuple.
            loss_fn: An instance of a LossFunction subclass to calculate individual losses.

        Returns:
            The computed control parameter lambda.
        """
        pass


class ConformalRiskControl(Method):
    """
    Implements the Conformal Risk Control (CRC) decision rule.

    This method finds lambda_crc such that the expected value of L^+ (which
    recovers the CRC objective) is less than or equal to alpha.
    """

    def __init__(self, alpha: float, B: float, lambda_search_range: Tuple[float, float]):
        """
        Initializes the ConformalRiskControl method.

        Args:
            alpha: The target risk threshold.
            B: The maximum possible loss.
            lambda_search_range: A tuple (min_lambda, max_lambda) defining the
                                 bounds for searching the control parameter lambda.
        """
        super().__init__(alpha, B, lambda_search_range)

    def _objective_fn(self, lambda_val: float, cal_data: List[Tuple], loss_fn: LossFunction) -> float:
        """
        Calculates the objective function for CRC: (1/(n+1)) * (Σ ℓ(z_i, λ) + B) - α.

        This function is monotonically non-increasing in lambda_val.
        We are looking for the smallest lambda_val where this function is <= 0.

        Args:
            lambda_val: The current control parameter value.
            cal_data: A list of calibration data samples.
            loss_fn: An instance of a LossFunction subclass.

        Returns:
            The value of the objective function.
        """
        n = len(cal_data)
        if n == 0:
            # If no calibration data, the sum of losses is 0.
            # The objective becomes (0 + B) / (1) - alpha = B - alpha
            # If B <= alpha, we could choose any lambda, but we need to return
            # a value consistent with the search. Let's define this case to prevent errors
            # for the objective. For practical CRC, n > 0 is assumed.
            return self.B - self.alpha

        sum_of_losses = sum(loss_fn.calculate_individual_loss(z_i, lambda_val) for z_i in cal_data)
        mean_expected_loss = (sum_of_losses + self.B) / (n + 1)
        return mean_expected_loss - self.alpha

    def compute_lambda(self, cal_data: List[Tuple], loss_fn: LossFunction) -> float:
        """
        Computes the Conformal Risk Control (CRC) lambda using a root-finding algorithm.

        It finds the smallest lambda such that the CRC objective is <= self.alpha.
        Since the objective function is monotonically non-increasing, we can use
        scipy.optimize.brentq to find the root.

        Args:
            cal_data: A list of calibration data samples.
            loss_fn: An instance of a LossFunction subclass.

        Returns:
            The computed CRC lambda_crc.
        """
        min_lambda, max_lambda = self.lambda_search_range

        # Evaluate objective at the boundaries of the search range
        obj_at_min_lambda = self._objective_fn(min_lambda, cal_data, loss_fn)
        obj_at_max_lambda = self._objective_fn(max_lambda, cal_data, loss_fn)

        # Case 1: If the condition is already met at the minimum possible lambda,
        # then the infimum is the minimum lambda.
        if obj_at_min_lambda <= 0:
            return min_lambda

        # Case 2: If the condition is not met even at the maximum possible lambda,
        # it implies that controlling risk to alpha is not achievable within this range.
        # In such cases, we return the maximum lambda as a practical upper bound,
        # which represents the least conservative choice under the unmet condition.
        if obj_at_max_lambda > 0:
            # print(f"Warning: CRC condition (risk <= {self.alpha}) not met even at max_lambda={max_lambda}. Returning max_lambda.")
            return max_lambda

        # Case 3: A root exists within (min_lambda, max_lambda)
        # scipy.optimize.brentq requires the function values at the endpoints
        # to have opposite signs. Our objective function is non-increasing,
        # so obj_at_min_lambda > 0 and obj_at_max_lambda <= 0.
        try:
            # The root found is the infimum lambda that satisfies the condition.
            lambda_crc = optimize.brentq(self._objective_fn, min_lambda, max_lambda,
                                         args=(cal_data, loss_fn))
            return lambda_crc
        except ValueError as e:
            # This should ideally not happen if the earlier checks are correct and
            # the range is appropriately chosen such that a root is bracketed.
            # print(f"Error in brentq for CRC: {e}. Returning max_lambda as fallback.")
            return max_lambda


class RCPSBaseline(Method):
    """
    Placeholder for the Risk-controlling Prediction Sets (RCPS) method.

    The precise implementation requires details from Bates et al. (2021)
    regarding the application of the Hoeffding upper confidence bound.
    This class currently raises a NotImplementedError as per the design.
    """

    def __init__(self, alpha: float, B: float, lambda_search_range: Tuple[float, float], beta_rcps: float):
        """
        Initializes the RCPSBaseline method.

        Args:
            alpha: The target risk threshold.
            B: The maximum possible loss.
            lambda_search_range: A tuple (min_lambda, max_lambda) defining the
                                 bounds for searching the control parameter lambda.
            beta_rcps: The confidence level for the Hoeffding bound (e.g., config.beta).
        """
        super().__init__(alpha, B, lambda_search_range)
        if not (0 <= beta_rcps <= 1):
            raise ValueError("beta_rcps must be between 0 and 1.")
        self.beta_rcps = beta_rcps

    def compute_lambda(self, cal_data: List[Tuple], loss_fn: LossFunction) -> float:
        """
        Placeholder implementation for computing RCPS lambda.

        Raises:
            NotImplementedError: This method is not yet implemented and
                                 requires further details from Bates et al. (2021).
        """
        # As per the design and "Anything UNCLEAR", this is a placeholder.
        # A proper implementation would follow Bates et al. (2021) to
        # incorporate the Hoeffding bound for risk control.
        raise NotImplementedError(
            "RCPSBaseline.compute_lambda is a placeholder and requires "
            "implementation based on Bates et al. (2021) and Hoeffding UCB details."
        )

